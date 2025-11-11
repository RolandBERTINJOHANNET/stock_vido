#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_context_check.py — DEBUG / VERBOSE (OpenAI-only, no HF)

Semantic fitness check for Pixabay stock clips using OpenAI multimodal, with
negative-case prompt synthesis for Text-to-Video (T2V).

Pipeline:
  1. Download candidate video (Pixabay URL).
  2. Extract a few JPEG frames with ffmpeg.
  3. Call OpenAI (gpt-4.1-mini by default) with frames + analysis prompt
     → get a factual description.
  4. Call OpenAI again with narration sentence + description
     → get "[yes]" / "[no]" + short explanation.
  5. If decision is "[no]": call OpenAI (gpt-4.1) again **WITHOUT** any frames
     or video description — use ONLY the narration sentence plus the explicit
     T2V purpose context — to produce a single-sentence T2V prompt (camera-ready).
  6. Return structured result, including "t2v_prompt" if generated.

Requirements:
  - OPENAI_API_KEY must be set.
  - ffmpeg must be installed and in PATH.
  - Designed to be imported and called from video1.py:

        from video_context_check import check_video_fitness

        fitness = check_video_fitness(sentence, chosen_url, t2v_purpose_context="Restaurant worker training b-roll.")
        if fitness["final_decision"] != "yes":
            print(f"Use T2V prompt: {fitness['t2v_prompt']}")

Public API:
    check_video_fitness(
        sentence: str,
        video_url: str,
        *,
        openai_api_key: str | None = None,
        vision_model: str = "gpt-4.1-mini",
        text_model: str = "gpt-4.1-mini",
        t2v_prompt_model: str = "gpt-4.1",
        t2v_purpose_context: str | None = None,   # NEW: extra context for the prompt writer
        verbose: bool = True,
    ) -> dict

Returns:
    {
      "description": str,          # vision analysis (from frames)
      "questionnaire": dict,       # reserved
      "final_decision": "yes" | "no",
      "reason": str,               # one-line justification from decision step
      "t2v_prompt": str | None     # present only when decision == "no"
    }
"""

from __future__ import annotations
import os
import time
import base64
import shutil
import tempfile
import traceback
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import requests
from openai import OpenAI


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
_VIDEO_PROMPT = """You are a precise vision-language analyst.

You are given several frames sampled from a short stock video clip.
Based ONLY on these frames, do the following:

1. Describe the video content in one concise sentence.
2. Then briefly answer:
   - What is the main object or subject?
   - What is the setting or environment?
   - What kind of action is taking place?
   - What overall mood or atmosphere is suggested?

Be specific and factual. Do not hallucinate content that is not visible.
"""

_DECISION_PROMPT = """You are checking if a stock video matches the narration context.

Narration sentence:
"{sentence}"

Video description and details:
{video_description}

Question:
Does this video semantically and visually fit the narration context?
The context is : a broll for a restaurant workers training video.
Rules:
- Answer "[yes]" if it clearly fits or is neutral but acceptable.
- Answer "[no]" if it is misleading, off-topic, or clashes with the narration.
Then, on the same line, give one short reason.

Examples:
[yes] Neutral office background that fits a general training intro.
[no] Gym workout footage that does not match a kitchen training narration.
"""

# NEW — sentence-only T2V prompt writer instruction (no frames/description allowed)
_T2V_SENTENCE_ONLY_INSTRUCTION = """You are writing EXACTLY ONE sentence for a text-to-video generator to create a replacement clip.

Purpose:
- The stock footage we found was WRONG for the narration.
- Your job is to generate a camera-ready prompt that will produce a NEW clip that CLEARLY MATCHES the narration line.

Context for the final video:
{purpose_context}

Strict rules:
- Base your prompt ONLY on the narration sentence and the purpose/context above.
- DO NOT reference or replicate any previously seen stock footage or its visuals.
- 18–30 words preferred (do not include a word count).
- Be specific: include subject, setting, action, and briefly a camera motion or composition cue.
- Keep it neutral, brand-safe, and workplace-safe; avoid trademarks/logos and identifiable faces.
- No hashtags, quotes, brackets, bullet points, or scene numbers.
- Output ONLY the prompt sentence, nothing else.
"""


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _download_video_to_temp(video_url: str, verbose: bool = True) -> Path:
    """Download remote video URL to a temporary file."""
    t0 = time.perf_counter()
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="vc_video_", suffix=".mp4")
    os.close(tmp_fd)
    tmp = Path(tmp_path)

    if verbose:
        print(f"    [VC] Downloading video to temp file: {tmp}")

    with requests.get(video_url, stream=True, timeout=30.0) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)

    dt = time.perf_counter() - t0
    if verbose:
        size_mb = tmp.stat().st_size / (1024 * 1024)
        print(f"    [VC] Downloaded in {dt:.2f}s ({size_mb:.1f} MB)")
    return tmp


def _extract_frames_ffmpeg(video_path: Path, num_frames: int = 4, verbose: bool = True) -> List[Path]:
    """
    Extract up to `num_frames` frames using ffmpeg.

    Strategy:
      - 1 frame per second, capped by num_frames.
      - Outputs JPEGs in a temp directory.

    Returns:
      list[Path] of extracted frame files.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="vc_frames_"))
    pattern = out_dir / "frame_%03d.jpg"

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", "fps=1",
        "-vframes", str(num_frames),
        "-qscale:v", "3",
        str(pattern),
        "-y",
        "-loglevel", "error",
    ]

    if verbose:
        print(f"    [VC] Extracting up to {num_frames} frames with ffmpeg...")
        print(f"    [VC] Command: {' '.join(cmd)}")

    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"    [VC][ERROR] ffmpeg extraction failed after {dt:.2f}s → {e}")
        if verbose:
            traceback.print_exc(limit=2)
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    frames = sorted(out_dir.glob("frame_*.jpg"))
    dt = time.perf_counter() - t0

    if verbose:
        print(f"    [VC] Extracted {len(frames)} frame(s) in {dt:.2f}s")
        if not frames:
            print("    [VC][WARN] No frames extracted from video.")

    if not frames:
        shutil.rmtree(out_dir, ignore_errors=True)
    return frames


def _encode_image_as_data_url(path: Path) -> str:
    """Convert an image file to a data URL for OpenAI image input."""
    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _collapse_to_one_line(s: str) -> str:
    """Collapse whitespace and strip quotes/brackets to keep one clean sentence."""
    s = s.replace("\n", " ").replace("\r", " ")
    # Strip surrounding quotes/brackets if present
    s = s.strip().strip("'").strip('"').strip("[](){}").strip()
    # Collapse multiple spaces
    s = " ".join(s.split())
    return s


# -----------------------------------------------------------------------------
# OpenAI helpers
# -----------------------------------------------------------------------------
def _describe_video_with_gpt(
    video_url: str,
    *,
    openai_api_key: str,
    vision_model: str,
    verbose: bool,
) -> Tuple[str, List[Path]]:
    """
    Use OpenAI multimodal model to describe the video:
      - download video
      - extract frames
      - send frames + analysis prompt

    Returns:
      (description_text, frame_paths)
    """
    client = OpenAI(api_key=openai_api_key)

    # 1) Download video
    video_path = _download_video_to_temp(video_url, verbose=verbose)

    try:
        # 2) Extract frames
        frames = _extract_frames_ffmpeg(video_path, num_frames=4, verbose=verbose)
        if not frames:
            raise RuntimeError("No frames extracted; cannot analyze video.")

        if verbose:
            print("    [VC] Preparing frames for OpenAI vision call...")

        # 3) Build content: frames as image inputs + prompt text
        content = []
        for i, frame_path in enumerate(frames, start=1):
            data_url = _encode_image_as_data_url(frame_path)
            if verbose:
                print(f"    [VC] Frame {i}: {frame_path} (encoded as data URL)")
            content.append({
                "type": "image_url",
                "image_url": {"url": data_url},
            })

        content.append({
            "type": "text",
            "text": _VIDEO_PROMPT,
        })

        if verbose:
            print(f"    [VC] Calling OpenAI vision model: {vision_model}")

        t0 = time.perf_counter()
        completion = client.chat.completions.create(
            model=vision_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=300,
            temperature=0,
        )
        dt = time.perf_counter() - t0

        desc = (completion.choices[0].message.content or "").strip()
        if verbose:
            print(f"    [VC] OpenAI vision response in {dt:.2f}s")
            print(f"    [VC] Description (truncated): {desc[:300]}...")

        return desc, frames

    finally:
        # Cleanup temporary video file (keep frames until decision is done)
        try:
            if video_path.exists():
                video_path.unlink(missing_ok=True)
                if verbose:
                    print(f"    [VC] Cleaned temp video file: {video_path}")
        except Exception:
            if verbose:
                print("    [VC][WARN] Failed to clean temp video file (ignored).")


def _ask_fit_decision(
    *,
    sentence: str,
    desc: str,
    openai_api_key: str,
    text_model: str,
    verbose: bool,
) -> Tuple[str, str]:
    """
    Calls OpenAI to decide [yes]/[no] given narration + video description.
    """
    if verbose:
        print("    [VC] Preparing decision prompt...")
        print(f"    [VC] Narration: {sentence}")
        print(f"    [VC] Description (first 160): {desc[:160]}...")

    client = OpenAI(api_key=openai_api_key)
    prompt = _DECISION_PROMPT.format(
        sentence=sentence,
        video_description=desc,
    )

    t0 = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=text_model,
            temperature=0,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        dt = time.perf_counter() - t0
        resp = (completion.choices[0].message.content or "").strip()
        if verbose:
            print(f"    [VC] Decision response in {dt:.2f}s: {resp}")

        low = resp.lower()
        if low.startswith("[yes]"):
            return "yes", resp
        if low.startswith("[no]"):
            return "no", resp

        # Fallback: anything ambiguous → "no"
        return "no", f"unclear → {resp}"

    except Exception as e:
        dt = time.perf_counter() - t0
        print(f"    [VC][ERROR] Decision call failed after {dt:.2f}s → {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc(limit=2)
        raise


# NEW — sentence-only T2V prompt builder (no frames or description)
def _build_t2v_prompt_from_sentence(
    *,
    sentence: str,
    purpose_context: str,
    openai_api_key: str,
    t2v_prompt_model: str,
    verbose: bool,
) -> str:
    """
    Build a one-sentence, camera-ready T2V prompt using ONLY the narration sentence
    and the explicit purpose/context. No frames or video description are provided.
    """
    if verbose:
        print("    [VC] Building T2V prompt from sentence + purpose context (no frames)...")

    client = OpenAI(api_key=openai_api_key)

    instruction = _T2V_SENTENCE_ONLY_INSTRUCTION.format(
        purpose_context=purpose_context.strip()
    )
    user_text = (
        instruction
        + "\n\nNarration sentence to match:\n"
        + f"\"{sentence}\""
    )

    print("t2v prompt : ", user_text)

    t0 = time.perf_counter()
    completion = client.chat.completions.create(
        model=t2v_prompt_model,
        temperature=0.2,
        max_tokens=120,
        messages=[{"role": "user", "content": user_text}],
    )
    dt = time.perf_counter() - t0
    raw = (completion.choices[0].message.content or "").strip()
    if verbose:
        print(f"    [VC] T2V prompt (sentence-only) response in {dt:.2f}s: {raw}")

    return _collapse_to_one_line(raw)


# -----------------------------------------------------------------------------
# Public orchestration
# -----------------------------------------------------------------------------
def check_video_fitness(
    sentence: str,
    video_url: str,
    *,
    openai_api_key: str | None = None,
    vision_model: str = "gpt-4.1-mini",
    text_model: str = "gpt-4.1-mini",
    t2v_prompt_model: str = "gpt-4.1",   # use full 4.1 for prompt synth
    t2v_purpose_context: str | None = None,  # NEW: pass domain/purpose context for better prompts
    verbose: bool = True,
) -> dict:
    """
    Full pipeline:
      - Describe video via OpenAI multimodal (frames + vision_model).
      - Decide fit via text_model.
      - If decision == "no": build a T2V prompt using GPT-4.1 with ONLY the narration
        sentence + the provided/default purpose context (NO frames/description).
      - Return structured result.

    Returns:
        {
          "description": str,
          "questionnaire": dict,             # reserved for future structured fields
          "final_decision": "yes" | "no",
          "reason": str,
          "t2v_prompt": str | None
        }
    """
    print("\n  [CHECK] Starting semantic video fitness check")
    print(f"  [CHECK] Sentence: {sentence}")
    print(f"  [CHECK] Video: {video_url}")

    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for OpenAI multimodal calls.")

    # Default context if none provided — generic training/e-learning b-roll
    purpose_context = (t2v_purpose_context or
        "General restorant workers training/e-learning b-roll to illustrate the narration clearly. "
        "Prefer neutral professional settings that match the topic implied by the sentence. "
        "Keep content brand-safe, workplace-appropriate, and free of identifiable faces or logos."
    )

    frames: List[Path] = []
    try:
        # (1) Vision description (from frames)
        desc, frames = _describe_video_with_gpt(
            video_url,
            openai_api_key=openai_api_key,
            vision_model=vision_model,
            verbose=verbose,
        )

        # (2) Fit decision
        decision, reason = _ask_fit_decision(
            sentence=sentence,
            desc=desc,
            openai_api_key=openai_api_key,
            text_model=text_model,
            verbose=verbose,
        )

        # (3) If negative, synthesize a one-sentence T2V prompt (sentence-only)
        t2v_prompt: Optional[str] = None
        if decision == "no":
            try:
                t2v_prompt = _build_t2v_prompt_from_sentence(
                    sentence=sentence,
                    purpose_context=purpose_context,
                    openai_api_key=openai_api_key,
                    t2v_prompt_model=t2v_prompt_model,
                    verbose=verbose,
                )
            except Exception as e:
                print(f"  [CHECK][WARN] T2V prompt synthesis failed → {type(e).__name__}: {e}")
                t2v_prompt = None

        print(f"  [CHECK][DONE] Final decision: {decision.upper()}")
        print(f"  [CHECK][REASON] {reason[:300]}")
        if decision == "no":
            print(f"  [CHECK][T2V] Prompt: {t2v_prompt}")

        return {
            "description": desc,
            "questionnaire": {},
            "final_decision": decision,
            "reason": reason,
            "t2v_prompt": t2v_prompt,
        }

    finally:
        # Cleanup extracted frames directory if present
        try:
            if frames:
                frame_dir = frames[0].parent
                shutil.rmtree(frame_dir, ignore_errors=True)
                if verbose:
                    print(f"    [VC] Cleaned frame directory: {frame_dir}")
        except Exception:
            if verbose:
                print("    [VC][WARN] Failed to clean frame directory (ignored).")
