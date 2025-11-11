#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video1.py — Pixabay proposer/downloader with semantic fitness & Grok fallback

Public API (unchanged)
----------------------
propose_and_download_clips(raw_script, timestamps, *,
                           openai_api_key=None,
                           pixabay_api_key=None,
                           out_dir="./stock_clips",
                           model="gpt-4.1-mini",
                           per_page=10,
                           max_llm_retries=3,
                           request_timeout=15.0) -> list[dict]

What’s new
----------
- After picking a Pixabay candidate, we call `video_context_check.check_video_fitness`.
- If fitness == "yes": we download the Pixabay clip as before.
- If fitness == "no": we call Kie.ai Grok Imagine (`grok_text2video.text_to_video`)
  with the one-sentence prompt produced by the fitness checker (which is built
  ONLY from the narration sentence + purpose context), save the returned MP4, and
  use THAT clip instead of the Pixabay one.
- If Grok fails or the prompt is missing, we gracefully fall back to the Pixabay clip.

Env
---
OPENAI_API_KEY       : required (LLM query + fitness)
PIXABAY_KEY          : required (Pixabay search)
KIE_API_KEY          : required if Grok fallback is used (or pass via grok_text2video.api_key)
T2V_PURPOSE_CONTEXT  : optional; extra context for the T2V prompt writer
"""

from __future__ import annotations
import os
import re
import json
import time
import string
from pathlib import Path
from typing import List, Tuple, Optional

import requests

try:
    from openai import OpenAI
except Exception:
    from openai import OpenAI  # type: ignore

# -------------------------------------------------------------------------
# Optional semantic fitness + T2V prompt synth
# -------------------------------------------------------------------------
try:
    from video_context_check import check_video_fitness
except Exception:
    check_video_fitness = None  # Safe fallback if module missing

# -------------------------------------------------------------------------
# Optional Grok text-to-video fallback
# -------------------------------------------------------------------------
try:
    from grok_t2v_once import text_to_video, GrokT2VError
except Exception:
    text_to_video = None  # type: ignore
    GrokT2VError = RuntimeError  # type: ignore
    print("NO GROM!!")

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
API_BASE = "https://pixabay.com/api/videos/"
DEFAULT_SIZE_PREF = ["medium", "small", "large", "tiny"]
_SENT_END_RE = re.compile(r'(?<=[\.\!\?\u2026])\s+')

GROK_ASPECT_RATIO = "3:2"   # adjust if your workflow prefers 16:9, etc.
GROK_MODE = "normal"
T2V_PURPOSE_CONTEXT = os.getenv("T2V_PURPOSE_CONTEXT") or (
    "General restaurant workers training/e-learning b-roll to illustrate the narration clearly. "
    "Prefer neutral professional settings that match the sentence topic. "
    "Keep content brand-safe, workplace-appropriate, and free of trademarks/logos and identifiable faces."
)

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _split_into_sentences(raw: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END_RE.split(raw.strip()) if p.strip()]
    return parts

def _normalize_two_words(s: str) -> str:
    s = s.strip().lower()
    s = "".join(ch if ch in string.ascii_letters + " " else " " for ch in s)
    toks = [t for t in s.split() if t]
    if len(toks) < 2:
        toks = (toks + ["imagery"])[:2]
    return " ".join(toks[:2])

def _collapse_one_line(s: str) -> str:
    """Collapse any whitespace/newlines to a single-line string with spaces only."""
    return " ".join((s or "").replace("\n", " ").replace("\r", " ").split())

def _call_llm_two_word_query(client: OpenAI, model: str, sentence: str, attempt: int, note: str | None = None) -> str:
    sys = (
        "You output EXACTLY TWO ENGLISH WORDS for a Pixabay stock video search query.\n"
        "Two words only, lowercase ASCII, no punctuation, no quotes, no extra text."
    )
    user = f"Sentence: {sentence}\n"
    if note:
        user += f"Note: {note}\n"
    user += "Return only two words."
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=8,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
    )
    _ = time.perf_counter() - t0
    text = (resp.choices[0].message.content or "").strip()
    query = _normalize_two_words(text)
    return query

def _pick_best_rendition(hit: dict, size_pref: list[str]) -> tuple[str, float | None, str]:
    vids = hit.get("videos") or {}
    for k in size_pref:
        info = vids.get(k)
        if info and info.get("url"):
            url = info["url"]
            dur = float(hit.get("duration")) if hit.get("duration") is not None else None
            return url, dur, k
    raise RuntimeError("No playable rendition in hit")

def _search_pixabay(q: str, key: str, per_page: int, timeout: float, page: int = 1) -> list[dict]:
    params = {
        "key": key,
        "q": q,
        "video_type": "film",
        "safesearch": "true",
        "per_page": per_page,
        "order": "popular",
        "page": page,
    }
    t0 = time.perf_counter()
    r = requests.get(API_BASE, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    hits = data.get("hits", [])
    _ = time.perf_counter() - t0
    return hits

def _slugify(text: str, maxlen: int = 60) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:maxlen] or "clip"

def _head_size_mb(url: str, timeout: float = 10.0) -> Optional[float]:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        clen = r.headers.get("Content-Length")
        if clen is not None:
            return int(clen) / (1024 * 1024)
    except Exception:
        return None
    return None

def _download(url: str, dest: Path, timeout: float = 60.0) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                if chunk:
                    f.write(chunk)
    _ = time.perf_counter() - t0  # elapsed unused but preserved for symmetry

# -------------------------------------------------------------------------
# Main logic
# -------------------------------------------------------------------------
def propose_and_download_clips(
    raw_script: str,
    timestamps: List[Tuple[bytes, float, float]],
    *,
    openai_api_key: Optional[str] = None,
    pixabay_api_key: Optional[str] = None,
    out_dir: str = "./stock_clips",
    model: str = "gpt-4.1-mini",
    per_page: int = 10,
    max_llm_retries: int = 3,
    request_timeout: float = 15.0,
    size_pref: Optional[list] = None,
    page_jitter: int = 3,
) -> list[dict]:
    """
    Returns:
      list of dictionaries like:
      {
        "sentence_index": int,
        "sentence": str,
        "query": str,                    # two-word query chosen by LLM
        "pixabay_id": int,               # candidate hit id (even if Grok used)
        "source_url": str,               # Pixabay URL chosen (for traceability)
        "orig_duration": float | None,   # Pixabay's duration field if present
        "local_path": str,               # final chosen clip (Pixabay OR Grok)
        "source": "pixabay" | "grok" | "pixabay_fallback",
        # Fitness fields (when check ran):
        "description": str,
        "questionnaire": dict,
        "final_decision": "yes" | "no",
        "reason": str,
        "t2v_prompt": str | None
      }
    """
    t_total0 = time.perf_counter()

    sentences = _split_into_sentences(raw_script)
    if len(sentences) != len(timestamps):
        raise ValueError(f"Sentence count ({len(sentences)}) != timestamps count ({len(timestamps)}).")

    openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    pix_key = pixabay_api_key or os.getenv("PIXABAY_KEY")
    if not pix_key:
        raise RuntimeError("Missing PIXABAY_KEY.")

    size_pref = size_pref or DEFAULT_SIZE_PREF

    client = OpenAI(api_key=openai_key)
    used_urls: set[str] = set()
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for idx, sentence in enumerate(sentences):
        note = None
        last_candidate: dict | None = None

        # --------------------------
        # Step 1: Propose a candidate Pixabay URL
        # --------------------------
        for attempt in range(1, max_llm_retries + 1):
            query = _call_llm_two_word_query(client, model, sentence, attempt, note)
            page = 1 + ((idx + attempt - 1) % max(1, page_jitter))
            hits = _search_pixabay(query, pix_key, per_page=per_page, timeout=request_timeout, page=page)

            if not hits:
                note = f"'{query}' returned no results; try a different angle with two english words."
                continue

            chosen_hit = None
            chosen_url = None
            chosen_dur = None
            chosen_size_key = None
            dup_candidates = 0

            for h in hits:
                try:
                    url, dur, skey = _pick_best_rendition(h, size_pref)
                except Exception:
                    continue
                if url in used_urls:
                    dup_candidates += 1
                    continue
                chosen_hit = h
                chosen_url = url
                chosen_dur = dur
                chosen_size_key = skey
                break

            if chosen_hit is None:
                # As last resort, allow duplicate URL to make progress
                for h in hits:
                    try:
                        url, dur, skey = _pick_best_rendition(h, size_pref)
                        chosen_hit, chosen_url, chosen_dur, chosen_size_key = h, url, dur, skey
                        break
                    except Exception:
                        continue

            if chosen_hit is None or chosen_url is None:
                note = f"Could not extract a playable rendition for '{query}'. Try a different angle."
                continue

            hit_id = int(chosen_hit.get("id"))
            _ = _head_size_mb(chosen_url)

            last_candidate = {
                "sentence_index": idx,
                "sentence": sentence,
                "query": query,
                "pixabay_id": hit_id,
                "source_url": chosen_url,
                "orig_duration": chosen_dur,
            }

            used_urls.add(chosen_url)  # reserve this URL globally
            break

        if last_candidate is None:
            raise RuntimeError(f"Exhausted attempts for sentence {idx}: '{sentence}'.")

        # --------------------------
        # Step 2: Semantic fitness (maybe) → decide Pixabay vs Grok
        # --------------------------
        decision = "yes"
        t2v_prompt: Optional[str] = None
        grok_used = False

        if check_video_fitness is not None:
            try:
                # Pass purpose context so the T2V prompt writer knows our intent
                fitness = check_video_fitness(
                    sentence,
                    last_candidate["source_url"],
                    t2v_purpose_context=T2V_PURPOSE_CONTEXT,
                )
                last_candidate.update(fitness)  # add description, reason, decision, t2v_prompt
                decision = fitness.get("final_decision", "yes")
                t2v_prompt = fitness.get("t2v_prompt")
            except Exception as e:
                # If the check fails, proceed with Pixabay quietly
                last_candidate.setdefault("final_decision", "yes")
                last_candidate.setdefault("reason", f"fitness_check_failed: {type(e).__name__}")
                last_candidate.setdefault("description", "")
                last_candidate.setdefault("questionnaire", {})
                last_candidate.setdefault("t2v_prompt", None)

        # --------------------------
        # Step 3: Fetch & save the chosen clip
        # --------------------------
        if decision == "no" and text_to_video is not None:
            # Prefer the synthesized T2V prompt; fallback to narration sentence
            prompt = _collapse_one_line(t2v_prompt or sentence)

            try:
                mp4_bytes = text_to_video(
                    prompt=prompt,
                    # api_key=os.getenv("KIE_API_KEY"),  # optional explicit pass
                    aspect_ratio=GROK_ASPECT_RATIO,
                    mode=GROK_MODE,
                    poll_interval=5,
                    timeout=10 * 60,
                )
                # Save Grok output
                slug = _slugify(f"{idx+1:02d}-grok-{last_candidate['query']}-id{last_candidate['pixabay_id']}")
                local_path = out_dir_p / f"{slug}.mp4"
                with open(local_path, "wb") as f:
                    f.write(mp4_bytes)

                last_candidate["local_path"] = str(local_path)
                last_candidate["source"] = "grok"
                last_candidate["t2v_prompt"] = prompt
                grok_used = True

            except GrokT2VError as e:
                print("grokerror!!")
                # Grok failed — fall back to downloading the Pixabay candidate
                last_candidate["source"] = "pixabay_fallback"
                slug = _slugify(f"{idx+1:02d}-{last_candidate['query']}-id{last_candidate['pixabay_id']}")
                local_path = out_dir_p / f"{slug}.mp4"
                _download(last_candidate["source_url"], local_path, timeout=90.0)
                last_candidate["local_path"] = str(local_path)

        if not grok_used:
            # If fitness was "yes" OR Grok not available OR Grok failed → use Pixabay
            if "local_path" not in last_candidate:
                slug = _slugify(f"{idx+1:02d}-{last_candidate['query']}-id{last_candidate['pixabay_id']}")
                local_path = out_dir_p / f"{slug}.mp4"
                _download(last_candidate["source_url"], local_path, timeout=90.0)
                last_candidate["local_path"] = str(local_path)
            last_candidate.setdefault("source", "pixabay")

        results.append(last_candidate)

    _ = time.perf_counter() - t_total0
    return results


if __name__ == "__main__":
    demo_script = """Welcome to today’s short training session. We’re making Pasta Puttanesca, Sicilian style—a bright, briny classic that’s fast, flavorful, and perfect for service. By the end, you’ll know exactly what to do and why each step matters, so every plate tastes consistent and delicious.

Step 1 — Prepare the olive oil container.
Before we heat a pan, get the olive oil ready for smooth, controlled pouring. Make two quick slits in the top of the can. This gives you easy access and better control, helping you avoid messy glugs and oily countertops.
"""
    ts = [
        (b"", 0.0, 2.5), (b"", 2.5, 8.5), (b"", 8.5, 16.0),
        (b"", 16.0, 19.0), (b"", 19.0, 23.0),
        (b"", 23.0, 25.5), (b"", 25.5, 30.5),
    ]

    try:
        clips = propose_and_download_clips(
            demo_script,
            ts,
            per_page=12,
            max_llm_retries=3,
            size_pref=DEFAULT_SIZE_PREF,
            page_jitter=3,
        )
        print(json.dumps(clips, indent=2))
    except Exception as e:
        print("[DEMO] ERROR:", repr(e))
