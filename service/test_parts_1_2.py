#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
part_1_2_test.py — End-to-end tryout (audio+video+captions)

What this does:
1) Uses Part-1 to tag + TTS your hardcoded script -> returns (timestamps, stitched_mp3_bytes)
2) Uses Part-2 to fetch/trim/fade/concat stock clips -> returns silent mp4 bytes matching timestamps
3) Builds an SRT from the timestamps + sentence text
4) Burns captions on the silent video
5) Muxes the stitched MP3 with the captioned video -> final_preview.mp4

Env required:
  OPENAI_API_KEY   (Step-1 tagging and Step-2 LLM queries for keywords)
  ELEVEN_API_KEY   (Step-1 ElevenLabs TTS)
  PIXABAY_KEY      (Step-2 stock search)

Optional:
  ELEVEN_VOICE_ID or ELEVEN_VOICE_NAME
  FFMPEG / FFPROBE env to point to binaries (else they must be in PATH)

Outputs (in ./tryout_out/):
  - stitched_audio.mp3
  - visual_silent.mp4
  - captions.srt
  - video_with_captions.mp4
  - final_preview.mp4   (video+audio, with burned captions)
"""

from __future__ import annotations
import os, re, sys, json, tempfile, subprocess
from pathlib import Path
from typing import List, Tuple

# --- Imports from your modules (already implemented in your project) ---
from audio_orchestrator import get_audio_and_timestamps
from video_orchestrator import build_stock_video_from_script
from mini_edit import FFMPEG, FFPROBE, burn_subtitles, probe_duration

# ------------------------
# Hardcoded test script
# ------------------------
TEST_SCRIPT = """Welcome to today’s short training session. We’re making Pasta Puttanesca, Sicilian style—a bright, briny classic that’s fast, flavorful, and perfect for service. By the end, you’ll know exactly what to do and why each step matters, so every plate tastes consistent and delicious.

Step 1 — Prepare the olive oil container.
Before we heat a pan, get the olive oil ready for smooth, controlled pouring. Make two quick slits in the top of the can. This gives you easy access and better control, helping you avoid messy glugs and oily countertops.
"""

# ---------- helpers ----------

def _require_env(keys: List[str]) -> None:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + "\nSet them or pass through your process runner."
        )

_SENT_PAT = re.compile(r"\s*(.+?[\.!?…])(?:\s+|$)", re.S)

def split_sentences_like_step1(text: str) -> List[str]:
    """
    Mirrors the simple terminal-punctuation splitter used in docs for Step-1/2.
    Keeps only non-empty sentences that end with . ! ? or …
    """
    out = []
    for m in _SENT_PAT.finditer(text.strip()):
        s = m.group(1).strip()
        if s:
            out.append(s)
    return out

def srt_timestamp(t: float) -> str:
    """Convert seconds -> SRT hh:mm:ss,mmm"""
    if t < 0:
        t = 0.0
    hrs = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"

def build_srt(sentences: List[str], timestamps: List[Tuple[bytes, float, float]]) -> str:
    if len(sentences) != len(timestamps):
        raise ValueError(
            f"Sentence count ({len(sentences)}) != timestamps count ({len(timestamps)}). "
            "Ensure your test splitter matches Step-1 segmentation."
        )
    lines = []
    for i, (seg_bytes, start, end) in enumerate(timestamps, start=1):
        lines.append(str(i))
        lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
        lines.append(sentences[i-1])
        lines.append("")  # blank line
    return "\n".join(lines) + "\n"

def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\n{proc.stdout}")

# ---------- main pipeline ----------

def main():
    out_dir = Path("./tryout_out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0) Check env upfront (fail fast)
    _require_env(["OPENAI_API_KEY", "ELEVEN_API_KEY", "PIXABAY_KEY"])

    # 1) Part-1: audio + timestamps
    print("[1/5] Generating stitched audio + timestamps via Part-1 ...")
    timestamps, stitched_audio = get_audio_and_timestamps(
        raw_script=TEST_SCRIPT,
        add_random_pauses=False,   # keep clean mapping for the tryout
        temperature=0.2,
        sleep_between_calls=0.05,  # gentle cushion
    )
    audio_path = out_dir / "stitched_audio.mp3"
    write_bytes(audio_path, stitched_audio)
    total_audio_s = timestamps[-1][2] if timestamps else 0.0
    print(f"    -> segments: {len(timestamps)} | total audio: {total_audio_s:.2f}s | saved: {audio_path}")

    # 2) Part-2: build silent video matching timestamps
    print("[2/5] Building silent stock video via Part-2 ...")
    visual_bytes = build_stock_video_from_script(
        raw_script=TEST_SCRIPT,
        timestamps=timestamps,
        model="gpt-4.1-mini",
        fade_duration=0.3,
        target_fps=30,
        target_size=None,  # inherit from first clip
    )
    visual_path = out_dir / "visual_silent.mp4"
    write_bytes(visual_path, visual_bytes)
    try:
        vdur = probe_duration(str(visual_path))
        print(f"    -> saved: {visual_path} | duration (probe): {vdur:.2f}s")
    except Exception as e:
        print(f"    (warn) ffprobe failed to read duration: {e}")

    # 3) Build SRT from timestamps + our sentence splitter
    print("[3/5] Creating captions SRT from timestamps ...")
    sentences = split_sentences_like_step1(TEST_SCRIPT)
    srt_text = build_srt(sentences, timestamps)
    srt_path = out_dir / "captions.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_text)
    print(f"    -> saved: {srt_path}")

    # 4) Burn captions on the silent video
    print("[4/5] Burning captions onto video ...")
    captioned_path = out_dir / "video_with_captions.mp4"
    burn_subtitles(str(visual_path), str(srt_path), str(captioned_path))
    try:
        vcap_dur = probe_duration(str(captioned_path))
        print(f"    -> saved: {captioned_path} | duration (probe): {vcap_dur:.2f}s")
    except Exception as e:
        print(f"    (warn) ffprobe failed to read captioned duration: {e}")

    # 5) Mux stitched audio with the captioned video (copy video stream, shortest)
    print("[5/5] Muxing stitched audio with captioned video ...")
    final_path = out_dir / "final_preview.mp4"
    cmd = [
        FFMPEG, "-y",
        "-i", str(captioned_path),
        "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy",
        "-shortest",
        str(final_path),
    ]
    run(cmd)
    try:
        f_dur = probe_duration(str(final_path))
        print(f"    -> saved: {final_path} | duration (probe): {f_dur:.2f}s")
    except Exception as e:
        print(f"    (warn) ffprobe failed to read final duration: {e}")

    # Sanity: audio total ~= final video duration (allow tiny drift)
    print("\n=== Sanity Check ===")
    print(f"Total audio (from timestamps): {total_audio_s:.3f}s")
    print(f"Final video (probe):           {f_dur:.3f}s (approx)")
    print("Expect them to be nearly equal. If they differ, check fades and concat settings.")

def render_final_video(raw_script: str) -> bytes:
    """
    Public API
    ----------
    render_final_video(raw_script: str) -> bytes
      1) Part-1: tag+TTS per sentence → (timestamps, stitched MP3 bytes)
      2) Part-2: Pixabay (+ fitness + Grok fallback) → silent MP4 (timing-matched)
      3) Build SRT from (sentences, timestamps), burn captions
      4) Mux captioned video + stitched audio (copy video, shortest)
      5) Return final MP4 bytes (no files written to repo)
    """
    # 1) Audio + timestamps (deterministic config)
    timestamps, stitched_audio = get_audio_and_timestamps(
        raw_script=raw_script,
        add_random_pauses=False,
        temperature=0.2,
        sleep_between_calls=0.0,
    )

    # 2) Silent video matching the timestamps
    visual_bytes = build_stock_video_from_script(
        raw_script=raw_script,
        timestamps=timestamps,
        model="gpt-4.1-mini",
        fade_duration=0.3,
        target_fps=30,
        target_size=None,  # inherit from first clip
    )

    # 3) Build captions using the same simple splitter used in the tryout
    sentences = split_sentences_like_step1(raw_script)
    srt_text = build_srt(sentences, timestamps)

    # 4) Burn SRT and mux with stitched audio in a temp workspace
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        v_silent = td / "visual_silent.mp4"
        v_captioned = td / "video_with_captions.mp4"
        srt_path = td / "captions.srt"
        a_path = td / "audio.mp3"
        final_path = td / "final.mp4"

        # write temp artifacts
        v_silent.write_bytes(visual_bytes)
        a_path.write_bytes(stitched_audio)
        srt_path.write_text(srt_text, encoding="utf-8")

        # burn subtitles
        burn_subtitles(str(v_silent), str(srt_path), str(v_captioned))

        # mux audio (copy video, shortest)
        cmd = [
            FFMPEG, "-y",
            "-i", str(v_captioned),
            "-i", str(a_path),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy",
            "-shortest",
            str(final_path),
        ]
        run(cmd)

        return final_path.read_bytes()



if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
