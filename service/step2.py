#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 — TTS Orchestration (robust ffmpeg fallback):
- Input: a "tagged" script where each sentence begins with a bracketed tag (output of Step 1).
- For each sentence: call ElevenLabs TTS (via elabs_audio.speak_tagged_sentence).
- Measure durations → build (audio_bytes, start_s, end_s) tuples.
- Stitch to a single MP3 (44100 Hz, 128 kbps).
- Return: (timestamps_list, stitched_audio_bytes).

Public API:
    synthesize_audio_from_tagged_script(tagged_script: str)
        -> tuple[list[tuple[bytes, float, float]], bytes]

Dev/CLI demo:
- Uses Step 1's TEST_SCRIPT on __main__:
  1) add_emotion_tags_to_script (Step 1),
  2) synthesize_audio_from_tagged_script (Step 2),
  3) writes 'stitched_demo.mp3' and prints a timeline.

Env needed at runtime:
  ELEVEN_API_KEY   (required by elabs_audio)
  ELEVEN_VOICE_ID / ELEVEN_VOICE_NAME (optional, voice selection)

Optional packages:
  mutagen (preferred for MP3 duration)
  pydub (optional for concatenation and/or duration; needs ffmpeg)

Optional env overrides for CLI fallbacks:
  FFMPEG_BIN  (default: "ffmpeg")
  FFPROBE_BIN (default: "ffprobe")
"""

from __future__ import annotations
import io
import os
import re
import sys
import time
import subprocess
import tempfile
from typing import List, Tuple

# --- Local deps: Step 1 and ElevenLabs wrapper ---
# Ensure these files exist alongside step2.py:
#   - step1.py (from previous step)
#   - elabs_audio.py (provided by you)
from step1 import add_emotion_tags_to_script, TEST_SCRIPT
from elabs_audio import speak_tagged_sentence

# --- Optional duration helpers (mutagen, pydub are optional) ---
_DURATION_BACKENDS = {
    "mutagen": True,
    "pydub": True,
}
try:
    from mutagen.mp3 import MP3 as MutagenMP3
except Exception:
    _DURATION_BACKENDS["mutagen"] = False

# On Python 3.13, pydub requires 'pyaudioop' (drop-in for removed stdlib audioop).
# We keep pydub optional and gracefully fall back to ffmpeg CLI if missing.
try:
    from pydub import AudioSegment
    from pydub.utils import which as _pydub_which
except Exception:
    AudioSegment = None  # type: ignore[assignment]
    _pydub_which = None  # type: ignore[assignment]
    _DURATION_BACKENDS["pydub"] = False

# Allow binary overrides via env
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.environ.get("FFPROBE_BIN", "ffprobe")


# ==============================
# Sentence splitting (tagged)
# ==============================
_SENTENCE_RE = re.compile(r'(.+?(?:[.!?]))(\s+|$)', re.DOTALL)

def _split_sentences_with_separators(text: str) -> List[Tuple[str, str]]:
    """
    Returns [(sentence, separator_after), ...], preserving whitespace/newlines.
    Sentences end on ., !, or ? (greedy minimal match).
    Trailing text without terminal punctuation becomes its own sentence.
    """
    parts: List[Tuple[str, str]] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(text):
        sent = m.group(1)
        sep = m.group(2)
        parts.append((sent, sep))
        pos = m.end()
    if pos < len(text):
        tail = text[pos:]
        if tail.strip():
            parts.append((tail, ""))
    return parts


# ==============================
# Duration measurement
# ==============================
def _mp3_duration_seconds(data: bytes) -> float:
    """
    Best-effort duration extraction for MP3 bytes.
    Try: mutagen → pydub → ffprobe CLI.
    """
    # 1) mutagen (no ffmpeg needed)
    if _DURATION_BACKENDS.get("mutagen", False):
        try:
            bio = io.BytesIO(data)
            return float(MutagenMP3(bio).info.length)
        except Exception:
            pass

    # 2) pydub (requires ffmpeg)
    if _DURATION_BACKENDS.get("pydub", False) and AudioSegment is not None:
        try:
            # Ensure ffmpeg is discoverable by pydub; if not, force common paths
            try:
                if _pydub_which is not None and _pydub_which("ffmpeg") is None:
                    # Typical Ubuntu paths; adjust if necessary.
                    AudioSegment.converter = "/usr/bin/ffmpeg"
                    AudioSegment.ffmpeg = "/usr/bin/ffmpeg"
                    AudioSegment.ffprobe = "/usr/bin/ffprobe"
            except Exception:
                pass
            seg = AudioSegment.from_file(io.BytesIO(data), format="mp3")
            return len(seg) / 1000.0
        except Exception:
            pass

    # 3) ffprobe CLI (final fallback)
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            cmd = [
                FFPROBE_BIN, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                tmp.name,
            ]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            return float(out.decode("utf-8").strip())
    except Exception:
        pass

    raise RuntimeError(
        "Unable to measure MP3 duration. Install 'mutagen' or 'pydub'+ffmpeg, or ensure ffprobe is available."
    )


# ==============================
# Concatenation (export MP3)
# ==============================
def _concat_mp3_segments(segments: List[bytes], bitrate: str = "128k") -> bytes:
    """
    Concatenate MP3 segments and return MP3 bytes.
    Try pydub+ffmpeg; if unavailable or fails, use ffmpeg CLI concat demuxer.
    """
    # --- Try pydub path first ---
    if _DURATION_BACKENDS.get("pydub", False) and AudioSegment is not None:
        try:
            # Ensure ffmpeg is found; if not, force common paths
            try:
                if _pydub_which is not None and _pydub_which("ffmpeg") is None:
                    AudioSegment.converter = "/usr/bin/ffmpeg"
                    AudioSegment.ffmpeg = "/usr/bin/ffmpeg"
                    AudioSegment.ffprobe = "/usr/bin/ffprobe"
            except Exception:
                pass

            combined = None
            for blob in segments:
                seg = AudioSegment.from_file(io.BytesIO(blob), format="mp3")
                combined = seg if combined is None else (combined + seg)

            buff = io.BytesIO()
            # Re-encode to MP3 44.1kHz @ 128kbps to match ElevenLabs default
            combined.export(buff, format="mp3", bitrate=bitrate, parameters=["-ar", "44100"])
            return buff.getvalue()
        except Exception:
            # fall through to ffmpeg CLI
            pass

    # --- Fallback: ffmpeg concat demuxer ---
    # We first try stream copy (-c copy); if that fails, we re-encode to libmp3lame 128k 44.1kHz.
    with tempfile.TemporaryDirectory() as td:
        list_path = os.path.join(td, "list.txt")
        out_path = os.path.join(td, "out.mp3")
        part_paths = []
        for i, blob in enumerate(segments):
            p = os.path.join(td, f"{i:03d}.mp3")
            with open(p, "wb") as f:
                f.write(blob)
            part_paths.append(p)
        with open(list_path, "w") as lf:
            for p in part_paths:
                # Use concat demuxer file list format
                lf.write(f"file '{p}'\n")

        # 1) Attempt stream copy (gapless, fastest)
        cmd_copy = [
            FFMPEG_BIN, "-y", "-v", "error",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_path,
        ]
        try:
            subprocess.run(cmd_copy, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            # 2) Fallback: re-encode (consistent output)
            cmd_reenc = [
                FFMPEG_BIN, "-y", "-v", "error",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100",
                out_path,
            ]
            subprocess.run(cmd_reenc, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with open(out_path, "rb") as f:
            return f.read()


# =========================================================
# Public API
# =========================================================
def synthesize_audio_from_tagged_script(
    tagged_script: str,
    sleep_between_calls: float = 0.0,
) -> Tuple[List[Tuple[bytes, float, float]], bytes]:
    """
    Given a TAGGED script (each sentence starts with a bracketed tag):
      - Split into sentences (ignores separators for TTS calls),
      - Call ElevenLabs TTS per sentence (strict 1 sentence),
      - Compute precise timestamps,
      - Concatenate all audio into one MP3,
      - Return (timestamps_list, stitched_audio_bytes).

    Returns:
      timestamps_list: list of (audio_bytes, start_sec, end_sec)
          where 'audio_bytes' are the per-sentence MP3 bytes.
      stitched_audio_bytes: the single concatenated MP3 bytes.
    """
    # Split into sentences; ignore empty fragments
    pairs = _split_sentences_with_separators(tagged_script)
    sentences = [s.strip() for (s, _sep) in pairs if s.strip()]

    # TTS per sentence
    per_audio: List[bytes] = []
    durations: List[float] = []
    for idx, line in enumerate(sentences, start=1):
        # ElevenLabs function enforces: starts with [tag], exactly one sentence, no newlines.
        audio = speak_tagged_sentence(line)
        dur = _mp3_duration_seconds(audio)
        per_audio.append(audio)
        durations.append(dur)
        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    # Build timestamps
    timestamps: List[Tuple[bytes, float, float]] = []
    t = 0.0
    for blob, d in zip(per_audio, durations):
        start = t
        end = t + d
        timestamps.append((blob, start, end))
        t = end

    # Stitch all
    stitched = _concat_mp3_segments(per_audio)

    return timestamps, stitched


# =========================================================
# CLI Demo (uses Step 1's TEST_SCRIPT)
# =========================================================
def _print_timeline(sentences: List[str], timestamps: List[Tuple[bytes, float, float]]) -> None:
    width = len(str(len(sentences)))
    for i, (line, (_blob, s, e)) in enumerate(zip(sentences, timestamps), start=1):
        print(f"{str(i).zfill(width)} | {s:8.3f}s → {e:8.3f}s | {line}")

def main() -> int:
    print("=== Step 2 Demo: Synthesize audio from tagged script (robust) ===")
    print("Requires ELEVEN_API_KEY (and optionally ELEVEN_VOICE_ID/NAME).")
    print("Optional: mutagen (duration), pydub+ffmpeg (concat); otherwise we fall back to ffmpeg CLI.\n")

    # 1) Use Step 1 to add tags to the hardcoded test script
    try:
        print("[1/3] Running Step 1 (add_emotion_tags_to_script) ...")
        tagged = add_emotion_tags_to_script(TEST_SCRIPT)
    except Exception as e:
        print(f"Step 1 failed: {e}", file=sys.stderr)
        return 2

    # 2) Run Step 2
    try:
        print("[2/3] Synthesizing with ElevenLabs, please wait ...")
        timestamps, stitched = synthesize_audio_from_tagged_script(
            tagged_script=tagged,
            sleep_between_calls=0.0
        )
    except Exception as e:
        print(f"Step 2 failed: {e}", file=sys.stderr)
        return 3

    # 3) Save stitched audio & print timeline
    out_path = os.path.abspath("stitched_demo.mp3")
    try:
        with open(out_path, "wb") as f:
            f.write(stitched)
        print(f"\n[3/3] Saved final stitched audio → {out_path}")
    except Exception as e:
        print(f"Could not write output MP3: {e}", file=sys.stderr)

    # Print a clean textual timeline
    pairs = _split_sentences_with_separators(tagged)
    sentences = [s.strip() for (s, _sep) in pairs if s.strip()]
    print("\n--- TIMELINE ---")
    _print_timeline(sentences, timestamps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
