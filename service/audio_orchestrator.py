#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator — raw script → tagged script → per-sentence TTS → stitched MP3.

Public API:
    get_audio_and_timestamps(
        raw_script: str,
        *,
        model: str = "gpt-4.1-mini",
        batch_size: int = 5,
        temperature: float = 0.2,
        max_retries: int = 2,
        openai_api_key: str | None = None,
        add_random_pauses: bool = False,
        pause_p: float = 0.5,
        pause_seed: int | None = None,
        sleep_between_calls: float = 0.0,
    ) -> tuple[list[tuple[bytes, float, float]], bytes]

Returns:
    (timestamps_list, stitched_audio_bytes)
      - timestamps_list: [(segment_bytes, start_s, end_s), ...]
      - stitched_audio_bytes: single MP3 with all segments concatenated

Notes:
  - Requires ELEVEN_API_KEY in the environment for TTS (used by elabs_audio via step2).
  - Optionally uses OPENAI_API_KEY for tagging (step1).
"""

from __future__ import annotations
import os
import sys
from typing import List, Tuple

# Step 1: tagging + optional [pause]
from step1 import (
    add_emotion_tags_to_script,
    randomly_append_pause_tags,
    TEST_SCRIPT,   # for demo
)

# Step 2: per-sentence TTS + stitching (robust ffmpeg fallback inside)
from step2 import synthesize_audio_from_tagged_script

# (Optional import to make the dependency explicit; not used directly here)
try:
    from elabs_audio import speak_tagged_sentence  # noqa: F401
except Exception:
    # elabs_audio is used by step2; we don't need it here if step2 is importable.
    pass


def get_audio_and_timestamps(
    raw_script: str,
    *,
    model: str = "gpt-4.1-mini",
    batch_size: int = 5,
    temperature: float = 0.2,
    max_retries: int = 2,
    openai_api_key: str | None = None,
    add_random_pauses: bool = False,
    pause_p: float = 0.5,
    pause_seed: int | None = None,
    sleep_between_calls: float = 0.0,
) -> Tuple[List[Tuple[bytes, float, float]], bytes]:
    """
    Full pipeline:
      1) Tag each sentence with an emotion tag (LLM; sentences unchanged).
      2) Optionally append [pause] to ~p of sentences (before punctuation).
      3) ElevenLabs TTS per sentence + stitching to a single MP3.

    Returns:
      timestamps_list, stitched_audio_bytes
    """
    # 1) Tagging
    tagged = add_emotion_tags_to_script(
        raw_text=raw_script,
        model=model,
        batch_size=batch_size,
        temperature=temperature,
        max_retries=max_retries,
        openai_api_key=openai_api_key,
    )

    # 2) Optional random [pause]
    if add_random_pauses:
        tagged = randomly_append_pause_tags(tagged, p=pause_p, seed=pause_seed)

    # 3) TTS per sentence + stitching
    timestamps, stitched = synthesize_audio_from_tagged_script(
        tagged_script=tagged,
        sleep_between_calls=sleep_between_calls,
    )
    return timestamps, stitched


# ----------------
# Demo CLI runner
# ----------------
def main() -> int:
    print("=== Orchestrator demo: raw script → audio+timestamps ===")
    if not os.getenv("ELEVEN_API_KEY"):
        print("Warning: ELEVEN_API_KEY not set — TTS will fail.", file=sys.stderr)

    # Use the test script from step1
    raw = TEST_SCRIPT
    try:
        timestamps, stitched = get_audio_and_timestamps(
            raw_script=raw,
            add_random_pauses=True,     # demo: sprinkle [pause]
            pause_seed=42,
            sleep_between_calls=0.0,
        )
    except Exception as e:
        print(f"Pipeline failed: {e}", file=sys.stderr)
        return 2

    # Save stitched output
    out_path = os.path.abspath("final_stitched.mp3")
    try:
        with open(out_path, "wb") as f:
            f.write(stitched)
        print(f"Saved stitched audio → {out_path}")
        print(f"Segments: {len(timestamps)}")
        if timestamps:
            total = timestamps[-1][2]
            print(f"Total duration: {total:.2f}s")
    except Exception as e:
        print(f"Could not write MP3: {e}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
