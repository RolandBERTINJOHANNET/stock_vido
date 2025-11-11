#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_orchestrator.py — Step 2 orchestrator (stock selection + trimming + fades).

Public API
----------
build_stock_video_from_script(raw_script: str,
                              timestamps: list[tuple[bytes, float, float]],
                              *,
                              openai_api_key: str | None = None,
                              pixabay_api_key: str | None = None,
                              downloads_dir: str = "./stock_clips",
                              model: str = "gpt-4.1-mini",
                              fade_duration: float = 0.3,
                              target_fps: int = 30,
                              target_size: tuple[int, int] | None = None) -> bytes

- Uses video1.propose_and_download_clips → unique, downloaded clips per sentence.
- Uses video2.cut_and_stitch_clips → exact-length segments with non-offset fades.
- Returns: MP4 bytes (H.264). Duration matches exactly the final audio duration from step1.
"""

from __future__ import annotations
from typing import List, Tuple, Optional

from video1 import propose_and_download_clips
from video2 import cut_and_stitch_clips


def build_stock_video_from_script(
    raw_script: str,
    timestamps: List[Tuple[bytes, float, float]],
    *,
    openai_api_key: Optional[str] = None,
    pixabay_api_key: Optional[str] = None,
    downloads_dir: str = "./stock_clips",
    model: str = "gpt-4.1-mini",
    fade_duration: float = 0.3,
    target_fps: int = 30,
    target_size: tuple[int, int] | None = None,
) -> bytes:
    """
    High-level entrypoint for Step 2.

    Args:
        raw_script: The *raw* (un-tagged) script text used in step1 audio.
        timestamps: The sentence-aligned timestamps from step1:
                    list of (segment_bytes, start_sec, end_sec)
        openai_api_key / pixabay_api_key: Optional explicit keys; else read env.
        downloads_dir: Where to store the downloaded stock videos.
        model: The same model as step1 (default 'gpt-4.1-mini').
        fade_duration: Visual fade-in/out seconds per segment (no offset added).
        target_fps: Output FPS.
        target_size: Optional (width, height). Defaults to the first clip's size.

    Returns:
        MP4 bytes comprising all sentence clips stitched together with fades.
    """
    clips = propose_and_download_clips(
        raw_script,
        timestamps,
        openai_api_key=openai_api_key,
        pixabay_api_key=pixabay_api_key,
        out_dir=downloads_dir,
        model=model,
    )
    stitched = cut_and_stitch_clips(
        clips,
        timestamps,
        fade_duration=fade_duration,
        target_fps=target_fps,
        target_size=target_size,
    )
    return stitched


if __name__ == "__main__":
    demo_script = "Wash your hands. Keep your station clean."
    ts = [(b"", 0.0, 2.0), (b"", 2.0, 5.0)]  # example (same shape as step1)
    mp4_bytes = build_stock_video_from_script(demo_script, ts)
    with open("demo_step2.mp4", "wb") as f:
        f.write(mp4_bytes)
    print("Wrote demo_step2.mp4")
