#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video2.py — FFmpeg-only implementation (no MoviePy).

Builds a stitched, silent MP4 that matches sentence-level audio timings exactly:
- per sentence: trim or loop to the exact duration (end - start)
- add non-crossfade fades (in/out to black) that DO NOT change segment length
- normalize all segments (same size/fps/codec) for concat-demuxer copy
- concat segments losslessly and return final MP4 as bytes

Requirements:
- ffmpeg/ffprobe available (or set FFMPEG / FFPROBE env vars)
- mini_edit.py in the same directory (we reuse its constants & concat helper)
"""

from __future__ import annotations
import json, math, os, subprocess, tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# Reuse your mini_edit constants/helpers
from mini_edit import FFMPEG, FFPROBE, concat_normalized

# -------------------------- low-level helpers --------------------------

def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error ({proc.returncode}):\n{proc.stdout}")

def _probe_resolution(path: str) -> tuple[int, int]:
    """Return (width, height) of the first video stream via ffprobe."""
    cmd = [FFPROBE, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "json", path]
    out = subprocess.check_output(cmd, text=True)
    j = json.loads(out)
    s = j.get("streams", [{}])[0]
    return int(s.get("width", 1280)), int(s.get("height", 720))

def _segment_with_fades_and_norm(src: str,
                                 dst: str,
                                 desired: float,
                                 *,
                                 size: tuple[int, int],
                                 fps: int,
                                 fade_duration: float,
                                 preset: str,
                                 crf: int) -> None:
    """
    Create a per-sentence segment at exact `desired` seconds, with fades and normalized params.
    If src is shorter than desired, we loop it using -stream_loop N. Audio is stripped (-an).
    """
    # Robust safety on tiny durations
    desired = max(0.1, float(desired))
    fd = max(0.0, min(float(fade_duration), desired / 2.0))

    # How many loops do we need?
    # If src >= desired: no looping. Else: loops = ceil(desired / src) - 1
    # We probe src duration using ffprobe (fast).
    src_dur = _probe_duration_quick(src)
    loops = max(0, math.ceil(desired / max(src_dur, 0.001)) - 1)

    w, h = size
    scale = f"scale={w}:{h},setsar=1"
    fpsf = f"fps={fps}"

    # Fades are duration-neutral: fade in at 0, fade out starting at desired - fd
    # Note: place fps AFTER fades to avoid fractional-frame trimming issues.
    if fd > 0:
        vf = f"{scale},format=yuv420p,fade=t=in:st=0:d={fd},fade=t=out:st={max(0.0, desired - fd)}:d={fd},{fpsf}"
    else:
        vf = f"{scale},format=yuv420p,{fpsf}"

    # Build command
    cmd = [FFMPEG, "-y"]
    if loops > 0:
        cmd += ["-stream_loop", str(loops)]
    cmd += [
        "-i", src,
        "-t", f"{desired}",
        "-an",  # no audio in visual track; audio comes from stitched MP3 later
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        dst,
    ]
    _run(cmd)

def _probe_duration_quick(path: str) -> float:
    cmd = [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    out = subprocess.check_output(cmd, text=True)
    return float(json.loads(out)["format"]["duration"])

# ------------------------------ public API ------------------------------

def cut_and_stitch_clips(
    clips: List[dict],
    timestamps: List[Tuple[bytes, float, float]],
    *,
    fade_duration: float = 0.3,
    target_fps: int = 30,
    target_size: Optional[tuple[int, int]] = None,
    preset: str = "veryfast",
    crf: int = 20,
) -> bytes:
    """
    Build a single stitched MP4 (bytes) with per-segment fades that DO NOT change segment durations.

    Args:
        clips: output list from video1.propose_and_download_clips (needs 'local_path').
        timestamps: list of (segment_bytes, start_sec, end_sec).
        fade_duration: per-segment fade in & out to black (seconds), duration-neutral.
        target_fps: output fps for all segments.
        target_size: (w, h). If None, use the resolution of the first clip.
        preset/crf: libx264 settings used for per-segment normalization.

    Returns:
        Final stitched **silent** MP4 as bytes.
    """
    if len(clips) != len(timestamps):
        raise ValueError(f"Clip count ({len(clips)}) != timestamps count ({len(timestamps)})")

    if not clips:
        raise ValueError("No clips provided.")

    # Determine target size if not specified
    first_path = clips[0]["local_path"]
    if target_size is None:
        target_size = _probe_resolution(first_path)

    tmp_dir = Path(tempfile.mkdtemp(prefix="video2_ffmpeg_"))
    seg_paths: list[str] = []

    try:
        # Create normalized, faded, exact-duration segments
        for i, (c, ts) in enumerate(zip(clips, timestamps)):
            src = c["local_path"]
            _, start_s, end_s = ts
            desired = max(0.0, float(end_s) - float(start_s))
            # Safety for zero/negative durations
            if desired <= 0.0:
                desired = 0.1

            out_seg = tmp_dir / f"seg_{i:03d}.mp4"
            _segment_with_fades_and_norm(
                src=src,
                dst=str(out_seg),
                desired=desired,
                size=target_size,
                fps=target_fps,
                fade_duration=fade_duration,
                preset=preset,
                crf=crf,
            )
            seg_paths.append(str(out_seg))

        # Concat all segments (all share same codec/params -> fast, no re-encode)
        final_path = tmp_dir / "stitched.mp4"
        concat_normalized(seg_paths, str(final_path))

        data = final_path.read_bytes()
        return data

    finally:
        # Best-effort cleanup of temp dir
        try:
            for p in sorted(tmp_dir.glob("*")):
                try: p.unlink()
                except Exception: pass
            tmp_dir.rmdir()
        except Exception:
            pass

# ------------------------------ CLI demo --------------------------------

if __name__ == "__main__":
    # This module is meant to be used from the step-2 orchestrator.
    print("video2 (FFmpeg) ready: import cut_and_stitch_clips(...)")
