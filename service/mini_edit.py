
# mini_edit.py
# Minimal FFmpeg-based utilities for robust, deployable video editing from Python.
# Requires: ffmpeg available on PATH (or provide an absolute path in FFMPEG env/argument).

import subprocess, tempfile, os, shlex, json

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")

def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error ({proc.returncode}):\n{proc.stdout}")

def probe_duration(path: str) -> float:
    cmd = [
        FFPROBE, "-v", "error", "-show_entries", "format=duration",
        "-of", "json", path
    ]
    out = subprocess.check_output(cmd, text=True)
    return float(json.loads(out)["format"]["duration"])

def cut(src: str, dst: str, start: float, end: float, reencode: bool = False, fps: int | None = None):
    """Trim a segment [start,end] from src to dst. If reencode=False it tries stream copy (fast)."""
    dur = max(0.0, end - start)
    if not reencode:
        cmd = [FFMPEG, "-y", "-ss", f"{start}", "-i", src, "-t", f"{dur}", "-c", "copy", dst]
    else:
        cmd = [FFMPEG, "-y", "-ss", f"{start}", "-i", src, "-t", f"{dur}", "-map", "0",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k"]
        if fps: cmd += ["-r", str(fps)]
        cmd += [dst]
    _run(cmd)

def normalize(src: str, dst: str, size: str = "1280x720", fps: int = 30, audio_bitrate: str = "160k"):
    """Transcode to a predictable MP4 (H.264/AAC), good for concat."""
    cmd = [FFMPEG, "-y", "-i", src,
           "-vf", f"scale={size},setsar=1", "-r", str(fps),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000", "-ac", "2",
           dst]
    _run(cmd)

def concat_normalized(mp4s: list[str], dst: str):
    """Concatenate already-normalized MP4s (matching codecs) using concat demuxer (no reencode)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in mp4s:
            f.write(f"file '{p}'\n")
        listfile = f.name
    try:
        cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", listfile, "-c", "copy", dst]
        _run(cmd)
    finally:
        os.remove(listfile)

def side_by_side(left: str, right: str, dst: str, height: int = 720, gap: int = 0, mix_audio: bool = False):
    """
    Place two clips side-by-side. Both are scaled to the same HEIGHT, widths preserved.
    If mix_audio=True, audio tracks are mixed; else take left clip's audio.
    """
    vf = (
        f"[0:v]scale=-1:{height},setsar=1[v0];"
        f"[1:v]scale=-1:{height},setsar=1[v1];"
        f"[v0][v1]hstack=inputs=2[v]"
    )
    maps = ["-map", "[v]"]
    if mix_audio:
        af = "[0:a][1:a]amix=inputs=2:normalize=0[a]"
        maps += ["-map", "[a]"]
        filter_complex = vf + ";" + af
    else:
        # Use left audio by default (map 0:a if exists)
        filter_complex = vf
        maps += ["-map", "0:a?"]

    cmd = [FFMPEG, "-y", "-i", left, "-i", right,
           "-filter_complex", filter_complex,
           *maps,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k",
           dst]
    _run(cmd)

def overlay_pip(background: str, overlay: str, dst: str, x: int = 30, y: int = 30, ow: int | None = 480, oh: int | None = None):
    """Picture-in-picture: scale overlay to (ow,oh) then place at (x,y)."""
    scale = f"scale={ow}:{oh}" if ow and oh else (f"scale={ow}:-1" if ow else "scale=-1:480")
    vf = f"[1:v]{scale}[ov];[0:v][ov]overlay={x}:{y}[v]"
    cmd = [FFMPEG, "-y", "-i", background, "-i", overlay,
           "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", dst]
    _run(cmd)

def burn_subtitles(src: str, srt_path: str, dst: str):
    """Burn an .srt file into the video (requires FFmpeg with libass/subtitles)."""
    # If your path has weird chars, escaping may be needed on Windows.
    vf = f"subtitles={shlex.quote(srt_path)}"
    cmd = [FFMPEG, "-y", "-i", src,
           "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k",
           dst]
    _run(cmd)
