"""FFmpeg rendering engine.

Renders scene-by-scene (spec section 10/38): each scene clip is produced
independently and released, then concatenated via FFmpeg's concat demuxer
so we never hold multiple decoded videos in memory at once. This keeps
peak RAM usage low even on an 8GB local machine.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from core.exceptions import RenderError


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg command failed: {' '.join(cmd)}\n{proc.stderr[-3000:]}")


def concat_clips(clip_paths: list[str], output_path: str) -> str:
    """Concatenate pre-rendered scene clips (all same codec/resolution) via the
    concat demuxer — no re-encoding of each clip individually is required
    beforehand, only a stream copy at concat time when formats match."""
    if not clip_paths:
        raise RenderError("No clips supplied to concat_clips")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for clip in clip_paths:
            f.write(f"file '{Path(clip).resolve()}'\n")
        list_path = f.name

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy", output_path,
    ]
    try:
        _run(cmd)
    except RenderError:
        # Fall back to re-encoding concat if stream copy fails due to
        # mismatched parameters between clips.
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path,
        ]
        _run(cmd)
    finally:
        Path(list_path).unlink(missing_ok=True)

    return output_path


def mux_audio(video_path: str, audio_path: str, output_path: str) -> str:
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-shortest", output_path,
    ]
    _run(cmd)
    return output_path


def burn_captions(video_path: str, srt_path: str, output_path: str) -> str:
    # subtitles filter requires the srt path to not need extra escaping in
    # the common case; for paths with special chars this would need
    # escaping, kept simple here since project paths are controlled.
    vf = f"subtitles={srt_path}"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", vf, "-c:a", "copy", output_path]
    _run(cmd)
    return output_path


def render_still_frame_video(image_path: str, duration_seconds: float, output_path: str,
                              width: int, height: int, fps: int) -> str:
    """Trivial fallback: a static (non-animated) clip from a still image."""
    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path,
        "-t", str(duration_seconds), "-vf", f"scale={width}:{height}",
        "-pix_fmt", "yuv420p", "-r", str(fps), output_path,
    ]
    _run(cmd)
    return output_path


def get_video_info(path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    info: dict = {}
    if proc.returncode == 0:
        for line in proc.stdout.strip().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                info[k] = v
    return info
