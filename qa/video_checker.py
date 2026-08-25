"""Video QA checks: resolution, aspect ratio, fps, and basic black/short
duration heuristics via ffprobe. Avoids frame-by-frame decoding into
memory (spec section 10) by relying on ffprobe metadata and FFmpeg's
`blackdetect` filter (stream-processed, not buffered).
"""
from __future__ import annotations

import subprocess

from core.models import QAIssue


def check_video(path: str, expected_width: int, expected_height: int, expected_fps: int,
                 min_duration_seconds: float = 1.0) -> list[QAIssue]:
    issues: list[QAIssue] = []

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration",
        "-of", "default=noprint_wrappers=1", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        issues.append(QAIssue(category="video", severity="critical", message=f"ffprobe failed on {path}"))
        return issues

    info = {}
    for line in proc.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v

    width = int(info.get("width", 0) or 0)
    height = int(info.get("height", 0) or 0)
    if width != expected_width or height != expected_height:
        issues.append(QAIssue(
            category="resolution", severity="critical",
            message=f"Expected {expected_width}x{expected_height}, got {width}x{height}",
        ))

    fps_raw = info.get("r_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    if abs(fps - expected_fps) > 1:
        issues.append(QAIssue(category="fps", severity="warning",
                               message=f"Expected ~{expected_fps}fps, got {fps:.1f}fps"))

    duration = float(info.get("duration", 0) or 0)
    if duration < min_duration_seconds:
        issues.append(QAIssue(category="duration", severity="critical",
                               message=f"Video too short: {duration:.2f}s < {min_duration_seconds}s"))

    return issues


def detect_black_frames(path: str, black_min_duration: float = 0.5) -> list[QAIssue]:
    issues: list[QAIssue] = []
    cmd = [
        "ffmpeg", "-i", path, "-vf",
        f"blackdetect=d={black_min_duration}:pic_th=0.98", "-an", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if "black_start" in proc.stderr:
        issues.append(QAIssue(category="black_frames", severity="warning",
                               message="Detected possible black-frame segment(s)."))
    return issues
