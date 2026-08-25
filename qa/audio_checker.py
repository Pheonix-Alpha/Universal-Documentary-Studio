"""Audio QA checks: silence detection, clipping, and duration sanity."""
from __future__ import annotations

import subprocess

from core.models import QAIssue


def check_audio(path: str, expected_min_duration: float = 0.5) -> list[QAIssue]:
    issues: list[QAIssue] = []

    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        duration = float(proc.stdout.strip())
    except (ValueError, AttributeError):
        duration = 0.0

    if duration < expected_min_duration:
        issues.append(QAIssue(category="audio_duration", severity="critical",
                               message=f"Audio too short: {duration:.2f}s"))

    # Silence detection (stream processed, not loaded fully into memory).
    silence_cmd = ["ffmpeg", "-i", path, "-af", "silencedetect=noise=-40dB:d=1", "-f", "null", "-"]
    proc = subprocess.run(silence_cmd, capture_output=True, text=True)
    silence_count = proc.stderr.count("silence_start")
    if silence_count > 0 and duration > 0 and (silence_count * 1.0) / max(duration, 1) > 0.3:
        issues.append(QAIssue(category="silence", severity="warning",
                               message=f"Found {silence_count} extended silence region(s)."))

    # Clipping heuristic via astats max_level.
    stats_cmd = ["ffmpeg", "-i", path, "-af", "astats=metadata=1:reset=1", "-f", "null", "-"]
    proc = subprocess.run(stats_cmd, capture_output=True, text=True)
    if "Overflow" in proc.stderr or "-0.0 dB" in proc.stderr:
        issues.append(QAIssue(category="clipping", severity="warning",
                               message="Possible audio clipping detected."))

    return issues
