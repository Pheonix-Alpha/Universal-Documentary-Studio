"""Caption engine.

Generates SRT and WebVTT captions from a list of (text, start, end)
segments. Segments are produced upstream from scene narration + duration
(see agents/audio_agent.py), keeping this module a pure formatter.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaptionSegment:
    text: str
    start_seconds: float
    end_seconds: float


def _format_srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_srt(segments: list[CaptionSegment], output_path: str) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_timestamp(seg.start_seconds)} --> {_format_srt_timestamp(seg.end_seconds)}")
        lines.append(seg.text)
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def write_vtt(segments: list[CaptionSegment], output_path: str) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_format_vtt_timestamp(seg.start_seconds)} --> {_format_vtt_timestamp(seg.end_seconds)}")
        lines.append(seg.text)
        lines.append("")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def segments_from_scenes(scene_texts: list[tuple[str, float]]) -> list[CaptionSegment]:
    """Build caption segments from a list of (narration_text, duration_seconds)."""
    segments = []
    cursor = 0.0
    for text, duration in scene_texts:
        segments.append(CaptionSegment(text=text, start_seconds=cursor, end_seconds=cursor + duration))
        cursor += duration
    return segments
