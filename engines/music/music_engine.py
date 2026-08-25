"""Music engine.

Selects a licensed/free music cue by mood from `assets/music/<mood>/`.
If no library track exists for a mood (e.g. in a fresh checkout or in
MOCK_MODE), synthesizes a short, quiet placeholder ambient tone via
FFmpeg so the mixing/rendering pipeline always has a real file to work
with.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

MOODS = [
    "cinematic", "dramatic", "mystery", "technology", "historical",
    "emotional", "corporate", "ambient", "neutral",
]

_MOOD_FREQ_HZ = {
    "cinematic": 110, "dramatic": 98, "mystery": 130, "technology": 220,
    "historical": 90, "emotional": 150, "corporate": 180, "ambient": 100,
    "neutral": 120,
}


def find_or_synthesize_cue(mood: str, duration_seconds: float, library_dir: str, output_dir: str) -> str:
    mood = mood if mood in MOODS else "neutral"
    mood_dir = Path(library_dir) / mood
    if mood_dir.exists():
        candidates = sorted(mood_dir.glob("*.mp3")) + sorted(mood_dir.glob("*.wav"))
        if candidates:
            return str(candidates[0])

    freq = _MOOD_FREQ_HZ.get(mood, 120)
    out_path = str(Path(output_dir) / f"music_{mood}.wav")
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency={freq}:duration={max(1.0, duration_seconds)}",
        "-af", "volume=0.15", out_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path
