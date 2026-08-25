"""SFX engine.

Selects a licensed/free SFX clip by category from `assets/sfx/<category>/`.
Falls back to a short synthesized "whoosh"-style placeholder tone via
FFmpeg when no library asset is available.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

CATEGORIES = [
    "ambience", "city", "office", "machinery", "technology",
    "nature", "transition", "historical_atmosphere",
]


def find_or_synthesize_sfx(category: str, duration_seconds: float, library_dir: str, output_dir: str) -> str:
    category = category if category in CATEGORIES else "transition"
    cat_dir = Path(library_dir) / category
    if cat_dir.exists():
        candidates = sorted(cat_dir.glob("*.mp3")) + sorted(cat_dir.glob("*.wav"))
        if candidates:
            return str(candidates[0])

    out_path = str(Path(output_dir) / f"sfx_{category}.wav")
    duration = max(0.3, min(duration_seconds, 2.0))
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-af", "volume=0.08,afade=t=out:st=0:d=" + str(duration),
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path
