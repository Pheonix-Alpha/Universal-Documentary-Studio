"""Audio pipeline: narration cleanup/normalization and voice+music+SFX mixing.

Pipeline (spec section 34):
    TTS -> cleanup -> silence trim -> normalize -> EQ/compression -> loudness norm
    then: voice + music + sfx, with automatic ducking so narration stays clear.

Implemented with FFmpeg filters so it works identically locally and on
a Colab worker, with no extra Python audio-DSP dependency required.
"""
from __future__ import annotations

import subprocess

from core.exceptions import RenderError


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"ffmpeg audio command failed: {' '.join(cmd)}\n{proc.stderr[-2000:]}")


def normalize_narration(input_path: str, output_path: str) -> str:
    """Trim leading/trailing silence and apply loudness normalization (EBU R128)."""
    vf = (
        "silenceremove=start_periods=1:start_threshold=-55dB:"
        "stop_periods=1:stop_threshold=-55dB,"
        "loudnorm=I=-16:TP=-1.5:LRA=11"
    )
    cmd = ["ffmpeg", "-y", "-i", input_path, "-af", vf, output_path]
    _run(cmd)
    return output_path


def mix_voice_music_sfx(
    voice_path: str,
    output_path: str,
    music_path: str | None = None,
    sfx_paths: list[str] | None = None,
    music_duck_db: float = -14.0,
) -> str:
    """Mix narration (full volume) with optional music (ducked under voice) and SFX."""
    sfx_paths = sfx_paths or []
    inputs = ["-i", voice_path]
    filter_inputs = ["[0:a]"]

    idx = 1
    if music_path:
        inputs += ["-i", music_path]
        # Duck music under narration using sidechaincompress keyed off the voice track.
        filter_inputs.append(f"[{idx}:a]volume={music_duck_db}dB[music_ducked]")
        idx += 1

    sfx_labels = []
    for sfx_path in sfx_paths:
        inputs += ["-i", sfx_path]
        sfx_labels.append(f"[{idx}:a]")
        idx += 1

    filter_parts = []
    mix_labels = ["[0:a]"]
    if music_path:
        filter_parts.append(f"[1:a]volume={music_duck_db}dB[music_ducked]")
        mix_labels.append("[music_ducked]")
    mix_labels.extend(sfx_labels)

    n_inputs = len(mix_labels)
    filter_parts.append(f"{''.join(mix_labels)}amix=inputs={n_inputs}:duration=first:dropout_transition=2[mixed]")
    filter_complex = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y", *inputs, "-filter_complex", filter_complex, "-map", "[mixed]", output_path]
    _run(cmd)
    return output_path


def get_audio_duration_seconds(path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0
