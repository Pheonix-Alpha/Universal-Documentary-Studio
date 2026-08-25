"""Ken Burns / cinematic camera-move engine.

This is the critical low-GPU fallback (spec section 23): turns a single
still image into a motivated, subtle cinematic shot using FFmpeg's
zoompan filter, driven by the scene's camera_movement/lens choice rather
than random motion. Expressions are kept intentionally simple/robust
(FFmpeg's zoompan expression parser is picky) while still producing
visually distinct motion per movement type.
"""
from __future__ import annotations

import subprocess

from core.exceptions import RenderError
from core.models import CameraMovement

_MOVEMENT_EXPR = {
    CameraMovement.STATIC:        ("1.02", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    CameraMovement.SLOW_PUSH_IN:  ("min(zoom+0.0008,1.18)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    CameraMovement.SLOW_PULL_OUT: ("if(eq(on,0),1.18,max(zoom-0.0008,1.0))", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    CameraMovement.ZOOM:          ("min(zoom+0.0018,1.3)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    CameraMovement.DOLLY:         ("min(zoom+0.0012,1.22)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    CameraMovement.PAN_LEFT:      ("1.12", "iw/2-(iw/zoom/2)-on*1.2", "ih/2-(ih/zoom/2)"),
    CameraMovement.PAN_RIGHT:     ("1.12", "iw/2-(iw/zoom/2)+on*1.2", "ih/2-(ih/zoom/2)"),
    CameraMovement.TILT:          ("1.12", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)+on*0.8"),
    CameraMovement.TRACKING:      ("1.1",  "iw/2-(iw/zoom/2)+on*0.6", "ih/2-(ih/zoom/2)"),
    CameraMovement.ORBIT:         ("1.12", "iw/2-(iw/zoom/2)+8*sin(on/20)", "ih/2-(ih/zoom/2)"),
    CameraMovement.PARALLAX:      ("min(zoom+0.0004,1.1)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"),
    CameraMovement.HANDHELD:      ("1.06", "iw/2-(iw/zoom/2)+3*sin(on/5)", "ih/2-(ih/zoom/2)+2*cos(on/7)"),
    CameraMovement.CRANE:         ("1.1",  "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)-on*0.7"),
}


def animate_image(
    image_path: str,
    output_path: str,
    duration_seconds: float,
    movement: CameraMovement = CameraMovement.STATIC,
    width: int = 1920,
    height: int = 1080,
    fps: int = 24,
) -> str:
    frames = max(1, int(duration_seconds * fps))
    zoom_expr, x_expr, y_expr = _MOVEMENT_EXPR.get(movement, _MOVEMENT_EXPR[CameraMovement.STATIC])

    vf = (
        f"scale={width * 2}:{height * 2},"
        f"zoompan=z='{zoom_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={width}x{height}:fps={fps}"
    )

    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", image_path,
        "-vf", vf, "-t", str(duration_seconds),
        "-pix_fmt", "yuv420p", output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RenderError(f"Ken Burns animation failed: {proc.stderr[-2000:]}")
    return output_path
