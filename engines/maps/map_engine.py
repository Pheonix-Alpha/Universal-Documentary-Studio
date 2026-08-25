"""Map engine.

Renders a stylized (non-satellite, offline-safe) map showing labeled
points and optional routes between them. Uses plain normalized
coordinates (0-1) supplied by the caller rather than depending on any
live geocoding/tile service, so it works fully offline in MOCK_MODE and
CI.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass
class MapPoint:
    label: str
    x: float  # normalized 0..1
    y: float  # normalized 0..1


def render_map(points: list[MapPoint], title: str, output_path: str,
                width: int = 1920, height: int = 1080, show_route: bool = False) -> str:
    img = Image.new("RGB", (width, height), (14, 26, 30))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((60, 50), title, fill=(230, 230, 230), font=font)

    # Simple "landmass" backdrop shape for visual context (stylized, not geo-accurate).
    draw.rectangle([width * 0.1, height * 0.15, width * 0.9, height * 0.9], outline=(60, 90, 90), width=2)

    pixel_points = [(p.label, int(width * p.x), int(height * p.y)) for p in points]

    if show_route and len(pixel_points) > 1:
        for i in range(len(pixel_points) - 1):
            _, x0, y0 = pixel_points[i]
            _, x1, y1 = pixel_points[i + 1]
            draw.line([(x0, y0), (x1, y1)], fill=(232, 180, 92), width=3)

    for label, x, y in pixel_points:
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=(90, 160, 232))
        draw.text((x + 12, y - 6), label, fill=(220, 220, 220), font=font)

    img.save(output_path)
    return output_path
