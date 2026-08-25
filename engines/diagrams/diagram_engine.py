"""Diagram engine.

Renders simple, clean process/flow diagrams (box -> arrow -> box) from a
list of step labels. Sufficient for "how it works" / technical
explanation scenes without needing an external diagramming library.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def render_flow_diagram(steps: list[str], title: str, output_path: str,
                         width: int = 1920, height: int = 1080) -> str:
    img = Image.new("RGB", (width, height), (16, 18, 26))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((60, 50), title, fill=(230, 230, 230), font=font)

    n = max(1, len(steps))
    box_w = min(280, (width - 120) // n - 40)
    box_h = 100
    gap = (width - 120 - box_w * n) // max(1, n - 1) if n > 1 else 0
    y = height // 2 - box_h // 2

    x = 60
    centers = []
    for step in steps:
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=12,
                                outline=(232, 180, 92), width=3, fill=(28, 32, 44))
        draw.text((x + 16, y + box_h // 2 - 6), step[:26], fill=(230, 230, 230), font=font)
        centers.append((x + box_w, y + box_h // 2))
        x += box_w + gap

    for i in range(len(centers) - 1):
        x0, cy = centers[i]
        x1, _ = centers[i + 1]
        draw.line([(x0, cy), (x1 - gap, cy)], fill=(150, 150, 160), width=3)
        draw.polygon([(x1 - gap - 10, cy - 6), (x1 - gap - 10, cy + 6), (x1 - gap, cy)], fill=(150, 150, 160))

    img.save(output_path)
    return output_path
