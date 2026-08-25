"""Timeline engine.

Renders a horizontal timeline image from verified (date, label) events.
The image is then handed to the animation engine (Ken Burns / pan) so it
becomes a moving shot rather than a static slide.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont


@dataclass
class TimelineEvent:
    date: str
    label: str


def render_timeline(events: list[TimelineEvent], title: str, output_path: str,
                     width: int = 1920, height: int = 1080) -> str:
    img = Image.new("RGB", (width, height), (18, 20, 28))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((60, 50), title, fill=(230, 230, 230), font=font)

    line_y = height // 2
    draw.line([(80, line_y), (width - 80, line_y)], fill=(120, 120, 140), width=4)

    n = max(1, len(events))
    usable_width = width - 160
    for i, event in enumerate(events):
        x = 80 + (usable_width * i / max(1, n - 1)) if n > 1 else width // 2
        draw.ellipse([x - 10, line_y - 10, x + 10, line_y + 10], fill=(232, 180, 92))
        label_y = line_y - 60 if i % 2 == 0 else line_y + 30
        draw.text((x - 40, label_y), event.date, fill=(232, 180, 92), font=font)
        draw.text((x - 60, label_y + 16), event.label[:30], fill=(220, 220, 220), font=font)

    img.save(output_path)
    return output_path
