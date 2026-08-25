"""Graphics engine: title cards / lower-thirds / text-animation source frames.

Produces a still frame with clean typography that the animation engine
then turns into a subtle text-animation shot (fade/slide handled at
render time via FFmpeg).
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def render_text_card(headline: str, subtext: str, output_path: str,
                      width: int = 1920, height: int = 1080) -> str:
    img = Image.new("RGB", (width, height), (10, 10, 14))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((width * 0.1, height * 0.42), headline[:80], fill=(240, 240, 240), font=font)
    if subtext:
        draw.text((width * 0.1, height * 0.52), subtext[:100], fill=(180, 180, 190), font=font)

    img.save(output_path)
    return output_path
