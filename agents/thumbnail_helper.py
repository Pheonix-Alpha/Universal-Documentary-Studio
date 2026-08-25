"""ThumbnailAgent: generates 3-5 thumbnail concepts (spec section 41).

Thumbnails are built from actual generated scene imagery plus a bold
readable headline drawn directly from the video's real title/topic --
never a fabricated or exaggerated claim.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from core.logging import get_logger

logger = get_logger(__name__)


def _draw_headline(img: Image.Image, headline: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    w, h = img.size
    bar_h = int(h * 0.22)
    draw.rectangle([0, h - bar_h, w, h], fill=(0, 0, 0))
    draw.text((20, h - bar_h + 14), headline[:60], fill=(255, 255, 255), font=font)
    return img


class ThumbnailAgent:
    def run(self, source_image_paths: list[str], title_options: list[str], output_dir: str,
             width: int = 1280, height: int = 720, count: int = 4) -> list[str]:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        count = max(3, min(5, count))
        outputs = []

        if not source_image_paths:
            logger.warning("No source images available for thumbnails; skipping visual composition.")
            return outputs

        for i in range(count):
            src = source_image_paths[i % len(source_image_paths)]
            headline = title_options[i % len(title_options)] if title_options else ""
            try:
                img = Image.open(src).convert("RGB").resize((width, height))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not open source image %s for thumbnail: %s", src, exc)
                continue
            img = _draw_headline(img, headline)
            out_path = str(Path(output_dir) / f"thumbnail_{i+1}.png")
            img.save(out_path)
            outputs.append(out_path)

        return outputs
