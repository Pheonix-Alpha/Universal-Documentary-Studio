"""Mock ImageGenerator: produces real PNG files (via Pillow) so the rest
of the pipeline (animation, rendering) has actual media to work with,
without needing a GPU or diffusion model installed. Deterministic per
prompt so repeated runs / caching behave sensibly.
"""
from __future__ import annotations

import hashlib
import textwrap

from PIL import Image, ImageDraw, ImageFont

from adapters.image_generation.base import ImageGenerationRequest, ImageGenerationResult, ImageGenerator

_PALETTES = [
    ((20, 24, 35), (210, 200, 170)),
    ((30, 20, 20), (220, 180, 140)),
    ((15, 30, 30), (180, 220, 210)),
    ((25, 25, 45), (200, 200, 230)),
    ((35, 25, 15), (230, 210, 170)),
]


def _seed_from_prompt(prompt: str) -> int:
    return int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)


class MockImageGenerator(ImageGenerator):
    model_name = "mock-image-v1"
    provider = "mock"

    def generate(self, request: ImageGenerationRequest, output_path: str) -> ImageGenerationResult:
        seed = request.seed if request.seed is not None else _seed_from_prompt(request.prompt)
        bg, fg = _PALETTES[seed % len(_PALETTES)]

        img = Image.new("RGB", (request.width, request.height), bg)
        draw = ImageDraw.Draw(img)

        # Simple deterministic "compositional" shapes so frames differ visually.
        rng = seed
        for i in range(4):
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            x0 = (rng % request.width)
            rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
            y0 = (rng % request.height)
            w = request.width // 4
            h = request.height // 4
            draw.ellipse(
                [x0 - w // 2, y0 - h // 2, x0 + w // 2, y0 + h // 2],
                outline=fg, width=3,
            )

        caption = textwrap.fill(request.prompt, width=40)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.rectangle([0, request.height - 90, request.width, request.height], fill=(0, 0, 0))
        draw.text((20, request.height - 80), f"[MOCK IMAGE] {caption[:120]}", fill=fg, font=font)

        img.save(output_path)

        return ImageGenerationResult(
            file_path=output_path,
            model_name=self.model_name,
            provider=self.provider,
            seed=seed,
            metadata={"prompt": request.prompt, "width": request.width, "height": request.height},
        )
