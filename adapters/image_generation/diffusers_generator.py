"""Real diffusers-backed ImageGenerator (SDXL / SD-Turbo family).

torch/diffusers are only imported when this class is actually
instantiated -- i.e. only when ModelLifecycleManager has already decided
(via ModelRegistry + ResourceManager) that a real image model is both
selected and resource-compatible. mock_mode / CI / no-GPU environments
never need these heavy packages installed; if they're missing, the
ImportError raised here is caught by ModelLifecycleManager and treated
as "this candidate failed to load, try the next one" (spec section 11).
"""
from __future__ import annotations

from pathlib import Path

from adapters.image_generation.base import ImageGenerationRequest, ImageGenerationResult, ImageGenerator
from core.logging import get_logger

logger = get_logger(__name__)


class DiffusersImageGenerator(ImageGenerator):
    """Loads a HuggingFace `diffusers` text-to-image pipeline on demand."""

    provider = "diffusers"

    def __init__(self, model_name: str, hf_repo_id: str | None, cache_dir: str, use_cuda: bool = True):
        if not hf_repo_id:
            raise ValueError(f"No hf_repo_id configured for model '{model_name}'.")

        self.model_name = model_name
        self.hf_repo_id = hf_repo_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        import torch  # noqa: F401  -- absence here is the fallback signal
        from diffusers import AutoPipelineForText2Image

        self._torch = torch
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        dtype = torch.float16 if self.use_cuda else torch.float32

        logger.info(
            "Loading diffusers image pipeline model=%s repo=%s cache=%s cuda=%s",
            model_name, hf_repo_id, self.cache_dir, self.use_cuda,
        )
        # Only ever request the fp16-variant safetensors weights when
        # running fp16 (the common GPU case): without `variant="fp16"`,
        # diffusers downloads the full fp32 checkpoint and casts it after
        # the fact, which is roughly 2x the download/disk footprint for no
        # benefit once cast to float16 anyway. Falls back to the default
        # (fp32) weights if a repo doesn't publish an fp16 variant.
        load_kwargs = dict(torch_dtype=dtype, cache_dir=str(self.cache_dir), use_safetensors=True)
        try:
            if self.use_cuda:
                self._pipe = AutoPipelineForText2Image.from_pretrained(hf_repo_id, variant="fp16", **load_kwargs)
            else:
                self._pipe = AutoPipelineForText2Image.from_pretrained(hf_repo_id, **load_kwargs)
        except Exception as exc:  # noqa: BLE001 - e.g. repo has no fp16 variant published
            logger.warning("fp16 variant unavailable for %s (%s); falling back to default weights", hf_repo_id, exc)
            self._pipe = AutoPipelineForText2Image.from_pretrained(hf_repo_id, **load_kwargs)
        if self.use_cuda:
            self._pipe = self._pipe.to("cuda")

    def generate(self, request: ImageGenerationRequest, output_path: str) -> ImageGenerationResult:
        generator = None
        if request.seed is not None:
            device = "cuda" if self.use_cuda else "cpu"
            generator = self._torch.Generator(device=device).manual_seed(request.seed)

        result = self._pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=request.width,
            height=request.height,
            generator=generator,
        )
        image = result.images[0]

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)

        return ImageGenerationResult(
            file_path=str(out_path),
            model_name=self.model_name,
            provider=self.provider,
            seed=request.seed,
            metadata={"prompt": request.prompt, "hf_repo_id": self.hf_repo_id, "width": request.width, "height": request.height},
        )

    def unload(self) -> None:
        if getattr(self, "_pipe", None) is not None:
            del self._pipe
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.exception("Error while releasing CUDA memory for %s", self.model_name)
