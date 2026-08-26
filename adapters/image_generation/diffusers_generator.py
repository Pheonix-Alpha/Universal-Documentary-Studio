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

# SDXL is trained/benchmarked around ~1024x1024 (1,048,576 px). Generating
# directly at much larger targets (e.g. this app's 1920x1080 long-form
# canvas, ~2,073,600 px) roughly doubles attention/activation memory versus
# native resolution and is what actually exhausts VRAM on cards like a T4 --
# independent of how much VRAM the model's registry entry declares, since
# that figure is a per-model constant that has no idea what resolution a
# caller will ask for. Every downstream consumer of this image
# (engines/animation/ken_burns.py, engines/rendering/ffmpeg_renderer.py)
# already scales its input to the final target resolution via its own
# ffmpeg `scale=` filter, so there is no need to generate at the caller's
# exact requested size -- we generate at a safe, model-native size and let
# those later ffmpeg stages do the upscale.
_MAX_GENERATION_DIM = 1024
_DIM_MULTIPLE = 8


def _clamp_generation_size(width: int, height: int, max_dim: int = _MAX_GENERATION_DIM) -> tuple[int, int]:
    """Scale (width, height) down to fit within max_dim on the long edge,
    preserving aspect ratio, rounded to a multiple of 8 (required by SDXL's
    U-Net). Never scales up -- if the request is already small, it's left
    alone."""
    longest = max(width, height)
    if longest <= max_dim:
        scale = 1.0
    else:
        scale = max_dim / longest

    def _round(v: float) -> int:
        v = int(v * scale)
        return max(_DIM_MULTIPLE, (v // _DIM_MULTIPLE) * _DIM_MULTIPLE)

    return _round(width), _round(height)


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
            # Extra VRAM safety margin on top of the resolution clamp below
            # -- cheap (small/no measurable quality cost, modest speed
            # cost) and makes this robust even on tighter cards than a T4.
            try:
                self._pipe.enable_attention_slicing()
            except Exception:  # noqa: BLE001 - not all pipelines support this
                logger.debug("enable_attention_slicing unavailable for %s", self.model_name)
            try:
                self._pipe.enable_vae_slicing()
            except Exception:  # noqa: BLE001
                logger.debug("enable_vae_slicing unavailable for %s", self.model_name)

    def generate(self, request: ImageGenerationRequest, output_path: str) -> ImageGenerationResult:
        generator = None
        if request.seed is not None:
            device = "cuda" if self.use_cuda else "cpu"
            generator = self._torch.Generator(device=device).manual_seed(request.seed)

        gen_width, gen_height = _clamp_generation_size(request.width, request.height)
        if (gen_width, gen_height) != (request.width, request.height):
            logger.info(
                "Scene requested %dx%d; generating at %dx%d (model-native cap) to stay within VRAM -- "
                "downstream ffmpeg stages upscale to the final resolution.",
                request.width, request.height, gen_width, gen_height,
            )

        result = self._pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            width=gen_width,
            height=gen_height,
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
            metadata={
                "prompt": request.prompt, "hf_repo_id": self.hf_repo_id,
                "requested_width": request.width, "requested_height": request.height,
                "generated_width": gen_width, "generated_height": gen_height,
            },
        )

    def unload(self) -> None:
        if getattr(self, "_pipe", None) is not None:
            del self._pipe
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.exception("Error while releasing CUDA memory for %s", self.model_name)
