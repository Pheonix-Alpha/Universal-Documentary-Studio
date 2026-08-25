"""Real diffusers-backed VideoGenerator (Stable Video Diffusion family).

Like DiffusersImageGenerator, torch/diffusers are only imported when this
class is instantiated; a missing/broken install surfaces as an
ImportError that ModelLifecycleManager catches and treats as "try the
next compatible model" (ultimately falling back to MockVideoGenerator so
the pipeline always completes -- spec sections 6/11/25).
"""
from __future__ import annotations

from pathlib import Path

from adapters.image_generation.base import ImageGenerationRequest
from adapters.image_generation.mock_generator import MockImageGenerator
from adapters.video_generation.base import VideoGenerationRequest, VideoGenerationResult, VideoGenerator
from core.logging import get_logger

logger = get_logger(__name__)


class DiffusersVideoGenerator(VideoGenerator):
    """Loads a HuggingFace Stable Video Diffusion (img2vid) pipeline on demand.

    SVD requires a source still frame. If the caller doesn't supply one via
    `request.source_image_path`, a placeholder is synthesized first via
    MockImageGenerator -- exactly what MockVideoGenerator already does --
    so behavior is consistent whether the real or mock video engine runs.
    """

    provider = "diffusers"

    def __init__(self, model_name: str, hf_repo_id: str | None, cache_dir: str, use_cuda: bool = True):
        if not hf_repo_id:
            raise ValueError(f"No hf_repo_id configured for model '{model_name}'.")

        self.model_name = model_name
        self.hf_repo_id = hf_repo_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        import torch
        from diffusers import StableVideoDiffusionPipeline

        self._torch = torch
        self.use_cuda = bool(use_cuda and torch.cuda.is_available())
        dtype = torch.float16 if self.use_cuda else torch.float32

        logger.info(
            "Loading diffusers video pipeline model=%s repo=%s cache=%s cuda=%s",
            model_name, hf_repo_id, self.cache_dir, self.use_cuda,
        )
        # Same rationale as DiffusersImageGenerator: request only the
        # fp16-variant safetensors weights when running fp16, instead of
        # downloading the full fp32 checkpoint (~2x the size) and casting
        # it after the fact.
        load_kwargs = dict(torch_dtype=dtype, cache_dir=str(self.cache_dir), use_safetensors=True)
        try:
            if self.use_cuda:
                self._pipe = StableVideoDiffusionPipeline.from_pretrained(hf_repo_id, variant="fp16", **load_kwargs)
            else:
                self._pipe = StableVideoDiffusionPipeline.from_pretrained(hf_repo_id, **load_kwargs)
        except Exception as exc:  # noqa: BLE001 - e.g. repo has no fp16 variant published
            logger.warning("fp16 variant unavailable for %s (%s); falling back to default weights", hf_repo_id, exc)
            self._pipe = StableVideoDiffusionPipeline.from_pretrained(hf_repo_id, **load_kwargs)
        if self.use_cuda:
            self._pipe = self._pipe.to("cuda")

    def generate(self, request: VideoGenerationRequest, output_path: str) -> VideoGenerationResult:
        from diffusers.utils import export_to_video, load_image

        source_image = request.source_image_path
        if source_image is None:
            source_image = output_path.rsplit(".", 1)[0] + "_source.png"
            MockImageGenerator().generate(
                ImageGenerationRequest(prompt=request.prompt, width=request.width, height=request.height),
                output_path=source_image,
            )

        image = load_image(source_image).resize((request.width, request.height))
        frames = self._pipe(image, decode_chunk_size=8).frames[0]

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(out_path), fps=request.fps)

        return VideoGenerationResult(
            file_path=str(out_path),
            model_name=self.model_name,
            provider=self.provider,
            metadata={"prompt": request.prompt, "source_image": source_image, "hf_repo_id": self.hf_repo_id},
        )

    def unload(self) -> None:
        if getattr(self, "_pipe", None) is not None:
            del self._pipe
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort cleanup
            logger.exception("Error while releasing CUDA memory for %s", self.model_name)
