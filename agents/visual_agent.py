"""VisualAgent: scenes.json -> assets.json (spec section 21/24/25/26).

For each scene, attempts the requested visual technique and always has a
safe fallback chain:

    licensed/real media (if available with known license)
        -> AI generated image
        -> image animation (Ken Burns) over a generated/placeholder image
        -> chart / timeline / map / diagram (for those visual types directly)

AI video generation is attempted only if a compatible model/worker is
available (spec 25); otherwise the scene falls back to image animation
so the documentary can always be completed.
"""
from __future__ import annotations

from pathlib import Path

from adapters.image_generation.base import ImageGenerationRequest
from adapters.image_generation.diffusers_generator import DiffusersImageGenerator
from adapters.image_generation.mock_generator import MockImageGenerator
from adapters.media_sources.base import MediaProvider
from adapters.video_generation.base import VideoGenerationRequest
from adapters.video_generation.diffusers_generator import DiffusersVideoGenerator
from adapters.video_generation.mock_generator import MockVideoGenerator
from core.logging import get_logger
from core.model_lifecycle import LoadedModel, ModelLifecycleManager
from core.models import (
    Asset, AssetManifest, AssetOrigin, LicenseRecord, Scene, ScenePlan, VisualType,
)
from core.resource_manager import ModelRequirement, ResourceManager
from core.scheduler import Job, Scheduler
from engines.animation.ken_burns import animate_image
from engines.charts.chart_engine import render_bar_chart
from engines.diagrams.diagram_engine import render_flow_diagram
from engines.graphics.text_card import render_text_card
from engines.maps.map_engine import MapPoint, render_map
from engines.timelines.timeline_engine import TimelineEvent, render_timeline
from models.capabilities import ModelTask
from models.registry import ModelRegistry

logger = get_logger(__name__)


def _build_prompt(scene: Scene) -> str:
    return (
        f"{scene.visual_objective} documentary realism, {scene.lighting} lighting, "
        f"{scene.lens.value} lens, {scene.composition} composition"
    )


class VisualAgent:
    def __init__(
        self,
        model_registry: ModelRegistry,
        resource_manager: ResourceManager,
        scheduler: Scheduler,
        media_provider: MediaProvider,
        assets_dir: str,
        video_width: int = 1920,
        video_height: int = 1080,
        fps: int = 24,
        model_lifecycle: ModelLifecycleManager | None = None,
    ):
        self.model_registry = model_registry
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.media_provider = media_provider
        self.assets_dir = Path(assets_dir)
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps

        # Real model selection/download/load/unload always goes through the
        # lifecycle manager -- this is what previously got skipped: the
        # registry's pick never reached generation because these two
        # attributes were hard-coded to the mocks regardless of mock_mode.
        self.model_lifecycle = model_lifecycle or ModelLifecycleManager(
            model_registry=model_registry, resource_manager=resource_manager, mock_mode=True,
        )
        self._image_model: LoadedModel | None = None
        self._video_model: LoadedModel | None = None

    def run(self, scene_plan: ScenePlan) -> AssetManifest:
        assets: list[Asset] = []
        try:
            for scene in scene_plan.scenes:
                asset = self._generate_for_scene(scene)
                assets.append(asset)
        finally:
            # Never leave a heavy model resident in GPU/RAM/disk once this
            # batch is done, even if a scene raised partway through.
            self.release_models()
        return AssetManifest(assets=assets)

    def regenerate_scene(self, scene: Scene) -> Asset:
        """Regenerate a single scene (used by the human-review UI), then
        immediately release any model it acquired -- a one-off regen should
        never leave a heavy model resident afterward."""
        try:
            return self._generate_for_scene(scene)
        finally:
            self.release_models()

    def release_models(self) -> None:
        """Explicitly unload + clean up any models acquired by this agent."""
        self._release_image_model()
        self._release_video_model()

    # ------------------------------------------------------------------
    # Lazy model acquisition (cached for the duration of one run()/regen)
    # ------------------------------------------------------------------

    def _release_image_model(self) -> None:
        if self._image_model is not None:
            self._image_model.release()
            self._image_model = None

    def _release_video_model(self) -> None:
        if self._video_model is not None:
            self._video_model.release()
            self._video_model = None

    def _get_image_generator(self):
        # Image and video models are each large enough (e.g. SDXL ~7GB +
        # SVD ~9GB fp16) that holding both resident on GPU at once can
        # exceed VRAM on cards like a T4 (14.56GB total). No single scene
        # ever needs both simultaneously -- video is attempted, and only
        # on failure/absence does a scene fall back to image generation --
        # so release the other model type before loading this one instead
        # of letting both accumulate across the run.
        self._release_video_model()
        if self._image_model is None:
            self._image_model = self.model_lifecycle.acquire(
                task=ModelTask.IMAGE_GENERATION,
                mock_factory=lambda: MockImageGenerator(),
                real_factory=lambda capability, workdir: DiffusersImageGenerator(
                    model_name=capability.model_name,
                    hf_repo_id=capability.hf_repo_id,
                    cache_dir=str(workdir),
                    use_cuda=self.resource_manager.detect().cuda_available,
                ),
            )
        return self._image_model.generator

    def _get_video_generator(self):
        self._release_image_model()
        if self._video_model is None:
            self._video_model = self.model_lifecycle.acquire(
                task=ModelTask.VIDEO_GENERATION,
                mock_factory=lambda: MockVideoGenerator(),
                real_factory=lambda capability, workdir: DiffusersVideoGenerator(
                    model_name=capability.model_name,
                    hf_repo_id=capability.hf_repo_id,
                    cache_dir=str(workdir),
                    use_cuda=self.resource_manager.detect().cuda_available,
                ),
            )
        return self._video_model.generator

    # ------------------------------------------------------------------

    def _generate_for_scene(self, scene: Scene) -> Asset:
        clip_path = str(self.assets_dir / f"scene_{scene.index:03d}.mp4")

        if scene.visual_type == VisualType.CHART:
            return self._make_chart_scene(scene, clip_path)
        if scene.visual_type == VisualType.TIMELINE:
            return self._make_timeline_scene(scene, clip_path)
        if scene.visual_type == VisualType.MAP:
            return self._make_map_scene(scene, clip_path)
        if scene.visual_type == VisualType.DIAGRAM:
            return self._make_diagram_scene(scene, clip_path)
        if scene.visual_type == VisualType.TEXT_ANIMATION:
            return self._make_text_scene(scene, clip_path)

        # real/licensed media path first
        licensed_asset = self._try_licensed_media(scene, clip_path)
        if licensed_asset is not None:
            return licensed_asset

        if scene.visual_type == VisualType.GENERATED_VIDEO:
            video_asset = self._try_generated_video(scene, clip_path)
            if video_asset is not None:
                return video_asset
            logger.info("Scene %d: falling back from generated_video to image animation.", scene.index)

        # default / fallback: generated image -> Ken Burns animation
        return self._make_image_animation_scene(scene, clip_path)

    # ------------------------------------------------------------------
    # Visual technique implementations
    # ------------------------------------------------------------------

    def _try_licensed_media(self, scene: Scene, clip_path: str) -> Asset | None:
        if scene.visual_type not in (VisualType.REAL_MEDIA, VisualType.PUBLIC_DOMAIN_MEDIA, VisualType.LICENSED_MEDIA):
            return None
        results = self.media_provider.search(scene.visual_objective, media_type="image")
        if not results:
            logger.info("Scene %d: no licensed media found, falling back.", scene.index)
            return None
        best = results[0]
        # A real implementation would download `best.url` here; omitted in
        # mock mode since MockMediaProvider never returns results.
        return Asset(
            scene_id=scene.scene_id,
            visual_type=scene.visual_type,
            origin=AssetOrigin.EXTERNAL_MEDIA,
            file_path=None,
            provider="media_provider",
            license=best.license,
            metadata={"url": best.url},
        )

    def _try_generated_video(self, scene: Scene, clip_path: str) -> Asset | None:
        requirement = ModelRequirement(minimum_vram_gb=12.0, recommended_vram_gb=16.0, requires_cuda=True)

        job = Job(task_type="video_generation", project_id="", scene_id=scene.scene_id,
                   gpu_required=True, requirement=requirement)

        def remote_executor(_job: Job):
            request = VideoGenerationRequest(
                prompt=_build_prompt(scene), duration_seconds=scene.duration_seconds,
                width=self.video_width, height=self.video_height, fps=self.fps,
            )
            return self._get_video_generator().generate(request, clip_path)

        result = self.scheduler.run_job(job, remote_executor=remote_executor, fallback_executor=None)
        if result.result is None:
            return None

        gen_result = result.result
        return Asset(
            scene_id=scene.scene_id,
            visual_type=VisualType.GENERATED_VIDEO,
            origin=AssetOrigin.AI_GENERATED,
            file_path=gen_result.file_path,
            provider=gen_result.provider,
            license=LicenseRecord(asset_id=scene.scene_id, license="internal-generated", commercial_use=True),
            metadata={**gen_result.metadata, "model": gen_result.model_name},
        )

    def _make_image_animation_scene(self, scene: Scene, clip_path: str) -> Asset:
        image_path = str(self.assets_dir / f"scene_{scene.index:03d}_source.png")
        request = ImageGenerationRequest(prompt=_build_prompt(scene), width=self.video_width, height=self.video_height)
        gen_result = self._get_image_generator().generate(request, image_path)

        animate_image(
            image_path=gen_result.file_path, output_path=clip_path,
            duration_seconds=scene.duration_seconds, movement=scene.camera_movement,
            width=self.video_width, height=self.video_height, fps=self.fps,
        )
        return Asset(
            scene_id=scene.scene_id,
            visual_type=VisualType.IMAGE_ANIMATION,
            origin=AssetOrigin.AI_GENERATED,
            file_path=clip_path,
            provider=gen_result.provider,
            license=LicenseRecord(asset_id=scene.scene_id, license="internal-generated", commercial_use=True),
            metadata={"prompt": request.prompt, "source_image": gen_result.file_path, "model": gen_result.model_name},
        )

    def _make_chart_scene(self, scene: Scene, clip_path: str) -> Asset:
        image_path = str(self.assets_dir / f"scene_{scene.index:03d}_chart.png")
        labels = ["Y1", "Y2", "Y3", "Y4"]
        values = [10, 25, 18, 32]
        render_bar_chart(labels, values, scene.visual_objective[:40], image_path,
                          width_px=self.video_width, height_px=self.video_height)
        animate_image(image_path, clip_path, scene.duration_seconds, movement=scene.camera_movement,
                       width=self.video_width, height=self.video_height, fps=self.fps)
        return Asset(scene_id=scene.scene_id, visual_type=VisualType.CHART, origin=AssetOrigin.HUMAN_CREATED,
                      file_path=clip_path, provider="chart_engine",
                      license=LicenseRecord(asset_id=scene.scene_id, license="internal-generated", commercial_use=True),
                      metadata={"chart_type": "bar"})

    def _make_timeline_scene(self, scene: Scene, clip_path: str) -> Asset:
        image_path = str(self.assets_dir / f"scene_{scene.index:03d}_timeline.png")
        events = [TimelineEvent(date=f"Y{i}", label=scene.visual_objective[:20]) for i in range(1, 4)]
        render_timeline(events, scene.visual_objective[:40], image_path, width=self.video_width, height=self.video_height)
        animate_image(image_path, clip_path, scene.duration_seconds, movement=scene.camera_movement,
                       width=self.video_width, height=self.video_height, fps=self.fps)
        return Asset(scene_id=scene.scene_id, visual_type=VisualType.TIMELINE, origin=AssetOrigin.HUMAN_CREATED,
                      file_path=clip_path, provider="timeline_engine",
                      license=LicenseRecord(asset_id=scene.scene_id, license="internal-generated", commercial_use=True),
                      metadata={})

    def _make_map_scene(self, scene: Scene, clip_path: str) -> Asset:
        image_path = str(self.assets_dir / f"scene_{scene.index:03d}_map.png")
        points = [MapPoint(label="A", x=0.3, y=0.4), MapPoint(label="B", x=0.6, y=0.6)]
        render_map(points, scene.visual_objective[:40], image_path, width=self.video_width, height=self.video_height,
                   show_route=True)
        animate_image(image_path, clip_path, scene.duration_seconds, movement=scene.camera_movement,
                       width=self.video_width, height=self.video_height, fps=self.fps)
        return Asset(scene_id=scene.scene_id, visual_type=VisualType.MAP, origin=AssetOrigin.HUMAN_CREATED,
                      file_path=clip_path, provider="map_engine",
                      license=LicenseRecord(asset_id=scene.scene_id, license="internal-generated", commercial_use=True),
                      metadata={})

    def _make_diagram_scene(self, scene: Scene, clip_path: str) -> Asset:
        image_path = str(self.assets_dir / f"scene_{scene.index:03d}_diagram.png")
        steps = ["Input", "Process", "Output"]
        render_flow_diagram(steps, scene.visual_objective[:40], image_path, width=self.video_width, height=self.video_height)
        animate_image(image_path, clip_path, scene.duration_seconds, movement=scene.camera_movement,
                       width=self.video_width, height=self.video_height, fps=self.fps)
        return Asset(scene_id=scene.scene_id, visual_type=VisualType.DIAGRAM, origin=AssetOrigin.HUMAN_CREATED,
                      file_path=clip_path, provider="diagram_engine",
                      license=LicenseRecord(asset_id=scene.scene_id, license="internal-generated", commercial_use=True),
                      metadata={})

    def _make_text_scene(self, scene: Scene, clip_path: str) -> Asset:
        image_path = str(self.assets_dir / f"scene_{scene.index:03d}_text.png")
        render_text_card(scene.narration[:80], "", image_path, width=self.video_width, height=self.video_height)
        animate_image(image_path, clip_path, scene.duration_seconds, movement=scene.camera_movement,
                       width=self.video_width, height=self.video_height, fps=self.fps)
        return Asset(scene_id=scene.scene_id, visual_type=VisualType.TEXT_ANIMATION, origin=AssetOrigin.HUMAN_CREATED,
                      file_path=clip_path, provider="graphics_engine",
                      license=LicenseRecord(asset_id=scene.scene_id, license="internal-generated", commercial_use=True),
                      metadata={})
