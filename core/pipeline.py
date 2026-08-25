"""Pipeline orchestrator.

Wires every agent together, driving the ProjectManager's state machine
and saving a checkpoint after each stage completes so a crashed/resumed
run continues from the last successful stage rather than restarting
(spec section 8/51).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adapters.media_sources.mock_provider import MockMediaProvider
from adapters.research.mock_provider import MockResearchProvider
from adapters.tts.mock_engine import MockVoiceEngine
from adapters.tts.piper_engine import PiperVoiceEngine
from agents.audio_agent import AudioAgent
from agents.fact_checker import FactChecker
from agents.metadata_agent import MetadataAgent
from agents.qa_agent import QAAgent
from agents.research_agent import ResearchAgent
from agents.scene_agent import SceneAgent
from agents.script_agent import ScriptAgent
from agents.shorts_agent import ShortsAgent
from agents.story_agent import StoryAgent
from agents.thumbnail_helper import ThumbnailAgent
from agents.visual_agent import VisualAgent
from core.exceptions import QAFailure
from core.logging import get_logger
from core.model_lifecycle import LifecycleEvent, ModelLifecycleManager, StatusCallback
from core.models import (
    AssetManifest, FactCheckReport, MusicManifest, ProjectState, Research, ScenePlan,
    Script, Story, VoiceManifest,
)
from core.project_manager import ProjectManager
from core.resource_manager import ResourceManager
from core.scheduler import Scheduler
from engines.captions.caption_engine import segments_from_scenes
from engines.rendering.ffmpeg_renderer import concat_clips, mux_audio
from models.registry import ModelRegistry

logger = get_logger(__name__)


def console_status_printer(event: LifecycleEvent) -> None:
    """Default status_callback: a first pass at the "show everything, not
    'something is happening...'" UI requirement (spec section 4), for
    contexts (CLI, tests) that don't have a Gradio panel to render into."""
    bits = [f"[{event.task}] {event.status.value}"]
    if event.model_name:
        bits.append(f"model={event.model_name}")
    if event.message:
        bits.append(event.message)
    print("    · " + " — ".join(bits))


@dataclass
class PipelineResult:
    research: Research
    fact_check: FactCheckReport
    story: Story
    script: Script
    scene_plan: ScenePlan
    asset_manifest: AssetManifest
    voice_manifest: VoiceManifest
    music_manifest: MusicManifest
    render_path: str | None
    qa_report: object
    shorts: list
    short_render_paths: list
    thumbnail_paths: list
    metadata: object


class DocumentaryPipeline:
    """Runs (or resumes) the full topic -> long-form + shorts pipeline for
    one project. Every stage checks `ProjectManager` for an existing
    checkpoint first, so re-running `run_full()` on a project that already
    has partial progress resumes rather than redoes work.
    """

    def __init__(
        self, pm: ProjectManager, video_width: int = 1920, video_height: int = 1080, fps: int = 24,
        short_width: int = 1080, short_height: int = 1920, short_fps: int = 30,
        status_callback: StatusCallback | None = None,
        runtime_dir: str | None = None,
    ):
        self.pm = pm
        self.video_width = video_width
        self.video_height = video_height
        self.fps = fps

        self.resource_manager = ResourceManager(vram_safety_margin_gb=1.5)
        self.model_registry = ModelRegistry()
        self.scheduler = Scheduler(self.resource_manager, remote_available=False, allow_local_gpu=pm.config.local_gpu_enabled)

        # One ModelLifecycleManager shared by every agent in this pipeline,
        # so at most one heavy generative model is ever resident in
        # GPU/RAM/disk at a time (spec section 3/10): image generation
        # downloads, runs, and fully unloads before video generation's
        # model is even selected.
        self.model_lifecycle = ModelLifecycleManager(
            model_registry=self.model_registry, resource_manager=self.resource_manager,
            runtime_dir=runtime_dir or str(pm.store.dir_for("tmp") / "model_runtime"),
            mock_mode=pm.config.mock_mode,
            status_callback=status_callback or console_status_printer,
        )

        self.research_agent = ResearchAgent(MockResearchProvider())
        self.fact_checker = FactChecker()
        self.story_agent = StoryAgent()
        self.script_agent = ScriptAgent()
        self.scene_agent = SceneAgent()
        self.visual_agent = VisualAgent(
            model_registry=self.model_registry, resource_manager=self.resource_manager,
            scheduler=self.scheduler, media_provider=MockMediaProvider(),
            assets_dir=str(pm.store.dir_for("assets")),
            video_width=video_width, video_height=video_height, fps=fps,
            model_lifecycle=self.model_lifecycle,
        )

        # Long-form and shorts each get their own voice engine instance so
        # that any per-call/streaming state (buffers, sample queues, etc.)
        # kept by a real engine like PiperVoiceEngine is never shared across
        # concurrent or interleaved narration jobs.
        voice_engine = self._create_voice_engine()
        short_voice_engine = self._create_voice_engine()

        self.audio_agent = AudioAgent(
            voice_engine=voice_engine, audio_dir=str(pm.store.dir_for("audio")),
            captions_dir=str(pm.store.dir_for("captions")),
            music_library_dir="assets/music", sfx_library_dir="assets/sfx",
            voice=pm.config.voice, language=pm.config.language,
        )
        self.shorts_agent = ShortsAgent()
        self.metadata_agent = MetadataAgent()
        self.qa_agent = QAAgent()
        self.thumbnail_agent = ThumbnailAgent()
        # Shorts render vertically (9:16) regardless of the long-form target size.
        self.short_width, self.short_height, self.short_fps = short_width, short_height, short_fps
        self._short_visual_agent = VisualAgent(
            model_registry=self.model_registry, resource_manager=self.resource_manager,
            scheduler=self.scheduler, media_provider=MockMediaProvider(),
            assets_dir=str(pm.store.dir_for("shorts")),
            video_width=short_width, video_height=short_height, fps=short_fps,
            model_lifecycle=self.model_lifecycle,
        )
        self._short_audio_agent = AudioAgent(
            voice_engine=short_voice_engine, audio_dir=str(pm.store.dir_for("shorts")),
            captions_dir=str(pm.store.dir_for("shorts")),
            music_library_dir="assets/music", sfx_library_dir="assets/sfx",
            voice=pm.config.voice, language=pm.config.language,
        )

    # ------------------------------------------------------------------

    def _create_voice_engine(self):
        """Create the configured TTS engine.

        Mock mode uses the deterministic mock engine.
        Real mode selects the configured provider.
        """
        if self.pm.config.mock_mode:
            logger.info("TTS: using MockVoiceEngine because mock_mode=true")
            return MockVoiceEngine()

        tts_config = self.pm.config.tts
        provider = tts_config.provider.lower()

        if provider == "piper":
            model_path = tts_config.model_path

            if not model_path:
                raise ValueError(
                    "Piper TTS selected but tts.model_path is empty."
                )

            logger.info(
                "TTS: using PiperVoiceEngine model=%s",
                model_path,
            )

            return PiperVoiceEngine(
                model_path=model_path,
                use_cuda=tts_config.use_cuda,
            )

        if provider == "mock":
            # mock_mode=false at the project level but the *TTS* provider
            # specifically is still "mock" (e.g. Piper weights weren't
            # available yet) -- honor that per-capability fallback rather
            # than failing the whole pipeline over one missing model.
            logger.info("TTS: using MockVoiceEngine because tts.provider=mock")
            return MockVoiceEngine()

        raise ValueError(
            f"Unsupported TTS provider: {tts_config.provider}"
        )

    # ------------------------------------------------------------------

    def run_research(self) -> Research:
        existing = self.pm.load_model("research.json", Research)
        if existing:
            logger.info("Resuming: research already complete.")
            return existing
        self.pm.transition(ProjectState.RESEARCHING, force=True)
        research = self.research_agent.run(self.pm.config.topic, depth=self.pm.config.research_depth)
        self.pm.save_model("research.json", research)
        self.pm.transition(ProjectState.RESEARCH_COMPLETE)
        return research

    def run_fact_check(self, research: Research) -> FactCheckReport:
        existing = self.pm.load_model("facts.json", FactCheckReport)
        if existing:
            logger.info("Resuming: fact check already complete.")
            return existing
        self.pm.transition(ProjectState.FACT_CHECKING, force=True)
        report = self.fact_checker.run(research)
        self.pm.save_model("facts.json", report)
        if report.has_critical_failure:
            logger.warning("Critical fact-check failures found; proceeding with flagged claims (mock mode).")
        self.pm.transition(ProjectState.FACT_CHECK_COMPLETE, force=True)
        return report

    def run_story(self, research: Research) -> Story:
        existing = self.pm.load_model("story.json", Story)
        if existing:
            return existing
        story = self.story_agent.run(research)
        self.pm.save_model("story.json", story)
        return story

    def run_script(self, research: Research, story: Story) -> Script:
        existing = self.pm.load_model("script.json", Script)
        if existing:
            logger.info("Resuming: script already complete.")
            return existing
        self.pm.transition(ProjectState.SCRIPTING, force=True)
        script = self.script_agent.run(research, story)
        self.pm.save_model("script.json", script)
        self.pm.transition(ProjectState.SCRIPT_COMPLETE)
        return script

    def run_scenes(self, script: Script, story: Story) -> ScenePlan:
        existing = self.pm.load_model("scenes.json", ScenePlan)
        if existing:
            logger.info("Resuming: scene plan already complete.")
            return existing
        self.pm.transition(ProjectState.SCENE_PLANNING, force=True)
        plan = self.scene_agent.run(script, story)
        self.pm.save_model("scenes.json", plan)
        self.pm.transition(ProjectState.SCENES_COMPLETE)
        return plan

    def run_assets(self, scene_plan: ScenePlan) -> AssetManifest:
        existing = self.pm.load_model("assets.json", AssetManifest)
        if existing:
            logger.info("Resuming: assets already complete.")
            return existing
        self.pm.transition(ProjectState.ASSET_GENERATION, force=True)
        manifest = self.visual_agent.run(scene_plan)
        self.pm.save_model("assets.json", manifest)
        self.pm.transition(ProjectState.ASSETS_COMPLETE)
        return manifest

    def run_audio(self, scene_plan: ScenePlan) -> tuple[VoiceManifest, MusicManifest]:
        existing_voice = self.pm.load_model("voice.json", VoiceManifest)
        existing_music = self.pm.load_model("music.json", MusicManifest)
        if existing_voice and existing_music:
            logger.info("Resuming: audio already complete.")
            return existing_voice, existing_music
        self.pm.transition(ProjectState.AUDIO_GENERATION, force=True)
        voice_manifest, music_manifest = self.audio_agent.run(scene_plan)
        self.pm.save_model("voice.json", voice_manifest)
        self.pm.save_model("music.json", music_manifest)
        self.pm.transition(ProjectState.AUDIO_COMPLETE)
        return voice_manifest, music_manifest

    def run_render(
        self, asset_manifest: AssetManifest, voice_manifest: VoiceManifest,
    ) -> str:
        render_dir = self.pm.store.dir_for("renders")
        output_path = str(render_dir / "long_form.mp4")

        if self.pm.store.has_checkpoint("render_manifest.json"):
            logger.info("Resuming: render already complete.")
            return output_path

        self.pm.transition(ProjectState.RENDERING, force=True)

        clip_paths = [a.file_path for a in asset_manifest.assets if a.file_path]
        video_only_path = str(render_dir / "video_only.mp4")
        concat_clips(clip_paths, video_only_path)

        # Concatenate per-scene narration into one track matching the video.
        voice_paths = [t.file_path for t in voice_manifest.tracks]
        combined_voice_path = str(render_dir / "voice_combined.wav")
        concat_clips(voice_paths, combined_voice_path)  # concat demuxer also works for audio-only files

        mux_audio(video_only_path, combined_voice_path, output_path)

        from core.models import RenderManifest, RenderManifestEntry, RenderTarget
        manifest = RenderManifest(entries=[RenderManifestEntry(
            target=RenderTarget.LONG_FORM, output_path=output_path,
            width=self.video_width, height=self.video_height, fps=self.fps,
            duration_seconds=sum(t.duration_seconds for t in voice_manifest.tracks),
        )])
        self.pm.save_model("render_manifest.json", manifest)
        self.pm.transition(ProjectState.RENDER_COMPLETE)
        return output_path

    def run_qa(
        self, research: Research, asset_manifest: AssetManifest, render_path: str, scene_plan: ScenePlan,
    ) -> object:
        existing = self.pm.store.load_checkpoint("qa.json")
        from core.models import QAReport
        if existing:
            logger.info("Resuming: QA already complete.")
            return QAReport.model_validate(existing)

        self.pm.transition(ProjectState.QA, force=True)
        scene_texts = [(s.narration, s.duration_seconds) for s in scene_plan.scenes]
        segments = segments_from_scenes(scene_texts)
        total_duration = sum(s.duration_seconds for s in scene_plan.scenes)

        report = self.qa_agent.run(
            research=research, asset_manifest=asset_manifest, render_path=render_path,
            expected_width=self.video_width, expected_height=self.video_height, expected_fps=self.fps,
            caption_segments=segments, total_duration_seconds=total_duration,
        )
        self.pm.save_model("qa.json", report)

        from core.models import QAStatus
        if report.status == QAStatus.REJECT:
            self.pm.transition(ProjectState.QA_FAILED, force=True)
        else:
            self.pm.transition(ProjectState.HUMAN_REVIEW, force=True)
        return report

    def run_shorts(self, research: Research, script: Script) -> tuple[list, list[str]]:
        """Build and render each short as its own independent vertical clip."""
        shorts = self.shorts_agent.run(research, script, short_count=self.pm.config.short_count)
        shorts_dir = self.pm.store.dir_for("shorts")
        render_paths = []

        for short in shorts:
            out_path = str(shorts_dir / f"{short.short_id}.mp4")
            if Path(out_path).exists():
                render_paths.append(out_path)
                continue

            asset_manifest = self._short_visual_agent.run(short.scene_plan)
            voice_manifest, _music_manifest = self._short_audio_agent.run(short.scene_plan)

            clip_paths = [a.file_path for a in asset_manifest.assets if a.file_path]
            video_only = str(shorts_dir / f"{short.short_id}_video_only.mp4")
            concat_clips(clip_paths, video_only)

            voice_paths = [t.file_path for t in voice_manifest.tracks]
            combined_voice = str(shorts_dir / f"{short.short_id}_voice.wav")
            concat_clips(voice_paths, combined_voice)

            mux_audio(video_only, combined_voice, out_path)
            render_paths.append(out_path)

        return shorts, render_paths

    def run_thumbnails(self, asset_manifest: AssetManifest, metadata) -> list[str]:
        thumbnails_dir = self.pm.store.dir_for("thumbnails")
        source_images = [
            a.metadata.get("source_image") for a in asset_manifest.assets
            if a.metadata.get("source_image")
        ]
        if not source_images:
            # Fall back to any still frame we can find among generated PNGs.
            source_images = [
                str(p) for p in self.pm.store.dir_for("assets").glob("*_source.png")
            ]
        return self.thumbnail_agent.run(
            source_images, metadata.title_options, str(thumbnails_dir),
        )

    # ------------------------------------------------------------------

    def run_full(self) -> PipelineResult:
        research = self.run_research()
        fact_check = self.run_fact_check(research)
        story = self.run_story(research)
        script = self.run_script(research, story)
        scene_plan = self.run_scenes(script, story)
        asset_manifest = self.run_assets(scene_plan)
        voice_manifest, music_manifest = self.run_audio(scene_plan)
        render_path = self.run_render(asset_manifest, voice_manifest)
        qa_report = self.run_qa(research, asset_manifest, render_path, scene_plan)
        shorts, short_render_paths = self.run_shorts(research, script)
        metadata = self.metadata_agent.run(script, research.topic)
        thumbnail_paths = self.run_thumbnails(asset_manifest, metadata)

        return PipelineResult(
            research=research, fact_check=fact_check, story=story, script=script,
            scene_plan=scene_plan, asset_manifest=asset_manifest, voice_manifest=voice_manifest,
            music_manifest=music_manifest, render_path=render_path, qa_report=qa_report, shorts=shorts,
            short_render_paths=short_render_paths, thumbnail_paths=thumbnail_paths, metadata=metadata,
        )