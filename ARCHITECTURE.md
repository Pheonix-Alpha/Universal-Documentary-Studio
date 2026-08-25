# Architecture

## High-level flow

```
TOPIC
  -> ResearchAgent        -> research.json
  -> FactChecker          -> facts.json
  -> StoryAgent           -> story.json
  -> ScriptAgent          -> script.json
  -> SceneAgent           -> scenes.json
  -> VisualAgent          -> assets.json      (per-scene visual generation)
  -> AudioAgent           -> voice.json, music.json, captions/*.srt|.vtt
  -> FFmpeg renderer      -> renders/long_form.mp4
  -> QAAgent              -> qa.json
  -> ShortsAgent          -> shorts/*.mp4     (independent vertical re-cuts)
  -> ThumbnailAgent       -> thumbnails/*.png
  -> MetadataAgent        -> titles, description, chapters, keywords
  -> HUMAN REVIEW         (Gradio dashboard; nothing auto-publishes)
```

Every arrow above is a checkpoint boundary: `core/pipeline.py`'s
`DocumentaryPipeline` checks `ProjectManager` for an existing checkpoint
before running a stage, so re-invoking `run_full()` on a partially
completed project resumes rather than redoes work.

## Directory layout

```
app/            CLI entry point, Gradio UI, app-level config loading
core/           data models, state machine, resource manager, scheduler,
                project manager, pipeline orchestrator, cache facade
agents/         one module per pipeline stage (see flow above)
engines/        the actual media-generation code (FFmpeg, Pillow, matplotlib)
adapters/       provider-agnostic interfaces + mock implementations
                (research, image gen, video gen, TTS, ASR, media sources)
models/         ModelCapability schema + ModelRegistry
qa/             one checker per QA category + score aggregation
storage/        ProjectStore (checkpoints) and CacheStore (content-addressed cache)
projects/       per-project output (created at runtime)
config/         config.yaml (app settings) and models.yaml (model catalog)
notebooks/colab/ the Colab GPU worker notebook
tests/          unit, integration, and end-to-end tests (all MOCK_MODE)
```

## Resource management

`core/resource_manager.py` probes the actual runtime (GPU name/VRAM, RAM,
CPU, disk, CUDA) rather than assuming a fixed machine. It classifies the
worker into `LIGHT` / `MEDIUM` / `HEAVY` / `VERY_HEAVY` based on VRAM
thresholds and enforces a configurable safety margin
(`resource.vram_safety_margin_gb`, default 1.5GB) so a job is never
scheduled against 100% of available VRAM.

`models/registry.py`'s `ModelRegistry` holds a declarative catalog of
model capabilities (min/recommended VRAM, license, commercial-use flag,
supported tasks, quality rank) and returns the *highest-quality
compatible* model for a given available-VRAM budget — never a
hard-coded model name. If nothing fits, callers get `None` and must use
a non-AI fallback (this is enforced in `agents/visual_agent.py`, not in
the registry itself).

`core/scheduler.py` implements the remote-GPU-first decision flow:

```
remote GPU available and sufficient?  -> execute remotely
local GPU explicitly opted in?        -> execute locally
lightweight fallback available?       -> use fallback
otherwise                              -> queue the job (paused)
```

## Memory efficiency

Scenes are generated, rendered, and released one at a time
(`engines/rendering/ffmpeg_renderer.py` uses FFmpeg's concat demuxer to
stitch already-encoded per-scene clips rather than holding decoded frames
in memory). No engine loads an entire video or every generated asset
into RAM at once.

## State machine

`core/models.py`'s `STATE_TRANSITIONS` graph is the single source of
truth for legal project states. `core/state.py`'s `StateMachine` raises
`StateTransitionError` on an illegal transition unless the caller
explicitly passes `force=True` (used for regeneration flows like
`QA_FAILED -> ASSET_GENERATION`).

## MOCK_MODE

Every adapter has a mock implementation that produces **real files** —
actual PNGs (Pillow), actual WAV files sized to narration length, actual
MP4 clips (FFmpeg) — rather than empty stubs. This means the entire
pipeline, including FFmpeg rendering and FFprobe-based QA, is exercised
in CI without any GPU, model download, or paid API key. See
`adapters/*/mock_*.py`.

## Human review

`ProjectState.HUMAN_REVIEW` is a required stop: `run_qa` transitions
either to `HUMAN_REVIEW` or `QA_FAILED`, never directly to `APPROVED`.
Only an explicit call (via the Gradio dashboard's APPROVE button, or
`ProjectManager.transition(ProjectState.APPROVED)`) can move a project
further.
