"""
Simplified main-screen UI (spec section 54 + launcher requirement #8/#10).

This is the screen a normal user sees: Topic, Video type, Duration,
Language, Voice, Number of Shorts, and a single GENERATE button — no
per-stage buttons required. Advanced per-stage controls (the existing
`app.ui` dashboard) stay available in a collapsed "Advanced" section for
development/debugging, unchanged.

The GENERATE flow drives the exact same stage methods `run_full()` calls,
in the exact same order — it does not reimplement pipeline logic — but it
calls them one at a time (instead of via run_full()) so this UI can
stream a live progress bar and a status/error log back to the browser
after every stage. This gives the "is it stuck or not" visibility the
Colab launcher spec asks for, without changing DocumentaryPipeline itself.
"""

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Iterator

import gradio as gr

from app.config import load_app_config
from core.models import ProjectConfig, ProjectState, TTSConfig
from core.pipeline import DocumentaryPipeline
from core.project_manager import ProjectManager

# (label, weight) — weight is a rough relative-time share used only to
# make the progress bar move at a believable pace; it has no effect on
# pipeline behavior.
_STAGES = [
    ("Research", 10),
    ("Fact-checking", 5),
    ("Story structure", 5),
    ("Script", 10),
    ("Scene planning", 10),
    ("Generating visual assets", 20),
    ("Generating audio", 15),
    ("Rendering", 15),
    ("Quality control", 5),
    ("Generating Shorts", 5),
    ("Metadata", 2),
    ("Thumbnails", 3),
]
_TOTAL_WEIGHT = sum(w for _, w in _STAGES)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log_line(kind: str, message: str) -> str:
    marker = {"info": "  ", "ok": "✓ ", "error": "✗ ", "start": "▶ "}.get(kind, "  ")
    return f"[{_timestamp()}] {marker}{message}"


def run_generation(
    topic: str,
    video_type: str,
    language: str,
    duration: float,
    voice: str,
    short_count: int,
) -> Iterator[tuple]:
    """Generator driving the full pipeline stage-by-stage, yielding
    (progress_fraction, status_line, log_text, video_path) after every
    stage so Gradio can render a live % bar + scrolling log."""

    log_lines: list[str] = []

    def emit(progress: float, status: str, video=None):
        return progress, status, "\n".join(log_lines), video

    if not topic or not topic.strip():
        log_lines.append(_log_line("error", "No topic provided."))
        yield emit(0.0, "Error: please enter a topic.")
        return

    log_lines.append(_log_line("info", f"Video type: {video_type} (documentary pipeline; format is fixed for now)"))
    log_lines.append(_log_line("start", f"Creating project for topic: {topic!r}"))
    yield emit(0.0, "Starting...")

    try:
        app_config = load_app_config()
        config = ProjectConfig(
            topic=topic,
            language=language,
            target_duration_minutes=float(duration),
            short_count=int(short_count),
            voice=voice,
            research_depth=app_config["pipeline"]["research_depth"],
            mock_mode=app_config["mock_mode"],
            local_gpu_enabled=app_config["local_gpu_enabled"],
            tts=TTSConfig(**app_config["tts"]),
        )
        pm = ProjectManager(projects_root=app_config["projects_root"], config=config)
        video_cfg = app_config["video"]
        pipeline = DocumentaryPipeline(
            pm,
            video_width=video_cfg["long_form"]["width"], video_height=video_cfg["long_form"]["height"],
            fps=video_cfg["long_form"]["fps"], short_width=video_cfg["short"]["width"],
            short_height=video_cfg["short"]["height"], short_fps=video_cfg["short"]["fps"],
        )
    except Exception as exc:  # noqa: BLE001
        log_lines.append(_log_line("error", f"Failed to create project: {exc}"))
        log_lines.append(traceback.format_exc(limit=3))
        yield emit(0.0, "Error creating project — see log below.")
        return

    log_lines.append(_log_line("ok", f"Project created: {config.project_id} (mock_mode={config.mock_mode})"))

    done_weight = 0
    research = fact_check = story = script = scene_plan = None
    asset_manifest = voice_manifest = music_manifest = None
    render_path = qa_report = metadata = None

    # Each entry: (label, callable). Every callable takes no args and
    # closes over the variables above, mirroring run_full()'s exact
    # sequence and dependencies — this is intentionally the same call
    # graph as DocumentaryPipeline.run_full(), just interruptible/observable.
    def _stage_research():
        nonlocal research
        research = pipeline.run_research()
        return f"{len(research.sources)} sources, {len(research.claims)} claims."

    def _stage_fact_check():
        nonlocal fact_check
        fact_check = pipeline.run_fact_check(research)
        total = len(fact_check.results)
        failed = sum(1 for r in fact_check.results if r.status.value == "failed")
        return f"{total - failed}/{total} claims passed" + (f" ({failed} failed)" if failed else ".")

    def _stage_story():
        nonlocal story
        story = pipeline.run_story(research)
        primary = story.structures[0].value if story.structures else "chronology"
        return f"Structure: {primary}."

    def _stage_script():
        nonlocal script
        script = pipeline.run_script(research, story)
        return f"{len(script.sections)} sections (~{script.estimated_duration_seconds:.0f}s)."

    def _stage_scenes():
        nonlocal scene_plan
        scene_plan = pipeline.run_scenes(script, story)
        return f"{len(scene_plan.scenes)} scenes planned."

    def _stage_assets():
        nonlocal asset_manifest
        asset_manifest = pipeline.run_assets(scene_plan)
        return f"{len(asset_manifest.assets)} assets generated."

    def _stage_audio():
        nonlocal voice_manifest, music_manifest
        voice_manifest, music_manifest = pipeline.run_audio(scene_plan)
        return f"{len(voice_manifest.tracks)} narration tracks."

    def _stage_render():
        nonlocal render_path
        render_path = pipeline.run_render(asset_manifest, voice_manifest)
        return f"Rendered: {render_path}"

    def _stage_qa():
        nonlocal qa_report
        qa_report = pipeline.run_qa(research, asset_manifest, render_path, scene_plan)
        return f"QA score {qa_report.score} ({qa_report.status.value})."

    def _stage_shorts():
        nonlocal short_render_paths
        _shorts, short_render_paths = pipeline.run_shorts(research, script)
        return f"{len(short_render_paths)} Shorts rendered."

    def _stage_metadata():
        nonlocal metadata
        metadata = pipeline.metadata_agent.run(script, research.topic)
        return f"Titles, description, {len(metadata.chapters)} chapters generated."

    def _stage_thumbnails():
        thumbnail_paths = pipeline.run_thumbnails(asset_manifest, metadata)
        return f"{len(thumbnail_paths)} thumbnail concepts generated."

    short_render_paths: list = []
    stage_fns = [
        _stage_research, _stage_fact_check, _stage_story, _stage_script, _stage_scenes,
        _stage_assets, _stage_audio, _stage_render, _stage_qa, _stage_shorts,
        _stage_metadata, _stage_thumbnails,
    ]

    for (label, weight), fn in zip(_STAGES, stage_fns):
        log_lines.append(_log_line("start", f"{label}..."))
        progress = done_weight / _TOTAL_WEIGHT
        yield emit(progress, f"{label}...")
        try:
            detail = fn()
            done_weight += weight
            progress = done_weight / _TOTAL_WEIGHT
            log_lines.append(_log_line("ok", f"{label}: {detail}"))
            video = render_path if label == "Rendering" else None
            yield emit(progress, f"{label} complete.", video=video)
        except Exception as exc:  # noqa: BLE001
            log_lines.append(_log_line("error", f"{label} failed: {exc}"))
            log_lines.append(traceback.format_exc(limit=4))
            yield emit(done_weight / _TOTAL_WEIGHT, f"Error during {label} — see log below.")
            return

    pm.transition(ProjectState.HUMAN_REVIEW, force=True)
    log_lines.append(_log_line("ok", f"Pipeline complete. QA score {qa_report.score} ({qa_report.status.value})."))
    log_lines.append(_log_line("info", "Awaiting human review — open the Advanced tab to APPROVE / REGENERATE / REJECT."))
    yield emit(1.0, f"Done — QA score {qa_report.score} ({qa_report.status.value}). Awaiting human review.", video=render_path)


def build_simple_interface() -> gr.Blocks:
    with gr.Blocks(title="Universal Documentary Studio") as demo:
        gr.Markdown("# Universal Documentary Studio")

        with gr.Row():
            topic = gr.Textbox(label="Topic", placeholder="e.g. The History of the Indian Space Program", scale=3)
        with gr.Row():
            video_type = gr.Dropdown(label="Video type", choices=["Documentary"], value="Documentary")
            duration = gr.Number(label="Duration (minutes)", value=10.0)
            language = gr.Dropdown(label="Language", choices=["English", "en"], value="en")
        with gr.Row():
            voice = gr.Textbox(label="Voice", value="Ryan")
            short_count = gr.Slider(label="Number of Shorts", minimum=3, maximum=5, step=1, value=4)

        generate_btn = gr.Button("GENERATE", variant="primary", size="lg")

        gr.Markdown("## Status")
        progress_bar = gr.Slider(label="Progress", minimum=0, maximum=100, value=0, interactive=False)
        status_line = gr.Textbox(label="Current step", interactive=False)
        video_preview = gr.Video(label="Preview")
        log_box = gr.Textbox(label="Log (downloads, stage progress, errors)", lines=14, interactive=False, autoscroll=True)

        def _wrapped(topic_, video_type_, language_, duration_, voice_, short_count_):
            for progress, status, log, video in run_generation(
                topic_, video_type_, language_, duration_, voice_, short_count_
            ):
                yield round(progress * 100, 1), status, log, video

        generate_btn.click(
            _wrapped,
            inputs=[topic, video_type, language, duration, voice, short_count],
            outputs=[progress_bar, status_line, log_box, video_preview],
        )

        with gr.Accordion("Advanced (per-stage controls, human review)", open=False):
            gr.Markdown(
                "Full per-stage buttons (RESEARCH / SCRIPT / SCENE PLAN / ... / "
                "APPROVE / REGENERATE / REJECT) live in the advanced dashboard."
            )
            from app.ui import build_interface as _build_advanced

            _build_advanced()

    return demo


def launch_ui() -> None:
    demo = build_simple_interface()
    demo.launch()


if __name__ == "__main__":
    launch_ui()
