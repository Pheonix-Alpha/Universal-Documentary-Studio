"""Gradio UI (spec section 54): inputs for topic/language/duration/etc,
per-stage buttons, and a human-review dashboard with APPROVE / REGENERATE /
REJECT actions. Human approval is mandatory -- there is no "publish"
button because publishing is intentionally out of scope for automation.

Also implements the "UI should show everything" requirement (spec
section 4): a HARDWARE panel (GPU/VRAM/RAM/disk) and a live model
lifecycle log (WAITING -> CHECKING_RESOURCES -> ... -> COMPLETED, per
model, per stage) instead of a bare "Something is happening..." spinner.
"""
from __future__ import annotations

from typing import Optional

import gradio as gr

from app.config import load_app_config
from core.model_lifecycle import LifecycleEvent
from core.models import ProjectConfig, ProjectState, TTSConfig
from core.pipeline import DocumentaryPipeline
from core.project_manager import ProjectManager
from core.resource_manager import ResourceManager

_STATE: dict[str, object] = {"pm": None, "pipeline": None, "research": None, "story": None,
                             "script": None, "scene_plan": None, "asset_manifest": None,
                             "voice_manifest": None, "music_manifest": None, "render_path": None,
                             "qa_report": None, "lifecycle_log": []}

_MAX_LOG_LINES = 400


def _format_hardware(rm: ResourceManager) -> str:
    report = rm.detect()
    lines = ["### HARDWARE"]
    if report.gpu_available:
        lines.append(f"- ✓ GPU: {report.gpu_name}")
        lines.append(f"- ✓ VRAM: {report.vram_gb:.2f} GB (effective: {rm.effective_vram_gb():.2f} GB)")
    else:
        lines.append("- ✗ GPU: none detected (CPU-only / mock generation)")
    lines.append(f"- ✓ RAM: {report.ram_gb:.2f} GB")
    lines.append(f"- ✓ CPU cores: {report.cpu_cores}")
    lines.append(f"- ✓ Disk free: {report.disk_free_gb:.2f} GB")
    return "\n".join(lines)


def _lifecycle_callback(event: LifecycleEvent) -> None:
    """Appended to _STATE so build_interface can surface it after each
    pipeline stage call -- Gradio's simple `.click(fn, ...)` wiring here is
    synchronous, so we render the whole trail rather than streaming."""
    bits = [f"[{event.task}]", event.status.value]
    if event.model_name:
        bits.append(f"model={event.model_name}")
    if event.message:
        bits.append(f"— {event.message}")
    log: list = _STATE["lifecycle_log"]  # type: ignore[assignment]
    log.append(" ".join(bits))
    del log[:-_MAX_LOG_LINES]


def _lifecycle_text() -> str:
    log: list = _STATE.get("lifecycle_log") or []  # type: ignore[assignment]
    if not log:
        return "(no model lifecycle events yet)"
    return "\n".join(log)


def _new_project(topic, language, duration, shorts, voice, depth):
    app_config = load_app_config()
    config = ProjectConfig(
        topic=topic, language=language, target_duration_minutes=duration,
        short_count=int(shorts), voice=voice, research_depth=depth,
        mock_mode=app_config["mock_mode"], local_gpu_enabled=app_config["local_gpu_enabled"],
        tts=TTSConfig(**app_config["tts"]),
    )
    pm = ProjectManager(projects_root=app_config["projects_root"], config=config)
    video_cfg = app_config["video"]
    _STATE["lifecycle_log"] = []
    pipeline = DocumentaryPipeline(
        pm, video_width=video_cfg["long_form"]["width"], video_height=video_cfg["long_form"]["height"],
        fps=video_cfg["long_form"]["fps"], short_width=video_cfg["short"]["width"],
        short_height=video_cfg["short"]["height"], short_fps=video_cfg["short"]["fps"],
        status_callback=_lifecycle_callback,
    )
    _STATE["pm"] = pm
    _STATE["pipeline"] = pipeline
    hardware = _format_hardware(pipeline.resource_manager)
    mode = "MOCK_MODE (fast, no downloads)" if config.mock_mode else "REAL MODE (downloads real models as needed)"
    status = f"Created project {config.project_id} (state={pm.state_machine.current_state.value})\nMode: {mode}"
    return status, hardware, _lifecycle_text()


def _run_research():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    if pipeline is None:
        return "Create a project first.", _lifecycle_text()
    research = pipeline.run_research()
    _STATE["research"] = research
    return f"Research complete: {len(research.sources)} sources, {len(research.claims)} claims.", _lifecycle_text()


def _run_script():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    research = _STATE.get("research")
    if pipeline is None or research is None:
        return "Run research first.", _lifecycle_text()
    fact_check = pipeline.run_fact_check(research)
    story = pipeline.run_story(research)
    script = pipeline.run_script(research, story)
    _STATE["story"], _STATE["script"] = story, script
    return f"Script complete: {len(script.sections)} sections (~{script.estimated_duration_seconds:.0f}s).", _lifecycle_text()


def _run_scene_plan():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    script, story = _STATE.get("script"), _STATE.get("story")
    if pipeline is None or script is None:
        return "Run script first.", _lifecycle_text()
    scene_plan = pipeline.run_scenes(script, story)
    _STATE["scene_plan"] = scene_plan
    return f"Scene plan complete: {len(scene_plan.scenes)} scenes, {scene_plan.total_duration_seconds:.0f}s total.", _lifecycle_text()


def _run_assets():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    scene_plan = _STATE.get("scene_plan")
    if pipeline is None or scene_plan is None:
        return "Run scene planning first.", _lifecycle_text()
    asset_manifest = pipeline.run_assets(scene_plan)
    _STATE["asset_manifest"] = asset_manifest
    return f"Assets generated: {len(asset_manifest.assets)}.", _lifecycle_text()


def _run_audio():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    scene_plan = _STATE.get("scene_plan")
    if pipeline is None or scene_plan is None:
        return "Run scene planning first.", _lifecycle_text()
    voice_manifest, music_manifest = pipeline.run_audio(scene_plan)
    _STATE["voice_manifest"], _STATE["music_manifest"] = voice_manifest, music_manifest
    return f"Audio generated: {len(voice_manifest.tracks)} narration tracks.", _lifecycle_text()


def _run_render():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    asset_manifest, voice_manifest = _STATE.get("asset_manifest"), _STATE.get("voice_manifest")
    if pipeline is None or asset_manifest is None or voice_manifest is None:
        return "Generate assets and audio first.", None, _lifecycle_text()
    render_path = pipeline.run_render(asset_manifest, voice_manifest)
    _STATE["render_path"] = render_path
    return f"Render complete: {render_path}", render_path, _lifecycle_text()


def _run_qa():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    research, asset_manifest = _STATE.get("research"), _STATE.get("asset_manifest")
    render_path, scene_plan = _STATE.get("render_path"), _STATE.get("scene_plan")
    if pipeline is None or None in (research, asset_manifest, render_path, scene_plan):
        return "Complete rendering first.", _lifecycle_text()
    qa_report = pipeline.run_qa(research, asset_manifest, render_path, scene_plan)
    _STATE["qa_report"] = qa_report
    issue_lines = "\n".join(f"- [{i.severity}] {i.category}: {i.message}" for i in qa_report.issues) or "No issues found."
    return f"QA score: {qa_report.score} ({qa_report.status.value})\n{issue_lines}", _lifecycle_text()


def _run_full_pipeline():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    if pipeline is None:
        return "Create a project first.", None, _lifecycle_text()
    result = pipeline.run_full()
    _STATE.update({
        "research": result.research, "story": result.story, "script": result.script,
        "scene_plan": result.scene_plan, "asset_manifest": result.asset_manifest,
        "voice_manifest": result.voice_manifest, "music_manifest": result.music_manifest,
        "render_path": result.render_path, "qa_report": result.qa_report,
    })
    summary = (
        f"Pipeline complete. State: {_STATE['pm'].state_machine.current_state.value}\n"
        f"QA score: {result.qa_report.score} ({result.qa_report.status.value})\n"
        f"Shorts: {len(result.shorts)} | Thumbnails: {len(result.thumbnail_paths)}"
    )
    return summary, result.render_path, _lifecycle_text()


def _approve():
    pm: ProjectManager = _STATE["pm"]
    if pm is None:
        return "No project loaded."
    pm.transition(ProjectState.APPROVED, force=True)
    return f"Project approved. State: {pm.state_machine.current_state.value}"


def _reject():
    pm: ProjectManager = _STATE["pm"]
    if pm is None:
        return "No project loaded."
    pm.transition(ProjectState.FAILED, force=True)
    return f"Project rejected. State: {pm.state_machine.current_state.value}"


def _regenerate_scene(scene_index: int):
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    scene_plan = _STATE.get("scene_plan")
    if pipeline is None or scene_plan is None:
        return "Run scene planning first.", _lifecycle_text()
    try:
        scene = scene_plan.scenes[int(scene_index)]
    except (IndexError, ValueError):
        return "Invalid scene index.", _lifecycle_text()
    # regenerate_scene() acquires whatever model is needed and releases it
    # again immediately afterward -- a one-off regen never leaves a heavy
    # model resident.
    new_asset = pipeline.visual_agent.regenerate_scene(scene)
    return f"Regenerated scene {scene_index}: {new_asset.file_path}", _lifecycle_text()


def _regenerate_voice():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    scene_plan = _STATE.get("scene_plan")
    if pipeline is None or scene_plan is None:
        return "Run scene planning first."
    voice_manifest, music_manifest = pipeline.audio_agent.run(scene_plan)
    _STATE["voice_manifest"], _STATE["music_manifest"] = voice_manifest, music_manifest
    return "Voice regenerated for all scenes."


def _regenerate_script():
    pipeline: DocumentaryPipeline = _STATE["pipeline"]
    research, story = _STATE.get("research"), _STATE.get("story")
    if pipeline is None or research is None:
        return "Run research first."
    pipeline.pm.store.delete_checkpoint("script.json")
    pipeline.pm.store.delete_checkpoint("scenes.json")
    script = pipeline.run_script(research, story)
    _STATE["script"] = script
    return f"Script regenerated: {len(script.sections)} sections."


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Universal Documentary Studio") as demo:
        gr.Markdown("# Universal Documentary Studio — Daily Production Dashboard")

        with gr.Row():
            topic = gr.Textbox(label="Topic")
            language = gr.Textbox(label="Language", value="en")
            duration = gr.Number(label="Target duration (minutes)", value=10.0)
        with gr.Row():
            shorts = gr.Slider(label="Short count", minimum=3, maximum=5, step=1, value=4)
            voice = gr.Textbox(label="Voice", value="default")
            depth = gr.Dropdown(label="Research depth", choices=["quick", "standard", "deep"], value="standard")

        create_btn = gr.Button("CREATE PROJECT")
        status = gr.Textbox(label="Status", interactive=False)
        hardware_panel = gr.Markdown(label="Hardware")

        gr.Markdown("## Pipeline Stages")
        with gr.Row():
            research_btn = gr.Button("RESEARCH")
            script_btn = gr.Button("SCRIPT")
            scenes_btn = gr.Button("SCENE PLAN")
            assets_btn = gr.Button("GENERATE ASSETS")
        with gr.Row():
            audio_btn = gr.Button("GENERATE AUDIO")
            render_btn = gr.Button("RENDER")
            qa_btn = gr.Button("QA")
            full_btn = gr.Button("RUN FULL PIPELINE", variant="primary")

        stage_output = gr.Textbox(label="Stage output", interactive=False)
        video_preview = gr.Video(label="Preview")

        gr.Markdown(
            "## Model Lifecycle\n"
            "Every model this run selects, downloads, loads, generates with, "
            "and unloads shows up here — never just a bare spinner."
        )
        lifecycle_log = gr.Textbox(label="Lifecycle log", interactive=False, lines=14, max_lines=14, autoscroll=True)

        create_btn.click(_new_project, [topic, language, duration, shorts, voice, depth],
                          [status, hardware_panel, lifecycle_log])
        research_btn.click(_run_research, None, [stage_output, lifecycle_log])
        script_btn.click(_run_script, None, [stage_output, lifecycle_log])
        scenes_btn.click(_run_scene_plan, None, [stage_output, lifecycle_log])
        assets_btn.click(_run_assets, None, [stage_output, lifecycle_log])
        audio_btn.click(_run_audio, None, [stage_output, lifecycle_log])
        render_btn.click(_run_render, None, [stage_output, video_preview, lifecycle_log])
        qa_btn.click(_run_qa, None, [stage_output, lifecycle_log])
        full_btn.click(_run_full_pipeline, None, [stage_output, video_preview, lifecycle_log])

        gr.Markdown("## Human Review")
        with gr.Row():
            approve_btn = gr.Button("APPROVE", variant="primary")
            regen_scene_idx = gr.Number(label="Scene index to regenerate", value=0)
            regen_scene_btn = gr.Button("REGENERATE SCENE")
            regen_voice_btn = gr.Button("REGENERATE VOICE")
            regen_script_btn = gr.Button("REGENERATE SCRIPT")
            reject_btn = gr.Button("REJECT", variant="stop")

        review_output = gr.Textbox(label="Review output", interactive=False)
        approve_btn.click(_approve, None, review_output)
        regen_scene_btn.click(_regenerate_scene, regen_scene_idx, [review_output, lifecycle_log])
        regen_voice_btn.click(_regenerate_voice, None, review_output)
        regen_script_btn.click(_regenerate_script, None, review_output)
        reject_btn.click(_reject, None, review_output)

    return demo


def launch_ui() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    launch_ui()
