"""CLI entry point for Universal Documentary Studio.

Usage:
    python -m app.main --topic "How X Happened" [--ui] [--full]

Without --ui, runs the pipeline headlessly for one topic and prints a
summary. With --ui, launches the Gradio human-review dashboard instead.
"""
from __future__ import annotations

import argparse
import sys

from app.config import load_app_config
from app.startup import run_startup
from core.models import ProjectConfig, TTSConfig
from core.pipeline import DocumentaryPipeline
from core.project_manager import ProjectManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Universal Documentary Studio")
    parser.add_argument("--topic", type=str, help="Topic for the documentary.")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--duration", type=float, default=10.0, help="Target duration in minutes.")
    parser.add_argument("--shorts", type=int, default=4, help="Number of Shorts to produce (3-5).")
    parser.add_argument("--voice", type=str, default="default")
    parser.add_argument("--depth", type=str, default="standard", choices=["quick", "standard", "deep"])
    parser.add_argument("--ui", action="store_true", help="Launch the Gradio review dashboard instead.")
    parser.add_argument("--full", action="store_true", help="Run the full pipeline end-to-end for --topic.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app_config = load_app_config()
    run_startup(lightweight=True)

    if args.ui:
        from app.ui import launch_ui
        launch_ui()
        return 0

    if not args.topic:
        print("Provide --topic \"...\" (and optionally --full), or use --ui to launch the dashboard.")
        return 1

    config = ProjectConfig(
        topic=args.topic, language=args.language, target_duration_minutes=args.duration,
        short_count=args.shorts, voice=args.voice, research_depth=args.depth,
        mock_mode=app_config["mock_mode"], local_gpu_enabled=app_config["local_gpu_enabled"],
        tts=TTSConfig(**app_config["tts"]),
    )
    pm = ProjectManager(projects_root=app_config["projects_root"], config=config)
    video_cfg = app_config["video"]
    pipeline = DocumentaryPipeline(
        pm,
        video_width=video_cfg["long_form"]["width"], video_height=video_cfg["long_form"]["height"],
        fps=video_cfg["long_form"]["fps"],
        short_width=video_cfg["short"]["width"], short_height=video_cfg["short"]["height"],
        short_fps=video_cfg["short"]["fps"],
    )

    if args.full:
        result = pipeline.run_full()
        print(f"Project ID: {config.project_id}")
        print(f"State: {pm.state_machine.current_state.value}")
        print(f"Long-form render: {result.render_path}")
        print(f"Shorts: {result.short_render_paths}")
        print(f"Thumbnails: {result.thumbnail_paths}")
        print(f"QA score: {result.qa_report.score} ({result.qa_report.status.value})")
    else:
        research = pipeline.run_research()
        print(f"Project ID: {config.project_id}")
        print(f"Research complete: {len(research.sources)} sources, {len(research.claims)} claims.")
        print("Use --full to run the entire pipeline, or resume this project_id via --ui.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
