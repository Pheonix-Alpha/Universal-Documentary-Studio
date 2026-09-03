"""
Gradio UI - Clean Minimal with Worker Management
"""

import gradio as gr
from app import pipeline, model_manager, worker_client, config
import base64
import tempfile
import time


def _b64_video_to_tempfile(b64_data: str):
    """Gradio's Video/Gallery components need a file path (or URL), not a
    raw base64 string -- passing the base64 string straight through (as the
    old code did) rendered as a broken player. Decode to a temp .mp4 and
    hand back the path instead."""
    if not b64_data:
        return None
    try:
        raw = base64.b64decode(b64_data)
    except Exception:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def build_app():
    """Clean minimal UI with worker management"""

    with gr.Blocks(
        title="Universal Documentary Studio", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("""
        # 🎬 Universal Documentary Studio
        ### Enter a story, get a video. Everything else is automatic.
        """)

        with gr.Tabs():
            # ============================================================
            # TAB 1: Generate Video (Main)
            # ============================================================
            with gr.TabItem("🎬 Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Story input
                        story_input = gr.Textbox(
                            label="📝 Your Story",
                            placeholder="Enter your story here... (minimum 50 characters)",
                            lines=15,
                            value="""In the year 3024, humanity had colonized the solar system. 
                            Dr. Elena Vance, a xenobiologist, discovered an ancient signal from 
                            Europa's subsurface ocean. The signal wasn't random - it was a message 
                            from an unknown intelligence.""",
                        )

                        with gr.Row():
                            # "Auto" lets model_selector.py pick the best model
                            # for the job + the connected worker's resources.
                            # Manual choices are still here as an override.
                            video_model_choices = [
                                ("🧠 Auto (recommended)", "auto"),
                                (
                                    "ZeroScope (Fast, 7GB)",
                                    "cerspense/zeroscope_v2_576w",
                                ),
                                (
                                    "Stable Video Diffusion (Best, 9.5GB)",
                                    "stabilityai/stable-video-diffusion-img2vid",
                                ),
                                (
                                    "Realistic Vision (Light, 4.8GB)",
                                    "SG161222/Realistic_Vision_V5.1_noVAE",
                                ),
                            ]

                            video_model = gr.Dropdown(
                                label="🎥 Video Model",
                                choices=video_model_choices,
                                value="auto",  # Default value must be in choices
                                allow_custom_value=False,  # Don't allow custom values
                            )

                        with gr.Row():
                            duration = gr.Slider(
                                2, 8, value=4, label="⏱️ Clip Duration (seconds)"
                            )
                            generate_btn = gr.Button(
                                "🎬 Generate Video", variant="primary", size="lg"
                            )

                    with gr.Column(scale=1):
                        status = gr.Label(value="💤 Ready", label="Status")
                        progress = gr.Slider(
                            0, 100, value=0, label="📈 Progress", interactive=False
                        )
                        log = gr.Textbox(
                            label="📋 LIVE TERMINAL",
                            lines=18,
                            max_lines=30,
                            interactive=False,
                            autoscroll=True,
                        )

                # Output section
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(
                            label="🎞️ Generated Clips", columns=3, height=250
                        )
                    with gr.Column():
                        final_video = gr.Video(label="🎬 Final Video")

                # Model status (collapsible)
                with gr.Accordion("⚙️ Model Status (Auto)", open=False):
                    model_status = gr.JSON(label="Current Status", value={})

            # ============================================================
            # TAB 2: Workers (Where you add GPU workers)
            # ============================================================
            with gr.TabItem("⚙️ Workers"):
                gr.Markdown("""
                ### 🌐 Connect GPU Workers
                
                Add GPU workers to generate videos faster.
                
                **How to add a worker:**
                1. Run `python worker.py` in a separate GPU Colab
                2. Copy the `https://*.trycloudflare.com` URL
                3. Paste it below and click "Add Worker"
                """)

                # Worker add section
                with gr.Row():
                    with gr.Column(scale=3):
                        worker_url_input = gr.Textbox(
                            label="🔗 Worker URL",
                            placeholder="https://worker-123.trycloudflare.com",
                            value="",
                        )
                    with gr.Column(scale=2):
                        worker_label_input = gr.Textbox(
                            label="🏷️ Label (optional)",
                            placeholder="GPU Worker 1",
                            value="",
                        )
                    with gr.Column(scale=1):
                        add_worker_btn = gr.Button(
                            "➕ Add Worker", variant="primary", size="lg"
                        )

                worker_action_status = gr.Textbox(
                    label="📋 Status", lines=2, interactive=False
                )

                gr.Markdown("---")
                gr.Markdown("### 📋 Connected Workers")

                # Worker list
                worker_status_html = gr.HTML(
                    value="<div style='text-align: center; padding: 20px; color: #888;'>No workers connected. Add one above.</div>"
                )

                with gr.Row():
                    refresh_workers_btn = gr.Button(
                        "🔄 Refresh Workers", variant="secondary"
                    )

                # Worker remove section
                with gr.Row():
                    worker_to_remove = gr.Dropdown(
                        label="🗑️ Worker to Remove",
                        choices=[],  # Will be populated dynamically
                        value=None,
                        allow_custom_value=True,  # Allow because choices are dynamic
                    )
                    remove_worker_btn = gr.Button("❌ Remove Worker", variant="stop")

                # ---- Worker Functions ----

                def get_worker_choices():
                    """Get list of workers for dropdown"""
                    try:
                        workers = worker_client.list_workers()
                        return [
                            (f"{w.get('label', 'Unknown')}", w.get("id", ""))
                            for w in workers
                            if w.get("id")
                        ]
                    except:
                        return []

                def render_workers_html():
                    """Render workers as HTML with status indicators"""
                    try:
                        workers = worker_client.list_workers()
                    except:
                        workers = []

                    if not workers:
                        return """
                        <div style='text-align: center; padding: 30px; color: #888; border: 1px dashed #ddd; border-radius: 8px;'>
                            <p>🚫 No workers connected</p>
                            <p style='font-size: 14px;'>Add a worker URL from a GPU Colab running worker.py</p>
                        </div>
                        """

                    html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"

                    for worker in workers:
                        if worker.get("connected", False):
                            status_color = "#4CAF50"
                            status_icon = "🟢"
                            status_text = "Connected"
                        else:
                            status_color = "#f44336"
                            status_icon = "🔴"
                            status_text = "Disconnected"

                        load = worker.get("load", 0)
                        load_color = (
                            "#4CAF50"
                            if load < 2
                            else "#FF9800" if load < 5 else "#f44336"
                        )

                        html += f"""
                        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa;'>
                            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;'>
                                <div>
                                    <strong>{worker.get('label', 'Unknown')}</strong>
                                    <span style='margin-left: 10px; color: {status_color};'>{status_icon} {status_text}</span>
                                    <span style='margin-left: 10px; font-size: 12px; color: #666;'>Load: <span style='color: {load_color};'>{load}</span></span>
                                </div>
                                <div style='font-size: 12px; color: #666;'>
                                    {worker.get('device', 'unknown')}
                                </div>
                            </div>
                            <div style='font-size: 12px; color: #888; word-break: break-all; margin-top: 4px;'>
                                {worker.get('url', '')}
                            </div>
                        </div>
                        """

                    html += "</div>"
                    return html

                def add_worker_handler(url: str, label: str):
                    """Add a worker and return updated UI"""
                    if not url or not url.strip():
                        return (
                            "❌ Please enter a worker URL",
                            render_workers_html(),
                            get_worker_choices(),
                        )

                    try:
                        worker_id = worker_client.add_worker(
                            url.strip(), label.strip() or None
                        )
                        return (
                            f"✅ Added worker: {label or url}",
                            render_workers_html(),
                            get_worker_choices(),
                        )
                    except Exception as e:
                        return (
                            f"❌ Failed to add worker: {e}",
                            render_workers_html(),
                            get_worker_choices(),
                        )

                def remove_worker_handler(worker_id: str):
                    """Remove a worker and return updated UI"""
                    if not worker_id:
                        return (
                            "⚠️ Please select a worker to remove",
                            render_workers_html(),
                            get_worker_choices(),
                        )

                    try:
                        worker_client.remove_worker(worker_id)
                        return (
                            f"✅ Removed worker: {worker_id}",
                            render_workers_html(),
                            get_worker_choices(),
                        )
                    except Exception as e:
                        return (
                            f"❌ Failed to remove worker: {e}",
                            render_workers_html(),
                            get_worker_choices(),
                        )

                def refresh_workers_handler():
                    """Refresh worker list"""
                    return render_workers_html(), get_worker_choices()

                # ---- Wire up Worker Events ----

                add_worker_btn.click(
                    fn=add_worker_handler,
                    inputs=[worker_url_input, worker_label_input],
                    outputs=[
                        worker_action_status,
                        worker_status_html,
                        worker_to_remove,
                    ],
                )

                remove_worker_btn.click(
                    fn=remove_worker_handler,
                    inputs=[worker_to_remove],
                    outputs=[
                        worker_action_status,
                        worker_status_html,
                        worker_to_remove,
                    ],
                )

                refresh_workers_btn.click(
                    fn=refresh_workers_handler,
                    outputs=[worker_status_html, worker_to_remove],
                )

                # Auto-refresh on load
                demo.load(
                    fn=refresh_workers_handler,
                    outputs=[worker_status_html, worker_to_remove],
                )

            # ============================================================
            # TAB 3: API Keys (Optional)
            # ============================================================
            with gr.TabItem("🔑 API Keys"):
                gr.Markdown("""
                ### 🔐 API Keys (Optional)
                Add API keys for better scene analysis and image search.
                """)

                # The previous version of this tab rendered raw <input>/
                # <button> HTML with onclick handlers that dispatched
                # `save_key`/`clear_key` CustomEvents -- but nothing in the
                # Python app ever listened for those events, so clicking
                # Save/Clear did literally nothing and a key could only ever
                # be set via an environment variable before launch. Rebuilt
                # with real Gradio components wired to config.set_key/
                # unset so they actually work.
                def _key_status_line(key_id: str) -> str:
                    is_set = config.is_key_set(key_id)
                    return "✅ Set" if is_set else "❌ Not set"

                key_status_labels = {}
                key_inputs = {}

                for spec in config.KEY_SPECS:
                    key_id = spec["id"]
                    with gr.Group():
                        gr.Markdown(
                            f"**{spec['label']}** — {spec['note']}  \n"
                            f"[Get a key →]({spec.get('signup_url', '#')})"
                        )
                        with gr.Row():
                            key_inputs[key_id] = gr.Textbox(
                                label=f"{spec['label']} API key",
                                type="password",
                                placeholder="Enter API key...",
                                scale=3,
                            )
                            key_status_labels[key_id] = gr.Textbox(
                                value=_key_status_line(key_id),
                                label="Status",
                                interactive=False,
                                scale=1,
                            )
                        with gr.Row():
                            save_btn = gr.Button(
                                f"💾 Save {spec['label']} key", size="sm"
                            )
                            clear_btn = gr.Button(
                                f"🗑️ Clear {spec['label']} key",
                                size="sm",
                                variant="stop",
                            )

                        def _make_save(kid):
                            def _save(value):
                                if value and value.strip():
                                    config.set_key(kid, value)
                                return "", _key_status_line(kid)

                            return _save

                        def _make_clear(kid):
                            def _clear():
                                config.set_key(kid, "")
                                return "", _key_status_line(kid)

                            return _clear

                        save_btn.click(
                            fn=_make_save(key_id),
                            inputs=[key_inputs[key_id]],
                            outputs=[key_inputs[key_id], key_status_labels[key_id]],
                        )
                        clear_btn.click(
                            fn=_make_clear(key_id),
                            inputs=[],
                            outputs=[key_inputs[key_id], key_status_labels[key_id]],
                        )

        # ---- Video Generation Handler ----
        def generate_video_handler(story, model_id, duration):
            """Generate video with automatic everything"""
            if not story or len(story.strip()) < 20:
                yield (
                    "❌ Error",
                    0,
                    "⚠️ Please enter a longer story (minimum 20 characters)",
                    [],
                    None,
                    {},
                )
                return

            # Keep track of temp files we hand to the gallery so old scenes
            # from a previous run don't leak forever, but a running gallery
            # still needs its earlier entries to stay valid across yields.
            gallery_paths_by_scene = {}
            terminal_lines = [
                "╔══════════════════════════════════════════════╗",
                "║        UNIVERSAL DOCUMENTARY STUDIO         ║",
                "║              LIVE PIPELINE                  ║",
                "╚══════════════════════════════════════════════╝",
                "",
            ]

            try:
                for update in pipeline.run_video_pipeline(
                    story=story,
                    model_id=model_id,
                    duration_per_scene=duration,
                    auto_download=True,
                ):
                    stage = update.get("stage", "processing")
                    pct = update.get("pct", 0)
                    log_text = update.get("log", "")

                    if log_text:

                        # Add every new pipeline event to the terminal history.
                        terminal_lines.append(log_text)

                        # Keep the terminal from growing forever.
                        terminal_lines = terminal_lines[-60:]

                    terminal_display = "\n\n".join(terminal_lines)
                    gallery_items = update.get("gallery", [])
                    video_data = update.get("video", None)
                    model_status_data = update.get("model_status", {})

                    # gallery_items is a list of clip dicts with base64
                    # 'video_data' -- decode each to a temp .mp4 path (once
                    # per scene_id) since Gradio can't render raw base64.
                    gallery_display = []
                    for item in gallery_items:
                        if not isinstance(item, dict):
                            continue
                        scene_id = item.get("scene_id")
                        if scene_id in gallery_paths_by_scene:
                            gallery_display.append(gallery_paths_by_scene[scene_id])
                            continue
                        path = _b64_video_to_tempfile(item.get("video_data"))
                        if path:
                            gallery_paths_by_scene[scene_id] = path
                            gallery_display.append(path)

                    yield (
                        stage.capitalize() if stage != "error" else "❌ Error",
                        pct,
                        terminal_display,
                        gallery_display,
                        _b64_video_to_tempfile(video_data) if video_data else None,
                        model_status_data,
                    )

            except Exception as e:
                yield ("❌ Error", 0, f"❌ {str(e)}", [], None, {})

        # ---- Wire up generate ----
        generate_btn.click(
            fn=generate_video_handler,
            inputs=[story_input, video_model, duration],
            outputs=[status, progress, log, gallery, final_video, model_status],
        )

    return demo


def launch_app():
    """Launch the app"""
    print("🚀 Launching Universal Documentary Studio...")
    demo = build_app()
    # NOTE: gr.Blocks.launch() has no `theme` kwarg -- theme is only set at
    # Blocks(...) construction time (see build_app() above). Passing it here
    # used to raise "TypeError: launch() got an unexpected keyword argument
    # 'theme'" the moment the app tried to start.
    demo.queue(max_size=20).launch(share=True, debug=False)


if __name__ == "__main__":
    launch_app()
