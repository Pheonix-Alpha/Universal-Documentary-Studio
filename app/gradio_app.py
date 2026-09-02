"""
Gradio UI - Fixed video handling
"""

import gradio as gr
import base64
import tempfile
import os
from app import pipeline, model_manager, worker_client
import time


def build_app():
    """Clean minimal UI with proper video handling"""
    
    with gr.Blocks(title="Universal Documentary Studio", theme=gr.themes.Soft()) as demo:
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
                        story_input = gr.Textbox(
                            label="📝 Your Story",
                            placeholder="Enter your story here... (minimum 50 characters)",
                            lines=15,
                            value="""In the year 3024, humanity had colonized the solar system. 
                            Dr. Elena Vance, a xenobiologist, discovered an ancient signal from 
                            Europa's subsurface ocean. The signal wasn't random - it was a message 
                            from an unknown intelligence."""
                        )
                        
                        with gr.Row():
                            video_model_choices = [
                                ("ZeroScope (Fast, 7GB)", "cerspense/zeroscope_v2_576w"),
                                ("Stable Video Diffusion (Best, 9.5GB)", "stabilityai/stable-video-diffusion-img2vid"),
                                ("Realistic Vision (Light, 4.8GB)", "SG161222/Realistic_Vision_V5.1_noVAE")
                            ]
                            
                            video_model = gr.Dropdown(
                                label="🎥 Video Model",
                                choices=video_model_choices,
                                value="cerspense/zeroscope_v2_576w"
                            )
                        
                        with gr.Row():
                            duration = gr.Slider(2, 8, value=4, label="⏱️ Clip Duration (seconds)")
                            generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        status = gr.Label(value="💤 Ready", label="Status")
                        progress = gr.Slider(0, 100, value=0, label="📈 Progress", interactive=False)
                        log = gr.Textbox(label="📋 Log", lines=8, interactive=False)
                
                # Output section - using Video component with proper handling
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(label="🎞️ Generated Clips", columns=3, height=250)
                    with gr.Column():
                        final_video = gr.Video(label="🎬 Final Video", height=300)
                
                # Model status (collapsible)
                with gr.Accordion("⚙️ Model Status (Auto)", open=False):
                    model_status = gr.JSON(label="Current Status", value={})
            
            # ============================================================
            # TAB 2: Workers
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
                
                with gr.Row():
                    with gr.Column(scale=3):
                        worker_url_input = gr.Textbox(
                            label="🔗 Worker URL",
                            placeholder="https://worker-123.trycloudflare.com",
                            value=""
                        )
                    with gr.Column(scale=2):
                        worker_label_input = gr.Textbox(
                            label="🏷️ Label (optional)",
                            placeholder="GPU Worker 1",
                            value=""
                        )
                    with gr.Column(scale=1):
                        add_worker_btn = gr.Button("➕ Add Worker", variant="primary", size="lg")
                
                worker_action_status = gr.Textbox(label="📋 Status", lines=2, interactive=False)
                
                gr.Markdown("---")
                gr.Markdown("### 📋 Connected Workers")
                
                worker_status_html = gr.HTML(value="<div style='text-align: center; padding: 20px; color: #888;'>No workers connected. Add one above.</div>")
                
                with gr.Row():
                    refresh_workers_btn = gr.Button("🔄 Refresh Workers", variant="secondary")
                
                with gr.Row():
                    worker_to_remove = gr.Dropdown(
                        label="🗑️ Worker to Remove",
                        choices=[],
                        value=None,
                        allow_custom_value=True
                    )
                    remove_worker_btn = gr.Button("❌ Remove Worker", variant="stop")
                
                def get_worker_choices():
                    try:
                        workers = worker_client.list_workers()
                        return [(f"{w.get('label', 'Unknown')}", w.get('id', '')) for w in workers if w.get('id')]
                    except:
                        return []
                
                def render_workers_html():
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
                        if worker.get('connected', False):
                            status_color = "#4CAF50"
                            status_icon = "🟢"
                            status_text = "Connected"
                        else:
                            status_color = "#f44336"
                            status_icon = "🔴"
                            status_text = "Disconnected"
                        
                        load = worker.get('load', 0)
                        load_color = "#4CAF50" if load < 2 else "#FF9800" if load < 5 else "#f44336"
                        
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
                    if not url or not url.strip():
                        return "❌ Please enter a worker URL", render_workers_html(), get_worker_choices()
                    
                    try:
                        worker_id = worker_client.add_worker(url.strip(), label.strip() or None)
                        return f"✅ Added worker: {label or url}", render_workers_html(), get_worker_choices()
                    except Exception as e:
                        return f"❌ Failed to add worker: {e}", render_workers_html(), get_worker_choices()
                
                def remove_worker_handler(worker_id: str):
                    if not worker_id:
                        return "⚠️ Please select a worker to remove", render_workers_html(), get_worker_choices()
                    
                    try:
                        worker_client.remove_worker(worker_id)
                        return f"✅ Removed worker: {worker_id}", render_workers_html(), get_worker_choices()
                    except Exception as e:
                        return f"❌ Failed to remove worker: {e}", render_workers_html(), get_worker_choices()
                
                def refresh_workers_handler():
                    return render_workers_html(), get_worker_choices()
                
                add_worker_btn.click(
                    fn=add_worker_handler,
                    inputs=[worker_url_input, worker_label_input],
                    outputs=[worker_action_status, worker_status_html, worker_to_remove]
                )
                
                remove_worker_btn.click(
                    fn=remove_worker_handler,
                    inputs=[worker_to_remove],
                    outputs=[worker_action_status, worker_status_html, worker_to_remove]
                )
                
                refresh_workers_btn.click(
                    fn=refresh_workers_handler,
                    outputs=[worker_status_html, worker_to_remove]
                )
                
                demo.load(
                    fn=refresh_workers_handler,
                    outputs=[worker_status_html, worker_to_remove]
                )
            
            # ============================================================
            # TAB 3: API Keys
            # ============================================================
            with gr.TabItem("🔑 API Keys"):
                gr.Markdown("""
                ### 🔐 API Keys (Optional)
                Add API keys for better scene analysis and image search.
                """)
                
                api_status_html = gr.HTML(value="Loading...")
                refresh_keys_btn = gr.Button("🔄 Refresh Keys", variant="secondary")
                
                def render_keys_html():
                    from app import config
                    html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"
                    
                    for spec in config.KEY_SPECS:
                        key_id = spec['id']
                        is_set = config.is_key_set(key_id)
                        status_color = "#4CAF50" if is_set else "#f44336"
                        status_text = "✅ Set" if is_set else "❌ Not set"
                        
                        html += f"""
                        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fafafa;'>
                            <h4 style='margin: 0 0 4px 0;'>{spec['label']}</h4>
                            <p style='margin: 0 0 8px 0; font-size: 12px; color: #666;'>
                                Status: <span style='color: {status_color};'>{status_text}</span>
                            </p>
                            <div style='display: flex; gap: 8px; flex-wrap: wrap;'>
                                <input type='password' id='key_input_{key_id}' placeholder='Enter API key...' style='flex: 1; min-width: 150px; padding: 6px; border: 1px solid #ddd; border-radius: 4px;'>
                                <button onclick='saveKey("{key_id}")' style='background: #2196F3; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;'>
                                    💾 Save
                                </button>
                                <button onclick='clearKey("{key_id}")' style='background: #f44336; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer;'>
                                    🗑️ Clear
                                </button>
                            </div>
                            <p style='font-size: 11px; margin: 4px 0 0 0;'>
                                <a href='{spec.get('signup_url', '#')}' target='_blank' style='color: #2196F3;'>
                                    Get a key →
                                </a>
                            </p>
                        </div>
                        """
                    
                    html += """
                    <script>
                        function saveKey(keyId) {
                            const input = document.getElementById('key_input_' + keyId);
                            if (!input) return;
                            const value = input.value;
                            if (!value) { alert('Please enter a key'); return; }
                            const event = new CustomEvent('save_key', { detail: { keyId, value } });
                            document.dispatchEvent(event);
                        }
                        function clearKey(keyId) {
                            if (!confirm('Clear this API key?')) return;
                            const event = new CustomEvent('clear_key', { detail: { keyId } });
                            document.dispatchEvent(event);
                        }
                    </script>
                    """
                    
                    html += "</div>"
                    return html
                
                api_status_html.value = render_keys_html()
                refresh_keys_btn.click(
                    fn=render_keys_html,
                    outputs=[api_status_html]
                )
        
        # ---- Helper: Convert base64 to video file ----
        def base64_to_video_file(base64_data: str) -> str:
            """Convert base64 video data to a temporary file for Gradio"""
            if not base64_data:
                return None
            
            try:
                # Decode base64
                video_bytes = base64.b64decode(base64_data)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(video_bytes)
                    tmp_path = tmp.name
                
                return tmp_path
            except Exception as e:
                print(f"Error converting video: {e}")
                return None
        
        # ---- Video Generation Handler ----
        def generate_video_handler(story, model_id, duration):
            """Generate video with proper video handling"""
            if not story or len(story.strip()) < 20:
                yield (
                    "❌ Error",
                    0,
                    "⚠️ Please enter a longer story (minimum 20 characters)",
                    [],
                    None,
                    {}
                )
                return
            
            try:
                for update in pipeline.run_video_pipeline(
                    story=story,
                    model_id=model_id,
                    duration_per_scene=duration,
                    auto_download=True
                ):
                    stage = update.get('stage', 'processing')
                    pct = update.get('pct', 0)
                    log_text = update.get('log', '')
                    gallery_items = update.get('gallery', [])
                    video_data = update.get('video', None)
                    model_status_data = update.get('model_status', {})
                    
                    # Process gallery items - convert base64 to video files
                    gallery_display = []
                    for item in gallery_items:
                        if isinstance(item, dict):
                            if 'video_data' in item:
                                # Convert to video file for display
                                video_path = base64_to_video_file(item['video_data'])
                                if video_path:
                                    gallery_display.append(video_path)
                            elif 'image' in item:
                                gallery_display.append(item['image'])
                    
                    # Convert final video to file
                    final_video_path = None
                    if video_data:
                        final_video_path = base64_to_video_file(video_data)
                    
                    yield (
                        stage.capitalize() if stage != 'error' else "❌ Error",
                        pct,
                        log_text,
                        gallery_display if gallery_display else [],
                        final_video_path,
                        model_status_data
                    )
                    
            except Exception as e:
                import traceback
                yield (
                    "❌ Error",
                    0,
                    f"❌ {str(e)}",
                    [],
                    None,
                    {}
                )
        
        # ---- Wire up generate ----
        generate_btn.click(
            fn=generate_video_handler,
            inputs=[story_input, video_model, duration],
            outputs=[status, progress, log, gallery, final_video, model_status]
        )
    
    return demo


def launch_app():
    """Launch the app"""
    print("🚀 Launching Universal Documentary Studio...")
    demo = build_app()
    demo.queue(max_size=20).launch(share=True, debug=False, theme=gr.themes.Soft())


if __name__ == "__main__":
    launch_app()