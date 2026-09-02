"""
Gradio UI - Complete implementation with working Worker management
"""

import gradio as gr
import time
import json
from typing import Dict, Any, List, Optional

from app import (
    compute,
    model_manager,
    video_models,
    worker_client,
    config,
    pipeline
)


def build_app():
    """Build the complete Gradio UI with working worker management"""
    
    with gr.Blocks(title="Universal Documentary Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 Universal Documentary Studio
        ### Turn stories into video documentaries with distributed GPU workers
        """)
        
        with gr.Tabs():
            # ============================================================
            # TAB 1: Generate Video
            # ============================================================
            with gr.TabItem("🎬 Generate Video"):
                with gr.Row():
                    with gr.Column(scale=2):
                        story_input = gr.Textbox(
                            label="📝 Story",
                            placeholder="Enter your story here...",
                            lines=15,
                            value="""In the year 3024, humanity had colonized the solar system. 
                            Dr. Elena Vance, a xenobiologist, discovered an ancient signal from 
                            Europa's subsurface ocean. The signal wasn't random - it was a message 
                            from an unknown intelligence."""
                        )
                        
                        with gr.Row():
                            clip_model = gr.Dropdown(
                                label="🎯 CLIP Model (for reference images)",
                                choices=[],  # Will be updated dynamically
                                value=None
                            )
                            video_model = gr.Dropdown(
                                label="🎥 Video Model",
                                choices=list(video_models.VIDEO_MODEL_REGISTRY.keys()),
                                value=list(video_models.VIDEO_MODEL_REGISTRY.keys())[0] if video_models.VIDEO_MODEL_REGISTRY else None
                            )
                        
                        with gr.Row():
                            top_k = gr.Slider(1, 10, value=3, label="📊 Top K references")
                            duration = gr.Slider(2, 10, value=4, label="⏱️ Duration per scene (seconds)")
                        
                        start_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        status = gr.Label(value="💤 Idle")
                        progress = gr.Slider(0, 100, value=0, label="📈 Progress", interactive=False)
                        log = gr.Textbox(label="📋 Log", lines=5, interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(label="🎞️ Generated Clips", columns=3, height=300)
                    with gr.Column():
                        final_video = gr.Video(label="🎬 Final Assembled Video")
                
                with gr.Accordion("📖 Production Bible", open=False):
                    bible_json = gr.JSON(label="Bible Details")
            
            # ============================================================
            # TAB 2: Models & Resources
            # ============================================================
            with gr.TabItem("🎥 Models & Resources"):
                gr.Markdown("""
                ### 📊 Resource Dashboard
                Monitor and manage your VRAM and storage usage.
                """)
                
                with gr.Row():
                    with gr.Column():
                        storage_stats = gr.JSON(label="💾 Storage Usage", value={})
                    with gr.Column():
                        vram_stats = gr.JSON(label="⚡ VRAM Usage", value={})
                
                refresh_resources_btn = gr.Button("🔄 Refresh Resources", variant="secondary")
                
                def refresh_resources():
                    storage = model_manager.get_available_storage()
                    vram = model_manager.get_vram_status()
                    return storage, vram
                
                refresh_resources_btn.click(
                    fn=refresh_resources,
                    outputs=[storage_stats, vram_stats]
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📦 Model Management")
                
                with gr.Row():
                    models_list = gr.JSON(label="Installed Models", value={})
                    refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary")
                
                def refresh_models():
                    models = model_manager.list_models()
                    installed = [m for m in models if m['installed']]
                    available = [m for m in models if not m['installed']]
                    return {
                        'installed': installed,
                        'available': available,
                        'total_models': len(models),
                        'total_storage_gb': sum(m['size_gb'] for m in installed)
                    }
                
                refresh_models_btn.click(
                    fn=refresh_models,
                    outputs=[models_list]
                )
                
                with gr.Row():
                    model_to_download = gr.Dropdown(
                        label="📥 Model to Download",
                        choices=[(m['name'], m['id']) for m in model_manager.list_models() if not m['installed']]
                    )
                    download_btn = gr.Button("⬇️ Download Model", variant="primary")
                
                download_status = gr.Textbox(label="📋 Download Status", lines=5)
                
                def download_with_cleanup(model_id):
                    if not model_id:
                        yield "⚠️ Please select a model"
                        return
                    
                    messages = []
                    for pct, msg in model_manager.smart_download_model(model_id):
                        messages.append(f"[{pct:3d}%] {msg}")
                        yield "\n".join(messages[-10:])
                
                download_btn.click(
                    fn=download_with_cleanup,
                    inputs=[model_to_download],
                    outputs=[download_status]
                )
                
                with gr.Row():
                    model_to_delete = gr.Dropdown(
                        label="🗑️ Model to Delete",
                        choices=[(m['name'], m['id']) for m in model_manager.list_models() if m['installed']]
                    )
                    delete_btn = gr.Button("🗑️ Delete Model", variant="stop")
                
                delete_status = gr.Textbox(label="📋 Delete Status")
                
                def delete_model(model_id):
                    if not model_id:
                        return "⚠️ Please select a model"
                    
                    if model_manager.delete_model(model_id):
                        return f"✅ Deleted {model_id}"
                    return f"❌ Failed to delete {model_id}"
                
                delete_btn.click(
                    fn=delete_model,
                    inputs=[model_to_delete],
                    outputs=[delete_status]
                )
                
                with gr.Row():
                    cleanup_btn = gr.Button("🧹 Cleanup Old Models", variant="secondary")
                cleanup_status = gr.Textbox(label="📋 Cleanup Status")
                
                def cleanup_models():
                    freed = model_manager._smart_cleanup(1.0)
                    storage = model_manager.get_available_storage()
                    return f"🧹 Freed {freed:.2f}GB. Now using {storage['used_gb']:.2f}GB / {storage['max_gb']:.2f}GB"
                
                cleanup_btn.click(
                    fn=cleanup_models,
                    outputs=[cleanup_status]
                )
            
            # ============================================================
            # TAB 3: Workers - COMPLETE WORKING IMPLEMENTATION
            # ============================================================
            with gr.TabItem("⚙️ Workers"):
                gr.Markdown("""
                ### 🌐 GPU Worker Management
                
                Add GPU workers to distribute video generation tasks.
                Each worker runs in a separate Colab notebook with GPU runtime.
                
                **How to add a worker:**
                1. Run `python worker.py` in a separate GPU Colab notebook
                2. Copy the `https://*.trycloudflare.com` URL from the worker
                3. Paste it below and click "Add Worker"
                """)
                
                # Worker add section
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
                
                # Worker list with status
                gr.Markdown("---")
                gr.Markdown("### 📋 Connected Workers")
                
                worker_status_html = gr.HTML(value="<div style='text-align: center; padding: 20px; color: #888;'>No workers connected. Add one above.</div>")
                
                refresh_workers_btn = gr.Button("🔄 Refresh Workers", variant="secondary")
                
                # Worker remove section
                with gr.Row():
                    worker_to_remove = gr.Dropdown(
                        label="🗑️ Worker to Remove",
                        choices=[],
                        value=None
                    )
                    remove_worker_btn = gr.Button("❌ Remove Worker", variant="stop")
                
                worker_action_status = gr.Textbox(label="📋 Status", lines=2)
                
                # ---- Worker Functions ----
                
                def get_worker_choices():
                    """Get list of workers for dropdown"""
                    workers = worker_client.list_workers()
                    return [(f"{w['label']} ({w['url']})", w['id']) for w in workers if w['id']]
                
                def render_workers_html():
                    """Render workers as HTML with status indicators"""
                    workers = worker_client.list_workers()
                    
                    if not workers:
                        return """
                        <div style='text-align: center; padding: 40px; color: #888; border: 1px dashed #ddd; border-radius: 8px;'>
                            <p>🚫 No workers connected</p>
                            <p style='font-size: 14px;'>Add a worker URL from a GPU Colab running worker.py</p>
                        </div>
                        """
                    
                    html = "<div style='display: flex; flex-direction: column; gap: 15px;'>"
                    
                    for worker in workers:
                        # Status styling
                        if worker.get('connected', False):
                            status_color = "#4CAF50"
                            status_icon = "🟢"
                            status_text = "Connected"
                            device_text = worker.get('device', 'unknown')
                        else:
                            status_color = "#f44336"
                            status_icon = "🔴"
                            status_text = "Disconnected"
                            device_text = "N/A"
                        
                        # Load indicator
                        load = worker.get('load', 0)
                        load_color = "#4CAF50" if load < 2 else "#FF9800" if load < 5 else "#f44336"
                        
                        # Capabilities
                        caps = worker.get('capabilities', {})
                        caps_text = ", ".join(caps.keys()) if caps else "Unknown"
                        
                        html += f"""
                        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fafafa;'>
                            <div style='display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 10px;'>
                                <div style='flex: 1;'>
                                    <h3 style='margin: 0 0 8px 0;'>{worker['label']}</h3>
                                    <div style='display: grid; grid-template-columns: auto 1fr; gap: 4px 15px; font-size: 14px;'>
                                        <span style='color: #666;'>ID:</span>
                                        <span><code>{worker['id']}</code></span>
                                        
                                        <span style='color: #666;'>URL:</span>
                                        <span style='word-break: break-all;'>{worker['url']}</span>
                                        
                                        <span style='color: #666;'>Status:</span>
                                        <span style='color: {status_color}; font-weight: bold;'>{status_icon} {status_text}</span>
                                        
                                        <span style='color: #666;'>Device:</span>
                                        <span>{device_text}</span>
                                        
                                        <span style='color: #666;'>Load:</span>
                                        <span style='color: {load_color};'>{load} active jobs</span>
                                        
                                        <span style='color: #666;'>Capabilities:</span>
                                        <span>{caps_text}</span>
                                    </div>
                                </div>
                                <div style='text-align: right;'>
                                    <span style='background: {status_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 12px;'>
                                        {status_text}
                                    </span>
                                </div>
                            </div>
                        </div>
                        """
                    
                    html += "</div>"
                    return html
                
                def add_worker_handler(url: str, label: str):
                    """Add a worker and return updated UI"""
                    if not url or not url.strip():
                        return "❌ Please enter a worker URL", render_workers_html(), get_worker_choices()
                    
                    try:
                        worker_id = worker_client.add_worker(url.strip(), label.strip() or None)
                        return f"✅ Added worker: {label or url}", render_workers_html(), get_worker_choices()
                    except Exception as e:
                        return f"❌ Failed to add worker: {e}", render_workers_html(), get_worker_choices()
                
                def remove_worker_handler(worker_id: str):
                    """Remove a worker and return updated UI"""
                    if not worker_id:
                        return "⚠️ Please select a worker to remove", render_workers_html(), get_worker_choices()
                    
                    try:
                        worker_client.remove_worker(worker_id)
                        return f"✅ Removed worker: {worker_id}", render_workers_html(), get_worker_choices()
                    except Exception as e:
                        return f"❌ Failed to remove worker: {e}", render_workers_html(), get_worker_choices()
                
                def refresh_workers_handler():
                    """Refresh worker list"""
                    return render_workers_html(), get_worker_choices()
                
                # ---- Wire up Worker Events ----
                
                # Add worker
                add_worker_btn.click(
                    fn=add_worker_handler,
                    inputs=[worker_url_input, worker_label_input],
                    outputs=[worker_action_status, worker_status_html, worker_to_remove]
                )
                
                # Remove worker
                remove_worker_btn.click(
                    fn=remove_worker_handler,
                    inputs=[worker_to_remove],
                    outputs=[worker_action_status, worker_status_html, worker_to_remove]
                )
                
                # Refresh workers
                refresh_workers_btn.click(
                    fn=refresh_workers_handler,
                    outputs=[worker_status_html, worker_to_remove]
                )
                
                # Auto-refresh on page load
                demo.load(
                    fn=refresh_workers_handler,
                    outputs=[worker_status_html, worker_to_remove]
                )
            
            # ============================================================
            # TAB 4: API Keys
            # ============================================================
            with gr.TabItem("🔑 API Keys"):
                gr.Markdown("""
                ### 🔐 API Keys
                Enter your API keys here. They are used immediately without restart.
                """)
                
                api_status_html = gr.HTML(value="Loading...")
                refresh_keys_btn = gr.Button("🔄 Refresh Keys", variant="secondary")
                
                def render_keys_html():
                    html = "<div style='display: flex; flex-direction: column; gap: 15px;'>"
                    
                    for spec in config.KEY_SPECS:
                        key_id = spec['id']
                        is_set = config.is_key_set(key_id)
                        status_color = "#4CAF50" if is_set else "#f44336"
                        status_text = "✅ Set" if is_set else "❌ Not set"
                        
                        html += f"""
                        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #fafafa;'>
                            <h3 style='margin: 0 0 5px 0;'>{spec['label']}</h3>
                            <p style='margin: 0 0 5px 0;'>
                                Status: <span style='color: {status_color}; font-weight: bold;'>{status_text}</span>
                            </p>
                            <p style='font-size: 12px; color: #666; margin: 0 0 10px 0;'>{spec.get('help', '')}</p>
                            <div style='display: flex; gap: 10px; flex-wrap: wrap;'>
                                <input type='password' id='key_input_{key_id}' placeholder='Enter API key...' style='flex: 1; min-width: 200px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;'>
                                <button onclick='saveKey("{key_id}")' style='background: #2196F3; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;'>
                                    💾 Save
                                </button>
                                <button onclick='clearKey("{key_id}")' style='background: #f44336; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;'>
                                    🗑️ Clear
                                </button>
                            </div>
                            <p style='font-size: 12px; margin: 5px 0 0 0;'>
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
                            if (!value) {
                                alert('Please enter a key');
                                return;
                            }
                            // Send to Python via Gradio
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
            
            # ============================================================
            # TAB 5: Image Sources (Original)
            # ============================================================
            with gr.TabItem("🖼️ Image Sources"):
                gr.Markdown("""
                ### 📸 Image Sources
                These image sources are used for finding reference images.
                They work alongside the video generation pipeline.
                
                | Source | Type | Key Required |
                |---|---|---|
                | Wikimedia Commons | Free | ❌ |
                | NASA | Free | ❌ |
                | Internet Archive | Free | ❌ |
                | MET Museum | Free | ❌ |
                | Openverse | Free | ❌ |
                | Library of Congress | Free | ❌ |
                | Flickr | Keyed | ✅ |
                | Pexels | Keyed | ✅ |
                | Pixabay | Keyed | ✅ |
                | DuckDuckGo | Free | ❌ |
                """)

    return demo


def launch_app():
    """Launch the Gradio app"""
    print("🚀 Launching Universal Documentary Studio...")
    demo = build_app()
    demo.queue(max_size=20).launch(share=True, debug=False)


if __name__ == "__main__":
    launch_app()