"""
Gradio UI - Enhanced with resource management dashboard
"""

import gradio as gr
from typing import Dict, Any
import json
import time

from app import (
    model_manager,
    video_models,
    worker_client,
    config,
    pipeline
)


def build_app():
    with gr.Blocks(title="Universal Documentary Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 Universal Documentary Studio
        ### Turn stories into video documentaries
        """)
        
        with gr.Tabs():
            # ---- Tab 1: Generate Video ----
            with gr.TabItem("🎬 Generate Video"):
                # ... existing generation UI ...
                pass
            
            # ---- Tab 2: Model Management with Resource Dashboard ----
            with gr.TabItem("🎥 Models & Resources"):
                gr.Markdown("""
                ### 📊 Resource Dashboard
                Monitor and manage your VRAM and storage usage.
                """)
                
                # Resource stats
                with gr.Row():
                    with gr.Column():
                        storage_stats = gr.JSON(label="💾 Storage Usage", value={})
                    with gr.Column():
                        vram_stats = gr.JSON(label="⚡ VRAM Usage", value={})
                
                refresh_resources_btn = gr.Button("🔄 Refresh Resources", variant="secondary")
                
                def refresh_resources():
                    """Refresh storage and VRAM stats"""
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
                    # Model list
                    models_list = gr.JSON(label="Installed Models", value={})
                    refresh_models_btn = gr.Button("🔄 Refresh Models", variant="secondary")
                
                def refresh_models():
                    """Refresh the model list"""
                    models = model_manager.list_models()
                    installed = [m for m in models if m['installed']]
                    available = [m for m in models if not m['installed']]
                    return {
                        'installed': installed,
                        'available': available,
                        'total_models': len(models),
                        'total_storage': sum(m['size_gb'] for m in installed)
                    }
                
                refresh_models_btn.click(
                    fn=refresh_models,
                    outputs=[models_list]
                )
                
                # Model download/delete controls
                with gr.Row():
                    model_to_download = gr.Dropdown(
                        label="Model to Download",
                        choices=[(m['name'], m['id']) for m in model_manager.list_models() if not m['installed']]
                    )
                    download_btn = gr.Button("⬇️ Download Model", variant="primary")
                
                download_status = gr.Textbox(label="Download Status", lines=5)
                
                def download_with_cleanup(model_id):
                    """Download model with automatic cleanup"""
                    if not model_id:
                        return "Please select a model"
                    
                    messages = []
                    for pct, msg in model_manager.smart_download_model(model_id):
                        messages.append(f"[{pct:3d}%] {msg}")
                        yield "\n".join(messages[-10:])  # Show last 10 messages
                
                download_btn.click(
                    fn=download_with_cleanup,
                    inputs=[model_to_download],
                    outputs=[download_status]
                )
                
                with gr.Row():
                    model_to_delete = gr.Dropdown(
                        label="Model to Delete",
                        choices=[(m['name'], m['id']) for m in model_manager.list_models() if m['installed']]
                    )
                    delete_btn = gr.Button("🗑️ Delete Model", variant="stop")
                
                delete_status = gr.Textbox(label="Delete Status")
                
                def delete_model(model_id):
                    """Delete a model"""
                    if not model_id:
                        return "Please select a model"
                    
                    if model_manager.delete_model(model_id):
                        return f"✅ Deleted {model_id}"
                    return f"❌ Failed to delete {model_id}"
                
                delete_btn.click(
                    fn=delete_model,
                    inputs=[model_to_delete],
                    outputs=[delete_status]
                )
                
                # Cleanup button
                with gr.Row():
                    cleanup_btn = gr.Button("🧹 Cleanup Old Models", variant="secondary")
                cleanup_status = gr.Textbox(label="Cleanup Status")
                
                def cleanup_models():
                    """Force cleanup of old models"""
                    freed = model_manager._smart_cleanup(1.0)
                    storage = model_manager.get_available_storage()
                    return f"🧹 Freed {freed:.2f}GB. Now using {storage['used_gb']:.2f}GB / {storage['max_gb']:.2f}GB"
                
                cleanup_btn.click(
                    fn=cleanup_models,
                    outputs=[cleanup_status]
                )
            
            # ---- Tab 3: Workers ----
            with gr.TabItem("⚙️ Workers"):
                # ... existing worker UI ...
                pass
            
            # ---- Tab 4: API Keys ----
            with gr.TabItem("🔑 API Keys"):
                # ... existing API keys UI ...
                pass
    
    return demo

def launch_app():
    """Launch the Gradio app"""
    import gradio as gr
    demo = build_app()
    demo.queue(max_size=20).launch(share=True, debug=False)


if __name__ == "__main__":
    launch_app()