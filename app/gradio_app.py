"""
Gradio UI - Minimal & Clean
Only: Story Input → Generate → Output
Models auto-download when needed
"""

import gradio as gr
from app import pipeline, model_manager, video_models, worker_client
import time


def build_app():
    """Clean minimal UI with automatic model management"""
    
    with gr.Blocks(title="Universal Documentary Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 Universal Documentary Studio
        ### Enter a story, get a video. Everything else is automatic.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Story input - the only thing user needs to provide
                story_input = gr.Textbox(
                    label="📝 Your Story",
                    placeholder="Enter your story here... (minimum 50 characters)",
                    lines=20,
                    value="""In the year 3024, humanity had colonized the solar system. 
                    Dr. Elena Vance, a xenobiologist, discovered an ancient signal from 
                    Europa's subsurface ocean. The signal wasn't random - it was a message 
                    from an unknown intelligence."""
                )
                
                with gr.Row():
                    # Simple model selection - with auto-download
                    video_model = gr.Dropdown(
                        label="🎥 Video Model",
                        choices=[
                            ("ZeroScope (Fast, 7GB)", "cerspense/zeroscope_v2_576w"),
                            ("Stable Video Diffusion (Best, 9.5GB)", "stabilityai/stable-video-diffusion-img2vid"),
                            ("Realistic Vision (4.8GB)", "SG161222/Realistic_Vision_V5.1_noVAE")
                        ],
                        value="cerspense/zeroscope_v2_576w"
                    )
                
                with gr.Row():
                    duration = gr.Slider(2, 8, value=4, label="⏱️ Clip Duration (seconds)")
                    generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Status display
                status = gr.Label(value="💤 Ready", label="Status")
                progress = gr.Slider(0, 100, value=0, label="📈 Progress", interactive=False)
                log = gr.Textbox(label="📋 Log", lines=10, interactive=False)
        
        # Output section
        with gr.Row():
            with gr.Column():
                gallery = gr.Gallery(label="🎞️ Generated Clips", columns=3, height=300)
            with gr.Column():
                final_video = gr.Video(label="🎬 Final Video")
        
        # Hidden: Model status (optional expandable)
        with gr.Accordion("⚙️ Model Management (Auto)", open=False):
            gr.Markdown("""
            ### Models are automatically downloaded when needed and cleaned up when done.
            - ✅ Downloads models to the worker when first used
            - ✅ Automatically deletes old models to free space
            - ✅ Manages VRAM automatically
            """)
            model_status = gr.JSON(label="Model Status", value={})
        
        # ---- Video Generation Handler ----
        def generate_video_handler(story, model_id, duration):
            """Generate video with automatic model management"""
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
                # Run pipeline with auto-download
                for update in pipeline.run_video_pipeline(
                    story=story,
                    model_id=model_id,
                    duration_per_scene=duration,
                    auto_download=True  # NEW: auto-download models
                ):
                    stage = update.get('stage', 'processing')
                    pct = update.get('pct', 0)
                    log_text = update.get('log', '')
                    gallery_items = update.get('gallery', [])
                    video_data = update.get('video', None)
                    model_status_data = update.get('model_status', {})
                    
                    # Format gallery
                    gallery_display = []
                    for item in gallery_items:
                        if isinstance(item, dict):
                            if 'video_data' in item:
                                gallery_display.append(item['video_data'])
                            elif 'image' in item:
                                gallery_display.append(item['image'])
                    
                    yield (
                        stage.capitalize(),
                        pct,
                        log_text,
                        gallery_display if gallery_display else [],
                        video_data,
                        model_status_data
                    )
                    
            except Exception as e:
                import traceback
                yield (
                    "Error",
                    0,
                    f"❌ {str(e)}\n{traceback.format_exc()}",
                    [],
                    None,
                    {}
                )
        
        # Wire up
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