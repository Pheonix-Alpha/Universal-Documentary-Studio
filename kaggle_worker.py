"""
Kaggle Worker: Dual-GPU CogVideoX Raw Clip Generator
=====================================================
Run this on Kaggle (Dual T4 accelerator, internet ON).

Exposes a Gradio app with two API endpoints (auto-tunnelled via share=True):
  - /health   -> heartbeat probe, returns worker + GPU status
  - /generate -> runs the dual-GPU render job, returns two raw mp4 paths

After launch, copy the printed "Running on public URL" link and paste it
into colab_main.py when it asks for the Kaggle worker URL.
"""

import os
import sys
import subprocess
import threading
import time

# ---------------------------------------------------------------------------
# 1. Idempotent dependency install
# ---------------------------------------------------------------------------
REQUIRED = ["diffusers>=0.30.0", "transformers", "accelerate", "gradio>=4.44"]


def ensure_requirements():
    marker = "/kaggle/working/.deps_installed"
    if os.path.exists(marker):
        print("[setup] Dependencies already installed, skipping.")
        return
    print("[setup] Installing worker dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *REQUIRED], check=True)
    with open(marker, "w") as f:
        f.write("ok")
    print("[setup] Done.")


ensure_requirements()

import torch
import gradio as gr
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Redirect HuggingFace cache to unmetered OS ephemeral scratch space
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load two isolated pipelines, one per GPU (loaded once, stay resident)
# ---------------------------------------------------------------------------
print("[init] Loading independent GPU memory contexts...")
pipe_0 = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda:0")
pipe_1 = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda:1")
print("[init] Both pipelines resident in VRAM.")

_status = {0: 0, 1: 0, "busy": False}


def render_on_gpu(pipe, prompt, device_id, output_filename, status_dict):
    try:
        def step_callback(pipeline, step, timestep, callback_kwargs):
            status_dict[device_id] = int((step / 50) * 100)
            return callback_kwargs

        frames = pipe(
            prompt=prompt,
            num_inference_steps=50,
            num_frames=49,  # 6 seconds at 8 FPS
            guidance_scale=6.5,
            callback_on_step_end=step_callback,
        ).frames

        output_path = f"/kaggle/working/{output_filename}.mp4"
        export_to_video(frames, output_video_path=output_path, fps=8)
        status_dict[f"{device_id}_file"] = output_path
    except Exception as e:
        status_dict[f"{device_id}_error"] = str(e)


# ---------------------------------------------------------------------------
# 3. API-exposed functions
# ---------------------------------------------------------------------------
def health_check():
    """Heartbeat endpoint -- cheap, returns instantly."""
    return {
        "status": "ok",
        "busy": _status["busy"],
        "gpu0_mem_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        "gpu1_mem_gb": round(torch.cuda.memory_allocated(1) / 1e9, 2),
        "timestamp": time.time(),
    }


def generate(prompt_left, prompt_right, progress=gr.Progress()):
    """Runs both GPUs in parallel and returns the two raw clip paths."""
    _status.update({0: 0, 1: 0, "busy": True})
    status = _status

    t1 = threading.Thread(target=render_on_gpu, args=(pipe_0, prompt_left, 0, "raw_asset_gpu0", status))
    t2 = threading.Thread(target=render_on_gpu, args=(pipe_1, prompt_right, 1, "raw_asset_gpu1", status))
    t1.start()
    t2.start()

    while t1.is_alive() or t2.is_alive():
        avg = (status[0] + status[1]) / 200.0
        progress(avg, desc=f"GPU0 {status[0]}% | GPU1 {status[1]}%")
        t1.join(timeout=0.5)
        t2.join(timeout=0.5)

    status["busy"] = False
    if "0_error" in status or "1_error" in status:
        raise gr.Error(f"GPU0 error: {status.get('0_error')} | GPU1 error: {status.get('1_error')}")

    return status["0_file"], status["1_file"]


# ---------------------------------------------------------------------------
# 4. Gradio app -- the visible UI is optional; what matters is the API routes
# ---------------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Glass()) as app:
    gr.Markdown("# Kaggle Dual-GPU Worker")
    gr.Markdown("Copy the public URL Gradio prints below into `colab_main.py`.")

    with gr.Row():
        p0 = gr.Textbox(label="Prompt for GPU 0", value="Vector sticker of an astronaut, whiteboard background")
        p1 = gr.Textbox(label="Prompt for GPU 1", value="Vector sticker of a rocket ship, whiteboard background")
    btn = gr.Button("Run Dual Render (manual test)", variant="primary")
    out0 = gr.Video(label="Raw GPU0 clip")
    out1 = gr.Video(label="Raw GPU1 clip")
    btn.click(fn=generate, inputs=[p0, p1], outputs=[out0, out1], api_name="generate")

    # Hidden row purely to publish a /health API route for the Colab heartbeat
    with gr.Row(visible=False):
        health_btn = gr.Button("health")
        health_out = gr.JSON()
    health_btn.click(fn=health_check, outputs=health_out, api_name="health")

if __name__ == "__main__":
    app.launch(share=True)
