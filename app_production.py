"""
Kaggle Dual-T4 Text-to-Video Generation Engine
================================================
Genuinely parallel CogVideoX-2b inference across cuda:0 and cuda:1,
with:

  - A heartbeat/status system: each GPU worker (and the download/load
    step before it) continuously writes {stage, progress, prompt,
    last_heartbeat} to a shared status file. If the Gradio tab
    disconnects and reconnects (or you just open the file directly),
    you can immediately see what each GPU was doing and how recently
    it last checked in — a stale `last_heartbeat` means that worker
    has stalled or crashed, not just "still busy".

  - Idempotent, parallel model preparation: on startup, both GPUs
    check their local HF cache first. Only a GPU with a missing model
    downloads it; only a pipeline not yet constructed gets built and
    moved into VRAM. Both checks run in parallel threads so two cold
    starts overlap instead of stacking sequentially, and re-running
    setup mid-session is a no-op if everything is already in place.
"""

import os

# Must happen before any diffusers/transformers/torch-hub import that
# might trigger a download, or the redirect won't take effect.
os.environ["HF_HOME"] = "/tmp/hf_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)

import json
import threading
import time

import torch
import gradio as gr
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

MODEL_ID = "THUDM/CogVideoX-2b"
NUM_FRAMES = 49  # ~6s at 8fps
NUM_STEPS = 50
GUIDANCE_SCALE = 6.5
FPS = 8
HEARTBEAT_INTERVAL = 2  # seconds
STATUS_FILE = "/tmp/pipeline_status.json"

# ------------------------------------------------------------------
# Shared status / heartbeat layer
# ------------------------------------------------------------------
_status_lock = threading.Lock()
_status = {
    0: {"stage": "not_started", "progress": 0, "prompt": None, "last_heartbeat": None},
    1: {"stage": "not_started", "progress": 0, "prompt": None, "last_heartbeat": None},
}


def update_status(gpu_id, **fields):
    """Merge fields into this GPU's status, stamp the heartbeat, and
    persist to disk so any process/tab reading STATUS_FILE sees the
    latest known state even after a disconnect/reconnect."""
    with _status_lock:
        _status[gpu_id].update(fields)
        _status[gpu_id]["last_heartbeat"] = time.time()
        snapshot = json.loads(json.dumps(_status))  # cheap deep copy
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(snapshot, f)
    except OSError:
        pass  # disk hiccup shouldn't kill the worker
    return snapshot


def read_status():
    with _status_lock:
        return {
            0: dict(_status[0]),
            1: dict(_status[1]),
        }


def format_status_text():
    now = time.time()
    lines = []
    for gpu_id in (0, 1):
        s = read_status()[gpu_id]
        hb = s["last_heartbeat"]
        if hb is None:
            age_str = "never"
        else:
            age = now - hb
            age_str = f"{age:.1f}s ago" + (
                "  [STALE]" if age > HEARTBEAT_INTERVAL * 4 else ""
            )
        prompt_str = f' — "{s["prompt"]}"' if s.get("prompt") else ""
        lines.append(
            f"GPU {gpu_id}: {s['stage']} ({s['progress']}%) — last heartbeat {age_str}{prompt_str}"
        )
    return "\n".join(lines)


def _heartbeat_loop(gpu_id, stop_event):
    """Keeps last_heartbeat fresh between explicit progress updates,
    so a hung worker (stuck inside a library call with no callback
    firing) is visibly distinguishable from a busy one: the stage text
    stops changing AND the heartbeat age starts climbing."""
    while not stop_event.is_set():
        with _status_lock:
            _status[gpu_id]["last_heartbeat"] = time.time()
        try:
            with open(STATUS_FILE, "w") as f:
                json.dump(read_status(), f)
        except OSError:
            pass
        stop_event.wait(HEARTBEAT_INTERVAL)


# ------------------------------------------------------------------
# Idempotent, parallel model download + VRAM load
# ------------------------------------------------------------------
_pipelines = {}
_pipelines_lock = threading.Lock()


def _ensure_downloaded(gpu_id):
    """Skip the network entirely if the repo is already fully cached
    locally; only hit the hub if something's missing."""
    update_status(gpu_id, stage="checking_cache")
    try:
        snapshot_download(
            repo_id=MODEL_ID, cache_dir="/tmp/hf_cache", local_files_only=True
        )
        update_status(gpu_id, stage="cache_hit")
        return
    except Exception:
        pass  # not fully cached yet — fall through to a real download

    update_status(gpu_id, stage="downloading_model")
    snapshot_download(repo_id=MODEL_ID, cache_dir="/tmp/hf_cache")
    update_status(gpu_id, stage="downloaded")


def _prepare_pipeline(gpu_id):
    """Idempotent: if this GPU's pipeline is already built, return it
    immediately without touching disk or VRAM again."""
    with _pipelines_lock:
        if gpu_id in _pipelines:
            update_status(gpu_id, stage="idle")
            return _pipelines[gpu_id]

    _ensure_downloaded(gpu_id)

    update_status(gpu_id, stage="loading_into_vram")
    pipe = CogVideoXPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, cache_dir="/tmp/hf_cache"
    )
    # Per-instance offload bound to a specific device — true dual-GPU
    # isolation without either pipe fighting for the other's memory.
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    with _pipelines_lock:
        _pipelines[gpu_id] = pipe

    update_status(gpu_id, stage="idle", progress=0)
    return pipe


def ensure_pipelines_ready():
    """Runs both GPUs' download+load in parallel threads. Call once at
    startup; safe to call again later since each side is idempotent."""
    print("Preparing both pipelines (parallel download/load)...")
    threads = [
        threading.Thread(target=_prepare_pipeline, args=(0,)),
        threading.Thread(target=_prepare_pipeline, args=(1,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("Both pipelines ready.")


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------
def render_on_gpu(prompt, gpu_id, output_filename, result_dict):
    stop_event = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop, args=(gpu_id, stop_event), daemon=True
    )
    hb_thread.start()

    try:
        pipe = _prepare_pipeline(gpu_id)  # no-op if already loaded
        update_status(gpu_id, stage="generating", progress=0, prompt=prompt)

        def step_callback(pipeline, step, timestep, callback_kwargs):
            update_status(
                gpu_id, stage="generating", progress=int((step / NUM_STEPS) * 100)
            )
            return callback_kwargs

        result = pipe(
            prompt=prompt,
            num_inference_steps=NUM_STEPS,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            callback_on_step_end=step_callback,
        )
        frames = result.frames[0]

        update_status(gpu_id, stage="encoding_video", progress=100)
        output_path = f"/kaggle/working/{output_filename}.mp4"
        export_to_video(frames, output_video_path=output_path, fps=FPS)

        result_dict[f"{gpu_id}_file"] = output_path
        update_status(gpu_id, stage="done", progress=100)
    except Exception as e:
        result_dict[f"{gpu_id}_error"] = str(e)
        update_status(gpu_id, stage=f"error: {e}", progress=0)
    finally:
        stop_event.set()
        hb_thread.join(timeout=HEARTBEAT_INTERVAL + 1)


def orchestrator(prompt_left, prompt_right, progress=gr.Progress()):
    result = {}

    t1 = threading.Thread(
        target=render_on_gpu, args=(prompt_left, 0, "raw_asset_gpu0", result)
    )
    t2 = threading.Thread(
        target=render_on_gpu, args=(prompt_right, 1, "raw_asset_gpu1", result)
    )
    t1.start()
    t2.start()

    while t1.is_alive() or t2.is_alive():
        s = read_status()
        avg = (s[0]["progress"] + s[1]["progress"]) / 200.0
        progress(avg, desc=format_status_text().replace("\n", " | "))
        t1.join(timeout=0.5)
        t2.join(timeout=0.5)

    err0, err1 = result.get("0_error"), result.get("1_error")
    if err0:
        print(f"[GPU 0 ERROR] {err0}")
    if err1:
        print(f"[GPU 1 ERROR] {err1}")

    return result.get("0_file"), result.get("1_file")


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
with gr.Blocks(theme=gr.themes.Glass()) as app:
    gr.Markdown("# Dual-GPU Parallel Video Production Studio")

    status_box = gr.Textbox(
        label="Live GPU Status",
        value=format_status_text(),
        lines=3,
    )

    status_timer = gr.Timer(value=HEARTBEAT_INTERVAL)

    status_timer.tick(
        fn=format_status_text,
        outputs=status_box,
    )

    with gr.Row():
        with gr.Column():
            p0_box = gr.Textbox(
                label="Prompt for GPU 0",
                value="Vector sticker of an astronaut, whiteboard background, flat design",
            )
            p1_box = gr.Textbox(
                label="Prompt for GPU 1",
                value="Vector sticker of a rocket ship, whiteboard background, flat design",
            )
            btn = gr.Button("Run Dual Render Loop", variant="primary")
        with gr.Column():
            out0 = gr.Video(label="Output GPU 0")
            out1 = gr.Video(label="Output GPU 1")

    btn.click(fn=orchestrator, inputs=[p0_box, p1_box], outputs=[out0, out1])

if __name__ == "__main__":
    ensure_pipelines_ready()  # parallel download+load before serving any request
    app.launch(share=True)
