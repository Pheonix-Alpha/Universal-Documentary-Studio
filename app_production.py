"""
Kaggle Dual-T4 Text-to-Video Generation Engine
================================================
Parallel CogVideoX-2b inference across cuda:0 and cuda:1.

Features:
  - Dual GPU rendering
  - Live heartbeat/status system
  - Idempotent model download
  - Idempotent pipeline preparation
  - Parallel GPU preparation
  - Gradio live status timer
  - Optional distributed queue
"""

import os

# ---------------------------------------------------------------
# Hugging Face cache
# ---------------------------------------------------------------

os.environ["HF_HOME"] = "/tmp/hf_cache"
os.makedirs("/tmp/hf_cache", exist_ok=True)

# ---------------------------------------------------------------
# Imports
# ---------------------------------------------------------------

import json
import threading
import time

import torch
import gradio as gr

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download


# ---------------------------------------------------------------
# Optional distributed queue
# ---------------------------------------------------------------

try:
    from distributed_queue import Queue
except ImportError:
    Queue = None


# ---------------------------------------------------------------
# Distributed configuration
# ---------------------------------------------------------------

SERVICE_ACCOUNT_JSON = None
DRIVE_ROOT_FOLDER_ID = None

_queue = None

if Queue and SERVICE_ACCOUNT_JSON and DRIVE_ROOT_FOLDER_ID:
    _queue = Queue(
        SERVICE_ACCOUNT_JSON,
        DRIVE_ROOT_FOLDER_ID,
    )

    print(
        "Distributed queue connected — "
        "renders will be handed off to Colab workers."
    )

else:
    print(
        "Running in local-only mode "
        "(no distributed queue configured)."
    )


# ===============================================================
# Configuration
# ===============================================================

MODEL_ID = "THUDM/CogVideoX-2b"

NUM_FRAMES = 49
NUM_STEPS = 50

GUIDANCE_SCALE = 6.5

FPS = 8

HEARTBEAT_INTERVAL = 2

STATUS_FILE = "/tmp/pipeline_status.json"


# ===============================================================
# Shared Status / Heartbeat
# ===============================================================

_status_lock = threading.Lock()

_status = {
    0: {
        "stage": "not_started",
        "progress": 0,
        "prompt": None,
        "last_heartbeat": None,
    },
    1: {
        "stage": "not_started",
        "progress": 0,
        "prompt": None,
        "last_heartbeat": None,
    },
}


def _status_snapshot():
    """
    Return a safe copy of the in-memory status.

    IMPORTANT:
    Do NOT use JSON round-tripping here because JSON converts
    integer dictionary keys into strings.
    """

    with _status_lock:
        return {
            0: dict(_status[0]),
            1: dict(_status[1]),
        }


def _write_status_file():
    """
    Persist status to disk.

    JSON files necessarily store dictionary keys as strings.
    Therefore read_status() must normalize them back to integers.
    """

    snapshot = _status_snapshot()

    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(snapshot, f)

    except OSError:
        pass


def update_status(gpu_id, **fields):
    """
    Update one GPU's status and heartbeat.
    """

    with _status_lock:

        _status[gpu_id].update(fields)

        _status[gpu_id]["last_heartbeat"] = time.time()

        snapshot = {
            0: dict(_status[0]),
            1: dict(_status[1]),
        }

    # -----------------------------------------------------------
    # Persist status
    # -----------------------------------------------------------

    try:

        with open(STATUS_FILE, "w") as f:
            json.dump(snapshot, f)

    except OSError:
        pass

    # -----------------------------------------------------------
    # Optional distributed heartbeat
    # -----------------------------------------------------------

    if _queue:

        try:

            _queue.report_heartbeat(
                node_id=f"kaggle-gpu{gpu_id}",
                role="kaggle-generator",
                stage=snapshot[gpu_id]["stage"],
                progress=snapshot[gpu_id]["progress"],
                extra={
                    "prompt": snapshot[gpu_id]["prompt"]
                },
            )

        except Exception as e:

            print(
                f"[queue heartbeat warning] {e}"
            )

    return snapshot


def read_status():
    """
    Read current status.

    IMPORTANT:
    Always return integer GPU keys.

    This prevents:

        KeyError: 0

    caused by JSON converting:

        0 -> "0"
        1 -> "1"
    """

    with _status_lock:

        return {
            0: dict(_status[0]),
            1: dict(_status[1]),
        }


def format_status_text():
    """
    Convert GPU status into readable UI text.
    """

    now = time.time()

    status = read_status()

    lines = []

    for gpu_id in (0, 1):

        s = status[gpu_id]

        heartbeat = s.get("last_heartbeat")

        if heartbeat is None:

            age_str = "never"

        else:

            age = now - heartbeat

            age_str = (
                f"{age:.1f}s ago"
            )

            if age > HEARTBEAT_INTERVAL * 4:

                age_str += "  [STALE]"

        prompt = s.get("prompt")

        prompt_str = (
            f' — "{prompt}"'
            if prompt
            else ""
        )

        lines.append(
            f"GPU {gpu_id}: "
            f"{s.get('stage', 'unknown')} "
            f"({s.get('progress', 0)}%) "
            f"— last heartbeat {age_str}"
            f"{prompt_str}"
        )

    # -----------------------------------------------------------
    # Distributed worker status
    # -----------------------------------------------------------

    if _queue:

        try:

            all_status = _queue.list_all_status()

            for node_id, worker_status in sorted(
                all_status.items()
            ):

                if node_id.startswith("kaggle-gpu"):

                    continue

                stale_flag = (
                    "  [STALE]"
                    if worker_status.get("stale")
                    else ""
                )

                lines.append(
                    f"{node_id} "
                    f"({worker_status.get('role')})"
                    f": "
                    f"{worker_status.get('stage')}"
                    f" "
                    f"({worker_status.get('progress')})"
                    f" — last heartbeat "
                    f"{worker_status.get('age_seconds')}s ago"
                    f"{stale_flag}"
                )

        except Exception as e:

            lines.append(
                f"[queue status unavailable: {e}]"
            )

    return "\n".join(lines)


# ===============================================================
# Background Heartbeat
# ===============================================================

def _heartbeat_loop(gpu_id, stop_event):
    """
    Keeps the heartbeat alive while a GPU worker is running.
    """

    while not stop_event.is_set():

        with _status_lock:

            _status[gpu_id]["last_heartbeat"] = time.time()

            snapshot = {
                0: dict(_status[0]),
                1: dict(_status[1]),
            }

        try:

            with open(STATUS_FILE, "w") as f:

                json.dump(
                    snapshot,
                    f,
                )

        except OSError:

            pass

        stop_event.wait(
            HEARTBEAT_INTERVAL
        )


# ===============================================================
# Pipeline Management
# ===============================================================

_pipelines = {}

_pipelines_lock = threading.Lock()


def _ensure_downloaded(gpu_id):
    """
    Check whether CogVideoX is already cached.

    If cached:
        no network request.

    If missing:
        download model.
    """

    update_status(
        gpu_id,
        stage="checking_cache",
        progress=0,
    )

    try:

        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir="/tmp/hf_cache",
            local_files_only=True,
        )

        update_status(
            gpu_id,
            stage="cache_hit",
            progress=0,
        )

        return

    except Exception:

        pass

    # -----------------------------------------------------------
    # Download
    # -----------------------------------------------------------

    update_status(
        gpu_id,
        stage="downloading_model",
        progress=0,
    )

    snapshot_download(
        repo_id=MODEL_ID,
        cache_dir="/tmp/hf_cache",
    )

    update_status(
        gpu_id,
        stage="downloaded",
        progress=0,
    )


def _prepare_pipeline(gpu_id):
    """
    Prepare CogVideoX pipeline for one GPU.

    Idempotent:
    if already loaded, return existing pipeline.
    """

    # -----------------------------------------------------------
    # Already loaded?
    # -----------------------------------------------------------

    with _pipelines_lock:

        if gpu_id in _pipelines:

            update_status(
                gpu_id,
                stage="idle",
                progress=0,
            )

            return _pipelines[gpu_id]

    # -----------------------------------------------------------
    # Ensure model exists
    # -----------------------------------------------------------

    _ensure_downloaded(gpu_id)

    # -----------------------------------------------------------
    # Load model
    # -----------------------------------------------------------

    update_status(
        gpu_id,
        stage="loading_into_vram",
        progress=0,
    )

    pipe = CogVideoXPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir="/tmp/hf_cache",
    )

    # -----------------------------------------------------------
    # GPU-specific CPU offload
    # -----------------------------------------------------------

    pipe.enable_model_cpu_offload(
        gpu_id=gpu_id
    )

    # -----------------------------------------------------------
    # VAE memory optimization
    # -----------------------------------------------------------

    pipe.vae.enable_slicing()

    pipe.vae.enable_tiling()

    # -----------------------------------------------------------
    # Store pipeline
    # -----------------------------------------------------------

    with _pipelines_lock:

        _pipelines[gpu_id] = pipe

    update_status(
        gpu_id,
        stage="idle",
        progress=0,
    )

    return pipe


def ensure_pipelines_ready():
    """
    Prepare both GPU pipelines in parallel.
    """

    print(
        "Preparing both pipelines "
        "(parallel download/load)..."
    )

    thread_gpu0 = threading.Thread(
        target=_prepare_pipeline,
        args=(0,),
        name="prepare-gpu0",
    )

    thread_gpu1 = threading.Thread(
        target=_prepare_pipeline,
        args=(1,),
        name="prepare-gpu1",
    )

    thread_gpu0.start()
    thread_gpu1.start()

    thread_gpu0.join()
    thread_gpu1.join()

    print(
        "Both pipelines ready."
    )


# ===============================================================
# Generation
# ===============================================================

def render_on_gpu(
    prompt,
    gpu_id,
    output_filename,
    result_dict,
):

    stop_event = threading.Event()

    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(
            gpu_id,
            stop_event,
        ),
        daemon=True,
        name=f"heartbeat-gpu{gpu_id}",
    )

    heartbeat_thread.start()

    try:

        # -------------------------------------------------------
        # Get pipeline
        # -------------------------------------------------------

        pipe = _prepare_pipeline(
            gpu_id
        )

        # -------------------------------------------------------
        # Start generation
        # -------------------------------------------------------

        update_status(
            gpu_id,
            stage="generating",
            progress=0,
            prompt=prompt,
        )

        # -------------------------------------------------------
        # Generation callback
        # -------------------------------------------------------

        def step_callback(
            pipeline,
            step,
            timestep,
            callback_kwargs,
        ):

            progress_value = int(
                (step / NUM_STEPS) * 100
            )

            update_status(
                gpu_id,
                stage="generating",
                progress=progress_value,
                prompt=prompt,
            )

            return callback_kwargs

        # -------------------------------------------------------
        # Run CogVideoX
        # -------------------------------------------------------

        result = pipe(
            prompt=prompt,
            num_inference_steps=NUM_STEPS,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            callback_on_step_end=step_callback,
        )

        frames = result.frames[0]

        # -------------------------------------------------------
        # Encode MP4
        # -------------------------------------------------------

        update_status(
            gpu_id,
            stage="encoding_video",
            progress=100,
        )

        output_path = (
            f"/kaggle/working/"
            f"{output_filename}.mp4"
        )

        export_to_video(
            frames,
            output_video_path=output_path,
            fps=FPS,
        )

        result_dict[
            f"{gpu_id}_file"
        ] = output_path

        # -------------------------------------------------------
        # Optional distributed hand-off
        # -------------------------------------------------------

        if _queue:

            update_status(
                gpu_id,
                stage="submitting_to_queue",
                progress=100,
            )

            job_id = _queue.submit_job(
                prompt=prompt,
                video_local_path=output_path,
            )

            result_dict[
                f"{gpu_id}_job_id"
            ] = job_id

        # -------------------------------------------------------
        # Complete
        # -------------------------------------------------------

        update_status(
            gpu_id,
            stage="done",
            progress=100,
        )

    except Exception as e:

        result_dict[
            f"{gpu_id}_error"
        ] = str(e)

        update_status(
            gpu_id,
            stage=f"error: {e}",
            progress=0,
        )

        print(
            f"[GPU {gpu_id} ERROR] {e}"
        )

    finally:

        stop_event.set()

        heartbeat_thread.join(
            timeout=HEARTBEAT_INTERVAL + 1
        )


# ===============================================================
# Orchestrator
# ===============================================================

def orchestrator(
    prompt_left,
    prompt_right,
    progress=gr.Progress(),
):

    result = {}

    # -----------------------------------------------------------
    # GPU 0
    # -----------------------------------------------------------

    thread_gpu0 = threading.Thread(
        target=render_on_gpu,
        args=(
            prompt_left,
            0,
            "raw_asset_gpu0",
            result,
        ),
        name="render-gpu0",
    )

    # -----------------------------------------------------------
    # GPU 1
    # -----------------------------------------------------------

    thread_gpu1 = threading.Thread(
        target=render_on_gpu,
        args=(
            prompt_right,
            1,
            "raw_asset_gpu1",
            result,
        ),
        name="render-gpu1",
    )

    # -----------------------------------------------------------
    # Start both
    # -----------------------------------------------------------

    thread_gpu0.start()
    thread_gpu1.start()

    # -----------------------------------------------------------
    # Monitor progress
    # -----------------------------------------------------------

    while (
        thread_gpu0.is_alive()
        or thread_gpu1.is_alive()
    ):

        status = read_status()

        avg_progress = (
            status[0]["progress"]
            + status[1]["progress"]
        ) / 200.0

        progress(
            avg_progress,
            desc=format_status_text().replace(
                "\n",
                " | ",
            ),
        )

        thread_gpu0.join(
            timeout=0.5
        )

        thread_gpu1.join(
            timeout=0.5
        )

    # -----------------------------------------------------------
    # Errors
    # -----------------------------------------------------------

    error_gpu0 = result.get(
        "0_error"
    )

    error_gpu1 = result.get(
        "1_error"
    )

    if error_gpu0:

        print(
            f"[GPU 0 ERROR] "
            f"{error_gpu0}"
        )

    if error_gpu1:

        print(
            f"[GPU 1 ERROR] "
            f"{error_gpu1}"
        )

    # -----------------------------------------------------------
    # Return videos
    # -----------------------------------------------------------

    return (
        result.get("0_file"),
        result.get("1_file"),
    )


# ===============================================================
# Gradio UI
# ===============================================================

with gr.Blocks(
    theme=gr.themes.Glass()
) as app:

    gr.Markdown(
        "# Dual-GPU Parallel Video Production Studio"
    )

    # -----------------------------------------------------------
    # Live status
    # -----------------------------------------------------------

    status_box = gr.Textbox(
        label="Live GPU Status",
        value=format_status_text(),
        lines=3,
    )

    # -----------------------------------------------------------
    # Modern Gradio timer
    # -----------------------------------------------------------

    status_timer = gr.Timer(
        value=HEARTBEAT_INTERVAL
    )

    status_timer.tick(
        fn=format_status_text,
        outputs=status_box,
    )

    # -----------------------------------------------------------
    # Prompt inputs
    # -----------------------------------------------------------

    with gr.Row():

        with gr.Column():

            p0_box = gr.Textbox(
                label="Prompt for GPU 0",
                value=(
                    "Vector sticker of an astronaut, "
                    "whiteboard background, "
                    "flat design"
                ),
            )

            p1_box = gr.Textbox(
                label="Prompt for GPU 1",
                value=(
                    "Vector sticker of a rocket ship, "
                    "whiteboard background, "
                    "flat design"
                ),
            )

            btn = gr.Button(
                "Run Dual Render Loop",
                variant="primary",
            )

        # -------------------------------------------------------
        # Output videos
        # -------------------------------------------------------

        with gr.Column():

            out0 = gr.Video(
                label="Output GPU 0"
            )

            out1 = gr.Video(
                label="Output GPU 1"
            )

    # -----------------------------------------------------------
    # Button action
    # -----------------------------------------------------------

    btn.click(
        fn=orchestrator,
        inputs=[
            p0_box,
            p1_box,
        ],
        outputs=[
            out0,
            out1,
        ],
    )


# ===============================================================
# Main
# ===============================================================

if __name__ == "__main__":

    print("")
    print(
        "================================================"
    )
    print(
        " Starting Universal Documentary Studio"
    )
    print(
        "================================================"
    )
    print("")

    # -----------------------------------------------------------
    # Prepare both GPUs before serving UI
    # -----------------------------------------------------------

    ensure_pipelines_ready()

    # -----------------------------------------------------------
    # Launch Gradio
    # -----------------------------------------------------------

    app.launch(
        share=True
    )