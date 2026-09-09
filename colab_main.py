"""
Colab Main: Orchestrator ("main" system)
=========================================
Run this on Colab (single T4, internet ON).

1. Installs its own requirements and loads Real-ESRGAN/RIFE into VRAM once.
2. Takes a Kaggle worker URL as input and heartbeats it before doing anything.
3. Sends prompts to the Kaggle worker, auto-downloads the two raw clips it returns.
4. Runs the local mastering pipeline (colab_worker.py) on each clip.

colab_worker.py must sit next to this file (same folder) so it can be imported.
"""

import os
import sys
import time
import threading
import subprocess

# ---------------------------------------------------------------------------
# 1. Idempotent dependency install for the orchestrator itself
# ---------------------------------------------------------------------------
def ensure_requirements():
    marker = "/content/.main_deps_installed"
    if os.path.exists(marker):
        print("[setup] Main deps already installed, skipping.")
        return
    print("[setup] Installing orchestrator dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio_client"], check=True)
    with open(marker, "w") as f:
        f.write("ok")


ensure_requirements()

from gradio_client import Client
import colab_worker  # local mastering pipeline (Real-ESRGAN + RIFE)

# ---------------------------------------------------------------------------
# 2. Load local models into VRAM once, up front (before we even need them)
# ---------------------------------------------------------------------------
colab_worker.load_models()

# ---------------------------------------------------------------------------
# 3. Connect to the Kaggle worker + heartbeat
# ---------------------------------------------------------------------------
def connect_to_kaggle(url, retries=6, delay=5):
    """Blocks until the Kaggle worker answers /health, or raises after retries."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            client = Client(url)
            result = client.predict(api_name="/health")
            print(f"[heartbeat] Kaggle worker OK (attempt {attempt}): {result}")
            return client
        except Exception as e:
            last_err = e
            print(f"[heartbeat] Attempt {attempt}/{retries} failed: {e}")
            time.sleep(delay)
    raise ConnectionError(f"Could not reach Kaggle worker at {url}: {last_err}")


def heartbeat_monitor(client, stop_event, interval=15):
    """Background thread: pings /health every `interval` seconds while a job runs."""
    while not stop_event.is_set():
        try:
            result = client.predict(api_name="/health")
            print(
                f"[heartbeat] alive - busy={result.get('busy')} "
                f"gpu0={result.get('gpu0_mem_gb')}GB gpu1={result.get('gpu1_mem_gb')}GB"
            )
        except Exception as e:
            print(f"[heartbeat] WARNING - missed beat: {e}")
        stop_event.wait(interval)


# ---------------------------------------------------------------------------
# 4. Full pipeline: generate on Kaggle -> download -> master locally
# ---------------------------------------------------------------------------
def run_pipeline(kaggle_url, prompt_left, prompt_right, out_dir="/content/output"):
    os.makedirs(out_dir, exist_ok=True)
    client = connect_to_kaggle(kaggle_url)

    stop_event = threading.Event()
    hb_thread = threading.Thread(target=heartbeat_monitor, args=(client, stop_event), daemon=True)
    hb_thread.start()

    try:
        print("[pipeline] Requesting dual-GPU render from Kaggle...")
        # gradio_client downloads the returned video files locally and hands back their paths
        raw_gpu0_path, raw_gpu1_path = client.predict(
            prompt_left, prompt_right, api_name="/generate"
        )
        print(f"[pipeline] Received raw clips:\n  {raw_gpu0_path}\n  {raw_gpu1_path}")
    finally:
        stop_event.set()
        hb_thread.join(timeout=2)

    outputs = []
    for i, raw_path in enumerate([raw_gpu0_path, raw_gpu1_path]):
        final_path = os.path.join(out_dir, f"final_1080p_master_{i}.mp4")
        colab_worker.process_and_upscale_stream(raw_path, final_path)
        outputs.append(final_path)

    print("[pipeline] Complete. Final masters:\n  " + "\n  ".join(outputs))
    return outputs


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kaggle_url = input("Paste the Kaggle worker public URL (from kaggle_worker.py): ").strip()
    prompt_left = input("Prompt for GPU 0 [default astronaut]: ").strip() or \
        "Vector sticker of an astronaut, whiteboard background, flat design"
    prompt_right = input("Prompt for GPU 1 [default rocket]: ").strip() or \
        "Vector sticker of a rocket ship, whiteboard background, flat design"

    run_pipeline(kaggle_url, prompt_left, prompt_right)
