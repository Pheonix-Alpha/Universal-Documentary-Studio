"""
Colab Main: Orchestrator ("main" system)
=========================================
Run this on Colab (single T4, internet ON).

1. Installs its own requirements and loads Real-ESRGAN/RIFE into VRAM once.
2. Takes a Kaggle worker URL as input and heartbeats it before doing anything.
3. Takes ONE prompt from you, auto-splits it into a first-half / second-half
   pair, and sends both to the Kaggle worker to render concurrently
   (one segment per GPU).
4. Stitches the two returned segments into one continuous raw clip, then
   runs the local mastering pipeline (colab_worker.py) to produce a single
   1080p/24fps final video.

colab_worker.py must sit next to this file (same folder) so it can be imported.
Kaggle stays headless -- there is no prompt UI there, everything is driven
from here.
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
# 4. Single prompt -> two consecutive segment prompts
# ---------------------------------------------------------------------------
def build_two_part_prompts(user_prompt):
    """
    Splits one user prompt into a first-half / second-half pair so GPU0 and
    GPU1 can render both segments concurrently, which are then stitched into
    one longer clip.

    CogVideoX renders each half independently -- there's no frame-level
    conditioning carried from segment 1 into segment 2 -- so the stitch point
    is a hard cut, not a seamless continuous shot. Keeping both halves
    describing the same subject/setting (rather than two unrelated scenes)
    is what keeps that cut from looking jarring.
    """
    lowered = user_prompt.lower()
    for cue in (" then ", " and then ", "; "):
        if cue in lowered:
            idx = lowered.index(cue)
            part1 = user_prompt[:idx].strip().rstrip(",;")
            part2 = user_prompt[idx + len(cue):].strip()
            if part1 and part2:
                return part1, part2
    # No explicit sequencing cue in the prompt -- auto-wrap as two beats
    # of the same scene instead of guessing where to cut it.
    part1 = f"{user_prompt}, opening moment, establishing shot"
    part2 = f"{user_prompt}, continuing the same scene, a moment later"
    return part1, part2


# ---------------------------------------------------------------------------
# 5. Full pipeline: split -> generate on Kaggle (parallel) -> stitch -> master
# ---------------------------------------------------------------------------
def run_pipeline(kaggle_url, user_prompt, out_dir="/content/output"):
    os.makedirs(out_dir, exist_ok=True)
    client = connect_to_kaggle(kaggle_url)

    part1_prompt, part2_prompt = build_two_part_prompts(user_prompt)
    print(f"[pipeline] Segment 1 (GPU0): {part1_prompt}")
    print(f"[pipeline] Segment 2 (GPU1): {part2_prompt}")

    stop_event = threading.Event()
    hb_thread = threading.Thread(target=heartbeat_monitor, args=(client, stop_event), daemon=True)
    hb_thread.start()

    try:
        print("[pipeline] Rendering both segments concurrently on Kaggle...")
        raw_part1_path, raw_part2_path = client.predict(
            part1_prompt, part2_prompt, api_name="/generate"
        )
        print(f"[pipeline] Received segments:\n  {raw_part1_path}\n  {raw_part2_path}")
    finally:
        stop_event.set()
        hb_thread.join(timeout=2)

    stitched_path = os.path.join(out_dir, "raw_stitched.mp4")
    colab_worker.stitch_clips(raw_part1_path, raw_part2_path, stitched_path)

    final_path = os.path.join(out_dir, "final_1080p_master.mp4")
    colab_worker.process_and_upscale_stream(stitched_path, final_path)

    print(f"[pipeline] Complete. Final master:\n  {final_path}")
    return final_path


# ---------------------------------------------------------------------------
# 6. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    kaggle_url = input("Paste the Kaggle worker public URL (from kaggle_worker.py): ").strip()
    user_prompt = input("Describe the video you want: ").strip()

    run_pipeline(kaggle_url, user_prompt)
