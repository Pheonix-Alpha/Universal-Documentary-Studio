"""
Google Colab HD Mastering Engine
=================================
Takes a raw 720x480 (3:2) CogVideoX clip at 8fps and produces a
1920x1080 (16:9) master at 24fps using:
  - Real-ESRGAN x4plus for spatial upscaling (with real pretrained weights)
  - RIFE (Practical-RIFE HDv3) for genuine optical-flow frame interpolation
  - Pillarbox padding to preserve aspect ratio (no stretching)
  - Frame-by-frame disk streaming (no full-clip RAM buffering)

Adds:
  - Idempotent setup: weight files are only fetched if missing, and the
    upsampler/RIFE model objects are only constructed once per process
    (safe to call process_and_upscale_stream multiple times in one
    session without re-downloading or re-initializing).
  - Heartbeat/status file: written every couple of seconds with
    {stage, frames_done, total_frames, last_heartbeat}. Colab runtimes
    can silently drop the browser connection while a cell keeps running
    server-side — reading MASTER_STATUS_FILE after reconnecting tells
    you immediately whether it's still progressing (fresh heartbeat,
    frame count climbing) or actually died (stale heartbeat).

Run setup_colab.sh once first to fetch weights (also idempotent).
"""

import json
import os
import time

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Practical-RIFE's HDv3 wrapper. Adjust the import path if your clone
# layout differs from the setup script.
from train_log.RIFE_HDv3 import Model as RIFEModel

TARGET_W, TARGET_H = 1920, 1080
OUTPUT_FPS = 24
RIFE_PAD_MULTIPLE = 32
HEARTBEAT_INTERVAL = 2  # seconds
MASTER_STATUS_FILE = "/tmp/master_status.json"
ESRGAN_WEIGHTS_PATH = "RealESRGAN_x4plus.pth"
RIFE_WEIGHTS_DIR = "./train_log"

_status = {
    "stage": "not_started",
    "frames_done": 0,
    "total_frames": None,
    "last_heartbeat": None,
}


def _write_status():
    _status["last_heartbeat"] = time.time()
    try:
        with open(MASTER_STATUS_FILE, "w") as f:
            json.dump(_status, f)
    except OSError:
        pass


def update_status(**fields):
    _status.update(fields)
    _write_status()


def check_master_status():
    """Call this after reconnecting to Colab to see whether a
    background run is still alive. A fresh last_heartbeat with a
    climbing frames_done means it's healthy; a stale one means it
    stopped (crash, or the runtime itself was recycled)."""
    if not os.path.exists(MASTER_STATUS_FILE):
        print("No status file yet — nothing has run in this session.")
        return None
    with open(MASTER_STATUS_FILE) as f:
        s = json.load(f)
    age = time.time() - s["last_heartbeat"] if s["last_heartbeat"] else None
    stale = age is not None and age > HEARTBEAT_INTERVAL * 4
    print(
        f"stage={s['stage']}  frames_done={s['frames_done']}/{s['total_frames']}  "
        f"last_heartbeat={age:.1f}s ago{'  [STALE]' if stale else ''}"
    )
    return s


def _pillarbox_to_16_9(frame):
    """Pad a 3:2 frame to exactly a 16:9 canvas (white bars left/right),
    with exact integer width so there's no rounding drift."""
    h, w = frame.shape[:2]
    target_w = round(h * (16 / 9))
    total_pad = max(target_w - w, 0)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left  # absorbs any odd remainder
    return cv2.copyMakeBorder(
        frame, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )


# ------------------------------------------------------------------
# Idempotent, one-time model construction (weights already fetched by
# setup_colab.sh; this just avoids rebuilding objects on repeat calls)
# ------------------------------------------------------------------
_upsampler = None
_rife_model = None


def _get_upsampler():
    global _upsampler
    if _upsampler is not None:
        return _upsampler
    if not os.path.exists(ESRGAN_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"{ESRGAN_WEIGHTS_PATH} not found — run setup_colab.sh first."
        )
    update_status(stage="loading_esrgan")
    model_arch = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
    )
    _upsampler = RealESRGANer(
        scale=4, model_path=ESRGAN_WEIGHTS_PATH, model=model_arch, tile=400, half=True
    )
    return _upsampler


def _get_rife():
    global _rife_model
    if _rife_model is not None:
        return _rife_model
    if not os.path.isdir(RIFE_WEIGHTS_DIR) or not os.listdir(RIFE_WEIGHTS_DIR):
        raise FileNotFoundError(
            f"{RIFE_WEIGHTS_DIR} is missing or empty — run setup_colab.sh and "
            "follow its RIFE checkpoint instructions first."
        )
    update_status(stage="loading_rife")
    model = RIFEModel()
    model.load_model(RIFE_WEIGHTS_DIR, -1)
    model.eval()
    model.device()
    _rife_model = model
    return _rife_model


def _rife_interpolate(rife_model, frame_a_rgb, frame_b_rgb, timestep):
    """Runs RIFE between two HD RGB uint8 frames at the given timestep
    (0-1) and returns a clipped uint8 RGB frame at the original size.
    Handles the 32-pixel-multiple requirement internally."""
    h, w = frame_a_rgb.shape[:2]
    pad_h = (RIFE_PAD_MULTIPLE - h % RIFE_PAD_MULTIPLE) % RIFE_PAD_MULTIPLE
    pad_w = (RIFE_PAD_MULTIPLE - w % RIFE_PAD_MULTIPLE) % RIFE_PAD_MULTIPLE

    t0 = torch.from_numpy(frame_a_rgb.transpose(2, 0, 1)).to("cuda").float() / 255.0
    t1 = torch.from_numpy(frame_b_rgb.transpose(2, 0, 1)).to("cuda").float() / 255.0
    t0, t1 = t0.unsqueeze(0), t1.unsqueeze(0)

    if pad_h or pad_w:
        t0 = F.pad(t0, (0, pad_w, 0, pad_h))
        t1 = F.pad(t1, (0, pad_w, 0, pad_h))

    with torch.no_grad():
        mid = rife_model.inference(t0, t1, timestep=timestep)

    mid = mid[:, :, :h, :w]  # crop back to true size
    mid_np = mid[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0
    mid_np = np.clip(mid_np, 0, 255).astype(np.uint8)
    return mid_np


def process_and_upscale_stream(input_path, output_path):
    upsampler = _get_upsampler()  # no-op if already built
    rife_model = _get_rife()      # no-op if already built

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, OUTPUT_FPS, (TARGET_W, TARGET_H))

    update_status(stage="processing", frames_done=0, total_frames=total_frames)
    last_heartbeat_write = time.time()

    prev_hd_rgb = None
    success, frame = cap.read()
    frame_count = 0

    print("Streaming frame generation loop...")
    while success:
        # 1. Pillarbox 3:2 -> 16:9 canvas before upscaling, so nothing
        #    gets horizontally squashed.
        padded = _pillarbox_to_16_9(frame)

        # 2. Real-ESRGAN spatial upscale, then resize onto the exact
        #    1920x1080 output grid (aspect is already correct so this
        #    resize is a clean scale, not a distortion).
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        upscaled, _ = upsampler.enhance(rgb, outscale=4.0)
        current_hd_rgb = cv2.resize(
            upscaled, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LANCZOS4
        )

        # 3. Interpolate 8fps -> 24fps with RIFE (two true in-between
        #    frames per source-frame gap), writing straight to disk.
        if prev_hd_rgb is not None:
            mid1 = _rife_interpolate(rife_model, prev_hd_rgb, current_hd_rgb, 1 / 3)
            mid2 = _rife_interpolate(rife_model, prev_hd_rgb, current_hd_rgb, 2 / 3)

            out.write(cv2.cvtColor(prev_hd_rgb, cv2.COLOR_RGB2BGR))
            out.write(cv2.cvtColor(mid1, cv2.COLOR_RGB2BGR))
            out.write(cv2.cvtColor(mid2, cv2.COLOR_RGB2BGR))

        prev_hd_rgb = current_hd_rgb
        frame_count += 1

        # Heartbeat: update at most every HEARTBEAT_INTERVAL seconds so
        # this doesn't add per-frame disk I/O overhead, but still gives
        # a live, reconnect-safe view of progress.
        if time.time() - last_heartbeat_write >= HEARTBEAT_INTERVAL:
            update_status(stage="processing", frames_done=frame_count)
            last_heartbeat_write = time.time()
            print(f"  processed {frame_count}/{total_frames or '?'} source frames...")

        success, frame = cap.read()

    # Flush the final source frame (no successor to interpolate toward).
    if prev_hd_rgb is not None:
        out.write(cv2.cvtColor(prev_hd_rgb, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    update_status(stage="done", frames_done=frame_count)
    print(f"Master processing complete! Output saved to: {output_path}")


if __name__ == "__main__":
    process_and_upscale_stream("raw_asset_gpu0.mp4", "final_1080p_master.mp4")
