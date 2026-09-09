"""
Colab Worker: HD Mastering Core (Real-ESRGAN x4 + RIFE interpolation)
======================================================================
This module is imported by colab_main.py -- it is not meant to be run
directly. Call load_models() once, then process_and_upscale_stream()
for each raw clip you want mastered.
"""

import os
import sys
import subprocess

# opencv/numpy build cleanly under normal pip build isolation.
PLAIN_PIP = ["opencv-python", "numpy<2"]
# basicsr/realesrgan's setup.py needs `torch` importable *during the build*,
# which pip's isolated build env does not provide -> egg_info fails unless
# we disable build isolation so they build against the already-installed
# (Colab-provided) torch instead.
TORCH_DEPENDENT_PIP = ["basicsr", "realesrgan"]


def ensure_requirements():
    marker = "/content/.worker_deps_installed"
    if os.path.exists(marker):
        print("[setup] Worker deps already installed, skipping.")
    else:
        print("[setup] Installing colab_worker dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "setuptools", "wheel"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PLAIN_PIP], check=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", *TORCH_DEPENDENT_PIP],
            check=True,
        )
        with open(marker, "w") as f:
            f.write("ok")
        print("[setup] Done.")

    # RIFE inference code + weights aren't on pip -- clone once, reused after.
    if not os.path.exists("/content/ECCV2022-RIFE"):
        print("[setup] Cloning RIFE (frame interpolation) repo...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/megvii-research/ECCV2022-RIFE",
             "/content/ECCV2022-RIFE"],
            check=True,
        )
    if "/content/ECCV2022-RIFE" not in sys.path:
        sys.path.insert(0, "/content/ECCV2022-RIFE")


ensure_requirements()

import types
import cv2
import torch
import numpy as np
import torchvision.transforms.functional as _tv_functional

# basicsr (an unmaintained dependency of realesrgan) imports
# torchvision.transforms.functional_tensor, which newer torchvision removed
# in favor of torchvision.transforms.functional. Inject a compatibility
# shim module so that import succeeds without patching basicsr's source.
if "torchvision.transforms.functional_tensor" not in sys.modules:
    _shim = types.ModuleType("torchvision.transforms.functional_tensor")
    _shim.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _shim

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

_MODELS = {"upsampler": None, "rife": None}


def load_models(esrgan_weights_path="RealESRGAN_x4plus.pth", rife_dir="/content/ECCV2022-RIFE/train_log"):
    """Load Real-ESRGAN + RIFE into VRAM exactly once and keep them resident."""
    if _MODELS["upsampler"] is not None:
        print("[models] Already loaded, skipping.")
        return

    if not os.path.exists(esrgan_weights_path):
        print("[models] Fetching RealESRGAN_x4plus.pth ...")
        subprocess.run(
            ["wget", "-q",
             "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
             "-O", esrgan_weights_path],
            check=True,
        )

    model_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    _MODELS["upsampler"] = RealESRGANer(
        scale=4, model_path=esrgan_weights_path, model=model_arch, tile=400, half=True
    )

    from train_log.RIFE_HDv3 import Model as RIFEModel
    rife_model = RIFEModel()
    rife_model.load_model(rife_dir, -1)
    rife_model.eval()
    rife_model.device()
    _MODELS["rife"] = rife_model
    print("[models] Real-ESRGAN + RIFE resident in VRAM.")


def process_and_upscale_stream(input_path, output_path, target_size=(1920, 1080), target_fps=24):
    """
    Streams frame-by-frame: pads 3:2 -> 16:9, upscales x4, interpolates
    8fps -> 24fps with RIFE, writes directly to disk (no RAM frame cache).
    """
    if _MODELS["upsampler"] is None or _MODELS["rife"] is None:
        raise RuntimeError("Call load_models() before process_and_upscale_stream().")

    upsampler = _MODELS["upsampler"]
    rife_model = _MODELS["rife"]

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Could not open input video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, target_size)

    prev_frame_upscaled = None
    success, frame = cap.read()
    print(f"[process] Streaming {input_path} -> {output_path} ...")

    while success:
        h, w, _ = frame.shape
        target_w = int(h * (16 / 9))
        pad_w = max((target_w - w) // 2, 0)
        padded_frame = cv2.copyMakeBorder(frame, 0, 0, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        rgb_img = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB)
        upscaled, _ = upsampler.enhance(rgb_img, outscale=4.0)
        current_frame_hd = cv2.resize(upscaled, target_size, interpolation=cv2.INTER_LANCZOS4)

        if prev_frame_upscaled is not None:
            t0 = torch.from_numpy(prev_frame_upscaled.transpose(2, 0, 1)).to("cuda").float().unsqueeze(0) / 255.0
            t1 = torch.from_numpy(current_frame_hd.transpose(2, 0, 1)).to("cuda").float().unsqueeze(0) / 255.0

            mid1 = rife_model.inference(t0, t1, timestep=0.33)
            mid2 = rife_model.inference(t0, t1, timestep=0.66)

            f1 = (mid1[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            f2 = (mid2[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

            out.write(cv2.cvtColor(prev_frame_upscaled, cv2.COLOR_RGB2BGR))
            out.write(cv2.cvtColor(f1, cv2.COLOR_RGB2BGR))
            out.write(cv2.cvtColor(f2, cv2.COLOR_RGB2BGR))

        prev_frame_upscaled = current_frame_hd
        success, frame = cap.read()

    if prev_frame_upscaled is not None:
        out.write(cv2.cvtColor(prev_frame_upscaled, cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()
    print(f"[process] Done -> {output_path}")