#!/bin/bash
# ================================================================
# One-time environment setup for colab_master.py
# Run this in a Colab shell cell as:  !bash setup_colab.sh
# ================================================================
set -e

pip install -q basicsr realesrgan opencv-python-headless torch torchvision

# --- Real-ESRGAN x4plus weights (skip if already downloaded) ---
if [ -f "RealESRGAN_x4plus.pth" ]; then
  echo "RealESRGAN_x4plus.pth already present — skipping download."
else
  echo "Downloading RealESRGAN_x4plus.pth..."
  wget -q -O RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
fi

# --- Practical-RIFE (HDv3) for true optical-flow interpolation ---
if [ -d "rife_repo" ]; then
  echo "rife_repo already cloned — skipping."
else
  echo "Cloning Practical-RIFE..."
  git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git rife_repo
fi

mkdir -p train_log
if [ -n "$(ls -A train_log 2>/dev/null)" ]; then
  echo "train_log already populated — skipping checkpoint copy."
else
  cp -r rife_repo/train_log/* train_log/ 2>/dev/null || true
  # NOTE: RIFE's pretrained weight hosting has moved a few times.
  # Check https://github.com/hzwer/Practical-RIFE#model-weights for the
  # current download link, then place the extracted checkpoint files
  # into ./train_log/ if the copy above didn't already do it.
  if [ -z "$(ls -A train_log 2>/dev/null)" ]; then
    echo "train_log/ is still empty — download the RIFE HDv3 checkpoint"
    echo "manually from the Practical-RIFE README and extract it there."
  fi
fi

echo "Setup complete. Verify RealESRGAN_x4plus.pth and train_log/ exist before running."
