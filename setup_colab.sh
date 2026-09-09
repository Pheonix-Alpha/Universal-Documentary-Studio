#!/bin/bash

# ================================================================
# Universal Documentary Studio
# ================================================================
# COLAB MASTERING ENVIRONMENT
#
# Pipeline:
#
#   Kaggle raw video
#          ↓
#        RIFE
#          ↓
#   Frame interpolation
#          ↓
#    Real-ESRGAN
#          ↓
#      Upscaling
#          ↓
#       FFmpeg
#          ↓
#   Final 1080p master
#
# Target:
#   Google Colab
#   NVIDIA T4
#   Python 3.13
#
# IMPORTANT:
#   - Do NOT reinstall torch
#   - Do NOT pip install BasicSR
#   - Do NOT pip install Real-ESRGAN
#   - Use source repositories directly
# ================================================================

set -e


# ================================================================
# CONFIGURATION
# ================================================================

REALESRGAN_REPO="https://github.com/xinntao/Real-ESRGAN.git"
BASICSR_REPO="https://github.com/XPixelGroup/BasicSR.git"
RIFE_REPO="https://github.com/hzwer/Practical-RIFE.git"

REALESRGAN_DIR="realesrgan_repo"
BASICSR_DIR="basicsr_repo"
RIFE_DIR="rife_repo"

REALESRGAN_MODEL="RealESRGAN_x4plus.pth"

RIFE_CHECKPOINT_DIR="train_log"

STATUS_DIR="master_status"


# ================================================================
# HEADER
# ================================================================

echo ""
echo "================================================"
echo " Universal Documentary Studio"
echo " Colab Mastering Setup"
echo "================================================"
echo ""


# ================================================================
# 1. VERIFY PYTHON / PYTORCH / CUDA
# ================================================================

echo "[1/8] Verifying Python / PyTorch / CUDA..."

python - <<'PY'

import sys
import torch

print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not detected. "
        "Switch Colab runtime to NVIDIA T4."
    )

print(
    "GPU    :",
    torch.cuda.get_device_name(0)
)

print(
    "VRAM   :",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2
    ),
    "GB"
)

PY

echo ""
echo "GPU environment OK."


# ================================================================
# 2. INSTALL ONLY RUNTIME DEPENDENCIES
# ================================================================

echo ""
echo "[2/8] Installing mastering runtime dependencies..."

pip install -q \
    opencv-python-headless \
    scipy \
    tqdm \
    imageio \
    imageio-ffmpeg \
    pillow \
    google-api-python-client \
    google-auth \
    google-auth-httplib2

echo "Runtime dependencies installed."


# ================================================================
# 3. PREPARE BASICSR SOURCE
# ================================================================

echo ""
echo "[3/8] Preparing BasicSR source..."

#
# DO NOT:
#
#     pip install basicsr
#
# BasicSR's legacy setup.py causes the Python 3.13
# metadata-generation failure.
#
# Instead we clone the source and import it directly.
#

if [ -d "$BASICSR_DIR/.git" ]; then

    echo "BasicSR repository already exists."
    echo "Skipping clone."

else

    if [ -d "$BASICSR_DIR" ]; then
        rm -rf "$BASICSR_DIR"
    fi

    echo "Cloning BasicSR source..."

    git clone \
        --depth 1 \
        "$BASICSR_REPO" \
        "$BASICSR_DIR"

fi

echo "BasicSR source ready."


# ================================================================
# 4. PREPARE REAL-ESRGAN SOURCE
# ================================================================

echo ""
echo "[4/8] Preparing Real-ESRGAN source..."

#
# DO NOT pip install Real-ESRGAN.
#
# We use the repository source directly.
#

if [ -d "$REALESRGAN_DIR/.git" ]; then

    echo "Real-ESRGAN repository already exists."
    echo "Skipping clone."

else

    if [ -d "$REALESRGAN_DIR" ]; then
        rm -rf "$REALESRGAN_DIR"
    fi

    echo "Cloning Real-ESRGAN source..."

    git clone \
        --depth 1 \
        "$REALESRGAN_REPO" \
        "$REALESRGAN_DIR"

fi

echo "Real-ESRGAN source ready."


# ================================================================
# ADD SOURCE REPOSITORIES TO PYTHON PATH
# ================================================================

echo ""
echo "Configuring Python source paths..."

export PYTHONPATH="$(pwd)/$BASICSR_DIR:$(pwd)/$REALESRGAN_DIR:${PYTHONPATH:-}"

echo "PYTHONPATH configured."


# ================================================================
# TEST BASICSR SOURCE
# ================================================================

echo ""
echo "Testing BasicSR source import..."

python - <<'PY'

import sys

print("Python path:")
for p in sys.path:
    print("  ", p)

import basicsr

print("")
print("BasicSR source import: OK")
print("BasicSR path:", basicsr.__file__)

PY


# ================================================================
# TEST REAL-ESRGAN SOURCE
# ================================================================

echo ""
echo "Testing Real-ESRGAN source import..."

python - <<'PY'

import realesrgan

print("Real-ESRGAN source import: OK")
print("Real-ESRGAN path:", realesrgan.__file__)

PY


# ================================================================
# 5. REAL-ESRGAN MODEL
# ================================================================

echo ""
echo "[5/8] Checking Real-ESRGAN x4plus model..."

if [ -s "$REALESRGAN_MODEL" ]; then

    echo "$REALESRGAN_MODEL already exists."

    ls -lh "$REALESRGAN_MODEL"

else

    echo "Downloading Real-ESRGAN x4plus model..."

    wget -q \
        --show-progress \
        -O "$REALESRGAN_MODEL" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    if [ ! -s "$REALESRGAN_MODEL" ]; then

        echo ""
        echo "ERROR: Real-ESRGAN model download failed."

        exit 1

    fi

    echo ""
    echo "Real-ESRGAN model downloaded."

    ls -lh "$REALESRGAN_MODEL"

fi


# ================================================================
# 6. PRACTICAL-RIFE
# ================================================================

echo ""
echo "[6/8] Preparing Practical-RIFE..."

if [ -d "$RIFE_DIR/.git" ]; then

    echo "Practical-RIFE already exists."
    echo "Skipping clone."

else

    if [ -d "$RIFE_DIR" ]; then
        rm -rf "$RIFE_DIR"
    fi

    echo "Cloning Practical-RIFE..."

    git clone \
        --depth 1 \
        "$RIFE_REPO" \
        "$RIFE_DIR"

fi

echo "Practical-RIFE repository ready."


# ================================================================
# RIFE CHECKPOINT
# ================================================================

echo ""
echo "Checking RIFE checkpoint..."

mkdir -p "$RIFE_CHECKPOINT_DIR"

if [ -n "$(find "$RIFE_CHECKPOINT_DIR" -type f -print -quit 2>/dev/null)" ]; then

    echo "RIFE checkpoint already exists."

elif [ -d "$RIFE_DIR/train_log" ] && \
     [ -n "$(find "$RIFE_DIR/train_log" -type f -print -quit 2>/dev/null)" ]; then

    echo "Found checkpoint inside RIFE repository."

    cp -r \
        "$RIFE_DIR/train_log/." \
        "$RIFE_CHECKPOINT_DIR/"

else

    echo ""
    echo "WARNING:"
    echo "RIFE pretrained checkpoint was NOT found."
    echo ""
    echo "Repository exists:"
    echo "    $RIFE_DIR/"
    echo ""
    echo "Checkpoint directory:"
    echo "    $RIFE_CHECKPOINT_DIR/"
    echo ""

fi


# ================================================================
# 7. FFMPEG
# ================================================================

echo ""
echo "[7/8] Checking FFmpeg..."

if command -v ffmpeg >/dev/null 2>&1; then

    echo "FFmpeg already installed."

else

    echo "Installing FFmpeg..."

    apt-get update -qq
    apt-get install -y -qq ffmpeg

fi

echo ""
ffmpeg -version | head -1


# ================================================================
# 8. FINAL VERIFICATION
# ================================================================

echo ""
echo "================================================"
echo " FINAL ENVIRONMENT VERIFICATION"
echo "================================================"
echo ""


# ------------------------------------------------
# GPU
# ------------------------------------------------

python - <<'PY'

import torch

print("GPU      :", torch.cuda.get_device_name(0))
print("CUDA     :", torch.version.cuda)
print("PyTorch  :", torch.__version__)

PY


# ------------------------------------------------
# OpenCV
# ------------------------------------------------

python - <<'PY'

import cv2

print("OpenCV   :", cv2.__version__)

PY


# ------------------------------------------------
# BasicSR
# ------------------------------------------------

python - <<'PY'

import basicsr

print("BasicSR  : OK")
print("Path     :", basicsr.__file__)

PY


# ------------------------------------------------
# Real-ESRGAN
# ------------------------------------------------

python - <<'PY'

import realesrgan

print("ESRGAN   : OK")
print("Path     :", realesrgan.__file__)

PY


# ------------------------------------------------
# Real-ESRGAN MODEL
# ------------------------------------------------

echo ""
echo "Real-ESRGAN model:"

if [ -s "$REALESRGAN_MODEL" ]; then

    echo "STATUS: FOUND"
    ls -lh "$REALESRGAN_MODEL"

else

    echo "STATUS: MISSING"

fi


# ------------------------------------------------
# RIFE
# ------------------------------------------------

echo ""
echo "RIFE repository:"

if [ -d "$RIFE_DIR" ]; then

    echo "STATUS: FOUND"

else

    echo "STATUS: MISSING"

fi


# ------------------------------------------------
# RIFE CHECKPOINT
# ------------------------------------------------

echo ""
echo "RIFE checkpoint:"

if [ -n "$(find "$RIFE_CHECKPOINT_DIR" -type f -print -quit 2>/dev/null)" ]; then

    echo "STATUS: FOUND"
    echo ""

    find "$RIFE_CHECKPOINT_DIR" -type f | head -20

else

    echo "STATUS: MISSING"

fi


# ------------------------------------------------
# FFMPEG
# ------------------------------------------------

echo ""
echo "FFmpeg:"

if command -v ffmpeg >/dev/null 2>&1; then

    echo "STATUS: FOUND"

else

    echo "STATUS: MISSING"

fi


# ================================================================
# STATUS DIRECTORY
# ================================================================

mkdir -p "$STATUS_DIR"


# ================================================================
# COMPLETE
# ================================================================

echo ""
echo "================================================"
echo " SETUP COMPLETE"
echo "================================================"
echo ""

echo "Mastering stack:"
echo ""
echo "  PyTorch       : READY"
echo "  CUDA/T4       : READY"
echo "  BasicSR       : SOURCE"
echo "  Real-ESRGAN   : SOURCE"
echo "  ESRGAN Model  : CHECK ABOVE"
echo "  RIFE          : CHECK ABOVE"
echo "  RIFE Model    : CHECK ABOVE"
echo "  FFmpeg        : CHECK ABOVE"
echo ""

echo "Do NOT run colab_master.py yet."
echo ""
echo "Next:"
echo "  1. RIFE standalone test"
echo "  2. Real-ESRGAN standalone test"
echo "  3. FFmpeg test"
echo "  4. Full mastering test"
echo ""