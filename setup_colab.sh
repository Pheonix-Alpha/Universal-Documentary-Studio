%%writefile setup_colab.sh
#!/bin/bash

# ================================================================
# Universal Documentary Studio
# ================================================================
# COLAB MASTERING ENVIRONMENT
#
# Purpose:
#   - Receive raw videos from Kaggle
#   - RIFE frame interpolation
#   - Real-ESRGAN AI upscaling
#   - FFmpeg final encoding
#
# Target:
#   Google Colab
#   NVIDIA T4
#   Existing CUDA-enabled PyTorch
#
# IMPORTANT:
#   Do NOT reinstall torch/torchvision.
#   Do NOT install the legacy PyPI BasicSR package.
# ================================================================

set -e


# ================================================================
# CONFIGURATION
# ================================================================

RIFE_REPO="https://github.com/hzwer/Practical-RIFE.git"
REALESRGAN_REPO="https://github.com/xinntao/Real-ESRGAN.git"

REALESRGAN_MODEL="RealESRGAN_x4plus.pth"

RIFE_DIR="rife_repo"
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

gpu = torch.cuda.get_device_name(0)

vram = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print("GPU    :", gpu)
print("VRAM   :", round(vram, 2), "GB")

PY

echo ""
echo "GPU environment OK."


# ================================================================
# 2. INSTALL MASTERING RUNTIME DEPENDENCIES
# ================================================================

echo ""
echo "[2/8] Installing mastering dependencies..."

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

echo "Dependencies installed."


# ================================================================
# 3. INSTALL REAL-ESRGAN
# ================================================================

echo ""
echo "[3/8] Preparing Real-ESRGAN..."

#
# We intentionally DO NOT run:
#
#     pip install basicsr
#
# BasicSR's legacy setup.py installation is incompatible
# with the modern Python environment used by Colab.
#
# Real-ESRGAN is installed from source without dependency
# resolution so pip does not pull the broken legacy BasicSR
# package automatically.
#

if python -c "import realesrgan" >/dev/null 2>&1; then

    echo "Real-ESRGAN already importable."
    echo "Skipping installation."

else

    echo "Installing Real-ESRGAN from source..."

    pip install -q \
        --no-deps \
        --no-build-isolation \
        "git+${REALESRGAN_REPO}"

fi


# ================================================================
# REAL-ESRGAN IMPORT TEST
# ================================================================

echo ""
echo "Testing Real-ESRGAN import..."

if python -c "import realesrgan" >/dev/null 2>&1; then

    echo "Real-ESRGAN import: OK"

else

    echo ""
    echo "ERROR: Real-ESRGAN cannot be imported."
    echo ""
    echo "Detailed error:"
    echo ""

    python -c "import realesrgan"

    exit 1

fi


# ================================================================
# 4. DOWNLOAD REAL-ESRGAN MODEL
# ================================================================

echo ""
echo "[4/8] Checking Real-ESRGAN x4plus model..."

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
    echo "Model downloaded:"
    ls -lh "$REALESRGAN_MODEL"

fi


# ================================================================
# 5. CLONE PRACTICAL-RIFE
# ================================================================

echo ""
echo "[5/8] Preparing Practical-RIFE..."

if [ -d "$RIFE_DIR/.git" ]; then

    echo "Practical-RIFE already cloned."
    echo "Skipping clone."

else

    if [ -d "$RIFE_DIR" ]; then

        echo "Removing incomplete RIFE directory..."
        rm -rf "$RIFE_DIR"

    fi

    echo "Cloning Practical-RIFE..."

    git clone \
        --depth 1 \
        "$RIFE_REPO" \
        "$RIFE_DIR"

fi

echo "RIFE repository ready."


# ================================================================
# 6. PREPARE RIFE CHECKPOINT DIRECTORY
# ================================================================

echo ""
echo "[6/8] Preparing RIFE checkpoint..."

mkdir -p "$RIFE_CHECKPOINT_DIR"


# ------------------------------------------------
# Check existing checkpoint
# ------------------------------------------------

if [ -n "$(find "$RIFE_CHECKPOINT_DIR" -type f -print -quit 2>/dev/null)" ]; then

    echo "RIFE checkpoint already exists."

# ------------------------------------------------
# Check repository checkpoint
# ------------------------------------------------

elif [ -d "$RIFE_DIR/train_log" ] && \
     [ -n "$(find "$RIFE_DIR/train_log" -type f -print -quit 2>/dev/null)" ]; then

    echo "RIFE checkpoint found inside repository."
    echo "Copying checkpoint..."

    cp -r \
        "$RIFE_DIR/train_log/." \
        "$RIFE_CHECKPOINT_DIR/"

# ------------------------------------------------
# Check model directory
# ------------------------------------------------

elif [ -d "$RIFE_DIR/train_log" ]; then

    echo "RIFE train_log directory exists."
    echo "Checking for checkpoint files..."

    find "$RIFE_DIR/train_log" -type f | head -20 || true

else

    echo ""
    echo "WARNING:"
    echo "RIFE pretrained checkpoint was not found."
    echo ""
    echo "Repository:"
    echo "    $RIFE_DIR/"
    echo ""
    echo "Expected checkpoint directory:"
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

    echo "FFmpeg not found."
    echo "Installing FFmpeg..."

    apt-get update -qq
    apt-get install -y -qq ffmpeg

fi

echo ""
ffmpeg -version | head -1


# ================================================================
# 8. FINAL ENVIRONMENT VERIFICATION
# ================================================================

echo ""
echo "================================================"
echo " FINAL ENVIRONMENT VERIFICATION"
echo "================================================"
echo ""


# ------------------------------------------------
# Python / GPU
# ------------------------------------------------

python - <<'PY'

import sys
import torch

print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.version.cuda)
print("GPU    :", torch.cuda.get_device_name(0))

PY


# ------------------------------------------------
# OpenCV
# ------------------------------------------------

python - <<'PY'

import cv2

print("OpenCV :", cv2.__version__)

PY


# ------------------------------------------------
# Real-ESRGAN
# ------------------------------------------------

python - <<'PY'

try:

    import realesrgan

    print("RealESRGAN : OK")

except Exception as e:

    print("RealESRGAN : FAILED")
    print("Error      :", e)

PY


# ------------------------------------------------
# Real-ESRGAN model
# ------------------------------------------------

echo ""
echo "Real-ESRGAN model:"

if [ -s "$REALESRGAN_MODEL" ]; then

    ls -lh "$REALESRGAN_MODEL"
    echo "STATUS: FOUND"

else

    echo "STATUS: MISSING"

fi


# ------------------------------------------------
# RIFE repository
# ------------------------------------------------

echo ""
echo "RIFE repository:"

if [ -d "$RIFE_DIR" ]; then

    echo "STATUS: FOUND"
    echo "PATH  : $RIFE_DIR"

else

    echo "STATUS: MISSING"

fi


# ------------------------------------------------
# RIFE checkpoint
# ------------------------------------------------

echo ""
echo "RIFE checkpoint:"

if [ -n "$(find "$RIFE_CHECKPOINT_DIR" -type f -print -quit 2>/dev/null)" ]; then

    echo "STATUS: FOUND"
    echo ""
    echo "Checkpoint files:"
    find "$RIFE_CHECKPOINT_DIR" -type f | head -20

else

    echo "STATUS: MISSING"
    echo ""
    echo "RIFE checkpoint must be added before"
    echo "the mastering pipeline can run interpolation."

fi


# ------------------------------------------------
# FFmpeg
# ------------------------------------------------

echo ""
echo "FFmpeg:"

if command -v ffmpeg >/dev/null 2>&1; then

    echo "STATUS: FOUND"

else

    echo "STATUS: MISSING"

fi


# ================================================================
# CREATE STATUS DIRECTORY
# ================================================================

mkdir -p "$STATUS_DIR"


# ================================================================
# SETUP COMPLETE
# ================================================================

echo ""
echo "================================================"
echo " SETUP COMPLETE"
echo "================================================"
echo ""

echo "Environment:"
echo "  Python       : READY"
echo "  PyTorch      : READY"
echo "  CUDA         : READY"
echo "  Real-ESRGAN  : CHECK ABOVE"
echo "  RIFE         : CHECK ABOVE"
echo "  FFmpeg       : CHECK ABOVE"

echo ""
echo "Required assets:"
echo "  RealESRGAN_x4plus.pth"
echo "  rife_repo/"
echo "  train_log/"
echo ""

echo "Next pipeline stage:"
echo ""
echo "  1. Test RIFE"
echo "  2. Test Real-ESRGAN"
echo "  3. Test FFmpeg"
echo "  4. Test colab_master.py"
echo "  5. Process Kaggle raw assets"
echo "  6. Generate final 1080p master"
echo ""