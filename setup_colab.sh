#!/bin/bash
# ================================================================
# Universal Documentary Studio
# Colab Mastering Environment
#
# Usage after every Colab session restart:
#
#     !bash setup_colab.sh
#
# This script is SAFE TO RUN MULTIPLE TIMES.
# It only installs/downloads things that are missing.
# ================================================================

set -e

echo ""
echo "================================================"
echo " Universal Documentary Studio"
echo " Colab Environment Setup"
echo "================================================"

# ------------------------------------------------
# 0. Project directory
# ------------------------------------------------

PROJECT_DIR="$(pwd)"

echo ""
echo "[0/7] Project:"
echo "      $PROJECT_DIR"


# ------------------------------------------------
# 1. Verify GPU / PyTorch
# ------------------------------------------------

echo ""
echo "[1/7] Checking GPU and PyTorch..."

python - <<'PY'
import torch

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.version.cuda)
print("CUDA OK :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is not available. "
        "Change Colab Runtime to GPU and run again."
    )

print("GPU     :", torch.cuda.get_device_name(0))
print("VRAM    :", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3,
    2
), "GB")
PY


# ------------------------------------------------
# 2. Install lightweight runtime dependencies
# ------------------------------------------------

echo ""
echo "[2/7] Checking runtime dependencies..."

pip install -q \
    opencv-python-headless \
    scipy \
    tqdm \
    imageio \
    imageio-ffmpeg


# ------------------------------------------------
# 3. BasicSR
#
# IMPORTANT:
# Do NOT install the old PyPI BasicSR 1.4.2.
# Python 3.13 breaks its setup.py metadata.
#
# Install directly from GitHub if missing.
# ------------------------------------------------

echo ""
echo "[3/7] Checking BasicSR..."

if python -c "import basicsr" 2>/dev/null; then

    echo "BasicSR already installed."

else

    echo "BasicSR not found."
    echo "Installing BasicSR from GitHub..."

    pip install -q \
        --no-build-isolation \
        git+https://github.com/XPixelGroup/BasicSR.git

fi


# ------------------------------------------------
# 4. Real-ESRGAN
#
# Install only if import is missing.
# ------------------------------------------------

echo ""
echo "[4/7] Checking Real-ESRGAN..."

if python -c "import realesrgan" 2>/dev/null; then

    echo "Real-ESRGAN already installed."

else

    echo "Real-ESRGAN not found."
    echo "Installing Real-ESRGAN from GitHub..."

    pip install -q \
        --no-build-isolation \
        git+https://github.com/xinntao/Real-ESRGAN.git

fi


# ------------------------------------------------
# 5. Real-ESRGAN model
# ------------------------------------------------

echo ""
echo "[5/7] Checking Real-ESRGAN model..."

MODEL_FILE="RealESRGAN_x4plus.pth"

if [ -s "$MODEL_FILE" ]; then

    echo "$MODEL_FILE already exists."
    ls -lh "$MODEL_FILE"

else

    echo "Model not found."
    echo "Downloading $MODEL_FILE..."

    wget -q --show-progress \
        -O "$MODEL_FILE" \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

    test -s "$MODEL_FILE"

    echo "Model downloaded successfully."

fi


# ------------------------------------------------
# 6. Practical-RIFE repository
# ------------------------------------------------

echo ""
echo "[6/7] Checking Practical-RIFE..."

if [ -d "rife_repo/.git" ]; then

    echo "Practical-RIFE already cloned."

else

    if [ -d "rife_repo" ]; then
        echo "Incomplete rife_repo detected."
        echo "Removing incomplete directory..."
        rm -rf rife_repo
    fi

    echo "Cloning Practical-RIFE..."

    git clone --depth 1 \
        https://github.com/hzwer/Practical-RIFE.git \
        rife_repo

fi


# ------------------------------------------------
# RIFE checkpoint directory
# ------------------------------------------------

mkdir -p train_log

echo ""
echo "Checking RIFE checkpoint..."

if [ -n "$(find train_log -type f -print -quit 2>/dev/null)" ]; then

    echo "RIFE checkpoint files found:"
    find train_log -maxdepth 2 -type f | head -20

else

    echo "No RIFE checkpoint found in ./train_log."

    # Some Practical-RIFE versions may contain weights
    # inside the cloned repository.
    if [ -d "rife_repo/train_log" ] && \
       [ -n "$(find rife_repo/train_log -type f -print -quit 2>/dev/null)" ]; then

        echo "Copying available RIFE checkpoint files..."

        cp -r rife_repo/train_log/. train_log/

    fi

fi


# ------------------------------------------------
# 7. Final verification
# ------------------------------------------------

echo ""
echo "================================================"
echo " FINAL VERIFICATION"
echo "================================================"

python - <<'PY'
import torch
import cv2

print("")
print("PyTorch     :", torch.__version__)
print("CUDA        :", torch.version.cuda)
print("GPU         :", torch.cuda.get_device_name(0))
print("OpenCV      :", cv2.__version__)

try:
    import basicsr
    print("BasicSR     : OK")
except Exception as e:
    print("BasicSR     : FAILED")
    print("Error       :", e)
    raise

try:
    import realesrgan
    print("Real-ESRGAN : OK")
except Exception as e:
    print("Real-ESRGAN : FAILED")
    print("Error       :", e)
    raise
PY


echo ""
echo "Real-ESRGAN model:"
if [ -s "$MODEL_FILE" ]; then
    ls -lh "$MODEL_FILE"
else
    echo "ERROR: $MODEL_FILE missing"
    exit 1
fi


echo ""
echo "Practical-RIFE:"
if [ -d "rife_repo" ]; then
    echo "Repository : OK"
else
    echo "Repository : FAILED"
    exit 1
fi


echo ""
echo "RIFE checkpoint:"
if [ -n "$(find train_log -type f -print -quit 2>/dev/null)" ]; then
    echo "Checkpoint : FOUND"
    find train_log -maxdepth 2 -type f | head -20
else
    echo "Checkpoint : NOT FOUND"
    echo ""
    echo "WARNING:"
    echo "RIFE repository exists, but no checkpoint was found."
    echo "Do NOT run the mastering pipeline until the RIFE"
    echo "checkpoint is available."
fi


echo ""
echo "================================================"
echo " SETUP COMPLETE"
echo "================================================"
echo ""
echo "You can now verify/run the mastering pipeline."
echo ""