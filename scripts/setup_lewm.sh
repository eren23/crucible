#!/bin/bash
# Setup script for running Le-WM on a RunPod 4090
# Usage: ssh pod "bash -s" < scripts/setup_lewm.sh
set -euo pipefail

echo "=== Le-WM Setup on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

WORKDIR=/workspace/le-wm
STABLEWM_HOME=/workspace/stable-wm-data
export STABLEWM_HOME

# 1. Install uv (fast Python package manager)
echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. Clone Le-WM
echo "=== Cloning Le-WM ==="
if [ -d "$WORKDIR" ]; then
    echo "Already cloned, pulling latest..."
    cd "$WORKDIR" && git pull
else
    git clone https://github.com/lucas-maes/le-wm.git "$WORKDIR"
    cd "$WORKDIR"
fi

# 3. Create Python 3.10 venv and install deps
echo "=== Creating Python 3.10 environment ==="
uv venv --python=3.10 .venv
source .venv/bin/activate

echo "=== Installing stable-worldmodel[train,env] ==="
uv pip install "stable-worldmodel[train,env]"
uv pip install gdown  # For downloading data from Google Drive

# 4. Download pusht dataset from Google Drive
echo "=== Downloading pusht dataset ==="
mkdir -p "$STABLEWM_HOME"

# Download the entire data folder from Google Drive
# Folder ID: 1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e
gdown --folder "https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e" -O "$STABLEWM_HOME/" --remaining-ok || {
    echo "WARNING: gdown folder download failed. Trying individual files..."
    echo "You may need to download data manually from:"
    echo "https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e"
}

# Extract any .tar.zst archives
for f in "$STABLEWM_HOME"/*.tar.zst; do
    if [ -f "$f" ]; then
        echo "Extracting $f..."
        tar --zstd -xvf "$f" -C "$STABLEWM_HOME/"
        rm "$f"  # Clean up archive after extraction
    fi
done

# 5. Configure: disable WandB by default (can enable later)
echo "=== Configuring training ==="
export WANDB_MODE=disabled

# 6. Print summary
echo ""
echo "=== Setup Complete ==="
echo "Workdir: $WORKDIR"
echo "Data dir: $STABLEWM_HOME"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo ""
echo "Contents of data dir:"
ls -la "$STABLEWM_HOME/" 2>/dev/null || echo "(empty)"
echo ""
echo "To train: cd $WORKDIR && source .venv/bin/activate && WANDB_MODE=disabled STABLEWM_HOME=$STABLEWM_HOME python train.py data=pusht"
