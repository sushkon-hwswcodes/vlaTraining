#!/usr/bin/env bash
# setup_gpu_machine.sh
#
# Run this once on a fresh rented GPU machine to get training ready.
# Usage:
#   bash setup_gpu_machine.sh
#
# After it completes, start training with:
#   bash start_training.sh

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> [1/7] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
  gcc g++ build-essential \
  libgl1 libvulkan1 vulkan-tools libvulkan-dev \
  xvfb

echo "==> [2/7] Installing ManiSkill from source..."
cd "$REPO_DIR"
pip install -e . --quiet

echo "==> [3/7] Installing PPO training extras..."
pip install tensorboard "tyro>=0.8.5" --quiet

echo "==> [4/7] Pinning NumPy < 2 (required for PyTorch 2.2.x compatibility)..."
pip install "numpy<2" --quiet

echo "==> [5/7] Fixing CUDA symlink for GPU physics..."
LIBCUDA_SO=/usr/lib/x86_64-linux-gnu/libcuda.so
if [ ! -f "$LIBCUDA_SO" ]; then
  sudo ln -s "${LIBCUDA_SO}.1" "$LIBCUDA_SO"
  echo "    symlink created: $LIBCUDA_SO"
else
  echo "    symlink already exists, skipping."
fi

echo "==> [6/7] Downloading ReplicaCAD scene assets..."
python -m mani_skill.utils.download_asset ReplicaCAD

echo "==> [7/7] Configuring git..."
git -C "$REPO_DIR" config user.name  "sushkon-hwswcodes"
git -C "$REPO_DIR" config user.email "thisissush@gmail.com"
if [ -z "$GITHUB_TOKEN" ]; then
  echo "    WARNING: GITHUB_TOKEN not set — git push from training won't work."
  echo "    Set it with: export GITHUB_TOKEN=<your_pat>"
else
  git -C "$REPO_DIR" remote set-url origin \
    "https://${GITHUB_TOKEN}@github.com/sushkon-hwswcodes/vlaTraining.git"
  echo "    git remote configured with token."
fi

echo ""
echo "==> Setup complete. Run 'bash start_training.sh' to begin."
