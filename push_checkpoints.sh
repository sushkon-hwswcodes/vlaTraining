#!/usr/bin/env bash
# push_checkpoints.sh
#
# Force-adds the latest checkpoint(s) from runs/ and pushes to remote.
# Usage:
#   bash push_checkpoints.sh                  # push latest ckpt from newest run
#   bash push_checkpoints.sh path/to/ckpt.pt  # push a specific checkpoint

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [ -n "$1" ]; then
  CKPT_FILES=("$1")
else
  # Find the most recent run directory
  LATEST_RUN=$(ls -td examples/baselines/ppo/runs/*/ 2>/dev/null | head -1)
  if [ -z "$LATEST_RUN" ]; then
    echo "No run directories found under examples/baselines/ppo/runs/"
    exit 1
  fi
  # Grab the latest checkpoint in that run
  mapfile -t CKPT_FILES < <(ls -t "${LATEST_RUN}"ckpt_*.pt 2>/dev/null | head -3)
  if [ ${#CKPT_FILES[@]} -eq 0 ]; then
    echo "No .pt checkpoints found in $LATEST_RUN"
    exit 1
  fi
fi

echo "==> Staging checkpoints:"
for f in "${CKPT_FILES[@]}"; do
  echo "    $f"
  git add -f "$f"
done

# Also stage the tfevents log if present (useful for TensorBoard)
LATEST_RUN=$(dirname "${CKPT_FILES[0]}")
TFEVENT=$(ls "${LATEST_RUN}"/events.out.tfevents.* 2>/dev/null | head -1)
if [ -n "$TFEVENT" ]; then
  echo "    $TFEVENT (tfevents)"
  git add -f "$TFEVENT"
fi

if git diff --cached --quiet; then
  echo "Nothing new to commit."
  exit 0
fi

RUN_NAME=$(basename "$LATEST_RUN")
git commit -m "checkpoint: ${RUN_NAME} — $(date '+%Y-%m-%d %H:%M')"

echo "==> Pushing to remote..."
git push
echo "==> Done."
