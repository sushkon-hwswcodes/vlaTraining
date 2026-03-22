#!/usr/bin/env bash
# start_training.sh
#
# Starts a tmux session with the virtual display + PPO training.
# Run this every time you spin up the machine (after setup_gpu_machine.sh).
#
# Usage:
#   bash start_training.sh [env_id] [checkpoint_path]
#
# Examples:
#   bash start_training.sh
#   bash start_training.sh PickCube-v1
#   bash start_training.sh PickCubeReplicaCAD-v1
#   bash start_training.sh PickCubeReplicaCAD-v1 runs/PickCubeReplicaCAD-v1__ppo__1__<ts>/ckpt_11.pt

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPO_DIR="$REPO_DIR/examples/baselines/ppo"

ENV_ID="${1:-PickCubeReplicaCAD-v1}"
CHECKPOINT="${2:-}"

# Pick sensible defaults per env
if [[ "$ENV_ID" == *"ReplicaCAD"* ]]; then
  NUM_ENVS=256
  NUM_STEPS=200
else
  NUM_ENVS=2048
  NUM_STEPS=50
fi

# Build the training command
TRAIN_CMD="cd $PPO_DIR && DISPLAY=:1 python ppo.py \
  --env_id=\"$ENV_ID\" \
  --num_envs=$NUM_ENVS \
  --update_epochs=8 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --eval_freq=10 \
  --num-steps=$NUM_STEPS"

if [ -n "$CHECKPOINT" ]; then
  TRAIN_CMD="$TRAIN_CMD --checkpoint=\"$CHECKPOINT\""
fi

SESSION="training"

echo "==> Starting virtual display..."
pkill Xvfb 2>/dev/null || true
Xvfb :1 -screen 0 1280x1024x24 &
sleep 1
export DISPLAY=:1

echo "==> Pulling latest code from git..."
git -C "$REPO_DIR" pull --ff-only

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "==> tmux session '$SESSION' already exists — attaching."
  tmux attach -t "$SESSION"
else
  echo "==> Launching tmux session '$SESSION'..."
  echo "    Env:        $ENV_ID"
  echo "    Num envs:   $NUM_ENVS"
  echo "    Num steps:  $NUM_STEPS"
  [ -n "$CHECKPOINT" ] && echo "    Checkpoint: $CHECKPOINT"
  echo ""
  echo "    Detach with Ctrl+B then D. Reattach with: tmux attach -t $SESSION"
  echo ""
  tmux new-session -s "$SESSION" -d
  tmux send-keys -t "$SESSION" "export DISPLAY=:1" Enter
  tmux send-keys -t "$SESSION" "$TRAIN_CMD" Enter
  tmux attach -t "$SESSION"
fi
