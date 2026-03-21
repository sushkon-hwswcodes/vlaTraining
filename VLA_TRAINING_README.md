# VLA Training — ManiSkill Experiments

This repo tracks our VLA (Vision-Language-Action) training experiments using [ManiSkill](https://github.com/haosulab/ManiSkill), a GPU-accelerated robot simulation framework.

---

## Environment Setup

### System Requirements
- Linux (Ubuntu 20.04+)
- NVIDIA GPU with CUDA 12.x+
- Python 3.10
- `build-essential` (gcc, g++) for compiling dependencies

### Installation

**1. Install system dependencies**
```bash
sudo apt-get update && sudo apt-get install -y \
  gcc g++ build-essential \
  libgl1 libvulkan1 vulkan-tools libvulkan-dev \
  xvfb
```

**2. Install ManiSkill from source**
```bash
cd /workspace/ManiSkill
pip install -e .
pip install tensorboard
```

**3. Fix known dependency conflicts**
```bash
# NumPy 2.x is incompatible with PyTorch 2.2.x — downgrade
pip install "numpy<2"
```

**4. Fix CUDA library symlink (required for GPU physics)**
```bash
sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
```

**5. Start virtual display (required for headless GPU rendering)**
```bash
Xvfb :1 -screen 0 1280x1024x24 &
export DISPLAY=:1
```

> **Why Xvfb?** The NVIDIA Vulkan ICD (`libGLX_nvidia.so.0`) requires an X server to initialize even in headless environments. Without it you get `vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed`.

---

## Experiments

### Experiment 1 — Random Action Demo

Sanity check: runs a Panda arm on `PickCube-v1` with purely random actions. Confirms the simulation + GPU rendering pipeline is working.

```bash
DISPLAY=:1 python -m mani_skill.examples.demo_random_action \
  -e "PickCube-v1" \
  --render-mode rgb_array \
  --record-dir ./recordings
```

**Output:** `recordings/0.mp4`

What you see:
- Red cube = object to pick up
- Green ball = goal position (where cube needs to go)
- Robot arm flails randomly — no policy, just physics verification

---

### Experiment 2 — Motion Planning Expert (PickCube-v1)

Runs a **classical robotics pipeline** (IK + collision-free motion planning) as an intelligent baseline. No neural network — uses full simulator state access.

```bash
DISPLAY=:1 python -m mani_skill.examples.motionplanning.panda.run \
  --env-id "PickCube-v1" \
  --num-traj 5 \
  --save-video \
  --record-dir ./demos
```

**Output:** `demos/PickCube-v1/motionplanning/{0..4}.mp4`

How it works:
1. **Inverse Kinematics (IK)** — computes joint angles to reach target pose
2. **Motion Planning (OMPL)** — finds collision-free path through joint space
3. **Execution** — follows planned trajectory in the simulator

Available tasks with motion planning solutions:
- `PickCube-v1`, `StackCube-v1`, `PegInsertionSide-v1`
- `LiftPegUpright-v1`, `PlaceSphere-v1`, `PlugCharger-v1`
- `PushCube-v1`, `PullCube-v1`, `PullCubeTool-v1`, `DrawTriangle-v1`, `DrawSVG-v1`

---

### Experiment 3 — PPO State-Based Training (PickCube-v1)

Trains a neural network policy using **Proximal Policy Optimization (PPO)** on robot joint state observations.

```bash
cd examples/baselines/ppo

DISPLAY=:1 python ppo.py \
  --env_id="PickCube-v1" \
  --num_envs=2048 \
  --update_epochs=8 \
  --num_minibatches=32 \
  --total_timesteps=10_000_000 \
  --eval_freq=10 \
  --num-steps=50
```

**Key flags:**
| Flag | Value | Purpose |
|------|-------|---------|
| `--num_envs` | 2048 | Parallel GPU-simulated environments |
| `--total_timesteps` | 10M | More is better — 2M is not enough for PickCube |
| `--num-steps` | 50 | Episode length; PickCube needs ≥50 steps |
| `--eval_freq` | 10 | Evaluate every 10 updates |

**Checkpoints saved to:** `runs/PickCube-v1__ppo__<seed>__<timestamp>/`

**Monitor training:**
```bash
tensorboard --logdir runs/
```

**Evaluate a checkpoint:**
```bash
DISPLAY=:1 python ppo.py \
  --env_id="PickCube-v1" \
  --evaluate \
  --checkpoint=runs/PickCube-v1__ppo__1__<timestamp>/final_ckpt.pt \
  --num_eval_envs=1 \
  --num-eval-steps=200
```

> **Note from run 1:** Training with `--num-steps=20` and `--total_timesteps=2_000_000` did not converge (success_once=0.0). PickCube requires longer episodes and more timesteps. Use `--num-steps=50` and `--total_timesteps=10_000_000` minimum.

---

### Experiment 4 — PPO RGB Training (Work in Progress)

Trains a CNN-based policy from raw **RGB camera observations** instead of joint states. More realistic but slower to train.

```bash
cd examples/baselines/ppo

DISPLAY=:1 python ppo_rgb.py \
  --env_id="PickCube-v1" \
  --num_envs=256 \
  --update_epochs=8 \
  --num_minibatches=8 \
  --total_timesteps=10_000_000
```

**Changes from upstream:**
- `ppo_rgb.py` has been modified to support **RGBD** observations (RGB + depth channels)
- Depth obs is passed through its own CNN branch before concatenation with RGB features

---

## Faster PPO Variant

`ppo_fast.py` uses CUDA graphs and other optimizations for maximum throughput:

```bash
cd examples/baselines/ppo

DISPLAY=:1 python ppo_fast.py \
  --env_id="PushCube-v1" \
  --num_envs=4096 \
  --num-steps=4 \
  --update_epochs=8 \
  --num_minibatches=32 \
  --total_timesteps=50_000_000 \
  --cudagraphs
```

> **Start here for quick validation** — `PushCube-v1` is simpler than `PickCube-v1` and converges reliably in ~2M steps.

---

## Available Baselines

| Baseline | Script | Obs Mode | Notes |
|----------|--------|----------|-------|
| PPO (state) | `ppo/ppo.py` | Joint states | Fastest to train |
| PPO (fast) | `ppo/ppo_fast.py` | Joint states | CUDA graphs, max throughput |
| PPO (RGB) | `ppo/ppo_rgb.py` | RGB + Depth | Modified to support RGBD |
| SAC | `sac/` | Joint states | Off-policy, sample efficient |
| Diffusion Policy | `diffusion_policy/` | State / RGB | Imitation learning |
| BC | `bc/` | State / RGB | Behavioral cloning |
| ACT | `act/` | State / RGB | Action chunking transformer |
| RFCL | `rfcl/` | State | Reverse curriculum |
| RLPD | `rlpd/` | State | RL with prior data |
| TDMPCv2 | `tdmpc2/` | State | Model-based RL |

---

## Re-rendering with Ray-Tracing

After collecting trajectories, re-render with higher-quality visuals:

```bash
DISPLAY=:1 python -m mani_skill.trajectory.replay_trajectory \
  --traj-path=path/to/trajectory.h5 \
  --use-env-states \
  --shader="rt-fast" \
  --save-video \
  --allow-failure \
  -o none
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `libcuda.so: cannot open shared object file` | `sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so` |
| `RuntimeError: Numpy is not available` | `pip install "numpy<2"` |
| `vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed` | Start Xvfb: `Xvfb :1 -screen 0 1280x1024x24 &` then `export DISPLAY=:1` |
| `ImportError: libGL.so.1` | `sudo apt-get install -y libgl1` |
| `ModuleNotFoundError: No module named 'tensorboard'` | `pip install tensorboard` |
| `Failed building wheel for toppra` (gcc missing) | `sudo apt-get install -y gcc g++ build-essential` |

