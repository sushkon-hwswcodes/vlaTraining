# Research Report: VLA Training Experiments
**Date:** April 2026  
**Repos:** [vlaTraining](https://github.com/sushkon-hwswcodes/vlaTraining) · [llmandRobots](https://github.com/sushkon-hwswcodes/llmandRobots)

---

## Overview

This report documents two parallel experiments in robot manipulation research:

1. **RL Baseline** — Training a PPO policy from scratch in ManiSkill/SAPIEN simulation
2. **Code-as-Policy** — Using a local LLM (Qwen2.5-Coder-7B) to generate robot control code zero-shot, built on the CaP-X framework

The goal is to compare these two fundamentally different paradigms on the same manipulation task (cube lifting), then extend both to more general settings (arbitrary object shapes/sizes).

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 (8GB VRAM) |
| CUDA | 12.8 (driver only, no toolkit) |
| Python | 3.10.13 |
| OS | Ubuntu 22.04 |

---

## Experiment 1: RL Baseline (ManiSkill + PPO)

### Setup
- **Simulator:** ManiSkill/SAPIEN (GPU-accelerated, fork of upstream ManiSkill)
- **Task:** `PickCube-v1` — Franka Panda arm picks a red cube and places it at a randomized goal position
- **Algorithm:** Proximal Policy Optimization (PPO), state-based observations (joint angles, velocities, object/goal poses)
- **Environment:** 512 parallel GPU environments

### Infrastructure fixes required
- Missing `libX11.so.6` → set `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`
- Missing NVIDIA Vulkan ICD → copied `nvidia_icd.json` to `/usr/share/vulkan/icd.d/`
- Headless rendering → started Xvfb virtual display (`DISPLAY=:1`)

### Custom environment: PickCubeReplicaCAD-v1
Built a custom ManiSkill environment (`mani_skill/envs/tasks/tabletop/pick_cube_replicacad.py`) placing the PickCube task inside a photorealistic ReplicaCAD apartment scene with a Fetch robot. Required tuning `collision_stack_size` to 64MB to handle the heavy scene geometry.

### Training run
```
env:            PickCube-v1
num_envs:       512
total_timesteps: 10,000,000
update_epochs:  8
num_minibatches: 32
num_steps:      50
```

### Results

| Epoch | Success Rate | Avg Return |
|---|---|---|
| 1 | 0% | 2.91 |
| 151 | **100%** | 37.80 |

- **Stopped at epoch 161** (38% through training) — task was already solved at epoch 151
- **Checkpoint saved:** `ckpt_161.pt`
- **Eval video saved:** `videos/16.mp4`
- Speed: ~4,600 steps/sec on RTX 3070

### Key finding
PPO converges to **100% success on PickCube-v1 in ~4M timesteps** (~15 minutes on RTX 3070 with 512 parallel envs).

---

## Experiment 2: Code-as-Policy (CaP-X + Qwen2.5-Coder-7B)

### Motivation
Rather than learning a policy through RL trial-and-error, can a language model generate Python code that directly solves the task zero-shot? This approach requires no training data or gradient updates — the LLM reasons about the task and writes executable robot control code.

### Framework: CaP-X
Forked [capgym/cap-x](https://github.com/capgym/cap-x) (MIT licensed, NVIDIA/Berkeley/Stanford/CMU, AAAI 2026). CaP-X is a Code-as-Policy framework where:
- An LLM receives the task description + robot observations
- Generates Python code that calls a robot control API
- Code is executed live against the simulator
- Errors are fed back to the LLM for retry (multi-turn)

### Our modifications to CaP-X
1. **Ollama integration** (`capx/llm/client.py`): Added `OLLAMA_MODELS`, `is_ollama_model()`, routing of `ollama/` prefixed models to `localhost:11434/v1/chat/completions`. No proxy server needed — Ollama already exposes an OpenAI-compatible endpoint.
2. **Robosuite 1.5.2 compatibility** (`capx/envs/simulators/robosuite_base.py`): Removed `skip_render_images` kwarg dropped in robosuite 1.5.2.
3. **New task config** (`env_configs/cube_lifting_qwen/franka_qwen_privileged.yaml`): Cube lifting with privileged state API — no perception stack (SAM3, ContactGraspNet) required.

### Setup
- **Simulator:** Robosuite 1.5.2 / MuJoCo (CPU-based — no GPU conflict with LLM)
- **Task:** Franka cube lifting (same task, different sim)
- **LLM:** `Qwen2.5-Coder-7B-Instruct` (Q4_K_M quantization) served via Ollama locally
- **Observations:** Privileged state — ground truth object pose, goal pose passed directly to LLM
- **IK:** PyRoKi server (CPU, runs as local HTTP service on port 8116)
- **Rendering:** OSMesa (software, headless)

### How it works
```
Episode start
    │
    ▼
LLM receives:
  - system prompt (API docs, coordinate system, units)
  - object pose: (x, y, z, quaternion)
  - goal pose: (x, y, z, quaternion)
  - task: "lift the red cube"
    │
    ▼
LLM generates Python code:
  obj = get_object_pose("red_cube")
  grasp_pos, grasp_quat = sample_grasp_pose("red_cube")
  goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
  close_gripper()
  lift_position = grasp_pos.copy(); lift_position[2] += 0.1
  goto_pose(lift_position, grasp_quat)
    │
    ▼
Code executes live against Robosuite sim
    │
    ▼
Result: reward, task_completed, video saved
```

### Infrastructure fixes required
- `libX11.so.6` not on path → `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu`
- MuJoCo EGL context failure → switched to `MUJOCO_GL=osmesa`, installed `libosmesa6`
- `sam3`, `curobo` not buildable (no CUDA toolkit/nvcc) → skipped perception stack, used privileged state config only

### 20-Trial Benchmark Results

**Model:** `ollama/qwen2.5-coder:7b-instruct-q4_K_M`  
**Task:** Franka cube lifting, privileged state, zero-shot (no training)

| Metric | Result |
|---|---|
| **Task success rate** | **2/20 (10%)** |
| Code execution success | 13/20 (65%) |
| Average reward | 0.194 |
| Successful trials | #6, #8 |
| Avg time per trial | ~27 seconds |

### Failure mode analysis (preliminary)
- **35% of trials:** Code crashes (syntax errors, wrong API usage, missing `import numpy`) — sandbox exits with rc=1
- **55% of trials:** Code runs but grasping fails — robot moves to approximate position but misses the cube or doesn't close gripper in time
- **10% of trials:** Full success — grasp, lift, task complete

The model understands the task structure (approach → grasp → lift) but lacks precise spatial reasoning for reliable grasping at the correct z-offset.

---

## Head-to-Head Comparison

| | RL (PPO) | Code-as-Policy (Qwen-7B) |
|---|---|---|
| **Success rate** | **100%** | 10% |
| Training required | Yes (~4M steps, ~15 min) | No |
| GPU usage | Heavy (512 parallel envs) | None (CPU sim + CPU LLM) |
| Generalizes to new positions | Yes (randomized per episode) | Yes (reads pose each time) |
| Generalizes to new object shapes | Needs retraining | Potentially zero-shot |
| Cost per trial | ~0 (after training) | ~27 sec LLM inference |
| Interpretable | No (neural net) | Yes (readable Python code) |
| Sim used | ManiSkill/SAPIEN | Robosuite/MuJoCo |

---

## Repo Structure

```
sushkon-hwswcodes/
│
├── vlaTraining/                    # ManiSkill + RL experiments
│   ├── mani_skill/                 # Full ManiSkill source (forked)
│   ├── examples/baselines/ppo/
│   │   ├── ppo.py                  # PPO training script
│   │   └── runs/PickCube-v1__ppo__1__*/
│   │       ├── ckpt_161.pt         # Best checkpoint (100% success)
│   │       └── videos/16.mp4       # Eval video at epoch 161
│   ├── mani_skill/envs/tasks/tabletop/
│   │   └── pick_cube_replicacad.py # Custom ReplicaCAD env
│   ├── auto_commit.sh              # Auto-commits every 17 min
│   ├── push_checkpoints.sh         # Force-push checkpoints to git
│   ├── setup_gpu_machine.sh        # One-time setup
│   └── start_training.sh           # Training launcher
│
└── llmandRobots/                   # Code-as-Policy experiments (forked CaP-X)
    ├── capx/
    │   ├── llm/client.py           # Patched: Ollama routing added
    │   └── envs/simulators/
    │       └── robosuite_base.py   # Patched: robosuite 1.5.2 compat
    ├── env_configs/
    │   └── cube_lifting_qwen/
    │       └── franka_qwen_privileged.yaml  # Our task config
    ├── outputs/                    # Trial results, videos, generated code
    │   └── ollama_qwen2.5-coder:7b-instruct-q4_K_M/
    │       └── qwen_cube_lifting_privileged/
    │           ├── trial_06_*/video_turn_00.mp4  # Success video
    │           ├── trial_08_*/video_turn_00.mp4  # Success video
    │           └── ...
    └── benchmark_20trials.log      # Full 20-trial run log
```

---

## What's Working

- [x] ManiSkill + PPO pipeline (GPU-accelerated, 512 parallel envs)
- [x] Custom PickCubeReplicaCAD-v1 environment
- [x] Qwen2.5-Coder-7B running locally via Ollama (no cloud API)
- [x] CaP-X integrated with Ollama (no proxy needed)
- [x] Full trial pipeline: env reset → LLM code gen → execution → video → metrics
- [x] 20-trial benchmark complete
- [x] Both repos on GitHub, auto-commits running

## What's Not Done Yet

- [ ] Multi-turn error recovery (retry on failure — likely biggest win for CaP-X)
- [ ] Shape/size generalization task (core research contribution)
- [ ] RL vs CaP-X on same simulator (currently different sims)
- [ ] Failure mode analysis (reading the generated code from failed trials)
- [ ] CaP-X with larger LLM for comparison (frontier model upper bound)
- [ ] PickCubeReplicaCAD-v1 training run (harder task)

---

## Immediate Open Questions

1. **How much does multi-turn recovery help?** CaP-X supports it (M-tier configs exist). Enabling it on the Qwen config is ~1 line change. Could easily push 10% → 30%+.

2. **What exactly is Qwen failing at?** Need to read the generated code from failed trials and categorize: wrong API call, wrong coordinates, wrong grasp approach, gripper timing?

3. **How does Qwen compare to a frontier model on this task?** Running the same 20 trials with GPT-4o or Claude would give an upper bound on what CaP-X can achieve on this task.

4. **Shape/size generalization:** Robosuite supports randomizing object geometry. How do we expose shape info to the LLM, and does it generalize?

5. **Same simulator for fair comparison:** RL was trained in ManiSkill, CaP-X runs in Robosuite. For a clean comparison, both should be on the same sim.
