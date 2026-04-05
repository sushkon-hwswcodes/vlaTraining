# Experiment Status
**Last updated:** April 2026  
**Repos:** [vlaTraining](https://github.com/sushkon-hwswcodes/vlaTraining) · [llmandRobots](https://github.com/sushkon-hwswcodes/llmandRobots)

---

## Current Status: Phase 2 in progress — benchmark completed once; rerun needed with corrected success metric

---

## Completed Work

### Baseline: Cube Lifting (PickCube-v1) — ManiSkill + PPO
- **Result:** 100% success at epoch 151, ~4M timesteps, ~15 min on RTX 3070
- **Checkpoint:** `examples/baselines/ppo/runs/PickCube-v1__ppo__1__*/ckpt_161.pt`
- **Video:** `videos/16.mp4`
- **Repo:** `sushkon-hwswcodes/vlaTraining`

### Phase 1: Cube Lifting Baseline — Code-as-Policy (CaP-X + Qwen2.5-Coder-7B)
All runs in `sushkon-hwswcodes/llmandRobots`, outputs under `outputs/ollama_qwen2.5-coder:7b-instruct-q4_K_M/`

| Run | Config change | Success | Notes |
|---|---|---|---|
| Single-turn | baseline | 2/20 (10%) | `benchmark_20trials.log` |
| Multi-turn | added multi_turn_prompt | 6/20 (30%) | `benchmark_multiturn.log` |
| Prompt v2 | forbid open_gripper + example | 17/20 (85%) | `benchmark_prompt_v2.log` |
| **Prompt v2 + temp=0.3** | **lower temperature** | **20/20 (100%)** | `benchmark_temp03.log` — **current baseline** |

**Key findings:**
- `open_gripper()` at end was dominant failure (35% of trials)
- Temperature 1.0 → 0.3 eliminated remaining failures
- Multi-turn recovery rarely needed once prompt was fixed

**Active config:** `env_configs/cube_lifting_qwen/franka_qwen_privileged.yaml` (prompt v2, multiturn enabled)  
**Run command:**
```bash
cd /root/vlaTraining/cap-x
export MUJOCO_GL=osmesa && export DISPLAY=:1 && export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PYTHONPATH=/root/vlaTraining/cap-x python3 -u -m capx.envs.launch \
  --config-path env_configs/cube_lifting_qwen/franka_qwen_privileged.yaml \
  --model "ollama/qwen2.5-coder:7b-instruct-q4_K_M" \
  --total-trials 20 --num-workers 1 --temperature 0.3
```

### Infrastructure improvements
- Added `frontview` overview camera to all trial videos (`video_turn_XX_overview.mp4`)
- Auto-commit cron every 17 min on vlaTraining

---

## In Progress: Phase 2 — Shape Generalization

**Goal:** Test whether Qwen-7B can adapt grasp code when told the object shape/size.  
**Objects:** Box, cylinder, ball — randomly chosen per episode  
**Method:** Privileged state — LLM calls `get_object_shape()` → adapts grasp z-offset

### Files created:
| File | Purpose |
|---|---|
| `capx/third_party/robosuite/robosuite/environments/manipulation/lift_shape.py` | LiftShape env — random shape per reset |
| `capx/envs/simulators/robosuite_shape_lift.py` | Low-level env wrapper using LiftShape |
| `capx/envs/simulators/__init__.py` | Registers `franka_robosuite_shape_lift_low_level` |
| `capx/integrations/franka/control_privileged.py` | Added `get_object_shape()`, fixed bbox extents |
| `env_configs/shape_generalization/franka_qwen_shape.yaml` | Phase 2 benchmark config |
| `capx/third_party/robosuite/robosuite/environments/manipulation/lift_shape.py` | Updated success check to require lift-from-reset + grasp contact |

### Phase 2 benchmark result (first run)
- **Run:** `benchmark_shape_generalization.log`
- **Result:** `6/30` task-complete (`20%`), average reward `0.236`
- **Issue identified:** Success condition in inherited Lift logic used fixed threshold (`object_center_z > table_z + 0.04`), which can overcount success for larger randomized shapes.
- **Fix applied:** In `LiftShape`, success now requires both:
  - object lifted by `>0.04m` above its own reset height, and
  - active grasp contact (`_check_grasp`) by the gripper.

### Next step: re-run benchmark with corrected success metric
```bash
cd /root/vlaTraining/cap-x
export MUJOCO_GL=osmesa && export DISPLAY=:1 && export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
PYTHONPATH=/root/vlaTraining/cap-x python3 -u -m capx.envs.launch \
  --config-path env_configs/shape_generalization/franka_qwen_shape.yaml \
  --model "ollama/qwen2.5-coder:7b-instruct-q4_K_M" \
  --total-trials 30 --num-workers 1 --temperature 0.3 | tee benchmark_shape_generalization_after_fix.log
```

---

## Planned: Phase 2.5 — Real-World Objects (Text Description)
Same Qwen-7B, same privileged-state setup, but LIBERO-PRO/HOPE dataset objects (soup cans, milk cartons, etc.) instead of geometric primitives. LLM uses named object knowledge + text dimensions.

## Planned: Phase 3 — Vision Input
Swap to Qwen2.5-VL (multimodal), pass rendered image, no text shape description. LLM infers shape visually.

## Planned: Phase 4 — Frontier Model Upper Bound
Same Phase 2 benchmark with GPT-4o or similar.

## Planned: Phase 5 — RL vs CaP-X Same Simulator
Port both to same sim for clean comparison.

---

## Blockers
| Blocker | Affects |
|---|---|
| No CUDA toolkit (nvcc) | SAM3, ContactGraspNet, curobo — not needed until Phase 3+ |
| Qwen2.5-VL not yet in Ollama | Phase 3 only |

---

## Environment Setup (if reconnecting)
```bash
# MuJoCo / Robosuite rendering
export MUJOCO_GL=osmesa
export DISPLAY=:1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Repos
# vlaTraining: /root/vlaTraining  (ManiSkill + RL)
# llmandRobots (cap-x): /root/vlaTraining/cap-x  (CaP-X + Code-as-Policy)

# Ollama (local LLM server) — should be running
ollama list  # check qwen2.5-coder:7b-instruct-q4_K_M is available
```
