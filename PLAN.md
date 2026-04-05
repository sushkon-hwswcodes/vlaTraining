# Experiment Plan: Shape-Generalized Grasping via Code-as-Policy
**Created:** April 2026  
**Baseline:** 2/20 (10%) success on cube lifting, zero-shot, single-turn, Qwen2.5-Coder-7B

---

## Goal

Enable a robot arm to pick up objects of **arbitrary shapes and sizes** using a local LLM (Code-as-Policy), with no task-specific training. The LLM receives a description of the object and generates executable grasping code.

---

## Phase 1 — Fix the Baseline: Multi-Turn Recovery

**Problem:** 35% of current failures are code crashes (syntax errors, wrong API usage) that a second attempt would fix. 55% are grasp misses that error feedback could correct. Multi-turn is already built into CaP-X.

**What to do:**
1. Add `multi_turn_prompt` to `env_configs/cube_lifting_qwen/franka_qwen_privileged.yaml` (copy from `franka_robosuite_cube_lifting_multiturn.yaml`, keep `privileged: true` and `FrankaControlPrivilegedApi`)
2. Set `max_turns: 3` (LLM gets up to 3 attempts per episode)
3. Re-run 20-trial benchmark with identical setup
4. Record: success rate, turns-per-success, failure mode breakdown

**Expected outcome:** 30–50% success (up from 10%)

**Definition of done:** 20-trial run complete, results logged, compared to single-turn baseline.

---

## Phase 2 — Shape Generalization: Privileged Shape Description

**Problem:** The LLM currently receives only the object pose. It doesn't know if it's grasping a cube, cylinder, or sphere. Grasp geometry (approach axis, finger width, z-offset) depends on shape.

**What to do:**

### 2a. Extend the Robosuite environment
- Randomize object geometry per episode: `box`, `cylinder`, `sphere`
- Randomize size within a range (e.g., box: 2–6cm per side; cylinder: r=1.5–4cm, h=5–15cm)
- File to modify: `capx/envs/tasks/franka/franka_lift.py` — add shape/size sampling in `reset()`

### 2b. Extend the privileged state API
- Add `get_object_shape()` API call returning:
  ```python
  {
    "shape": "cylinder",        # "box" | "cylinder" | "sphere"
    "size": [r_m, h_m],         # shape-specific dimensions in meters
  }
  ```
- File to modify: `capx/envs/tasks/franka/franka_privileged_api.py` (or equivalent)

### 2c. Update the system prompt
- Explain shape semantics and how they affect grasp strategy:
  - Box: top-down grasp, finger gap = longest side + margin
  - Cylinder: side grasp from equatorial plane, finger gap = 2r + margin  
  - Sphere: top-down or side, finger gap = 2r + margin
- Provide example code per shape type in the prompt (few-shot)

### 2d. Create new config
- `env_configs/cube_lifting_qwen/franka_qwen_shape_generalization.yaml`
- Inherits multiturn setup from Phase 1

### 2e. Benchmark
- 30 trials: 10 per shape type (box, cylinder, sphere)
- Vary sizes within each type
- Metrics: success rate per shape, overall, vs. cube-only baseline

**Expected outcome:** Demonstrates that natural language shape description enables zero-shot grasp adaptation. Even partial success (e.g., 40% box, 20% cylinder, 10% sphere) is a publishable result showing the LLM generalizes.

**Definition of done:** 30-trial benchmark complete, per-shape breakdown analyzed.

---

## Phase 3 — Remove Privileged State: Vision Input

**Problem:** Privileged shape info is unrealistic. In the real world, the robot must infer shape from sensors.

**What to do:**
1. Swap Qwen2.5-Coder-7B for **Qwen2.5-VL-7B** (multimodal, fits in 8GB VRAM, available via Ollama)
2. Capture a rendered top-down + side view from Robosuite camera at episode start
3. Pass image + task description to LLM (no explicit shape metadata)
4. LLM infers shape visually and generates grasp code
5. Re-run same 30-trial mixed-shape benchmark

**Expected outcome:** Measures the cost of removing privileged perception. If VL model matches Phase 2, the pipeline is perception-independent.

**Definition of done:** 30-trial VL benchmark complete, compared to Phase 2 results.

---

## Phase 4 (Optional) — Upper Bound: Frontier Model Comparison

Run the same Phase 2 benchmark with a frontier model (GPT-4o or Claude via API) to establish the upper bound of what Code-as-Policy can achieve on this task with a stronger reasoner.

**Metrics:** Success rate vs. Qwen-7B at same task. Quantifies the capability gap between local 7B and frontier models.

---

## Phase 5 (Optional) — Same Simulator for RL vs. CaP-X Comparison

Currently: RL trained in ManiSkill/SAPIEN, CaP-X runs in Robosuite. For a clean comparison:
- Port the shape-generalization task to ManiSkill, or
- Port the RL baseline to Robosuite

This makes the head-to-head comparison (RL vs. CaP-X on arbitrary shapes) scientifically valid.

---

## Current Blockers

| Blocker | Phase | Status |
|---|---|---|
| No CUDA toolkit (nvcc) | — | Blocks SAM3, ContactGraspNet, curobo — **not needed for Phases 1–3** |
| Qwen2.5-VL-7B not yet pulled in Ollama | Phase 3 | Pull when Phase 2 is complete |
| Frontier model API key | Phase 4 | Deferred |

---

## Success Metrics Summary

| Phase | Metric | Baseline | Target |
|---|---|---|---|
| 1 — Multi-turn | Success rate, cube only | 10% | 30–50% |
| 2 — Shape generalization | Success rate, mixed shapes | N/A | >20% avg, measurable per-shape |
| 3 — Vision input | Success rate, mixed shapes, no privileged state | N/A | Within 10pp of Phase 2 |
| 4 — Frontier model | Success rate, mixed shapes | N/A | Upper bound reference |

---

## Immediate Next Step

**Phase 1:** Add multi-turn prompt to `franka_qwen_privileged.yaml` and re-run 20 trials.
