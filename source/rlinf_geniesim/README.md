# Genie Sim RL — Reinforcement Learning with Genie Sim Simulation

This module connects [Genie Sim](https://github.com/AgibotTech/genie_sim) with
[RLinf](https://github.com/RLinf/RLinf) for robot reinforcement learning,
featuring **Isaac Sim + MuJoCo dual-simulator** architecture and
**SpaceMouse human-in-the-loop** training.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  RLinf (training framework)                                          │
│    Task Env (e.g. PlaceWorkpieceEnv)                                 │
│      ├── _extract_states()   52-dim SHM state → 26-dim model state   │
│      ├── _expand_actions()    7-dim model action → 14-dim SHM action │
│      └── _compute_reward()   dense reward from body poses            │
│    GenieSimBaseEnv  →  GenieSimShmClient                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  Shared Memory (SHM)
                           │  ├── Frame SHM  (camera images)
                           │  ├── Ctrl SHM   (per-env state/action/info)
                           │  └── Step SHM   (request-reply sync)
┌──────────────────────────┴───────────────────────────────────────────┐
│  sim_server.py  (Genie Sim side)                                      │
│    GenieSimVectorEnv  ← manages MuJoCo lifecycle + signal-based sync │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐       ┌──────────────────────┐  │
│  │ MuJoCo env_0 │   │ MuJoCo env_1 │  ...  │ Isaac Sim renderer   │  │
│  │ 1000 Hz      │   │ 1000 Hz      │       │ 30 Hz (GridCloner)   │  │
│  └──────────────┘   └──────────────┘       └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

- **MuJoCo** handles physics at 1000 Hz per environment — one process per env, isolated by ROS 2 namespace
- **Isaac Sim** provides photo-realistic rendering via `GridCloner`
- **Shared memory (SHM)** is the only data channel between RLinf and sim (zero-copy for camera images)
- **EE control mode** — IK (damped Jacobian) runs inside MuJoCo; actions are delta EE pose targets
- **Task Env pattern** — SHM transports full-dimensional state/action; each task maps to/from a smaller model space via `_extract_states()` / `_expand_actions()`

### Reward Design

The `place_workpiece` task uses three reward components:

| Component | Description |
|-----------|-------------|
| `r_alive` | Exponentially decaying reward based on 3D distance and orientation error to target |
| `r_below` | Penalty when workpiece drops below target height |
| `r_success` | Sparse one-time reward when workpiece is placed at target and held still |

---

## Quick Start

### Prerequisites

- NVIDIA GPU (RTX 3090+, VRAM ≥ 24GB)
- Docker with NVIDIA Container Toolkit
- 3Dconnexion SpaceMouse (for data collection)

### 1. Clone Repositories

```bash
mkdir workspace && cd workspace
git clone https://github.com/AgibotTech/genie_sim.git
git clone -b dev/geniesim https://github.com/RLinf/RLinf.git
```

For Genie Sim installation and asset downloads, refer to the
[Genie Sim documentation](https://agibot-world.com/sim-evaluation/docs/#/v3).

#### Scene assets (`geniesim_assets`)

Scene/robot assets ship as the separate **`geniesim_assets`** pip package. Make
your checkout available at `source/rlinf_geniesim/assets/` (`.gitignore`d) with a
bind-mount — the container entrypoint editable-installs it on start:

```bash
sudo mount --bind /path/to/geniesim_assets \
  genie_sim/source/rlinf_geniesim/assets
```

> Use `mount --bind` (not a symlink). Undo with `sudo umount genie_sim/source/rlinf_geniesim/assets`.

### 2. Build Docker Images

Two layers: the **GenieSim base** (built with the `geniesim` client) and the
**RLinf training image** on top.

**Step 1 — Build the GenieSim base image** with the `geniesim` client (install
the CLI per the repo [README](../../README.md) § 3.1):

```bash
geniesim docker build      # → registry.agibot.com/genie-sim/geniesim3:latest
```

**Step 2 — Build the RLinf integration image** (Genie Sim + Isaac Sim + MuJoCo +
ROS 2 → `geniesim-rlinf:latest`):

```bash
cd workspace
bash genie_sim/source/rlinf_geniesim/scripts/build_geniesim_rlinf_image.sh
```

**Step 3 — Build the training image** (RLinf + PyTorch + training deps →
`geniesim-rlinf-train:latest`):

```bash
cd RLinf
docker build \
  --build-arg BUILD_TARGET=embodied-geniesim \
  -t geniesim-rlinf-train:latest \
  -f docker/Dockerfile \
  .
```

**Step 4 — Verify** GPU access in the final image:

```bash
docker run --rm --gpus all geniesim-rlinf-train:latest nvidia-smi
```

### 3. Download Pretrained Weights

```bash
cd RLinf/examples/embodiment/config
# For mainland China: export HF_ENDPOINT=https://hf-mirror.com
hf download RLinf/RLinf-ResNet10-pretrained --local-dir .
```

### 4. Collect Demonstrations

Connect the SpaceMouse via USB, then:

```bash
cd workspace
bash RLinf/rlinf/envs/geniesim/scripts/run.sh collect --num-demos 50
```

SpaceMouse controls:

| Action | Effect |
|--------|--------|
| Translate device | Move right arm end-effector (x/y/z) |
| Rotate device | Rotate right arm end-effector (roll/pitch/yaw) |
| Press left button | Save demo → environment resets |
| Press right button | Discard demo → environment resets |

Demos are saved to `RLinf/sac_demo/` (the `--save-dir` default, inside the RLinf repo).

### 5. Convert Demonstrations

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh convert
```

### 6. Start Training

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train
```

During training, env_0 accepts real-time SpaceMouse intervention while remaining
environments are driven by the policy.

Override Hydra parameters:

```bash
# Adjust discount factor
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.gamma=0.97

# Adjust BC regularization
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.bc_coef=5.0
```

### 7. Monitor Training

```bash
tensorboard --logdir workspace/results/
```

Key metrics: `critic_loss`, `q_values`, `eval/success_rate`, `entropy`, `bc_loss`.

### 8. Debug Shell

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh shell
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `run.sh collect --num-demos N` | Collect N demonstrations |
| `run.sh convert` | Convert demos to replay buffer |
| `run.sh train` | Start SAC + SpaceMouse HIL training |
| `run.sh shell` | Interactive container shell |
| `run.sh help` | Show all commands |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `PermissionError` on `/dev/shm/geniesim_*` | Run `bash RLinf/rlinf/envs/geniesim/scripts/cleanup_stale.sh` |
| Stale `.geniesim_idle` causing hang | Same cleanup script above |
| Isaac Sim startup timeout | Increase `startup_timeout_sec` in env YAML |
| GPU out of memory | Reduce `env.train.total_num_envs` via Hydra override |

## License

Mozilla Public License Version 2.0 — see `LICENSE` in the repository root.
