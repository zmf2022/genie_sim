# genie_sim_engine

The physics engine. Drives the simulator's body / joint state, exposes
it to ROS 2 via a pybind C++ bridge, and supports three physics
backends behind a single launch surface.

## What's in here

- **Three physics backends, picked via `physics_engine:=…`**:
  - `isaac_physx` — PhysX 5 inside Isaac Sim. The stable rigid path.
  - `isaac_newton` — Isaac Sim's Newton wrapper. Rigid-only,
    experimental.
  - `newton` — Newton-standalone. Kit-free; the only backend that
    supports cloth + soft body alongside rigid.
- **Two entry points** that share the same scene assets:
  - `scripts/genie_sim_engine_isaacsim.py` for the Isaac Sim engines.
  - `scripts/genie_sim_engine_newton.py` for Newton-standalone.
- **A C++ ROS 2 bridge** (`gsi::RosBridge`) that publishes `/clock`,
  `/joint_states`, `/tf`, `/odom`, and the camera topics.
- **An assemble pipeline** (`scripts/assemble_*.py`) that converts
  URDF → USD, stages cameras + render products into a manifest, and
  caches expensive steps so the second launch is fast.

## Quick check

```bash
# Confirm the engine package is built and discoverable
ros2 pkg list | grep genie_sim_engine
```

The engine itself is launched indirectly — via `genie_sim_bringup`'s
`app.launch.py` with a `launcher_config:=…`.

## When you'd touch this package

- Tuning physics, drives, contact, mass, friction, or mimic-joint
  behaviour.
- Adding a debug publisher, a new visualizer, or extending the
  Newton-standalone build pipeline (`scripts/engine/newton/setup/`).
- Changing the assemble pipeline (URDF→USD, manifest schema, robot
  caching).
- Working on cloth / soft-body solver wiring.

## Mechanics & deep-dives

- [AGENTS.md](AGENTS.md) — routing rules + dispatch invariants
  (URDF→USD per Isaac Sim version, mimic joints, asset layouts, gotchas).
- [docs/](docs/) — engine-internal deep-dives:
  [engines.md](docs/engines.md),
  [pipeline.md](docs/pipeline.md),
  [asset_consumption.md](docs/asset_consumption.md),
  [param_injection_eval.md](docs/param_injection_eval.md),
  [perf.md](docs/perf.md),
  [ovrtx_sync.md](docs/ovrtx_sync.md),
  [newton_quirks.md](docs/newton_quirks.md).
