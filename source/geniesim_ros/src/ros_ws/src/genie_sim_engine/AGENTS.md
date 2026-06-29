# genie_sim_engine

C++ ROS 2 pybind bridge + Python physics entry points for the Genie Sim stack.

## Entry points

| Script | When to use |
|---|---|
| `genie_sim_engine_isaacsim.py` | Isaac Sim engines (`isaac_physx`, `isaac_newton`) |
| `genie_sim_engine_newton.py` | Newton-standalone (`physics_engine:=newton`), Kit-free by default |

## Engines

| `physics_engine` | Class | Notes |
|---|---|---|
| `isaac_physx` (default) | `IsaacPhysXEngine` (`kit/isaac_physx.py`) | PhysX 5 via `omni.physx`. Reference path. |
| `isaac_newton` | `IsaacNewtonEngine` (`kit/isaac_newton.py`) | Newton wrapper, MuJoCo-Warp rigid only, no cloth. |
| `newton` | `NewtonHeadlessEngine` (`engine/newton/engine.py`) | Newton-direct, Kit-free. Only cloth-capable engine. Supports `physics_solver:=mujoco-warp` or Featherstone. |

## Scripts layout

```
scripts/
├── common/          # Kit-free: early_params, params (+ EngineNodeParams), scene_config, loop, session
├── kit/             # Kit-requiring: bootstrap, stage, isaac_physx, isaac_newton
├── engine/
│   ├── base.py      # PhysicsEngine ABC + factory (PhysicsEngine.create)
│   ├── _mimic.py    # shared mimic-joint broadcast (parse_mimic, expand_targets)
│   └── newton/      # newton-standalone implementation (Kit-free)
│       ├── engine.py        # NewtonHeadlessEngine — the only concrete subclass
│       ├── engine_base.py   # _NewtonStandaloneBase: mixin composition + shared init
│       ├── setup/           # build-phase mixins (stage / model / normalize / solver / debug_pubs / init_pose / runtime)
│       └── visualizers/     # RViz markers (DeformableMarker, DeformablePointCloud, ObjectMarker) + InlineOvrtxVisualizer
├── genie_sim_engine_isaacsim.py
└── genie_sim_engine_newton.py
```

Build phases compose into `_NewtonStandaloneBase` via plain multiple
inheritance — no metaclasses, no auto-discovery. See
`engine/newton/setup/__init__.py` for the table-of-contents.

## Detailed docs

- [docs/pipeline.md](docs/pipeline.md) — assemble pipeline, stage cache, robot routes, manifest, collision policy, material overrides, fix_base, mimic joints
- [docs/engines.md](docs/engines.md) — engine comparison: solver, joint control, state readback, cloth, diagnostics, Option C USD seed
- [docs/asset_consumption.md](docs/asset_consumption.md) — how each engine path consumes the assembled `robot.usda` + payloads, runtime injections per path, when variant selection matters, validation tools per path
- [docs/param_injection_eval.md](docs/param_injection_eval.md) — evaluation: do `isaac_newton` and newton-standalone reuse the mujoco-warp param injection? (TL;DR: no — two parallel implementations; the USD overlay is the only shared bridge)
- [docs/perf.md](docs/perf.md) — stats output glossary, bottleneck signatures, tuning levers (PhysX GPU pipeline, render_hz, CPU affinity, SCHED_FIFO)
