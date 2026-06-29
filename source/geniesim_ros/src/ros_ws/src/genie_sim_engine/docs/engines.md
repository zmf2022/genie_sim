# Engine comparison

Three concrete `PhysicsEngine` implementations, selected by `physics_engine` ROS parameter.

## Engine paths

| `physics_engine` | Class | File | Purpose |
|---|---|---|---|
| `isaac_physx` (default) | `IsaacPhysXEngine` | [`kit/isaac_physx.py`](../scripts/kit/isaac_physx.py) | PhysX 5 via `omni.physx`. Reference path. |
| `isaac_newton` | `IsaacNewtonEngine` | [`kit/isaac_newton.py`](../scripts/kit/isaac_newton.py) | Newton wrapper via `isaacsim.physics.newton`. MuJoCo-Warp rigid only, no cloth. |
| `newton` | `NewtonHeadlessEngine` | [`engine/newton/`](../scripts/engine/newton/) | Newton-direct, Kit-free. The only cloth-capable engine. |

The factory `PhysicsEngine.create(...)` lives in [`engine/base.py`](../scripts/engine/base.py).
`isaac_physx` and `isaac_newton` require the isaacsim entry point; `newton` can run Kit-free.

`isaac_newton` note: `physics_solver` is pinned to `mujoco-warp` at
`DeclareLaunchArgument(choices=["mujoco-warp"])` — the wrapper's default `MuJoCoSolverConfig` is
what actually runs regardless. Wrong values can't be typed. For cloth, use `physics_engine:=newton`.

## Solver and stage

| Concern | `isaac_physx` | `isaac_newton` | `newton` |
|---|---|---|---|
| Solver | PhysX 5 | Newton — MuJoCo-Warp (wrapper default) | Newton — `SolverMuJoCo` (`physics_solver:=mujoco-warp`) or `SolverFeatherstone` (default; cloth-capable) |
| Stage management | `IsaacSimStage` (`kit/stage.py`) | `IsaacSimStage` | self-managed via `get_current_stage()` + `add_usd` |
| Articulation handle | `SingleArticulation` (numpy/torch backend) | `SingleArticulation` (warp backend) | `newton.Model` + `state_0/state_1` warp buffers |
| `step()` | `physx_sim.simulate(dt) + fetch_results()` | `physx_sim.simulate(dt) + fetch_results()` | manual substep loop + CUDA graph capture |
| `tick_extras` | no-op | no-op (Fabric writeback handled by wrapper extension) | cloth Fabric writeback + body-transform writeback every tick |

## Joint control

| Concern | `isaac_physx` | `isaac_newton` | `newton` |
|---|---|---|---|
| Control law | PhysX force-mode PD: `τ = kp·(q⋆−q) + kd·(qd⋆−qd)` | Newton wrapper accel-mode PD (same form, accel-unit gains) | Velocity injection: `joint_qd ← q⋆−q` — kinematic, no PD gains |
| Command path | `articulation.apply_action(ArticulationAction(...))` | same call → `_view_input` wraps numpy → `wp.array` | `_target_joint_pos.assign(numpy)` — in-place GPU memcpy |
| Mimic enforcement | `PhysxMimicJointAPI` rigid constraint at C++ solver | software broadcast via `engine/_mimic.expand_targets` | same shared helper |
| `init_joint_pos` | USD `drive:*:targetPosition` seed + `set_joint_positions` | same call surface, inputs via `_view_input` | writes `model.joint_q` + `control.joint_target_pos` directly |

## Option C — USD writes as model-build seed

`kit.stage._configure_drives` authors `drive:angular:physics:*` USD attributes for **all** engines:

- **PhysX boot seed**: `World.reset()` runs ~5 init ticks before `_init_articulation` sets the
  tensor handle. Without a seed PhysX uses the URDF importer defaults (kp=625, kd=0 for grippers)
  and the gripper droops for ~80 ms.
- **Newton-wrapper actuator-existence seed**: Newton's USD importer reads `stiffness/damping`
  at `ModelBuilder.finalize()` to decide whether to allocate a POSITION actuator. `stiffness=0,
  damping=0` → `EFFORT` mode → no position actuator → `apply_action`'s `joint_target_pos` has
  nothing to push against. Gripper-mimic followers need a Newton-only DriveAPI seed because the
  URDF→USD converter leaves them at `stiffness=0`.

The **runtime tensor handle** (`_physics_view.set_dof_stiffnesses` / `set_dof_dampings` /
`set_dof_max_efforts`) is the single source of truth for gain values at solve time on both
backends, called after `World.reset()` in `_init_articulation`.

One remaining `physics_engine == "isaac_newton"` branch in `_init_articulation`: Newton's
`Articulation.set_gains` hardcodes `device="cpu"` for indices; combining with Newton's CUDA
tensors from `get_dof_stiffnesses()` explodes. The workaround calls `_physics_view.set_dof_stiffnesses`
directly with a `torch.tensor([0], dtype=torch.int32, device="cuda:0")` index.

## State readback

The Newton wrapper does NOT write joint state back to USD `state:angular:physics:position`
attributes that PhysX writes — so `snapshot_joint_states` returns zeros under Newton.

| Concern | `isaac_physx` | `isaac_newton` | `newton` |
|---|---|---|---|
| Joint state source | USD attributes (PhysX writes every tick) | `IsaacSimStage.get_joint_states` reads `articulation.get_joint_positions()` via `_view_readback` | reads `state_0.joint_q` directly (warp) |
| Body transforms | USD-local from `_xform_to_xyzwxyz_local` | same as PhysX | world-space from `state_0.body_q`, `(x,y,z,qw,qx,qy,qz)` layout |
| Tensor format | numpy / torch | `wp.array` only — `_view_input` shim converts numpy → CUDA torch | warp arrays end-to-end |

## Cloth, render, viewport

| Concern | `isaac_physx` | `isaac_newton` | `newton` |
|---|---|---|---|
| Cloth | not supported | not supported — use `physics_engine:=newton` | `add_cloth_mesh` on `ModelBuilder`; VBD (default), XPBD, Style3D |
| Cloth sidecar | unused | unused | required when `newton.entries` is non-empty |
| Viewport / render mode | `configure_viewport_for_debug(render_mode=...)` | same | same — called after `_build`/`_warmup`/`_setup_usdrt` |
| Fabric writeback | wrapper extension auto-writes | wrapper extension auto-writes | engine writes cloth points + body transforms via dedicated kernels every tick |

## Diagnostics and CLI knobs

| Knob | `isaac_physx` | `isaac_newton` | `newton` |
|---|---|---|---|
| `render_mode` | wired via `configure_viewport_for_debug` | same | same |
| `physics_solver_substep` / `physics_solver_iterations` | ignored | writes `NewtonStage.cfg.num_substeps` and `cfg.solver_cfg.iterations` | passed to `SolverFeatherstone` or `SolverMuJoCo` directly |
| `physics_solver` | unused | pinned to `mujoco-warp`; value is accepted cosmetically | `mujoco-warp` → `SolverMuJoCo` (1 substep, no cloth); other → `SolverFeatherstone` (10 substeps, cloth via `newton.solver.prefer`) |
| NaN-on-blowup | not applicable | `IsaacSimStage.get_joint_states` once-shot guard | `_StateMixin.get_joint_states` once-shot guard |
| Stats | basic timings | same | `_StatsMixin` with sync-vs-copy split, render-tick breakdown |

## What's shared vs deliberately divergent

**Shared:**
- `engine/_mimic.py` — `parse_mimic(stage, logger)` and `expand_targets(cmd_positions, followers)`
  used by both Newton paths (PhysX calls it too; it's a no-op there since the constraint solver
  already glues followers).
- `kit/stage.py` — `IsaacSimStage` is reused by `isaac_physx` AND `isaac_newton`. Under
  Option C the only engine-conditional code is the `set_gains` CPU-pin workaround in
  `_init_articulation`.
- `common/loop.py` — `EngineRunLoop.spin(render_hook, exit_check)` is the shared step scaffold
  for both entry points.
- `common/session.py` — `EngineSession` owns all setup from physics-params loading through
  `set_topology` (manifest, scene yaml, `_core.init_ros/init_scheduler`, `PhysicsEngine.create`,
  `snapshot_body_transforms`). `SimpleLogger(prefix)` replaces the per-file `_SimpleLogger` class.
  Entry points define their render_hook closure then call `session.run(render_hook, exit_check)`.
  `session.post_step_hooks` accepts `Callable[[float], None]` callbacks executed after
  `tick_extras()` each tick — extend step behaviour without subclassing (IsaacLab EventManager
  pattern).
- `common/params.py` — `EngineNodeParams` (typed dataclass, `from_dict` classmethod) coerces the
  raw ROS `{str: str}` params dict so `EngineSession.__init__` only touches typed attributes
  (`ep.physics_hz`, `ep.fake_slam`, etc.).  Mirrors IsaacLab `@configclass`.

**Intentionally distinct:** solver step, cloth integration, render tick orchestration, stats.
These differ at the API level and abstracting them would cost more in indirection than the
dedupe is worth.
