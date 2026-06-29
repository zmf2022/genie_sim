# genie_sim_bringup

Launch orchestration and configuration for the GenieSim ROS 2 stack.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE)

**Maintenance contract**: when you add a launch arg, a new launcher YAML key,
or a new config file, update this file in the same diff.

**Support tier**: G2 is the only Tier-1 robot in this stack.  Scene yamls,
launchers, and physics tuning are validated end-to-end on
`scene_flat_g2_*.yaml` + `launcher_newton_*.yaml` against the
`genie/g2/*` description.  Every other robot — `agilex/aloha`,
`agilex/piper`, `arx/x5`, `arx/acone`, `franka/fr3`,
`universal_robots/ur5` — is Tier 2: buildable but not continuously
validated through the bringup, drive-gain, and physics-tuning paths
described below.  See
[`genie_sim_robot_model/AGENTS.md → Support tiers`](../genie_sim_robot_model/AGENTS.md#support-tiers)
for the full policy.

---

## Layout

```
genie_sim_bringup/
├── launch/
│   ├── app.launch.py              ← top-level entry point (requires launcher_config)
│   ├── physics_isaacsim.launch.py ← Isaac Sim physics sub-launch
│   ├── physics_mujoco.launch.py   ← MuJoCo physics sub-launch
│   └── utils.py                   ← shared launch helpers (pure I/O, no ROS state)
├── config/
│   ├── launcher_ovrtx_isaac_physx.yaml  ← Isaac Sim 5.1 + PhysX + OVRtx renderer (stable default)
│   ├── launcher_ovrtx_isaac_newton.yaml ← Isaac Sim 6.0 + Isaac's Newton wrapper + OVRtx (rigid only, experimental)
│   ├── launcher_newton_mjwarp.yaml      ← newton-standalone, MuJoCo-Warp rigid (no cloth)
│   ├── launcher_newton_mjvbd.yaml       ← newton-standalone, MuJoCo-Warp rigid + VBD cloth
│   ├── launcher_newton_mjxpbd.yaml      ← newton-standalone, MuJoCo-Warp rigid + XPBD cloth
│   ├── launcher_newton_fsvbd.yaml       ← newton-standalone, Featherstone rigid + VBD cloth
│   ├── launcher_newton_avbd.yaml        ← newton-standalone, Featherstone rigid + augmented-VBD cloth
│   └── scene_*.yaml                     ← scene configs (robot + cameras + optional newton + optional debug)
├── scripts/
│   ├── gripper_cmds.py                  ← test driver for /joint_command gripper setpoints
│   └── wbc_cmds.py                      ← whole-body-control command driver
└── rviz/                          ← RViz2 config files
```

> **Newton standalone is experimental** — `launcher_newton_*` and
> `launcher_ovrtx_isaac_newton` print a yellow `[ENTER]` gate before
> startup. The stable rigid-body path is `launcher_ovrtx_isaac_physx`.

---

## Launch architecture

### Entry point: `app.launch.py`

Requires `launcher_config` (no default — exits with a clear error if missing).
Reads the launcher YAML to determine the physics engine, then delegates to one
of two sub-launches:

| `launcher.physics.engine` | Sub-launch |
|---|---|
| `genie_sim_engine` (default) | `physics_isaacsim.launch.py` |
| `mujoco_geniesim` | `physics_mujoco.launch.py` |

Common args declared here and forwarded to sub-launches:
`scene`, `physics_hz`, `render_hz`, `headless`, `always_regenerate_robot_usd`,
`physics_engine`, `physics_solver`, `robot_model`, `body`, `arm`, `gripper`,
`use_sim_time`, `remap_tf`, `fake_slam`, `log_level`, `interaction_tools`,
`launcher_config`.

### `physics_isaacsim.launch.py`

Owns the Isaac Sim physics pipeline. Declares its own args:
`physics_hz`, `render_hz`, `headless`, `always_regenerate_robot_usd`,
`physics_engine` (`isaac_physx` | `isaac_newton` | `newton`),
`physics_solver` (`mujoco-warp` — the only supported value; pinned at
`choices=["mujoco-warp"]` so wrong values can't be typed. Honored by
`isaac_newton`; ignored by `isaac_physx` (PhysX has its own scheme) and
`newton` (newton-standalone hardcodes Featherstone for rigid and reads
its cloth solver from the scene yaml's `newton.solver.prefer`)),
`physics_solver_substep`, `physics_solver_iterations`, `render_mode`
(`raster` | `pathtrace` | `offline`).

Orchestration order (all gated behind the assemble pipeline):
```
[assemble_robot?] → [assemble_scene] → [assemble_newton] → [genie_sim_engine + industrial_bridge? + renders?]
```

`assemble_newton` is unconditionally chained (after the
launcher-yaml/CLI simplification) — it self-gates on whether the scene
yaml carries a non-empty `newton.entries` block, so an
`isaac_physx`-only scene exits the stage in milliseconds without
writing anything.

`physics_engine` / `physics_solver` / `physics_solver_substep` /
`physics_solver_iterations` / `render_mode` are forwarded as Node
params to the active `genie_sim_engine_*.py` script — **do not rename
these keys** without updating `runtime.bootstrap.py` and the engine
entry points (`genie_sim_engine_isaacsim.py` / `genie_sim_engine_newton.py`).

### `utils.py`

Pure helper module — no ROS state, no side effects. Key functions:

| Function | Purpose |
|---|---|
| `perform(context, name)` | Safe `LaunchConfiguration` resolver |
| `resolve_scene_yaml_robot_params()` | Merge scene YAML robot block with CLI overrides |
| `resolve_bringup_config_file()` | Resolve a config name to an absolute path |
| `load_launcher_yaml()` | Parse launcher YAML → `(launcher_section, plug_yaml, params_path)` |
| `resolve_physics_engine()` | Extract ROS engine id / package / executable from launcher YAML |
| `make_assemble_pipeline()` | Build the `assemble_robot → assemble_scene → assemble_newton → runtime` chain |
| `make_industrial_bridge_node()` | Construct the industrial_bridge Node |
| `make_render_ovrtx_node()` | Construct the render_ovrtx Node |
| `make_render_isaacsim_node()` | Construct the render_isaacsim Node |
| `resolve_active_renders()` | Filter render list to installed packages |
| `resolve_assets_folder()` | Resolve `ASSETS_PATH` from `geniesim_assets` |

---

## Launcher YAML schema

```yaml
launcher:
  physics:
    engine: genie_sim_engine        # ROS engine id
    engines:
      genie_sim_engine:
        package: genie_sim_engine
        executable: genie_sim_engine_isaacsim.py
        name: genie_sim_engine
    industrial_bridge: ""           # optional node name
  renders:
    - render_ovrtx                  # or render_isaacsim

render_ovrtx:                       # params block for the render node
  ros__parameters:
    plugin:
      - genie_sim_render/RosImagePublisherPlugin
```

`physics_engine` / `physics_solver` / `physics_solver_substep` /
`physics_solver_iterations` / `render_mode` are **launch CLI args
with launcher-YAML defaults** — each launcher YAML's
`genie_sim_engine.ros__parameters` block sets the engine-appropriate
defaults (e.g. `launcher_newton_fsvbd.yaml` defaults to
`physics_engine: newton, physics_hz: 60, physics_solver_substep: 10,
physics_solver_iterations: 5`), and any `key:=value` on the
`ros2 launch` CLI overrides on top. Precedence: CLI > launcher YAML >
`DeclareLaunchArgument` default. See `_OVERRIDABLE_FROM_YAML` in
[physics_isaacsim.launch.py](launch/physics_isaacsim.launch.py) for
the full overridable set.

---

## Scene YAML robot block

```yaml
robot:
  robot_model: genie
  body: default
  arm: default
  gripper: default
  robot_prefix: genie
  init_joint_pos:
    arm_l_joint1: -45.0   # degrees for revolute, metres for prismatic
  robot_source:
    package: genie_sim_robot_model
    urdf:                 # presence triggers URDF→USD route
      xacro_relpath: xacro/robot.xacro
```

URDF mimic joints are handled by the URDF→USD importer (PhysX
`PhysxMimicJointAPI` / Newton `NewtonMimicAPI`); there is no scene-yaml
`mimic:` block to author. The Newton paths (`isaac_newton` and
`newton`) additionally broadcast each master target across its
followers in software (`engine/_mimic.expand_targets`) — Newton's
`apply_action` only drives DOFs whose names are passed in. See
[genie_sim_engine/AGENTS.md → Mimic joints](../genie_sim_engine/AGENTS.md#mimic-joints--importer-authored-runtime-broadcast-on-newton)
for the design.

`init_joint_pos` is forwarded as `init_joint_pos_json` (JSON-encoded) to the
physics node — deliberately kept out of the stage manifest so editing it does
not invalidate the assemble cache.

---

## Scene YAML `newton:` block

Active under `physics_engine:=newton` (newton-standalone) or
`physics_engine:=isaac_newton` (Isaac's Newton wrapper). Ignored by
`isaac_physx`. Defines Newton-native particle features (cloth, soft
body, static box colliders) that the rigid-body USD bridge doesn't
cover.

`assemble_newton.py` reads this block, validates it, extracts mesh
geometry from the referenced USD files, and writes
`newton_solvers.json` into the stage directory. The runtime
cloth-inject hook reads that sidecar to materialize particles in the
Newton `ModelBuilder` between `add_usd` and `finalize`.
`assemble_newton.py` is unconditionally chained in the launch graph —
it self-gates on a non-empty `newton.entries` block, so scenes without
particle features pass through silently.

```yaml
newton:
  solver:
    # Required when entries contain cloth/particles. Pick explicitly:
    #   "vbd"     — unconditionally stable, garment-quality (newton-standalone only)
    #   "xpbd"    — PBD; the only particle solver the isaac_newton wrapper exposes
    #   "style3d" — higher-quality garment drape at greater per-step cost
    #               (newton-standalone only)
    # No "auto" — every path declares its solver explicitly so the yaml matches
    # the runtime.
    prefer: vbd

  contact:
    # Cloth ↔ rigid contact stiffness/damping/friction.
    # Model-global in Newton 1.0.0 — no per-entry override yet.
    # VBD-only — XPBD ignores these (kept for cross-launcher parity).
    soft_ke: 1.0e4
    soft_kd: 1.0e-2
    soft_mu: 0.25

  entries:
    - kind: cloth                    # required; "cloth" or "box"
      name: tshirt                   # required; unique per scene — used as the USD prim
                                     # name and as the runtime particle-writeback key
      mesh_usd: scenes/blank/Female_T_Shirt.usd
                                     # relative to --base-path (assets folder), or absolute
      pose: [tx, ty, tz, qx, qy, qz, qw]  # spawn pose in world frame; default: identity
      vel:  [vx, vy, vz]             # initial linear velocity; default: [0,0,0]
      params:
        density:        0.02         # kg/m^2
        tri_ke:         1.0e4        # stretch stiffness
        tri_ka:         1.0e4        # area preservation
        tri_kd:         1.5e-6       # stretch damping
        edge_ke:        5.0          # bend stiffness
        edge_kd:        1.0e-2       # bend damping
        particle_radius: 0.008       # metres; contact + self-collision offset
```

All `params` keys for `kind: cloth` are **required** — `assemble_newton`
exits non-zero with a clear error if any is missing. `solver` and
`contact` sub-blocks are optional (shown defaults apply when absent).

**Static / dynamic split** (same rule as the rest of the manifest):

| Field | Baked into `newton_solvers.json` | Live from scene yaml at runtime |
|---|---|---|
| vertices + indices (from `mesh_usd`) | yes — extracted once | no |
| `pose` / `vel` | yes | no |
| `params` (density, stiffness, …) | **no** | **yes — re-read on every launch** |
| `contact` block | advisory only | re-read live |

Iterating cloth stiffness / damping doesn't require re-running
`assemble_newton` — change the yaml and relaunch. The geometry itself
(vertex positions, topology) is baked; changing `mesh_usd` requires
deleting the stage cache.

---

## Routing rules

- Launch arg declarations → `app.launch.py` (common) or `physics_isaacsim.launch.py` (own)
- Launcher YAML parsing → `utils.py:load_launcher_yaml`
- Scene YAML parsing → `utils.py:resolve_scene_yaml_robot_params`
- Assemble pipeline wiring → `utils.py:make_assemble_pipeline` (always chains `assemble_newton`; the script self-gates on a non-empty scene `newton.entries` block)
- Physics engine resolution → `utils.py:resolve_physics_engine`
- Newton extras assembly → `genie_sim_engine/scripts/assemble_newton.py`
- Gripper test commands → `scripts/gripper_cmds.py`
