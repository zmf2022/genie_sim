---
name: moveit-wbc
description: >
  Bring up MoveIt 2 + whole-body-control RViz for the Genie G2 family,
  using `ros2 launch genie_sim_moveit wbc.launch.py`. Covers
  arm × gripper selection, the three packaged IK plugins (KDL-coupled,
  bio_ik-coupled, relaxed-IK), the GenieBioIK human-prior A/B switch,
  and the `use_ros2_control:=false` mode for direct `/joint_command`
  driving.
  Trigger: When the user asks to "start moveit", "wbc rviz", "plan with
  moveit", "switch IK plugin", "drive joint_command directly",
  "compare human priors", or pairs MoveIt with a `scene_*_g2_*`
  launched by the `launch-scene` skill.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_ros:launch-scene       # a `scene_*_g2_*` must be running
inputs:
  - name: arm
    desc: "arm variant (`crs` / `crsB`)"
    required: false
    default: crsB
  - name: gripper
    desc: "gripper variant (`swiftpicker` / `omnipicker` / `none`)"
    required: false
    default: swiftpicker
  - name: use_ros2_control
    desc: "`false` to drive via /joint_command directly (move_group only)"
    required: false
    default: "true"
  - name: use_human_priors
    desc: GenieBioIK priors on (default) vs vanilla BioIK
    required: false
    default: "true"
outputs:
  - desc: "`move_group` + `robot_state_publisher` + `rviz2` (MoveIt-Plan-Execute panel) up; optionally ros2_control controllers spawned"
---

## When to Use

- A `scene_*_g2_{op,sp}` is running (via the `launch-scene` skill) and
  the user wants planning + RViz on top.
- User wants to switch the active IK plugin or A/B the GenieBioIK
  "human priors" (torso-straight, chassis-pin, head LookAt).
- User wants to drive the arm directly via `/joint_command` (e.g.
  `wok_flip_cmds.py`) without ros2_control fighting the publisher.

Do **not** use for:
- Robots other than Genie G2 — the SRDF, coupled-joint constraints,
  and ros2_controllers manifest are G2-only. Other robots run on the
  engine but don't have a packaged MoveIt config.
- Launching the simulator itself → `launch-scene` skill.
- Building the workspace → `build-workspace` skill.

## Critical Patterns

1. **`arm` and `gripper` are independent.** Defaults are
   `arm:=crsB gripper:=swiftpicker`. The launch wires both into the
   URDF filename and the SRDF xacro mappings, so any
   `(arm, gripper)` combo that has a corresponding URDF under
   `genie_sim_robot_model/urdf/` works.
2. **Match MoveIt's `(arm, gripper)` to the scene's robot variant.**
   `scene_pnp_g2_op` → `gripper:=omnipicker`,
   `scene_wbc_g2_sp` → `gripper:=swiftpicker`. Mismatched values
   produce a URDF MoveIt can plan against but a robot the engine
   refuses to drive.
3. **Pick the IK plugin via `use_human_priors`, not by editing yaml.**
   `use_human_priors:=true` (default) loads
   `config/kinematics.yaml` (GenieBioIK with priors);
   `use_human_priors:=false` loads `config/kinematics_vanilla.yaml`
   (upstream BioIK, every prior weight zeroed). Both files share the
   plugin selection / IK links / timeouts so the only A/B variable is
   the goal stack.
4. **`use_ros2_control:=false` for `/joint_command` driving.** When an
   external publisher (teleop, wok-flip script) writes
   `/joint_command`, the ros2_control node would republish at the CM
   update rate and fight it — set this flag to leave `move_group`
   alone (the `/compute_ik`, `/compute_fk` services) so there's a
   single source on the topic.

## The matrix

| Scene (from `launch-scene`) | Recommended `wbc.launch.py` args |
|---|---|
| `scene_pnp_g2_op` | `arm:=crsB gripper:=omnipicker` |
| `scene_wbc_g2_sp` | `arm:=crsB gripper:=swiftpicker` (defaults) |
| `scene_flat_g2_sp` / `_chef` / `_shirt` | defaults (`crsB` + `swiftpicker`) |

| IK plugin / mode | Args |
|---|---|
| GenieBioIK with human priors (default) | `use_human_priors:=true` |
| Upstream BioIK (priors zeroed, for A/B) | `use_human_priors:=false` |
| `move_group` only — `/compute_ik` services, no controllers | `use_ros2_control:=false` |

## Workflow

### Step 1 — Confirm the scene is up

`launch-scene` skill should already be running in another shell.
Verify:

```bash
ros2 topic list | grep -E "joint_states|tf|clock"
ros2 topic hz /clock                    # sim-time pacing should be live
```

### Step 2 — Pick `arm` + `gripper`

Ask via `AskUserQuestion` if not given. Default to the scene's
gripper (see matrix above).

### Step 3 — Launch MoveIt

```bash
source devel/setup.bash

# Default (Genie G2 + crsB + swiftpicker + human priors + ros2_control):
ros2 launch genie_sim_moveit wbc.launch.py

# crs arm + omnipicker:
ros2 launch genie_sim_moveit wbc.launch.py arm:=crs gripper:=omnipicker

# A/B vanilla BioIK (priors disabled):
ros2 launch genie_sim_moveit wbc.launch.py use_human_priors:=false

# Direct /joint_command (e.g. external teleop / wok_flip_cmds.py):
ros2 launch genie_sim_moveit wbc.launch.py use_ros2_control:=false
```

### Step 4 — Plan in RViz

The launch starts `move_group`, `robot_state_publisher`, and
`rviz2` with the MoveIt-Plan-Execute panel. Drag the interactive
marker on the IK link, hit **Plan**, then **Execute**.

### Step 5 — (optional) Drive the chassis as a single chain

`simple_body_controller` claims the same body-joint position
interfaces as `simple_waist_controller` + `simple_torso_controller`,
so it's configured but not auto-spawned. To drive the body as one
chain:

```bash
ros2 control switch_controllers \
  --activate simple_body_controller \
  --deactivate simple_waist_controller simple_torso_controller
```

## Commands (copy-paste summary for the user)

```bash
# Inside the container, after sourcing the overlay:
source devel/setup.bash

# Most common: match a pnp scene
ros2 launch genie_sim_moveit wbc.launch.py arm:=crsB gripper:=omnipicker

# A/B the human priors
ros2 launch genie_sim_moveit wbc.launch.py use_human_priors:=false

# move_group only (for external /joint_command publishers)
ros2 launch genie_sim_moveit wbc.launch.py use_ros2_control:=false
```

## Notes

- The MoveIt URDF joint limits are widened by ±0.01 rad / ±0.001 m
  at launch time to absorb MuJoCo soft-limit drift —
  `CheckStartStateBounds` reads URDF limits directly, so the
  per-pipeline `start_state_max_bounds_error` doesn't help. Helper
  lives in `genie_sim_robot_model.urdf_utils.pad_urdf_joint_limits`.
- Default workspace half-extent for OMPL planar-joint bounds is set
  to 200 m via `ompl.default_workspace_bounds` — without this,
  `FixWorkspaceBounds` clamps `planar_joint/trans_{x,y}` to a narrow
  box that excludes the actual map → base_link pose.
- `fix_start_state:=true` lets jazzy's `CheckStartStateBounds` adapter
  actively renormalise joint values that drift past `<limit>` by
  sub-mrad under contact load (default upstream is false).
- The MoveIt config is **distro-agnostic** (Humble + Jazzy). Distro
  drift is absorbed by `moveit_compat.hpp` in
  `genie_sim_moveit_plugins/`, never by `#ifdef` in the config
  package.

## Resources

- **Launch file**: [source/geniesim_ros/src/ros_ws/src/genie_sim_moveit/launch/wbc.launch.py](../../src/ros_ws/src/genie_sim_moveit/launch/wbc.launch.py)
- **Per-gripper SRDF fragments**: [source/geniesim_ros/src/ros_ws/src/genie_sim_moveit/config/srdf_grippers/](../../src/ros_ws/src/genie_sim_moveit/config/srdf_grippers/)
- **IK plugins**: [source/geniesim_ros/src/ros_ws/src/genie_sim_moveit_plugins/](../../src/ros_ws/src/genie_sim_moveit_plugins/)
- **MoveIt config overview**: [source/geniesim_ros/src/ros_ws/src/genie_sim_moveit/AGENTS.md](../../src/ros_ws/src/genie_sim_moveit/AGENTS.md)
