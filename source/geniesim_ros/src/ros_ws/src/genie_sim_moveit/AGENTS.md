# genie_sim_moveit

MoveIt 2 **configuration package** for the Genie G2 family — SRDF,
kinematics + planner configs, joint-limit overrides, ros2_control wiring,
and the canonical `move_group` / RViz / WBC launch files.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_moveit/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE)

**See [`README.md`](README.md) for the architectural overview** (planar
+ prismatic floating base for WBC, controllers ordering,
robot_description plumbing). This file only documents the agent-facing
routing rules; don't duplicate the README here.

**Maintenance contract**: when you add a launch file, change a controller
spawn order, or alter the `move_group` parameter surface, update both
`README.md` and this file in the same diff.

---

## Layout

```
genie_sim_moveit/
├── config/
│   ├── genie.urdf.xacro            ← top-level robot description (consumed by RSP)
│   ├── genie.srdf.xacro            ← planning groups + virtual_joint(s); includes per-gripper fragment via $(arg gripper)
│   ├── srdf_grippers/              ← per-gripper SRDF fragments (gripper_* groups, passive mimic joints, gripper-link disable_collisions)
│   │   ├── swiftpicker.srdf.xacro
│   │   └── omnipicker.srdf.xacro
│   ├── genie.ros2_control.xacro    ← ros2_control hardware/controller bindings
│   ├── moveit_controllers.yaml     ← MoveIt → controller_manager wiring
│   ├── ros2_controllers.yaml       ← controller_manager spawn manifest
│   ├── kinematics.yaml             ← active IK plugin (relaxed-IK by default)
│   ├── kinematics_vanilla.yaml     ←   alternate: stock KDL for A/B testing
│   ├── ompl_planning.yaml          ← OMPL pipeline tuning
│   ├── pilz_cartesian_limits.yaml  ← Pilz pipeline cartesian limits
│   ├── joint_limits.yaml           ← per-joint vel/accel overrides
│   ├── initial_positions.yaml      ← canonical start pose
│   └── moveit.rviz                 ← MoveIt-Plan-Execute RViz config
├── launch/
│   ├── wbc.launch.py               ← move_group + WBC-tuned RViz (default workflow)
│   ├── demo.launch.py              ← MoveIt setup-assistant style demo
│   ├── move_group.launch.py        ← bare move_group (no RViz)
│   ├── moveit_rviz.launch.py       ← RViz only
│   ├── rsp.launch.py               ← robot_state_publisher only
│   ├── spawn_controllers.launch.py ← controller_manager spawner
│   ├── joint_states_relay.launch.py ← joint_states bridge for the WBC graph
│   ├── warehouse_db.launch.py      ← MoveIt warehouse (motion library)
│   ├── setup_assistant.launch.py   ← MoveIt Setup Assistant re-entry
│   └── moveit_launch_utils.py      ← shared helpers (URDF/SRDF loading, params)
├── scripts/
│   └── moveit_joint_states_bridge.py ← stitches sim joint_states into MoveIt's view
└── README.md
```

---

## Cross-distro contract

This package is **distro-agnostic** (ROS 2 Humble + Jazzy). Distro-specific
header / API differences are absorbed by `moveit_compat.hpp` in
`../genie_sim_moveit_plugins/` — never `#ifdef` here. If a launch file or
yaml needs to switch on distro, the cleaner fix is almost always to push
the difference into the compat header.

---

## Robot scope

Targets **Genie G2** (`crs` / `crsB` arms × `omnipicker` / `swiftpicker`
grippers). The SRDF, kinematics, and coupled-joint constraints are tuned
for G2 — porting to another platform means a new `.srdf`, new
`coupled_constraints.yaml` in `../genie_sim_moveit_plugins/config/`, and
re-deriving `joint_limits.yaml` from URDF.

---

## Routing rules

- Default entry point → `launch/wbc.launch.py`
- Hardware/controller bindings → `config/genie.ros2_control.xacro`
- Active IK plugin selection → `config/kinematics.yaml`
- IK plugins themselves → `../genie_sim_moveit_plugins/`
- Robot URDF / mesh source → `../genie_sim_robot_model/`
- Controller plugins loaded by `controller_manager` → `../genie_sim_ros_control/`
- Architectural deep-dive → [`README.md`](README.md)
