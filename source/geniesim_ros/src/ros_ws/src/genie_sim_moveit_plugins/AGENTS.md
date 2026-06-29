# genie_sim_moveit_plugins

MoveIt 2 **motion-control plugins** for GenieSim — three inverse-kinematics
plugins (KDL-coupled, bio_ik-coupled, relaxed-IK) plus an RRT-Connect +
TOPP-RA planner.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_moveit_plugins/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE)

**See [`README.md`](README.md) for the per-plugin manual** (constraints
loaded, parameter surface, when to pick which). This file only documents
agent-facing routing.

**Maintenance contract**: when you add a plugin or change pluginlib class
naming, update `plugin_<name>.xml`, the
`pluginlib_export_plugin_description_file(...)` calls in `CMakeLists.txt`,
the `README.md` plugin table, and this file in the same diff. New plugins
go into their own subdirectory under `src/`.

---

## Layout

```
genie_sim_moveit_plugins/
├── include/genie_sim_moveit_plugins/
│   ├── coupled_constraints.hpp   ← coupled-joint YAML loader + A·q ≤ b check
│   └── moveit_compat.hpp         ← Humble↔Jazzy moveit shims (single point of #ifdef)
├── src/
│   ├── kdl_coupled/              ← KDL + hard coupled-joint guard
│   ├── bio_ik_coupled/           ← bio_ik + soft coupled-joint penalty
│   ├── relaxed_ik/               ← standalone DLS + nullspace tasks
│   └── genie_planner/            ← RRT-Connect + TOPP-RA planning pipeline
├── config/
│   ├── coupled_constraints.yaml  ← G2 coupled-joint A·q ≤ b (joint-name keyed)
│   ├── kinematics_relaxed.yaml   ← params for relaxed_ik plugin
│   ├── relaxed_ik_params.yaml    ← weight / tolerance presets
│   └── genie_planning.yaml       ← genie_planner tuning
├── plugin_kdl_coupled.xml
├── plugin_bio_ik_coupled.xml
├── plugin_relaxed_ik.xml
├── plugin_genie_planner.xml
└── README.md
```

---

## Plugins

| pluginlib class | Base | Folder |
|---|---|---|
| `genie_sim_moveit_plugins/KDLKinematicsPlugin` | `kinematics::KinematicsBase` | `src/kdl_coupled/` |
| `genie_sim_moveit_plugins/BioIKPlugin`         | `kinematics::KinematicsBase` | `src/bio_ik_coupled/` |
| `genie_sim_moveit_plugins/GenieRelaxedIK`       | `kinematics::KinematicsBase` | `src/relaxed_ik/` |
| `genie_sim_moveit_plugins/GeniePlannerManager`  | `planning_interface::PlannerManager` | `src/genie_planner/` |

The KDL / bio_ik / relaxed-IK plugins share a single coupled-joint policy
loader (`include/.../coupled_constraints.hpp`) — fix coupled-joint bugs
there, not in each plugin.

---

## Robot scope

The shipped `config/coupled_constraints.yaml` references **G2-specific**
joint names (`idx0X_body_jointN`, `idxXX_arm_{l,r}_jointN`) and the `A·q ≤ b`
inequalities encode G2's mechanical coupling envelope. Loading these
plugins on a different robot will silently no-op (joint names not found)
or apply mismatched inequalities (if names happen to collide). Port to
another platform by supplying a new yaml via the
`coupled_constraints_file` parameter (`relaxed_ik.coupled_constraints_file`
for the relaxed-IK plugin).

---

## Cross-distro contract

Every Humble ↔ Jazzy difference (field renames, `.h` vs `.hpp` headers,
`solve()` return-type differences) lives **in `moveit_compat.hpp`**. Any
`#ifdef ROS_DISTRO_*` outside that header is a bug — fix the header
instead.

---

## Routing rules

- Coupled-joint policy (shared) → `include/genie_sim_moveit_plugins/coupled_constraints.hpp`
- Distro shims (shared) → `include/genie_sim_moveit_plugins/moveit_compat.hpp`
- Per-plugin source → `src/<plugin>/`
- Plugin selection (which IK to use) → `../genie_sim_moveit/config/kinematics.yaml`
- Per-plugin manual → [`README.md`](README.md)
