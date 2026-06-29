# genie_sim_moveit_plugins

MoveIt 2 motion-control plugins for GenieSim — three inverse-kinematics
plugins (KDL, bio_ik, relaxed-IK) plus an RRT-Connect / TOPP-RA planner.
Cross-distro (ROS 2 Humble ↔ Jazzy) via the compat shims under
`include/genie_sim_moveit_plugins/`.

> **Robot scope: Genie G2 only.** The IK plugins consume a coupled-joint
> constraint set tuned for the G2 platform — `config/coupled_constraints.yaml`
> references G2-specific joint names (`idx0X_body_jointN`, `idxXX_arm_{l,r}_jointN`)
> and the `A·q ≤ b` inequalities encode G2's mechanical coupling envelope.
> Loading these plugins on a different robot will either skip every entry
> (joint names not found in the active group — silent no-op) or, if joint
> names happen to collide, apply inequalities that don't match the new
> hardware. To port to another robot, supply a new `coupled_constraints.yaml`
> via the `coupled_constraints_file` parameter (or `relaxed_ik.coupled_constraints_file`
> for the relaxed-IK plugin).

## Layout

```
.
├── include/genie_sim_moveit_plugins/   # public headers (shared by all plugins)
│   ├── coupled_constraints.hpp         #   coupled-joint YAML loader + Aq≤b check
│   └── moveit_compat.hpp               #   Humble↔Jazzy moveit API/header shims
├── src/
│   ├── kdl_coupled/                    # plugin: KDL + coupled-joint guard
│   ├── bio_ik_coupled/                 # plugin: bio_ik + coupled-joint soft penalty
│   ├── relaxed_ik/                     # plugin: standalone DLS + nullspace tasks
│   └── genie_planner/                  # plugin: RRT-Connect + TOPP-RA
├── plugin_<name>.xml                   # pluginlib descriptors (one per plugin)
├── config/                             # default params consumed by the plugins
└── launch/                             # ros2 launch entrypoints (demo + benchmark)
```

Each plugin folder under `src/` owns its `.cpp`, any plugin-private headers, and
nothing else. Headers shared across plugins live under
`include/genie_sim_moveit_plugins/` and are the only files installed as the
package's public API.

## Plugins

| Folder              | XML                            | pluginlib class                                  | Base                          |
|---------------------|--------------------------------|--------------------------------------------------|-------------------------------|
| `kdl_coupled/`      | `plugin_kdl_coupled.xml`       | `genie_sim_moveit_plugins/KDLKinematicsPlugin`   | `kinematics::KinematicsBase`  |
| `bio_ik_coupled/`   | `plugin_bio_ik_coupled.xml`    | `genie_sim_moveit_plugins/BioIKPlugin`           | `kinematics::KinematicsBase`  |
| `relaxed_ik/`       | `plugin_relaxed_ik.xml`        | `genie_sim_moveit_plugins/GenieRelaxedIK`        | `kinematics::KinematicsBase`  |
| `genie_planner/`    | `plugin_genie_planner.xml`     | `genie_sim_moveit_plugins/GeniePlannerManager`   | `planning_interface::PlannerManager` |

### kdl_coupled
Thin subclass of MoveIt's stock KDL plugin. Installs an `IKCallbackFn` that
rejects samples violating the coupled-joint inequalities (`A·q ≤ b − margin`)
loaded from `config/coupled_constraints.yaml`. Use when you want KDL's
behaviour but need hard guarding of body-joint coupling.

### bio_ik_coupled
Loads upstream `bio_ik/BioIKKinematicsPlugin` via pluginlib and augments its
goal set:
* `CoupledBoundGoal` — soft quadratic penalty for `A·q − (b − margin)` violations
* `MinimalDisplacementGoal`, `CenterJointsGoal`, `AvoidJointLimitsGoal` — weighted
  regularizers (weights configurable via params)
* Same callback guard as kdl_coupled, applied after bio_ik returns.

### relaxed_ik
Standalone damped-least-squares solver with a nullspace-projected gradient
that bundles several soft tasks: bound penalty, coupled-joint penalty,
center-pull, velocity/acceleration damping, optional FCL-distance guard and
CoM balance check. Parameters live under `config/relaxed_ik_params.yaml`.

### genie_planner
`PlannerManager` exposing two contexts:
* RRT-Connect in joint space (`src/genie_planner/rrt_connect_planner.{hpp,cpp}`)
* TOPP-RA time parameterisation (`src/genie_planner/topp_ra.{hpp,cpp}`)

Used by the MoveIt move_group when the kinematics_plugin is `genie_planner`.

## Cross-distro compat (`moveit_compat.hpp`)

`include/genie_sim_moveit_plugins/moveit_compat.hpp` exposes two layers of
Humble ↔ Jazzy bridging, gated by the `GENIE_MOVEIT_USE_NEW_API` compile
definition (set by CMake when `$ENV{ROS_DISTRO}` is jazzy/kilted/rolling):

1. **`PlanningContext::solve()` API drift** — Jazzy returns `void`, Humble
   returns `bool`. Use `GENIE_MOVEIT_SOLVE_RT`, `GENIE_MOVEIT_SOLVE_OK`,
   `GENIE_MOVEIT_SOLVE_ERR` in override declarations.
2. **MoveIt header rename `.h → .hpp`** — Jazzy deprecates every `.h` form.
   Include moveit headers via the `MOVEIT_H_*` macros (e.g.
   `#include MOVEIT_H_KINEMATICS_BASE`); the macros route to `.hpp` on
   Jazzy+ and `.h` on Humble. Extend the macro list whenever a new
   moveit_core header is pulled in.

Field-name renames (e.g. `error_code_` → `error_code`) use the
`GENIE_MOVEIT_FIELD(name)` token-pasting macro.

## Build

```bash
colcon build --packages-select genie_sim_moveit_plugins
```

Produces four shared libraries under `lib/`:

```
libgenie_sim_moveit_plugins_kdl_coupled_plugin.so
libgenie_sim_moveit_plugins_bio_ik_coupled_plugin.so
libgenie_sim_moveit_plugins_relaxed_ik_plugin.so
libgenie_sim_moveit_plugins_planner_plugin.so
```

## Adding a new plugin

1. `mkdir src/<name>/` and drop the `.cpp` plus any private headers there.
2. Author `plugin_<name>.xml` at the package root with the pluginlib class
   record (`<library path="lib<...>">` / `<class name="..." type="..." …>`).
3. Add the `add_library(genie_<name>_plugin SHARED src/<name>/<...>.cpp)`
   block and the matching `pluginlib_export_plugin_description_file(<base>
   plugin_<name>.xml)` to `CMakeLists.txt`.
4. If the plugin needs a moveit header, include it via `MOVEIT_H_*` from
   `moveit_compat.hpp` rather than hard-coding `.h` or `.hpp`.
