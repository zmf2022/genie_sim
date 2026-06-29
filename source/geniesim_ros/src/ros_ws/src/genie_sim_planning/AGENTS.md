# genie_sim_planning

Python tooling + small CLI scripts for the GenieSim chassis pipeline
(teleop, navigation, follow_trajectory).

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_planning/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE) (PythonRobotics-derived LQR helpers under MIT — see [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md))

**Status: scheduled for refactor.** The C++ four-wheel-steering servo and
its supporting plugins / OSQP solvers / tracking differentiator already
moved out to [`../genie_sim_ros_control/genie_sim_controllers/`](../genie_sim_ros_control/genie_sim_controllers/) (ros2_control
controller form). What remains here is the Python side and a few demo
scripts; expect more code to migrate out as the refactor lands.

**Maintenance contract**: when you add a script or change one of the
helper modules, update this file in the same diff. Before adding new
**C++** servo code here, default to extending `genie_sim_controllers`
instead — this package is intentionally Python-only at this point.

---

## Layout

```
genie_sim_planning/
├── genie_sim_planning/             ← importable Python package
│   ├── __init__.py
│   ├── kinematics.py               ← geometry / IK helpers
│   ├── math_utils.py               ← vector / quaternion / SO(2) utilities
│   ├── path_utils.py               ← waypoint smoothing, resampling
│   ├── planner.py                  ← high-level planner glue
│   ├── qos.py                      ← shared QoS presets
│   └── tf_utils.py                 ← tf2 lookups + frame conversions
└── scripts/
    ├── simple_follow_trajectory.py ← FollowJointTrajectory demo driver
    ├── simple_navigation.py        ← /cmd_twist demo from Nav2 goals
    └── simple_move_base.py         ← bare /cmd_twist publisher (teleop-style)
```

---

## Where things moved

The **servo loop, MPC, ServoBase plugins, and OSQP solver glue** used to
live in this package as a standalone C++ node (`servo_node`). They are
now plugins under `genie_sim_controllers`:

| Old location (here, removed)     | New location |
|---|---|
| `src/servo_node.cpp` / `servo_core.cpp` | `../genie_sim_ros_control/genie_sim_controllers/src/` |
| `src/plugins/*.cpp` (ServoBase strategies) | `../genie_sim_ros_control/genie_sim_controllers/src/plugins/` |
| `src/solvers/osqp_*.cpp` | `../genie_sim_ros_control/genie_sim_controllers/src/solvers/` |
| `include/.../{servo_*,four_wheel_car_twist_problem,tracking_differentiator}.hpp` | `../genie_sim_ros_control/genie_sim_controllers/include/genie_sim_controllers/` |

Don't add new C++ chassis-servo code back into this package — extend
`genie_sim_controllers` instead.

---

## Routing rules

- Reusable Python helpers → `genie_sim_planning/`
- Demo / quick-and-dirty scripts → `scripts/`
- Real chassis servo (C++) → `../genie_sim_ros_control/genie_sim_controllers/`
- Planar-base trajectory dispatcher → `../genie_sim_ros_control/genie_sim_control/src/planar_base_controller.cpp`
