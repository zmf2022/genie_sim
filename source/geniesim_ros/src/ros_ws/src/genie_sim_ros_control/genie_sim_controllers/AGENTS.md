# genie_sim_controllers

ros2_control **controller plugins** for GenieSim. Currently ships the
four-wheel-steering chassis servo and its supporting linear-MPC solvers
(OSQP) + tracking-differentiator smoother.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_ros_control/genie_sim_controllers/](.)
License: [Mozilla Public License Version 2.0](../../../../../LICENSE)

**Maintenance contract**: when you add or rename a controller plugin or a
servo strategy, update `plugin_*.xml`, the
`pluginlib_export_plugin_description_file(...)` call in `CMakeLists.txt`,
and this file in the same diff. New ServoBase subclasses go under
`src/plugins/` so the `pluginlib` discovery line in `servo_core.cpp` picks
them up.

---

## Layout

```
genie_sim_controllers/
├── include/genie_sim_controllers/
│   ├── chassis_servo_controller.hpp   ← ros2_control wrapper (ControllerInterface)
│   ├── servo_node.hpp                 ← standalone executable form (legacy)
│   ├── servo_core.hpp                 ← shared servo loop (used by both forms)
│   ├── servo_base.hpp                 ← ServoBase pluginlib API
│   ├── four_wheel_car_twist_problem.hpp ← MPC problem definition (4ws kinematics)
│   └── tracking_differentiator.hpp    ← reference-signal smoother
├── src/
│   ├── chassis_servo_controller.cpp   ← ControllerInterface implementation
│   ├── servo_node.cpp                 ← legacy standalone node
│   ├── servo_core.cpp                 ← /cmd_twist → /cmd_4ws solver loop
│   ├── main.cpp                       ← rclcpp main for the standalone node
│   ├── plugins/                       ← ServoBase strategies (pluginlib)
│   │   ├── general_servo.cpp
│   │   ├── optimal_servo.cpp          ← MPC (OSQP) path
│   │   ├── parking_servo.cpp
│   │   ├── selftest_servo.cpp
│   │   └── spin_servo.cpp
│   └── solvers/                       ← OSQP-backed quadratic-program glue
│       ├── osqp_solver.cpp
│       ├── problem.cpp
│       ├── quadratic_function.cpp
│       ├── relative_function.cpp
│       └── scalar_function.cpp
├── config/
│   ├── genie_g2.yaml                  ← G2 platform tuning (wheel base, gains)
│   └── genie_g2u.yaml                 ← G2-U variant tuning
├── launch/servo_4ws.launch.py         ← bring up the standalone servo_node form
├── scripts/teleop_joy.py              ← joystick driver publishing /cmd_twist
└── plugin_chassis_servo_controller.xml
```

---

## Plugins

### `genie_sim_controllers/ChassisServoController`
`base_class_type=controller_interface::ControllerInterface`

Four-wheel-steering chassis servo packaged as a **ros2_control controller**.
Doesn't claim any hardware interfaces — same topic-driven pattern as
`PlanarBaseController` (see `../genie_sim_control/`). Subscribes
`/cmd_twist` + `/joint_states`, runs the `servo_core` loop, publishes
`/cmd_4ws`.

### Internal: `ServoBase` strategies (`src/plugins/*.cpp`)

Pluginlib-discovered strategies inside the servo. `servo_core` selects one
at runtime based on the requested mode (`general` / `optimal` / `parking`
/ `spin` / `selftest`). `optimal_servo` is the MPC variant — it builds a
`FourWheelCarTwistProblem` and solves it with the OSQP glue under
`src/solvers/`.

---

## Two deployment forms

| Form | Entry point | When to use |
|---|---|---|
| ros2_control controller (default) | `chassis_servo_controller` plugin in `ros2_controllers.yaml` | Run alongside MoveIt / other controllers under the same `controller_manager`. |
| Standalone executable (legacy) | `servo_4ws.launch.py` → `servo_node` | Quick teleop / hardware bring-up without ros2_control overhead. |

Both forms share `servo_core` — fix bugs there, not in either wrapper.

---

## Routing rules

- ros2_control controller wrapper → `src/chassis_servo_controller.cpp`
- Servo loop (the actual control law) → `src/servo_core.cpp`
- ServoBase strategy plugins → `src/plugins/*.cpp`
- OSQP problem glue → `src/solvers/*.cpp`
- Platform tuning (gains, geometry) → `config/genie_g2*.yaml`
- `/cmd_twist` producer (P-controlled by base trajectory) → `../genie_sim_control/src/planar_base_controller.cpp`
