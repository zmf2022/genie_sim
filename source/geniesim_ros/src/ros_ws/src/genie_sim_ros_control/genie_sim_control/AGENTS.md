# genie_sim_control

ros2_control **hardware-interface + controller plugins** that bridge MoveIt /
ros2_control trajectory execution to the simulated robot over ROS topics.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_ros_control/genie_sim_control/](.)
License: [Mozilla Public License Version 2.0](../../../../../LICENSE)

**Maintenance contract**: when you add or rename a plugin, update the
corresponding `plugin_*.xml`, the `pluginlib_export_plugin_description_file(...)`
call in `CMakeLists.txt`, and this file in the same diff.

---

## Layout

```
genie_sim_control/
├── include/genie_sim_control/
│   ├── genie_sim_robot_interface.hpp  ← SystemInterface (hardware bridge)
│   └── planar_base_controller.hpp     ← ControllerInterface (planar virtual_joint)
├── src/
│   ├── genie_sim_robot_interface.cpp
│   └── planar_base_controller.cpp
├── plugin_genie_sim_robot_interface.xml ← pluginlib: hardware_interface::SystemInterface
└── plugin_planar_base_controller.xml    ← pluginlib: controller_interface::ControllerInterface
```

---

## Plugins

### `genie_sim_control/GenieSimRobotInterface`
`base_class_type=hardware_interface::SystemInterface`

ros2_control **hardware interface** that talks to the GenieSim simulator
through plain ROS topics rather than a real driver. Used by `genie_sim_moveit`'s
`ros2_controllers.yaml` to wire the MoveIt JointTrajectoryController to whatever
physics engine is running.

### `genie_sim_control/PlanarBaseController`
`base_class_type=controller_interface::ControllerInterface`

Trajectory dispatcher for an SRDF **planar `virtual_joint`** (the kind MoveIt
uses to plan in `x / y / yaw` of a mobile base). Exposes a
`FollowJointTrajectory` action and publishes `geometry_msgs/Twist` —
typically on `/cmd_twist` — using P feedback against the TF chain
`map → base_link`. Doesn't claim any hardware interfaces (topic-driven);
the actual wheel-side closure lives in `genie_sim_controllers`'
`ChassisServoController`.

---

## Routing rules

- Hardware bridge (sim ↔ ros2_control) → `src/genie_sim_robot_interface.cpp`
- Planar-base trajectory dispatcher → `src/planar_base_controller.cpp`
- Wheel-side servo (consumer of `/cmd_twist`) → `../genie_sim_controllers/`
- ros2_control wiring (which plugins to load) → `../genie_sim_moveit/config/ros2_controllers.yaml`
