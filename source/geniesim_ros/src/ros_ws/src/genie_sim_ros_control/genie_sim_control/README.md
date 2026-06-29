# genie_sim_control

The bridge between MoveIt / ros2_control and the GenieSim simulator.
Provides the ros2_control plugins that turn MoveIt's
`FollowJointTrajectory` actions and planner output into simulator
commands.

## Plugins

### `GenieSimRobotInterface` (`hardware_interface::SystemInterface`)

The ros2_control hardware interface used by `genie_sim_moveit`. It
doesn't talk to real hardware — it talks to whichever physics engine
is running, over plain ROS topics. From MoveIt's perspective the
simulator looks like any other ros2_control hardware backend.

### `PlanarBaseController` (`controller_interface::ControllerInterface`)

A trajectory dispatcher for the SRDF **planar `virtual_joint`** that
MoveIt uses to plan in `x / y / yaw` for mobile bases. Exposes a
`FollowJointTrajectory` action, runs a P controller against the
`map → base_link` TF chain, and publishes `geometry_msgs/Twist` on
`/cmd_twist`.

The wheel-side closure of that loop lives in `genie_sim_controllers`
(the 4WS chassis servo).

## When you'd touch this package

- Adding a new hardware interface plugin (e.g. a different topic
  layout, a hardware-in-the-loop variant).
- Tuning the planar-base P controller gains.
- Changing the `/cmd_twist` output schema or the trajectory dispatch
  semantics.

## Mechanics

See [AGENTS.md](AGENTS.md) for the pluginlib registration, where the
ros2_control wiring lives (`genie_sim_moveit/config/ros2_controllers.yaml`),
and how the planar-base controller hands off to the chassis servo.
