# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
"""Shared launch helpers for genie_sim_moveit."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from moveit_configs_utils.launch_utils import add_debuggable_node, DeclareBooleanLaunchArg

# Sim publishes /joint_states (SensorData) with all physics-tracked joints,
# including chassis wheel joints that are not claimed by ros2_control. We relay
# /joint_states directly to /moveit/joint_states (Reliable) so MoveIt's PSM
# observes the complete robot state. The hardware interface and
# joint_state_broadcaster still operate on their own topics for ros2_control's
# internal state, but they are NOT the source of truth for MoveIt: the broadcaster
# only republishes the 24 ros2_control-claimed joints, which would leave the 8
# wheel joints "missing" from MoveIt's view.
#
# The planar (base) DoF is supplied to MoveIt via the SRDF planar virtual_joint
# resolved from /tf (map -> base_link), not via /joint_states.
SIM_JOINT_STATES_TOPIC = "/joint_states"
MOVEIT_JOINT_STATES_TOPIC = "/moveit/joint_states"
MOVEIT_MOVE_GROUP_REMAPPINGS = [("joint_states", MOVEIT_JOINT_STATES_TOPIC)]


def moveit_joint_states_bridge_node(*, use_sim_time: bool = False) -> Node:
    """Launch the merged QoS-bridge + ride-height-synthesis node.

    Subscribes to the simulator's ``/joint_states`` (SensorData QoS),
    appends a synthetic ``base_footprint_to_base_link`` entry whose
    position is the live ``odom -> base_link`` Z translation from /tf,
    and republishes to ``/moveit/joint_states`` (Reliable QoS) for
    MoveIt's PlanningSceneMonitor.  See
    ``scripts/moveit_joint_states_bridge.py`` for the full rationale.
    """
    return Node(
        package="genie_sim_moveit",
        executable="moveit_joint_states_bridge.py",
        name="moveit_joint_states_bridge",
        output="screen",
        parameters=[
            {
                "input_topic": SIM_JOINT_STATES_TOPIC,
                "output_topic": MOVEIT_JOINT_STATES_TOPIC,
                "use_sim_time": use_sim_time,
            }
        ],
    )


def generate_genie_move_group_launch(moveit_config) -> LaunchDescription:
    """Like moveit_configs_utils.generate_move_group_launch, with joint_states relay + remap."""
    ld = LaunchDescription()

    ld.add_action(moveit_joint_states_bridge_node())
    ld.add_action(DeclareBooleanLaunchArg("debug", default_value=False))
    ld.add_action(DeclareBooleanLaunchArg("allow_trajectory_execution", default_value=True))
    ld.add_action(DeclareBooleanLaunchArg("publish_monitored_planning_scene", default_value=True))
    ld.add_action(
        DeclareLaunchArgument(
            "capabilities",
            default_value=moveit_config.move_group_capabilities["capabilities"],
        )
    )
    ld.add_action(
        DeclareLaunchArgument(
            "disable_capabilities",
            default_value=moveit_config.move_group_capabilities["disable_capabilities"],
        )
    )
    ld.add_action(DeclareBooleanLaunchArg("monitor_dynamics", default_value=False))

    should_publish = LaunchConfiguration("publish_monitored_planning_scene")
    move_group_configuration = {
        "publish_robot_description_semantic": True,
        "allow_trajectory_execution": LaunchConfiguration("allow_trajectory_execution"),
        "capabilities": ParameterValue(LaunchConfiguration("capabilities"), value_type=str),
        "disable_capabilities": ParameterValue(LaunchConfiguration("disable_capabilities"), value_type=str),
        "publish_planning_scene": should_publish,
        "publish_geometry_updates": should_publish,
        "publish_state_updates": should_publish,
        "publish_transforms_updates": should_publish,
        "monitor_dynamics": False,
    }

    add_debuggable_node(
        ld,
        package="moveit_ros_move_group",
        executable="move_group",
        commands_file=str(moveit_config.package_path / "launch" / "gdb_settings.gdb"),
        output="screen",
        parameters=[moveit_config.to_dict(), move_group_configuration],
        remappings=MOVEIT_MOVE_GROUP_REMAPPINGS,
        extra_debug_args=["--debug"],
        additional_env={"DISPLAY": os.environ.get("DISPLAY", "")},
    )
    return ld
