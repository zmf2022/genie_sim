# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_moveit_launch_utils():
    here = os.path.dirname(os.path.realpath(__file__))
    spec = importlib.util.spec_from_file_location("_moveit_launch_utils", os.path.join(here, "moveit_launch_utils.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_launch_description():
    mlu = _load_moveit_launch_utils()
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            mlu.moveit_joint_states_bridge_node(use_sim_time=False),
        ]
    )
