# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import importlib.util
import os

from moveit_configs_utils import MoveItConfigsBuilder


def _load_moveit_launch_utils():
    here = os.path.dirname(os.path.realpath(__file__))
    spec = importlib.util.spec_from_file_location("_moveit_launch_utils", os.path.join(here, "moveit_launch_utils.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_launch_description():
    mlu = _load_moveit_launch_utils()
    moveit_config = MoveItConfigsBuilder("genie", package_name="genie_sim_moveit").to_moveit_configs()
    return mlu.generate_genie_move_group_launch(moveit_config)
