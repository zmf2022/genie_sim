# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base import BasePolicy
import time


class BaselinePolicy(BasePolicy):
    def __init__(self, task_name) -> None:
        super().__init__(task_name)

    def act(self, observations, **kwargs):
        step_num = kwargs.get("step_num", None)
        while self.sim_ros_node.buffer_empty():
            time.sleep(0.1)

        self.sim_ros_node.parse_joint_command()
        return

    def reset(self):
        target_position = [
            0.34906611,
            0.34987221,
            0,
            0.436332313,
            -0.66857928,
            0.67156327,
            0.2008844,
            -0.20287371,
            0.27921745,
            -0.282218840,
            -1.28203404,
            1.28208637,
            0.84163094,
            -0.84068865,
            1.51518357,
            -1.51710308,
            -0.18715125,
            0.18636601,
            1,
            -1,
            1,
            -1,
            0,
            1,
            0,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            0,
            0,
        ]
        return target_position
