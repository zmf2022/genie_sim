# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import threading
import os
import rclpy
import json
from base_utils.ros_utils import SimROSNode

ROOT_DIR = os.environ.get("SIM_REPO_ROOT")


class BasePolicy:
    def __init__(self, task_name) -> None:
        rclpy.init()
        # load robot_cfg
        with open(
            os.path.join(ROOT_DIR, "benchmark/ader/eval_tasks", f"{task_name}.json"),
            "r",
        ) as f:
            task_content = json.load(f)
        self.robot_cfg_file = task_content["robot"]["robot_cfg"]
        with open(
            os.path.join(
                ROOT_DIR, "server/source/genie.sim.lab/robot_cfg/", self.robot_cfg_file
            ),
            "r",
        ) as f:
            self.robot_cfg = json.load(f)
        self.sim_ros_node = SimROSNode(robot_cfg=self.robot_cfg)
        self.spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.sim_ros_node,)
        )
        self.spin_thread.start()

    def shutdown(self):
        if rclpy.ok():
            self.sim_ros_node.destroy_node()
            rclpy.shutdown()

        if self.spin_thread.is_alive():
            self.spin_thread.join(timeout=5)

    def reset(self):
        """Called at the beginning of an episode."""
        pass

    def set_robot(self, robot, cam_dict):
        pass

    def act(self, observations, **kwargs) -> np.ndarray:
        """Act based on the observations."""
        pass


class RandomPolicy(BasePolicy):
    def __init__(self, action_space=1):
        self.action_space = action_space

    def act(self, observations, **kwargs):
        action = np.random.uniform(low=-1, high=1, size=(self.action_dim,))
        return action

    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        return "rgbd"

    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        return None
