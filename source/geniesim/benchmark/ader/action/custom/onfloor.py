# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.benchmark.ader.action.common_actions import EvalExitAction
import numpy as np
from collections import deque
from geniesim.utils.logger import Logger

logger = Logger()


class Onfloor(EvalExitAction):
    def __init__(self, env, obj_name, height):
        super().__init__(env)
        self.obj_name = obj_name
        self._done_flag = False
        self.z_ref = float(height)
        self.z_threshold = 0.3
        self.env = env

    def _analyze_obj_name(self, obj_name):
        if obj_name.startswith("/World"):
            return obj_name

        return "/World/Objects/" + obj_name

    def get_obj_pose(self, obj_name):
        pose = self.env.robot.get_prim_world_pose(self._analyze_obj_name(obj_name))
        return np.array(pose)

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        obj_pose = self.get_obj_pose(self.obj_name)
        current_z = obj_pose[2, 3]
        if abs(current_z - self.z_ref) < self.z_threshold:
            self._done_flag = True
            self.progress_info["SCORE"] = 0
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        pass
