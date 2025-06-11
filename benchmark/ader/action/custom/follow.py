# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from benchmark.ader.action.common_actions import EvaluateAction, ActionBase, ActionEvent
import numpy as np
from collections import deque
from base_utils.logger import Logger
import ast

logger = Logger()


class Follow(EvaluateAction):
    def __init__(self, env, obj_name, bbox, gripper_id):
        super().__init__(env)
        self.obj_name = obj_name
        self._done_flag = False
        self.bbox = ast.literal_eval(bbox)
        self.env = env
        self.gripper_id = gripper_id

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        aa, bb = self.get_obj_aabb(self.obj_name, self.bbox)
        g_pose = self.env.robot.get_ee_pose(ee_type="gripper", id=self.gripper_id)
        g_pos = g_pose[:3, 3].reshape(-1)
        if self.aabb_contains_point(g_pos, (aa, bb)):
            logger.info(f"Follow obj {self.obj_name}")
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [Follow] {self.obj_name} evt: {event.value}")
        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            self.progress_info["SCORE"] = 0
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
