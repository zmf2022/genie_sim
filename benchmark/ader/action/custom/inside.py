# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from benchmark.ader.action.common_actions import EvaluateAction, ActionBase, ActionEvent
import numpy as np
from base_utils.logger import Logger

logger = Logger()


class Inside(EvaluateAction):
    def __init__(self, env, active_obj, passive_obj):
        super().__init__(env)
        self.active_obj = active_obj
        self.passive_obj = passive_obj
        self._done_flag = False
        self._pass_frame = 0

    def update(self, delta_time: float) -> float:
        pose_A = self.get_obj_pose(self.active_obj)
        pos_A = pose_A[:3, 3].reshape(
            -1,
        )
        aa_B, bb_B = self.get_obj_aabb_new(self.passive_obj)
        if self.aabb_contains_point(pos_A, (aa_B, bb_B)):
            self._pass_frame += 1
        else:
            self._pass_frame = 0
        if self._pass_frame > 2:
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [Inside] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            self.progress_info["SCORE"] = 0
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
