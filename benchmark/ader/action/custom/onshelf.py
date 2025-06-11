# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from benchmark.ader.action.common_actions import EvaluateAction, ActionBase, ActionEvent
import numpy as np

from base_utils.logger import Logger

logger = Logger()


class OnShelf(EvaluateAction):
    def __init__(self, env, obj_name):
        super().__init__(env)
        self.obj_name = obj_name
        self._done_flag = False
        self.success_frame = 0
        self.success_time = 5

    def update(self, delta_time: float) -> float:
        [x, y, z] = self.get_obj_pose(self.obj_name)[0:3, 3]
        if "004" in self.obj_name:
            if abs(x + 4.42) < 0.04 and abs(y - 0.03) < 0.04 and abs(z - 1.13) < 0.05:
                self.success_frame += 1
        if "001" in self.obj_name:
            if abs(x + 4.42) < 0.03 and abs(y - 0.08) < 0.007 and abs(z - 1.13) < 0.05:
                self.success_frame += 1

        if "003" in self.obj_name:
            if abs(x + 4.42) < 0.03 and abs(y + 0.155) < 0.015 and abs(z - 1.13) < 0.05:
                self.success_frame += 1
        if self.success_frame > self.success_time:
            self.success_frame = 0
            logger.info(
                "\n======================\n\n\nTask Success!!!\n\n\n======================"
            )
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [OnShelf] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            pass
