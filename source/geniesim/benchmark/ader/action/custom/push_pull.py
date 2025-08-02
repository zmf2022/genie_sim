# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim.benchmark.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
import numpy as np
from collections import deque
from geniesim.utils.logger import Logger

logger = Logger()


class PushPull(EvaluateAction):
    def __init__(self, env, obj_name, thresh_min, thresh_max):
        super().__init__(env)
        self.obj_name = obj_name
        self._done_flag = False
        self.thresh_min = float(thresh_min)
        self.thresh_max = float(thresh_max)
        self.env = env
        self._pass_frame = 0

    def update(self, delta_time: float) -> float:
        if not self.is_running():
            return 0.0
        prismatic_joint = self.get_prismatic_joint(self.obj_name)

        for v in prismatic_joint:
            if self.thresh_min == 0.0:
                condition_met = all(v <= self.thresh_max for v in prismatic_joint)
            else:
                condition_met = any(
                    self.thresh_min <= v <= self.thresh_max for v in prismatic_joint
                )

            if condition_met:
                self._pass_frame += 1
                break

        if self._pass_frame > 1:
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [PushPull] evt: %d" % (event.value))

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
