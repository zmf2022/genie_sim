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


class PickUpOnGripper(EvaluateAction):
    def __init__(self, env, obj_name, gripper_id):
        super().__init__(env)
        self.obj_name = obj_name
        self._done_flag = False
        self.gripper_id = gripper_id

        # Threshold Parameters
        self.z_threshold = 0.01
        self.pos_threshold = 0.06
        self.rot_threshold = 10

        # Status record
        self.initial_z = None
        self.stage = 1
        self.pose_history = deque(maxlen=3)  # Pose History Queue

    def calculate_rot_diff(self, R1, R2):
        """Calculate the angle difference (degree) between two rotation matrices"""
        # Calculate the relative rotation matrix
        R_rel = R1 @ R2.T
        # Calculate the rotation angle
        theta = np.arccos((np.trace(R_rel) - 1) / 2)
        return np.degrees(theta)

    def check_stage_1(self, obj_pose):
        """Stage 1 Detection: Object Z-axis lifting"""
        current_z = obj_pose[2, 3]

        if self.initial_z is None:
            self.initial_z = current_z
            return False

        z_diff = current_z - self.initial_z
        return z_diff > self.z_threshold

    def check_stage_2(self, gripper_pose, obj_pose):
        """Stage 2 Detection: Two-frame stability"""
        # Record current frame data
        if len(self.pose_history) == 2:
            self.pose_history.popleft()

        self.pose_history.append((gripper_pose.copy(), obj_pose.copy()))

        if len(self.pose_history) < 2:
            return False

        # Calculate the differences between consecutive frames
        diffs = []
        for i in range(1, len(self.pose_history)):
            prev_gripper, prev_obj = self.pose_history[i - 1]
            curr_gripper, curr_obj = self.pose_history[i]

            # Displacement difference
            delta_gripper = np.linalg.norm(curr_gripper[:3, 3] - prev_gripper[:3, 3])
            delta_obj = np.linalg.norm(curr_obj[:3, 3] - prev_obj[:3, 3])
            pos_diff = abs(delta_obj - delta_gripper)

            # Rotation difference
            rot_diff = self.calculate_rot_diff(
                curr_gripper[:3, :3], prev_gripper[:3, :3]
            )

            diffs.append((pos_diff, rot_diff))
        # Check whether all frames meet the criteria
        return all(
            (pos < self.pos_threshold) and (rot < self.rot_threshold)
            for pos, rot in diffs
        )

    def update(self, delta_time: float) -> float:
        link_prim_path = (
            "/G1/gripper_r_center_link"
            if self.gripper_id == "right"
            else "/G1/gripper_l_center_link"
        )
        current_gripper_pose = self.get_world_pose_matrix(link_prim_path)
        current_obj_pose = self.get_obj_pose(self.obj_name)

        if self.stage == 1:
            if self.check_stage_1(current_obj_pose):
                self.stage = 2

        if self.stage == 2:
            if self.check_stage_2(current_gripper_pose, current_obj_pose):
                self._done_flag = True
            else:
                self.stage = 1

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [PickUpOnGripper] {self.gripper_id} evt: {event.value}")

        if event == ActionEvent.STARTED:
            pass
        elif event == ActionEvent.PAUSED:
            pass
        elif event == ActionEvent.CANCELED:
            pass
        elif event == ActionEvent.FINISHED:
            self.progress_info["SCORE"] = 1
