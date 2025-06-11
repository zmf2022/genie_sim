# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
import cv2
import os
from .base import BasePolicy
from .model_infer.dummy_infer import DummyInfer
from .model_infer.dummy_infer import get_actions_lerobot
from scipy.spatial.transform import Rotation as R
import ik_solver
import base_utils


def get_rgb_image(observation, camera_prim):
    cam_info = observation["camera"][camera_prim]["camera_info"]
    rgb_image = observation["camera"][camera_prim]["rgb_camera"].reshape(
        cam_info["height"], cam_info["width"], 4
    )[:, :, :3]
    return rgb_image


def mat2xyzrpy(mat):
    rpy = R.from_matrix(mat[0:3, 0:3]).as_euler("xyz", degrees=False)
    xyz = mat[0:3, 3]
    xyzrpy = np.concatenate([xyz, rpy])
    return xyzrpy


def xyzrpy2mat(xyzrpy):
    rot = R.from_euler("xyz", xyzrpy[3:6]).as_matrix()
    mat = np.eye(4)
    mat[0:3, 0:3] = rot
    mat[0:3, 3] = xyzrpy[0:3]
    return mat


class DummyPolicy(BasePolicy):
    def __init__(self, task_name, action_step=30, save_res=False):
        super().__init__(task_name)

        # init DummyInfer
        self.task_name = task_name
        self.action_step = action_step
        self.demo_infer = DummyInfer()

        # init config
        self.INIT_POSE = []
        self.MODEL_CONFIG = []

        # init cpp ik_solver
        self.head_init_position = self.INIT_POSE["body_state"][:2]
        self.waist_init_position = self.INIT_POSE["body_state"][2:4]
        self.solver = ik_solver.Solver(
            urdf_path=str(
                os.path.join(os.path.dirname(base_utils.__file__), "IK-SDK", "G1.urdf")
            ),
            config_path=str(
                os.path.join(
                    os.path.dirname(base_utils.__file__), "IK-SDK/config", "solver.yaml"
                )
            ),
            use_relaxed_ik=True,
            use_elbow=False,
        )
        self.solver.initialize_states(
            left_arm_init=np.array(self.INIT_POSE["init_arm"][:7], dtype=np.float32),
            right_arm_init=np.array(self.INIT_POSE["init_arm"][7:], dtype=np.float32),
            head_init=np.array(self.head_init_position, dtype=np.float32),
        )
        q_full = np.zeros(18)
        q_full[0] = self.waist_init_position[1]
        q_full[1] = self.waist_init_position[0]
        self.base_T_center = self.solver.compute_fk(
            q=q_full, start_link="base_link", end_link="arm_base_link"
        )
        self.center_T_base = np.linalg.inv(self.base_T_center)

        pass

    def reset(self):
        # return initial pose
        pass

    def act(self, observations, **kwargs):
        step_num = kwargs.get("step_num", None)

        # infer action every step_num steps
        if step_num % self.action_step == 0:
            self.observations = observations
            rgb_inputs, joints_state = self.deal_observation_data(
                observations, step_num=step_num
            )

            payload = {
                "observation.images.cam_right_wrist": rgb_inputs[2],
                "observation.images.cam_left_wrist": rgb_inputs[1],
                "observation.images.cam_top": rgb_inputs[0],
                "instruction": {
                    "conversation_type": 0,
                    "job_description": "",
                    "sub_job_description": "",
                },
            }
            raw_action, model_outputs = get_actions_lerobot(
                self.demo_infer,
                payload,
                False,
                "",
            )

            # calc ik/fk
            raw_action = self.compute_abs_eef_cpp(raw_action, joints_state)
            self.action_list = self.eef_actions_to_joint_cpp(raw_action, joints_state)

        # generate target_position according to cur_action
        cur_action = self.action_list[0]
        target_position = []
        action_dict = {
            "type": "joint",
            "position": target_position,
            "is_trajectory": True,
        }
        self.action_list = self.action_list[1:]
        return action_dict

    def compute_abs_eef_cpp(self, actions, joints_state):
        left_joint_state = joints_state[0:7]
        right_joint_state = joints_state[7:14]
        actions_np = np.array(actions)
        left_arm_T = self.solver.compute_part_fk(
            q_part=np.array(left_joint_state, dtype=np.float32),
            part=ik_solver.RobotPart.LEFT_ARM,
            from_base=False,
        )
        right_arm_T = self.solver.compute_part_fk(
            q_part=np.array(right_joint_state, dtype=np.float32),
            part=ik_solver.RobotPart.RIGHT_ARM,
            from_base=False,
        )
        left_arm_base = self.base_T_center @ left_arm_T
        eefrot_left_xyzrpy = mat2xyzrpy(left_arm_base)
        right_arm_base = self.base_T_center @ right_arm_T
        eefrot_right_xyzrpy = mat2xyzrpy(right_arm_base)
        eefrot_left_xyzrpy_last = eefrot_left_xyzrpy
        eefrot_right_xyzrpy_last = eefrot_right_xyzrpy
        abs_eef_actions = []
        for _, action in enumerate(actions_np):
            if self.MODEL_CONFIG["delta_diff"]:
                eefrot_left_xyzrpy_cur = eefrot_left_xyzrpy_last + action[0:6]
                eefrot_right_xyzrpy_cur = eefrot_right_xyzrpy_last + action[6:12]
                eefrot_left_xyzrpy_last = eefrot_left_xyzrpy_cur
                eefrot_right_xyzrpy_last = eefrot_right_xyzrpy_cur
            else:
                eefrot_left_xyzrpy_cur = eefrot_left_xyzrpy + action[0:6]
                eefrot_right_xyzrpy_cur = eefrot_right_xyzrpy + action[6:12]
            eefrot_left_mat_cur_center = self.center_T_base @ xyzrpy2mat(
                eefrot_left_xyzrpy_cur
            )
            eefrot_left_xyzrpy_cur_center = mat2xyzrpy(eefrot_left_mat_cur_center)
            eefrot_right_mat_cur_center = self.center_T_base @ xyzrpy2mat(
                eefrot_right_xyzrpy_cur
            )
            eefrot_right_xyzrpy_cur_center = mat2xyzrpy(eefrot_right_mat_cur_center)
            abs_eef_actions.append(
                eefrot_left_xyzrpy_cur_center.tolist()
                + eefrot_right_xyzrpy_cur_center.tolist()
                + action[12:14].tolist()
            )
        return abs_eef_actions

    def eef_actions_to_joint_cpp(self, eef_actions, joints_state):
        joint_actions = []
        for _, action in enumerate(eef_actions):
            eefrot_left_cur = np.array(action[:6], dtype=np.float32)
            eefrot_right_cur = np.array(action[6:12], dtype=np.float32)
            self.solver.update_target_mat(
                part=ik_solver.RobotPart.LEFT_ARM,
                target_pos=eefrot_left_cur[:3],
                target_rot=xyzrpy2mat(eefrot_left_cur)[0:3, 0:3],
            )
            self.solver.update_target_mat(
                part=ik_solver.RobotPart.RIGHT_ARM,
                target_pos=eefrot_right_cur[:3],
                target_rot=xyzrpy2mat(eefrot_right_cur)[0:3, 0:3],
            )
            left_joints = self.solver.solve_left_arm()
            right_joints = self.solver.solve_right_arm()
            l_gripper = (
                action[12:13] if type(action) == list else action[12:13].tolist()
            )
            r_gripper = (
                action[13:14] if type(action) == list else action[13:14].tolist()
            )
            joint_actions.append(
                left_joints.tolist() + right_joints.tolist() + l_gripper + r_gripper
            )
        return joint_actions
