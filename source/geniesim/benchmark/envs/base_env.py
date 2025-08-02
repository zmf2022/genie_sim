# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import time
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from geniesim.robot.genie_robot import IsaacSimRpcRobot
from geniesim.benchmark.ader.action.action_manager import ActionManager
from geniesim.benchmark.ader.action.common_actions import ActionBase


class BaseEnv(object):
    def __init__(self, robot: IsaacSimRpcRobot) -> None:
        self.robot = robot
        self.action_executor = ActionManager()
        self.last_update_time = time.time()
        self.has_done = False

    def do_action(self, slot: str, name: str, action: ActionBase):
        self.action_executor.start(slot, name, action)

    def cancel_action(self, slot):
        self.action_executor.stop(slot)

    def exist_eval_action(self):
        return self.action_executor.exist_action("eval")

    def do_eval_action(self):
        self.do_action("eval", self.init_task_config["task"], self.task.action)

    def cancel_eval(self):
        self.reset()
        self.has_done = True

    def reset(self):
        self.robot.reset()

    def exist_eval_action(self):
        return self.action_executor.exist_action("eval")

    def action_update(self):
        if not self.exist_eval_action():
            self.has_done = True
            return
        delta_time = time.time() - self.last_update_time
        self.action_executor.update(delta_time)
        self.last_update_time = time.time()
        if self.has_done:
            self.cancel_action("eval")

    def add_object(self, object_info: dict):
        name = object_info["object_id"]
        usd_path = object_info.get("model_path")
        if not usd_path:
            return
        position = np.array(object_info["position"])
        quaternion = np.array(object_info["quaternion"])
        if "scale" not in object_info:
            object_info["scale"] = [1, 1, 1]
            size = object_info.get("size", np.array([]))
            if isinstance(size, list):
                size = np.array(size)
            if size.shape[0] == 0:
                size = np.array(100)
            if np.max(size) > 10:
                object_info["scale"] = [0.001, 0.001, 0.001]
        add_particle = False
        particle_position = [0, 0, 0]
        particle_scale = [1, 1, 1]
        if "add_particle" in object_info:
            add_particle = True
            particle_position = object_info["add_particle"]["position"]
            particle_scale = object_info["add_particle"]["scale"]
        if isinstance(object_info["scale"], list):
            scale = np.array(object_info["scale"])
        else:
            scale = np.array([object_info["scale"]] * 3)
        material = (
            "general" if "material" not in object_info else object_info["material"]
        )
        mass = object_info.get("mass", 0.01)
        com = object_info.get("com", [0, 0, 0])
        model_type = object_info.get("model_type", "convexDecomposition")
        static_friction = object_info.get("static_friction", 0.5)
        dynamic_friction = object_info.get("dynamic_friction", 0.5)
        self.robot.client.add_object(
            usd_path=usd_path,
            prim_path="/World/Objects/%s" % name,
            label_name=name,
            target_position=position,
            target_quaternion=quaternion,
            target_scale=scale,
            material=material,
            color=[1, 1, 1],
            mass=mass,
            add_particle=add_particle,
            particle_position=particle_position,
            particle_scale=particle_scale,
            com=com,
            model_type=model_type,
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
        )

    def generate_layout(self, task_file, init=True):
        with open(task_file, "r") as f:
            task_info = json.load(f)

        if init:
            for object_info in task_info["objects"]:
                if "box" not in object_info["name"]:
                    continue
                self.add_object(object_info)
            time.sleep(1)

            for object_info in task_info["objects"]:
                if "obj" not in object_info["name"]:
                    continue
                self.add_object(object_info)
            time.sleep(1)

        self.task_info = task_info
        self.robot.target_object = task_info["target_object"]
        self.arm = task_info["arm"]
        return task_info

    def execute(self, commands):
        for command in commands:
            type = command["action"]
            content = command["content"]
            if type == "move" or type == "rotate":
                if self.robot.move(content) == False:
                    return False
                else:
                    self.last_pose = content["matrix"]
            elif type == "open_gripper":
                id = "left" if "gripper" not in content else content["gripper"]
                width = 0.1 if "width" not in content else content["width"]
                self.robot.open_gripper(id=id, width=width)

            elif type == "close_gripper":
                id = "left" if "gripper" not in content else content["gripper"]
                force = 50 if "force" not in content else content["force"]
                self.robot.close_gripper(id=id, force=force)
                if "attach_obj" in content:
                    self.robot.client.AttachObj(prim_paths=[content["attach_obj"]])

            else:
                Exception("Not implemented.")
        return True

    def start_recording(self, task_name, camera_prim_list):
        self.robot.client.start_recording(
            task_name=task_name,
            fps=30,
            data_keys={
                "camera": {
                    "camera_prim_list": camera_prim_list,
                    "render_depth": False,
                    "render_semantic": False,
                },
                "pose": [],
                "joint_position": True,
                "gripper": True,
            },
        )

    def stop_recording(self, success):
        self.robot.client.stop_recording()

    def grasp(
        self,
        target_gripper_pose,
        gripper_id=None,
        use_pre_grasp=False,
        use_pick_up=False,
        grasp_width=0.1,
    ):
        gripper_id = "left" if gripper_id is None else gripper_id
        pick_up_pose = np.copy(target_gripper_pose)
        pick_up_pose[2, 3] = pick_up_pose[2, 3] + 0.1  # lift up 10cm after grasped obj

        commands = []
        commands.append(
            {
                "action": "open_gripper",
                "content": {
                    "gripper": gripper_id,
                    "width": grasp_width,
                },
            }
        )
        if use_pre_grasp:
            pre_pose = np.array(
                [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, -0.05], [0, 0, 0, 1]]
            )
            pre_grasp_pose = target_gripper_pose @ pre_pose
            commands.append(
                {
                    "action": "move",
                    "content": {
                        "matrix": pre_grasp_pose,
                        "type": "matrix",
                        "comment": "pre_grasp",
                        "trajectory_type": "ObsAvoid",
                    },
                }
            )

        grasp_trajectory_type = "Simple" if use_pre_grasp else "ObsAvoid"
        commands.append(
            {
                "action": "move",
                "content": {
                    "matrix": target_gripper_pose,
                    "type": "matrix",
                    "comment": "grasp",
                    "trajectory_type": grasp_trajectory_type,
                },
            }
        )
        commands.append(
            {
                "action": "close_gripper",
                "content": {
                    "gripper": gripper_id,
                    "force": 50,
                },
            }
        )

        if use_pick_up:
            commands.append(
                {
                    "action": "move",
                    "content": {
                        "matrix": pick_up_pose,
                        "type": "matrix",
                        "comment": "pick_up",
                        "trajectory_type": "Simple",
                    },
                }
            )

        return commands

    def get_observation(self):
        observation_raw = self.robot.get_observation(self.data_keys)
        self.observation = observation_raw["camera"][self.robot.base_camera]

    def place(
        self,
        target_gripper_pose,
        current_gripper_pose=None,
        gripper2part=None,
        gripper_id=None,
    ):
        gripper_id = "left" if gripper_id is None else gripper_id

        pre_place_pose = np.copy(target_gripper_pose)
        pre_place_pose[2, 3] += 0.05

        commands = [
            {
                "action": "move",
                "content": {
                    "matrix": pre_place_pose,
                    "type": "matrix",
                    "comment": "pre_place",
                    "trajectory_type": "ObsAvoid",
                },
            },
            {
                "action": "move",
                "content": {
                    "matrix": target_gripper_pose,
                    "type": "matrix",
                    "comment": "place",
                    "trajectory_type": "Simple",
                },
            },
            {
                "action": "open_gripper",
                "content": {
                    "gripper": gripper_id,
                },
            },
        ]
        return commands

    def pour(
        self,
        target_gripper_pose,
        current_gripper_pose=None,
        gripper2part=None,
        gripper_id=None,
    ):
        def interpolate_rotation_matrices(rot_matrix1, rot_matrix2, num_interpolations):
            rot1 = R.from_matrix(rot_matrix1)
            rot2 = R.from_matrix(rot_matrix2)
            quat1 = rot1.as_quat()
            quat2 = rot2.as_quat()
            times = [0, 1]
            slerp = Slerp(times, R.from_quat([quat1, quat2]))
            interp_times = np.linspace(0, 1, num_interpolations)
            interp_rots = slerp(interp_times)
            interp_matrices = interp_rots.as_matrix()
            return interp_matrices

        target_part_pose = target_gripper_pose @ np.linalg.inv(gripper2part)
        current_part_pose = current_gripper_pose @ np.linalg.inv(gripper2part)
        commands = []
        rotations = interpolate_rotation_matrices(
            current_part_pose[:3, :3], target_part_pose[:3, :3], 5
        )
        for i, rotation in enumerate(rotations):
            target_part_pose_step = np.copy(target_part_pose)
            target_part_pose_step[:3, :3] = rotation
            target_gripper_pose_step = target_part_pose_step @ gripper2part

            commands.append(
                {
                    "action": "move",
                    "content": {
                        "matrix": target_gripper_pose_step,
                        "type": "matrix",
                        "comment": "pour_sub_rotate_%d" % i,
                        "trajectory_type": "Simple",
                    },
                }
            )
        return commands
