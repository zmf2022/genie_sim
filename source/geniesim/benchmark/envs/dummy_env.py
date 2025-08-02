# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import time
import glob
import pickle
import numpy as np
import os
from scipy.spatial.transform import Rotation

from .base_env import BaseEnv

from geniesim.utils.logger import Logger

logger = Logger()  # Create singleton instance


from geniesim.robot import Robot

from geniesim.planner.manip_solver import load_task_solution
from geniesim.benchmark.tasks.demo_task import DemoTask
from geniesim.benchmark.tasks.demo_task import DemoTask


class DummyEnv(BaseEnv):
    def __init__(self, robot: Robot, task_file: str, init_task_config, need_setup=True):
        super().__init__(robot)
        self.attached_obj_id = None
        self.current_episode = 0
        self.current_step = 0
        self.init_task_config = init_task_config
        self.camera_list = self.init_task_config["recording_setting"]["camera_list"]
        self.specific_task_name = self.init_task_config["specific_task_name"]

        # Initialize the scene layout according to task_file
        self.load(task_file)
        if need_setup:
            self.load_task_setup()

    def load_task_setup(self):
        if "task_name" not in self.task_info:
            self.task = DummyTask(self)
        else:
            try:
                self.task = DemoTask(self)
            except ImportError:
                raise Exception("bddl is not available.")

    def load(self, task_file):
        self.generate_layout(task_file)
        self.robot.open_gripper(id="right")
        self.robot.open_gripper(id="left")
        task_info = json.load(open(task_file, "rb"))
        self.task_info = task_info
        self.policy_stages, self.policy_objects = load_task_solution(task_info)
        self.policy_objects = self.update_objects(self.policy_objects)

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode.
        """
        self.current_episode += 1
        self.current_step = 0

    def get_observation(self):
        """
        # Example
            data_keys = {
                "camera": {
                    "camera_prim_list": self.camera_list[:1],
                    "render_depth": False,
                    "render_semantic": False,
                },
                # "pose": ["/World/G1/gripper_center"],
                "joint_position": True,
                "gripper": True,
            }
        """
        observation_raw = self.robot.client.get_observation(self.data_keys)
        joint = self.robot.client.get_joint_positions()

        return observation_raw

    def reset(self, data_keys=None):
        if data_keys is None:
            data_keys = {
                "camera": {
                    "camera_prim_list": self.camera_list,
                    "render_depth": False,
                    "render_semantic": False,
                },
                # "pose": ["/World/G1/gripper_center"],
                "joint_position": True,
                "gripper": True,
            }
        self.data_keys = data_keys
        self.task.reset(self)
        self.reset_variables()
        observaion = self.get_observation()
        return observaion

    def step(self, actions):
        observaion = None
        self.current_step += 1
        need_update = False
        if self.current_step != 1 and self.current_step % 30 == 0:
            observaion = self.get_observation()
            self.task.step(self)
            self.action_update()
            need_update = True

        return observaion, self.has_done, need_update, self.task.task_progress

    def start_recording(self, task_name, camera_prim_list, fps, extra_objects_prim=[]):
        self.robot.client.start_recording(
            task_name=task_name,
            fps=fps,
            data_keys={
                "camera": {
                    "camera_prim_list": camera_prim_list,
                    "render_depth": False,
                    "render_semantic": False,
                },
                "pose": extra_objects_prim,
                "joint_position": True,
                "gripper": True,
            },
        )

    def generate_layout(self, task_file):
        self.task_file = task_file
        with open(task_file, "r") as f:
            task_info = json.load(f)

        # add mass for stable manipulation
        for stage in task_info["stages"]:
            if stage["action"] in ["place", "insert", "pour"]:
                obj_id = stage["passive"]["object_id"]
                for i in range(len(task_info["objects"])):
                    if task_info["objects"][i]["object_id"] == obj_id:
                        task_info["objects"][i]["mass"] = 10
                        break

        self.articulated_objs = []
        for object_info in task_info["objects"]:
            is_articulated = object_info.get("is_articulated", False)
            if is_articulated:
                self.articulated_objs.append(object_info["object_id"])
            object_info["material"] = "general"
            self.add_object(object_info)
        time.sleep(2)

        self.arm = task_info["arm"]

        """ For G1: Fix camera rotaton to look at target object """
        task_related_objs = []
        for stage in task_info["stages"]:
            for type in ["active", "passive"]:
                obj_id = stage[type]["object_id"]
                if obj_id == "gripper" or obj_id in task_related_objs:
                    continue
                task_related_objs.append(obj_id)

        material_infos = []
        for key in task_info["object_with_material"]:
            material_infos += task_info["object_with_material"][key]
        if len(material_infos):
            self.robot.client.SetMaterial(material_infos)
            time.sleep(0.3)

        light_infos = []
        for key in task_info["lights"]:
            light_infos += task_info["lights"][key]
        if len(light_infos):
            self.robot.client.SetLight(light_infos)
            time.sleep(0.3)

    def update_objects(self, objects, arm="right"):
        # update gripper pose
        objects["gripper"].obj_pose = self.robot.get_ee_pose(ee_type="gripper", id=arm)

        # update object pose
        for obj_id in objects:
            if obj_id == "gripper":
                continue

            if "/" in obj_id:
                obj_name = obj_id.split("/")[0]
                part_name = obj_id.split("/")[1]

                object_joint_state = self.robot.client.get_object_joint(
                    "/World/Objects/%s" % obj_name
                )
                for joint_name, joint_position, joint_velocity in zip(
                    object_joint_state.joint_names,
                    object_joint_state.joint_positions,
                    object_joint_state.joint_velocities,
                ):
                    if joint_name[-1] == part_name[-1]:
                        objects[obj_id].joint_position = joint_position
                        objects[obj_id].joint_velocity = joint_velocity

            objects[obj_id].obj_pose = self.robot.get_prim_world_pose(
                "/World/Objects/%s" % obj_id
            )
            if (
                "simple_place" in objects[obj_id].info
                and objects[obj_id].info["simple_place"]
            ):
                down_direction_world = (
                    np.linalg.inv(objects[obj_id].obj_pose) @ np.array([0, 0, -1, 1])
                )[:3]
                down_direction_world = (
                    down_direction_world / np.linalg.norm(down_direction_world) * 0.08
                )
                objects[obj_id].elements["active"]["place"][
                    "direction"
                ] = down_direction_world

        return objects

    def check_task_file(self, task_file):
        with open(task_file, "r") as f:
            task_info = json.load(f)

        objs_dir = {}
        objs_interaction = {}
        for obj_info in task_info["objects"]:
            obj_id = obj_info["object_id"]
            objs_dir[obj_id] = obj_info["data_info_dir"]
            if "interaction" in obj_info:
                objs_interaction[obj_id] = obj_info["interaction"]
            else:
                objs_interaction[obj_id] = json.load(
                    open(obj_info["data_info_dir"] + "/interaction.json")
                )["interaction"]

        for stage in task_info["stages"]:
            active_obj_id = stage["active"]["object_id"]
            passive_obj_id = stage["passive"]["object_id"]
            if active_obj_id != "gripper":
                if active_obj_id not in objs_dir:
                    logger.info("Active obj not in objs_dir: %s" % active_obj_id)
                    return False
            if passive_obj_id != "gripper":
                if passive_obj_id not in objs_dir:
                    logger.info("Passive obj not in objs_dir: %s" % passive_obj_id)
                    return False
            data_root = os.path.dirname(os.path.dirname(__file__)) + "/assets"
            if stage["action"] in ["grasp", "pick"]:
                passive_obj_id = stage["passive"]["object_id"]
                obj_dir = objs_dir[passive_obj_id]
                primitive = stage["passive"]["primitive"]
                if primitive is None:
                    file = "grasp_pose/grasp_pose.pkl"
                else:
                    file = objs_interaction[passive_obj_id]["passive"]["grasp"][
                        primitive
                    ]
                    if isinstance(file, list):
                        file = file[0]
                grasp_file = os.path.join(data_root, obj_dir, file)
                if not os.path.exists(grasp_file):
                    logger.warning("-- Grasp file not exist: %s" % grasp_file)
                    return False

                _data = pickle.load(open(grasp_file, "rb"))
                if len(_data["grasp_pose"]) == 0:
                    logger.warning("-- Grasp file empty: %s" % grasp_file)
                    return False
        return True
