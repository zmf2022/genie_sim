# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import numpy as np
import sys
from pathlib import Path

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)
import geniesim.utils.system_utils as system_utils
import threading
from concurrent.futures import ThreadPoolExecutor
import grpc

# cam
from aimdk.protocol.hal.sensors import rs2_camera_pb2
from aimdk.protocol.hal.sensors import rs2_camera_pb2_grpc
from aimdk.protocol.sim import sim_camera_service_pb2
from aimdk.protocol.sim import sim_camera_service_pb2_grpc

# arm
from aimdk.protocol.hal.arm import arm_pb2
from aimdk.protocol.hal.arm import arm_pb2_grpc

# joint
from aimdk.protocol.hal.joint import joint_channel_pb2
from aimdk.protocol.hal.joint import joint_channel_pb2_grpc

# hand
from aimdk.protocol.sim import sim_gripper_service_pb2
from aimdk.protocol.sim import sim_gripper_service_pb2_grpc

# object
from aimdk.protocol.sim import sim_object_service_pb2
from aimdk.protocol.sim import sim_object_service_pb2_grpc

# observation
from aimdk.protocol.sim import sim_observation_service_pb2
from aimdk.protocol.sim import sim_observation_service_pb2_grpc


class CameraService(rs2_camera_pb2_grpc.CameraService):
    def __init__(self, server_function):
        self.server_function = server_function

    def SetCameraInfo(self, camera_info, width, height, ppx, ppy, fx, fy):
        camera_info.width = width
        camera_info.height = height
        camera_info.ppx = ppx
        camera_info.ppy = ppy
        camera_info.fx = fx
        camera_info.fy = fy

    def GetCameraData(self, req, rsp):
        rsp = rs2_camera_pb2.GetCameraDataResponse()
        rsp.serial_no = req.serial_no
        current_camera = self.server_function.blocking_start_server(
            data={
                "Cam_prim_path": req.serial_no,
                "isRGB": True,
                "isDepth": True,
                "isSemantic": False,
                "isGN": False,
            },
            Command=1,
        )
        camera_info = current_camera["camera_info"]
        self.SetCameraInfo(
            rsp.color_info,
            camera_info["width"],
            camera_info["height"],
            camera_info["ppx"],
            camera_info["ppy"],
            camera_info["fx"],
            camera_info["fy"],
        )
        rgb_camera = current_camera["rgb"]
        rsp.color_image.format = "rgb"
        if rgb_camera is not None:
            rsp.color_image.data = rgb_camera.tobytes()
        depth_camera = current_camera["depth"]
        rsp.depth_image.format = "depth"
        if depth_camera is not None:
            rsp.depth_image.data = depth_camera.tobytes()
        return rsp


class SimCameraService(sim_camera_service_pb2_grpc.SimCameraService):
    def __init__(self, server_function):
        self.server_function = server_function

    def GetSemanticData(self, req, rsp):
        rsp = sim_camera_service_pb2.GetSemanticResponse()
        rsp.serial_no = req.serial_no
        current_camera = self.server_function.blocking_start_server(
            data={
                "Cam_prim_path": req.serial_no,
                "isRGB": False,
                "isDepth": False,
                "isSemantic": True,
                "isGN": False,
            },
            Command=1,
        )
        semantic_image = current_camera["semantic"][0]
        label_ids = current_camera["semantic"][1]
        rsp.semantic_mask.name = "semantic"
        if semantic_image is not None:
            rsp.semantic_mask.data = semantic_image.tobytes()
        for key in label_ids:
            label = sim_camera_service_pb2.SemanticLabel()
            label.label_name = key
            label.label_id = label_ids[key]
            rsp.label_dict.append(label)
        return rsp


class ArmService(arm_pb2_grpc.G1ArmControlService):
    def __init__(self, server_function):
        self.server_function = server_function

    def LinearMove(self, req, rsp):
        rsp = arm_pb2.LinearMoveRsp()
        isArmRight = False
        if req.robot_name == "right":
            isArmRight = True
        target_position = np.array(
            [req.pose.position.x, req.pose.position.y, req.pose.position.z]
        )
        target_rotation = np.array(
            [req.pose.rpy.rw, req.pose.rpy.rx, req.pose.rpy.ry, req.pose.rpy.rz]
        )
        is_backend = req.is_block
        isSuccess = self.server_function.blocking_start_server(
            data={
                "isArmRight": isArmRight,
                "target_position": target_position,
                "target_rotation": target_rotation,
                "is_backend": is_backend,
                "ee_interpolation": req.ee_interpolation,
                "distance_frame": req.distance_frame,
            },
            Command=2,
        )
        rsp.errmsg = str(isSuccess)
        return rsp


class JointService(joint_channel_pb2_grpc.JointControlService):
    def __init__(self, server_function):
        self.server_function = server_function

    def GetJointPosition(self, req, rsp):
        rsp = joint_channel_pb2.GetJointRsp()
        joint_positions = self.server_function.blocking_start_server(
            data="get_joints", Command=8
        )
        for joint_name in joint_positions:
            joint_state = joint_channel_pb2.JointState()
            joint_state.name = joint_name
            joint_state.position = joint_positions[joint_name]
            rsp.states.append(joint_state)
        return rsp

    def SetJointPosition(self, req, rsp):
        rsp = joint_channel_pb2.SetJointRsp()
        joint_num = len(req.commands)
        target_joint_position = []
        target_joint_indices = []
        is_trajectory = req.is_trajectory
        for index in range(joint_num):
            v = req.commands[index].position
            idc = req.commands[index].sequence
            if v is None:
                continue
            if not np.isfinite(v):
                continue
            target_joint_position.append(v)
            target_joint_indices.append(idc)
        self.server_function.blocking_start_server(
            data={
                "target_joints_position": np.array(target_joint_position),
                "is_trajectory": is_trajectory,
                "target_joints_indices": target_joint_indices,
            },
            Command=3,
        )
        rsp.errmsg = "Move Joint"
        return rsp

    def GetEEPose(self, req, rsp):
        rsp = joint_channel_pb2.GetEEPoseRsp()
        is_right = req.is_right
        pose = self.server_function.blocking_start_server(
            data={"isRight": is_right}, Command=18
        )

        if pose:
            position, rotation = pose
            (
                rsp.ee_pose.position.x,
                rsp.ee_pose.position.y,
                rsp.ee_pose.position.z,
            ) = position
            (
                rsp.ee_pose.rpy.rw,
                rsp.ee_pose.rpy.rx,
                rsp.ee_pose.rpy.ry,
                rsp.ee_pose.rpy.rz,
            ) = rotation
        else:
            rsp.ee_pose.position.x = 0
            rsp.ee_pose.position.y = 0
            rsp.ee_pose.position.z = 0
            rsp.ee_pose.rpy.rw = 1
            rsp.ee_pose.rpy.rx = 0
            rsp.ee_pose.rpy.ry = 0
            rsp.ee_pose.rpy.rz = 0
        return rsp

    def GetIKStatus(self, req, rsp):
        rsp = joint_channel_pb2.GetIKStatusRsp()
        target_poses = []
        for pose in req.target_pose:
            target_position = np.array(
                [pose.position.x, pose.position.y, pose.position.z]
            )
            target_rotation = np.array(
                [pose.rpy.rw, pose.rpy.rx, pose.rpy.ry, pose.rpy.rz]
            )
            target_poses.append(
                {"position": target_position, "rotation": target_rotation}
            )
        is_right = req.is_right
        ObsAvoid = req.ObsAvoid
        ik_result = self.server_function.blocking_start_server(
            data={
                "target_poses": target_poses,
                "isRight": is_right,
                "ObsAvoid": ObsAvoid,
            },
            Command=19,
        )
        for result in ik_result:
            ik_status = joint_channel_pb2.IKStatus()
            ik_status.isSuccess = result[0]
            for key, value in result[1].items():
                ik_status.joint_names.append(key)
                ik_status.joint_positions.append(value)
            rsp.IKStatus.append(ik_status)
        return rsp

    def SetJointState(self, req, rsp):
        rsp = joint_channel_pb2.SetJointStateRsp()


class ObjectService(sim_object_service_pb2_grpc.SimObjectService):
    def __init__(self, server_function):
        self.server_function = server_function

    def AddObject(self, req, rsp):
        rsp = sim_object_service_pb2.AddObjectRsp()
        usd_path = req.usd_path
        prim_path = req.prim_path
        label_name = req.label_name
        object_mass = req.object_mass
        object_com = np.array([req.object_com.x, req.object_com.y, req.object_com.z])
        object_color = np.array(
            [req.object_color.r, req.object_color.g, req.object_color.b]
        )
        object_material = req.object_material
        target_position = np.array(
            [
                req.object_pose.position.x,
                req.object_pose.position.y,
                req.object_pose.position.z,
            ]
        )
        target_rotation = np.array(
            [
                req.object_pose.rpy.rw,
                req.object_pose.rpy.rx,
                req.object_pose.rpy.ry,
                req.object_pose.rpy.rz,
            ]
        )
        target_scale = np.array(
            [req.object_scale.x, req.object_scale.y, req.object_scale.z]
        )
        add_particle = req.add_particle
        particle_position = np.array(
            [req.particle_position.x, req.particle_position.y, req.particle_position.z]
        )
        particle_scale = np.array(
            [req.particle_scale.x, req.particle_scale.y, req.particle_scale.z]
        )
        particle_color = np.array(
            [req.particle_color.r, req.particle_color.g, req.particle_color.b]
        )
        model_type = req.model_type
        static_friction = req.static_friction
        dynamic_friction = req.dynamic_friction
        msg = self.server_function.blocking_start_server(
            data={
                "usd_object_path": str(system_utils.assets_path()) + "/" + usd_path,
                "usd_object_prim_path": prim_path,
                "usd_label_name": label_name,
                "usd_object_position": target_position,
                "usd_object_rotation": target_rotation,
                "usd_object_scale": target_scale,
                "object_color": object_color,
                "object_material": object_material,
                "object_mass": object_mass,
                "add_particle": add_particle,
                "particle_position": particle_position,
                "particle_scale": particle_scale,
                "particle_color": particle_color,
                "object_com": object_com,
                "model_type": model_type,
                "static_friction": static_friction,
                "dynamic_friction": dynamic_friction,
            },
            Command=6,
        )
        rsp.label_name = label_name
        return rsp

    def GetObjectPose(self, req, rsp):
        rsp = sim_object_service_pb2.GetObjectPoseRsp()
        prim_path = req.prim_path
        object_pose = self.server_function.blocking_start_server(
            data={"object_prim_path": prim_path}, Command=5
        )
        rsp.prim_path = prim_path
        if object_pose:
            position, rotation = object_pose
            (
                rsp.object_pose.position.x,
                rsp.object_pose.position.y,
                rsp.object_pose.position.z,
            ) = position
            (
                rsp.object_pose.rpy.rw,
                rsp.object_pose.rpy.rx,
                rsp.object_pose.rpy.ry,
                rsp.object_pose.rpy.rz,
            ) = rotation
        else:
            rsp.object_pose.position.x = 0
            rsp.object_pose.position.y = 0
            rsp.object_pose.position.z = 0
            rsp.object_pose.rpy.rw = 1
            rsp.object_pose.rpy.rx = 0
            rsp.object_pose.rpy.ry = 0
            rsp.object_pose.rpy.rz = 0
        return rsp

    def GetObjectJoint(self, req, rsp):
        rsp = sim_object_service_pb2.GetObjectJointRsp()
        prim_path = req.prim_path
        joint_state = self.server_function.blocking_start_server(
            data={"object_prim_path": prim_path}, Command=26
        )
        for name in joint_state["joint_names"]:
            rsp.joint_names.append(name)
        for position in joint_state["joint_positions"]:
            rsp.joint_positions.append(position)
        for velocity in joint_state["joint_velocities"]:
            rsp.joint_velocities.append(velocity)
        return rsp

    def SetTargetPoint(self, req, rsp):
        rsp = sim_object_service_pb2.SetTargetPointRsp()
        target_position = np.array(
            [req.point_position.x, req.point_position.y, req.point_position.z]
        )
        rsp.msg = self.server_function.blocking_start_server(
            data={"target_position": target_position}, Command=27
        )
        return rsp


class GripperService(sim_gripper_service_pb2_grpc.SimGripperService):
    def __init__(self, server_function):
        self.server_function = server_function

    def SetGripperState(self, req, rsp):
        rsp = sim_gripper_service_pb2.SetGripperStateRsp()
        gripper_command = req.gripper_command
        is_right = req.is_right
        front_msg = "left"
        if is_right:
            front_msg = "right"
        opened_width = req.opened_width
        msg = self.server_function.blocking_start_server(
            data={
                "gripper_state": gripper_command,
                "is_gripper_right": is_right,
                "opened_width": opened_width,
            },
            Command=9,
        )
        rsp.msg = front_msg + " " + msg
        return rsp

    def GetGripperState(self, req, rsp):
        rsp = sim_gripper_service_pb2.GetGripperStateRsp()
        is_right = req.is_right
        gripper_state = self.server_function.blocking_start_server(
            data={"isGripperRight": is_right}, Command=4
        )
        position, rotation = gripper_state
        (
            rsp.gripper_pose.position.x,
            rsp.gripper_pose.position.y,
            rsp.gripper_pose.position.z,
        ) = position
        (
            rsp.gripper_pose.rpy.rw,
            rsp.gripper_pose.rpy.rx,
            rsp.gripper_pose.rpy.ry,
            rsp.gripper_pose.rpy.rz,
        ) = rotation
        return rsp


class ObservationService(sim_observation_service_pb2_grpc.SimObservationService):
    def __init__(self, server_function):
        self.server_function = server_function

    def SetCameraInfo(self, camera_info, width, height, ppx, ppy, fx, fy):
        camera_info.width = width
        camera_info.height = height
        camera_info.ppx = ppx
        camera_info.ppy = ppy
        camera_info.fx = fx
        camera_info.fy = fy

    def GetObservation(self, req, rsp):
        rsp = sim_observation_service_pb2.GetObservationRsp()
        isCam = req.isCam
        isJoint = req.isJoint
        isPose = req.isPose
        isGripper = req.isGripper
        render_depth = req.CameraReq.render_depth
        render_semantic = req.CameraReq.render_semantic
        startRecording = req.startRecording
        stopRecording = req.stopRecording
        fps = req.fps
        task_name = req.task_name
        camera_prim_list = []
        object_prim = req.objectPrims
        result = self.server_function.blocking_start_server(
            data={
                "startRecording": startRecording,
                "stopRecording": stopRecording,
                "isCam": isCam,
                "isJoint": isJoint,
                "isPose": isPose,
                "isGripper": isGripper,
                "camera_prim_list": camera_prim_list,
                "render_depth": render_depth,
                "render_semantic": render_semantic,
                "object_prim": object_prim,
                "fps": fps,
                "task_name": task_name,
            },
            Command=11,
        )
        if startRecording or stopRecording:
            rsp.recordingState = result
            return rsp

        if isPose:
            for _pose in result["object"]:
                object_rsp = sim_observation_service_pb2.ObjectRsp()
                position, rotation = _pose
                (
                    object_rsp.object_pose.position.x,
                    object_rsp.object_pose.position.y,
                    object_rsp.object_pose.position.z,
                ) = position
                (
                    object_rsp.object_pose.rpy.rw,
                    object_rsp.object_pose.rpy.rx,
                    object_rsp.object_pose.rpy.ry,
                    object_rsp.object_pose.rpy.rz,
                ) = rotation
                rsp.pose.append(object_rsp)
        if isCam:
            for _cam in result["camera"]:
                camera_data = sim_observation_service_pb2.CameraRsp()
                camera_info = _cam["camera_info"]
                self.SetCameraInfo(
                    camera_data.camera_info,
                    camera_info["width"],
                    camera_info["height"],
                    camera_info["ppx"],
                    camera_info["ppy"],
                    camera_info["fx"],
                    camera_info["fy"],
                )
                rgb_camera = _cam["rgb"]
                camera_data.rgb_camera.format = "rgb"
                if rgb_camera is not None:
                    camera_data.rgb_camera.data = rgb_camera.tobytes()
                depth_camera = _cam["depth"]
                camera_data.depth_camera.format = "depth"
                if depth_camera is not None:
                    camera_data.depth_camera.data = depth_camera.tobytes()

                if _cam["semantic"] is not None:
                    semantic_image = _cam["semantic"][0]
                    label_ids = _cam["semantic"][1]
                    camera_data.semantic_mask.name = "semantic"
                    if semantic_image is not None:
                        camera_data.semantic_mask.data = semantic_image.tobytes()
                    for key in label_ids:
                        label = sim_observation_service_pb2.SemanticDict()
                        label.label_name = key
                        label.label_id = label_ids[key]
                        camera_data.label_dict.append(label)
                rsp.camera.append(camera_data)
        if isJoint:
            joint_positions = result["joint"]
            for joint_name in joint_positions:
                joint_state = sim_observation_service_pb2.JointState()
                joint_state.name = joint_name
                joint_state.position = joint_positions[joint_name]
                rsp.joint.left_arm.append(joint_state)
        if isGripper:
            left_pose, left_rotation = result["gripper"]["left"]
            right_pose, right_rotation = result["gripper"]["right"]
            (
                rsp.gripper.right_gripper.position.x,
                rsp.gripper.right_gripper.position.y,
                rsp.gripper.right_gripper.position.z,
            ) = right_pose
            (
                rsp.gripper.right_gripper.rpy.rw,
                rsp.gripper.right_gripper.rpy.rx,
                rsp.gripper.right_gripper.rpy.ry,
                rsp.gripper.right_gripper.rpy.rz,
            ) = right_rotation

            (
                rsp.gripper.left_gripper.position.x,
                rsp.gripper.left_gripper.position.y,
                rsp.gripper.left_gripper.position.z,
            ) = left_pose
            (
                rsp.gripper.left_gripper.rpy.rw,
                rsp.gripper.left_gripper.rpy.rx,
                rsp.gripper.left_gripper.rpy.ry,
                rsp.gripper.left_gripper.rpy.rz,
            ) = left_rotation
        return rsp

    def Reset(self, req, rsp):
        rsp = sim_observation_service_pb2.ResetRsp()
        Reset = req.reset
        rsp.msg = self.server_function.blocking_start_server(
            data={"reset", Reset}, Command=12
        )
        return rsp

    def AttachObj(self, req, rsp):
        rsp = sim_observation_service_pb2.AttachRsp()
        prim_paths = []
        for prim in req.obj_prims:
            prim_paths.append(prim)
        is_right = req.is_right
        rsp.msg = self.server_function.blocking_start_server(
            data={"obj_prim_paths": prim_paths, "is_right": is_right}, Command=13
        )
        return rsp

    def DetachObj(self, req, rsp):
        rsp = sim_observation_service_pb2.DetachRsp()
        detach = req.detach
        rsp.msg = self.server_function.blocking_start_server(
            data={"detach", detach}, Command=14
        )
        return rsp

    def MultiMove(self, req, rsp):
        rsp = sim_observation_service_pb2.MultiMoveRsp()
        target_poses = []
        for pose in req.poses:
            target_position = np.array(
                [pose.position.x, pose.position.y, pose.position.z]
            )
            target_rotation = np.array(
                [pose.rpy.rw, pose.rpy.rx, pose.rpy.ry, pose.rpy.rz]
            )
            target_poses.append([target_position, target_rotation])
        isArmRight = False
        if req.robot_name == "right":
            isArmRight = True
        is_plan = req.plan
        plan = req.cmd_plan

        result = self.server_function.blocking_start_server(
            data={"isArmRight": isArmRight, "poses": target_poses, "isPlan": is_plan},
            Command=15,
        )
        for plan in result["cmd_plans"]:
            Cmd_plan = sim_observation_service_pb2.CmdPlan()
            for name in plan["names"]:
                Cmd_plan.joint_names.append(name)
            for _plan in plan["positions"]:
                single_plan = sim_observation_service_pb2.SinglePlan()
                for position in _plan:
                    single_plan.joint_pos.append(position)
                Cmd_plan.joint_plans.append(single_plan)
            rsp.cmd_plans.append(Cmd_plan)
        rsp.msg = str(result["msg"])
        return rsp

    def GetObjectsOfType(self, req, rsp):
        rsp = sim_observation_service_pb2.GetObjectsOfTypeRsp()
        obj_type = req.obj_type
        result = self.server_function.blocking_start_server(
            data={"obj_type": obj_type}, Command=20
        )
        for prim_path in result:
            rsp.prim_paths.append(prim_path)
        return rsp

    def TaskStatus(self, req, rsp):
        rsp = sim_observation_service_pb2.TaskStatusRsp()
        isSuccess = req.isSuccess
        failStep = []
        for step in req.failStep:
            failStep.append(step)
        rsp.msg = self.server_function.blocking_start_server(
            data={"isSuccess": isSuccess, "failStep": failStep}, Command=16
        )
        return rsp

    def Exit(self, req, rsp):
        rsp = sim_observation_service_pb2.ExitRsp()
        exit = req.exit
        rsp.msg = self.server_function.blocking_start_server(
            data={"exit": exit}, Command=17
        )
        return rsp

    def AddCamera(self, req, rsp):
        rsp = sim_observation_service_pb2.AddCameraRsp()
        target_position = np.array(
            [
                req.camera_pose.position.x,
                req.camera_pose.position.y,
                req.camera_pose.position.z,
            ]
        )
        target_rotation = np.array(
            [
                req.camera_pose.rpy.rw,
                req.camera_pose.rpy.rx,
                req.camera_pose.rpy.ry,
                req.camera_pose.rpy.rz,
            ]
        )
        rsp.msg = self.server_function.blocking_start_server(
            data={
                "camera_prim": req.camera_prim,
                "camera_position": target_position,
                "camera_rotation": target_rotation,
                "focus_length": req.focus_length,
                "horizontal_aperture": req.horizontal_aperture,
                "vertical_aperture": req.vertical_aperture,
                "width": req.width,
                "height": req.height,
                "is_local": req.is_local,
            },
            Command=22,
        )
        return rsp

    def InitRobot(self, req, rsp):
        rsp = sim_observation_service_pb2.InitRobotRsp()
        target_position = np.array(
            [
                req.robot_pose.position.x,
                req.robot_pose.position.y,
                req.robot_pose.position.z,
            ]
        )
        target_rotation = np.array(
            [
                req.robot_pose.rpy.rw,
                req.robot_pose.rpy.rx,
                req.robot_pose.rpy.ry,
                req.robot_pose.rpy.rz,
            ]
        )
        rsp.msg = self.server_function.blocking_start_server(
            data={
                "robot_cfg_file": req.robot_cfg_file,
                "robot_usd_path": req.robot_usd_path,
                "scene_usd_path": req.scene_usd_path,
                "robot_position": target_position,
                "robot_rotation": target_rotation,
                "stand_type": req.stand_type,
                "stand_size_x": req.stand_size_x,
                "stand_size_y": req.stand_size_y,
            },
            Command=21,
        )
        return rsp

    def DrawLine(self, req, rsp):
        rsp = sim_observation_service_pb2.DrawLineRsp()
        point_list_1 = []
        point_list_2 = []
        colors = []
        sizes = []
        for point in req.point_list_1:
            point_list_1.append((point.x, point.y, point.z))
        for point in req.point_list_2:
            point_list_2.append((point.x, point.y, point.z))
        for color in req.colors:
            colors.append((color.x, color.y, color.z, 1))
        for size in req.sizes:
            sizes.append(size)
        rsp.msg = self.server_function.blocking_start_server(
            data={
                "point_list_1": point_list_1,
                "point_list_2": point_list_2,
                "colors": colors,
                "sizes": sizes,
            },
            Command=23,
        )
        return rsp

    def ClearLine(self, req, rsp):
        rsp = sim_observation_service_pb2.ClearLineRsp()
        rsp.msg = self.server_function.blocking_start_server(
            data={"name": req.name}, Command=31
        )
        return rsp

    def SetObjectPose(self, req, rsp):
        rsp = sim_observation_service_pb2.SetObjectPoseRsp()
        object_poses = []
        for object in req.object_pose:
            object_pose = {}
            object_pose["prim_path"] = object.prim_path
            object_pose["position"] = [
                object.pose.position.x,
                object.pose.position.y,
                object.pose.position.z,
            ]
            object_pose["rotation"] = [
                object.pose.rpy.rw,
                object.pose.rpy.rx,
                object.pose.rpy.ry,
                object.pose.rpy.rz,
            ]
            object_poses.append(object_pose)
        object_joints = []
        for joint in req.object_joint:
            object_joint = {}
            object_joint["prim_path"] = joint.prim_path
            object_joint["object_joint"] = []
            for joint in joint.joint_cmd:
                object_joint["object_joint"].append(joint.position)
            object_joints.append(object_joint)
        joint_num = len(req.joint_cmd)
        target_joint_position = np.zeros(joint_num)
        for index in range(joint_num):
            target_joint_position[index] = req.joint_cmd[index].position
        rsp.msg = self.server_function.blocking_start_server(
            data={
                "object_poses": object_poses,
                "joint_position": target_joint_position,
                "object_joints": object_joints,
            },
            Command=24,
        )
        return rsp

    def SetTrajectoryList(self, req, rsp):
        rsp = sim_observation_service_pb2.SetTrajectoryListRsp()
        trajectory_list = []
        is_block = req.is_block
        for trajectory_point in req.trajectory_point:
            position = np.array(
                [
                    trajectory_point.position.x,
                    trajectory_point.position.y,
                    trajectory_point.position.z,
                ]
            )
            rotation = np.array(
                [
                    trajectory_point.rpy.rw,
                    trajectory_point.rpy.rx,
                    trajectory_point.rpy.ry,
                    trajectory_point.rpy.rz,
                ]
            )
            pose = position, rotation
            trajectory_list.append(pose)
        rsp.msg = self.server_function.blocking_start_server(
            data={"trajectory_list": trajectory_list, "is_block": is_block}, Command=25
        )
        return rsp

    def SetFrameState(self, req, rsp):
        rsp = sim_observation_service_pb2.SetFrameStateRsp()
        frame_state = req.frame_state
        rsp.msg = self.server_function.blocking_start_server(
            data={"frame_state": frame_state}, Command=28
        )
        return rsp

    def SetMaterial(self, req, rsp):
        rsp = sim_observation_service_pb2.SetMaterialRsp()
        materials = []
        for mat in req.materials:
            materials.append(
                {
                    "object_prim": mat.object_prim,
                    "material_name": mat.material_name,
                    "material_path": mat.material_path,
                    "label_name": mat.label_name,
                }
            )
        rsp.msg = self.server_function.blocking_start_server(data=materials, Command=29)
        return rsp

    def SetLight(self, req, rsp):
        rsp = sim_observation_service_pb2.SetLightRsp()
        lights = []
        for light in req.lights:
            lights.append(
                {
                    "light_type": light.light_type,
                    "light_prim": light.light_prim,
                    "light_temperature": light.light_temperature,
                    "light_intensity": light.light_intensity,
                    "light_rotation": [
                        light.light_rotation.rw,
                        light.light_rotation.rx,
                        light.light_rotation.ry,
                        light.light_rotation.rz,
                    ],
                    "light_texture": light.light_texture,
                }
            )
        rsp.msg = self.server_function.blocking_start_server(data=lights, Command=30)
        return rsp

    def OmniCmdChangeProperty(self, req, rsp):
        rsp = sim_observation_service_pb2.OmniCmdChangePropertyRsp()
        cmd = {}
        cmd["prop_path"] = req.prop_path
        if req.WhichOneof("value") == "bool_value":
            cmd["value"] = req.bool_value
        if req.WhichOneof("value") == "str_value":
            cmd["value"] = req.str_value

        rsp.msg = self.server_function.blocking_start_server(data=cmd, Command=32)
        return rsp

    def GetPartiPointNumInbbox(self, req, rsp):
        rsp = sim_observation_service_pb2.GetPartiPointNumInbboxRsp()
        cmd = {}
        cmd["prim_path"] = req.prim_path
        cmd["bbox_3d"] = []
        for v in req.bbox:
            cmd["bbox_3d"].append(v)

        ret = self.server_function.blocking_start_server(data=cmd, Command=33)
        rsp.num = ret["num"]
        return rsp

    def GetObjectAABB(self, req, rsp):
        rsp = sim_observation_service_pb2.GetObjectAABBRsp()
        cmd = {}
        cmd["prim_path"] = req.prim_path

        ret = self.server_function.blocking_start_server(data=cmd, Command=34)
        for val in ret["points"]:
            rsp.bbox.append(val)
        return rsp

    def GetWorldPose(self, req, rsp):
        rsp = sim_observation_service_pb2.GetWorldPoseRsp()
        cmd = {}
        cmd["prim_path"] = req.prim_path

        ret = self.server_function.blocking_start_server(data=cmd, Command=35)
        for val in ret["pos"]:
            rsp.pos.append(val)
        for val in ret["quat"]:
            rsp.quat.append(val)
        return rsp


class GrpcServer:
    def __init__(self, server_function):
        self.server_function = server_function

    def start(self):
        server_thread = threading.Thread(target=self.server)
        server_thread.start()

    def server(self):
        self._server = grpc.server(
            ThreadPoolExecutor(max_workers=10),
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ],
        )
        rs2_camera_pb2_grpc.add_CameraServiceServicer_to_server(
            CameraService(self.server_function), self._server
        )
        sim_camera_service_pb2_grpc.add_SimCameraServiceServicer_to_server(
            SimCameraService(self.server_function), self._server
        )
        arm_pb2_grpc.add_G1ArmControlServiceServicer_to_server(
            ArmService(self.server_function), self._server
        )
        joint_channel_pb2_grpc.add_JointControlServiceServicer_to_server(
            JointService(self.server_function), self._server
        )
        sim_object_service_pb2_grpc.add_SimObjectServiceServicer_to_server(
            ObjectService(self.server_function), self._server
        )
        sim_gripper_service_pb2_grpc.add_SimGripperServiceServicer_to_server(
            GripperService(self.server_function), self._server
        )
        sim_observation_service_pb2_grpc.add_SimObservationServiceServicer_to_server(
            ObservationService(self.server_function), self._server
        )
        self.stop()
        self._server.add_insecure_port("0.0.0.0:50051")
        self._server.start()

    def stop(self):
        if self._server:
            self._server.stop(0)
