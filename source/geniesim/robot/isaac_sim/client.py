# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import grpc
import numpy as np
import sys, os
import time
import json

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.append(current_directory)

# camera
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

# gripper
from aimdk.protocol.sim import sim_gripper_service_pb2
from aimdk.protocol.sim import sim_gripper_service_pb2_grpc

# object
from aimdk.protocol.sim import sim_object_service_pb2
from aimdk.protocol.sim import sim_object_service_pb2_grpc

# observation
from aimdk.protocol.sim import sim_observation_service_pb2
from aimdk.protocol.sim import sim_observation_service_pb2_grpc


from geniesim.utils.logger import Logger

logger = Logger()  # Create singleton instance


# All rotation angles in the current code are in units of angles
class Rpc_Client:
    def __init__(self, client_host, robot_urdf="G1_120s.urdf"):
        for i in range(600):
            try:
                self.channel = grpc.insecure_channel(
                    client_host,
                    options=[("grpc.max_receive_message_length", 50 * 1024 * 1024)],
                )
                grpc.channel_ready_future(self.channel).result(timeout=5)
                self.robot_urdf = robot_urdf
                break
            except grpc.FutureTimeoutError as e:
                logger.error(f"Failed to connect to gRPC server[{i}]: {e}")
                time.sleep(3)
                if i >= 599:
                    raise e
            except grpc.RpcError as e:
                logger.error(f"Failed to connect to gRPC server[{i}]: {e}")
                time.sleep(3)
                if i >= 599:
                    raise e

    def set_frame_state(
        self,
        action: str,
        substage_id: int,
        active_id: str,
        passive_id: str,
        if_attached: bool,
    ):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetFrameStateReq()
        frame_state = {
            "action": action,
            "substage_id": substage_id,
            "if_attached": if_attached,
            "active_id": active_id,
            "passive_id": passive_id,
        }
        _frame_state: str = json.dumps(frame_state)
        # logger.info(_frame_state)
        req.frame_state = _frame_state
        response = stub.SetFrameState(req)
        return response

    def capture_frame(self, camera_prim_path):
        stub = rs2_camera_pb2_grpc.CameraServiceStub(self.channel)
        req = rs2_camera_pb2.GetCameraDataRequest()
        req.serial_no = camera_prim_path
        response = stub.GetCameraData(req)
        return response

    def capture_semantic_frame(self, camera_prim_path):
        stub = sim_camera_service_pb2_grpc.SimCameraServiceStub(self.channel)
        req = sim_camera_service_pb2.GetSemanticRequest()
        req.serial_no = "/World/G1/base/collisions/camera"
        req.serial_no = camera_prim_path
        response = stub.GetSemanticData(req)

        return response

    def moveto(
        self,
        target_position,
        target_quaternion,
        arm_name,
        is_backend=True,
        ee_interpolation=False,
        distance_frame=0.0008,
    ):
        stub = arm_pb2_grpc.G1ArmControlServiceStub(self.channel)
        req = arm_pb2.LinearMoveReq()
        req.robot_name = arm_name
        req.pose.position.x, req.pose.position.y, req.pose.position.z = target_position
        req.pose.rpy.rw, req.pose.rpy.rx, req.pose.rpy.ry, req.pose.rpy.rz = (
            target_quaternion
        )
        req.is_block = is_backend
        req.ee_interpolation = ee_interpolation
        req.distance_frame = distance_frame
        response = stub.LinearMove(req)
        return response

    def set_joint_positions(
        self, target_joint_position, is_trajectory, joint_indices=None
    ):
        stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        req = joint_channel_pb2.SetJointReq()
        req.is_trajectory = is_trajectory
        for idx, pos in enumerate(target_joint_position):
            joint_position = joint_channel_pb2.JointCommand()
            if joint_indices is None:
                joint_position.sequence = idx
            else:
                joint_position.sequence = joint_indices[idx]
            joint_position.position = pos
            req.commands.append(joint_position)
        response = stub.SetJointPosition(req)
        return response

    def get_joint_positions(self):
        stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        req = joint_channel_pb2.GetJointReq()
        req.serial_no = "robot"
        response = stub.GetJointPosition(req)
        return response

    # The current value of usd_path is set to the directory under data such as genie3D/01.usd
    # prim_path needs to be specified in "/World/Objects/xxx" to facilitate reset
    def add_object(
        self,
        usd_path,
        prim_path,
        label_name,
        target_position,
        target_quaternion,
        target_scale,
        color,
        material,
        mass,
        add_particle=True,
        particle_position=[0, 0, 0],
        particle_scale=[0.1, 0.1, 0.1],
        particle_color=[1, 1, 1],
        com=[0, 0, 0],
        model_type="convexDecomposition",
        static_friction=0.5,
        dynamic_friction=0.5,
    ):
        stub = sim_object_service_pb2_grpc.SimObjectServiceStub(self.channel)
        req = sim_object_service_pb2.AddObjectReq()
        req.usd_path = usd_path
        req.prim_path = prim_path
        req.label_name = label_name
        req.object_color.r, req.object_color.g, req.object_color.b = color
        (
            req.object_pose.position.x,
            req.object_pose.position.y,
            req.object_pose.position.z,
        ) = target_position
        (
            req.object_pose.rpy.rw,
            req.object_pose.rpy.rx,
            req.object_pose.rpy.ry,
            req.object_pose.rpy.rz,
        ) = target_quaternion
        req.object_scale.x, req.object_scale.y, req.object_scale.z = target_scale
        req.object_material = material
        req.object_mass = mass
        req.object_com.x, req.object_com.y, req.object_com.z = com
        req.add_particle = add_particle
        req.particle_position.x, req.particle_position.y, req.particle_position.z = (
            particle_position
        )
        req.particle_scale.x, req.particle_scale.y, req.particle_scale.z = (
            particle_scale
        )
        req.particle_color.r, req.particle_color.g, req.particle_color.b = (
            particle_color
        )
        req.model_type = model_type
        req.static_friction = static_friction
        req.dynamic_friction = dynamic_friction
        response = stub.AddObject(req)
        return response

    def get_object_pose(self, prim_path):
        stub = sim_object_service_pb2_grpc.SimObjectServiceStub(self.channel)
        req = sim_object_service_pb2.GetObjectPoseReq()
        req.prim_path = prim_path
        response = stub.GetObjectPose(req)
        return response

    def get_object_joint(self, prim_path):
        stub = sim_object_service_pb2_grpc.SimObjectServiceStub(self.channel)
        req = sim_object_service_pb2.GetObjectJointReq()
        req.prim_path = prim_path
        response = stub.GetObjectJoint(req)
        return response

    def set_gripper_state(self, gripper_command, is_right, opened_width):
        stub = sim_gripper_service_pb2_grpc.SimGripperServiceStub(self.channel)
        req = sim_gripper_service_pb2.SetGripperStateReq()
        req.gripper_command = gripper_command
        req.is_right = is_right
        req.opened_width = opened_width
        response = stub.SetGripperState(req)
        return response

    def get_gripper_state(self, is_right):
        stub = sim_gripper_service_pb2_grpc.SimGripperServiceStub(self.channel)
        req = sim_gripper_service_pb2.GetGripperStateReq()
        req.is_right = is_right
        response = stub.GetGripperState(req)
        return response

    # Client side gets all observations of a certain frame
    def get_observation(self, data_keys):
        observation = {}
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetObservationReq()
        if "camera" in data_keys:
            req.isCam = True
            req.CameraReq.render_depth = data_keys["camera"]["render_depth"]
            req.CameraReq.render_semantic = data_keys["camera"]["render_semantic"]
            for prim in data_keys["camera"]["camera_prim_list"]:
                req.CameraReq.camera_prim_list.append(prim)
        if "pose" in data_keys:
            req.isPose = True
            for pose in data_keys["pose"]:
                req.objectPrims.append(pose)
        req.isJoint = data_keys["joint_position"]
        req.isGripper = data_keys["gripper"]
        response = stub.GetObservation(req)

        # protobuf to dictionary
        camera_datas = {}
        for camera_data, camera_prim in zip(
            response.camera, data_keys["camera"]["camera_prim_list"]
        ):
            cam_data = {
                "rgb_camera": np.frombuffer(
                    camera_data.rgb_camera.data, dtype=np.uint8
                ),
                "depth_camera": np.frombuffer(
                    camera_data.depth_camera.data, dtype=np.float32
                ),
                "camera_info": {
                    "width": camera_data.camera_info.width,
                    "height": camera_data.camera_info.height,
                    "ppx": camera_data.camera_info.ppx,
                    "ppy": camera_data.camera_info.ppy,
                    "fx": camera_data.camera_info.fx,
                    "fy": camera_data.camera_info.fy,
                },
            }
            camera_datas[camera_prim] = cam_data
        joint_datas = {}
        for joint in response.joint.left_arm:
            joint_datas[joint.name] = joint.position
        object_datas = {}
        if "pose" in data_keys:
            for object, obj_name in zip(response.pose, data_keys["pose"]):
                object_data = {
                    "position": np.array(
                        [
                            object.object_pose.position.x,
                            object.object_pose.position.y,
                            object.object_pose.position.z,
                        ]
                    ),
                    "rotation": np.array(
                        [
                            object.object_pose.rpy.rw,
                            object.object_pose.rpy.rx,
                            object.object_pose.rpy.ry,
                            object.object_pose.rpy.rz,
                        ]
                    ),
                }
                object_datas[obj_name] = object_data
        gripper_datas = {
            "left": {
                "position": np.array(
                    [
                        response.gripper.left_gripper.position.x,
                        response.gripper.left_gripper.position.y,
                        response.gripper.left_gripper.position.z,
                    ]
                ),
                "rotation": np.array(
                    [
                        response.gripper.left_gripper.rpy.rw,
                        response.gripper.left_gripper.rpy.rx,
                        response.gripper.left_gripper.rpy.ry,
                        response.gripper.left_gripper.rpy.rz,
                    ]
                ),
            },
            "right": {
                "position": np.array(
                    [
                        response.gripper.right_gripper.position.x,
                        response.gripper.right_gripper.position.y,
                        response.gripper.right_gripper.position.z,
                    ]
                ),
                "rotation": np.array(
                    [
                        response.gripper.left_gripper.rpy.rw,
                        response.gripper.right_gripper.rpy.rx,
                        response.gripper.right_gripper.rpy.ry,
                        response.gripper.right_gripper.rpy.rz,
                    ]
                ),
            },
        }
        observation = {
            "camera": camera_datas,
            "joint": joint_datas,
            "pose": object_datas,
            "gripper": gripper_datas,
        }
        return observation

    # Client starts recording
    def start_recording(self, data_keys, fps, task_name):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetObservationReq()
        req.startRecording = True
        req.fps = fps
        req.task_name = task_name
        if "camera" in data_keys:
            req.isCam = True
            req.CameraReq.render_depth = data_keys["camera"]["render_depth"]
            req.CameraReq.render_semantic = data_keys["camera"]["render_semantic"]
            for prim in data_keys["camera"]["camera_prim_list"]:
                req.CameraReq.camera_prim_list.append(prim)
        if data_keys["pose"]:
            req.isPose = True
            for pose in data_keys["pose"]:
                req.objectPrims.append(pose)
        req.isJoint = data_keys["joint_position"]
        req.isGripper = data_keys["gripper"]
        response = stub.GetObservation(req)
        return response

    # Client-side start and end recording
    def stop_recording(self):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetObservationReq()
        req.stopRecording = True
        response = stub.GetObservation(req)
        return response

    def reset(self):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.ResetReq()
        req.reset = True
        response = stub.Reset(req)
        return response

    def AttachObj(self, prim_paths):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.AttachReq()
        for prim in prim_paths:
            req.obj_prims.append(prim)
        response = stub.AttachObj(req)
        return response

    def DetachObj(self):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.DetachReq()
        req.detach = True
        response = stub.DetachObj(req)
        return response

    def MultiPlan(self, robot_name, target_poses):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.MultiMoveReq()
        req.plan = True
        req.robot_name = robot_name
        req.plans_index = 0
        for pose in target_poses:
            _pose = sim_observation_service_pb2.SE3RpyPose()
            _pose.position.x, _pose.position.y, _pose.position.z = pose[0]
            _pose.rpy.rw, _pose.rpy.rx, _pose.rpy.ry, _pose.rpy.rz = pose[1]
            req.poses.append(_pose)
        response = stub.MultiMove(req)
        return response

    def MultiMove(self, robot_name, cmd_plan):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.MultiMoveReq()
        req.plan = False
        req.robot_name = robot_name
        req.cmd_plan = cmd_plan
        response = stub.MultiMove(req)
        return response

    def SendTaskStatus(self, isSuccess, fail_stage_step):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.TaskStatusReq()
        for step in fail_stage_step:
            req.failStep.append(step)
        req.isSuccess = isSuccess
        response = stub.TaskStatus(req)
        return response

    def Exit(self):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.ExitReq()
        req.exit = True
        response = stub.Exit(req)
        return response

    def GetEEPose(self, is_right):
        stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        req = joint_channel_pb2.GetEEPoseReq()
        req.is_right = is_right
        response = stub.GetEEPose(req)
        return response

    def GetIKStatus(self, target_poses, is_right, ObsAvoid=False):
        import pinocchio

        stub = joint_channel_pb2_grpc.JointControlServiceStub(self.channel)
        req = joint_channel_pb2.GetIKStatusReq()
        req.is_right = is_right
        req.ObsAvoid = ObsAvoid
        urdf = (
            os.path.dirname(os.path.abspath(__file__))
            + "/robot_urdf/"
            + self.robot_urdf
        )
        model = pinocchio.buildModelFromUrdf(urdf)
        joint_positions = pinocchio.neutral(model)
        data = model.createData()
        joint_names = []
        for name in model.names:
            if name != "universe":
                joint_names.append(str(name))

        for pose in target_poses:
            _pose = sim_observation_service_pb2.SE3RpyPose()
            _pose.position.x, _pose.position.y, _pose.position.z = pose["position"]
            _pose.rpy.rw, _pose.rpy.rx, _pose.rpy.ry, _pose.rpy.rz = pose["rotation"]
            req.target_pose.append(_pose)
        response = stub.GetIKStatus(req)
        result = []
        for ik_status in response.IKStatus:
            joint_datas = {}
            for name, position in zip(ik_status.joint_names, ik_status.joint_positions):
                joint_datas[name] = position

            joint_positions = []
            for name in model.names:
                if name == "universe":
                    continue
                if name not in joint_datas:
                    joint_positions.append(0)
                else:
                    joint_positions.append(joint_datas[name])
            joint_positions = np.array(joint_positions)

            pinocchio.forwardKinematics(model, data, joint_positions)
            if "G1" in self.robot_urdf:
                J = pinocchio.computeJointJacobian(model, data, joint_positions, 24)
            else:
                J = pinocchio.computeJointJacobian(model, data, joint_positions, 7)
            manip = np.sqrt(np.linalg.det(np.dot(J, J.T)))
            result.append(
                {
                    "status": ik_status.isSuccess,
                    "Jacobian": manip,
                    "joint_positions": joint_positions,
                    "joint_names": joint_names,
                }
            )
        return result

    def GetManipulability(self, joint_data):
        import pinocchio

        urdf = (
            os.path.dirname(os.path.abspath(__file__))
            + "/robot_urdf/"
            + self.robot_urdf
        )
        model = pinocchio.buildModelFromUrdf(urdf)
        data = model.createData()
        joint_id = 24
        joint_positions = []
        for name in model.names:
            if name == "universe":
                continue
            # if name not in joint_data:
            if name not in [
                "joint_lift_body",
                "joint_body_pitch",
                "Joint1_r",
                "Joint2_r",
                "Joint3_r",
                "Joint4_r",
                "Joint5_r",
                "Joint6_r",
                "Joint7_r",
                "right_Left_1_Joint",
                "right_Left_0_Joint",
                "right_Left_Support_Joint",
                "right_Right_1_Joint",
                "right_Right_0_Joint",
                "right_Right_Support_Joint",
            ]:
                joint_positions.append(0)
            else:
                joint_positions.append(joint_data[name])
        joint_positions = np.array(joint_positions)
        # logger.info(joint_positions)
        pinocchio.forwardKinematics(model, data, joint_positions)
        J = pinocchio.computeFrameJacobian(model, data, joint_positions, joint_id)
        manip = np.sqrt(np.linalg.det(np.dot(J, J.T)))
        return manip

    def GetObjectsOfType(self, obj_type):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetObjectsOfTypeReq()
        req.obj_type = obj_type
        response = stub.GetObjectsOfType(req)
        return response

    def AddCamera(
        self,
        camera_prim,
        camera_position,
        camera_rotation,
        width,
        height,
        focus_length,
        horizontal_aperture,
        vertical_aperture,
        is_local,
    ):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.AddCameraReq()
        req.camera_prim = camera_prim
        (
            req.camera_pose.position.x,
            req.camera_pose.position.y,
            req.camera_pose.position.z,
        ) = camera_position
        (
            req.camera_pose.rpy.rw,
            req.camera_pose.rpy.rx,
            req.camera_pose.rpy.ry,
            req.camera_pose.rpy.rz,
        ) = camera_rotation
        req.focus_length = focus_length
        req.horizontal_aperture = horizontal_aperture
        req.vertical_aperture = vertical_aperture
        req.width = width
        req.height = height
        req.is_local = is_local
        response = stub.AddCamera(req)
        return response

    def InitRobot(
        self,
        robot_cfg,
        robot_usd,
        scene_usd,
        init_position=[0, 0, 0],
        init_rotation=[1, 0, 0, 0],
    ):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.InitRobotReq()
        req.robot_cfg_file, req.robot_usd_path, req.scene_usd_path = (
            robot_cfg,
            robot_usd,
            scene_usd,
        )
        (
            req.robot_pose.position.x,
            req.robot_pose.position.y,
            req.robot_pose.position.z,
        ) = init_position
        (
            req.robot_pose.rpy.rw,
            req.robot_pose.rpy.rx,
            req.robot_pose.rpy.ry,
            req.robot_pose.rpy.rz,
        ) = init_rotation
        response = stub.InitRobot(req)
        return response

    def DrawLine(self, point_list_1, point_list_2, colors, sizes):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.DrawLineReq()
        for point in point_list_1:
            _point = sim_observation_service_pb2.Vec3()
            _point.x, _point.y, _point.z = point
            req.point_list_1.append(_point)
        for point in point_list_2:
            _point = sim_observation_service_pb2.Vec3()
            _point.x, _point.y, _point.z = point
            req.point_list_2.append(_point)
        for color in colors:
            _color = sim_observation_service_pb2.Vec3()
            _color.x, _color.y, _color.z = color
            req.colors.append(_color)
        for size in sizes:
            req.sizes.append(size)
        response = stub.DrawLine(req)
        return response

    def SetObjectPose(self, object_info, joint_cmd, object_joints=[]):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetObjectPoseReq()
        for object in object_info:
            object_pose = sim_observation_service_pb2.ObjectPose()
            object_pose.prim_path = object["prim_path"]
            (
                object_pose.pose.position.x,
                object_pose.pose.position.y,
                object_pose.pose.position.z,
            ) = object["position"]
            (
                object_pose.pose.rpy.rw,
                object_pose.pose.rpy.rx,
                object_pose.pose.rpy.ry,
                object_pose.pose.rpy.rz,
            ) = object["rotation"]
            req.object_pose.append(object_pose)
        for pos in joint_cmd:
            cmd = sim_observation_service_pb2.JointCommand()
            cmd.position = pos
            req.joint_cmd.append(cmd)

        for j in object_joints:
            oj = sim_observation_service_pb2.ObjectJoint()
            oj.prim_path = j["prim_path"]
            for jp in j["joint_cmd"]:
                cmd = sim_observation_service_pb2.JointCommand()
                cmd.position = jp
                oj.joint_cmd.append(cmd)
            req.object_joint.append(oj)

        response = stub.SetObjectPose(req)
        return response

    def SetTrajectoryList(self, trajectory_list, is_block=False):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetTrajectoryListReq()
        req.is_block = is_block
        for point in trajectory_list:
            pose = sim_observation_service_pb2.SE3RpyPose()
            pose.position.x, pose.position.y, pose.position.z = point[0]
            pose.rpy.rw, pose.rpy.rx, pose.rpy.ry, pose.rpy.rz = point[1]
            req.trajectory_point.append(pose)
        response = stub.SetTrajectoryList(req)
        return response

    def SetTargetPoint(self, position):
        stub = sim_object_service_pb2_grpc.SimObjectServiceStub(self.channel)
        req = sim_object_service_pb2.SetTargetPointReq()
        req.point_position.x, req.point_position.y, req.point_position.z = position
        response = stub.SetTargetPoint(req)
        return response

    def SetMaterial(self, material_info):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetMaterailReq()
        for mat in material_info:
            logger.debug(mat)
            mat_info = sim_observation_service_pb2.MaterialInfo()
            mat_info.object_prim = mat["object_prim"]
            mat_info.material_name = mat["material_name"]
            mat_info.material_path = mat["material_path"]
            req.materials.append(mat_info)
        response = stub.SetMaterial(req)
        return response

    def SetLight(self, light_info):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.SetLightReq()
        for light in light_info:
            logger.debug(light)
            light_cfg = sim_observation_service_pb2.LightCfg()
            light_cfg.light_type = light["light_type"]
            light_cfg.light_prim = light["light_prim"]
            light_cfg.light_temperature = light["light_temperature"]
            light_cfg.light_intensity = light["light_intensity"]
            (
                light_cfg.light_rotation.rw,
                light_cfg.light_rotation.rx,
                light_cfg.light_rotation.ry,
                light_cfg.light_rotation.rz,
            ) = light["rotation"]
            light_cfg.light_texture = light["texture"]
            req.lights.append(light_cfg)
        response = stub.SetLight(req)
        return response

    def OmniCmdChangeProperty(self, prop_path, value):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.OmniCmdChangePropertyReq()
        req.prop_path = prop_path
        if isinstance(value, bool):
            req.bool_value = value
        if isinstance(value, str):
            req.str_value = value
        response = stub.OmniCmdChangeProperty(req)
        return response

    def GetPartiPointNumInbbox(self, prim_path, bbox=None):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetPartiPointNumInbboxReq()
        req.prim_path = prim_path
        for v in bbox:
            req.bbox.append(v)

        response = stub.GetPartiPointNumInbbox(req)
        return response

    def GetObjectAABB(self, prim_path):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetObjectAABBReq()
        req.prim_path = prim_path

        response = stub.GetObjectAABB(req)
        return response

    def GetWorldPose(self, prim_path):
        stub = sim_observation_service_pb2_grpc.SimObservationServiceStub(self.channel)
        req = sim_observation_service_pb2.GetWorldPoseReq()
        req.prim_path = prim_path

        response = stub.GetWorldPose(req)
        return response


# Call Example
def run():
    rpc_client = Rpc_Client(client_host="localhost:50051")
    try:
        while True:
            send_msg = input("input Command: ")
            if send_msg == "30":
                result = rpc_client.GetEEPose(is_right=True)
                logger.info(result)
            if send_msg == "point":
                rpc_client.SetTargetPoint([0, 1, 1])
            if send_msg == "test_mat":
                rpc_client.SetMaterial(
                    [
                        {
                            "object_prim": "/World/huxing/a_keting/yangbanjian_A_016",
                            "material_name": "Ash",
                            "material_path": "materials/wood/Ash",
                        },
                        {
                            "object_prim": "/World/huxing/a_Base/Floor_003",
                            "material_name": "Ash",
                            "material_path": "materials/wood/Ash",
                        },
                    ]
                )
                rpc_client.SetLight(
                    [
                        {
                            "light_type": "Distant",
                            "light_prim": "/World/huxing/DistantLight",
                            "light_temperature": 2000,
                            "light_intensity": 1000,
                            "rotation": [1, 0.5, 0.5, 0.5],
                            "texture": "",
                        },
                        {
                            "light_type": "Dome",
                            "light_prim": "/World/DomeLight",
                            "light_temperature": 6500,
                            "light_intensity": 1000,
                            "rotation": [1, 0, 0, 0],
                            "texture": "materials/hdri/abandoned_hall_01_4k.hdr",
                        },
                        {
                            "light_type": "Rect",
                            "light_prim": "/World/RectLight",
                            "light_temperature": 4500,
                            "light_intensity": 1000,
                            "rotation": [1, 0, 0, 0],
                            "texture": "",
                        },
                        {
                            "light_type": "Rect",
                            "light_prim": "/World/RectLight_01",
                            "light_temperature": 8500,
                            "light_intensity": 1000,
                            "rotation": [1, 0, 0, 0],
                            "texture": "",
                        },
                    ]
                )

            if send_msg == "908":
                pose_1 = [1, 1, 1], [1, 0, 0, 0]
                pose_2 = [1, 0, 1], [1, 0, 0, 0]
                trajecory_list = [pose_1, pose_2] * 10
                result = rpc_client.SetTrajectoryList(trajecory_list, True)
            if send_msg == "31":
                target_poses = []
                for i in range(1):
                    x = 0.57597359649470348
                    y = -0.45669529659303565
                    z = 1.0198275517174573
                    rw = 0.066194084385260935
                    rx = 0.70713063274436749
                    ry = 0.70071350186850612
                    rz = 0.0677141028599091
                    pose = {"position": [x, y, z], "rotation": [rw, rx, ry, rz]}
                    target_poses.append(pose)

                result = rpc_client.GetIKStatus(
                    target_poses=target_poses, is_right=False
                )
                logger.info(result)
            if send_msg == "1":
                # camera
                rpc_client.capture_frame(
                    camera_prim_path="/World/G1/base_link/Head_Camera"
                )
                result = rpc_client.capture_semantic_frame(
                    camera_prim_path="/World/G1/base_link/Head_Camera"
                )
                logger.info(result)
            if send_msg == "2":
                # Arm
                x = np.random.uniform(0.3, 0.7)
                y = np.random.uniform(-0.3, 0)
                z = np.random.uniform(0.1, 0.5)
                rpc_client.moveto(
                    target_position=[x, y, z],
                    target_quaternion=np.array([0.0, -0.0, -1.0, 0.0]),
                    arm_name="left",
                    is_backend=True,
                    ee_interpolation=True,
                )
            if send_msg == "20":
                result = rpc_client.MultiPlan(
                    robot_name="left",
                    target_poses=[
                        [
                            [
                                0.5169683509992052,
                                0.1313259611510117,
                                1.1018942820728093,
                            ],
                            [0.40020943, 0.57116637, 0.69704651, -0.16651593],
                        ],
                        [
                            [
                                0.5610938560120418,
                                0.048608636026916924,
                                1.0269891277236924,
                            ],
                            [0.40020943, 0.57116637, 0.69704651, -0.16651593],
                        ],
                        [
                            [
                                0.5610938560120418,
                                0.048608636026916924,
                                1.2269891277236924,
                            ],
                            [0.40020943, 0.57116637, 0.69704651, -0.16651593],
                        ],
                    ],
                )
                logger.info(result[0])

            if send_msg == "44":
                import time

                # test cabinet
                rpc_client.InitRobot(
                    "G1_120s.json",
                    "G1.usd",
                    "omnimanip_Simple_Room_01/simple_room.usd",
                    init_position=[-0.4, 0, -0.55],
                )
                rpc_client.reset()
                time.sleep(1)
                rpc_client.DetachObj()
                rpc_client.reset()
                time.sleep(1)
                rpc_client.add_object(
                    usd_path="objects/guanglun/storagefurniture/storagefurniture011/Storagefurniture011.usd",
                    prim_path="/World/Objects/storagefurniture",
                    label_name="test_storagefurniture",
                    target_position=np.array([0.64, -0.3, 0.40]),
                    target_quaternion=np.array([0.0, 0.0, 0.70711, 0.70711]),
                    target_scale=np.array([1, 1, 1]),
                    color=np.array([1, 0, 1]),
                    material="Plastic",
                    mass=0.01,
                )
                time.sleep(1)
                rpc_client.set_gripper_state(
                    gripper_command="open", is_right=True, opened_width=1
                )

                target_poses = []
                start_time = time.time()
                for i in range(50):
                    x = np.random.uniform(0, 0.2)
                    y = np.random.uniform(-0.3, 0.3)
                    z = np.random.uniform(0.96, 1.2)
                    pose = {"position": [x, y, z], "rotation": [1, x, y, z]}
                    target_poses.append(pose)
                result = rpc_client.GetIKStatus(
                    target_poses=target_poses, is_right=False
                )
                result = rpc_client.GetIKStatus(
                    target_poses=target_poses, is_right=True
                )
                rpc_client.DetachObj()
                logger.info("open")
                rpc_client.get_object_pose(
                    "/World/Objects/storagefurniture/Storagefurniture011_Handle1"
                )
                rpc_client.get_object_pose(
                    "/World/Objects/storagefurniture/Storagefurniture011_Handle2"
                )
                obj_pose = rpc_client.get_object_pose(
                    "/World/Objects/storagefurniture/Storagefurniture011_Handle3"
                )
                obj_xyz = np.array(
                    [
                        obj_pose.object_pose.position.x,
                        obj_pose.object_pose.position.y,
                        obj_pose.object_pose.position.z,
                    ]
                )
                ee_pose = rpc_client.GetEEPose(is_right=True)
                ee_quat = np.array(
                    [
                        ee_pose.ee_pose.rpy.rw,
                        ee_pose.ee_pose.rpy.rx,
                        ee_pose.ee_pose.rpy.ry,
                        ee_pose.ee_pose.rpy.rz,
                    ]
                )
                goal_xyz = obj_xyz.copy()
                goal_xyz[0] = goal_xyz[0] + 0.3
                goal_xyz[2] = goal_xyz[2] + 0.55
                logger.info(goal_xyz)
                rpc_client.moveto(
                    target_position=goal_xyz,
                    target_quaternion=ee_quat,
                    arm_name="left",
                    is_backend=False,
                    ee_interpolation=False,
                    distance_frame=0.001,
                )
                obj_pose = rpc_client.get_object_pose(
                    "/World/Objects/storagefurniture/Storagefurniture011_Handle3"
                )
                obj_xyz = np.array(
                    [
                        obj_pose.object_pose.position.x,
                        obj_pose.object_pose.position.y,
                        obj_pose.object_pose.position.z,
                    ]
                )
                ee_pose = rpc_client.GetEEPose(is_right=True)
                ee_quat = np.array(
                    [
                        ee_pose.ee_pose.rpy.rw,
                        ee_pose.ee_pose.rpy.rx,
                        ee_pose.ee_pose.rpy.ry,
                        ee_pose.ee_pose.rpy.rz,
                    ]
                )
                goal_xyz = obj_xyz.copy()
                rpc_client.get_object_pose(
                    "/World/Objects/storagefurniture/Storagefurniture011_Handle1"
                )
                rpc_client.get_object_pose(
                    "/World/Objects/storagefurniture/Storagefurniture011_Handle2"
                )
                rpc_client.moveto(
                    target_position=goal_xyz,
                    target_quaternion=ee_quat,
                    arm_name="left",
                    is_backend=True,
                    ee_interpolation=False,
                    distance_frame=0.001,
                )
                rpc_client.set_gripper_state(
                    gripper_command="close", is_right=True, opened_width=0.02
                )
                time.sleep(1)
                obj_pose = rpc_client.get_object_pose(
                    "/World/Objects/storagefurniture/Storagefurniture011_Handle3"
                )
                obj_xyz = np.array(
                    [
                        obj_pose.object_pose.position.x,
                        obj_pose.object_pose.position.y,
                        obj_pose.object_pose.position.z,
                    ]
                )
                ee_pose = rpc_client.GetEEPose(is_right=True)
                ee_quat = np.array(
                    [
                        ee_pose.ee_pose.rpy.rw,
                        ee_pose.ee_pose.rpy.rx,
                        ee_pose.ee_pose.rpy.ry,
                        ee_pose.ee_pose.rpy.rz,
                    ]
                )
                goal_xyz = obj_xyz.copy()
                goal_xyz[0] = goal_xyz[0] - 0.1
                rpc_client.moveto(
                    target_position=goal_xyz,
                    target_quaternion=ee_quat,
                    arm_name="left",
                    is_backend=True,
                    ee_interpolation=False,
                    distance_frame=0.001,
                )
            if send_msg == "21":
                rpc_client.MultiMove(robot_name="left", plan_index=0)
            if send_msg == "22":
                rpc_client.MultiMove(robot_name="left", plan_index=1)
            if send_msg == "23":
                rpc_client.MultiMove(robot_name="left", plan_index=2)
            if send_msg == "24":
                rpc_client.start_recording()

            if send_msg == "3":
                # joint
                joint_states = rpc_client.get_joint_positions().states
                joint_datas = {}
                for joint in joint_states:
                    joint_datas[joint.name] = joint.position
                result = rpc_client.GetManipulability(joint_datas)
                logger.info(result)
            if send_msg == "4":
                # hand
                rpc_client.set_gripper_state(
                    gripper_command="open", is_right=True, opened_width=0.1
                )
            if send_msg == "5":
                # object

                import time

                # rpc_client.reset()
                rpc_client.add_object(
                    usd_path="Collected_cabinet_000/cabinet_000.usd",
                    prim_path="/World/Objects/cabinet",
                    label_name="test_cabinet",
                    target_position=np.array([0.8, 0.0, -0.1]),
                    target_quaternion=np.array([0.0, 0.0, 0.70711, 0.70711]),
                    target_scale=np.array([1, 1, 1]),
                    color=np.array([1, 0, 1]),
                    material="Plastic",
                    mass=0.01,
                )
                time.sleep(1)
                result = rpc_client.get_object_joint("/World/Objects/cabinet")
                logger.info(result)
            if send_msg == "6":
                # Get all information
                rpc_client.get_joint_positions()
            if send_msg == "7":
                rpc_client.set_gripper_state(
                    gripper_command="close", is_right=True, opened_width=0.02
                )
            if send_msg == "8":
                result = rpc_client.get_observation(
                    data_keys={
                        "camera": {
                            "camera_prim_list": ["/World/G1/base_link/Head_Camera"],
                            "render_depth": True,
                            "render_semantic": True,
                        },
                        "pose": ["/World/G1/base_link/Head_Camera"],
                        "joint_position": True,
                        "gripper": True,
                    }
                )
                logger.info(result["camera"])
            if send_msg == "9":
                rpc_client.start_recording(
                    data_keys={
                        "camera": {
                            "camera_prim_list": ["/World/G1/base_link/Head_Camera"],
                            "render_depth": True,
                            "render_semantic": True,
                        },
                        "pose": ["/World/G1/base_link/Head_Camera"],
                        "joint_position": True,
                        "gripper": True,
                    },
                    fps=30,
                    task_name="test",
                )
                time.sleep(1)
                rpc_client.stop_recording()
                rpc_client.SendTaskStatus(False)
            if send_msg == "10":
                rpc_client.stop_recording()
            if send_msg == "11":
                rpc_client.reset()
            if send_msg == "13":
                result = rpc_client.Exit()
                logger.info(result)

            if send_msg == "112":
                rpc_client.InitRobot(
                    robot_cfg="G1.json",
                    robot_usd="G1/G1.usd",
                    scene_usd="Pick_Place_G1_Yellow_Table.usd",
                )

            if send_msg == "113":
                position = [1.2, 0.0, 1.2]
                rotation = [0.61237, 0.35355, 0.35355, 0.61237]
                rotation = [0.65328, 0.2706, 0.2706, 0.65328]
                width = 640
                height = 480
                rpc_client.AddCamera(
                    "/World/Sensors/Head_Camera",
                    position,
                    rotation,
                    width,
                    height,
                    18.14756,
                    20.955,
                    15.2908,
                    False,
                )
                response = rpc_client.capture_frame(
                    camera_prim_path="/World/Sensors/Head_Camera"
                )
                # cam_info
                cam_info = {
                    "W": response.color_info.width,
                    "H": response.color_info.height,
                    "K": np.array(
                        [
                            [response.color_info.fx, 0, response.color_info.ppx],
                            [0, response.color_info.fy, response.color_info.ppy],
                            [0, 0, 1],
                        ]
                    ),
                    "scale": 1,
                }
            if send_msg == "114":
                position = [0.05, 0.0, 0.0]
                rotation = [0.06163, 0.70442, 0.70442, 0.06163]
                width = 640
                height = 480
                rpc_client.AddCamera(
                    "/panda/panda_hand/Hand_Camera_1",
                    position,
                    rotation,
                    width,
                    height,
                    18.14756,
                    20.955,
                    15.2908,
                    True,
                )
                response = rpc_client.capture_frame(
                    camera_prim_path="/panda/panda_hand/Hand_Camera_1"
                )
                # cam_info
                cam_info = {
                    "W": response.color_info.width,
                    "H": response.color_info.height,
                    "K": np.array(
                        [
                            [response.color_info.fx, 0, response.color_info.ppx],
                            [0, response.color_info.fy, response.color_info.ppy],
                            [0, 0, 1],
                        ]
                    ),
                    "scale": 1,
                }
                # rgb
                rgb = np.frombuffer(response.color_image.data, dtype=np.uint8).reshape(
                    cam_info["H"], cam_info["W"], 4
                )[:, :, :3]
                import cv2

                cv2.imwrite("hand_img.png", rgb)

    except Exception as e:
        logger.error("failed.{}".format(e))
        return False


if __name__ == "__main__":
    run()
