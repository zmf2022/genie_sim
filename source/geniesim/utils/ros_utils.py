# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)
from rclpy.node import Node
from rclpy.parameter import Parameter

from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Bool
from collections import deque
import threading, time

import cv2
from cv_bridge import CvBridge
import numpy as np

QOS_PROFILE_LATEST = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)

QOS_PROFILE_VOLATILE = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=30,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class TimerROSNode(Node):
    def __init__(self, node_name="timer_node"):
        super().__init__(node_name)
        self.loop_rate = self.create_rate(30.0)


class SimROSNode(Node):
    def __init__(self, robot_cfg, node_name="sim_publisher"):
        super().__init__(
            node_name,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        # core
        self.robot_cfg = robot_cfg
        self.init_pose = None
        self.bridge = CvBridge()

        # pub
        self.pub_head_img = self.create_publisher(
            CompressedImage,
            "/sim/head_img",
            QOS_PROFILE_LATEST,
        )
        self.pub_head_depth_img = self.create_publisher(
            CompressedImage,
            "/sim/head_depth_img",
            QOS_PROFILE_LATEST,
        )
        self.pub_left_wrist_img = self.create_publisher(
            CompressedImage,
            "/sim/left_wrist_img",
            QOS_PROFILE_LATEST,
        )
        self.pub_left_wrist_depth_img = self.create_publisher(
            CompressedImage,
            "/sim/left_wrist_depth_img",
            QOS_PROFILE_LATEST,
        )
        self.pub_right_wrist_img = self.create_publisher(
            CompressedImage,
            "/sim/right_wrist_img",
            QOS_PROFILE_LATEST,
        )
        self.pub_right_wrist_depth_img = self.create_publisher(
            CompressedImage,
            "/sim/right_wrist_depth_img",
            QOS_PROFILE_LATEST,
        )
        self.pub_joint_state = self.create_publisher(
            JointState,
            "/sim/joint_state",
            QOS_PROFILE_LATEST,
        )
        self.pub_joint_command = self.create_publisher(
            JointState,
            "/joint_command",
            QOS_PROFILE_LATEST,
        )
        self.pub_infer_start = self.create_publisher(
            Bool,
            "/sim/infer_start",
            QOS_PROFILE_LATEST,
        )

        # sub
        self.sub_js = self.create_subscription(
            JointState,
            "/sim/target_joint_state",
            self.callback_joint_command,
            QOS_PROFILE_LATEST,
        )
        self.sub_js = self.create_subscription(
            JointState,
            "/joint_states",
            self.callback_joint_state,
            1,
        )
        self.sub_img_l = self.create_subscription(
            Image,
            "/genie_sim/Left_Camera_rgb",
            self.callback_rgb_image_l,
            1,
        )
        self.sub_img_r = self.create_subscription(
            Image,
            "/genie_sim/Right_Camera_rgb",
            self.callback_rgb_image_r,
            1,
        )
        self.sub_img_head = self.create_subscription(
            Image,
            "/genie_sim/Head_Camera_rgb",
            self.callback_rgb_image_head,
            1,
        )
        self.sub_img_depth = self.create_subscription(
            Image,
            "/genie_sim/Head_Camera_depth",
            self.callback_depth_image,
            1,
        )
        self.sub_img_l = self.create_subscription(
            Image,
            "/genie_sim/Left_Camera_depth",
            self.callback_depth_image_l,
            1,
        )
        self.sub_img_r = self.create_subscription(
            Image,
            "/genie_sim/Right_Camera_depth",
            self.callback_depth_image_r,
            1,
        )
        self.sub_tf = self.create_subscription(
            TFMessage,
            "/tf",
            self.callback_tf,
            QOS_PROFILE_VOLATILE,
        )

        # msg
        self.lock = threading.Lock()
        self.message_buffer = deque(maxlen=30)
        self.lock_joint_state = threading.Lock()
        self.obs_joint_state = JointState()
        self.cur_joint_state = JointState()
        self.lock_tf = threading.Lock()
        self.tf_msg = TFMessage()

        # loop
        self.loop_rate = self.create_rate(30.0)

    def callback_rgb_image_head(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            success, compressed_data = cv2.imencode(".png", cv_image)
            if success:
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = "png"
                compressed_msg.data = compressed_data.tobytes()

                self.pub_head_img.publish(compressed_msg)
            else:
                self.get_logger().error("Failed to compress image")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def callback_rgb_image_l(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            success, compressed_data = cv2.imencode(".png", cv_image)
            if success:
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = "png"
                compressed_msg.data = compressed_data.tobytes()

                self.pub_left_wrist_img.publish(compressed_msg)
            else:
                self.get_logger().error("Failed to compress image")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def callback_rgb_image_r(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            success, compressed_data = cv2.imencode(".png", cv_image)
            if success:
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = "png"
                compressed_msg.data = compressed_data.tobytes()

                self.pub_right_wrist_img.publish(compressed_msg)
            else:
                self.get_logger().error("Failed to compress image")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def callback_depth_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            cv_image = np.nan_to_num(cv_image, nan=0.0, posinf=0.0, neginf=0.0)
            cv_image = (cv_image * 1000).astype(np.uint16)

            success, compressed_data = cv2.imencode(".png", cv_image)
            if success:
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = "png"
                compressed_msg.data = compressed_data.tobytes()

                self.pub_head_depth_img.publish(compressed_msg)
            else:
                self.get_logger().error("Failed to compress image")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def callback_depth_image_l(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            cv_image = np.nan_to_num(cv_image, nan=0.0, posinf=0.0, neginf=0.0)
            cv_image = (cv_image * 1000).astype(np.uint16)

            success, compressed_data = cv2.imencode(".png", cv_image)
            if success:
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = "png"
                compressed_msg.data = compressed_data.tobytes()

                self.pub_left_wrist_depth_img.publish(compressed_msg)
            else:
                self.get_logger().error("Failed to compress image")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def callback_depth_image_r(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            cv_image = np.nan_to_num(cv_image, nan=0.0, posinf=0.0, neginf=0.0)
            cv_image = (cv_image * 1000).astype(np.uint16)

            success, compressed_data = cv2.imencode(".png", cv_image)
            if success:
                compressed_msg = CompressedImage()
                compressed_msg.header = msg.header
                compressed_msg.format = "png"
                compressed_msg.data = compressed_data.tobytes()

                self.pub_right_wrist_depth_img.publish(compressed_msg)
            else:
                self.get_logger().error("Failed to compress image")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def callback_joint_state(self, msg):
        self.cur_joint_state = msg

        joint_name_state_dict = {}
        for idx, name in enumerate(msg.name):
            joint_name_state_dict[name] = msg.position[idx]

        msg_remap = JointState()
        msg_remap.header = msg.header
        msg_remap.name = []
        msg_remap.velocity = []
        msg_remap.effort = []

        # fmt: off
        gripper_control_joints = self.robot_cfg["gripper"]["gripper_control_joint"]
        msg_remap.position.append(joint_name_state_dict["idx21_arm_l_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx22_arm_l_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx23_arm_l_joint3"])
        msg_remap.position.append(joint_name_state_dict["idx24_arm_l_joint4"])
        msg_remap.position.append(joint_name_state_dict["idx25_arm_l_joint5"])
        msg_remap.position.append(joint_name_state_dict["idx26_arm_l_joint6"])
        msg_remap.position.append(joint_name_state_dict["idx27_arm_l_joint7"])
        msg_remap.position.append(joint_name_state_dict["idx61_arm_r_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx62_arm_r_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx63_arm_r_joint3"])
        msg_remap.position.append(joint_name_state_dict["idx64_arm_r_joint4"])
        msg_remap.position.append(joint_name_state_dict["idx65_arm_r_joint5"])
        msg_remap.position.append(joint_name_state_dict["idx66_arm_r_joint6"])
        msg_remap.position.append(joint_name_state_dict["idx67_arm_r_joint7"])
        msg_remap.position.append(joint_name_state_dict[gripper_control_joints["left"].split("/")[-1]])
        msg_remap.position.append(joint_name_state_dict[gripper_control_joints["right"].split("/")[-1]])
        msg_remap.position.append(joint_name_state_dict["idx11_head_joint1"])
        msg_remap.position.append(joint_name_state_dict["idx12_head_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx02_body_joint2"])
        msg_remap.position.append(joint_name_state_dict["idx01_body_joint1"])
        # fmt: on

        with self.lock_joint_state:
            self.obs_joint_state = msg_remap

    def callback_tf(self, msg):
        self.tf_msg = msg

    def get_base_transform(self):
        with self.lock_tf:
            for t in self.tf_msg.transforms:
                if "base_link" == t.child_frame_id:
                    position = np.array(
                        [
                            t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z,
                        ]
                    )
                    rotation = np.array(
                        [
                            t.transform.rotation.w,
                            t.transform.rotation.x,
                            t.transform.rotation.y,
                            t.transform.rotation.z,
                        ]
                    )
                    return (position, rotation)

    def publish_observation_msgs(self):
        with self.lock_joint_state:
            self.pub_joint_state.publish(self.obs_joint_state)

    def publish_msg(self, img_data_list, depth_image_data_list, joint_data):
        # pub joint_state
        self.publish_observation_msgs()

    def callback_joint_command(self, msg):
        with self.lock:
            self.message_buffer.append(msg)

    def pub_init_pose_command(self):
        cmd_msg = JointState()
        cmd_msg.name = [
            "idx21_arm_l_joint1",
            "idx22_arm_l_joint2",
            "idx23_arm_l_joint3",
            "idx24_arm_l_joint4",
            "idx25_arm_l_joint5",
            "idx26_arm_l_joint6",
            "idx27_arm_l_joint7",
            "idx61_arm_r_joint1",
            "idx62_arm_r_joint2",
            "idx63_arm_r_joint3",
            "idx64_arm_r_joint4",
            "idx65_arm_r_joint5",
            "idx66_arm_r_joint6",
            "idx67_arm_r_joint7",
            "idx41_gripper_l_outer_joint1",
            "idx81_gripper_r_outer_joint1",
            "idx11_head_joint1",
            "idx12_head_joint2",
            "idx02_body_joint2",
            "idx01_body_joint1",
        ]
        cmd_msg.velocity = [float("nan")] * 20
        cmd_msg.effort = [float("nan")] * 20
        cmd_msg.position = [0.0] * 20
        cmd_msg.position[0] = self.init_pose["init_arm"][0]
        cmd_msg.position[1] = self.init_pose["init_arm"][1]
        cmd_msg.position[2] = self.init_pose["init_arm"][2]
        cmd_msg.position[3] = self.init_pose["init_arm"][3]
        cmd_msg.position[4] = self.init_pose["init_arm"][4]
        cmd_msg.position[5] = self.init_pose["init_arm"][5]
        cmd_msg.position[6] = self.init_pose["init_arm"][6]
        cmd_msg.position[7] = self.init_pose["init_arm"][7]
        cmd_msg.position[8] = self.init_pose["init_arm"][8]
        cmd_msg.position[9] = self.init_pose["init_arm"][9]
        cmd_msg.position[10] = self.init_pose["init_arm"][10]
        cmd_msg.position[11] = self.init_pose["init_arm"][11]
        cmd_msg.position[12] = self.init_pose["init_arm"][12]
        cmd_msg.position[13] = self.init_pose["init_arm"][13]
        cmd_msg.position[14] = self.init_pose["init_hand"][0]
        cmd_msg.position[15] = self.init_pose["init_hand"][1]
        cmd_msg.position[16] = self.init_pose["body_state"][0]
        cmd_msg.position[17] = self.init_pose["body_state"][1]
        cmd_msg.position[18] = self.init_pose["body_state"][2]
        cmd_msg.position[19] = self.init_pose["body_state"][3]

        self.pub_joint_command.publish(cmd_msg)

    def get_full_msgs(self):
        with self.lock:
            return len(self.message_buffer) == 30

    def buffer_empty(self):
        with self.lock:
            return len(self.message_buffer) == 0

    def parse_joint_command(self):
        with self.lock:
            if len(self.message_buffer) != 0:
                msg = self.message_buffer.popleft()
                infer_start = Bool()
                infer_start.data = msg.header.frame_id == "-1"
                self.pub_infer_start.publish(infer_start)
                self.pub_joint_command.publish(msg)

            return None

    def get_joint_state(self):
        with self.lock:
            return self.cur_joint_state

    def set_joint_state(self, name, position):
        with self.lock:
            cmd_msg = JointState()
            cmd_msg.position = position
            cmd_msg.name = name
            cmd_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_joint_command.publish(cmd_msg)
