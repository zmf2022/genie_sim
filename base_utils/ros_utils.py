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
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

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
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

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
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

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
        # self.pub_joint_state.publish(msg_remap)

    def callback_tf(self, msg):
        self.tf_msg = msg

    def get_base_transform(self):
        with self.lock_tf:
            for t in self.tf_msg.transforms:
                if "base_link" == t.child_frame_id:
                    # print("ROS x", t.transform.translation.x)

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

        return
        # fmt: off
        timestamp = self.get_clock().now().to_msg()
        # pub rgb msg
        self.msg_head_img = self.bridge.cv2_to_compressed_imgmsg(img_data_list[0])
        self.msg_head_img.header.stamp = timestamp
        self.pub_head_img.publish(self.msg_head_img)
        self.msg_left_wrist_img = self.bridge.cv2_to_compressed_imgmsg(img_data_list[1])
        self.msg_left_wrist_img.header.stamp = timestamp
        self.pub_left_wrist_img.publish(self.msg_left_wrist_img)
        self.msg_right_wrist_img = self.bridge.cv2_to_compressed_imgmsg(img_data_list[2])
        self.msg_right_wrist_img.header.stamp = timestamp
        self.pub_right_wrist_img.publish(self.msg_right_wrist_img)

        # pub depth msg
        if depth_image_data_list != None:
            self.msg_head_depth_img = self.bridge.cv2_to_compressed_imgmsg(depth_image_data_list[0], dst_format="png",)
            self.msg_head_depth_img.header.stamp = timestamp
            self.pub_head_depth_img.publish(self.msg_head_depth_img)
            self.msg_left_wrist_depth_img = self.bridge.cv2_to_compressed_imgmsg(depth_image_data_list[1], dst_format="png",)
            self.msg_left_wrist_depth_img.header.stamp = timestamp
            self.pub_left_wrist_depth_img.publish(self.msg_left_wrist_depth_img)
            self.msg_right_wrist_depth_img = self.bridge.cv2_to_compressed_imgmsg(depth_image_data_list[2], dst_format="png",)
            self.msg_right_wrist_depth_img.header.stamp = timestamp
            self.pub_right_wrist_depth_img.publish(self.msg_right_wrist_depth_img)

        # pub joint msg
        self.joint_msg = JointState()
        self.joint_msg.position = joint_data
        self.joint_msg.header.stamp = timestamp
        self.pub_joint_state.publish(self.joint_msg)
        self.get_logger().info(f"Published sim observation at {timestamp}")
        # fmt: on
        self.message_buffer = deque(maxlen=30)

    def publish_perf_cb_msg(self, perf_cb_status):
        self.publisher_perf_cb_status.publish(perf_cb_status)

    def callback_joint_command(self, msg):
        cmd_msg = JointState()
        cmd_msg.header = msg.header
        cmd_msg.header.stamp = self.get_clock().now().to_msg()
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
        # cmd_msg.velocity = [float("nan")] * 20
        # cmd_msg.effort = [float("nan")] * 20
        cmd_msg.position = [0.0] * 20
        cmd_msg.position[0] = msg.position[0]
        cmd_msg.position[1] = msg.position[1]
        cmd_msg.position[2] = msg.position[2]
        cmd_msg.position[3] = msg.position[3]
        cmd_msg.position[4] = msg.position[4]
        cmd_msg.position[5] = msg.position[5]
        cmd_msg.position[6] = msg.position[6]
        cmd_msg.position[7] = msg.position[7]
        cmd_msg.position[8] = msg.position[8]
        cmd_msg.position[9] = msg.position[9]
        cmd_msg.position[10] = msg.position[10]
        cmd_msg.position[11] = msg.position[11]
        cmd_msg.position[12] = msg.position[12]
        cmd_msg.position[13] = msg.position[13]
        cmd_msg.position[14] = min(1.0, max(0, msg.position[14]))
        cmd_msg.position[15] = min(1.0, max(0, msg.position[15]))
        cmd_msg.position[16] = self.init_pose["body_state"][0]
        cmd_msg.position[17] = self.init_pose["body_state"][1]
        cmd_msg.position[18] = self.init_pose["body_state"][2]
        cmd_msg.position[19] = self.init_pose["body_state"][3]

        with self.lock:
            self.message_buffer.append(cmd_msg)

        # self.pub_joint_command.publish(cmd_msg)

    def pub_init_pose_command(self):
        cmd_msg = JointState()
        # cmd_msg.header = msg.header
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
                # self.get_logger().info(f"msg buffer length {len(self.message_buffer)}")
                msg = self.message_buffer.popleft()
                pos = msg.position
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                joint_info = {
                    "timestamp": timestamp,
                    "left_arm": pos[0:7],
                    "right_arm": pos[7:14],
                    "gripper": pos[14:16],
                    "head": [-v for v in pos[16:18]][::-1],  # yaw, pitch
                    "waist": pos[18:20][::-1],  # lift, pitch
                    "base_velocity": pos[20:22],
                }

                # for idx, name in enumerate(self.cur_joint_state.name):
                #     if name == "idx81_gripper_r_outer_joint1":
                #         print(
                #             "R  js: ",
                #             self.cur_joint_state.position[idx],
                #             self.cur_joint_state.velocity[idx],
                #             self.cur_joint_state.effort[idx],
                #         )
                #         print(
                #             "  cmd: ",
                #             msg.position[15],
                #         )

                cmd_msg = JointState()
                cmd_msg.header = msg.header
                cmd_msg.header.stamp = self.get_clock().now().to_msg()
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
                # cmd_msg.velocity = [float("nan")] * 20
                # cmd_msg.effort = [float("nan")] * 20
                cmd_msg.position = [0.0] * 20
                cmd_msg.position[0] = msg.position[0]
                cmd_msg.position[1] = msg.position[1]
                cmd_msg.position[2] = msg.position[2]
                cmd_msg.position[3] = msg.position[3]
                cmd_msg.position[4] = msg.position[4]
                cmd_msg.position[5] = msg.position[5]
                cmd_msg.position[6] = msg.position[6]
                cmd_msg.position[7] = msg.position[7]
                cmd_msg.position[8] = msg.position[8]
                cmd_msg.position[9] = msg.position[9]
                cmd_msg.position[10] = msg.position[10]
                cmd_msg.position[11] = msg.position[11]
                cmd_msg.position[12] = msg.position[12]
                cmd_msg.position[13] = msg.position[13]
                cmd_msg.position[14] = min(1.0, max(0, msg.position[14]))
                cmd_msg.position[15] = min(1.0, max(0, msg.position[15]))
                cmd_msg.position[16] = self.init_pose["body_state"][0]
                cmd_msg.position[17] = self.init_pose["body_state"][1]
                cmd_msg.position[18] = self.init_pose["body_state"][2]
                cmd_msg.position[19] = self.init_pose["body_state"][3]

                self.pub_joint_command.publish(cmd_msg)

                return joint_info

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
