# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import os, sys
import json
import numpy as np
import glob
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import geniesim.utils.system_utils as system_utils
from geniesim.benchmark.envs.dummy_env import DummyEnv
from geniesim.robot.genie_robot import IsaacSimRpcRobot
from geniesim.layout.task_generate import TaskGenerator
from vr_server import VRServer
from scipy.spatial.transform import Rotation as R
import math
from pynput import keyboard
import threading

from geniesim.utils.logger import Logger
from geniesim.utils.eval_utils import *
from geniesim.utils.ros_utils import SimROSNode
from copy import deepcopy
import rclpy
import time
import signal

logger = Logger()


def handle_exit(signum, frame):
    raise KeyboardInterrupt(f"Receive signal {signum}, start cleaning...")


signal.signal(signal.SIGINT, handle_exit)  # Ctrl+C


def scale_quat_to_euler(q, s):
    rotation = R.from_quat(q)
    rpy = rotation.as_euler("xzy", degrees=False)
    r, p, y = rpy
    if r >= 0:
        r_trans = math.pi - r
    else:
        r_trans = -math.pi - r
    p_trans = -p
    y_trans = y
    rpy_trans = np.array([p_trans, -r_trans, y_trans])
    rpy_scaled = rpy_trans * s

    return rpy_scaled


class TeleOp(object):
    def __init__(self, args):
        if args.task_name != "":
            self.task_name = args.task_name
        else:
            raise ValueError("Invalid task_name or task_id")
        self.args = args
        self.task_config = None
        self.mode = args.mode
        self.record = args.record
        self.fps = args.fps
        self.episodes_per_instance = 1
        self.host_ip = args.host_ip
        self.port = args.port
        if self.mode == "pico":
            self.init_pico_control()
        elif self.mode == "keyboard":
            self.init_keyboard_control()
        else:
            raise ValueError("Invalid mode")
        self.reset_flg = False
        self.switch_flg = False
        self.single_evaluate_ret = deepcopy(EVAL_TEMPLATE)
        self.eval_out_dir = os.path.join(system_utils.teleop_root_path(), "output")

    def init_pico_control(self):
        self.vr_server = VRServer(self.host_ip, self.port)

    def init_keyboard_control(self):
        self.reset_command()
        self.gripper_command = "open"
        self.current_gripper_state = "open"
        self.current_arm_type = "right"
        default_cmd = {
            "pos": np.array([0.0, 0.0, 0.0]),
            "rot": np.array([0.0, 0.0, 0.0]),
        }
        self.last_command = [default_cmd, default_cmd]

    def reset_command(self):
        self.command_ = np.array([0.0, 0.0, 0.0])
        self.rotation_command = np.array([0.0, 0.0, 0.0])
        self.robot_command = np.array([0.0, 0.0, 0.0])
        self.robot_rotation_command = np.array([0.0, 0.0, 0.0])
        self.waist_command = np.array([0.0, 0.0])
        self.head_command = np.array([0.0, 0.0])

    def sub_keyboard_event(self):
        self.pressed_keys = set()

        def on_press(key):
            try:
                if key.char == "w":
                    self.command_ += np.array([0.01, 0.0, 0.0])
                elif key.char == "s":
                    self.command_ += np.array([-0.01, 0.0, 0.0])
                elif key.char == "a":
                    self.command_ += np.array([0.0, 0.01, 0.0])
                elif key.char == "d":
                    self.command_ += np.array([0.0, -0.01, 0.0])
                elif key.char == "q":
                    self.command_ += np.array([0.0, 0.0, 0.01])
                elif key.char == "e":
                    self.command_ += np.array([0.0, 0.0, -0.01])
                elif key.char == "j":
                    self.rotation_command += np.array([-0.02, 0.0, 0.0])
                elif key.char == "l":
                    self.rotation_command += np.array([0.02, 0.0, 0.0])
                elif key.char == "i":
                    self.rotation_command += np.array([0.0, 0.02, 0.0])
                elif key.char == "k":
                    self.rotation_command += np.array([0.0, -0.02, 0.0])
                elif key.char == "u":
                    self.rotation_command += np.array([0.0, 0.0, 0.02])
                elif key.char == "o":
                    self.rotation_command += np.array([0.0, 0.0, -0.02])
                elif key.char == "c":
                    if keyboard.Key.ctrl in self.pressed_keys:
                        self.gripper_command = "open"
                    else:
                        self.gripper_command = "close"
                elif key.char == "r":
                    self.reset_flg = True
            except AttributeError:
                self.pressed_keys.add(key)
                if key == keyboard.Key.up:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.0, -0.01])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.01, 0.0])
                    else:
                        self.robot_command += np.array([0.02, 0.0, 0.0])
                elif key == keyboard.Key.down:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.0, 0.01])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([-0.01, 0.0])
                    else:
                        self.robot_command += np.array([-0.02, 0.0, 0.0])
                elif key == keyboard.Key.left:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([0.01, 0.0])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.0, 0.01])
                    else:
                        self.robot_rotation_command += np.array([0.0, 0.0, 0.02])
                elif key == keyboard.Key.right:
                    if keyboard.Key.shift in self.pressed_keys:
                        self.head_command += np.array([-0.01, 0.0])
                    elif keyboard.Key.ctrl in self.pressed_keys:
                        self.waist_command += np.array([0.0, -0.01])
                    else:
                        self.robot_rotation_command += np.array([0.0, 0.0, -0.02])
                elif key == keyboard.Key.tab and keyboard.Key.ctrl in self.pressed_keys:
                    if self.current_arm_type == "right":
                        self.current_arm_type = "left"
                        self.last_command[1] = {
                            "pos": self.command_,
                            "rot": self.rotation_command,
                        }
                    else:
                        self.current_arm_type = "right"
                        self.last_command[0] = {
                            "pos": self.command_,
                            "rot": self.rotation_command,
                        }
                    self.switch_flg = True

        def on_release(key):
            self.pressed_keys.discard(key)
            if key == keyboard.Key.up or key == keyboard.Key.down:
                self.robot_command = np.array([0.0, 0.0, 0.0])
            elif key == keyboard.Key.left or key == keyboard.Key.right:
                self.robot_rotation_command = np.array([0.0, 0.0, 0.0])

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

    def parse_joint_pose(self, precision=3):
        l_arm, r_arm, body = [], [], []
        joint_pos, joint_name = self.get_joint_state()
        for i in range(7):
            l_arm.append(joint_pos[i * 2 + 4])
            r_arm.append(joint_pos[i * 2 + 5])
        for i in range(4):
            body.append(joint_pos[i])

        return {
            "l_arm": [round(v, precision) for v in l_arm],
            "r_arm": [round(v, precision) for v in r_arm],
            "ori_pos": [round(v, precision) for v in joint_pos],
            "joint_name": joint_name,
            "body": body,
        }

    def parse_pico_command(self, content):
        """
        Args:
            content (list(dict)): parse pico control command to local frame
        """
        [l_sig, r_sig] = content
        ret = {"l": {}, "r": {}}
        ret["l"]["position"] = np.array(
            [l_sig["position"]["z"], -l_sig["position"]["x"], l_sig["position"]["y"]]
        )
        ret["l"]["quaternion"] = np.array(
            [
                l_sig["rotation"]["w"],
                l_sig["rotation"]["x"],
                l_sig["rotation"]["y"],
                l_sig["rotation"]["z"],
            ]
        )
        ret["l"]["axisMode"] = "reset" if l_sig["axisClick"] == "true" else "move"
        ret["l"]["axisX"] = l_sig["axisX"]
        ret["l"]["axisY"] = l_sig["axisY"]
        ret["l"]["gripper"] = 1 - l_sig["indexTrig"]
        ret["l"]["On"] = l_sig["handTrig"] > 0.9
        ret["l"]["reset"] = l_sig["keyOne"] == "true"
        ret["l"]["reserved"] = l_sig["keyTwo"] == "true"

        ret["r"]["position"] = np.array(
            [r_sig["position"]["z"], -r_sig["position"]["x"], r_sig["position"]["y"]]
        )
        ret["r"]["quaternion"] = np.array(
            [
                r_sig["rotation"]["w"],
                r_sig["rotation"]["x"],
                r_sig["rotation"]["y"],
                r_sig["rotation"]["z"],
            ]
        )
        ret["r"]["axisX"] = r_sig["axisX"]
        ret["r"]["axisY"] = r_sig["axisY"]
        ret["r"]["axisMode"] = "head" if r_sig["axisClick"] == "true" else "waist"
        ret["r"]["gripper"] = 1 - r_sig["indexTrig"]
        ret["r"]["On"] = r_sig["handTrig"] > 0.9
        ret["r"]["reset"] = r_sig["keyOne"] == "true"
        ret["r"]["resetbh"] = r_sig["keyTwo"] == "true"
        self.pico_command = ret

    def parse_head_control(self):
        if self.pico_command["r"]["axisMode"] == "head":
            delta_angle = 2e-3
            thresh = 0.5
            head_yaw = self.pico_command["r"]["axisX"]
            head_pitch = self.pico_command["r"]["axisY"]
            if head_yaw > thresh:
                self.init_pos[2] -= delta_angle
            if head_yaw < -thresh:
                self.init_pos[2] += delta_angle
            if head_pitch > thresh:
                self.init_pos[3] -= delta_angle
            if head_pitch < -thresh:
                self.init_pos[3] += delta_angle

    def parse_waist_control(self):
        if self.pico_command["r"]["axisMode"] == "waist":
            delta_angle = 1e-4
            thresh = 0.5
            waist_lift = self.pico_command["r"]["axisY"]
            waist_pitch = self.pico_command["r"]["axisX"]
            if waist_lift > thresh:
                self.init_pos[0] += delta_angle
            if waist_lift < -thresh:
                self.init_pos[0] -= delta_angle
            if waist_pitch > thresh:
                self.init_pos[1] += delta_angle * 5
            if waist_pitch < -thresh:
                self.init_pos[1] -= delta_angle * 5

    def parse_arm_control(self, coef_pos, coef_quat):
        on_l, on_r = self.pico_command["l"]["On"], self.pico_command["r"]["On"]
        rescaled_pos_l = coef_pos * self.pico_command["l"]["position"]
        rescaled_pos_r = coef_pos * self.pico_command["r"]["position"]
        rescaled_rpy_l = scale_quat_to_euler(
            self.pico_command["l"]["quaternion"], coef_quat
        )
        rescaled_rpy_r = scale_quat_to_euler(
            self.pico_command["r"]["quaternion"], coef_quat
        )
        reset_l = self.pico_command["l"]["reset"]
        reset_r = self.pico_command["r"]["reset"]
        reset_bh = self.pico_command["r"]["resetbh"]

        jp_l = self.env.robot.get_joint_from_deltapos(
            xyz=rescaled_pos_l, rpy=rescaled_rpy_l, id="left", isOn=on_l
        )
        jp_r = self.env.robot.get_joint_from_deltapos(
            xyz=rescaled_pos_r, rpy=rescaled_rpy_r, id="right", isOn=on_r
        )
        if jp_l.any() != 0.0:
            self.init_pos[4:18:2] = [round(v, 3) for v in jp_l]
        if jp_r.any() != 0.0:
            self.init_pos[5:19:2] = [round(v, 3) for v in jp_r]

        if reset_l:
            logger.info("reset left arm...")
            self.init_pos[4:18:2] = self.init_l_arm
            joint_info = self.parse_joint_pose()
            joint_info["l_arm"] = self.init_l_arm
            self.env.robot.initialize_solver(joint_info)
        if reset_r:
            logger.info("reset right arm...")
            self.init_pos[5:19:2] = self.init_r_arm
            joint_info = self.parse_joint_pose()
            joint_info["r_arm"] = self.init_r_arm
            self.env.robot.initialize_solver(joint_info)
        if reset_bh:
            logger.info("reset body and head...")
            self.init_pos[0:4] = self.init_body

    def parse_gripper_control(self):
        l_idx, r_idx = self.gripper_active_joint
        self.init_pos[l_idx] = self.pico_command["l"]["gripper"]
        self.init_pos[r_idx] = self.pico_command["r"]["gripper"]

    def apply_base_control(self):
        x = self.pico_command["l"]["axisY"]
        y = -self.pico_command["l"]["axisX"]
        mode = self.pico_command["l"]["axisMode"]
        if mode == "move":
            target_x, target_yaw = 0, 0
            if x > 0.5:
                target_x = 0.01 * x
            if x < -0.5:
                target_x = 0.01 * x
            if y < -0.5:
                target_yaw = -0.1
            if y > 0.5:
                target_yaw = 0.1

            if target_x != 0.0 or target_yaw != 0.0:
                pos_diff = np.array([target_x, 0, 0])
                rot_diff = np.array([0, 0, target_yaw])
                self.env.robot.update_odometry(pos_diff, rot_diff)
        else:
            self.reset_robot_pose()

    def initialize(self):
        joint_info = self.parse_joint_pose()
        self.init_pos = joint_info["ori_pos"]
        self.init_l_arm = joint_info["l_arm"]
        self.init_r_arm = joint_info["r_arm"]
        self.init_body = joint_info["body"]
        self.joint_name = joint_info["joint_name"]
        self.env.robot.initialize_solver(joint_info)
        self.robot_init_pos, self.robot_init_rot = self.env.robot.get_init_pose()
        self.reset_command()
        self.current_step = 0
        self.eval_interval = 30

    def reset_robot_pose(self):
        self.env.robot.set_base_pose(
            target_pos=self.robot_init_pos, target_rot=self.robot_init_rot
        )
        self.env.robot.reset_odometry()

    def update_eval_ret(self, task_progress):
        self.single_evaluate_ret["result"]["progress"] = task_progress

    def run_eval(self):
        if self.current_step != 0 and self.current_step % self.eval_interval == 0:
            self.env.action_update()
            self.update_eval_ret(self.env.task.task_progress)
        self.current_step += 1

    def run_pico_control(self, with_physics=True):
        now_t = self.sim_ros_node.get_clock().now().nanoseconds * 1e-9
        while now_t <= 3.0:
            print("wait server for %.3fs" % now_t)
            now_t = self.sim_ros_node.get_clock().now().nanoseconds * 1e-9
            time.sleep(0.1)
        self.initialize()
        coef_pos = 0.8
        coef_quat = 0.8
        self.env.do_eval_action()
        while True:
            pico_cmd = self.vr_server.on_update()
            if pico_cmd:
                logger.info(f"Episode status {self.env.has_done}")
                self.parse_pico_command(pico_cmd)
                self.parse_arm_control(coef_pos, coef_quat)
                self.parse_gripper_control()
                self.parse_head_control()
                self.parse_waist_control()
                self.apply_base_control()
                self.set_joint_state(self.joint_name, self.init_pos)
                self.run_eval()
                if self.env.has_done:
                    break
            else:
                logger.info("waiting for pico cmd")

    def run_keyboard_control(self, with_physics=True):
        self.initialize()
        self.sub_keyboard_event()
        self.env.do_eval_action()
        while True:
            with self.lock:
                if self.reset_flg:
                    self.reset_command()
                    self.reset_robot_pose()
                    self.reset_flg = False
                if self.switch_flg:
                    idx = 0 if self.current_arm_type == "left" else 1
                    self.command_ = self.last_command[idx]["pos"]
                    self.rotation_command = self.last_command[idx]["rot"]
                    self.switch_flg = False
                if self.current_arm_type == "right":
                    jp = self.env.robot.get_joint_from_deltapos(
                        xyz=self.command_,
                        rpy=self.rotation_command,
                        id="right",
                        isOn=True,
                    )
                    if jp.any() != 0.0:
                        self.init_pos[5:19:2] = jp
                    self.init_pos[self.gripper_active_joint[1]] = (
                        0.0 if self.gripper_command == "close" else 1.0
                    )
                else:
                    jp = self.env.robot.get_joint_from_deltapos(
                        xyz=self.command_,
                        rpy=self.rotation_command,
                        id="left",
                        isOn=True,
                    )
                    if jp.any() != 0.0:
                        self.init_pos[4:18:2] = jp
                    self.init_pos[self.gripper_active_joint[0]] = (
                        0.0 if self.gripper_command == "close" else 1.0
                    )
                logger.info(
                    f"Base moving speed: {self.robot_command[0]:.2f}, yaw rate: {self.robot_rotation_command[2]:.2f}"
                )
                self.env.robot.update_odometry(
                    self.robot_command, self.robot_rotation_command
                )
                self.init_pos[0:2] += self.waist_command
                self.init_pos[2:4] += self.head_command
                self.set_joint_state(self.joint_name, self.init_pos)
                self.run_eval()
                if self.env.has_done:
                    break

    def load_task_config(self, task):
        task_config_file = os.path.join(
            system_utils.teleop_root_path(), "tasks", task + ".json"
        )
        logger.info(f"task config file {task_config_file}")
        if not os.path.exists(task_config_file):
            raise ValueError("Task config file not found: {}".format(task_config_file))
        with open(task_config_file) as f:
            self.task_config = json.load(f)

    def spin_ros_node(self):
        while not self.stop_spin_thread_flag:
            rclpy.spin_once(self.sim_ros_node, timeout_sec=0.01)

    def start_ros_node(self):
        rclpy.init()
        with open(
            os.path.join(
                system_utils.app_root_path(),
                "robot_cfg/",
                self.robot_cfg,
            ),
            "r",
        ) as f:
            self.robot_cfg_content = json.load(f)
        self.sim_ros_node = SimROSNode(robot_cfg=self.robot_cfg_content)
        self.stop_spin_thread_flag = False
        self.spin_thread = threading.Thread(target=self.spin_ros_node, daemon=True)
        self.spin_thread.start()
        self.lock = threading.Lock()

    def stop_ros_node(self):
        if self.spin_thread.is_alive():
            self.stop_spin_thread_flag = True
            self.spin_thread.join(timeout=1.0)
        self.sim_ros_node.destroy_node()
        rclpy.shutdown()

    def get_joint_state(self):
        js = self.sim_ros_node.get_joint_state()
        return js.position, js.name

    def set_joint_state(self, name, position):
        self.sim_ros_node.set_joint_state(name, position)

    def config_gripper(self):
        if "omnipicker" in self.robot_cfg:
            self.gripper_active_joint = [19, 21]
        elif "120s" in self.robot_cfg:
            self.gripper_active_joint = [19, 21]
        else:
            raise ValueError(f"Robot {self.robot_cfg} gripper not supported.")

    def run(self):
        self.load_task_config(self.task_name)
        self.robot_cfg = self.task_config["robot"]["robot_cfg"]
        self.start_ros_node()
        self.config_gripper()

        # init robot and scene
        scene_info = self.task_config["scene"]
        workspace_id = scene_info["scene_id"].split("/")[-1]
        if workspace_id in scene_info["function_space_objects"]:
            robot_init_pose = self.task_config["robot"]["robot_init_pose"][workspace_id]
        else:
            robot_init_pose = self.task_config["robot"]["robot_init_pose"]
        self.task_config["specific_task_name"] = self.task_name

        logger.info(f"scene_usd {self.task_config['scene']['scene_usd']}")
        robot = IsaacSimRpcRobot(
            robot_cfg=self.robot_cfg,
            scene_usd=self.task_config["scene"]["scene_usd"],
            client_host=self.args.client_host,
            position=robot_init_pose["position"],
            rotation=robot_init_pose["quaternion"],
            gripper_control_type=1,
        )

        # init state
        task_generator = TaskGenerator(self.task_config)
        task_folder = str(system_utils.teleop_root_path()) + "/saved_task/%s" % (
            self.task_config["task"]
        )
        task_generator.generate_tasks(
            save_path=task_folder,
            task_num=self.episodes_per_instance,
            task_name=self.task_config["task"],
        )
        robot_position = task_generator.robot_init_pose["position"]
        robot_rotation = task_generator.robot_init_pose["quaternion"]
        self.task_config["robot"]["robot_init_pose"]["position"] = robot_position
        self.task_config["robot"]["robot_init_pose"]["quaternion"] = robot_rotation
        specific_task_files = glob.glob(task_folder + "/*.json")
        self.eval_result = []
        for episode_id in range(self.episodes_per_instance):
            episode_file_path = specific_task_files[episode_id]
            env = DummyEnv(robot, episode_file_path, self.task_config)
            self.env = env
            init_pose = self.task_config["robot"].get("init_arm_pose")
            recording_objects_prim = self.task_config.get("recording_setting").get(
                "objects_prim"
            )
            if init_pose:
                robot.set_init_pose(init_pose)

            env.reset()

            if self.record:
                self.env.start_recording(
                    task_name=self.task_name,
                    camera_prim_list=[],
                    fps=self.fps,
                    extra_objects_prim=recording_objects_prim,
                )
            if self.mode == "pico":
                self.run_pico_control()
            elif self.mode == "keyboard":
                self.run_keyboard_control()
            else:
                raise ValueError("Invalid mode: {}".format(self.mode))

    def post_process(self):
        if self.record:
            self.env.stop_recording(True)

        summarize_scores(self.single_evaluate_ret, self.task_name)
        self.eval_result.append(self.single_evaluate_ret)
        dump_eval_result(self.eval_out_dir, self.eval_result)
        logger.info("Episode done...")
        self.env.robot.client.Exit()
        self.stop_ros_node()


def main():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--client_host", type=str, default="localhost:50051", help="The client")
    parser.add_argument("--fps", type=int, default=30, help="Set fps of the recording")
    parser.add_argument("--task_name", default="iros_stamp_the_seal", type=str, help="Selected task to run")
    parser.add_argument("--mode", type=str, default="pico", help="Choose teleop mode: pico or keyboard")
    parser.add_argument("--record", action="store_true", help="Enable data recording")
    parser.add_argument("--host_ip", type=str, default="172.19.33.248", help="Set vr host ip")
    parser.add_argument("--port", type=int, default=8080, help="Set vr port")
    # fmt: on
    args = parser.parse_args()
    task = TeleOp(args)
    try:
        task.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt, teleop stop")
    finally:
        try:
            task.post_process()
        except Exception as e:
            logger.error("Error in post process: {}".format(e))


if __name__ == "__main__":
    main()
