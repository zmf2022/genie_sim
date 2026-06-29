# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .base_nodes import *
from std_msgs.msg import String, Bool
import threading


class PIROSNode(SimNode):
    """ROS bridge for control-plane signals (instruction / reset / infer_start / shuffle).

    The image-forwarding path (/genie_sim/*_rgb → /record/*_rgb republish) and
    the recording control topics (/record/static_info, /record/sub_task_name,
    /record/episode_done, /record/episode_ack) have been removed — the
    benchmark now drives the in-process LocalRecorder directly via
    ``api_core.local_recorder``.
    """

    def __init__(self, robot_name="G1_120s", node_name="pi_node", device="cuda:0"):
        super().__init__(robot_name=robot_name, node_name=node_name)

        # Subscribe to /sim/instruction, /sim/reset, /sim/infer_start, /sim/shuffle
        self.sub_instruction = self.create_subscription(String, "/sim/instruction", self.callback_instruction, 1)
        self.sub_reset = self.create_subscription(Bool, "/sim/reset", self.callback_reset, 1)
        self.sub_infer_start = self.create_subscription(Bool, "/sim/infer_start", self.callback_infer_start, 1)
        self.sub_shuffle = self.create_subscription(Bool, "/sim/shuffle", self.callback_shuffle, 1)

        self.instruction_lock = threading.Lock()
        self.instruction_msg = ""
        self.reset_lock = threading.Lock()
        self.reset_msg = False
        self.infer_start_lock = threading.Lock()
        self.infer_start_msg = False
        self.shuffle_lock = threading.Lock()
        self.shuffle_msg = False

        self.sub_task_name = ""
        self.sub_task_name_lock = threading.Lock()

        # Hooks the recorder can install to be notified when ROS callbacks fire.
        self._instruction_listener = None

    # ----- Recorder hook ----------------------------------------------------

    def set_instruction_listener(self, callback):
        self._instruction_listener = callback

    # ----- Static / dynamic info: kept as no-ops for compat ----------------

    def pub_static_info_msg(self, msg: str):
        # Recording metadata is now consumed in-process via DataCourier →
        # LocalRecorder.update_instruction; nothing to publish here.
        return

    def pub_dynamic_info_msg(self, msg: str):
        return

    # ----- Sim control subscriptions ---------------------------------------

    def callback_instruction(self, msg):
        with self.instruction_lock:
            self.instruction_msg = msg.data
        if self._instruction_listener is not None:
            try:
                self._instruction_listener(msg.data)
            except Exception:
                pass

    def callback_reset(self, msg):
        with self.reset_lock:
            self.reset_msg = msg.data

    def callback_infer_start(self, msg):
        with self.infer_start_lock:
            self.infer_start_msg = msg.data

    def callback_shuffle(self, msg):
        with self.shuffle_lock:
            self.shuffle_msg = msg.data

    def get_instruction(self):
        with self.instruction_lock:
            return self.instruction_msg

    def get_reset(self):
        with self.reset_lock:
            return self.reset_msg

    def get_infer_start(self):
        with self.infer_start_lock:
            return self.infer_start_msg

    def get_shuffle(self):
        with self.shuffle_lock:
            return self.shuffle_msg

    # ----- Sub-task name plumbing (in-memory only) -------------------------

    def set_sub_task_name(self, sub_task_name: str):
        with self.sub_task_name_lock:
            self.sub_task_name = sub_task_name

    def get_sub_task_name(self):
        with self.sub_task_name_lock:
            return self.sub_task_name
