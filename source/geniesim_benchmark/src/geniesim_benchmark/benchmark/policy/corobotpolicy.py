# -*- coding: utf-8 -*-
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import time
import pickle
from copy import deepcopy
from typing import Dict

from .base import BasePolicy
from geniesim_benchmark.plugins.logger import Logger
from geniesim_benchmark.utils.comm.retry import run_with_inference_retry
from geniesim_benchmark.utils.generalization_utils import apply_camera_image_augmentation
from collections import deque
import numpy as np
import cv2
from geniesim_benchmark.utils import msgpack_numpy
import websockets.sync.client
from scipy.spatial.transform import Rotation as R
from geniesim_benchmark.utils.infer_post_process import process_action, get_arm_states
from geniesim_benchmark.utils.name_utils import ROBOT_CONFIGS, DEFAULT_ROBOT_CONFIG
from geniesim_benchmark.utils.ikfk_utils import get_shared_ikfk_solver
from geniesim_benchmark.utils.comm.websocket_client import ws_connect_compat

logger = Logger()

ROOT_DIR = os.environ.get("SIM_REPO_ROOT")

_OPEN_TIMEOUT_SEC = 30
_PING_INTERVAL_SEC = 20
_PING_TIMEOUT_SEC = 60


class CoRobotPolicy(BasePolicy):
    def __init__(
        self,
        task_name,
        host_ip,
        port,
        sub_task_name="",
        debug=False,
        preview=False,
        robot_cfg="",
    ):
        super().__init__(task_name=task_name, sub_task_name=sub_task_name)
        self.ts_str = time.strftime("%Y%m%d_%H%M", time.localtime(time.time()))
        self.initialized = False
        self.preview = preview
        self.debug = debug
        self._ws_uri = f"ws://{host_ip}:{port}" if port is not None else f"ws://{host_ip}"
        self._ws = None
        self._server_metadata = None
        self.infer_cnt = 0
        self._camera_dirt_cache: Dict[tuple, np.ndarray] = {}
        self._current_episode_idx = 0
        self._episode_done = False
        self._task_progress = []
        self.robot_cfg = robot_cfg
        self._robot_config = ROBOT_CONFIGS.get(robot_cfg, DEFAULT_ROBOT_CONFIG)
        # Embodiment tag the server branches on; falls back to the raw cfg key.
        self._robot_type = self._robot_config.get("robot_type", robot_cfg)
        self._label_state = self._robot_config["label_state"]
        self._process_gripper_action = self._robot_config["process_gripper_action"]
        self._arm_dim = len(self._robot_config.get("arm_joints", [])) or 14
        self._gripper_dim = len(self._robot_config.get("gripper_joints", [])) or 2
        # Process-wide shared IK/FK solver, used for EEF_ABS control and FK
        # observations; JOINT_ABS control does not depend on it.
        self._ikfk_solver = get_shared_ikfk_solver(
            arm_init_joint_position=[0.0] * self._arm_dim,
            head_init_position=[0.0] * 3,
            waist_init_position=[0.0] * 5,
            robot_cfg=robot_cfg,
        )

    def _ensure_connection(self):
        """Make a single connect attempt if currently disconnected.

        Transient failures propagate so the outer retry helper counts them
        against the budget — looping here would block one infer() call
        forever and bypass the budget."""
        if self._ws is not None:
            return
        logger.info(f"Connecting to policy server at {self._ws_uri}...")
        self._ws = ws_connect_compat(
            self._ws_uri,
            compression=None,
            max_size=None,
            open_timeout=_OPEN_TIMEOUT_SEC,
            ping_interval=_PING_INTERVAL_SEC,
            ping_timeout=_PING_TIMEOUT_SEC,
        )
        self._server_metadata = msgpack_numpy.unpackb(self._ws.recv())
        logger.info(f"Connected to policy server, metadata: {self._server_metadata}")

    def _drop_connection(self):
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def set_episode_idx(self, idx):
        self._current_episode_idx = idx

    def update_task_status(self, done, task_progress):
        self._episode_done = done
        self._task_progress = task_progress
        if done:
            self.action_buffer.clear()

    @staticmethod
    def _extract_scores(task_progress):
        scores = []
        ignored = {"ActionList", "ActionSetWaitAny", "StepOut"}
        for item in task_progress:
            cls = item.get("class_name", "")
            if cls in ignored:
                continue
            prog = item.get("progress") or {}
            entry = {
                "name": cls,
                "score": prog.get("SCORE", 0) if isinstance(prog, dict) else 0,
                "status": prog.get("STATUS", "PENDING") if isinstance(prog, dict) else "PENDING",
            }
            scores.append(entry)
        return scores

    @staticmethod
    def _encode_image_jpeg(image_rgb: np.ndarray, quality: int = 95) -> dict:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return {
            "encoding": "JPEG",
            "image_data": buf.tobytes(),
            "height": image_rgb.shape[0],
            "width": image_rgb.shape[1],
        }

    @staticmethod
    def _encode_depth(depth_map: np.ndarray, scale: int) -> dict:
        depth = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth * scale, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        return {
            "encoding": "RAW_UINT16",
            "image_data": depth.tobytes(),
            "height": depth.shape[0],
            "width": depth.shape[1],
        }

    def _split_states(self, states):
        if isinstance(states, dict):
            arm = list(states["left_arm"]) + list(states["right_arm"])
            gripper = list(states["left_gripper"]) + list(states["right_gripper"])
            return arm, gripper, list(states["waist"]), list(states["head"])
        # legacy flat list: arm + gripper + waist + head
        s = list(states)
        arm = s[0 : self._arm_dim]
        gripper = s[self._arm_dim : self._arm_dim + self._gripper_dim]
        remaining = s[self._arm_dim + self._gripper_dim :]
        n = len(remaining)
        if n >= 5:
            waist = remaining[:5]
            head = remaining[5:]
        elif n >= 2:
            waist = remaining[:2]
            head = remaining[2:]
        else:
            waist = remaining
            head = []
        return arm, gripper, waist, head

    def _pre_process_obs(self, obs, gen_config):
        obs = deepcopy(obs)
        self._label_state(obs, self._robot_config)
        if gen_config is not None:
            obs["images"] = apply_camera_image_augmentation(self._camera_dirt_cache, obs["images"], gen_config)
        return obs

    @staticmethod
    def _resize_image(image_rgb, width=640, height=480):
        """Resize image to uniform resolution for stacking."""
        if image_rgb.shape[0] != height or image_rgb.shape[1] != width:
            return cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        return image_rgb

    def get_payload(self, obs, task_instruction, gen_config):
        obs = self._pre_process_obs(obs, gen_config)

        arm_states, gripper_states, waist_states, head_states = self._split_states(obs["states"])

        # Binarize gripper: effector (mm, after label_state conversion) > threshold → 1 (closed)
        # This must match agibot_to_lerobot.py `VLA_GRIPPER_STATE_BINARY_THRESHOLD = 10.0`.
        gripper_state_binary_threshold = 10.0
        gripper_states_binary = [
            1.0 if g > gripper_state_binary_threshold else 0.0 for g in gripper_states
        ]

        # Resize all cameras to uniform 640x480, send as HWC format
        head_img = self._resize_image(obs["images"]["head"])
        left_hand_img = self._resize_image(obs["images"]["left_hand"])
        right_hand_img = self._resize_image(obs["images"]["right_hand"])

        payload = {
            "observation.images.top_head": head_img,
            "observation.images.hand_left": left_hand_img,
            "observation.images.hand_right": right_hand_img,
            "observation.state": np.concatenate([arm_states, gripper_states_binary]),
            "task": task_instruction,
        }
        
        if self.debug:
            logger.debug(f"task: {payload['task']}")
            logger.debug(f"state shape: {payload['observation.state'].shape}")
            cv2.imwrite("head.png", cv2.cvtColor(obs["images"]["head"], cv2.COLOR_RGB2BGR))
            cv2.imwrite("left_hand.png", cv2.cvtColor(obs["images"]["left_hand"], cv2.COLOR_RGB2BGR))
            cv2.imwrite("right_hand.png", cv2.cvtColor(obs["images"]["right_hand"], cv2.COLOR_RGB2BGR))
            debug_dir = os.path.join(ROOT_DIR, "debug_preview")
            os.makedirs(debug_dir, exist_ok=True)
            pkl_path = os.path.join(debug_dir, f"debug_{self.infer_cnt:04d}.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"payload": payload, "obs": obs}, f)
            logger.debug(f"Dumped debug pkl to {pkl_path}")

        if self.preview:
            ts = int(time.time() * 1000)
            debug_dir = os.path.join(ROOT_DIR, "debug_preview")
            os.makedirs(debug_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_head.png"),
                cv2.cvtColor(obs["images"]["head"], cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_left_hand.png"),
                cv2.cvtColor(obs["images"]["left_hand"], cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(debug_dir, f"preview_{self.infer_cnt:04d}_{ts}_right_hand.png"),
                cv2.cvtColor(obs["images"]["right_hand"], cv2.COLOR_RGB2BGR),
            )
            logger.info(f"[Preview] Saved images to {debug_dir}/preview_{self.infer_cnt:04d}_{ts}_*.png")
            self.infer_cnt += 1
            return None

        return payload

    def reset(self):
        self.action_buffer.clear()
        self._episode_done = False
        self._task_progress = []

    @staticmethod
    def _parse_result(result_dict):
        """Parse server result into action list.
        
        OpenPI server returns: {"actions": numpy_array, "server_timing": {...}}
        where actions.shape = (H, D) with D = arm_dim + gripper_dim = 16
        """
        actions_array = result_dict.get("actions")
        if actions_array is None:
            raise ValueError(f"Server response missing 'actions' key. Got keys: {list(result_dict.keys())}")
        
        actions_array = np.array(actions_array)
        if actions_array.ndim == 1:
            actions_array = actions_array.reshape(1, -1)
        
        arm_dim = 14
        gripper_dim = 2
        action_dim = actions_array.shape[1]
        
        actions = []
        for i in range(actions_array.shape[0]):
            action = actions_array[i]
            actions.append({
                "arm": action[:arm_dim],                         # First 14 dims: arm joints
                "gripper": action[arm_dim:arm_dim + gripper_dim],  # Next 2 dims: gripper (explicit slice, not [:] to end)
                "kind": "JOINT_ABS",
            })
        logger.info(f"[_parse_result] action_dim={action_dim}, arm_dim={arm_dim}, gripper_dim={gripper_dim}")
        return actions

    def _post_process_action(self, raw_entry, cur_arm):
        """Post-process action based on kind (JOINT_ABS or EEF_ABS).

        Args:
            raw_entry: dict with "arm", "gripper", and "kind" keys
            cur_arm: current arm joint states

        Returns:
            Processed action dict with "arm", "gripper" keys
        """
        kind = raw_entry.get("kind", "JOINT_ABS")
        raw_arm = raw_entry["arm"]
        raw_gripper = raw_entry["gripper"]

        if kind == "EEF_ABS" and self._ikfk_solver is not None:
            # Model EEF poses are already expressed in the arm_base_link frame
            # the IK solver operates in ([x, y, z, qx, qy, qz, qw]), so they go
            # straight to IK with no reframing.
            left_eef = np.asarray(raw_arm[:7], dtype=np.float64)
            right_eef = np.asarray(raw_arm[7:14], dtype=np.float64)

            left_xyzrpy = np.concatenate([left_eef[:3], R.from_quat(left_eef[3:7]).as_euler("xyz")])
            right_xyzrpy = np.concatenate([right_eef[:3], R.from_quat(right_eef[3:7]).as_euler("xyz")])

            eef_action = np.concatenate([left_xyzrpy, right_xyzrpy, raw_gripper[:1], raw_gripper[1:2]])
            joint_action = self._ikfk_solver.eef_actions_to_joint([eef_action.tolist()], cur_arm, [0.0, 0.0])[0]
            arm = [float(v) for v in joint_action[: self._arm_dim]]
            gripper_raw = joint_action[self._arm_dim : self._arm_dim + self._gripper_dim]

            logger.info(f"[EEF_ABS] IK result joints: {[round(v, 4) for v in arm]}")
        else:
            # JOINT_ABS: process directly
            action_flat = np.concatenate([raw_arm, raw_gripper])
            action_flat = process_action(None, cur_arm, action_flat, type="abs_joint", smooth_alpha=0.5)
            arm = [float(v) for v in action_flat[: self._arm_dim]]
            gripper_raw = action_flat[self._arm_dim : self._arm_dim + self._gripper_dim]

        gripper = self._process_gripper_action(gripper_raw, self._robot_config)

        result = {"arm": arm, "gripper": gripper}

        if "waist" in raw_entry:
            result["waist"] = [float(v) for v in raw_entry["waist"]]

        return result

    def infer(self, payload):
        """Send one inference request. Returns True on success.

        On transient connection failures the socket is dropped and the
        exception is re-raised so the outer retry helper can classify and
        count it. Server-side semantic errors (RuntimeError) propagate as
        fatal — they will not be retried.
        """
        try:
            self._ensure_connection()
            data = msgpack_numpy.packb(payload)
            logger.info(f"Sending payload to server, size={len(data)} bytes")
            self._ws.send(data)
            response = self._ws.recv()
            if isinstance(response, str):
                raise RuntimeError(f"Server error: {response}")
            result = msgpack_numpy.unpackb(response)
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(f"Server returned error: {result['error']}")
            # OpenPI format: result contains actions directly, not wrapped in result["result"]
            actions = self._parse_result(result)
            n = max(len(actions), 1)
            self.action_buffer = deque(actions, maxlen=n)
            return True
        except Exception as e:
            logger.warning(f"Model inference failed: {type(e).__name__}: {str(e)}")
            self._drop_connection()
            raise

    def act(self, observation, **kwargs):
        if len(self.action_buffer) == 0:
            logger.info("CoRobotPolicy: calling model infer")
            task_instruction = kwargs.get("task_instruction", "")
            gen_config = kwargs.get("gen_config")
            logger.info(f"\nInstruction: {task_instruction}\n")
            payload = self.get_payload(observation, task_instruction, gen_config)

            if payload is None:
                return None

            run_with_inference_retry(
                lambda: self.infer(payload),
                log=logger,
                label="CoRobotPolicy.infer",
            )
            self.infer_cnt += 1

        raw_entry = self.action_buffer.popleft()
        cur_arm = get_arm_states(observation["states"], self._arm_dim)
        return self._post_process_action(raw_entry, cur_arm)
