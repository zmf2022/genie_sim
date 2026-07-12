# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Convert agibot episode data to LeRobot v2.1 format.

Public API
----------
- :func:`convert_agibot_to_lerobot` — programmatic entry; takes paths,
  returns a manifest dict.
- :func:`convert_cli` — argv wrapper used by
  ``geniesim dataset convert agibot-to-lerobot``.

The converter auto-detects single-episode vs batch from the input
layout: if ``agibot_dir/aligned_joints.h5`` exists, it's treated as a
single episode; otherwise the directory is scanned for episode
subdirectories.

History
-------
This module was previously ``scripts/convert_agibot_to_lerobot.py`` at
the repo root. Schema constants, dimension offsets, ffmpeg encoder
flags, and the parquet writer settings are preserved verbatim — only
the entry-point shape changed (CLI lives in ``commands/dataset.py``,
the Python API is now importable).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ============================================================
# LeRobot v2.1 parquet schema constants
# ============================================================

SUPPORTED_FORMATS = ("agibot", "vla")

AGIBOT_STATE_LEN = 159
AGIBOT_ACTION_LEN = 40

VLA_STATE_LEN = 16
VLA_ACTION_LEN = 16


def _arrow_schema_huggingface_metadata(state_len: int, action_len: int):
    return {
        "info": {
            "features": {
                "observation.state": {
                    "feature": {"dtype": "float32", "_type": "Value"},
                    "length": state_len,
                    "_type": "Sequence",
                },
                "action": {
                    "feature": {"dtype": "float32", "_type": "Value"},
                    "length": action_len,
                    "_type": "Sequence",
                },
                "episode_index": {"dtype": "int64", "_type": "Value"},
                "frame_index": {"dtype": "int64", "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
                "task_index": {"dtype": "int64", "_type": "Value"},
                "timestamp": {"dtype": "float32", "_type": "Value"},
            }
        }
    }


# ============================================================
# Data mapping constants (from lerobot meta/info.json)
# ============================================================

# --- VLA format: state 21 dims, action 21 dims ---
# State: arm_joint_pos(14) + gripper_pos(2)
VLA_STATE_JOINT_POSITION = 0
VLA_STATE_GRIPPER_POSITION = 14

# Action: arm_joint_pos(14) + gripper_pos(2)
VLA_ACTION_JOINT_POSITION = 0
VLA_ACTION_GRIPPER_POSITION = 14

# --- Agibot format: state 159 dims, action 40 dims ---
# State vector: 159 dimensions
STATE_LEFT_EFFECTOR_POS = 0
STATE_RIGHT_EFFECTOR_POS = 1
STATE_END_POSITION = 2
STATE_END_ORIENTATION = 8
STATE_END_ARM_ORIENTATION = 16
STATE_END_ARM_POSITION = 24
STATE_JOINT_POSITION = 30
STATE_JOINT_EFFORT = 44
STATE_JOINT_VELOCITY = 58
STATE_HEAD_POSITION = 72
STATE_WAIST_POSITION = 75
STATE_ROBOT_POSITION = 80
STATE_ROBOT_ORIENTATION = 83

# Action vector: 40 dimensions
ACTION_LEFT_EFFECTOR = 0
ACTION_RIGHT_EFFECTOR = 1
ACTION_END_POSITION = 2
ACTION_END_ORIENTATION = 8
ACTION_JOINT_POSITION = 16
ACTION_HEAD_POSITION = 30
ACTION_WAIST_POSITION = 33
ACTION_ROBOT_VELOCITY = 38


# ============================================================
# Pre-flight checks
# ============================================================


def _require_ffmpeg() -> None:
    """Verify ffmpeg is on PATH; raise a clear error if not."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not on PATH. Install it first:\n"
            "    Ubuntu / Debian:  sudo apt install ffmpeg\n"
            "    macOS (Homebrew): brew install ffmpeg"
        )


def _require_heavy_deps():
    """Lazy import the heavy deps so import-time cost stays low for callers
    that just want to ``from geniesim_benchmark.dataset.convert import …``.
    Returns (h5py, np, pa, pq)."""
    try:
        import h5py
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "agibot→LeRobot conversion requires h5py + numpy + pyarrow. "
            f"Install with: pip install h5py numpy pyarrow\n"
            f"Underlying error: {exc}"
        ) from exc
    return h5py, np, pa, pq


# ============================================================
# Format helpers
# ============================================================


def _resolve_format(fmt: str) -> tuple:
    """Return (state_len, action_len) for the given format string."""
    if fmt == "agibot":
        return AGIBOT_STATE_LEN, AGIBOT_ACTION_LEN
    if fmt == "vla":
        return VLA_STATE_LEN, VLA_ACTION_LEN
    raise ValueError(f"Unknown format {fmt!r}. Supported: {SUPPORTED_FORMATS}")


# ============================================================
# Core data building functions
# ============================================================


def _build_state_agibot(state_dict: dict, joint_all_dict: dict, extrinsic_data: dict, frame_idx: int = 0):
    import numpy as np
    import h5py  # noqa: F401  (needed for downstream type guards)

    state = np.zeros(159, dtype=np.float32)

    state[STATE_LEFT_EFFECTOR_POS] = state_dict["left_effector"]["position"][0]
    state[STATE_RIGHT_EFFECTOR_POS] = state_dict["right_effector"]["position"][0]

    ep = state_dict["end"]["position"]
    eo = state_dict["end"]["orientation"]
    eao = state_dict["end"]["arm_orientation"]
    eap = state_dict["end"]["arm_position"]
    state[STATE_END_POSITION : STATE_END_POSITION + 6] = ep.flatten()
    state[STATE_END_ORIENTATION : STATE_END_ORIENTATION + 8] = eo.flatten()
    state[STATE_END_ARM_ORIENTATION : STATE_END_ARM_ORIENTATION + 8] = eao.flatten()
    state[STATE_END_ARM_POSITION : STATE_END_ARM_POSITION + 6] = eap.flatten()

    state[STATE_JOINT_POSITION : STATE_JOINT_POSITION + 14] = joint_all_dict["joint"]["position"][:14]
    state[STATE_JOINT_EFFORT : STATE_JOINT_EFFORT + 14] = joint_all_dict["joint"]["effort"][:14]
    state[STATE_JOINT_VELOCITY : STATE_JOINT_VELOCITY + 14] = joint_all_dict["joint"]["velocity"][:14]

    state[STATE_HEAD_POSITION : STATE_HEAD_POSITION + 3] = state_dict["head"]["position"]
    state[STATE_WAIST_POSITION : STATE_WAIST_POSITION + 5] = state_dict["waist"]["position"]
    state[STATE_ROBOT_POSITION : STATE_ROBOT_POSITION + 3] = state_dict["robot"]["position"]
    state[STATE_ROBOT_ORIENTATION : STATE_ROBOT_ORIENTATION + 4] = state_dict["robot"]["orientation"]

    # Extrinsics
    if "hand_left_rgbd" in extrinsic_data:
        entry = extrinsic_data["hand_left_rgbd"][frame_idx]
        state[87:96] = np.array(entry["rotation"], dtype=np.float32).flatten()
        state[141:144] = np.array(entry["translation"], dtype=np.float32)

    if "hand_right_rgbd" in extrinsic_data:
        entry = extrinsic_data["hand_right_rgbd"][frame_idx]
        state[96:105] = np.array(entry["rotation"], dtype=np.float32).flatten()
        state[144:147] = np.array(entry["translation"], dtype=np.float32)

    if "head_front_rgbd" in extrinsic_data:
        entry = extrinsic_data["head_front_rgbd"][frame_idx]
        state[123:132] = np.array(entry["rotation"], dtype=np.float32).flatten()
        state[153:156] = np.array(entry["translation"], dtype=np.float32)

    return state


def _build_action_agibot(action_dict: dict):
    import numpy as np

    action = np.zeros(40, dtype=np.float32)

    action[ACTION_LEFT_EFFECTOR] = action_dict["left_effector"]["position"][0]
    action[ACTION_RIGHT_EFFECTOR] = action_dict["right_effector"]["position"][0]
    action[ACTION_END_POSITION : ACTION_END_POSITION + 6] = action_dict["end"]["position"].flatten()
    action[ACTION_END_ORIENTATION : ACTION_END_ORIENTATION + 8] = action_dict["end"]["orientation"].flatten()
    action[ACTION_JOINT_POSITION : ACTION_JOINT_POSITION + 14] = action_dict["joint"]["position"]
    action[ACTION_HEAD_POSITION : ACTION_HEAD_POSITION + 3] = action_dict["head"]["position"]
    action[ACTION_WAIST_POSITION : ACTION_WAIST_POSITION + 5] = action_dict["waist"]["position"]
    action[ACTION_ROBOT_VELOCITY : ACTION_ROBOT_VELOCITY + 2] = action_dict["robot"]["velocity"]

    return action


def _build_state_vla(state_dict: dict, joint_all_dict: dict, extrinsic_data: dict, frame_idx: int = 0):
    import numpy as np

    state = np.zeros(VLA_STATE_LEN, dtype=np.float32)
    state[VLA_STATE_JOINT_POSITION : VLA_STATE_JOINT_POSITION + 14] = joint_all_dict["joint"]["position"][:14]
    state[VLA_STATE_GRIPPER_POSITION] = state_dict["left_effector"]["position"][0]
    state[VLA_STATE_GRIPPER_POSITION + 1] = state_dict["right_effector"]["position"][0]
    return state


def _build_action_vla(action_dict: dict):
    import numpy as np

    action = np.zeros(VLA_ACTION_LEN, dtype=np.float32)
    action[VLA_ACTION_JOINT_POSITION : VLA_ACTION_JOINT_POSITION + 14] = action_dict["joint"]["position"]
    action[VLA_ACTION_GRIPPER_POSITION] = action_dict["left_effector"]["position"][0]
    action[VLA_ACTION_GRIPPER_POSITION + 1] = action_dict["right_effector"]["position"][0]
    return action


_BUILDERS = {
    "agibot": (_build_state_agibot, _build_action_agibot),
    "vla": (_build_state_vla, _build_action_vla),
}


def build_state(state_dict: dict, joint_all_dict: dict, extrinsic_data: dict, frame_idx: int = 0, fmt: str = "vla"):
    return _BUILDERS[fmt][0](state_dict, joint_all_dict, extrinsic_data, frame_idx)


def build_action(action_dict: dict, fmt: str = "vla"):
    return _BUILDERS[fmt][1](action_dict)


def read_h5_frame(h5_file, frame_id: str) -> tuple:
    """Read action and state from a single frame in aligned_joints.h5."""
    import h5py
    import numpy as np

    frame = h5_file[frame_id]

    def read_group(grp):
        result = {}
        for k, item in grp.items():
            if isinstance(item, h5py.Dataset):
                val = item[()]
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    result[k] = val.item()
                elif isinstance(val, np.ndarray):
                    result[k] = val
                else:
                    result[k] = val
            elif isinstance(item, h5py.Group):
                result[k] = read_group(item)
        return result

    return read_group(frame["action"]), read_group(frame["state"]), frame["main_timestamp"][()]


# ============================================================
# File loading utilities
# ============================================================


def load_extrinsics(agibot_dir: Path) -> dict:
    """Load extrinsic calibration files. Returns dict mapping key → list of per-frame data."""
    extrinsics: dict = {}
    extrinsic_files = {
        "hand_left_rgbd": "extrinsic_end_T_hand_left_rgbd_aligned.json",
        "hand_right_rgbd": "extrinsic_end_T_hand_right_rgbd_aligned.json",
        "head_front_rgbd": "extrinsic_end_T_head_front_rgbd_aligned.json",
    }
    sensor_dir = agibot_dir / "parameters" / "sensor"
    for key, filename in extrinsic_files.items():
        filepath = sensor_dir / filename
        if filepath.exists():
            with filepath.open() as f:
                raw = json.load(f)
                extrinsics[key] = [
                    {
                        "rotation": item["extrinsic"]["rotation_matrix"]
                        if "extrinsic" in item
                        else item["rotation"],
                        "translation": item["extrinsic"]["translation_vector"]
                        if "extrinsic" in item
                        else item["translation"],
                    }
                    for item in raw
                ]
    return extrinsics


def load_lerobot_reference(lerobot_dir: Path):
    """Load reference state vector from existing lerobot dataset.

    Returns None if the reference parquet isn't present — callers
    silently skip the extrinsic-fill step in that case.
    """
    import numpy as np
    import pyarrow.parquet as pq

    parquet_path = lerobot_dir / "data" / "chunk-000" / "episode_000000.parquet"
    if parquet_path.exists():
        table = pq.read_table(parquet_path)
        return np.array(table.column("observation.state")[0].as_py(), dtype=np.float32)
    return None


def fill_extrinsics_from_lerobot(extrinsic_data: dict, template) -> dict:
    """Fill missing extrinsic fields using lerobot reference data."""
    if template is None:
        return extrinsic_data

    keys_to_fill = {
        "head_left_fisheye": (105, 114, 147, 150),
        "head_right_fisheye": (114, 123, 150, 153),
        "head_back_fisheye": (132, 141, 156, 159),
    }
    for key, (rs, re, ts, te) in keys_to_fill.items():
        if key not in extrinsic_data:
            rot = template[rs:re].reshape(3, 3).tolist()
            trans = template[ts:te].tolist()
            extrinsic_data[key] = [{"rotation": rot, "translation": trans}] * template.shape[0]

    return extrinsic_data


TARGET_VIDEO_SIZE = "640:480"


def encode_videos(agibot_dir: Path, output_dir: Path, episode_index: int, fps: float = 30.0, fmt: str = "vla"):
    """Re-encode camera videos to unified 640×480 with ffmpeg.

    Prefers pre-encoded MP4 from ``observations/videos/`` as source; falls
    back to raw camera images. All outputs are rescaled to 640×480 so
    ``np.stack(cams, axis=2)`` works across camera views.

    Output path: videos/{chunk}/{video_key}/episode_{index:06d}.mp4
    """
    chunk_name = f"chunk-{episode_index // 1000:03d}"
    src_dir = agibot_dir / "observations" / "videos"
    camera_dir = agibot_dir / "camera"

    include_depth = fmt != "vla"
    video_map = {
        "observation.images.top_head": ("head_color", "jpg", False),
        "observation.images.hand_left": ("hand_left_color", "jpg", False),
        "observation.images.hand_right": ("hand_right_color", "jpg", False),
    }
    if include_depth:
        video_map["observation.images.top_head_depth"] = ("head_depth", "png", True)
        video_map["observation.images.hand_left_depth"] = ("hand_left_depth", "png", True)
        video_map["observation.images.hand_right_depth"] = ("hand_right_depth", "png", True)

    for vid_key, (src_name, ext, is_depth) in video_map.items():
        dst_path = output_dir / "videos" / chunk_name / vid_key / f"episode_{episode_index:06d}.mp4"
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        src_path = src_dir / f"{src_name}.mp4"
        if src_path.exists():
            _encode_video(str(src_path), str(dst_path), is_depth, fps)
            print(f"  Encoded {vid_key} (from pre-encoded MP4)")
        else:
            _encode_from_images(camera_dir, dst_path, src_name, ext, is_depth, fps)
            print(f"  Encoded {vid_key} (from raw images)")


def _encode_video(input_path: str, output_path: str, is_depth: bool, fps: float):
    """Re-encode (and scale) a single video file to target size."""
    _require_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale={TARGET_VIDEO_SIZE}",
        "-r", str(int(fps)),
    ]
    if is_depth:
        cmd += ["-c:v", "png", "-sws_flags", "neighbor"]
    else:
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "30"]
    cmd += ["-movflags", "+faststart", output_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Re-encoding {input_path} failed: {result.stderr[-500:]}")


def _encode_from_images(camera_dir, video_path, image_stem, ext, is_depth, fps):
    """Fallback: encode video from raw camera images using ffmpeg."""
    _require_ffmpeg()
    input_pattern = str(camera_dir / f"%d/{image_stem}.{ext}")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(int(fps)),
        "-i", input_pattern,
        "-vf", f"scale={TARGET_VIDEO_SIZE}",
        "-r", str(int(fps)),
    ]
    if is_depth:
        cmd += ["-c:v", "png", "-pix_fmt", "gray16le", "-sws_flags", "neighbor"]
    else:
        cmd += ["-f", "image2", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "30"]
    cmd += ["-movflags", "+faststart", str(video_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Encoding {image_stem} failed: {result.stderr[-500:]}")


def detect_episodes(agibot_root: Path) -> list:
    """Auto-detect episode directories inside agibot_root."""
    entries = sorted(agibot_root.iterdir(), key=lambda p: p.name)
    episodes = []
    for p in entries:
        if p.is_dir() and (p / "aligned_joints.h5").exists():
            episodes.append(p)
    return episodes


# ============================================================
# Episode conversion
# ============================================================


def convert_episode(
    agibot_dir: Path, output_dir: Path, episode_index: int, lerobot_template, fps: float = 30.0, fmt: str = "vla"
) -> dict:
    """Convert a single agibot episode. Returns dict with stats."""
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq

    state_len, action_len = _resolve_format(fmt)

    print(f"\n{'='*60}")
    print(f"Converting episode {episode_index}: {agibot_dir}")

    extrinsic_data = load_extrinsics(agibot_dir)
    extrinsic_data = fill_extrinsics_from_lerobot(extrinsic_data, lerobot_template)

    h5_path = agibot_dir / "aligned_joints.h5"
    h5_all_path = agibot_dir / "aligned_joints_all.h5"

    camera_dir = agibot_dir / "camera"
    frame_ids = sorted(
        [d.name for d in camera_dir.iterdir() if d.is_dir()],
        key=lambda x: int(x),
    )

    with h5py.File(str(h5_path), "r") as f_h5, h5py.File(str(h5_all_path), "r") as f_h5_all:
        joint_pos_all = f_h5_all["state"]["joint"]["position"][:]
        joint_eff_all = f_h5_all["state"]["joint"]["effort"][:]
        joint_vel_all = f_h5_all["state"]["joint"]["velocity"][:]

        states, actions, timestamps = [], [], []
        for i, fid in enumerate(frame_ids):
            action_dict, state_dict, ts_ns = read_h5_frame(f_h5, fid)
            joint_all_dict = {
                "joint": {"position": joint_pos_all[i], "effort": joint_eff_all[i], "velocity": joint_vel_all[i]}
            }
            states.append(build_state(state_dict, joint_all_dict, extrinsic_data, frame_idx=i, fmt=fmt))
            actions.append(build_action(action_dict, fmt=fmt))
            timestamps.append(ts_ns)

    num_frames = len(frame_ids)

    # Use ideal timestamps: frame_index / fps (matches v2.1 reference format)
    ideal_timestamps = [i / fps for i in range(num_frames)]

    # Build parquet using fixed_size_list to match reference schema exactly
    # pa.list_(dtype, length) creates FixedSizeListType (matching reference parquet schema)
    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32(), state_len)),
            "action": pa.array(actions, type=pa.list_(pa.float32(), action_len)),
            "episode_index": pa.array([episode_index] * num_frames, type=pa.int64()),
            "frame_index": pa.array(list(range(num_frames)), type=pa.int64()),
            "index": pa.array(list(range(num_frames)), type=pa.int64()),
            "task_index": pa.array([0] * num_frames, type=pa.int64()),
            "timestamp": pa.array(ideal_timestamps, type=pa.float32()),
        }
    )

    # Add huggingface schema metadata (present in reference v2.1 parquet files)
    metadata = {b"huggingface": json.dumps(_arrow_schema_huggingface_metadata(state_len, action_len)).encode()}
    table = table.replace_schema_metadata(metadata)

    chunk_name = f"chunk-{episode_index // 1000:03d}"
    parquet_dir = output_dir / "data" / chunk_name
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = parquet_dir / f"episode_{episode_index:06d}.parquet"

    # Write parquet with row_group_size=1000 to match LeRobot v2.1 format
    pq.write_table(table, str(parquet_path), row_group_size=1000)
    print(f"  Wrote {parquet_path} ({num_frames} frames)")

    # Encode videos
    encode_videos(agibot_dir, output_dir, episode_index, fps=fps, fmt=fmt)

    return {"episode_index": episode_index, "length": num_frames}


# ============================================================
# Meta file generation
# ============================================================


def load_camera_parameters(agibot_dir: Path) -> dict:
    """Load camera intrinsic/extrinsic parameters from agibot sensor directory."""
    sensor_dir = agibot_dir / "parameters" / "sensor"
    if not sensor_dir.exists():
        return {}

    cam_params: dict = {}
    for fpath in sensor_dir.iterdir():
        if not fpath.name.endswith(".json"):
            continue
        with fpath.open() as f:
            cam_params[fpath.name] = json.load(f)
    return cam_params


def generate_meta(output_dir: Path, episode_info: list, agibot_dirs: list, fmt: str = "vla"):
    """Generate info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl."""
    import numpy as np
    import pyarrow.parquet as pq

    state_len, action_len = _resolve_format(fmt)

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # tasks.jsonl
    with (meta_dir / "tasks.jsonl").open("w") as f:
        f.write(json.dumps({"task_index": 0, "task": "Pop the popcorn"}, ensure_ascii=False) + "\n")

    # episodes.jsonl
    with (meta_dir / "episodes.jsonl").open("w") as f:
        for ep in episode_info:
            f.write(
                json.dumps(
                    {
                        "episode_index": ep["episode_index"],
                        "tasks": ["Pop the popcorn"],
                        "length": ep["length"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # episodes_stats.jsonl
    stats_episodes = []
    for ep in episode_info:
        ep_idx = ep["episode_index"]
        chunk_name = f"chunk-{ep_idx // 1000:03d}"
        parquet_path = output_dir / "data" / chunk_name / f"episode_{ep_idx:06d}.parquet"
        if parquet_path.exists():
            table = pq.read_table(str(parquet_path))
            rows = table.to_pylist()
            num_frames = len(rows)
            state_col = np.array([r["observation.state"] for r in rows], dtype=np.float32)
            action_col = np.array([r["action"] for r in rows], dtype=np.float32)
            frame_index_col = np.array([r["frame_index"] for r in rows], dtype=np.int64)
            index_col = np.array([r["index"] for r in rows], dtype=np.int64)
            timestamp_col = np.array([r["timestamp"] for r in rows], dtype=np.float32)

            # Video fields: per-frame stats as nested arrays (matches reference format)
            video_keys = ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
            video_stats = {}
            for key in video_keys:
                # For videos we don't have pixel data in parquet, use zeros as placeholder
                # (matching the reference which also has zeros for video stats)
                video_stats[key] = {
                    "min": [[[0.0]], [[0.0]], [[0.0]]],
                    "max": [[[0.0]], [[0.0]], [[0.0]]],
                    "mean": [[[0.0]], [[0.0]], [[0.0]]],
                    "std": [[[0.0]], [[0.0]], [[0.0]]],
                    "count": [0],
                }

            stats = {
                "episode_index": ep_idx,
                "stats": {
                    **video_stats,
                    "observation.state": {
                        "mean": state_col.mean(axis=0).tolist(),
                        "std": state_col.std(axis=0).tolist(),
                        "min": state_col.min(axis=0).tolist(),
                        "max": state_col.max(axis=0).tolist(),
                        "count": [num_frames],
                    },
                    "action": {
                        "mean": action_col.mean(axis=0).tolist(),
                        "std": action_col.std(axis=0).tolist(),
                        "min": action_col.min(axis=0).tolist(),
                        "max": action_col.max(axis=0).tolist(),
                        "count": [num_frames],
                    },
                    # Integer fields (episode_index, frame_index, index, task_index)
                    "episode_index": {
                        "min": [ep_idx],
                        "max": [ep_idx],
                        "mean": [float(ep_idx)],
                        "std": [0.0],
                        "count": [num_frames],
                    },
                    "frame_index": {
                        "min": [int(frame_index_col.min())],
                        "max": [int(frame_index_col.max())],
                        "mean": [float(frame_index_col.mean())],
                        "std": [float(frame_index_col.std())],
                        "count": [num_frames],
                    },
                    "index": {
                        "min": [int(index_col.min())],
                        "max": [int(index_col.max())],
                        "mean": [float(index_col.mean())],
                        "std": [float(index_col.std())],
                        "count": [num_frames],
                    },
                    "task_index": {
                        "min": [0],
                        "max": [0],
                        "mean": [0.0],
                        "std": [0.0],
                        "count": [num_frames],
                    },
                    "timestamp": {
                        "min": [float(timestamp_col.min())],
                        "max": [float(timestamp_col.max())],
                        "mean": [float(timestamp_col.mean())],
                        "std": [float(timestamp_col.std())],
                        "count": [num_frames],
                    },
                },
            }
            stats_episodes.append(stats)
        else:
            stats_episodes.append({})

    with (meta_dir / "episodes_stats.jsonl").open("w") as f:
        for s in stats_episodes:
            f.write(json.dumps(s) + "\n")

    # info.json
    total_frames = sum(ep["length"] for ep in episode_info)
    total_episodes = len(episode_info)
    total_chunks = max((ep["episode_index"] // 1000) for ep in episode_info) + 1

    # Build per-episode metadata from agibot dirs
    h5_path: dict = {}
    high_level_instruction: dict = {}
    instruction_segments: dict = {}
    camera_parameters: dict = {}
    intervention_info: dict = {}
    key_frame: dict = {}
    take_over: dict = {}

    for i, ep_dir in enumerate(agibot_dirs):
        h5_path[str(i)] = f"frame://{ep_dir}/aligned_joints.h5"
        high_level_instruction[str(i)] = {"high_level_instruction": ""}
        instruction_segments[str(i)] = []
        intervention_info[str(i)] = {}
        key_frame[str(i)] = {"single": [], "dual": []}
        take_over[str(i)] = []

        # Load camera parameters (intrinsics only for RGB cameras, matching reference)
        sensor_dir = ep_dir / "parameters" / "sensor"
        cam_params: dict = {}
        if sensor_dir.exists():
            # Only load intrinsic_rgb files (not depth), format as {name: content}
            for fpath in sensor_dir.iterdir():
                if fpath.name.startswith("intrinsic_") and fpath.name.endswith("_rgb.json"):
                    cam_name = fpath.name[:-5]  # Remove .json
                    with fpath.open() as f:
                        cam_params[cam_name] = json.load(f)
        camera_parameters[str(i)] = cam_params

    info = {
        "codebase_version": "v2.1",
        "robot_type": "g2a",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * 3,
        "total_chunks": total_chunks,
        "chunks_size": 1000,
        "fps": 30,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.images.top_head": {
                "dtype": "video",
                "video_info": {
                    "video.is_depth_map": False,
                    "video.fps": 30.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.hand_left": {
                "dtype": "video",
                "video_info": {
                    "video.is_depth_map": False,
                    "video.fps": 30.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.images.hand_right": {
                "dtype": "video",
                "video_info": {
                    "video.is_depth_map": False,
                    "video.fps": 30.0,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "has_audio": False,
                },
                "shape": [480, 640, 3],
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [state_len],
                "field_descriptions": _get_state_field_descriptions(fmt),
            },
            "action": {
                "dtype": "float32",
                "shape": [action_len],
                "field_descriptions": _get_action_field_descriptions(fmt),
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        },
        "data_version": "v0.1.5",
        "aid_info": {},
        # LeRobot v2.1 additional fields
        "camera_parameters": camera_parameters,
        "h5_path": h5_path,
        "high_level_instruction": high_level_instruction,
        "instruction_segments": instruction_segments,
        "intervention_info": intervention_info,
        "key_frame": key_frame,
        "take_over": take_over,
    }

    with (meta_dir / "info.json").open("w") as f:
        json.dump(info, f, indent=2)

    print(f"\nGenerated meta files in {meta_dir}/")


def _get_state_field_descriptions(fmt: str = "vla"):
    if fmt == "vla":
        return _get_state_field_descriptions_vla()
    return _get_state_field_descriptions_agibot()


def _get_action_field_descriptions(fmt: str = "vla"):
    if fmt == "vla":
        return _get_action_field_descriptions_vla()
    return _get_action_field_descriptions_agibot()


def _get_state_field_descriptions_agibot():
    return {
        "state/left_effector/position": {"description": "", "dimensions": 1, "indices": [0]},
        "state/right_effector/position": {"description": "", "dimensions": 1, "indices": [1]},
        "state/end/wrench": {"description": "", "dimensions": 0, "indices": []},
        "state/end/position": {"description": "", "dimensions": 6, "indices": list(range(2, 8))},
        "state/end/velocity": {"description": "", "dimensions": 0, "indices": []},
        "state/end/orientation": {"description": "", "dimensions": 8, "indices": list(range(8, 16))},
        "state/end/arm_orientation": {"description": "", "dimensions": 8, "indices": list(range(16, 24))},
        "state/end/arm_position": {"description": "", "dimensions": 6, "indices": list(range(24, 30))},
        "state/joint/position": {"description": "", "dimensions": 14, "indices": list(range(30, 44))},
        "state/joint/current_value": {"description": "", "dimensions": 0, "indices": []},
        "state/joint/effort": {"description": "", "dimensions": 14, "indices": list(range(44, 58))},
        "state/joint/velocity": {"description": "", "dimensions": 14, "indices": list(range(58, 72))},
        "state/head/position": {"description": "", "dimensions": 3, "indices": list(range(72, 75))},
        "state/waist/position": {"description": "", "dimensions": 5, "indices": list(range(75, 80))},
        "state/robot/position": {"description": "", "dimensions": 3, "indices": list(range(80, 83))},
        "state/robot/orientation": {"description": "", "dimensions": 4, "indices": list(range(83, 87))},
        "state/operator_event/action_src_status": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/controlled": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/rows": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/cols": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/resolution_x": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/resolution_y": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/normal_force": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/shear_force_x": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/shear_force_y": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/contact": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/valid": {"description": "", "dimensions": 0, "indices": []},
        "state/left_ee_force/err_code": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/controlled": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/rows": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/cols": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/resolution_x": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/resolution_y": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/normal_force": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/shear_force_x": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/shear_force_y": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/contact": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/valid": {"description": "", "dimensions": 0, "indices": []},
        "state/right_ee_force/err_code": {"description": "", "dimensions": 0, "indices": []},
        "extrinsic_end_T_hand_left_rgbd_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(87, 96)),
        },
        "extrinsic_end_T_hand_right_rgbd_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(96, 105)),
        },
        "extrinsic_end_T_head_left_fisheye_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(105, 114)),
        },
        "extrinsic_end_T_head_right_fisheye_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(114, 123)),
        },
        "extrinsic_end_T_head_front_rgbd_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(123, 132)),
        },
        "extrinsic_end_T_head_back_fisheye_aligned/rotation_matrix": {
            "description": "",
            "dimensions": 9,
            "indices": list(range(132, 141)),
        },
        "extrinsic_end_T_hand_left_rgbd_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(141, 144)),
        },
        "extrinsic_end_T_hand_right_rgbd_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(144, 147)),
        },
        "extrinsic_end_T_head_left_fisheye_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(147, 150)),
        },
        "extrinsic_end_T_head_right_fisheye_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(150, 153)),
        },
        "extrinsic_end_T_head_front_rgbd_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(153, 156)),
        },
        "extrinsic_end_T_head_back_fisheye_aligned/translation_vector": {
            "description": "",
            "dimensions": 3,
            "indices": list(range(156, 159)),
        },
    }


def _get_action_field_descriptions_agibot():
    return {
        "action/left_effector/position": {"description": "", "dimensions": 1, "indices": [0]},
        "action/right_effector/position": {"description": "", "dimensions": 1, "indices": [1]},
        "action/end/position": {"description": "", "dimensions": 6, "indices": list(range(2, 8))},
        "action/end/orientation": {"description": "", "dimensions": 8, "indices": list(range(8, 16))},
        "action/joint/position": {"description": "", "dimensions": 14, "indices": list(range(16, 30))},
        "action/head/position": {"description": "", "dimensions": 3, "indices": list(range(30, 33))},
        "action/waist/position": {"description": "", "dimensions": 5, "indices": list(range(33, 38))},
        "action/robot/velocity": {"description": "", "dimensions": 2, "indices": list(range(38, 40))},
    }


def _get_state_field_descriptions_vla():
    return {
        "state/joint_position": {
            "description": "arm joint angles (14 dims: left_arm[7] + right_arm[7])",
            "dimensions": 14,
            "indices": list(range(VLA_STATE_JOINT_POSITION, VLA_STATE_JOINT_POSITION + 14)),
        },
        "state/gripper_position": {
            "description": "gripper positions (left, right)",
            "dimensions": 2,
            "indices": list(range(VLA_STATE_GRIPPER_POSITION, VLA_STATE_GRIPPER_POSITION + 2)),
        },
    }


def _get_action_field_descriptions_vla():
    return {
        "action/joint_position": {
            "description": "arm joint target angles (14 dims: left_arm[7] + right_arm[7])",
            "dimensions": 14,
            "indices": list(range(VLA_ACTION_JOINT_POSITION, VLA_ACTION_JOINT_POSITION + 14)),
        },
        "action/gripper_position": {
            "description": "gripper target positions (left, right)",
            "dimensions": 2,
            "indices": list(range(VLA_ACTION_GRIPPER_POSITION, VLA_ACTION_GRIPPER_POSITION + 2)),
        },
    }


# ============================================================
# Public API
# ============================================================


def convert_agibot_to_lerobot(
    agibot_dir: Path,
    output_dir: Path,
    lerobot_ref_dir: Optional[Path] = None,
    fps: float = 30.0,
    fmt: str = "vla",
) -> dict:
    """Convert one or more agibot episodes to LeRobot v2.1 format.

    Auto-detects single vs batch from the ``agibot_dir`` layout: if
    ``agibot_dir/aligned_joints.h5`` exists, it's treated as a single
    episode; otherwise the directory is scanned for episode
    subdirectories (each containing ``aligned_joints.h5``).

    Parameters
    ----------
    agibot_dir
        Path to an episode dir (single) or its parent (batch).
    output_dir
        Where the LeRobot dataset is written (``data/``, ``videos/``,
        ``meta/`` are created here).
    lerobot_ref_dir
        Optional reference LeRobot dataset (must contain
        ``data/chunk-000/episode_000000.parquet``). When provided, the
        converter fills missing fisheye / head_back extrinsic columns
        from the reference. When ``None`` (the default), those
        columns are left empty.
    fps
        Video frame rate (default ``30.0``).
    fmt
        Output format: ``"vla"`` (16-dim state / 16-dim action — arm joints +
        gripper, suitable for pi0.5 and standard VLA training; default) or
        ``"agibot"`` (159-dim state / 40-dim action, full vector).

    Returns
    -------
    Manifest dict::

        {
            "episodes": [{"episode_index": int, "length": int}, ...],
            "total_episodes": int,
            "total_frames": int,
            "output_dir": str,
        }

    Raises
    ------
    RuntimeError
        If heavy deps (``h5py``/``pyarrow``/``numpy``) are missing, or no
        episode directories are found, or ffmpeg is missing when video
        fallback encoding is needed.
    """
    # Pre-flight checks (fail fast with friendly messages).
    _require_heavy_deps()

    agibot_dir = Path(agibot_dir).resolve()
    output_dir = Path(output_dir).resolve()

    lerobot_template = None
    if lerobot_ref_dir is not None:
        lerobot_template = load_lerobot_reference(Path(lerobot_ref_dir).resolve())

    # Auto-detect episodes
    if (agibot_dir / "aligned_joints.h5").exists():
        episodes = [agibot_dir]
    else:
        episodes = detect_episodes(agibot_dir)

    if not episodes:
        raise RuntimeError(
            f"No episode directories found in {agibot_dir}. " "Each episode dir must contain aligned_joints.h5."
        )

    print(f"Found {len(episodes)} episode(s) to convert:")
    for ep in episodes:
        print(f"  - {ep}")

    episode_info = []
    for i, ep_dir in enumerate(episodes):
        result = convert_episode(ep_dir, output_dir, i, lerobot_template, fps=fps, fmt=fmt)
        episode_info.append(result)

    generate_meta(output_dir, episode_info, episodes, fmt=fmt)

    total_frames = sum(ep["length"] for ep in episode_info)
    print(f"\nConversion complete! {len(episode_info)} episode(s), {total_frames} total frames.")

    return {
        "episodes": episode_info,
        "total_episodes": len(episode_info),
        "total_frames": total_frames,
        "output_dir": str(output_dir),
    }


# ============================================================
# CLI wrapper (called by `geniesim dataset convert agibot-to-lerobot`)
# ============================================================


def convert_cli(argv: list) -> int:
    """argparse front-end for :func:`convert_agibot_to_lerobot`.

    Returns a process exit code (``0`` on success, non-zero on
    failure). The ``geniesim dataset convert agibot-to-lerobot``
    dispatcher in ``geniesim_cli/commands/dataset.py`` is the only
    caller; the script is no longer invocable directly.
    """
    parser = argparse.ArgumentParser(
        prog="geniesim dataset convert agibot-to-lerobot",
        description="Convert agibot episode data to LeRobot v2.1 format. "
        "Supports single episode (--agibot-dir <one-episode>) "
        "and batch (--agibot-dir <parent-dir-of-episodes>) modes.",
    )
    parser.add_argument(
        "--agibot-dir",
        type=Path,
        required=True,
        help="agibot episode dir, or parent dir containing multiple episode subdirs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for LeRobot-format data",
    )
    parser.add_argument(
        "--lerobot-ref-dir",
        type=Path,
        default=None,
        help="Optional reference LeRobot dataset (must contain "
        "data/chunk-000/episode_000000.parquet) — used to fill missing "
        "fisheye / head_back extrinsic columns. Omit to leave them empty.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS (default 30.0)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=SUPPORTED_FORMATS,
        default="vla",
        help="Output schema format: 'vla' (16+16 dim: arm joints + gripper; "
        "default) or 'agibot' (full 159+40 dim vectors).",
    )
    args = parser.parse_args(argv)

    try:
        convert_agibot_to_lerobot(
            agibot_dir=args.agibot_dir,
            output_dir=args.output_dir,
            lerobot_ref_dir=args.lerobot_ref_dir,
            fps=args.fps,
            fmt=args.format,
        )
    except RuntimeError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    return 0
