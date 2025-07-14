# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from enum import Enum
import uuid
import os, json
from .logger import Logger
from collections import deque

logger = Logger()


class ErrorCode(Enum):
    INIT_VALUE = -1
    SUCCESS = 0
    ABNORMAL_INTERRUPTION = 1
    OUT_OF_MAX_STEP = 2

    UNKNOWN_ERROR = 500


EVAL_TEMPLATE = {
    "task_type": "benchmark",
    "task_uid": str(uuid.uuid4()),
    "task_name": "",
    "stage": "",
    "result": {"code": int(ErrorCode.INIT_VALUE.value), "step": 0, "msg": ""},
    "start_time": "",
    "end_time": "",
}

TASK_STEPS = {
    "iros_clear_the_countertop_waste": [
        "Follow",
        "PickUpOnGripper",
        "Inside",
        "Follow",
        "PickUpOnGripper",
        "Inside",
    ],
    "iros_open_drawer_and_store_items": [
        "PushPull",
        "Follow",
        "PickUpOnGripper",
        "Inside",
        "PushPull",
    ],
    "iros_heat_the_food_in_the_microwave": [
        "PushPull",
        "Follow",
        "PickUpOnGripper",
        "Inside",
        "PushPull",
        "TriggerAction",
    ],
    "iros_pack_moving_objects_from_conveyor": [
        "Follow",
        "PickUpOnGripper",
        "Follow",
        "Inside",
    ],
    "iros_pickup_items_from_the_freezer": [
        "PushPull",
        "Follow",
        "PickUpOnGripper",
        "Inside",
        "PushPull",
    ],
    "iros_restock_supermarket_items": [
        "Follow",
        "PickUpOnGripper",
        "Follow",
        "OnShelf",
    ],
    "iros_pack_in_the_supermarket": [
        "Follow",
        "PickUpOnGripper",
        "Follow",
        "Inside",
    ],
    "iros_make_a_sandwich": [
        "Follow",
        "PickUpOnGripper",
        "Cover",
        "Follow",
        "PickUpOnGripper",
        "Cover",
        "Follow",
        "PickUpOnGripper",
        "Cover",
        "Follow",
        "PickUpOnGripper",
        "Cover",
    ],
    "iros_clear_table_in_the_restaurant": [
        "Follow",
        "PickUpOnGripper",
        "Follow",
        "Ontop",
    ],
    "iros_stamp_the_seal": [
        "Follow",
        "PickUpOnGripper",
        "TriggerAction",
        "Follow",
        "Ontop",
    ],
}


def summarize_scores(single_evaluate_ret, task_name):
    if not task_name.startswith("iros_"):
        return

    eval_result = single_evaluate_ret["result"]
    episode_progress = eval_result.get("progress", [])

    eval_result["scores"] = {"STEPS": {}, "E2E": 0}
    for i in range(len(TASK_STEPS[task_name])):
        eval_result["scores"]["STEPS"][f"STEP{i}"] = 0.0
    if episode_progress == []:
        pass
    else:
        step_idx = 0
        substeps = deque(TASK_STEPS[task_name])
        for p in episode_progress:
            progress = p.get("progress")
            if not isinstance(progress, dict):
                continue
            if not progress.get("SCORE"):
                continue
            if substeps:
                if substeps[0] == p.get("class_name"):
                    substeps.popleft()
                    step = f"STEP{step_idx}"
                    eval_result["scores"]["STEPS"][step] = float(progress["SCORE"])
                    step_idx += 1
        step_keys = list(eval_result["scores"]["STEPS"].keys())
        if len(step_keys):
            last_step = step_keys[-1]
            if eval_result["scores"]["STEPS"][last_step] == 1:
                logger.info(f"E2E Succeed!!!")
                eval_result["scores"]["E2E"] = 1


def generate_eval_file_path(dir_path, prefix_name):
    count = 0
    while os.path.exists(os.path.join(dir_path, f"{prefix_name}_{count:02d}.json")):
        count += 1
    new_filename = f"{prefix_name}_{count:02d}.json"
    return os.path.join(dir_path, new_filename)


def dump_eval_result(out_dir, content):
    os.makedirs(out_dir, exist_ok=True)
    file_path = generate_eval_file_path(out_dir, "evaluate_ret")
    logger.info(f"Evaluation result file generated at {file_path}")
    with open(file_path, "w+") as f:
        json.dump(content, f, indent=4)
