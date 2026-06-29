# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os


def resolve_sub_usd_path(scene_cfg, benchmark_conf_path, sub_task_name, instance_id):
    if not sub_task_name:
        return ""

    override_root = scene_cfg.get("sub_usd_override_root", "")
    if override_root:
        return os.path.join(override_root, str(instance_id), "scene.usda")

    return os.path.join(
        benchmark_conf_path,
        "llm_task",
        sub_task_name,
        str(instance_id),
        "scene.usda",
    )
