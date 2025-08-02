# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import os
from .common_actions import *
from .custom import (
    PickUpOnGripper,
    Inside,
    FluidInside,
    Onfloor,
    Ontop,
    Cover,
    CheckParticleInBBox,
    PushPull,
    Follow,
    OnShelf,
    OnShelfCurobo,
    TriggerAction,
)

from geniesim.utils import benchmark_task_definitions_path

# PATHS
EVAL_CONFIGS_PATH = benchmark_task_definitions_path()


def get_definition_filename(behavior_activity, instance):
    ori_path = os.path.join(
        EVAL_CONFIGS_PATH,
        behavior_activity,
        f"problem{instance}.json",
    )

    if os.path.exists(ori_path):
        return ori_path

    return os.path.join(
        EVAL_CONFIGS_PATH,
        "default_problem.json",
    )


def scan_config(filename=None):
    if filename is not None:
        with open(filename) as f:
            eval_config = json.load(f)
    else:
        raise ValueError("No input JSON provided.")

    return eval_config


def get_object_scope(object_terms):
    scope = {}
    for object_cat in object_terms:
        for object_inst in object_terms[object_cat]:
            scope[object_inst] = None
    return scope


def do_parsing(behavior_activity, env):
    acts_filename = get_definition_filename(behavior_activity, 0)
    cfg = scan_config(acts_filename)
    problem_name = cfg["Problem"]
    objects = cfg["Objects"]
    init_progress = []
    acts = parse_action(cfg["Acts"][0], init_progress, env)
    return problem_name, objects, acts, init_progress


def record_act_obj(act, init_progress):
    init_progress.append(
        {"class_name": act.__class__.__name__, "id": hex(id(act)), "progress": ""}
    )


def parse_action(obj: dict, init_progress, env) -> ActionBase:
    for key, value in obj.items():
        if key == "ActionList":
            act = ActionList(env)
            record_act_obj(act, init_progress)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, init_progress, env))
            return act
        elif key == "ActionSetWaitAny":
            act = ActionSetWaitAny(env)
            record_act_obj(act, init_progress)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, init_progress, env))
            return act
        elif key == "ActionSetWaitAll":
            act = ActionSetWaitAll(env)
            record_act_obj(act, init_progress)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, init_progress, env))
            return act
        elif key == "ActionWaitForTime":
            act = ActionWaitForTime(env, wait_time=value)
            record_act_obj(act, init_progress)
            return act
        elif key == "PickUpOnGripper":
            params = value.split("|")
            act = PickUpOnGripper(env, params[0], params[1])
            record_act_obj(act, init_progress)
            return act
        elif key == "Timeout":
            act = TimeOut(env, time_out=float(value))
            record_act_obj(act, init_progress)
            return act
        elif key == "Inside":
            params = value.split("|")
            act = Inside(env, params[0], params[1], params[2])
            record_act_obj(act, init_progress)
            return act
        elif key == "FluidInside":
            params = value.split("|")
            act = FluidInside(env, params[0], params[1])
            record_act_obj(act, init_progress)
            return act
        elif key == "StepOut":
            act = StepOut(env, max_step=value)
            record_act_obj(act, init_progress)
            return act
        elif key == "Onfloor":
            params = value.split("|")
            act = Onfloor(env, obj_name=params[0], height=params[1])
            record_act_obj(act, init_progress)
            return act
        elif key == "Ontop":
            params = value.split("|")
            act = Ontop(env, params[0], params[1], float(params[2]), float(params[3]))
            record_act_obj(act, init_progress)
            return act
        elif key == "Cover":
            params = value.split("|")
            act = Cover(env, params[0], params[1])
            record_act_obj(act, init_progress)
            return act
        elif key == "PushPull":
            params = value.split("|")
            act = PushPull(env, params[0], params[1], params[2])
            record_act_obj(act, init_progress)
            return act
        elif key == "Follow":
            params = value.split("|")
            act = Follow(env, params[0], params[1], params[2])
            record_act_obj(act, init_progress)
            return act
        elif key == "CheckParticleInBBox":
            params = value.split("|")
            values = params[1].split(",")
            act = CheckParticleInBBox(
                env,
                int(params[0]),
                [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                    float(values[4]),
                    float(values[5]),
                ],
            )
            record_act_obj(act, init_progress)
            return act
        elif key == "OnShelf":
            params = value.split("|")
            act = OnShelf(
                env,
                params[0],
                params[1],
                params[2],
                params[3],
            )
            record_act_obj(act, init_progress)
            return act
        elif key == "OnShelfCurobo":
            params = value.split("|")
            act = OnShelfCurobo(
                env,
                params[0],
                params[1],
                params[2],
                params[3],
            )
            record_act_obj(act, init_progress)
            return act
        elif key == "TriggerAction":
            params = value.split("|")
            act = TriggerAction(env, params[0], params[1])
            record_act_obj(act, init_progress)
            return act
        else:
            raise ValueError(f"Unknown action type: {key}")
    return None
