# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .hook_base import HookBase


class TaskHook(HookBase):
    def __init__(self, policy):
        self.policy = policy

    def start_callback(self, env, _):
        pass

    def step_callback(self, env, action):
        pass

    def end_callback(self, env, _):
        pass

    def gather_results(self):
        pass
