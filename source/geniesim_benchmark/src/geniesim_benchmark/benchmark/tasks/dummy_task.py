# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np

from .base_task import BaseTask


class DummyTask(BaseTask):
    """
    No-op placeholder task: all hooks are inert stubs and it exposes no
    task observation (task_obs_dim = 0).
    """

    def __init__(self, env):
        super(DummyTask, self).__init__(env)
        self.task_obs_dim = 0

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        return

    def reset_agent(self, env):
        """
        Task-specific agent reset (no-op stub).

        :param env: environment instance
        """
        return

    def get_task_obs(self, env):
        """
        Task-specific observation (none — returns an empty array).

        :param env: environment instance
        :return: task-specific observation
        """
        return np.zeros(0, dtype=np.float32)
