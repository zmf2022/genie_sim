# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from abc import abstractmethod
from geniesim.benchmark.ader.action.action_parsing import do_parsing


class BaseTask:

    def __init__(self, env):
        self.config = env.task_info

        self.problem_name, self.objects, self.action, self.task_progress = do_parsing(
            env.specific_task_name, env
        )

    def update_progress(self, id, progress):
        for item in self.task_progress:
            if item["id"] == id:
                item["progress"] = progress
                break

    @abstractmethod
    def reset_scene(self, env):
        """
        Task-specific scene reset

        :param env: environment instance
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_agent(self, env):
        """
        Task-specific agent reset

        :param env: environment instance
        """
        raise NotImplementedError()

    def reset_variables(self, env):
        """
        Task-specific variable reset

        :param env: environment instance
        """
        return

    def reset(self, env):
        self.reset_variables(env)

    @abstractmethod
    def get_task_obs(self, env):
        """
        Get task-specific observation

        :param env: environment instance
        :return: task-specific observation (numpy array)
        """
        raise NotImplementedError()

    def step(self, env):
        """
        Perform task-specific step for every timestep

        :param env: environment instance
        """
        return
