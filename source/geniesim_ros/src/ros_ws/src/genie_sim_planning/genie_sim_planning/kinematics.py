# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import math

import numpy as np


class RateLimiter:
    def __init__(self, max_acc=1.0, max_jerk=5.0, dt=0.01):
        self.max_acc = max_acc
        self.max_jerk = max_jerk
        self.dt = dt
        self.prev = 0.0
        self.prev_dot = 0.0

    def step(self, target):
        desired_dot = (target - self.prev) / self.dt
        jerk = (desired_dot - self.prev_dot) / self.dt
        if jerk > self.max_jerk:
            desired_dot = self.prev_dot + self.max_jerk * self.dt
        elif jerk < -self.max_jerk:
            desired_dot = self.prev_dot - self.max_jerk * self.dt
        if desired_dot > self.max_acc:
            desired_dot = self.max_acc
        elif desired_dot < -self.max_acc:
            desired_dot = -self.max_acc
        self.prev += desired_dot * self.dt
        self.prev_dot = desired_dot
        return self.prev


class FourWheelSteeringRobot:
    def __init__(self, wheelbase, track_width):
        self.L = wheelbase
        self.W = track_width

    def inverse_kinematics(self, v, delta):
        if abs(delta) < 1e-6:
            return [v, v, v, v], [0.0, 0.0, 0.0, 0.0]

        R = self.L / math.tan(delta)

        theta_fl = math.atan(self.L / (R - self.W / 2))
        theta_fr = math.atan(self.L / (R + self.W / 2))
        theta_rl = -theta_fl
        theta_rr = -theta_fr

        v_fl = math.copysign(v * math.sqrt((R - self.W / 2) ** 2 + self.L**2) / abs(R), v)
        v_fr = math.copysign(v * math.sqrt((R + self.W / 2) ** 2 + self.L**2) / abs(R), v)
        v_rl = math.copysign(v * math.sqrt((R - self.W / 2) ** 2 + self.L**2) / abs(R), v)
        v_rr = math.copysign(v * math.sqrt((R + self.W / 2) ** 2 + self.L**2) / abs(R), v)

        return [v_fl, v_fr, v_rl, v_rr], [theta_fl, theta_fr, theta_rl, theta_rr]


def compute_ackermann_angles(curvature, L, W):
    if abs(curvature) < 1e-6:
        return 0.0, 0.0
    R = 1.0 / abs(curvature)
    if curvature > 0:
        delta_in = np.arctan(L / (R - W / 2))
        delta_out = np.arctan(L / (R + W / 2))
        return delta_in, delta_out
    else:
        delta_in = np.arctan(L / (R - W / 2))
        delta_out = np.arctan(L / (R + W / 2))
        return -delta_out, -delta_in
