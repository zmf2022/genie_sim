# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
#
# ``solve_dare`` and ``dlqr`` are adapted from Atsushi Sakai's
# PythonRobotics library (MIT-licensed; full text in
# ``../THIRD_PARTY_LICENSES.md``). Upstream reference:
#   https://github.com/AtsushiSakai/PythonRobotics
#   PathTracking/lqr_speed_steer_control/lqr_speed_steer_control.py

import math

import numpy as np
import scipy.linalg as la


def normalize_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def angle_distance(a, b):
    return abs(normalize_angle(a - b))


def solve_dare(A, B, Q, R):
    X = Q.copy()
    for _ in range(150):
        Xn = A.T @ X @ A - A.T @ X @ B @ la.inv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max() < 0.01:
            break
        X = Xn
    return Xn


def dlqr(A, B, Q, R):
    X = solve_dare(A, B, Q, R)
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
    return K
