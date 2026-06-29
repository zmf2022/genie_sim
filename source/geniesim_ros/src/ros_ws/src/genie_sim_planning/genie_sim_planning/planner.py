# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np
from pyclothoids import SolveG2


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def generate_clothoid(x0, y0, yaw0, k0, x1, y1, yaw1, k1, n_samples=500, v_nominal=0.6):
    dx = x1 - x0
    dy = y1 - y0
    goal_vec = np.array([dx, dy])
    heading_vec = np.array([np.cos(yaw0), np.sin(yaw0)])
    if np.dot(goal_vec, heading_vec) < 0:
        yaw0_fwd = wrap_to_pi(yaw0 + np.pi)
        yaw1_fwd = wrap_to_pi(yaw1 + np.pi)
        segments = SolveG2(x0, y0, yaw0_fwd, k0, x1, y1, yaw1_fwd, k1)
        v_sign = -v_nominal
    else:
        segments = SolveG2(x0, y0, yaw0, k0, x1, y1, yaw1, k1)
        v_sign = +v_nominal

    total_length = sum(seg.length for seg in segments)
    xs, ys, thetas, kappas, s_vals = [], [], [], [], []
    accumulated_s = 0.0
    for seg in segments:
        n_seg = max(2, int(n_samples * seg.length / total_length))
        s_segment = np.linspace(0, seg.length, n_seg, endpoint=False)
        for s in s_segment:
            xs.append(seg.X(s))
            ys.append(seg.Y(s))
            thetas.append(seg.Theta(s))
            kappas.append(seg.ThetaD(s))
            s_vals.append(accumulated_s + s)
        accumulated_s += seg.length
    last_seg = segments[-1]
    xs.append(last_seg.X(last_seg.length))
    ys.append(last_seg.Y(last_seg.length))
    thetas.append(last_seg.Theta(last_seg.length))
    kappas.append(last_seg.ThetaD(last_seg.length))
    s_vals.append(accumulated_s)

    if v_sign < 0:
        thetas = np.array([wrap_to_pi(theta - np.pi) for theta in thetas])
    else:
        thetas = np.array(thetas)

    xs = np.array(xs)
    ys = np.array(ys)
    kappas = np.array(kappas)
    s_vals = np.array(s_vals)
    vs = np.full_like(xs, v_sign)

    path_points = [(float(xs[i]), float(ys[i]), float(thetas[i]), float(vs[i])) for i in range(len(xs))]
    return path_points, (s_vals, xs, ys, thetas, kappas), v_sign
