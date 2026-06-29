# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from __future__ import annotations

import math

import rclpy
from tf2_ros import LookupException, ExtrapolationException

from .math_utils import normalize_angle


def lookup_pose(tf_buffer, target_frame: str, source_frame: str = "map"):
    try:
        tf = tf_buffer.lookup_transform(source_frame, target_frame, rclpy.time.Time())
        t = tf.transform.translation
        q = tf.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        return t.x, t.y, yaw
    except (LookupException, ExtrapolationException):
        return None


def pose_stamped_in_map(tf_buffer, msg, warn_fn=None):
    frame_id = msg.header.frame_id or "map"
    px = msg.pose.position.x
    py = msg.pose.position.y
    pz = msg.pose.position.z
    q = msg.pose.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    pose_yaw = math.atan2(siny, cosy)
    if frame_id == "map":
        return px, py, pz, pose_yaw
    try:
        tf = tf_buffer.lookup_transform("map", frame_id, rclpy.time.Time())
        t = tf.transform.translation
        tq = tf.transform.rotation
        tsiny = 2.0 * (tq.w * tq.z + tq.x * tq.y)
        tcosy = 1.0 - 2.0 * (tq.y * tq.y + tq.z * tq.z)
        tf_yaw = math.atan2(tsiny, tcosy)
        c, s = math.cos(tf_yaw), math.sin(tf_yaw)
        mx = c * px - s * py + t.x
        my = s * px + c * py + t.y
        mz = pz + t.z
        myaw = normalize_angle(pose_yaw + tf_yaw)
        return mx, my, mz, myaw
    except (LookupException, ExtrapolationException) as e:
        if warn_fn:
            warn_fn(f"TF {frame_id}->map unavailable: {e}")
        return None


def point_stamped_in_map(tf_buffer, msg, warn_fn=None):
    frame_id = msg.header.frame_id or "map"
    if frame_id == "map":
        return msg.point.x, msg.point.y
    try:
        tf = tf_buffer.lookup_transform("map", frame_id, rclpy.time.Time())
        t = tf.transform.translation
        q = tf.transform.rotation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        c, s = math.cos(yaw), math.sin(yaw)
        x = c * msg.point.x - s * msg.point.y + t.x
        y = s * msg.point.x + c * msg.point.y + t.y
        return x, y
    except (LookupException, ExtrapolationException) as e:
        if warn_fn:
            warn_fn(f"TF {frame_id}->map unavailable: {e}")
        return None
