# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


def publish_path(publisher, path_points, frame_id="map", stamp=None):
    msg = Path()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    for p in path_points:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        if stamp is not None:
            ps.header.stamp = stamp
        ps.pose.position.x = float(p["x"])
        ps.pose.position.y = float(p["y"])
        ps.pose.position.z = float(p.get("z", 0.0))
        yaw = float(p.get("yaw", 0.0))
        ps.pose.orientation.w = math.cos(yaw / 2)
        ps.pose.orientation.z = math.sin(yaw / 2)
        msg.poses.append(ps)
    publisher.publish(msg)
