#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import math
from enum import Enum
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from geometry_msgs.msg import PoseStamped, PointStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from tf2_ros import Buffer, TransformListener

from genie_sim_planning.math_utils import normalize_angle, dlqr
from rclpy.qos import qos_profile_system_default
from genie_sim_planning.tf_utils import (
    lookup_pose,
    pose_stamped_in_map,
    point_stamped_in_map,
)
from genie_sim_planning.path_utils import publish_path
from genie_sim_planning.planner import generate_clothoid


class NavStage(Enum):
    IDLE = 0
    FOLLOW_TRAJECTORY = 1
    MOVE_ROTATE = 2
    MOVE_GO = 3
    TURN_IN_PLACE = 4


class NavigationNode(Node):
    def __init__(self):
        super().__init__(
            "navigation",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )

        self.declare_parameter("wheelbase", 0.46)
        self.declare_parameter("track_width", 0.436)
        self.declare_parameter("control_rate", 100.0)
        self.declare_parameter("linear_speed", 1.0)
        self.declare_parameter("target_speed", 0.5)
        self.declare_parameter("goal_tolerance_xy", 0.05)
        self.declare_parameter("goal_tolerance_yaw", 0.10)
        self.declare_parameter("yaw_tolerance", 0.02)
        self.declare_parameter("k_omega", 5.0)
        self.declare_parameter("max_omega", 0.8)
        self.declare_parameter("mode", "move_base")

        self.L = float(self.get_parameter("wheelbase").value)
        self.W = float(self.get_parameter("track_width").value)
        self.dt = 1.0 / float(self.get_parameter("control_rate").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.target_speed = float(self.get_parameter("target_speed").value)
        self.goal_tolerance_xy = float(self.get_parameter("goal_tolerance_xy").value)
        self.goal_tolerance_yaw = float(self.get_parameter("goal_tolerance_yaw").value)
        self.yaw_tolerance = float(self.get_parameter("yaw_tolerance").value)
        self.k_omega = float(self.get_parameter("k_omega").value)
        self.max_omega = float(self.get_parameter("max_omega").value)
        self.navigation_mode = self.get_parameter("mode").value

        self.stage = NavStage.IDLE

        self.path = []
        self.path_dir = 0.0
        self.has_path = False
        self.lqr_e = 0.0
        self.lqr_e_th = 0.0

        self.has_goal = False
        self.goal_x = 0.0
        self.goal_y = 0.0

        self.has_turn_goal = False
        self.turn_target_yaw = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 1)
        self.create_subscription(PointStamped, "/clicked_point", self.clicked_point_cb, 1)
        self.create_subscription(Odometry, "/odom", self.odom_cb, qos_profile_system_default)
        self.create_subscription(String, "/navigation_mode", self.mode_cb, 1)

        self.pub_cmd_twist = self.create_publisher(Twist, "/cmd_twist", 1)
        from nav_msgs.msg import Path

        self.path_pub = self.create_publisher(Path, "/my_path", 1)

        self.odom = Odometry()
        self.timer = self.create_timer(self.dt, self.control_loop)

        self._throttle_last_ns = {}
        self.get_logger().info(f"NavigationNode online, mode={self.navigation_mode}")

    def _warn_throttle(self, key, period_sec, message):
        now_ns = int(self.get_clock().now().nanoseconds)
        last_ns = self._throttle_last_ns.get(key)
        period_ns = int(period_sec * 1e9)
        if last_ns is None or (now_ns - last_ns) >= period_ns:
            self._throttle_last_ns[key] = now_ns
            self.get_logger().warn(message)

    def _lookup_pose(self):
        return lookup_pose(self.tf_buffer, "base_link")

    def _pose_in_map(self, msg):
        return pose_stamped_in_map(self.tf_buffer, msg, warn_fn=lambda m: self._warn_throttle("tf_goal", 2.0, m))

    def _point_in_map(self, msg):
        return point_stamped_in_map(self.tf_buffer, msg, warn_fn=lambda m: self._warn_throttle("tf_point", 2.0, m))

    def _publish_path(self, path_points, frame_id="map"):
        publish_path(self.path_pub, path_points, frame_id, stamp=self.get_clock().now().to_msg())

    def publish_twist(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.pub_cmd_twist.publish(msg)

    def stop(self):
        self.stage = NavStage.IDLE
        self.has_goal = False
        self.has_path = False
        self.has_turn_goal = False

    def mode_cb(self, msg):
        new_mode = msg.data
        if new_mode in ("move_base", "clothoid"):
            self.navigation_mode = new_mode
            self.stop()
            self.get_logger().info(f"Navigation mode switched to: {self.navigation_mode}")
        else:
            self.get_logger().warn(f"Unknown nav mode: {new_mode}, ignoring")

    def odom_cb(self, msg):
        self.odom = msg

    def clicked_point_cb(self, msg):
        pose = self._lookup_pose()
        if pose is None:
            return
        x, y, yaw = pose
        pt = self._point_in_map(msg)
        if pt is None:
            return
        dx = pt[0] - x
        dy = pt[1] - y
        if math.hypot(dx, dy) < 1e-3:
            self.get_logger().info("Clicked point too close, ignoring.")
            return
        self.turn_target_yaw = math.atan2(dy, dx)
        self.has_turn_goal = True
        self.has_goal = False
        self.has_path = False
        self.stage = NavStage.TURN_IN_PLACE
        self.get_logger().info(f"Turn in place: target yaw = {math.degrees(self.turn_target_yaw):.1f} deg")

    def goal_cb(self, msg):
        pose = self._lookup_pose()
        if pose is None:
            self.get_logger().warn("Goal received but no robot pose, ignoring.")
            return

        x0, y0, yaw0 = pose
        goal_frame = msg.header.frame_id or "map"

        goal = self._pose_in_map(msg)
        if goal is None:
            self.get_logger().warn("Cannot transform goal to map frame, ignoring.")
            return
        x1, y1, z1, yaw1 = goal

        if z1 > 0.1:
            self.navigation_mode = "clothoid"
            self.get_logger().info("Goal z > 0.1, using clothoid mode")
        elif z1 < -0.1:
            self.navigation_mode = "move_base"
            self.get_logger().info("Goal z < -0.1, using move_base mode")

        self.has_turn_goal = False

        if self.navigation_mode == "clothoid":
            self._start_clothoid(x0, y0, yaw0, x1, y1, yaw1, goal_frame)
        else:
            self._start_move_base(x0, y0, yaw0, x1, y1, goal_frame)

    def _start_clothoid(self, x0, y0, yaw0, x1, y1, yaw1, frame_id="map"):
        self.lqr_e = 0.0
        self.lqr_e_th = 0.0

        path, (S, X, Y, YAW, KAPPA), direction = generate_clothoid(x0, y0, yaw0, 0.0, x1, y1, yaw1, 0.0, n_samples=100)

        self.path = []
        for xi, yi, yawi, ki, si in zip(X, Y, YAW, KAPPA, S):
            self.path.append(
                {
                    "x": float(xi),
                    "y": float(yi),
                    "z": 0.0,
                    "yaw": float(yawi),
                    "k": float(ki),
                    "s": float(si),
                }
            )

        self._publish_path(self.path)
        self.path_dir = 1.0 if direction > 0 else (-1.0 if direction < 0 else 0.0)
        self.has_path = True
        self.has_goal = False
        self.stage = NavStage.FOLLOW_TRAJECTORY
        self.get_logger().info(f"Clothoid path with {len(self.path)} points")

    def _start_move_base(self, x0, y0, yaw0, x1, y1, frame_id="map"):
        self.goal_x = x1
        self.goal_y = y1
        self.has_goal = True
        self.has_path = False
        self.stage = NavStage.MOVE_GO

        path = []
        for i in range(51):
            s = i / 50.0
            path.append(
                {
                    "x": x0 + s * (x1 - x0),
                    "y": y0 + s * (y1 - y0),
                    "z": 0.0,
                    "yaw": float(yaw0),
                }
            )
        self._publish_path(path)
        self.get_logger().info(f"Move base goal: ({x1:.3f}, {y1:.3f})")

    def control_loop(self):
        if self.stage == NavStage.IDLE:
            self.publish_twist(0.0, 0.0, 0.0)
            return

        pose = self._lookup_pose()
        if pose is None:
            return
        x, y, yaw = pose

        if self.stage == NavStage.TURN_IN_PLACE:
            self._control_turn_in_place(x, y, yaw)
        elif self.stage == NavStage.FOLLOW_TRAJECTORY:
            self._control_follow_trajectory(x, y, yaw)
        elif self.stage in (NavStage.MOVE_ROTATE, NavStage.MOVE_GO):
            self._control_move_base(x, y, yaw)

    def _control_turn_in_place(self, x, y, yaw):
        yaw_error = normalize_angle(self.turn_target_yaw - yaw)
        if abs(yaw_error) < self.yaw_tolerance:
            self.get_logger().info("Turn in place complete.")
            self.stop()
            self.publish_twist(0.0, 0.0, 0.0)
            return

        omega = max(-self.max_omega, min(self.max_omega, self.k_omega * yaw_error))
        self.publish_twist(0.0, 0.0, omega)

    def _control_move_base(self, x, y, yaw):
        dx = self.goal_x - x
        dy = self.goal_y - y
        dist = math.hypot(dx, dy)

        if dist < self.goal_tolerance_xy:
            self.get_logger().info("Goal reached.")
            self.stop()
            self.publish_twist(0.0, 0.0, 0.0)
            return

        v_mag = min(self.linear_speed, 0.6 * dist)

        c = math.cos(yaw)
        s = math.sin(yaw)
        vx_body = c * dx + s * dy
        vy_body = -s * dx + c * dy

        body_mag = math.hypot(vx_body, vy_body)
        if body_mag > 1e-6:
            vx_body = vx_body / body_mag * v_mag
            vy_body = vy_body / body_mag * v_mag

        self.publish_twist(vx_body, vy_body, 0.0)

    def _control_follow_trajectory(self, x, y, yaw):
        if not self.has_path or len(self.path) == 0:
            self.publish_twist(0.0, 0.0, 0.0)
            return

        speed = math.sqrt(self.odom.twist.twist.linear.x**2 + self.odom.twist.twist.linear.y**2)

        xs = np.array([p["x"] for p in self.path])
        ys = np.array([p["y"] for p in self.path])
        yaws = np.array([p["yaw"] for p in self.path])
        ks = np.array([p["k"] for p in self.path])

        goal_x = xs[-1]
        goal_y = ys[-1]
        goal_yaw = yaws[-1]

        dx = goal_x - x
        dy = goal_y - y
        dist_to_goal = math.hypot(dx, dy)
        yaw_error = normalize_angle(goal_yaw - yaw)

        if dist_to_goal < self.goal_tolerance_xy and abs(yaw_error) < self.goal_tolerance_yaw:
            self.get_logger().info("Trajectory goal reached!")
            self.stop()
            self.publish_twist(0.0, 0.0, 0.0)
            return

        dist2 = (xs - x) ** 2 + (ys - y) ** 2
        target_ind = int(np.argmin(dist2))

        v_for_lqr = max(abs(speed), 0.15)
        dl, target_ind, self.lqr_e, self.lqr_e_th = self._lqr_steering(x, y, yaw, v_for_lqr, xs, ys, yaws, ks)
        steer = max(-math.pi / 2, min(math.pi / 2, dl))
        steer *= self.path_dir

        v_cmd = min(self.target_speed, 0.6 * dist_to_goal)
        if abs(ks[target_ind]) > 1.0:
            v_cmd *= math.sqrt(1.0 / abs(ks[target_ind]))
        v_cmd = max(v_cmd, 0.05)
        v_cmd *= self.path_dir

        L_2 = self.L / 2.0
        omega = v_cmd * math.tan(steer) / L_2 if abs(L_2) > 1e-6 else 0.0

        vx_body = v_cmd
        vy_body = 0.0

        self.publish_twist(vx_body, vy_body, omega)

    def _lqr_steering(self, x, y, yaw, v, cx, cy, cyaw, ck):
        dx_arr = [x - icx for icx in cx]
        dy_arr = [y - icy for icy in cy]
        d = [idx**2 + idy**2 for (idx, idy) in zip(dx_arr, dy_arr)]
        ind = int(np.argmin(d))
        mind = math.sqrt(min(d))

        dxl = cx[ind] - x
        dyl = cy[ind] - y
        angle = normalize_angle(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        e = mind
        k = ck[ind]
        th_e = normalize_angle(yaw - cyaw[ind])

        L_2 = self.L / 2.0
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[0, 1] = self.dt
        A[1, 2] = v
        A[2, 2] = 1.0
        A[2, 3] = self.dt

        B = np.zeros((4, 1))
        B[3, 0] = v / L_2 if abs(L_2) > 1e-6 else 0.0

        Q_mat = np.eye(4)
        R_mat = np.eye(1)

        K = dlqr(A, B, Q_mat, R_mat)

        x_state = np.zeros((4, 1))
        x_state[0, 0] = e
        x_state[1, 0] = (e - self.lqr_e) / self.dt
        x_state[2, 0] = th_e
        x_state[3, 0] = (th_e - self.lqr_e_th) / self.dt

        ff = math.atan2(L_2 * k, 1)
        fb = normalize_angle((-K @ x_state)[0, 0])
        delta = ff + fb

        return delta, ind, e, th_e


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Guard against the ros2 launch SIGINT race (rcl_shutdown already called).
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
