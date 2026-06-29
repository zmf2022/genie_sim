#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
#
# ``lqr_steering_control`` and ``calc_nearest_index`` are adapted from
# Atsushi Sakai's PythonRobotics library (MIT-licensed; full text in
# ``../THIRD_PARTY_LICENSES.md``). Upstream reference:
#   https://github.com/AtsushiSakai/PythonRobotics
#   PathTracking/lqr_speed_steer_control/lqr_speed_steer_control.py

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

import math
import numpy as np

from tf2_ros import Buffer, TransformListener

from genie_sim_planning.math_utils import normalize_angle, dlqr
from rclpy.qos import qos_profile_system_default
from genie_sim_planning.tf_utils import lookup_pose
from genie_sim_planning.path_utils import publish_path
from genie_sim_planning.kinematics import (
    RateLimiter,
    FourWheelSteeringRobot,
)
from genie_sim_planning.planner import generate_clothoid

pi_2_pi = normalize_angle

L_2 = 0.46 / 2
_Q = np.eye(4)
_R = np.eye(1)
E_I = 0.0
K_I = 0.3


def lqr_steering_control(state, dt, cx, cy, cyaw, ck, pe, pth_e):
    global E_I, K_I

    ind, e = calc_nearest_index(state, cx, cy, cyaw)

    k = ck[ind]
    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    A = np.zeros((4, 4))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt

    B = np.zeros((4, 1))
    B[3, 0] = v / L_2

    K = dlqr(A, B, _Q, _R)

    x = np.zeros((4, 1))

    x[0, 0] = e
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt

    ff = math.atan2(L_2 * k, 1)
    fb = pi_2_pi((-K @ x)[0, 0])

    E_I += math.copysign(e, th_e) * dt

    delta = ff + fb - K_I * E_I

    return delta, ind, e, th_e


def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx**2 + idy**2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


class ClothoidFollower(Node):
    STEERING_MODE_ACKERMANN = 0
    STEERING_MODE_ACKERMANN_2 = 1

    def __init__(self):
        super().__init__("ackermann_clothoid_follower")

        # parameters
        self.declare_parameter("wheelbase", 0.46)
        self.declare_parameter("track_width", 0.436)
        self.declare_parameter("control_rate", 100.0)  # Hz
        self.declare_parameter("target_speed", 0.5)
        self.declare_parameter("goal_tolerance_xy", 0.05)
        self.declare_parameter("goal_tolerance_yaw", 0.10)
        self.declare_parameter("max_miss_distance", 0.5)

        # NEW: parameters for four‑wheel steering mode
        self.declare_parameter("use_4ws", "false")  # enable/disable 4WS mode
        self.declare_parameter("max_steer_angle_deg", "30")  # safety limit (degrees)

        # geometry
        self.L = float(self.get_parameter("wheelbase").value)  # wheel‑base (m)
        self.W = float(self.get_parameter("track_width").value)  # track width (m)

        # NEW: store steering‑limit in radians and the 4WS flag
        max_angle_deg = float(self.get_parameter("max_steer_angle_deg").value)
        self.max_angle_rad = math.radians(max_angle_deg)
        self.use_4ws = self.get_parameter("use_4ws").value.lower() == "true"

        # control
        self.dt = 1.0 / self.get_parameter("control_rate").value
        self.v_ff = float(self.get_parameter("target_speed").value)
        self.goal_tolerance_xy = self.get_parameter("goal_tolerance_xy").value
        self.goal_tolerance_yaw = self.get_parameter("goal_tolerance_yaw").value
        self.max_miss_distance = self.get_parameter("max_miss_distance").value

        # dynamic trajectory container
        self.path = []  # list[ dict(x, y, yaw, k, s) ]
        self.path_dir = 0.0  # +1 or -1 based on curvature direction
        self.has_path = False
        self.mode = self.STEERING_MODE_ACKERMANN

        # rate limiters
        self.v_limiter = RateLimiter(max_acc=0.1, max_jerk=4.0, dt=self.dt)
        self.delta_limiter = RateLimiter(max_acc=0.1, max_jerk=5.0, dt=self.dt)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subscribers
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 1)
        self.create_subscription(Odometry, "/odom", self.odom_cb, qos_profile_system_default)

        from nav_msgs.msg import Path

        self.path_pub = self.create_publisher(Path, "/my_path", 1)
        self.pub_joint_commmand = self.create_publisher(JointState, "/wheel_command", 1)
        self.pub_joint_commmand2 = self.create_publisher(JointState, "/joint_command_debug", 1)
        self.lqr_e, self.lqr_e_th = 0.0, 0.0

        # robot model (used only for the legacy Ackermann call)
        self.robot = FourWheelSteeringRobot(wheelbase=self.L, track_width=self.W)
        self.robot2 = FourWheelSteeringRobot(wheelbase=self.W, track_width=self.L)

        # timer loop
        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info("Ackermann clothoid-based follower online (waiting for /goal_pose)")

    # -------------------------------------------------------
    # 4WS helper functions (added)
    # -------------------------------------------------------
    def _allocate_4ws(self, v_cmd: float, kappa_cmd: float):
        """
        Convert a desired linear speed and curvature into per‑wheel speeds
        and steering angles. Returns a dict:
            {'fl':{speed, angle}, 'fr':..., 'rl':..., 'rr':...}
        The implementation follows the geometric derivation described in the
        design notes; it also enforces the configured steering limit.
        """
        # safety clamp on curvature based on max allowable steering angle
        if kappa_cmd != 0.0:
            kappa_max = math.tan(self.max_angle_rad) / self.L
            kappa_clamped = min(kappa_cmd, kappa_max)
        else:
            kappa_clamped = 0.0

        L, W = self.L, self.W
        # radius from the robot centre to each wheel (approx. equal for all)
        offset_radius = math.sqrt((L / 2) ** 2 + (W / 2) ** 2)

        # yaw rate that would produce curvature kappa_clamped
        omega = v_cmd / max(1e-6, (1.0 / kappa_clamped if kappa_clamped != 0 else float("inf")))
        base_speed = omega * offset_radius  # speed of each wheel due to turning

        # front steering angle via Ackermann geometry; rear is fractionally steered
        delta_front = math.copysign(math.atan(self.L * kappa_clamped), kappa_clamped)
        delta_rear = 0.5 * delta_front  # can be tuned if you want more rear steer

        results = {
            "fl": {"speed": base_speed, "angle": delta_front},
            "fr": {"speed": base_speed, "angle": delta_front},
            "rl": {"speed": -base_speed, "angle": delta_rear},
            "rr": {"speed": -base_speed, "angle": delta_rear},
        }

        # enforce absolute angle limits – if a wheel tries to exceed the limit,
        # scale down its angle (and thus overall curvature) until all are safe.
        for w in results:
            while abs(results[w]["angle"]) > self.max_angle_rad:
                scale = self.max_angle_rad / abs(results[w]["angle"])
                results[w]["angle"] *= scale
                kappa_clamped *= scale  # keep overall curvature consistent after scaling

        return results

    def _allocate_ackermann(self, v_cmd: float, curvature: float):
        """
        Legacy Ackermann allocator – unchanged from the original node.
        Returns a single steering angle that will be broadcast to opposite
        sides of the chassis.  Used only when ``use_4ws`` is False.
        """
        max_angle_rad = self.max_angle_rad
        delta_front = math.copysign(min(abs(math.atan(self.L * curvature)), max_angle_rad), curvature)
        return delta_front

    # -------------------------------------------------------
    # Pose lookup and path visualisation (unchanged)
    # -------------------------------------------------------
    def _lookup_pose(self):
        result = lookup_pose(self.tf_buffer, "base_link")
        if result is None:
            return None
        x, y, yaw = result
        if self.STEERING_MODE_ACKERMANN_2 == self.mode:
            return x, y, normalize_angle(yaw + math.pi / 2)
        return x, y, yaw

    def _publish_path(self, path_points, frame_id="map"):
        publish_path(self.path_pub, path_points, frame_id, stamp=self.get_clock().now().to_msg())

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def goal_cb(self, msg: PoseStamped):
        global E_I
        E_I = 0.0

        z1 = msg.pose.position.z
        if -0.1 <= z1 and z1 <= 0.1:
            self.get_logger().info("STEERING_MODE_ACKERMANN")
            self.mode = self.STEERING_MODE_ACKERMANN
        elif z1 > 0.1:
            self.get_logger().info("STEERING_MODE_ACKERMANN CRABBING")
            self.mode = self.STEERING_MODE_ACKERMANN_2
        elif z1 < -0.1:
            self.get_logger().info("STEERING UNCHANGED")

        pose = self._lookup_pose()
        if pose is None:
            self.get_logger().warn("Goal received but no robot pose available")
            return

        x0, y0, yaw0 = pose

        x1 = msg.pose.position.x
        y1 = msg.pose.position.y

        # extract yaw from quaternion in the goal message
        q = msg.pose.orientation
        siny = 2 * (q.w * q.z + q.x * q.y)
        cosy = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw1 = np.arctan2(siny, cosy)
        if self.STEERING_MODE_ACKERMANN_2 == self.mode:
            yaw1 = pi_2_pi(yaw1 + math.pi / 2)

        # ---- generate new clothoid ----
        path, (S, X, Y, YAW, KAPPA), direction = generate_clothoid(x0, y0, yaw0, 0.0, x1, y1, yaw1, 0.0, n_samples=100)

        # ---- convert to trajectory dictionary ----
        path = []
        for x, y, yaw, k, s in zip(X, Y, YAW, KAPPA, S):
            path.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "z": k,
                    "yaw": float(yaw),
                    "k": float(k),
                    "s": float(s),
                }
            )

        self._publish_path(path)

        self.path = path
        self.path_dir = 0
        if direction > 0:
            self.path_dir = 1.0
        elif direction < 0:
            self.path_dir = -1.0
        self.has_path = True

        self.get_logger().info(f"Received new goal, generated clothoid path with {len(path)} points")

    # -------------------------------------------------------
    # Main control loop (modified for four‑wheel mode switch)
    # -------------------------------------------------------
    def control_loop(self):
        """
        This is executed at 100 Hz. It:
          * reads the current odometry,
          * computes a desired curvature & speed via LQR + preview,
          * converts that into wheel commands (either Ackermann‑only or
            full four‑wheel steering, depending on ``use_4ws``),
          * publishes joint commands and a debug JointState.
        """
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = [
            "idx111_chassis_lwheel_front_joint1",
            "idx112_chassis_lwheel_front_joint2",
            "idx131_chassis_rwheel_front_joint1",
            "idx132_chassis_rwheel_front_joint2",
            "idx121_chassis_lwheel_rear_joint1",
            "idx122_chassis_lwheel_rear_joint2",
            "idx141_chassis_rwheel_rear_joint1",
            "idx142_chassis_rwheel_rear_joint2",
        ]
        cmd.position = [0.0] * 8
        cmd.velocity = [0.0] * 8

        # idle if we have not received a trajectory yet
        if not self.has_path or len(self.path) == 0:
            return

        pose = self._lookup_pose()
        if pose is None:
            return

        x, y, yaw = pose
        # optional flip when running in reverse direction
        # if self.path_dir == -1:
        #     yaw = pi_2_pi(yaw + math.pi)

        if self.odom.header == Header():
            self.get_logger().warn_throttle("invalid ODOM!")
        speed = math.sqrt(self.odom.twist.twist.linear.x**2 + self.odom.twist.twist.linear.y**2)

        goal_x = self.path[-1]["x"]
        goal_y = self.path[-1]["y"]
        dist_to_goal = math.hypot(goal_x - x, goal_y - y)
        start_x = self.path[-1]["x"]
        start_y = self.path[-1]["y"]
        dist_from_start = max(0.1, math.hypot(start_x - x, start_y - y))

        # nearest‑point index & error calculations
        xs = np.array([p["x"] for p in self.path])
        ys = np.array([p["y"] for p in self.path])
        yaws = np.array([p["yaw"] for p in self.path])
        ks = np.array([p["k"] for p in self.path])
        dist2 = (xs - x) ** 2 + (ys - y) ** 2
        idx = np.argmin(dist2)

        # target goal pose
        goal_x = xs[-1]
        goal_y = ys[-1]
        goal_yaw = yaws[-1]

        # error vectors
        dx = goal_x - x
        dy = goal_y - y
        dist_to_goal = math.hypot(dx, dy)
        yaw_error = normalize_angle(goal_yaw - yaw)
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # check for goal completion
        if dist_to_goal < self.goal_tolerance_xy and abs(yaw_error) < self.goal_tolerance_yaw:
            self.get_logger().warn("Robot reached the goal, stopping!")
            self.has_path = False
            self.path = []
            return

        # LQR‑based steering control (returns curvature dl)
        dl, target_ind, self.lqr_e, self.lqr_e_th = lqr_steering_control(
            State(
                x=x,
                y=y,
                yaw=yaw,
                v=abs(speed),
            ),
            self.dt,
            xs,
            ys,
            yaws,
            ks,
            self.lqr_e,
            self.lqr_e_th,
        )
        target_delta = min(dl, math.pi / 2)
        target_delta = max(target_delta, -math.pi / 2)

        steer_bicycle = target_delta
        v_cmd = 1.0 * min(0.6, dist_to_goal / 1.0)
        if abs(ks[target_ind]) > 1.0:
            v_cmd *= math.sqrt(1 / abs(ks[target_ind]))
        v_cmd = 1.0 * min(0.3, dist_from_start / 1.0)
        v_cmd *= self.path_dir
        steer_bicycle *= self.path_dir

        # ----------- FOUR‑WHEEL vs ACKERMANN SWITCH --------------
        if self.use_4ws:
            whl_vs, whl_as = self.robot.inverse_kinematics(v_cmd, steer_bicycle)
            alloc = self._allocate_4ws(v_cmd, steer_bicycle)
            from pprint import pprint

            pprint(alloc)
            # map allocated angles onto the position slots used by the legacy node
            cmd.position[0] = alloc["fl"]["angle"]
            cmd.position[2] = alloc["fr"]["angle"]
            cmd.position[4] = alloc["rl"]["angle"]
            cmd.position[6] = alloc["rr"]["angle"]

            wheel_radius = 0.07  # keep the same conversion constant you used before
            cmd.velocity[1] = alloc["fl"]["speed"] / wheel_radius
            cmd.velocity[3] = alloc["fr"]["speed"] / wheel_radius
            cmd.velocity[5] = alloc["rl"]["speed"] / wheel_radius
            cmd.velocity[7] = alloc["rr"]["speed"] / wheel_radius
        else:
            if self.STEERING_MODE_ACKERMANN == self.mode:
                whl_vs, whl_as = self.robot.inverse_kinematics(v_cmd, steer_bicycle)
                # **Legacy Ackermann path – unchanged from the original implementation**
                legacy_delta = self._allocate_ackermann(v_cmd, target_delta)
                cmd.position[0] = whl_as[0]  # front‑left joint command
                cmd.position[2] = whl_as[1]  # mirrored opposite side
                cmd.position[4] = whl_as[2]  # rear‑left (if your robot needs it)
                cmd.position[6] = whl_as[3]  # rear‑right mirror

                # the original code used whl_vs to fill velocities; keep that for safety
                wheel_radius = 0.07
                cmd.velocity[1] = whl_vs[0] / wheel_radius
                cmd.velocity[3] = whl_vs[1] / wheel_radius
                cmd.velocity[5] = whl_vs[2] / wheel_radius
                cmd.velocity[7] = whl_vs[3] / wheel_radius
            elif self.STEERING_MODE_ACKERMANN_2 == self.mode:
                whl_vs, whl_as = self.robot2.inverse_kinematics(v_cmd, steer_bicycle)
                # **Legacy Ackermann path – unchanged from the original implementation**
                legacy_delta = self._allocate_ackermann(v_cmd, target_delta)
                cmd.position[0] = whl_as[1] - math.pi / 2  # front‑left joint command
                cmd.position[2] = whl_as[3] - math.pi / 2  # mirrored opposite side
                cmd.position[4] = whl_as[0] - math.pi / 2  # rear‑left (if your robot needs it)
                cmd.position[6] = whl_as[2] - math.pi / 2  # rear‑right mirror

                # the original code used whl_vs to fill velocities; keep that for safety
                wheel_radius = 0.07
                cmd.velocity[1] = -whl_vs[1] / wheel_radius
                cmd.velocity[3] = -whl_vs[3] / wheel_radius
                cmd.velocity[5] = -whl_vs[0] / wheel_radius
                cmd.velocity[7] = -whl_vs[2] / wheel_radius

        # publish the main command (8‑element JointState)
        self.pub_joint_commmand.publish(cmd)

        # also publish a minimal debug message containing just speed & steer
        cmd_debug = JointState()
        cmd_debug.header.stamp = cmd.header.stamp
        cmd_debug.position = [v_cmd, steer_bicycle]

        self.pub_joint_commmand2.publish(cmd_debug)


# -------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ClothoidFollower()
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
