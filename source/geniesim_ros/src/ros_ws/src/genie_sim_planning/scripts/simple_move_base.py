#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import math
from enum import Enum
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from tf2_ros import Buffer, TransformListener

from genie_sim_planning.math_utils import normalize_angle, angle_distance
from rclpy.qos import qos_profile_system_default
from genie_sim_planning.tf_utils import (
    lookup_pose,
    pose_stamped_in_map,
    point_stamped_in_map,
)
from genie_sim_planning.path_utils import publish_path
from genie_sim_planning.kinematics import RateLimiter


class MoveStage(Enum):
    IDLE = 0
    ROTATE = 1
    GO = 2
    TURN_STEER = 3
    TURN_ROTATE = 4
    TURN_RESTORE = 5


class MoveBase(Node):
    """
    Simple two‑stage move_base‑like node:
      1) ROTATE  – turn robot towards target direction.
      2) GO      – drive straight towards target at configurable speed.
    """

    def __init__(self):
        super().__init__("move_base")

        # parameters
        self.declare_parameter("wheelbase", 0.46)
        self.declare_parameter("track_width", 0.436)
        self.declare_parameter("wheel_radius", 0.07)
        self.declare_parameter("control_rate", 100.0)  # Hz

        self.declare_parameter("linear_speed", 1.0)  # m/s, speed during go stage

        self.declare_parameter("goal_tolerance_xy", 0.05)
        self.declare_parameter("yaw_tolerance", 0.01)  # rad
        self.declare_parameter("k_omega", 5.0)  # yaw_error -> omega_cmd
        self.declare_parameter("max_omega", 0.8)  # rad/s

        # geometry
        self.L = float(self.get_parameter("wheelbase").value)
        self.W = float(self.get_parameter("track_width").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)

        # control
        self.dt = 1.0 / float(self.get_parameter("control_rate").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.goal_tolerance_xy = float(self.get_parameter("goal_tolerance_xy").value)
        self.yaw_tolerance = float(self.get_parameter("yaw_tolerance").value)
        self.k_omega = float(self.get_parameter("k_omega").value)
        self.max_omega = float(self.get_parameter("max_omega").value)

        # steering control to point wheels at target direction (no base yaw change)
        self.max_steer = math.radians(80.0)
        self.steer_tolerance = math.radians(3.0)
        # speed gain to ramp up/down with distance
        self.speed_gain = 0.6
        # smoothed steering target (rate-limited before delta_limiter) to avoid jerky delta_cmd
        self._last_desired_steer = 0.0

        # state
        self.stage = MoveStage.IDLE
        self.has_goal = False
        self.goal_x = 0.0
        self.goal_y = 0.0
        self._last_goal_dist = None  # track distance trend to avoid overshoot/backtracking
        # face clicked point: rotate base to face the point (yaw = direction from base to target)
        self.has_face_goal = False
        self.face_point_x = 0.0
        self.face_point_y = 0.0

        # turn in space state (three-stage: STEER -> ROTATE -> RESTORE)
        self.turn_in_space_target_yaw = 0.0

        # per-wheel smoothing + last-angle cache for minimal-angle policy
        self._wheels = ("fl", "fr", "rl", "rr")
        self._wheel_xy = {
            "fl": (self.L / 2.0, self.W / 2.0),
            "fr": (self.L / 2.0, -self.W / 2.0),
            "rl": (-self.L / 2.0, self.W / 2.0),
            "rr": (-self.L / 2.0, -self.W / 2.0),
        }
        self._last_steer_cmd = {w: 0.0 for w in self._wheels}
        self._steer_limiters = {w: RateLimiter(max_acc=3.0, max_jerk=40.0, dt=self.dt) for w in self._wheels}
        self._wheel_omega_limiters = {w: RateLimiter(max_acc=20.0, max_jerk=200.0, dt=self.dt) for w in self._wheels}

        # tuned for smoother start/stop
        self.v_limiter = RateLimiter(max_acc=0.3, max_jerk=1.5, dt=self.dt)
        # steering: faster response, smooth ramp, less overshoot (higher acc/jerk but slowdown near target)
        self.delta_limiter = RateLimiter(max_acc=2.0, max_jerk=20.0, dt=self.dt)

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # subs / pubs
        self.create_subscription(PoseStamped, "/goal_pose", self.goal_cb, 1)
        self.create_subscription(PointStamped, "/clicked_point", self.clicked_point_cb, 1)
        self.create_subscription(Odometry, "/odom", self.odom_cb, qos_profile_system_default)

        self.pub_joint_command = self.create_publisher(JointState, "/wheel_command", 1)
        self.pub_joint_debug = self.create_publisher(JointState, "/joint_command_debug", 1)
        from nav_msgs.msg import Path

        self.path_pub = self.create_publisher(Path, "/my_path", 1)

        self.odom = Odometry()

        # timer loop
        self.timer = self.create_timer(self.dt, self.control_loop)

        # log throttling (clock-aware)
        self._throttle_last_ns = {}

        self.get_logger().info("MoveBase node online (two‑stage: rotate then go).")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _warn_throttle(self, key: str, period_sec: float, message: str):
        now_ns = int(self.get_clock().now().nanoseconds)
        last_ns = self._throttle_last_ns.get(key)
        period_ns = int(period_sec * 1e9)
        if last_ns is None or (now_ns - last_ns) >= period_ns:
            self._throttle_last_ns[key] = now_ns
            self.get_logger().warn(message)

    def _lookup_pose(self):
        return lookup_pose(self.tf_buffer, "base_link")

    def _point_in_map(self, msg):
        return point_stamped_in_map(self.tf_buffer, msg, warn_fn=lambda m: self._warn_throttle("tf_point", 2.0, m))

    def _pose_in_map(self, msg):
        result = pose_stamped_in_map(self.tf_buffer, msg, warn_fn=lambda m: self._warn_throttle("tf_goal", 2.0, m))
        if result is None:
            return None
        return result[0], result[1]

    def _publish_path(self, path_points, frame_id="map"):
        publish_path(self.path_pub, path_points, frame_id, stamp=self.get_clock().now().to_msg())

    def _choose_min_angle(self, wheel: str, theta_raw: float):
        """
        For a steering joint, theta and theta+pi are equivalent if wheel speed is flipped.
        Pick the option closer to last commanded steering angle for smoothness.
        Returns (theta_cmd, flip_speed_sign).
        """
        last = self._last_steer_cmd[wheel]
        cand1 = normalize_angle(theta_raw)
        cand2 = normalize_angle(theta_raw + math.pi)
        if angle_distance(cand2, last) < angle_distance(cand1, last):
            return cand2, True
        return cand1, False

    def _allocate_turn_in_place(self, omega_cmd: float):
        """
        Direct per-wheel command for near turn-in-place.
        Uses v = omega x r in base frame. Returns dict wheel -> (steer_angle, wheel_omega_rad_s).
        """
        alloc = {}

        if abs(omega_cmd) < 1e-6:
            for w in self._wheels:
                alloc[w] = (self._last_steer_cmd[w], 0.0)
            return alloc

        for w in self._wheels:
            x_i, y_i = self._wheel_xy[w]
            v_x = -omega_cmd * y_i
            v_y = omega_cmd * x_i
            theta_raw = math.atan2(v_y, v_x)
            speed_raw = math.hypot(v_x, v_y)  # always positive

            theta_target, flip = self._choose_min_angle(w, theta_raw)
            theta_target = max(-self.max_steer, min(self.max_steer, theta_target))
            speed = -speed_raw if flip else speed_raw
            wheel_omega = speed / self.wheel_radius
            alloc[w] = (theta_target, wheel_omega)

        return alloc

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def clicked_point_cb(self, msg: PointStamped):
        """On clicked point: three-stage turn in space (STEER -> ROTATE -> RESTORE)."""
        pose = self._lookup_pose()
        if pose is None:
            return
        x, y, yaw = pose
        pt = self._point_in_map(msg)
        if pt is None:
            return
        self.face_point_x, self.face_point_y = pt[0], pt[1]
        dx_f = self.face_point_x - x
        dy_f = self.face_point_y - y
        if math.hypot(dx_f, dy_f) < 1e-3:
            self.get_logger().info("Face-point target is too close; ignoring.")
            return
        self.turn_in_space_target_yaw = math.atan2(dy_f, dx_f)
        self.has_face_goal = True
        self.stage = MoveStage.TURN_STEER
        self.get_logger().info(
            f"Turn in space: target yaw = {math.degrees(self.turn_in_space_target_yaw):.1f} deg, starting STEER stage."
        )

    def goal_cb(self, msg: PoseStamped):
        pose = self._lookup_pose()
        if pose is None:
            self.get_logger().warn("Goal received but no robot pose available, ignoring.")
            return

        x0, y0, yaw0 = pose
        goal_frame = msg.header.frame_id or "map"

        goal = self._pose_in_map(msg)
        if goal is None:
            self.get_logger().warn("Cannot transform goal to map frame, ignoring.")
            return
        self.goal_x, self.goal_y = goal
        self.has_goal = True
        self.stage = MoveStage.ROTATE

        path = []
        num_samples = 50
        for i in range(num_samples + 1):
            s = i / float(num_samples)
            path.append(
                {
                    "x": x0 + s * (self.goal_x - x0),
                    "y": y0 + s * (self.goal_y - y0),
                    "z": 0.0,
                    "yaw": float(yaw0),
                }
            )
        self._publish_path(path)

        self.get_logger().info(
            f"New goal received: ({self.goal_x:.3f}, {self.goal_y:.3f}), ROTATE wheels first, then GO."
        )

    # ------------------------------------------------------------------
    # main control loop: rotate then go
    # ------------------------------------------------------------------
    def control_loop(self):
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

        pose = self._lookup_pose()
        if pose is None:
            return

        x, y, yaw = pose

        # -------- Turn in space: three-stage (STEER -> ROTATE -> RESTORE) --------
        if self.has_face_goal:
            steer_index = {"fl": 0, "fr": 2, "rl": 4, "rr": 6}
            drive_index = {"fl": 1, "fr": 3, "rl": 5, "rr": 7}

            yaw_error = normalize_angle(self.turn_in_space_target_yaw - yaw)

            if self.stage == MoveStage.TURN_STEER:
                steer_direction = 1.0 if yaw_error >= 0 else -1.0
                alloc = self._allocate_turn_in_place(steer_direction * self.max_omega)

                max_steer_error = 0.0
                for w in self._wheels:
                    theta_target, _ = alloc[w]
                    theta_cmd = self._steer_limiters[w].step(theta_target)
                    self._last_steer_cmd[w] = theta_cmd

                    cmd.position[steer_index[w]] = theta_cmd
                    cmd.velocity[drive_index[w]] = 0.0

                    steer_error = abs(normalize_angle(theta_cmd - theta_target))
                    max_steer_error = max(max_steer_error, steer_error)

                self.pub_joint_command.publish(cmd)

                debug = JointState()
                debug.header.stamp = cmd.header.stamp
                debug.position = [float(max_steer_error), 0.0, 0.0]
                self.pub_joint_debug.publish(debug)

                if max_steer_error < self.steer_tolerance:
                    self.stage = MoveStage.TURN_ROTATE
                    self.get_logger().info("Steering aligned, switching to ROTATE stage.")

                return

            elif self.stage == MoveStage.TURN_ROTATE:
                omega_target = max(-self.max_omega, min(self.max_omega, self.k_omega * yaw_error))
                if abs(yaw_error) < self.yaw_tolerance:
                    omega_target = 0.0

                alloc = self._allocate_turn_in_place(omega_target)

                max_wheel_omega_abs = 0.0
                for w in self._wheels:
                    theta_target, wheel_omega_target = alloc[w]
                    theta_cmd = self._last_steer_cmd[w]
                    wheel_omega_cmd = self._wheel_omega_limiters[w].step(wheel_omega_target)

                    cmd.position[steer_index[w]] = theta_cmd
                    cmd.velocity[drive_index[w]] = wheel_omega_cmd
                    max_wheel_omega_abs = max(max_wheel_omega_abs, abs(wheel_omega_cmd))

                self.pub_joint_command.publish(cmd)

                debug = JointState()
                debug.header.stamp = cmd.header.stamp
                debug.position = [float(yaw_error), float(omega_target), float(max_wheel_omega_abs)]
                self.pub_joint_debug.publish(debug)

                if omega_target == 0.0 and max_wheel_omega_abs < 0.1:
                    self.stage = MoveStage.TURN_RESTORE
                    self.get_logger().info("Rotation complete, switching to RESTORE stage.")

                return

            elif self.stage == MoveStage.TURN_RESTORE:
                max_steer_error = 0.0
                for w in self._wheels:
                    theta_cmd = self._steer_limiters[w].step(0.0)
                    self._last_steer_cmd[w] = theta_cmd

                    cmd.position[steer_index[w]] = theta_cmd
                    cmd.velocity[drive_index[w]] = 0.0

                    max_steer_error = max(max_steer_error, abs(theta_cmd))

                self.pub_joint_command.publish(cmd)

                debug = JointState()
                debug.header.stamp = cmd.header.stamp
                debug.position = [float(max_steer_error), 0.0, 0.0]
                self.pub_joint_debug.publish(debug)

                if max_steer_error < self.steer_tolerance:
                    self.has_face_goal = False
                    self.stage = MoveStage.IDLE
                    self.get_logger().info("Turn in space complete: steering restored to default.")

                return

        # defaults when no goal
        v_target = 0.0
        delta_target = 0.0
        desired_steer = 0.0
        dist = 0.0
        motion_sign = 1.0  # +1 forward, -1 backward

        if self.has_goal:
            # compute goal vector and desired wheel direction (base yaw should stay unchanged)
            dx = self.goal_x - x
            dy = self.goal_y - y
            dist = math.hypot(dx, dy)
            target_yaw = math.atan2(dy, dx)

            # choose forward/backward so that steering angle is minimal
            steer_fwd = normalize_angle(target_yaw - yaw)
            steer_back = normalize_angle(target_yaw + math.pi - yaw)
            if abs(steer_fwd) <= abs(steer_back):
                desired_steer = steer_fwd
                motion_sign = 1.0
            else:
                desired_steer = steer_back
                motion_sign = -1.0

            # check goal reached (position + low speed)
            if dist < self.goal_tolerance_xy and abs(self.v_limiter.prev) < 0.02:
                self.get_logger().info("Goal reached smoothly, stopping and clearing goal.")
                self.stage = MoveStage.IDLE
                self.has_goal = False

        # simple stage management
        if self.stage == MoveStage.IDLE and self.has_goal:
            self.stage = MoveStage.ROTATE

        # choose target speed and steering based on stage (when a goal is active)
        if self.has_goal:
            if self.stage == MoveStage.ROTATE:
                # only rotate wheel steering joints towards desired direction, keep base static
                v_target = 0.0
                delta_target = desired_steer
            elif self.stage == MoveStage.GO:
                # move along steering direction with configurable speed
                # use current speed to compute stopping distance and start braking in time
                brake_margin = 0.05  # m
                a_brake = 0.25  # m/s^2, conservative decel to start braking earlier
                v_curr = abs(self.v_limiter.prev)
                stop_dist = v_curr * v_curr / (2.0 * a_brake) if a_brake > 1e-6 else 0.0

                if dist <= stop_dist + brake_margin:
                    # too close to stop safely at higher speed: command stop and let limiter brake
                    v_target_mag = 0.0
                else:
                    # far enough: accelerate/ramp up but respect distance
                    v_target_mag = min(self.linear_speed, self.speed_gain * dist)

                # near the goal, force a lower-speed approach for accuracy
                if dist < 0.8:
                    v_target_mag = min(v_target_mag, 0.30)

                v_target = motion_sign * v_target_mag
                delta_target = desired_steer
            else:
                v_target = 0.0
                delta_target = 0.0
        else:
            # no goal: smoothly restore steering wheels back to default 0.0
            v_target = 0.0
            delta_target = 0.0

        # rate-limit steering target: faster overall, smooth, slowdown near target to reduce overshoot
        max_steer_rate = math.radians(120.0)  # rad/s, faster steering
        steer_slowdown_rad = 0.2  # rad; below this error, scale down rate to approach gently
        steer_err = normalize_angle(delta_target - self._last_desired_steer)
        max_step = max_steer_rate * self.dt
        if abs(steer_err) < steer_slowdown_rad and abs(steer_err) > 1e-6:
            max_step = max_step * (abs(steer_err) / steer_slowdown_rad)
        steer_err = max(-max_step, min(max_step, steer_err))
        delta_target = normalize_angle(self._last_desired_steer + steer_err)
        self._last_desired_steer = delta_target

        # clamp steering
        delta_target = max(-self.max_steer, min(self.max_steer, delta_target))

        # rate‑limited commands for smooth motion
        v_cmd = self.v_limiter.step(v_target)
        delta_cmd = self.delta_limiter.step(delta_target)

        # stage transition: when wheels are aligned with desired direction, start GO
        if self.has_goal and self.stage == MoveStage.ROTATE:
            if abs(normalize_angle(delta_cmd - desired_steer)) < self.steer_tolerance:
                self.stage = MoveStage.GO
                self.get_logger().info("Wheel steering aligned, switching to GO stage.")

        # steering joints – all wheels point the same direction (crab motion)
        cmd.position[0] = delta_cmd  # front left
        cmd.position[2] = delta_cmd  # front right
        cmd.position[4] = delta_cmd  # rear left
        cmd.position[6] = delta_cmd  # rear right

        # wheel rotational velocities – equal speeds for pure translation
        cmd.velocity[1] = v_cmd / self.wheel_radius
        cmd.velocity[3] = v_cmd / self.wheel_radius
        cmd.velocity[5] = v_cmd / self.wheel_radius
        cmd.velocity[7] = v_cmd / self.wheel_radius
        # verbose steering debug for analyzing jerky delta_cmd
        # self.get_logger().info(
        #     f"[move_base] stage={self.stage.name} has_goal={self.has_goal} "
        #     f"dist={dist:.3f} v_tgt={v_target:.3f} v_cmd={v_cmd:.3f} "
        #     f"steer_des={desired_steer:.3f} delta_tgt={delta_target:.3f} delta_cmd={delta_cmd:.3f}"
        # )

        self.pub_joint_command.publish(cmd)

        # debug message: just v_cmd and delta_cmd
        debug = JointState()
        debug.header.stamp = cmd.header.stamp
        debug.position = [float(v_cmd), float(delta_cmd)]
        self.pub_joint_debug.publish(debug)


def main(args=None):
    rclpy.init(args=args)
    node = MoveBase()
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
