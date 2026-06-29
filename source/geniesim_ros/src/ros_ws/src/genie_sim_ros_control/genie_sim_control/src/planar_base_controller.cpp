// Copyright 2025 AgiBot
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#include "genie_sim_control/planar_base_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"

namespace genie_sim_control
{

namespace
{
double clamp(double v, double lo, double hi)
{
  return std::max(lo, std::min(v, hi));
}

double wrap_angle(double a)
{
  while (a > M_PI) {a -= 2.0 * M_PI;}
  while (a < -M_PI) {a += 2.0 * M_PI;}
  return a;
}

double yaw_from_quat(double qx, double qy, double qz, double qw)
{
  tf2::Quaternion q(qx, qy, qz, qw);
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return yaw;
}
}  // namespace

controller_interface::InterfaceConfiguration
PlanarBaseController::command_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::NONE, {}};
}

controller_interface::InterfaceConfiguration
PlanarBaseController::state_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::NONE, {}};
}

controller_interface::CallbackReturn PlanarBaseController::on_init()
{
  try {
    auto_declare<std::string>("joint_name", joint_name_);
    auto_declare<std::string>("cmd_twist_topic", cmd_twist_topic_);
    auto_declare<std::string>("odom_topic", odom_topic_);
    auto_declare<std::string>("map_frame", map_frame_);
    auto_declare<std::string>("base_frame", base_frame_);
    auto_declare<bool>("use_odom_state", use_odom_state_);
    auto_declare<double>("kp_xy", kp_xy_);
    auto_declare<double>("kp_yaw", kp_yaw_);
    auto_declare<double>("kd_xy", kd_xy_);
    auto_declare<double>("kd_yaw", kd_yaw_);
    auto_declare<double>("decel_window_xy", decel_window_xy_);
    auto_declare<double>("decel_window_yaw", decel_window_yaw_);
    auto_declare<double>("stop_band_v", stop_band_v_);
    auto_declare<double>("stop_band_w", stop_band_w_);
    auto_declare<double>("max_linear_speed", max_linear_speed_);
    auto_declare<double>("max_angular_speed", max_angular_speed_);
    auto_declare<double>("goal_tolerance_xy", goal_tolerance_xy_);
    auto_declare<double>("goal_tolerance_yaw", goal_tolerance_yaw_);
    auto_declare<double>("goal_time_tolerance", goal_time_tolerance_);
    auto_declare<double>("odom_timeout", odom_timeout_);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(
      rclcpp::get_logger("PlanarBaseController"),
      "Exception in on_init(): %s", e.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn PlanarBaseController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  auto node = get_node();

  joint_name_ = node->get_parameter("joint_name").as_string();
  cmd_twist_topic_ = node->get_parameter("cmd_twist_topic").as_string();
  odom_topic_ = node->get_parameter("odom_topic").as_string();
  map_frame_ = node->get_parameter("map_frame").as_string();
  base_frame_ = node->get_parameter("base_frame").as_string();
  use_odom_state_ = node->get_parameter("use_odom_state").as_bool();
  kp_xy_ = node->get_parameter("kp_xy").as_double();
  kp_yaw_ = node->get_parameter("kp_yaw").as_double();
  kd_xy_ = node->get_parameter("kd_xy").as_double();
  kd_yaw_ = node->get_parameter("kd_yaw").as_double();
  decel_window_xy_ = node->get_parameter("decel_window_xy").as_double();
  decel_window_yaw_ = node->get_parameter("decel_window_yaw").as_double();
  stop_band_v_ = node->get_parameter("stop_band_v").as_double();
  stop_band_w_ = node->get_parameter("stop_band_w").as_double();
  max_linear_speed_ = node->get_parameter("max_linear_speed").as_double();
  max_angular_speed_ = node->get_parameter("max_angular_speed").as_double();
  goal_tolerance_xy_ = node->get_parameter("goal_tolerance_xy").as_double();
  goal_tolerance_yaw_ = node->get_parameter("goal_tolerance_yaw").as_double();
  goal_time_tolerance_ = node->get_parameter("goal_time_tolerance").as_double();
  odom_timeout_ = node->get_parameter("odom_timeout").as_double();

  cmd_twist_pub_ = node->create_publisher<geometry_msgs::msg::Twist>(
    cmd_twist_topic_, rclcpp::QoS(1).reliable());

  if (use_odom_state_) {
    odom_sub_ = node->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PlanarBaseController::odom_callback, this, std::placeholders::_1));
  }

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  action_server_ = rclcpp_action::create_server<FollowJointTrajectory>(
    node, "~/follow_joint_trajectory",
    std::bind(
      &PlanarBaseController::handle_goal, this, std::placeholders::_1,
      std::placeholders::_2),
    std::bind(&PlanarBaseController::handle_cancel, this, std::placeholders::_1),
    std::bind(&PlanarBaseController::handle_accepted, this, std::placeholders::_1));

  RCLCPP_INFO(
    node->get_logger(),
    "PlanarBaseController configured: joint='%s', cmd_twist='%s', odom='%s' (active=%s), "
    "map='%s', base='%s', action='%s/follow_joint_trajectory'",
    joint_name_.c_str(), cmd_twist_topic_.c_str(), odom_topic_.c_str(),
    use_odom_state_ ? "yes" : "no (TF fallback)",
    map_frame_.c_str(), base_frame_.c_str(), node->get_name());

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn PlanarBaseController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  publish_zero_twist();
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn PlanarBaseController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  std::lock_guard<std::mutex> lk(goal_mutex_);
  if (current_goal_) {
    auto result = std::make_shared<FollowJointTrajectory::Result>();
    result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
    result->error_string = "Controller deactivated";
    current_goal_->abort(result);
    current_goal_.reset();
  }
  publish_zero_twist();
  return controller_interface::CallbackReturn::SUCCESS;
}

rclcpp_action::GoalResponse PlanarBaseController::handle_goal(
  const rclcpp_action::GoalUUID & /*uuid*/,
  std::shared_ptr<const FollowJointTrajectory::Goal> goal)
{
  // For an SRDF planar virtual_joint, MoveIt's TrajectoryExecutionManager
  // routes the trajectory into goal.multi_dof_trajectory (geometry_msgs/Transform
  // per waypoint), NOT goal.trajectory (single-DoF positions).
  // Empty goals are normal — moveit_simple_controller_manager forwards a stub
  // goal to every overlapping controller, even for arm-only plans.
  const auto & mdt = goal->multi_dof_trajectory;
  std::string joint_list;
  for (const auto & jn : mdt.joint_names) {
    if (!joint_list.empty()) {joint_list += ", ";}
    joint_list += jn;
  }
  RCLCPP_DEBUG(
    get_node()->get_logger(),
    "handle_goal: multi_dof joint_names=[%s] (%zu), %zu waypoints; single-dof %zu joints / %zu pts",
    joint_list.c_str(), mdt.joint_names.size(), mdt.points.size(),
    goal->trajectory.joint_names.size(), goal->trajectory.points.size());

  if (mdt.points.empty()) {
    RCLCPP_DEBUG(
      get_node()->get_logger(),
      "Accepting empty multi-dof trajectory (no-op for chassis); will succeed immediately");
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  bool found = false;
  for (const auto & jn : mdt.joint_names) {
    if (jn == joint_name_) {
      found = true;
      break;
    }
  }
  if (!found) {
    RCLCPP_WARN(
      get_node()->get_logger(),
      "Rejecting goal: multi_dof_trajectory does not contain joint '%s'",
      joint_name_.c_str());
    return rclcpp_action::GoalResponse::REJECT;
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse PlanarBaseController::handle_cancel(
  const std::shared_ptr<GoalHandle>/*goal_handle*/)
{
  return rclcpp_action::CancelResponse::ACCEPT;
}

void PlanarBaseController::handle_accepted(const std::shared_ptr<GoalHandle> goal_handle)
{
  const auto & mdt = goal_handle->get_goal()->multi_dof_trajectory;
  RCLCPP_INFO(
    get_node()->get_logger(),
    "[%s] Goal received: multi_dof joint_names.size=%zu, points.size=%zu",
    joint_name_.c_str(), mdt.joint_names.size(), mdt.points.size());
  for (size_t i = 0; i < mdt.joint_names.size(); ++i) {
    RCLCPP_INFO(
      get_node()->get_logger(), "  joint_names[%zu] = '%s'", i, mdt.joint_names[i].c_str());
  }
  if (!mdt.points.empty() && !mdt.points.front().transforms.empty()) {
    const auto & p0 = mdt.points.front();
    const auto & tf0 = p0.transforms.front();
    RCLCPP_INFO(
      get_node()->get_logger(),
      "  point[0]: transforms.size=%zu, t=%.3fs, "
      "tx=%.3f ty=%.3f tz=%.3f qz=%.3f qw=%.3f",
      p0.transforms.size(),
      rclcpp::Duration(p0.time_from_start).seconds(),
      tf0.translation.x, tf0.translation.y, tf0.translation.z,
      tf0.rotation.z, tf0.rotation.w);
  }

  if (mdt.points.empty()) {
    auto result = std::make_shared<FollowJointTrajectory::Result>();
    result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
    result->error_string = "Empty multi-dof trajectory (no-op for chassis)";
    goal_handle->succeed(result);
    RCLCPP_DEBUG(
      get_node()->get_logger(),
      "Empty trajectory accepted and succeeded immediately (no-op for chassis)");
    return;
  }

  std::lock_guard<std::mutex> lk(goal_mutex_);

  // Pre-empt any active goal.
  if (current_goal_) {
    auto result = std::make_shared<FollowJointTrajectory::Result>();
    result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
    result->error_string = "Pre-empted by new goal";
    current_goal_->abort(result);
  }

  current_goal_ = goal_handle;
  multi_dof_trajectory_ = mdt;
  trajectory_start_time_ = get_node()->now();

  // Resolve which entry in joint_names corresponds to our planar joint.
  planar_idx_ = -1;
  for (size_t i = 0; i < multi_dof_trajectory_.joint_names.size(); ++i) {
    if (multi_dof_trajectory_.joint_names[i] == joint_name_) {
      planar_idx_ = static_cast<int>(i);
      break;
    }
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Accepted multi-dof trajectory with %zu waypoints (planar joint '%s' at index %d of %zu)",
    multi_dof_trajectory_.points.size(), joint_name_.c_str(),
    planar_idx_, multi_dof_trajectory_.joint_names.size());
}

bool PlanarBaseController::lookup_base_pose(double & x, double & y, double & yaw)
{
  // Prefer the /odom subscription path: zero TF lookup latency, single hop,
  // direct from the simulator's odometry publisher. Fall back to TF only if
  // odom is disabled or stale.
  if (use_odom_state_) {
    std::lock_guard<std::mutex> lk(odom_mutex_);
    if (odom_received_) {
      const auto now = get_node()->now();
      const double age = (now - odom_stamp_).seconds();
      if (age < odom_timeout_) {
        x = odom_x_;
        y = odom_y_;
        yaw = odom_yaw_;
        return true;
      }
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 2000,
        "Odom stale (age=%.2fs > timeout=%.2fs); falling back to TF",
        age, odom_timeout_);
    } else {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 2000,
        "No odom received yet on '%s'; falling back to TF",
        odom_topic_.c_str());
    }
  }

  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = tf_buffer_->lookupTransform(
      map_frame_, base_frame_, tf2::TimePointZero,
      tf2::durationFromSec(0.05));
  } catch (const std::exception & e) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 2000,
      "TF lookup %s -> %s failed: %s",
      map_frame_.c_str(), base_frame_.c_str(), e.what());
    return false;
  }
  x = tf.transform.translation.x;
  y = tf.transform.translation.y;
  tf2::Quaternion q(
    tf.transform.rotation.x,
    tf.transform.rotation.y,
    tf.transform.rotation.z,
    tf.transform.rotation.w);
  double roll, pitch;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  return true;
}

void PlanarBaseController::odom_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lk(odom_mutex_);
  odom_x_ = msg->pose.pose.position.x;
  odom_y_ = msg->pose.pose.position.y;
  odom_yaw_ = yaw_from_quat(
    msg->pose.pose.orientation.x,
    msg->pose.pose.orientation.y,
    msg->pose.pose.orientation.z,
    msg->pose.pose.orientation.w);
  // nav_msgs/Odometry convention: twist is expressed in the *child* frame
  // (i.e. base_link). Rotate into the world frame so the D-term operates
  // in the same frame as the position error (ex, ey). Yaw rate is
  // frame-invariant for a planar base (z-axis is shared).
  const double c = std::cos(odom_yaw_);
  const double s = std::sin(odom_yaw_);
  const double vx_b = msg->twist.twist.linear.x;
  const double vy_b = msg->twist.twist.linear.y;
  odom_vx_w_ = c * vx_b - s * vy_b;
  odom_vy_w_ = s * vx_b + c * vy_b;
  odom_wz_ = msg->twist.twist.angular.z;
  odom_stamp_ = get_node()->now();
  odom_received_ = true;
}

bool PlanarBaseController::lookup_base_velocity(double & vx_w, double & vy_w, double & wz)
{
  std::lock_guard<std::mutex> lk(odom_mutex_);
  if (!odom_received_) {
    vx_w = vy_w = wz = 0.0;
    return false;
  }
  vx_w = odom_vx_w_;
  vy_w = odom_vy_w_;
  wz = odom_wz_;
  return true;
}

bool PlanarBaseController::sample_trajectory(
  double t_seconds, double & x, double & y, double & yaw) const
{
  if (multi_dof_trajectory_.points.empty() || planar_idx_ < 0) {return false;}
  const size_t pidx = static_cast<size_t>(planar_idx_);

  auto pos_at = [&](size_t pi, double & ox, double & oy, double & oyaw) {
      const auto & pt = multi_dof_trajectory_.points[pi];
      if (pidx >= pt.transforms.size()) {
        ox = oy = oyaw = 0.0;
        return;
      }
      const auto & tf = pt.transforms[pidx];
      ox = tf.translation.x;
      oy = tf.translation.y;
      oyaw = yaw_from_quat(tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w);
    };

  if (t_seconds <= 0.0) {
    pos_at(0, x, y, yaw);
    return true;
  }

  for (size_t i = 1; i < multi_dof_trajectory_.points.size(); ++i) {
    double t1 = rclcpp::Duration(multi_dof_trajectory_.points[i].time_from_start).seconds();
    if (t_seconds <= t1) {
      double t0 = rclcpp::Duration(multi_dof_trajectory_.points[i - 1].time_from_start).seconds();
      double dt = std::max(1e-6, t1 - t0);
      double a = (t_seconds - t0) / dt;
      double x0, y0, yaw0, x1, y1, yaw1;
      pos_at(i - 1, x0, y0, yaw0);
      pos_at(i, x1, y1, yaw1);
      x = x0 + a * (x1 - x0);
      y = y0 + a * (y1 - y0);
      yaw = yaw0 + a * wrap_angle(yaw1 - yaw0);
      return true;
    }
  }
  pos_at(multi_dof_trajectory_.points.size() - 1, x, y, yaw);
  return true;
}

bool PlanarBaseController::sample_trajectory_velocity(
  double t_seconds, double & vx, double & vy, double & wz) const
{
  // Feedforward velocity: prefer the explicit MultiDOFJointTrajectoryPoint::velocities
  // field populated by MoveIt's time-parameterization adapter (TOTG / Ruckig).
  // Fall back to a finite-difference of consecutive transforms if the producer
  // didn't fill velocities[].
  vx = vy = wz = 0.0;
  if (multi_dof_trajectory_.points.empty() || planar_idx_ < 0) {return false;}
  const size_t pidx = static_cast<size_t>(planar_idx_);
  const auto & points = multi_dof_trajectory_.points;

  // Past the end of the trajectory: zero feedforward (let feedback close the
  // residual error).
  const double t_end = rclcpp::Duration(points.back().time_from_start).seconds();
  if (t_seconds >= t_end) {return true;}

  // Locate the segment [i-1, i] containing t_seconds.
  size_t seg_i = 0;
  for (size_t i = 1; i < points.size(); ++i) {
    if (t_seconds <= rclcpp::Duration(points[i].time_from_start).seconds()) {
      seg_i = i;
      break;
    }
  }
  if (seg_i == 0) {seg_i = 1;}

  auto vel_at = [&](size_t pi, double & ovx, double & ovy, double & owz) -> bool {
      if (pi >= points.size()) {return false;}
      const auto & pt = points[pi];
      if (pidx >= pt.velocities.size()) {return false;}
      ovx = pt.velocities[pidx].linear.x;
      ovy = pt.velocities[pidx].linear.y;
      owz = pt.velocities[pidx].angular.z;
      return true;
    };

  double v0x, v0y, v0w, v1x, v1y, v1w;
  const bool have0 = vel_at(seg_i - 1, v0x, v0y, v0w);
  const bool have1 = vel_at(seg_i, v1x, v1y, v1w);
  const double t0 = rclcpp::Duration(points[seg_i - 1].time_from_start).seconds();
  const double t1 = rclcpp::Duration(points[seg_i].time_from_start).seconds();
  const double dt = std::max(1e-6, t1 - t0);
  const double a = clamp((t_seconds - t0) / dt, 0.0, 1.0);

  if (have0 && have1) {
    vx = v0x + a * (v1x - v0x);
    vy = v0y + a * (v1y - v0y);
    wz = v0w + a * (v1w - v0w);
    return true;
  }

  // Fall back to finite difference of poses across the segment.
  if (pidx < points[seg_i - 1].transforms.size() &&
    pidx < points[seg_i].transforms.size())
  {
    const auto & tf0 = points[seg_i - 1].transforms[pidx];
    const auto & tf1 = points[seg_i].transforms[pidx];
    vx = (tf1.translation.x - tf0.translation.x) / dt;
    vy = (tf1.translation.y - tf0.translation.y) / dt;
    const double y0 = yaw_from_quat(tf0.rotation.x, tf0.rotation.y, tf0.rotation.z, tf0.rotation.w);
    const double y1 = yaw_from_quat(tf1.rotation.x, tf1.rotation.y, tf1.rotation.z, tf1.rotation.w);
    wz = wrap_angle(y1 - y0) / dt;
    return true;
  }
  return false;
}

void PlanarBaseController::publish_zero_twist()
{
  if (!cmd_twist_pub_) {return;}
  geometry_msgs::msg::Twist t;
  cmd_twist_pub_->publish(t);
}

controller_interface::return_type PlanarBaseController::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  std::shared_ptr<GoalHandle> goal;
  double t_in_traj = 0.0;
  double traj_total = 0.0;
  bool finished = false;
  {
    std::lock_guard<std::mutex> lk(goal_mutex_);
    goal = current_goal_;
    if (goal) {
      t_in_traj = (time - trajectory_start_time_).seconds();
      if (!multi_dof_trajectory_.points.empty()) {
        traj_total =
          rclcpp::Duration(multi_dof_trajectory_.points.back().time_from_start).seconds();
      }
      if (t_in_traj > traj_total + goal_time_tolerance_) {
        finished = true;
      }
    }
  }

  if (!goal) {
    return controller_interface::return_type::OK;
  }

  double cx, cy, cyaw;
  if (!lookup_base_pose(cx, cy, cyaw)) {
    publish_zero_twist();
    return controller_interface::return_type::OK;
  }

  double dx, dy, dyaw;
  if (!sample_trajectory(t_in_traj, dx, dy, dyaw)) {
    publish_zero_twist();
    return controller_interface::return_type::OK;
  }

  double ex = dx - cx;
  double ey = dy - cy;
  double eyaw = wrap_angle(dyaw - cyaw);

  // Goal-completion check at/after the final waypoint:
  //   - if pose tolerance is met -> SUCCESSFUL (any time in
  //     [traj_total, traj_total + goal_time_tolerance]);
  //   - if t > traj_total + goal_time_tolerance still out of tolerance ->
  //     abort with GOAL_TOLERANCE_VIOLATED (matches joint_trajectory_controller).
  if (t_in_traj >= traj_total) {
    double fx, fy, fyaw;
    sample_trajectory(traj_total, fx, fy, fyaw);
    const double fex = fx - cx;
    const double fey = fy - cy;
    const double feyaw = wrap_angle(fyaw - cyaw);
    const bool xy_ok = std::hypot(fex, fey) < goal_tolerance_xy_;
    const bool yaw_ok = std::abs(feyaw) < goal_tolerance_yaw_;
    if (xy_ok && yaw_ok) {
      auto result = std::make_shared<FollowJointTrajectory::Result>();
      result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
      {
        std::lock_guard<std::mutex> lk(goal_mutex_);
        if (current_goal_) {
          current_goal_->succeed(result);
          current_goal_.reset();
        }
      }
      publish_zero_twist();
      return controller_interface::return_type::OK;
    } else if (finished) {
      auto result = std::make_shared<FollowJointTrajectory::Result>();
      result->error_code = FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED;
      result->error_string = "Goal time exceeded with pose tolerance not met "
        "(xy_err=" + std::to_string(std::hypot(fex, fey)) +
        " m, yaw_err=" + std::to_string(std::abs(feyaw)) + " rad)";
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Aborting goal: %s", result->error_string.c_str());
      {
        std::lock_guard<std::mutex> lk(goal_mutex_);
        if (current_goal_) {
          current_goal_->abort(result);
          current_goal_.reset();
        }
      }
      publish_zero_twist();
      return controller_interface::return_type::OK;
    }
    // else: within goal_time_tolerance window — keep driving with feedback only.
  }

  // Feedforward velocity from the trajectory (world frame), summed with PD
  // feedback. This matches the official mobile-base controller pattern: the
  // base tracks the planned velocity profile rather than perpetually trailing
  // a moving setpoint by `v_ref / kp`.
  //
  // The control law per axis is:
  //   v_cmd = clamp_window( v_ff + kp * pos_err - kd * v_meas, ... )
  // where:
  //   - v_ff       : trajectory-sampled feedforward velocity (zero past t_end)
  //   - kp * err   : P feedback on remaining position error
  //   - kd * v_meas: D damping on *measured* velocity (from /odom). This is
  //                  what kills the end-of-trajectory overshoot: as the base
  //                  approaches the goal with non-zero velocity, the D term
  //                  pulls v_cmd back toward zero even before err shrinks.
  //   - clamp_window: optional decel-window scaling that linearly tapers
  //                  the *magnitude* of v_cmd as |err| -> 0. Disabled when
  //                  decel_window_xy_ <= 0.
  // After the saturation clamp, a stop-band check forces an exact zero
  // twist when both |err| and |v_meas| are within their bands -- this
  // avoids streaming sub-cm twitches to the wheel servos.
  double vx_ff_w = 0.0, vy_ff_w = 0.0, wz_ff = 0.0;
  sample_trajectory_velocity(t_in_traj, vx_ff_w, vy_ff_w, wz_ff);

  double vx_m_w = 0.0, vy_m_w = 0.0, wz_m = 0.0;
  lookup_base_velocity(vx_m_w, vy_m_w, wz_m);

  double vx_world = vx_ff_w + kp_xy_ * ex - kd_xy_ * vx_m_w;
  double vy_world = vy_ff_w + kp_xy_ * ey - kd_xy_ * vy_m_w;
  double wz = wz_ff + kp_yaw_ * eyaw - kd_yaw_ * wz_m;

  // Decel-window: when within `decel_window_xy_` meters of the setpoint,
  // additionally clamp the linear command magnitude to a smooth ramp that
  // hits zero exactly at the setpoint. Same idea for yaw.
  if (decel_window_xy_ > 0.0) {
    const double err_xy = std::hypot(ex, ey);
    if (err_xy < decel_window_xy_) {
      const double scale = err_xy / decel_window_xy_;     // 0..1
      const double v_cap = scale * max_linear_speed_;
      const double v_mag = std::hypot(vx_world, vy_world);
      if (v_mag > v_cap && v_mag > 1e-9) {
        const double k = v_cap / v_mag;
        vx_world *= k;
        vy_world *= k;
      }
    }
  }
  if (decel_window_yaw_ > 0.0) {
    const double err_y = std::abs(eyaw);
    if (err_y < decel_window_yaw_) {
      const double scale = err_y / decel_window_yaw_;
      const double w_cap = scale * max_angular_speed_;
      if (std::abs(wz) > w_cap) {wz = std::copysign(w_cap, wz);}
    }
  }

  vx_world = clamp(vx_world, -max_linear_speed_, max_linear_speed_);
  vy_world = clamp(vy_world, -max_linear_speed_, max_linear_speed_);
  wz = clamp(wz, -max_angular_speed_, max_angular_speed_);

  // Stop-band: only engages past the end of the trajectory and when the base
  // is essentially at rest near the goal. Prevents the chassis driver from
  // amplifying sub-cm noise in /odom into wheel jitter.
  if (t_in_traj >= traj_total) {
    const double v_meas_mag = std::hypot(vx_m_w, vy_m_w);
    if (stop_band_v_ > 0.0 && v_meas_mag < stop_band_v_ &&
      std::hypot(ex, ey) < 0.5 * goal_tolerance_xy_)
    {
      vx_world = 0.0;
      vy_world = 0.0;
    }
    if (stop_band_w_ > 0.0 && std::abs(wz_m) < stop_band_w_ &&
      std::abs(eyaw) < 0.5 * goal_tolerance_yaw_)
    {
      wz = 0.0;
    }
  }

  double c = std::cos(cyaw);
  double s = std::sin(cyaw);
  double vx_body = c * vx_world + s * vy_world;
  double vy_body = -s * vx_world + c * vy_world;

  geometry_msgs::msg::Twist t_msg;
  t_msg.linear.x = vx_body;
  t_msg.linear.y = vy_body;
  t_msg.angular.z = wz;
  cmd_twist_pub_->publish(t_msg);

  return controller_interface::return_type::OK;
}

}  // namespace genie_sim_control

PLUGINLIB_EXPORT_CLASS(
  genie_sim_control::PlanarBaseController,
  controller_interface::ControllerInterface)
