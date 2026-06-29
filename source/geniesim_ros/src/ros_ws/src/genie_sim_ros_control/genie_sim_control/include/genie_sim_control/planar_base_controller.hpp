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

#ifndef GENIE_SIM_CONTROL__PLANAR_BASE_CONTROLLER_HPP_
#define GENIE_SIM_CONTROL__PLANAR_BASE_CONTROLLER_HPP_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "trajectory_msgs/msg/multi_dof_joint_trajectory.hpp"

namespace genie_sim_control
{

/// ros2_control controller plugin that bridges a MoveIt FollowJointTrajectory
/// action for an SRDF planar virtual_joint to a geometry_msgs/Twist topic
/// (e.g. /cmd_twist) consumed by the chassis driver.
///
/// The controller does not claim any command or state interfaces; it samples
/// the active trajectory in update() and uses TF (map -> base_link) for
/// closed-loop P feedback before publishing the Twist command.
class PlanarBaseController : public controller_interface::ControllerInterface
{
public:
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  using GoalHandle = rclcpp_action::ServerGoalHandle<FollowJointTrajectory>;

  PlanarBaseController() = default;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

private:
  // ----- action callbacks -----
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID & uuid,
    std::shared_ptr<const FollowJointTrajectory::Goal> goal);
  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandle> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle);

  // ----- helpers -----
  bool lookup_base_pose(double & x, double & y, double & yaw);
  // Returns the most recent world-frame measured base velocity. Returns false
  // if no /odom has arrived yet, in which case (vx, vy, wz) are set to 0.
  bool lookup_base_velocity(double & vx_w, double & vy_w, double & wz);
  bool sample_trajectory(double t_seconds, double & x, double & y, double & yaw) const;
  bool sample_trajectory_velocity(
    double t_seconds, double & vx, double & vy, double & wz) const;
  void publish_zero_twist();
  void odom_callback(nav_msgs::msg::Odometry::ConstSharedPtr msg);

  // ----- parameters -----
  std::string joint_name_{"planar_joint"};
  std::string cmd_twist_topic_{"/cmd_twist"};
  std::string odom_topic_{"/odom"};
  std::string map_frame_{"map"};
  std::string base_frame_{"base_link"};
  bool use_odom_state_{true};
  double kp_xy_{1.0};
  double kp_yaw_{1.5};
  // Velocity-damping (D) gains. Multiply the *measured* base velocity (from
  // /odom) and subtract from the commanded twist to suppress overshoot at
  // the end of a planned trajectory. 0 = pure P controller (legacy).
  double kd_xy_{0.0};
  double kd_yaw_{0.0};
  // Decel-window radius (meters / radians). Inside this window around the
  // current setpoint the commanded twist magnitude is additionally clamped
  // proportionally to the position error, so the base eases into the goal
  // instead of saturating up to the very last sample. 0 = disabled.
  double decel_window_xy_{0.0};
  double decel_window_yaw_{0.0};
  // Stop-band: when |pos_err| < tolerance/2 AND |measured_v| < this, command
  // an exact zero twist so the chassis driver does not get a stream of
  // sub-cm twitches that the wheels would amplify. 0 = disabled.
  double stop_band_v_{0.0};
  double stop_band_w_{0.0};
  double max_linear_speed_{0.5};
  double max_angular_speed_{1.0};
  double goal_tolerance_xy_{0.05};
  double goal_tolerance_yaw_{0.05};
  double goal_time_tolerance_{2.0};
  double odom_timeout_{0.5};

  // ----- runtime state -----
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_twist_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp_action::Server<FollowJointTrajectory>::SharedPtr action_server_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // protects the latest odom pose snapshot
  std::mutex odom_mutex_;
  bool odom_received_{false};
  rclcpp::Time odom_stamp_;
  double odom_x_{0.0};
  double odom_y_{0.0};
  double odom_yaw_{0.0};
  // World-frame measured base velocity (rotated from msg.twist.twist body
  // frame at odom_callback time). Used by the D-term and the stop-band.
  double odom_vx_w_{0.0};
  double odom_vy_w_{0.0};
  double odom_wz_{0.0};

  // protects current_goal_/trajectory_/start_time_
  std::mutex goal_mutex_;
  std::shared_ptr<GoalHandle> current_goal_;
  trajectory_msgs::msg::MultiDOFJointTrajectory multi_dof_trajectory_;
  rclcpp::Time trajectory_start_time_;
  // Index of our planar joint inside multi_dof_trajectory_.joint_names.
  int planar_idx_{-1};
};

}  // namespace genie_sim_control

#endif  // GENIE_SIM_CONTROL__PLANAR_BASE_CONTROLLER_HPP_
