#pragma once

#include <memory>
#include <string>
#include <vector>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>

#include "genie_sim_controllers/servo_core.hpp"

namespace genie_sim_controllers
{

/// ros2_control plugin: hosts ``ServoCore`` inside a controller_interface
/// lifecycle. Driven by the controller_manager's ``update()`` callback.
/// See also ``ServoNode`` for the standalone-rclcpp::Node form (same
/// ServoCore, wall-timer-driven).
///
/// Does not claim any hardware interfaces — runs entirely off topics
/// (``/cmd_twist`` + ``/joint_states`` in, ``/cmd_4ws`` out), same
/// pattern as ``PlanarBaseController``.
class ChassisServoController : public controller_interface::ControllerInterface
{
public:
  ChassisServoController() = default;

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
  void drainEventsToLogger();

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_twist_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr set_mode_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr cmd_4ws_pub_;

  std::vector<std::string> steer_joint_names_;
  std::vector<std::string> drive_joint_names_;

  ServoCore core_;
};

}  // namespace genie_sim_controllers
