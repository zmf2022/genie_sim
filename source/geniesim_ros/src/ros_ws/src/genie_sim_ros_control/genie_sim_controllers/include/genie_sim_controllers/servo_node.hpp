#pragma once

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>

#include "genie_sim_controllers/servo_core.hpp"

namespace genie_sim_controllers
{

/// Standalone rclcpp::Node form of the four-wheel-steering servo. Same
/// ``ServoCore`` as the ros2_control controller plugin
/// (``ChassisServoController``); the only difference is the surrounding
/// shell:
///
///   * tick is driven by a ``rclcpp::WallTimer`` at ``control_rate`` Hz
///     (the controller plugin uses the controller_manager's
///     ``update()`` callback instead).
///   * subscriptions / publisher / parameters live on this Node
///     instead of the controller's lifecycle Node.
///
/// Use this when you don't have a ``controller_manager`` in your
/// bringup graph (teleop-only, navigation-only, dev rigs). For the
/// ros2_control path, load ``ChassisServoController`` via the
/// controller_manager spawner — same parameter schema (the bundled
/// config files use the ``/**`` wildcard so a single YAML drives both
/// forms).
class ServoNode : public rclcpp::Node
{
public:
  explicit ServoNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void onTimer();
  void drainEventsToLogger();

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_twist_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_states_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr set_mode_sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr cmd_4ws_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;

  std::vector<std::string> steer_joint_names_;
  std::vector<std::string> drive_joint_names_;

  ServoCore core_;
};

}  // namespace genie_sim_controllers
