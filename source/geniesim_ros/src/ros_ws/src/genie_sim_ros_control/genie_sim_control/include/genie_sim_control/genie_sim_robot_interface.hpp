#pragma once

#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/handle.hpp>
#include <hardware_interface/hardware_info.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#ifdef GENIE_HW_USE_PARAMS_API
// Jazzy + newer: the on_init signature takes a params struct that
// bundles HardwareInfo together with rclcpp handles.  Header is
// only present on hardware_interface 4.x+.
#include <hardware_interface/types/hardware_component_interface_params.hpp>
#endif
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/state.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace genie_sim_control
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class GenieSimRobotInterface : public hardware_interface::SystemInterface
{
public:
#ifdef GENIE_HW_USE_PARAMS_API
  // Jazzy (hardware_interface 4.x) on_init signature.  The base
  // SystemInterface::on_init(params) populates info_ from
  // params.hardware_info, so this class's body keeps reading info_
  // unchanged.
  CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;
#else
  // Humble (hardware_interface 2.x/3.x) legacy signature.
  CallbackReturn on_init(const hardware_interface::HardwareInfo & info) override;
#endif
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);

  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_state_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_joint_command_;
  rclcpp::executors::SingleThreadedExecutor executor_;

  std::vector<std::string> joint_names_;

  std::vector<double> hw_position_;
  std::vector<double> hw_velocity_;
  std::vector<double> hw_effort_;

  std::vector<double> cmd_position_;
  std::vector<double> cmd_velocity_;
  std::vector<double> cmd_effort_;

  std::mutex state_mutex_;
  std::unordered_map<std::string, size_t> name_to_idx_;

  std::string joint_states_topic_;
  std::string joint_command_topic_;

  bool state_received_{false};
};

}  // namespace genie_sim_control
