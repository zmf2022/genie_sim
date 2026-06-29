#include "genie_sim_control/genie_sim_robot_interface.hpp"

#include <hardware_interface/types/hardware_interface_type_values.hpp>
#include <pluginlib/class_list_macros.hpp>

#include <algorithm>

namespace genie_sim_control
{

#ifdef GENIE_HW_USE_PARAMS_API
// ROS Jazzy (hardware_interface 4.x).  The base ``SystemInterface
// ::on_init(params)`` reads ``params.hardware_info`` into the
// base-class member ``info_`` so the body below — which iterates
// ``info_.joints`` — works unchanged across both APIs.
CallbackReturn GenieSimRobotInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
#else
// ROS Humble (hardware_interface 2.x/3.x) legacy signature.
CallbackReturn GenieSimRobotInterface::on_init(const hardware_interface::HardwareInfo & info)
{
  if (SystemInterface::on_init(info) != CallbackReturn::SUCCESS) {
    return CallbackReturn::ERROR;
  }
#endif

  joint_names_.clear();
  for (const auto & joint : info_.joints) {
    joint_names_.push_back(joint.name);
  }
  size_t n = joint_names_.size();

  hw_position_.assign(n, 0.0);
  hw_velocity_.assign(n, 0.0);
  hw_effort_.assign(n, 0.0);
  cmd_position_.assign(n, 0.0);
  cmd_velocity_.assign(n, 0.0);
  cmd_effort_.assign(n, 0.0);

  for (size_t i = 0; i < n; ++i) {
    name_to_idx_[joint_names_[i]] = i;
  }

  auto it_js = info_.hardware_parameters.find("joint_states_topic");
  joint_states_topic_ =
    (it_js != info_.hardware_parameters.end()) ? it_js->second : "/joint_states";

  auto it_jc = info_.hardware_parameters.find("joint_command_topic");
  joint_command_topic_ =
    (it_jc != info_.hardware_parameters.end()) ? it_jc->second : "/joint_command";

  return CallbackReturn::SUCCESS;
}

CallbackReturn GenieSimRobotInterface::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  node_ = rclcpp::Node::make_shared("genie_sim_robot_interface");

  sub_joint_state_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    joint_states_topic_, rclcpp::SensorDataQoS(),
    std::bind(&GenieSimRobotInterface::joint_state_callback, this, std::placeholders::_1));

  pub_joint_command_ = node_->create_publisher<sensor_msgs::msg::JointState>(
    joint_command_topic_, rclcpp::SensorDataQoS());

  executor_.add_node(node_);

  RCLCPP_INFO(
    node_->get_logger(),
    "[GenieSimHW] Configured: %zu joints, reading '%s', writing '%s'",
    joint_names_.size(), joint_states_topic_.c_str(), joint_command_topic_.c_str());

  return CallbackReturn::SUCCESS;
}

CallbackReturn GenieSimRobotInterface::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  std::lock_guard<std::mutex> lk(state_mutex_);
  cmd_position_ = hw_position_;
  std::fill(cmd_velocity_.begin(), cmd_velocity_.end(), 0.0);
  std::fill(cmd_effort_.begin(), cmd_effort_.end(), 0.0);
  return CallbackReturn::SUCCESS;
}

CallbackReturn GenieSimRobotInterface::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  return CallbackReturn::SUCCESS;
}

CallbackReturn GenieSimRobotInterface::on_cleanup(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  executor_.remove_node(node_);
  sub_joint_state_.reset();
  pub_joint_command_.reset();
  node_.reset();
  return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface>
GenieSimRobotInterface::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &hw_position_[i]);
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_VELOCITY, &hw_velocity_[i]);
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_EFFORT, &hw_effort_[i]);
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface>
GenieSimRobotInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> interfaces;
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_POSITION, &cmd_position_[i]);
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_VELOCITY, &cmd_velocity_[i]);
    interfaces.emplace_back(
      joint_names_[i], hardware_interface::HW_IF_EFFORT, &cmd_effort_[i]);
  }
  return interfaces;
}

hardware_interface::return_type GenieSimRobotInterface::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  executor_.spin_some(std::chrono::milliseconds(1));
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type GenieSimRobotInterface::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Don't publish a command until the sim's first /joint_states has been
  // received. Until then cmd_position_ is still the zero-initialized vector
  // (on_init), so publishing would command every joint to 0 — snapping the
  // arm (and any hand-attached payload, e.g. the chef wok) on startup, the
  // one-shot jerk seen when wbc.launch.py comes up against a running sim.
  // The joint_state_callback seeds cmd_position_ = hw_position_ on the first
  // message (state_received_), so the first command we publish here equals
  // the current pose and produces no motion.
  if (!state_received_) {
    return hardware_interface::return_type::OK;
  }
  if (pub_joint_command_) {
    auto msg = std::make_unique<sensor_msgs::msg::JointState>();
    msg->header.stamp = node_->now();
    msg->name.reserve(joint_names_.size());
    msg->position.reserve(joint_names_.size());
    msg->velocity.reserve(joint_names_.size());
    msg->effort.reserve(joint_names_.size());
    for (size_t i = 0; i < joint_names_.size(); ++i) {
      msg->name.push_back(joint_names_[i]);
      msg->position.push_back(cmd_position_[i]);
      msg->velocity.push_back(cmd_velocity_[i]);
      msg->effort.push_back(cmd_effort_[i]);
    }
    if (!msg->name.empty()) {
      pub_joint_command_->publish(std::move(msg));
    }
  }

  return hardware_interface::return_type::OK;
}

void GenieSimRobotInterface::joint_state_callback(
  const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::lock_guard<std::mutex> lk(state_mutex_);
  for (size_t i = 0; i < msg->name.size(); ++i) {
    auto it = name_to_idx_.find(msg->name[i]);
    if (it == name_to_idx_.end()) {
      continue;
    }
    size_t idx = it->second;
    if (i < msg->position.size()) {
      hw_position_[idx] = msg->position[i];
    }
    if (i < msg->velocity.size()) {
      hw_velocity_[idx] = msg->velocity[i];
    }
    if (i < msg->effort.size()) {
      hw_effort_[idx] = msg->effort[i];
    }
  }
  if (!state_received_) {
    for (size_t i = 0; i < cmd_position_.size(); ++i) {
      cmd_position_[i] = hw_position_[i];
    }
    state_received_ = true;
  }
}

}  // namespace genie_sim_control

PLUGINLIB_EXPORT_CLASS(
  genie_sim_control::GenieSimRobotInterface,
  hardware_interface::SystemInterface)
