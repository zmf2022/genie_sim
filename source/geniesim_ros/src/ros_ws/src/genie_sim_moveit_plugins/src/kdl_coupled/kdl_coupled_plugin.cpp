#include "genie_sim_moveit_plugins/moveit_compat.hpp"
#include MOVEIT_H_KDL_KINEMATICS_PLUGIN
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "genie_sim_moveit_plugins/coupled_constraints.hpp"

namespace genie_sim_moveit_plugins
{

// MoveIt's stock KDL plugin + a coupled-constraint guard installed as
// an IKCallbackFn so the solver rejects samples that violate the
// configured linear inequalities on body / arm joints.
class KDLKinematicsPlugin : public kdl_kinematics_plugin::KDLKinematicsPlugin
{
public:
  KDLKinematicsPlugin() = default;

  bool initialize(
    const rclcpp::Node::SharedPtr & node,
    const moveit::core::RobotModel & robot_model,
    const std::string & group_name,
    const std::string & base_frame,
    const std::vector<std::string> & tip_frames,
    double search_discretization) override
  {
    if (!kdl_kinematics_plugin::KDLKinematicsPlugin::initialize(
        node, robot_model, group_name, base_frame, tip_frames,
        search_discretization))
    {
      return false;
    }

    const auto * jmg = robot_model.getJointModelGroup(group_name);
    if (!jmg) {return true;}

    coupled_margin_ = declareOrGet<double>(node, "coupled_constraint_margin", 0.1);
    coupled_constraints_ = loadCoupledConstraints(
      node, activeJointIndex(jmg), "coupled_constraints_file", "[GenieKDL]");

    RCLCPP_INFO(
      node->get_logger(),
      "[GenieKDL] Initialized for '%s' with %zu coupled constraint groups (margin=%.3f)",
      group_name.c_str(), coupled_constraints_.size(), coupled_margin_);
    return true;
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return kdl_kinematics_plugin::KDLKinematicsPlugin::searchPositionIK(
      ik_pose, ik_seed_state, timeout, solution, wrapCallback({}), error_code, options);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    const std::vector<double> & consistency_limits,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return kdl_kinematics_plugin::KDLKinematicsPlugin::searchPositionIK(
      ik_pose, ik_seed_state, timeout, consistency_limits,
      solution, wrapCallback({}), error_code, options);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    std::vector<double> & solution,
    const IKCallbackFn & solution_callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return kdl_kinematics_plugin::KDLKinematicsPlugin::searchPositionIK(
      ik_pose, ik_seed_state, timeout, solution,
      wrapCallback(solution_callback), error_code, options);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    const std::vector<double> & consistency_limits,
    std::vector<double> & solution,
    const IKCallbackFn & solution_callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return kdl_kinematics_plugin::KDLKinematicsPlugin::searchPositionIK(
      ik_pose, ik_seed_state, timeout, consistency_limits,
      solution, wrapCallback(solution_callback), error_code, options);
  }

private:
  IKCallbackFn wrapCallback(const IKCallbackFn & user_cb) const
  {
    return wrapWithCoupledCheck(coupled_constraints_, coupled_margin_, user_cb);
  }

  std::vector<CoupledConstraint> coupled_constraints_;
  double coupled_margin_{0.1};
};

}  // namespace genie_sim_moveit_plugins

PLUGINLIB_EXPORT_CLASS(
  genie_sim_moveit_plugins::KDLKinematicsPlugin,
  kinematics::KinematicsBase)
