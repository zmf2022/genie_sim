#pragma once

#include "genie_sim_moveit_plugins/moveit_compat.hpp"
#include MOVEIT_H_PLANNING_INTERFACE
#include <rclcpp/rclcpp.hpp>
#include <string>

namespace genie_sim_moveit_plugins
{

class GeniePlanningContext : public planning_interface::PlanningContext
{
public:
  GeniePlanningContext(
    const std::string & name, const std::string & group,
    const moveit::core::RobotModelConstPtr & model);

  // MoveIt API: Humble returns ``bool``, Jazzy returns ``void``.
  // Macro picks per ``ROS_DISTRO``; see ``moveit_compat.hpp``.
  GENIE_MOVEIT_SOLVE_RT solve(planning_interface::MotionPlanResponse & res) override;
  GENIE_MOVEIT_SOLVE_RT solve(planning_interface::MotionPlanDetailedResponse & res) override;
  bool terminate() override;
  void clear() override;

private:
  moveit::core::RobotModelConstPtr robot_model_;
  bool terminated_{false};
};

class GeniePlannerManager : public planning_interface::PlannerManager
{
public:
  GeniePlannerManager() = default;
  ~GeniePlannerManager() override = default;

  bool initialize(
    const moveit::core::RobotModelConstPtr & model,
    const rclcpp::Node::SharedPtr & node,
    const std::string & parameter_namespace) override;

  bool canServiceRequest(
    const moveit_msgs::msg::MotionPlanRequest & req) const override;

  std::string getDescription() const override;

  void getPlanningAlgorithms(
    std::vector<std::string> & algs) const override;

  void setPlannerConfigurations(
    const planning_interface::PlannerConfigurationMap & pcs) override;

  planning_interface::PlanningContextPtr getPlanningContext(
    const planning_scene::PlanningSceneConstPtr & planning_scene,
    const planning_interface::MotionPlanRequest & req,
    moveit_msgs::msg::MoveItErrorCodes & error_code) const override;

private:
  moveit::core::RobotModelConstPtr robot_model_;
  rclcpp::Node::SharedPtr node_;
  planning_interface::PlannerConfigurationMap config_map_;
};

}  // namespace genie_sim_moveit_plugins
