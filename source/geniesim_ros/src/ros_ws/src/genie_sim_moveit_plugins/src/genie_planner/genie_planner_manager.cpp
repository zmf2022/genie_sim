#include "genie_planner_manager.hpp"
#include "rrt_connect_planner.hpp"
#include "topp_ra.hpp"

#include MOVEIT_H_PLANNING_SCENE
#include MOVEIT_H_CONVERSIONS
#include MOVEIT_H_ROBOT_TRAJECTORY
#include <pluginlib/class_list_macros.hpp>

namespace genie_sim_moveit_plugins
{

GeniePlanningContext::GeniePlanningContext(
  const std::string & name, const std::string & group,
  const moveit::core::RobotModelConstPtr & model)
: PlanningContext(name, group), robot_model_(model)
{
}

GENIE_MOVEIT_SOLVE_RT GeniePlanningContext::solve(
  planning_interface::MotionPlanResponse & res)
{
  terminated_ = false;
  const auto & req = getMotionPlanRequest();
  const auto & scene = getPlanningScene();
  const auto * jmg = robot_model_->getJointModelGroup(getGroupName());
  if (!jmg) {
    res.GENIE_MOVEIT_FIELD(error_code).val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    GENIE_MOVEIT_SOLVE_ERR;
  }

  const auto & bounds = jmg->getActiveJointModelsBounds();
  size_t ndof = jmg->getVariableCount();
  Eigen::VectorXd lower(ndof), upper(ndof);
  Eigen::VectorXd vel_lim(ndof), acc_lim(ndof);
  size_t idx = 0;
  for (const auto & jb : bounds) {
    for (const auto & b : *jb) {
      lower[idx] = b.min_position_;
      upper[idx] = b.max_position_;
      vel_lim[idx] = b.velocity_bounded_ ? b.max_velocity_ : 3.14;
      acc_lim[idx] = b.acceleration_bounded_ ? b.max_acceleration_ : 10.0;
      ++idx;
    }
  }

  moveit::core::RobotState start_state(robot_model_);
  start_state = scene->getCurrentState();
  if (!req.start_state.joint_state.name.empty()) {
    moveit::core::robotStateMsgToRobotState(req.start_state, start_state);
  }

  Eigen::VectorXd q_start(ndof);
  start_state.copyJointGroupPositions(jmg, q_start.data());

  moveit::core::RobotState goal_state(robot_model_);
  goal_state = start_state;
  for (const auto & gc : req.goal_constraints) {
    for (const auto & jc : gc.joint_constraints) {
      goal_state.setVariablePosition(jc.joint_name, jc.position);
    }
  }
  Eigen::VectorXd q_goal(ndof);
  goal_state.copyJointGroupPositions(jmg, q_goal.data());

  auto space = std::make_shared<StateSpace>(lower, upper);
  space->setValidChecker(
    [&](const Eigen::VectorXd & q) -> bool {
      moveit::core::RobotState test_state(robot_model_);
      test_state = start_state;
      test_state.setJointGroupPositions(jmg, q.data());
      test_state.update();
      return !scene->isStateColliding(test_state, getGroupName());
    });

  RRTConnect rrt(space);
  rrt.settings().max_iter = 10000;

  bool found = rrt.solve(q_start, {q_goal});
  if (!found || terminated_) {
    res.GENIE_MOVEIT_FIELD(error_code).val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    GENIE_MOVEIT_SOLVE_ERR;
  }

  rrt.simplifyPath();
  auto path = rrt.getPathSimplified();
  if (path.empty()) {path = rrt.getPath();}

  double speed_scale = req.max_velocity_scaling_factor > 0.0 ?
    req.max_velocity_scaling_factor : 0.1;
  double acc_scale = req.max_acceleration_scaling_factor > 0.0 ?
    req.max_acceleration_scaling_factor : 0.1;

  constexpr double OUTPUT_DT = 0.01;
  auto topp = toppRA(path, vel_lim * speed_scale, acc_lim * acc_scale, OUTPUT_DT);
  if (!topp.success) {
    res.GENIE_MOVEIT_FIELD(error_code).val = moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    GENIE_MOVEIT_SOLVE_ERR;
  }

  auto traj = std::make_shared<robot_trajectory::RobotTrajectory>(
    robot_model_, getGroupName());
  for (size_t i = 0; i < topp.positions.size(); ++i) {
    moveit::core::RobotState ws(robot_model_);
    ws = start_state;
    ws.setJointGroupPositions(jmg, topp.positions[i].data());
    ws.setJointGroupVelocities(jmg, topp.velocities[i].data());
    ws.update();
    traj->addSuffixWayPoint(ws, OUTPUT_DT);
  }

  res.GENIE_MOVEIT_FIELD(trajectory) = traj;
  res.GENIE_MOVEIT_FIELD(planning_time) = 0.0;
  res.GENIE_MOVEIT_FIELD(error_code).val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  GENIE_MOVEIT_SOLVE_OK;
}

GENIE_MOVEIT_SOLVE_RT GeniePlanningContext::solve(
  planning_interface::MotionPlanDetailedResponse & res)
{
  planning_interface::MotionPlanResponse simple;
#ifdef GENIE_MOVEIT_USE_NEW_API
  // Jazzy: void solve() — call it for side effects, no "ok" flag.
  solve(simple);
  const bool ok =
    simple.GENIE_MOVEIT_FIELD(error_code).val ==
    moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
#else
  // Humble: bool solve() returns success/failure directly.
  bool ok = solve(simple);
#endif
  res.GENIE_MOVEIT_FIELD(error_code) = simple.GENIE_MOVEIT_FIELD(error_code);
  if (simple.GENIE_MOVEIT_FIELD(trajectory)) {
    res.GENIE_MOVEIT_FIELD(trajectory).push_back(simple.GENIE_MOVEIT_FIELD(trajectory));
    res.GENIE_MOVEIT_FIELD(processing_time).push_back(simple.GENIE_MOVEIT_FIELD(planning_time));
    res.GENIE_MOVEIT_FIELD(description).emplace_back("GenieRRTConnect+TOPPRA");
  }
#ifdef GENIE_MOVEIT_USE_NEW_API
  (void)ok;  // void return — silence unused-variable if any
  return;
#else
  return ok;
#endif
}

bool GeniePlanningContext::terminate()
{
  terminated_ = true;
  return true;
}

void GeniePlanningContext::clear() {}

bool GeniePlannerManager::initialize(
  const moveit::core::RobotModelConstPtr & model,
  const rclcpp::Node::SharedPtr & node,
  const std::string & /*parameter_namespace*/)
{
  robot_model_ = model;
  node_ = node;
  RCLCPP_INFO(
    node_->get_logger(),
    "[GeniePlannerManager] Initialized for robot '%s'",
    model->getName().c_str());
  return true;
}

bool GeniePlannerManager::canServiceRequest(
  const moveit_msgs::msg::MotionPlanRequest & req) const
{
  return robot_model_->hasJointModelGroup(req.group_name);
}

std::string GeniePlannerManager::getDescription() const
{
  return "Genie RRT-Connect + TOPP-RA planner";
}

void GeniePlannerManager::getPlanningAlgorithms(
  std::vector<std::string> & algs) const
{
  algs = {"GenieRRTConnect"};
}

void GeniePlannerManager::setPlannerConfigurations(
  const planning_interface::PlannerConfigurationMap & pcs)
{
  config_map_ = pcs;
}

planning_interface::PlanningContextPtr GeniePlannerManager::getPlanningContext(
  const planning_scene::PlanningSceneConstPtr & planning_scene,
  const planning_interface::MotionPlanRequest & req,
  moveit_msgs::msg::MoveItErrorCodes & error_code) const
{
  if (!robot_model_->hasJointModelGroup(req.group_name)) {
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    return nullptr;
  }

  auto ctx = std::make_shared<GeniePlanningContext>(
    "GeniePlanningContext", req.group_name, robot_model_);
  ctx->setMotionPlanRequest(req);
  ctx->setPlanningScene(planning_scene);
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return ctx;
}

}  // namespace genie_sim_moveit_plugins

PLUGINLIB_EXPORT_CLASS(
  genie_sim_moveit_plugins::GeniePlannerManager,
  planning_interface::PlannerManager)
