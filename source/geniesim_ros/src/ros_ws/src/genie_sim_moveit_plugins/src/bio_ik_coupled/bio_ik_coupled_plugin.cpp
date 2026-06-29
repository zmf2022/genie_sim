#include <bio_ik/goal_types.h>
#include <bio_ik/bio_ik.h>

#include "genie_sim_moveit_plugins/moveit_compat.hpp"
#include MOVEIT_H_KINEMATICS_BASE
#include <pluginlib/class_list_macros.hpp>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/rclcpp.hpp>

#include "genie_sim_moveit_plugins/coupled_constraints.hpp"

namespace genie_sim_moveit_plugins
{

// Soft penalty added to bio_ik's cost: cost = Σ max(0, A·q − (b − margin))².
// bio_ik addresses variables by string name, so we hold them by name
// even though the canonical CoupledConstraint stores indices.
class CoupledBoundGoal : public bio_ik::Goal
{
  std::vector<std::string> variable_names_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  double margin_;

public:
  CoupledBoundGoal(
    std::vector<std::string> variable_names,
    Eigen::MatrixXd A,
    Eigen::VectorXd b,
    double margin,
    double weight)
  : variable_names_(std::move(variable_names)),
    A_(std::move(A)), b_(std::move(b)), margin_(margin)
  {
    weight_ = weight;
  }

  void describe(bio_ik::GoalContext & context) const override
  {
    Goal::describe(context);
    for (const auto & name : variable_names_) {
      context.addVariable(name);
    }
  }

  double evaluate(const bio_ik::GoalContext & context) const override
  {
    const size_t n_cols = variable_names_.size();
    double cost = 0.0;
    for (Eigen::Index r = 0; r < A_.rows(); ++r) {
      double s = 0.0;
      for (size_t c = 0; c < n_cols; ++c) {
        s += A_(r, c) * context.getVariablePosition(c);
      }
      double violation = s - (b_[r] - margin_);
      if (violation > 0.0) {cost += violation * violation;}
    }
    return cost;
  }
};

// bio_ik wrapper that injects coupled-constraint goals (soft penalty
// inside bio_ik's cost) plus a callback guard (hard reject after the
// solver returns) and forwards everything else to the upstream
// bio_ik/BioIKKinematicsPlugin.
class BioIKPlugin : public kinematics::KinematicsBase
{
public:
  BioIKPlugin() = default;

  ~BioIKPlugin() override
  {
    inner_.reset();
    loader_.reset();
  }

  bool initialize(
    const rclcpp::Node::SharedPtr & node,
    const moveit::core::RobotModel & robot_model,
    const std::string & group_name,
    const std::string & base_frame,
    const std::vector<std::string> & tip_frames,
    double search_discretization) override
  {
    node_ = node;
    storeValues(
      robot_model, group_name, base_frame, tip_frames, search_discretization);

    try {
      loader_ = std::make_shared<pluginlib::ClassLoader<kinematics::KinematicsBase>>(
        "moveit_core", "kinematics::KinematicsBase");
      inner_ = loader_->createSharedInstance("bio_ik/BioIKKinematicsPlugin");
    } catch (const pluginlib::PluginlibException & e) {
      RCLCPP_ERROR(node->get_logger(), "[GenieBioIK] Failed to load bio_ik: %s", e.what());
      return false;
    }

    if (!inner_->initialize(
        node, robot_model, group_name, base_frame, tip_frames, search_discretization))
    {
      RCLCPP_ERROR(node->get_logger(), "[GenieBioIK] bio_ik init failed");
      return false;
    }

    const auto * jmg = robot_model.getJointModelGroup(group_name);
    if (jmg) {
      active_joint_names_ = jmg->getActiveJointModelNames();
    }

    coupled_margin_ = declareOrGet<double>(node, "coupled_constraint_margin", 0.1);
    center_joints_weight_ = declareOrGet<double>(node, "bio_ik_center_joints_weight", 0.1);
    avoid_limits_weight_ = declareOrGet<double>(node, "bio_ik_avoid_limits_weight", 0.2);
    minimal_displacement_weight_ =
      declareOrGet<double>(node, "bio_ik_minimal_displacement_weight", 0.5);
    coupled_goal_weight_ = declareOrGet<double>(node, "bio_ik_coupled_goal_weight", 10.0);

    coupled_constraints_ = loadCoupledConstraints(
      node, activeJointIndex(jmg), "coupled_constraints_file", "[GenieBioIK]");

    RCLCPP_INFO(
      node->get_logger(),
      "[GenieBioIK] Initialized for '%s' with %zu coupled constraint groups "
      "(center=%.2f, limits=%.2f, disp=%.2f, coupled=%.2f)",
      group_name.c_str(), coupled_constraints_.size(),
      center_joints_weight_, avoid_limits_weight_,
      minimal_displacement_weight_, coupled_goal_weight_);
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
    auto bio_opts = makeBioIKOptions(ik_pose);
    return inner_->searchPositionIK(
      ik_pose, ik_seed_state, timeout, solution,
      wrapCallback({}), error_code, bio_opts ? *bio_opts : options);
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
    auto bio_opts = makeBioIKOptions(ik_pose);
    return inner_->searchPositionIK(
      ik_pose, ik_seed_state, timeout, consistency_limits, solution,
      wrapCallback({}), error_code, bio_opts ? *bio_opts : options);
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
    auto bio_opts = makeBioIKOptions(ik_pose);
    return inner_->searchPositionIK(
      ik_pose, ik_seed_state, timeout, solution,
      wrapCallback(solution_callback), error_code, bio_opts ? *bio_opts : options);
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
    auto bio_opts = makeBioIKOptions(ik_pose);
    return inner_->searchPositionIK(
      ik_pose, ik_seed_state, timeout, consistency_limits, solution,
      wrapCallback(solution_callback), error_code, bio_opts ? *bio_opts : options);
  }

  bool getPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return inner_->getPositionIK(ik_pose, ik_seed_state, solution, error_code, options);
  }

  bool getPositionFK(
    const std::vector<std::string> & link_names,
    const std::vector<double> & joint_angles,
    std::vector<geometry_msgs::msg::Pose> & poses) const override
  {
    return inner_->getPositionFK(link_names, joint_angles, poses);
  }

  const std::vector<std::string> & getJointNames() const override
  {
    return inner_->getJointNames();
  }

  const std::vector<std::string> & getLinkNames() const override
  {
    return inner_->getLinkNames();
  }

private:
  IKCallbackFn wrapCallback(const IKCallbackFn & user_cb) const
  {
    return wrapWithCoupledCheck(coupled_constraints_, coupled_margin_, user_cb);
  }

  std::unique_ptr<bio_ik::BioIKKinematicsQueryOptions> makeBioIKOptions(
    const geometry_msgs::msg::Pose & ik_pose) const
  {
    auto bio_opts = std::make_unique<bio_ik::BioIKKinematicsQueryOptions>();
    bio_opts->replace = true;

    if (!tip_frames_.empty()) {
      auto pose_goal = std::make_unique<bio_ik::PoseGoal>(
        tip_frames_[0],
        tf2::Vector3(ik_pose.position.x, ik_pose.position.y, ik_pose.position.z),
        tf2::Quaternion(
          ik_pose.orientation.x, ik_pose.orientation.y,
          ik_pose.orientation.z, ik_pose.orientation.w),
        1.0);
      pose_goal->setRotationScale(0.5);
      bio_opts->goals.push_back(std::move(pose_goal));
    }

    if (minimal_displacement_weight_ > 0.0) {
      bio_opts->goals.push_back(
        std::make_unique<bio_ik::MinimalDisplacementGoal>(minimal_displacement_weight_, true));
    }
    if (center_joints_weight_ > 0.0) {
      bio_opts->goals.push_back(
        std::make_unique<bio_ik::CenterJointsGoal>(center_joints_weight_, true));
    }
    if (avoid_limits_weight_ > 0.0) {
      bio_opts->goals.push_back(
        std::make_unique<bio_ik::AvoidJointLimitsGoal>(avoid_limits_weight_, true));
    }

    for (const auto & cc : coupled_constraints_) {
      std::vector<std::string> names;
      names.reserve(cc.group_indices.size());
      for (size_t idx : cc.group_indices) {
        names.push_back(active_joint_names_[idx]);
      }
      bio_opts->goals.push_back(
        std::make_unique<CoupledBoundGoal>(
          std::move(names), cc.A, cc.b, coupled_margin_, coupled_goal_weight_));
    }

    return bio_opts;
  }

  rclcpp::Node::SharedPtr node_;
  std::shared_ptr<pluginlib::ClassLoader<kinematics::KinematicsBase>> loader_;
  std::shared_ptr<kinematics::KinematicsBase> inner_;

  std::vector<std::string> active_joint_names_;
  std::vector<CoupledConstraint> coupled_constraints_;

  double coupled_margin_{0.1};
  double center_joints_weight_{0.1};
  double avoid_limits_weight_{0.2};
  double minimal_displacement_weight_{0.5};
  double coupled_goal_weight_{10.0};
};

}  // namespace genie_sim_moveit_plugins

PLUGINLIB_EXPORT_CLASS(
  genie_sim_moveit_plugins::BioIKPlugin,
  kinematics::KinematicsBase)
