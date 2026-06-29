/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016-2017, Philipp Sebastian Ruppel
 *  All rights reserved.
 *  ROS2 Humble port + coupled constraint support
 *********************************************************************/

#include <bio_ik/goal.h>

#include "forward_kinematics.h"
#include "ik_base.h"
#include "ik_parallel.h"
#include "problem.h"
#include "utils.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <kdl_parser/kdl_parser.hpp>
#ifdef BIO_IK_MOVEIT_USE_NEW_API
#include <moveit/kinematics_base/kinematics_base.hpp>
#else
#include <moveit/kinematics_base/kinematics_base.h>
#endif
#include <pluginlib/class_list_macros.hpp>

#include <tf2_eigen/tf2_eigen.hpp>

#ifdef BIO_IK_MOVEIT_USE_NEW_API
#include <moveit/robot_model/robot_model.hpp>
#include <moveit/robot_state/robot_state.hpp>
#else
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#endif

#include <atomic>
#include <mutex>
#include <random>
#include <tuple>
#include <type_traits>

#include <bio_ik/goal_types.h>

using namespace bio_ik;

namespace bio_ik
{

std::mutex bioIKKinematicsQueryOptionsMutex;
std::unordered_set<const void *> bioIKKinematicsQueryOptionsList;

BioIKKinematicsQueryOptions::BioIKKinematicsQueryOptions()
: replace(false), solution_fitness(0)
{
  std::lock_guard<std::mutex> lock(bioIKKinematicsQueryOptionsMutex);
  bioIKKinematicsQueryOptionsList.insert(this);
}

BioIKKinematicsQueryOptions::~BioIKKinematicsQueryOptions()
{
  std::lock_guard<std::mutex> lock(bioIKKinematicsQueryOptionsMutex);
  bioIKKinematicsQueryOptionsList.erase(this);
}

bool isBioIKKinematicsQueryOptions(const void * ptr)
{
  std::lock_guard<std::mutex> lock(bioIKKinematicsQueryOptionsMutex);
  return bioIKKinematicsQueryOptionsList.find(ptr) !=
         bioIKKinematicsQueryOptionsList.end();
}

const BioIKKinematicsQueryOptions *
toBioIKKinematicsQueryOptions(const void * ptr)
{
  if (isBioIKKinematicsQueryOptions(ptr)) {
    return (const BioIKKinematicsQueryOptions *)ptr;
  } else {
    return 0;
  }
}

}

namespace bio_ik_kinematics_plugin
{

template<class T>
static void lookupParam(
  const rclcpp::Node::SharedPtr & node,
  const std::string & param, T & val,
  const T & default_val)
{
  val = default_val;
  if (!node) {return;}
  std::string full_param = param;
  if (node->has_parameter(full_param)) {
    node->get_parameter(full_param, val);
  } else {
    try {
      node->declare_parameter(full_param, default_val);
    } catch (...) {
    }
    node->get_parameter(full_param, val);
  }
}

struct BioIKKinematicsPlugin : kinematics::KinematicsBase
{
  std::vector<std::string> joint_names, link_names;
  const moveit::core::JointModelGroup * joint_model_group;
  mutable std::unique_ptr<IKParallel> ik;
  mutable std::vector<double> state, temp;
  mutable std::unique_ptr<moveit::core::RobotState> temp_state;
  mutable std::vector<Frame> tipFrames;
  RobotInfo robot_info;
  bool enable_profiler;
  rclcpp::Node::SharedPtr node_;

  BioIKKinematicsPlugin() {enable_profiler = false;}

  const std::vector<std::string> & getJointNames() const override
  {
    return joint_names;
  }

  const std::vector<std::string> & getLinkNames() const override
  {
    return link_names;
  }

  bool getPositionFK(
    const std::vector<std::string> & link_names,
    const std::vector<double> & joint_angles,
    std::vector<geometry_msgs::msg::Pose> & poses) const override
  {
    return false;
  }

  bool getPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return false;
  }

  EigenSTL::vector_Isometry3d tip_reference_frames;

  mutable std::vector<std::unique_ptr<Goal>> default_goals;
  mutable std::vector<const bio_ik::Goal *> all_goals;

  IKParams ikparams;
  mutable Problem problem;

  bool load(const std::string & group_name)
  {
    joint_model_group = robot_model_->getJointModelGroup(group_name);
    if (!joint_model_group) {
      LOG("failed to get joint model group");
      return false;
    }

    joint_names.clear();

    for (auto * joint_model : joint_model_group->getJointModels()) {
      if (joint_model->getName() != base_frame_ &&
        joint_model->getType() != moveit::core::JointModel::UNKNOWN &&
        joint_model->getType() != moveit::core::JointModel::FIXED)
      {
        joint_names.push_back(joint_model->getName());
      }
    }

    auto tips2 = tip_frames_;
    joint_model_group->getEndEffectorTips(tips2);
    if (!tips2.empty()) {
      tip_frames_ = tips2;
    }

    link_names = tip_frames_;

    lookupParam(node_, "profiler", enable_profiler, false);

    robot_info = RobotInfo(robot_model_);

    ikparams.robot_model = robot_model_;
    ikparams.joint_model_group = joint_model_group;

    lookupParam(
      node_, "mode", ikparams.solver_class_name,
      std::string("bio2_memetic"));
    lookupParam(node_, "counter", ikparams.enable_counter, false);
    lookupParam(node_, "threads", ikparams.thread_count, 0);
    lookupParam(
      node_, "random_seed", ikparams.random_seed,
      static_cast<int>(std::random_device()()));

    lookupParam(node_, "dpos", ikparams.dpos, DBL_MAX);
    lookupParam(node_, "drot", ikparams.drot, DBL_MAX);
    lookupParam(node_, "dtwist", ikparams.dtwist, 1e-5);

    lookupParam(node_, "no_wipeout", ikparams.opt_no_wipeout, false);
    lookupParam(node_, "population_size", ikparams.population_size, 8);
    lookupParam(node_, "elite_count", ikparams.elite_count, 4);
    lookupParam(node_, "linear_fitness", ikparams.linear_fitness, false);

    temp_state.reset(new moveit::core::RobotState(robot_model_));

    ik.reset(new IKParallel(ikparams));

    {
      default_goals.clear();

      for (size_t i = 0; i < tip_frames_.size(); i++) {
        PoseGoal * goal = new PoseGoal();
        goal->setLinkName(tip_frames_[i]);
        double rotation_scale = 0.5;
        lookupParam(node_, "rotation_scale", rotation_scale, rotation_scale);
        bool position_only_ik = false;
        lookupParam(node_, "position_only_ik", position_only_ik, position_only_ik);
        if (position_only_ik) {
          rotation_scale = 0;
        }
        goal->setRotationScale(rotation_scale);
        default_goals.emplace_back(goal);
      }

      {
        double weight = 0;
        lookupParam(node_, "center_joints_weight", weight, weight);
        if (weight > 0.0) {
          auto * g = new bio_ik::CenterJointsGoal();
          g->setWeight(weight);
          default_goals.emplace_back(g);
        }
      }

      {
        double weight = 0;
        lookupParam(node_, "avoid_joint_limits_weight", weight, weight);
        if (weight > 0.0) {
          auto * g = new bio_ik::AvoidJointLimitsGoal();
          g->setWeight(weight);
          default_goals.emplace_back(g);
        }
      }

      {
        double weight = 0;
        lookupParam(node_, "minimal_displacement_weight", weight, weight);
        if (weight > 0.0) {
          auto * g = new bio_ik::MinimalDisplacementGoal();
          g->setWeight(weight);
          default_goals.emplace_back(g);
        }
      }
    }

    return true;
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
      robot_model, group_name, base_frame, tip_frames,
      search_discretization);
    if (!load(group_name)) {return false;}
    RCLCPP_INFO(
      node->get_logger(), "[BioIK] Initialized for '%s' (%zu tips)",
      group_name.c_str(), tip_frames.size());
    return true;
  }

  bool
  searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state, double timeout,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return searchPositionIK(
      std::vector<geometry_msgs::msg::Pose>{ik_pose},
      ik_seed_state, timeout, std::vector<double>(),
      solution, IKCallbackFn(), error_code, options);
  }

  bool
  searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state, double timeout,
    const std::vector<double> & consistency_limits,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return searchPositionIK(
      std::vector<geometry_msgs::msg::Pose>{ik_pose},
      ik_seed_state, timeout, consistency_limits,
      solution, IKCallbackFn(), error_code, options);
  }

  bool
  searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state, double timeout,
    std::vector<double> & solution,
    const IKCallbackFn & solution_callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return searchPositionIK(
      std::vector<geometry_msgs::msg::Pose>{ik_pose},
      ik_seed_state, timeout, std::vector<double>(),
      solution, solution_callback, error_code, options);
  }

  bool
  searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state, double timeout,
    const std::vector<double> & consistency_limits,
    std::vector<double> & solution,
    const IKCallbackFn & solution_callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    return searchPositionIK(
      std::vector<geometry_msgs::msg::Pose>{ik_pose},
      ik_seed_state, timeout, consistency_limits,
      solution, solution_callback, error_code, options);
  }

  bool
  searchPositionIK(
    const std::vector<geometry_msgs::msg::Pose> & ik_poses,
    const std::vector<double> & ik_seed_state, double timeout,
    const std::vector<double> & consistency_limits,
    std::vector<double> & solution,
    const IKCallbackFn & solution_callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions(),
    const moveit::core::RobotState * context_state = NULL) const
  {
    double t0 = wallTime();

    if (enable_profiler) {
      Profiler::start();
    }

    auto * bio_ik_options = toBioIKKinematicsQueryOptions(&options);

    state.resize(robot_model_->getVariableCount());
    if (context_state) {
      for (size_t i = 0; i < robot_model_->getVariableCount(); i++) {
        state[i] = context_state->getVariablePositions()[i];
      }
    } else {
      robot_model_->getVariableDefaultPositions(state);
    }

    solution = ik_seed_state;
    {
      int i = 0;
      for (auto & joint_name : getJointNames()) {
        auto * joint_model = robot_model_->getJointModel(joint_name);
        if (!joint_model) {
          continue;
        }
        for (size_t vi = 0; vi < joint_model->getVariableCount(); vi++) {
          state.at(joint_model->getFirstVariableIndex() + vi) =
            solution.at(i++);
        }
      }
    }

    if (!bio_ik_options || !bio_ik_options->replace) {
      tipFrames.clear();
      // Use getFrameTransform() (not getGlobalLinkTransform(string)) so that
      // a virtual model frame (e.g. SRDF planar virtual_joint with
      // parent_frame="map") resolves to identity instead of throwing
      // "Invalid link". This matches the pattern used by PickNik's
      // stretch_kinematics_plugin for mobile-base + arm composite groups.
      // See moveit_core RobotState::getFrameInfo (robot_state.cpp:1129).
      for (size_t i = 0; i < ik_poses.size(); i++) {
        Eigen::Isometry3d p, r;
        tf2::fromMsg(ik_poses[i], p);
        if (context_state) {
          r = context_state->getFrameTransform(getBaseFrame());
        } else {
          if (i == 0) {
            temp_state->setToDefaultValues();
          }
          r = temp_state->getFrameTransform(getBaseFrame());
        }
        tipFrames.emplace_back(r * p);
      }
    }

    problem.timeout = t0 + timeout;
    problem.initial_guess = state;

    {
      if (!bio_ik_options || !bio_ik_options->replace) {
        for (size_t i = 0; i < tip_frames_.size(); i++) {
          auto * goal = (PoseGoal *)default_goals[i].get();
          goal->setPosition(tipFrames[i].pos);
          goal->setOrientation(tipFrames[i].rot);
        }
      }

      all_goals.clear();

      if (!bio_ik_options || !bio_ik_options->replace) {
        for (auto & goal : default_goals) {
          all_goals.push_back(goal.get());
        }
      }

      if (bio_ik_options) {
        for (auto & goal : bio_ik_options->goals) {
          all_goals.push_back(goal.get());
        }
      }

      {
        problem.initialize(
          ikparams.robot_model, ikparams.joint_model_group,
          ikparams, all_goals, bio_ik_options);
      }
    }

    {
      ik->initialize(problem);
    }

    {
      ik->solve();
    }

    state = ik->getSolution();

    for (auto ivar : problem.active_variables) {
      auto v = state[ivar];
      if (robot_info.isRevolute(ivar) &&
        robot_model_->getMimicJointModels().empty())
      {
        auto r = problem.initial_guess[ivar];
        auto lo = robot_info.getMin(ivar);
        auto hi = robot_info.getMax(ivar);

        if (r < v - M_PI || r > v + M_PI) {
          v -= r;
          v /= (2 * M_PI);
          v += 0.5;
          v -= std::floor(v);
          v -= 0.5;
          v *= (2 * M_PI);
          v += r;
        }

        if (v > hi) {
          v -= std::ceil(std::max(0.0, v - hi) / (2 * M_PI)) * (2 * M_PI);
        }
        if (v < lo) {
          v += std::ceil(std::max(0.0, lo - v) / (2 * M_PI)) * (2 * M_PI);
        }

        if (v < lo) {
          v = lo;
        }
        if (v > hi) {
          v = hi;
        }
      }
      state[ivar] = v;
    }

    robot_model_->enforcePositionBounds(state.data());

    {
      solution.clear();
      for (auto & joint_name : getJointNames()) {
        auto * joint_model = robot_model_->getJointModel(joint_name);
        if (!joint_model) {
          continue;
        }
        for (size_t vi = 0; vi < joint_model->getVariableCount(); vi++) {
          solution.push_back(
            state.at(joint_model->getFirstVariableIndex() + vi));
        }
      }
    }

    if (bio_ik_options) {
      bio_ik_options->solution_fitness = ik->getSolutionFitness();
    }

    if (!ik->getSuccess() && !options.return_approximate_solution) {
      error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
      return false;
    }

    if (solution_callback) {
      solution_callback(ik_poses.front(), solution, error_code);
      return error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    } else {
      error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
      return true;
    }
  }

  bool supportsGroup(
    const moveit::core::JointModelGroup * jmg,
    std::string * error_text_out = 0) const override
  {
    return true;
  }
};
}

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <unordered_map>

namespace genie_sim_moveit_plugins
{

// Bring ``bio_ik_kinematics_plugin::lookupParam`` into this
// namespace so the lambdas inside ``BioIKPlugin::initialize`` can
// call it unqualified.  The original code relied on g++ 11's
// looser unqualified-template name lookup (Humble's compiler) —
// g++ 13 (Jazzy on Ubuntu 24.04) requires the template to be
// findable via qualified lookup OR a using-declaration before any
// call site.  This is a compiler-strictness fix, not a ROS API
// change, so no ROS_DISTRO macro guard is needed.
using bio_ik_kinematics_plugin::lookupParam;

struct GenieCoupledConstraint
{
  std::vector<std::string> joint_names;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
};

class CoupledBoundGoal : public bio_ik::Goal
{
  std::vector<std::string> variable_names_;
  Eigen::MatrixXd A_;
  Eigen::VectorXd b_;
  double margin_;

public:
  CoupledBoundGoal(
    const std::vector<std::string> & variable_names,
    const Eigen::MatrixXd & A,
    const Eigen::VectorXd & b,
    double margin,
    double weight = 10.0)
  : variable_names_(variable_names), A_(A), b_(b), margin_(margin)
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
    size_t n_cols = variable_names_.size();
    size_t n_rows = static_cast<size_t>(A_.rows());
    double cost = 0.0;
    for (size_t r = 0; r < n_rows; ++r) {
      double s = 0.0;
      for (size_t c = 0; c < n_cols; ++c) {
        s += A_(r, c) * context.getVariablePosition(c);
      }
      double violation = s - (b_[r] - margin_);
      if (violation > 0.0) {
        cost += violation * violation;
      }
    }
    return cost;
  }
};

struct BioIKPlugin : bio_ik_kinematics_plugin::BioIKKinematicsPlugin
{
  std::unordered_map<std::string, size_t> joint_name_to_idx_;
  std::unordered_map<std::string, size_t> variable_name_to_seed_idx_;
  std::vector<GenieCoupledConstraint> coupled_constraints_;
  double coupled_margin_{0.1};
  double center_joints_w_{0.1};
  double avoid_limits_w_{0.2};
  double minimal_disp_w_{0.5};
  double coupled_goal_w_{10.0};

  // Per-group "human-like posture" goals.
  //
  // Static torso posture (added once to default_goals at initialize):
  //   posture_joints   : list of joint variable names (e.g. body_link4/5)
  //   posture_targets  : list of target joint values (same length)
  //   posture_weight   : scalar; if <=0 no goal is added.
  //
  // Transient chassis pin (added per searchPositionIK call, pinned to seed):
  //   chassis_posture_joints : list of planar variable names (e.g.
  //                            planar_joint/x, planar_joint/y, planar_joint/theta)
  //   chassis_posture_weight : scalar; if <=0 or list empty, skipped.
  //
  // Transient lookat (added per call, target read from a robot link):
  //   lookat_link          : link to aim (e.g. head_link3)
  //   lookat_axis_xyz      : 3 doubles (link-frame axis to point at target)
  //   lookat_target_link   : robot link to look at (e.g. arm_l_end_link).
  //                          If empty OR the link is missing, the goal is
  //                          NOT added (free constraint = "nothing to look at").
  //   lookat_weight        : scalar; if <=0 the goal is not added.
  //
  // Application code can ALWAYS override / supplement these by passing
  // a BioIKKinematicsQueryOptions with explicit goals; the plugin's
  // transient injections are appended to that list, never replace it.
  std::vector<std::string> posture_joints_;
  std::vector<double> posture_targets_;
  double posture_weight_{0.0};

  std::vector<std::string> chassis_posture_joints_;
  double chassis_posture_weight_{0.0};

  std::string lookat_link_;
  std::array<double, 3> lookat_axis_{{1.0, 0.0, 0.0}};
  std::string lookat_target_link_;
  double lookat_weight_{0.0};

  bool initialize(
    const rclcpp::Node::SharedPtr & node,
    const moveit::core::RobotModel & robot_model,
    const std::string & group_name,
    const std::string & base_frame,
    const std::vector<std::string> & tip_frames,
    double search_discretization) override
  {
    if (!BioIKKinematicsPlugin::initialize(
        node, robot_model, group_name,
        base_frame, tip_frames,
        search_discretization))
    {
      return false;
    }

    const auto * jmg = robot_model.getJointModelGroup(group_name);
    if (jmg) {
      const auto & names = jmg->getActiveJointModelNames();
      for (size_t i = 0; i < names.size(); ++i) {
        joint_name_to_idx_[names[i]] = i;
      }
    }

    // Variable-level index (for multi-DOF joints like planar_joint that
    // expand into x / y / theta variables). Mirrors the seed layout used
    // by BioIKKinematicsPlugin::searchPositionIK: iterate getJointNames()
    // in order and concatenate each joint's variable values.
    {
      size_t cursor = 0;
      for (const auto & jname : getJointNames()) {
        const auto * jm = robot_model.getJointModel(jname);
        if (!jm) {continue;}
        for (const auto & vname : jm->getVariableNames()) {
          variable_name_to_seed_idx_[vname] = cursor++;
        }
      }
    }

    auto gp = [&](const std::string & name, auto def) {
        using T = decltype(def);
        T val = def;
        if (node->has_parameter(name)) {
          node->get_parameter(name, val);
        } else {
          try {node->declare_parameter(name, def);} catch (...) {}
        }
        return val;
      };

    coupled_margin_ = gp("coupled_constraint_margin", 0.1);
    center_joints_w_ = gp("bio_ik_center_joints_weight", 0.1);
    avoid_limits_w_ = gp("bio_ik_avoid_limits_weight", 0.2);
    minimal_disp_w_ = gp("bio_ik_minimal_displacement_weight", 0.5);
    coupled_goal_w_ = gp("bio_ik_coupled_goal_weight", 10.0);

    // Per-group posture / lookat configuration. lookupParam scopes parameter
    // names to the kinematics yaml entry of `group_name`, so different groups
    // (mobile_base_manipulator vs wbc_*) can declare different postures.
    auto lookup_str_vec = [&](const std::string & key) {
        std::vector<std::string> v;
        lookupParam(node, key, v, v);
        return v;
      };
    auto lookup_dbl_vec = [&](const std::string & key) {
        std::vector<double> v;
        lookupParam(node, key, v, v);
        return v;
      };

    posture_joints_ = lookup_str_vec("posture_joints");
    posture_targets_ = lookup_dbl_vec("posture_targets");
    {
      double w = 0.0;
      lookupParam(node, std::string("posture_weight"), w, w);
      posture_weight_ = w;
    }

    chassis_posture_joints_ = lookup_str_vec("chassis_posture_joints");
    {
      double w = 0.0;
      lookupParam(node, std::string("chassis_posture_weight"), w, w);
      chassis_posture_weight_ = w;
    }

    {
      std::string s;
      lookupParam(node, std::string("lookat_link"), s, s);
      lookat_link_ = s;
    }
    {
      std::vector<double> axis;
      lookupParam(node, std::string("lookat_axis_xyz"), axis, axis);
      if (axis.size() == 3) {
        lookat_axis_ = {axis[0], axis[1], axis[2]};
      }
    }
    {
      std::string s;
      lookupParam(node, std::string("lookat_target_link"), s, s);
      lookat_target_link_ = s;
    }
    {
      double w = 0.0;
      lookupParam(node, std::string("lookat_weight"), w, w);
      lookat_weight_ = w;
    }

    // Static torso-posture goal: added once, lives for the plugin lifetime.
    // Use secondary=true so it never trumps PoseGoal convergence; it just
    // biases redundancy resolution toward "torso straight".
    if (posture_weight_ > 0.0 &&
      !posture_joints_.empty() &&
      posture_joints_.size() == posture_targets_.size())
    {
      bool all_present = true;
      for (const auto & jn : posture_joints_) {
        if (joint_name_to_idx_.find(jn) == joint_name_to_idx_.end()) {
          RCLCPP_WARN(
            node->get_logger(),
            "[GenieBioIK] posture_joints entry '%s' not in group '%s'; "
            "skipping posture goal.",
            jn.c_str(), group_name.c_str());
          all_present = false;
          break;
        }
      }
      if (all_present) {
        for (size_t i = 0; i < posture_joints_.size(); ++i) {
          default_goals.push_back(
            std::make_unique<bio_ik::JointVariableGoal>(
              posture_joints_[i], posture_targets_[i],
              posture_weight_, /*secondary=*/ true));
        }
      }
    }

    loadCoupledConstraints(node);

    for (const auto & cc : coupled_constraints_) {
      default_goals.push_back(
        std::make_unique<CoupledBoundGoal>(
          cc.joint_names, cc.A, cc.b, coupled_margin_, coupled_goal_w_));
    }

    RCLCPP_INFO(
      node->get_logger(),
      "[GenieBioIK] Initialized for '%s' with %zu coupled constraints "
      "(center=%.2f, limits=%.2f, disp=%.2f, coupled=%.2f, posture=%.2f x%zu, "
      "chassis_pin=%.2f x%zu, lookat=%.2f link='%s'->'%s')",
      group_name.c_str(), coupled_constraints_.size(),
      center_joints_w_, avoid_limits_w_, minimal_disp_w_, coupled_goal_w_,
      posture_weight_, posture_joints_.size(),
      chassis_posture_weight_, chassis_posture_joints_.size(),
      lookat_weight_, lookat_link_.c_str(), lookat_target_link_.c_str());
    return true;
  }

  // RAII guard that pushes per-call transient goals onto default_goals
  // (chassis pinned to the IK seed, optional head LookAt) and pops them
  // when the call finishes. default_goals is a `mutable` member of
  // BioIKKinematicsPlugin and is consumed inside searchPositionIK by
  // borrowing raw pointers, so appending here is safe within a single
  // searchPositionIK invocation. NOT thread-safe; rely on MoveIt's
  // per-group plugin instance + caller-side serialization (the planning
  // pipeline already serializes IK calls per group).
  class TransientGoalGuard
  {
public:
    TransientGoalGuard(
      const BioIKPlugin & owner,
      const std::vector<double> & ik_seed_state)
    : owner_(owner)
    {
      orig_size_ = owner_.default_goals.size();

      // Chassis-pin goal: pin every chassis variable to its seed value.
      // Mirrors PickNik StretchKinematicsPlugin's "arm-first then base"
      // fallback by making BioIK pay a quadratic cost for moving the
      // chassis; the arms have to fail to reach before the optimizer
      // moves the base.
      if (owner_.chassis_posture_weight_ > 0.0 &&
        !owner_.chassis_posture_joints_.empty())
      {
        for (const auto & vname : owner_.chassis_posture_joints_) {
          auto it = owner_.variable_name_to_seed_idx_.find(vname);
          if (it == owner_.variable_name_to_seed_idx_.end()) {continue;}
          if (it->second >= ik_seed_state.size()) {continue;}
          owner_.default_goals.push_back(
            std::make_unique<bio_ik::JointVariableGoal>(
              vname, ik_seed_state[it->second],
              owner_.chassis_posture_weight_, /*secondary=*/ false));
        }
      }

      // LookAt goal: aim lookat_link's lookat_axis at lookat_target_link's
      // current position. The target position is read from a robot link
      // (e.g. arm_l_end_link) via RobotState::getFrameTransform on the
      // seed state, so the head tracks the active arm tip without any
      // tf2_ros buffer / spin. Empty target_link or missing link => no
      // goal (free constraint, "nothing to look at").
      if (owner_.lookat_weight_ > 0.0 &&
        !owner_.lookat_link_.empty() &&
        !owner_.lookat_target_link_.empty() &&
        owner_.robot_model_->hasLinkModel(owner_.lookat_target_link_))
      {
        moveit::core::RobotState rs(owner_.robot_model_);
        rs.setToDefaultValues();
        // Populate seed values into the RobotState. ik_seed_state is in
        // group variable order; mirror the layout used by the parent
        // searchPositionIK.
        size_t cursor = 0;
        for (const auto & jname : owner_.getJointNames()) {
          const auto * jm = owner_.robot_model_->getJointModel(jname);
          if (!jm) {continue;}
          for (size_t vi = 0; vi < jm->getVariableCount(); ++vi) {
            if (cursor < ik_seed_state.size()) {
              rs.setVariablePosition(
                jm->getFirstVariableIndex() + vi, ik_seed_state[cursor]);
            }
            ++cursor;
          }
        }
        rs.update();
        const Eigen::Isometry3d & target_tf = rs.getFrameTransform(
          owner_.lookat_target_link_);
        const Eigen::Vector3d t = target_tf.translation();
        owner_.default_goals.push_back(
          std::make_unique<bio_ik::LookAtGoal>(
            owner_.lookat_link_,
            tf2::Vector3(
              owner_.lookat_axis_[0],
              owner_.lookat_axis_[1],
              owner_.lookat_axis_[2]),
            tf2::Vector3(t.x(), t.y(), t.z()),
            owner_.lookat_weight_));
      }
    }

    ~TransientGoalGuard()
    {
      while (owner_.default_goals.size() > orig_size_) {
        owner_.default_goals.pop_back();
      }
    }

    TransientGoalGuard(const TransientGoalGuard &) = delete;
    TransientGoalGuard & operator=(const TransientGoalGuard &) = delete;

private:
    const BioIKPlugin & owner_;
    size_t orig_size_;
  };

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & options =
    kinematics::KinematicsQueryOptions()) const override
  {
    IKCallbackFn wrapped = makeCoupledCallback(IKCallbackFn());
    auto padded = padPoses(ik_pose, ik_seed_state);
    TransientGoalGuard guard(*this, ik_seed_state);
    return BioIKKinematicsPlugin::searchPositionIK(
      padded, ik_seed_state, timeout, std::vector<double>(),
      solution, wrapped, error_code, options);
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
    IKCallbackFn wrapped = makeCoupledCallback(IKCallbackFn());
    auto padded = padPoses(ik_pose, ik_seed_state);
    TransientGoalGuard guard(*this, ik_seed_state);
    return BioIKKinematicsPlugin::searchPositionIK(
      padded, ik_seed_state, timeout, consistency_limits,
      solution, wrapped, error_code, options);
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
    IKCallbackFn wrapped = makeCoupledCallback(solution_callback);
    auto padded = padPoses(ik_pose, ik_seed_state);
    TransientGoalGuard guard(*this, ik_seed_state);
    return BioIKKinematicsPlugin::searchPositionIK(
      padded, ik_seed_state, timeout, std::vector<double>(),
      solution, wrapped, error_code, options);
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
    IKCallbackFn wrapped = makeCoupledCallback(solution_callback);
    auto padded = padPoses(ik_pose, ik_seed_state);
    TransientGoalGuard guard(*this, ik_seed_state);
    return BioIKKinematicsPlugin::searchPositionIK(
      padded, ik_seed_state, timeout, consistency_limits,
      solution, wrapped, error_code, options);
  }

private:
  std::vector<geometry_msgs::msg::Pose> padPoses(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state) const
  {
    size_t n_tips = tip_frames_.size();
    if (n_tips <= 1) {
      return {ik_pose};
    }

    std::vector<double> full_state(robot_model_->getVariableCount());
    robot_model_->getVariableDefaultPositions(full_state.data());
    {
      size_t idx = 0;
      for (const auto & jname : getJointNames()) {
        auto * jm = robot_model_->getJointModel(jname);
        if (!jm) {continue;}
        for (size_t vi = 0; vi < jm->getVariableCount(); ++vi) {
          full_state[jm->getFirstVariableIndex() + vi] = ik_seed_state[idx++];
        }
      }
    }

    auto rs = std::make_shared<moveit::core::RobotState>(robot_model_);
    rs->setVariablePositions(full_state);
    rs->update();

    std::vector<geometry_msgs::msg::Pose> poses(n_tips);
    poses[0] = ik_pose;
    // getFrameTransform() handles virtual model frames (e.g. SRDF planar
    // virtual_joint parent_frame="map") by returning identity, which is
    // the mathematically correct value (the model frame IS the global
    // frame). Mirrors PickNik stretch_kinematics_plugin pattern.
    for (size_t i = 1; i < n_tips; ++i) {
      const Eigen::Isometry3d & base_tf = rs->getFrameTransform(getBaseFrame());
      const Eigen::Isometry3d & tip_tf = rs->getFrameTransform(tip_frames_[i]);
      Eigen::Isometry3d rel = base_tf.inverse() * tip_tf;
      poses[i] = tf2::toMsg(rel);
    }
    return poses;
  }

  bool checkCoupledConstraints(const std::vector<double> & jv) const
  {
    for (const auto & cc : coupled_constraints_) {
      size_t nr = static_cast<size_t>(cc.A.rows());
      size_t nc = static_cast<size_t>(cc.A.cols());
      Eigen::VectorXd q(nc);
      for (size_t c = 0; c < nc; ++c) {
        auto it = joint_name_to_idx_.find(cc.joint_names[c]);
        if (it == joint_name_to_idx_.end()) {return false;}
        q[c] = jv[it->second];
      }
      Eigen::VectorXd s = cc.A * q;
      for (size_t r = 0; r < nr; ++r) {
        if (s[r] > cc.b[r] - coupled_margin_) {return false;}}
    }
    return true;
  }

  IKCallbackFn makeCoupledCallback(const IKCallbackFn & user_cb) const
  {
    if (coupled_constraints_.empty()) {return user_cb;}
    return [this, user_cb](const geometry_msgs::msg::Pose & p,
             const std::vector<double> & j,
             moveit_msgs::msg::MoveItErrorCodes & ec) {
             if (!checkCoupledConstraints(j)) {
               ec.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
               return;
             }
             if (user_cb) {user_cb(p, j, ec);} else {
               ec.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
             }
           };
  }

  void loadCoupledConstraints(const rclcpp::Node::SharedPtr & node)
  {
    coupled_constraints_.clear();
    std::string config_path;
    if (node->has_parameter("coupled_constraints_file")) {
      node->get_parameter("coupled_constraints_file", config_path);
    } else {
      try {node->declare_parameter("coupled_constraints_file", std::string(""));} catch (...) {}
      node->get_parameter("coupled_constraints_file", config_path);
    }

    if (config_path.empty()) {
      try {
        config_path = ament_index_cpp::get_package_share_directory(
          "genie_sim_moveit_plugins") + "/config/coupled_constraints.yaml";
      } catch (...) {return;}
    }

    std::ifstream ifs(config_path);
    if (!ifs.is_open()) {return;}

    try {
      YAML::Node root = YAML::LoadFile(config_path);
      if (!root["coupled_constraints"]) {return;}

      for (const auto & entry : root["coupled_constraints"]) {
        auto jnames = entry["joint_names"].as<std::vector<std::string>>();
        auto bvec = entry["b"].as<std::vector<double>>();
        auto Arows = entry["A"];

        bool ok = true;
        for (const auto & jn : jnames) {
          if (joint_name_to_idx_.find(jn) == joint_name_to_idx_.end()) {ok = false; break;}}
        if (!ok || jnames.empty()) {continue;}

        size_t nr = Arows.size(), nc = jnames.size();
        GenieCoupledConstraint cc;
        cc.joint_names = jnames;
        cc.A = Eigen::MatrixXd::Zero(nr, nc);
        cc.b = Eigen::VectorXd::Zero(nr);
        for (size_t r = 0; r < nr; ++r) {
          auto row = Arows[r].as<std::vector<double>>();
          for (size_t c = 0; c < nc && c < row.size(); ++c) {
            cc.A(r, c) = row[c];
          }
          cc.b[r] = bvec[r];
        }
        coupled_constraints_.push_back(cc);
      }
    } catch (...) {}
  }
};

}  // namespace genie_sim_moveit_plugins

#undef LOG
#undef ERROR
PLUGINLIB_EXPORT_CLASS(
  bio_ik_kinematics_plugin::BioIKKinematicsPlugin,
  kinematics::KinematicsBase);
PLUGINLIB_EXPORT_CLASS(
  genie_sim_moveit_plugins::BioIKPlugin,
  kinematics::KinematicsBase);
