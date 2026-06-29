#include "genie_sim_moveit_plugins/moveit_compat.hpp"
#include MOVEIT_H_KINEMATICS_BASE
#include MOVEIT_H_JOINT_MODEL_GROUP
#include MOVEIT_H_LINK_MODEL
#include MOVEIT_H_ROBOT_STATE
#include MOVEIT_H_COLLISION_COMMON
#include MOVEIT_H_COLLISION_ENV_FCL
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

#include "genie_sim_moveit_plugins/coupled_constraints.hpp"

#include <Eigen/SVD>
#include <cmath>
#include <chrono>

namespace genie_sim_moveit_plugins
{

struct RelaxedIKOptions
{
  int ref_type = 1;
  bool use_groove_wrapper = false;

  double position_cost = 60.0;
  double rotation_cost = 6.0;
  double ref_rotation_cost = 0.15;

  double arm_ik_speed_scale = 4.0;
  int discontinuous_level = 2;

  double damp_v_cost = 8.0;
  double damp_a_cost = 5.0;
  double damp_j_cost = 0.0;

  double regular_cost = 0.01;

  double coupled_margin = 0.1;
  double coupled_penalty = 1000.0;

  double singularity_cost = 0.0;
  double singularity_lambda = 2.0;

  double bound_penalty = 1000.0;
  double bound_tolerance = 0.05;

  double collision_penalty = 1000.0;
  double collision_margin = 0.02;

  double balance_penalty = 10000.0;
  double balance_margin = 0.01;

  double pos_tolerance = 0.01;
  double ori_tolerance = 0.1;

  int max_iter = 100;
  double damping_lambda = 0.005;
  double seed_pull_weight = 2.0;
};

struct LinkMassInfo
{
  double mass;
  Eigen::Vector3d com_local;
  std::string link_name;
};

class GenieRelaxedIKPlugin : public kinematics::KinematicsBase
{
public:
  GenieRelaxedIKPlugin() = default;

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
    jmg_ = robot_model.getJointModelGroup(group_name);
    if (!jmg_) {return false;}

    ndof_ = jmg_->getVariableCount();
    const auto & bounds = jmg_->getActiveJointModelsBounds();
    lower_.resize(ndof_);
    upper_.resize(ndof_);
    vel_upper_.resize(ndof_);
    size_t idx = 0;
    for (const auto & jb : bounds) {
      for (const auto & b : *jb) {
        lower_[idx] = b.min_position_;
        upper_[idx] = b.max_position_;
        vel_upper_[idx] = b.velocity_bounded_ ? b.max_velocity_ : 3.14;
        ++idx;
      }
    }
    q_mid_ = (lower_ + upper_) / 2.0;
    q_range_ = upper_ - lower_;

    loadOptions(node);
    coupled_constraints_ = loadCoupledConstraints(
      node, activeJointIndex(jmg_),
      "relaxed_ik.coupled_constraints_file", "[GenieRelaxedIK]");
    initCollisionEnv(robot_model);
    // [PARTIAL PORT] CoM balance: uses URDF inertial data, not Pinocchio
    initMassInfo(robot_model);

    state_pre_0_ = q_mid_;
    state_pre_1_ = q_mid_;
    state_pre_2_ = q_mid_;
    state_initialized_ = false;

    RCLCPP_INFO(
      node->get_logger(),
      "[GenieRelaxedIK] Initialized for '%s' (%zu DOF, tip='%s') "
      "collision=%s balance=%s coupled=%zu",
      group_name.c_str(), ndof_, tip_frames[0].c_str(),
      opts_.collision_penalty > 1.0 ? "ON" : "OFF",
      opts_.balance_penalty > 1.0 ? "ON" : "OFF",
      coupled_constraints_.size());
    return true;
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & = kinematics::KinematicsQueryOptions()) const
  override
  {
    return solveIK(
      ik_pose, ik_seed_state, timeout, solution,
      IKCallbackFn(), error_code);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    const std::vector<double> &,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & = kinematics::KinematicsQueryOptions()) const
  override
  {
    return solveIK(
      ik_pose, ik_seed_state, timeout, solution,
      IKCallbackFn(), error_code);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    std::vector<double> & solution,
    const IKCallbackFn & solution_callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & = kinematics::KinematicsQueryOptions()) const
  override
  {
    return solveIK(
      ik_pose, ik_seed_state, timeout, solution,
      solution_callback, error_code);
  }

  bool searchPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    const std::vector<double> &,
    std::vector<double> & solution,
    const IKCallbackFn & solution_callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & = kinematics::KinematicsQueryOptions()) const
  override
  {
    return solveIK(
      ik_pose, ik_seed_state, timeout, solution,
      solution_callback, error_code);
  }

  bool getPositionIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    std::vector<double> & solution,
    moveit_msgs::msg::MoveItErrorCodes & error_code,
    const kinematics::KinematicsQueryOptions & = kinematics::KinematicsQueryOptions()) const
  override
  {
    return solveIK(
      ik_pose, ik_seed_state, 0.05, solution,
      IKCallbackFn(), error_code);
  }

  bool getPositionFK(
    const std::vector<std::string> & link_names,
    const std::vector<double> & joint_angles,
    std::vector<geometry_msgs::msg::Pose> & poses) const override
  {
    moveit::core::RobotState state(robot_model_);
    state.setJointGroupPositions(jmg_, joint_angles.data());
    state.update();
    poses.resize(link_names.size());
    for (size_t i = 0; i < link_names.size(); ++i) {
      poses[i] = tf2::toMsg(state.getGlobalLinkTransform(link_names[i]));
    }
    return true;
  }

  const std::vector<std::string> & getJointNames() const override
  {
    return jmg_->getActiveJointModelNames();
  }

  const std::vector<std::string> & getLinkNames() const override
  {
    return jmg_->getLinkModelNames();
  }

private:
  // ---------------------------------------------------------------
  // Initialization helpers
  // ---------------------------------------------------------------
  void loadOptions(const rclcpp::Node::SharedPtr & node)
  {
    auto get = [&](const std::string & name, auto & val) {
        val = declareOrGet(node, "relaxed_ik." + name, val);
      };
    get("ref_type", opts_.ref_type);
    get("use_groove_wrapper", opts_.use_groove_wrapper);
    get("position_cost", opts_.position_cost);
    get("rotation_cost", opts_.rotation_cost);
    get("ref_rotation_cost", opts_.ref_rotation_cost);
    get("arm_ik_speed_scale", opts_.arm_ik_speed_scale);
    get("discontinuous_level", opts_.discontinuous_level);
    get("damp_v_cost", opts_.damp_v_cost);
    get("damp_a_cost", opts_.damp_a_cost);
    get("damp_j_cost", opts_.damp_j_cost);
    get("regular_cost", opts_.regular_cost);
    get("coupled_margin", opts_.coupled_margin);
    get("coupled_penalty", opts_.coupled_penalty);
    get("singularity_cost", opts_.singularity_cost);
    get("singularity_lambda", opts_.singularity_lambda);
    get("bound_penalty", opts_.bound_penalty);
    get("bound_tolerance", opts_.bound_tolerance);
    get("collision_penalty", opts_.collision_penalty);
    get("collision_margin", opts_.collision_margin);
    get("balance_penalty", opts_.balance_penalty);
    get("balance_margin", opts_.balance_margin);
    get("pos_tolerance", opts_.pos_tolerance);
    get("ori_tolerance", opts_.ori_tolerance);
    get("max_iter", opts_.max_iter);
    get("damping_lambda", opts_.damping_lambda);
    get("seed_pull_weight", opts_.seed_pull_weight);
  }

  void initCollisionEnv(const moveit::core::RobotModel &)
  {
    if (opts_.collision_penalty < 1.0) {return;}
    try {
      collision_env_ = std::make_shared<collision_detection::CollisionEnvFCL>(robot_model_);
    } catch (const std::exception & e) {
      if (node_) {
        RCLCPP_WARN(
          node_->get_logger(),
          "[GenieRelaxedIK] CollisionEnvFCL init failed: %s", e.what());
      }
    }
  }

  // [PARTIAL PORT] CoM computed from URDF link inertial data via MoveIt2 LinkModel.
  // Original uses Pinocchio getCenterOfMassPosition()/getCenterOfMassJacobian()
  // which accounts for full tree. This version only considers links in the active group.
  void initMassInfo(const moveit::core::RobotModel & robot_model)
  {
    mass_info_.clear();
    total_mass_ = 0.0;
    if (opts_.balance_penalty < 1.0) {return;}

    const auto & urdf = robot_model.getURDF();
    if (!urdf) {return;}

    for (const auto & link_model : robot_model.getLinkModels()) {
      auto urdf_link = urdf->getLink(link_model->getName());
      if (!urdf_link || !urdf_link->inertial) {continue;}
      double m = urdf_link->inertial->mass;
      if (m < 1e-6) {continue;}
      Eigen::Vector3d com_local(
        urdf_link->inertial->origin.position.x,
        urdf_link->inertial->origin.position.y,
        urdf_link->inertial->origin.position.z);
      mass_info_.push_back({m, com_local, link_model->getName()});
      total_mass_ += m;
    }
  }

  // ---------------------------------------------------------------
  // Rotation error (axis-angle)
  // ---------------------------------------------------------------
  static Eigen::Vector3d rotationError(
    const Eigen::Matrix3d & R_desired,
    const Eigen::Matrix3d & R_current)
  {
    Eigen::Matrix3d R_err = R_desired * R_current.transpose();
    double trace = R_err.trace();
    double cos_angle = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    double angle = std::acos(cos_angle);

    if (std::abs(angle) < 1e-10) {
      return Eigen::Vector3d::Zero();
    }

    Eigen::Vector3d axis;
    axis << R_err(2, 1) - R_err(1, 2),
      R_err(0, 2) - R_err(2, 0),
      R_err(1, 0) - R_err(0, 1);
    double norm = axis.norm();
    if (norm < 1e-10) {
      return Eigen::Vector3d::Zero();
    }

    return axis * (angle / norm);
  }

  // ---------------------------------------------------------------
  // Collision check (boolean, no FD Jacobian)
  // ---------------------------------------------------------------
  // [PARTIAL PORT] Uses MoveIt2 FCL distanceSelf for a scalar minimum distance.
  // Original uses Pinocchio computeCollisionLinearApproximation() with per-pair
  // analytical Jacobians. Here we only check the scalar distance and return it
  // (no gradient). Used as a soft guard: if solution is in collision, we pull
  // back toward the seed via nullspace instead of computing expensive FD grads.
  double checkCollisionDistance(moveit::core::RobotState & state) const
  {
    if (!collision_env_) {return 1.0;}

    collision_detection::DistanceRequest req;
    req.enable_nearest_points = false;
    req.enable_signed_distance = true;
    req.type = collision_detection::DistanceRequestType::GLOBAL;
    collision_detection::DistanceResult res;
    collision_env_->distanceSelf(req, res, state);
    return res.minimum_distance.distance;
  }

  // ---------------------------------------------------------------
  // CoM balance check (scalar, no FD Jacobian)
  // ---------------------------------------------------------------
  // [PARTIAL PORT] Computes CoM from URDF link masses and MoveIt2 FK transforms.
  // Original uses Pinocchio getCenterOfMassJacobian() for analytical gradients.
  // Here we only compute the scalar CoM position for a soft guard check.
  Eigen::Vector2d computeCoM(moveit::core::RobotState & state) const
  {
    Eigen::Vector3d com = Eigen::Vector3d::Zero();
    for (const auto & info : mass_info_) {
      const Eigen::Isometry3d & T = state.getGlobalLinkTransform(info.link_name);
      Eigen::Vector3d world_com = T * info.com_local;
      com += info.mass * world_com;
    }
    if (total_mass_ > 1e-6) {
      com /= total_mass_;
    }
    return com.head<2>();
  }

  // ---------------------------------------------------------------
  // Coupled constraint cost
  // ---------------------------------------------------------------
  double computeCoupledCost(
    const Eigen::VectorXd & q,
    Eigen::VectorXd & grad) const
  {
    if (coupled_constraints_.empty()) {grad.setZero(); return 0.0;}

    double total_cost = 0.0;
    grad.setZero();

    for (const auto & cc : coupled_constraints_) {
      size_t n_rows = cc.A.rows();
      size_t n_cols = cc.A.cols();

      Eigen::VectorXd q_sub(n_cols);
      for (size_t c = 0; c < n_cols; ++c) {
        q_sub[c] = q[cc.group_indices[c]];
      }

      Eigen::VectorXd s = cc.A * q_sub;

      for (size_t r = 0; r < n_rows; ++r) {
        double violation = s[r] - cc.b[r] + opts_.coupled_margin;
        if (violation > 0.0) {
          total_cost += violation * violation;
          for (size_t c = 0; c < n_cols; ++c) {
            grad[cc.group_indices[c]] += 2.0 * violation * cc.A(r, c);
          }
        }
      }
    }

    return total_cost;
  }

  // ---------------------------------------------------------------
  // Main IK solver
  // ---------------------------------------------------------------
  bool solveIK(
    const geometry_msgs::msg::Pose & ik_pose,
    const std::vector<double> & ik_seed_state,
    double timeout,
    std::vector<double> & solution,
    const IKCallbackFn & solution_callback,
    moveit_msgs::msg::MoveItErrorCodes & error_code) const
  {
    Eigen::Isometry3d target;
    tf2::fromMsg(ik_pose, target);

    solution = ik_seed_state;
    Eigen::Map<Eigen::VectorXd> q(solution.data(), ndof_);
    Eigen::VectorXd q_seed = q;

    if (!state_initialized_) {
      state_pre_0_ = q_seed;
      state_pre_1_ = q_seed;
      state_pre_2_ = q_seed;
      state_initialized_ = true;
    }

    double target_dt = std::max(0.01, 0.02) * opts_.discontinuous_level;
    Eigen::VectorXd half_range = vel_upper_ * target_dt * opts_.arm_ik_speed_scale;

    Eigen::VectorXd center = state_pre_0_;
    for (size_t i = 0; i < ndof_; ++i) {
      double d = std::abs(q_seed[i] - state_pre_0_[i]);
      if (d > half_range[i]) {
        center[i] = q_seed[i];
      }
    }

    Eigen::VectorXd lb_current = lower_.cwiseMax(center - half_range);
    Eigen::VectorXd ub_current = upper_.cwiseMin(center + half_range);

    for (size_t i = 0; i < ndof_; ++i) {
      q[i] = std::clamp(q[i], lb_current[i], ub_current[i]);
    }

    Eigen::VectorXd max_joint_step = half_range;

    Eigen::Vector2d com_lb(-0.15 + opts_.balance_margin, -0.10 + opts_.balance_margin);
    Eigen::Vector2d com_ub(0.15 - opts_.balance_margin, 0.10 - opts_.balance_margin);

    moveit::core::RobotState state(robot_model_);

    const double LAMBDA = opts_.damping_lambda;
    const double SEED_W = opts_.seed_pull_weight;
    constexpr double NULL_GRAD_CAP = 2.0;

    auto start = std::chrono::steady_clock::now();
    int actual_iters = 0;

    for (int iter = 0; iter < opts_.max_iter; ++iter) {
      auto now = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(now - start).count();
      if (elapsed > timeout && timeout > 0) {break;}
      actual_iters = iter + 1;

      state.setJointGroupPositions(jmg_, q.data());
      state.update();

      const Eigen::Isometry3d & T_cur = state.getGlobalLinkTransform(getTipFrame());

      Eigen::Vector3d pos_err = target.translation() - T_cur.translation();
      Eigen::Vector3d rot_err = rotationError(target.rotation(), T_cur.rotation());

      if (pos_err.norm() < 5e-5 && rot_err.norm() < 5e-4) {
        break;
      }

      Eigen::MatrixXd J = state.getJacobian(jmg_);

      Eigen::VectorXd twist(6);
      twist.head<3>() = pos_err;
      twist.tail<3>() = rot_err;

      Eigen::MatrixXd JJt = J * J.transpose();
      JJt.diagonal().array() += LAMBDA * LAMBDA;
      Eigen::MatrixXd J_pinv = J.transpose() * JJt.inverse();

      Eigen::VectorXd dq = J_pinv * twist;

      Eigen::MatrixXd I_n = Eigen::MatrixXd::Identity(ndof_, ndof_);
      Eigen::MatrixXd N = I_n - J_pinv * J;

      Eigen::VectorXd null_grad = Eigen::VectorXd::Zero(ndof_);

      if (opts_.damp_v_cost > 0.0) {
        for (size_t i = 0; i < ndof_; ++i) {
          double range_sq = q_range_[i] > 1e-6 ? q_range_[i] * q_range_[i] : 1.0;
          null_grad[i] += opts_.damp_v_cost * (state_pre_0_[i] - q[i]) / range_sq;
        }
      }

      // [PARTIAL PORT] Pulls toward zero-acceleration trajectory point.
      if (opts_.damp_a_cost > 0.0) {
        Eigen::VectorXd accel_target = 2.0 * state_pre_0_ - state_pre_1_;
        for (size_t i = 0; i < ndof_; ++i) {
          double range_sq = q_range_[i] > 1e-6 ? q_range_[i] * q_range_[i] : 1.0;
          null_grad[i] += opts_.damp_a_cost * (accel_target[i] - q[i]) / range_sq;
        }
      }

      if (opts_.regular_cost > 0.0) {
        for (size_t i = 0; i < ndof_; ++i) {
          double range_sq = q_range_[i] > 1e-6 ? q_range_[i] * q_range_[i] : 1.0;
          null_grad[i] += opts_.regular_cost * (q_mid_[i] - q[i]) / range_sq;
        }
      }

      for (size_t i = 0; i < ndof_; ++i) {
        null_grad[i] += SEED_W * (q_seed[i] - q[i]);
      }

      if (opts_.ref_rotation_cost > 0.0 && opts_.ref_type == 1) {
        Eigen::Vector3d local_z(0, 0, 1);
        Eigen::Vector3d world_z = T_cur.rotation() * local_z;
        Eigen::Vector3d target_z(0, 0, 1);
        Eigen::Vector3d ref_err = world_z - target_z;
        if (ref_err.norm() > 1e-6) {
          Eigen::MatrixXd J_rot = J.bottomRows<3>();
          Eigen::MatrixXd skew = Eigen::MatrixXd::Zero(3, 3);
          skew(0, 1) = -local_z[2]; skew(0, 2) = local_z[1];
          skew(1, 0) = local_z[2]; skew(1, 2) = -local_z[0];
          skew(2, 0) = -local_z[1]; skew(2, 1) = local_z[0];
          Eigen::MatrixXd J_ref = -T_cur.rotation() * skew * J_rot;
          null_grad -= opts_.ref_rotation_cost * J_ref.transpose() * ref_err;
        }
      }

      if (opts_.bound_penalty > 0.0) {
        for (size_t i = 0; i < ndof_; ++i) {
          double tol = opts_.bound_tolerance;
          if (q[i] < lb_current[i] + tol) {
            null_grad[i] += opts_.bound_penalty * (lb_current[i] + tol - q[i]) / double(ndof_);
          }
          if (q[i] > ub_current[i] - tol) {
            null_grad[i] += opts_.bound_penalty * (ub_current[i] - tol - q[i]) / double(ndof_);
          }
        }
      }

      if (!coupled_constraints_.empty() && opts_.coupled_penalty > 0.0) {
        Eigen::VectorXd coupled_grad = Eigen::VectorXd::Zero(ndof_);
        double coupled_cost = computeCoupledCost(q, coupled_grad);
        if (coupled_cost > 0.0) {
          double g_norm = coupled_grad.norm();
          if (g_norm > 1e-8) {
            double scale = std::min(opts_.coupled_penalty / double(ndof_), NULL_GRAD_CAP / g_norm);
            null_grad -= scale * coupled_grad;
          }
        }
      }

      // [PARTIAL PORT] Collision: single scalar check, pull toward seed.
      if (collision_env_ && opts_.collision_penalty > 1.0 && (iter % 10 == 0)) {
        double dist = checkCollisionDistance(state);
        if (dist < opts_.collision_margin) {
          double severity = (opts_.collision_margin - dist) / opts_.collision_margin;
          for (size_t i = 0; i < ndof_; ++i) {
            null_grad[i] += severity * 5.0 * (q_seed[i] - q[i]);
          }
        }
      }

      // [PARTIAL PORT] Balance: single CoM check, pull toward joint center.
      if (opts_.balance_penalty > 1.0 && !mass_info_.empty() && (iter % 10 == 0)) {
        Eigen::Vector2d com = computeCoM(state);
        bool out = false;
        for (int k = 0; k < 2; ++k) {
          if (com[k] < com_lb[k] || com[k] > com_ub[k]) {out = true; break;}
        }
        if (out) {
          for (size_t i = 0; i < ndof_; ++i) {
            null_grad[i] += 3.0 * (q_mid_[i] - q[i]);
          }
        }
      }

      double ng_norm = null_grad.norm();
      if (ng_norm > NULL_GRAD_CAP) {
        null_grad *= (NULL_GRAD_CAP / ng_norm);
      }

      dq += N * null_grad;

      for (size_t i = 0; i < ndof_; ++i) {
        dq[i] = std::clamp(dq[i], -max_joint_step[i], max_joint_step[i]);
      }

      q += dq;

      for (size_t i = 0; i < ndof_; ++i) {
        q[i] = std::clamp(q[i], lb_current[i], ub_current[i]);
      }
    }

    auto end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

    state.setJointGroupPositions(jmg_, q.data());
    state.update();
    const Eigen::Isometry3d & achieved = state.getGlobalLinkTransform(getTipFrame());
    double pos_err_final = (target.translation() - achieved.translation()).norm();
    double ori_err_final = rotationError(target.rotation(), achieved.rotation()).norm();

    bool converged = (pos_err_final <= opts_.pos_tolerance &&
      ori_err_final <= opts_.ori_tolerance);

    log_count_++;
    if (!converged && (log_count_ % 50 == 1)) {
      RCLCPP_WARN(
        node_->get_logger(),
        "[GenieRelaxedIK] FAIL #%zu | iters=%d/%.1fms timeout=%.3fs | "
        "pos_err=%.4f(tol=%.4f) ori_err=%.4f(tol=%.4f) | "
        "ndof=%zu half_range[0]=%.4f seed_drift=%.4f",
        log_count_, actual_iters, total_ms, timeout,
        pos_err_final, opts_.pos_tolerance,
        ori_err_final, opts_.ori_tolerance,
        ndof_, half_range[0],
        (q_seed - state_pre_0_).norm());
    }
    if (converged && (log_count_ % 200 == 1)) {
      RCLCPP_INFO(
        node_->get_logger(),
        "[GenieRelaxedIK] OK #%zu | iters=%d/%.1fms | "
        "pos_err=%.5f ori_err=%.5f",
        log_count_, actual_iters, total_ms,
        pos_err_final, ori_err_final);
    }

    state_pre_2_ = state_pre_1_;
    state_pre_1_ = state_pre_0_;
    state_pre_0_ = converged ? Eigen::VectorXd(q) : q_seed;

    if (!converged) {
      error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
      return false;
    }

    if (solution_callback) {
      geometry_msgs::msg::Pose achieved_pose = tf2::toMsg(achieved);
      solution_callback(achieved_pose, solution, error_code);
      if (error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        return false;
      }
    }

    error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    return true;
  }

  // ---------------------------------------------------------------
  // Members
  // ---------------------------------------------------------------
  rclcpp::Node::SharedPtr node_;
  const moveit::core::JointModelGroup * jmg_{nullptr};
  size_t ndof_{0};
  RelaxedIKOptions opts_;

  Eigen::VectorXd lower_, upper_, vel_upper_;
  Eigen::VectorXd q_mid_, q_range_;

  std::vector<CoupledConstraint> coupled_constraints_;

  std::shared_ptr<collision_detection::CollisionEnvFCL> collision_env_;

  std::vector<LinkMassInfo> mass_info_;
  double total_mass_{0.0};

  mutable Eigen::VectorXd state_pre_0_, state_pre_1_, state_pre_2_;
  mutable bool state_initialized_{false};
  mutable size_t log_count_{0};
};

}  // namespace genie_sim_moveit_plugins

PLUGINLIB_EXPORT_CLASS(
  genie_sim_moveit_plugins::GenieRelaxedIKPlugin,
  kinematics::KinematicsBase)
