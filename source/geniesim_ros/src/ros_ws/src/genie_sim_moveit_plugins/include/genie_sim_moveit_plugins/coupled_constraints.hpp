// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// License: Mozilla Public License Version 2.0

#pragma once

// Shared helpers for IK plugins that enforce linear inequality
// constraints across coupled joints (A·q ≤ b − margin).
//
// All three IK plugins (kdl_coupled, bio_ik_coupled, relaxed_ik) load
// the same YAML schema and run the same linear-inequality check; this
// header is the single source of truth so the plugins keep only what
// is genuinely solver-specific.

#include "genie_sim_moveit_plugins/moveit_compat.hpp"
#include MOVEIT_H_KINEMATICS_BASE
#include MOVEIT_H_JOINT_MODEL_GROUP
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace genie_sim_moveit_plugins
{

struct CoupledConstraint
{
  std::vector<size_t> group_indices;
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
};

// Build active-joint-name → group-index map for a JointModelGroup.
inline std::unordered_map<std::string, size_t> activeJointIndex(
  const moveit::core::JointModelGroup * jmg)
{
  std::unordered_map<std::string, size_t> out;
  if (!jmg) {return out;}
  const auto & names = jmg->getActiveJointModelNames();
  out.reserve(names.size());
  for (size_t i = 0; i < names.size(); ++i) {
    out[names[i]] = i;
  }
  return out;
}

// Declare-or-get a parameter with a default. Swallows redeclare
// races (e.g. a second plugin instance hitting the same node).
template<typename T>
T declareOrGet(
  const rclcpp::Node::SharedPtr & node,
  const std::string & name,
  const T & default_value)
{
  T value = default_value;
  if (node->has_parameter(name)) {
    node->get_parameter(name, value);
  } else {
    try {
      node->declare_parameter(name, default_value);
    } catch (...) {
    }
    node->get_parameter(name, value);
  }
  return value;
}

// Load coupled constraints from YAML. ``path_param_name`` is the
// full ROS parameter name to read the override path from (e.g.
// ``coupled_constraints_file`` or ``relaxed_ik.coupled_constraints_file``).
// When the parameter is empty, falls back to
// ``<share>/genie_sim_moveit_plugins/config/coupled_constraints.yaml``.
// Constraint entries that reference unknown joints are skipped.
inline std::vector<CoupledConstraint> loadCoupledConstraints(
  const rclcpp::Node::SharedPtr & node,
  const std::unordered_map<std::string, size_t> & joint_name_to_idx,
  const std::string & path_param_name,
  const std::string & log_tag)
{
  std::vector<CoupledConstraint> out;
  std::string config_path = declareOrGet<std::string>(node, path_param_name, "");

  if (config_path.empty()) {
    try {
      config_path =
        ament_index_cpp::get_package_share_directory("genie_sim_moveit_plugins") +
        "/config/coupled_constraints.yaml";
    } catch (...) {
      return out;
    }
  }

  std::ifstream ifs(config_path);
  if (!ifs.is_open()) {
    RCLCPP_WARN(
      node->get_logger(),
      "%s No coupled constraints file: %s", log_tag.c_str(), config_path.c_str());
    return out;
  }

  try {
    YAML::Node root = YAML::LoadFile(config_path);
    if (!root["coupled_constraints"]) {return out;}

    for (const auto & entry : root["coupled_constraints"]) {
      auto joint_names = entry["joint_names"].as<std::vector<std::string>>();
      auto b_vec = entry["b"].as<std::vector<double>>();
      auto A_rows = entry["A"];

      std::vector<size_t> group_idx;
      group_idx.reserve(joint_names.size());
      bool all_found = true;
      for (const auto & jn : joint_names) {
        auto it = joint_name_to_idx.find(jn);
        if (it == joint_name_to_idx.end()) {all_found = false; break;}
        group_idx.push_back(it->second);
      }
      if (!all_found || group_idx.empty()) {continue;}

      size_t n_rows = A_rows.size();
      size_t n_cols = joint_names.size();
      CoupledConstraint cc;
      cc.group_indices = std::move(group_idx);
      cc.A = Eigen::MatrixXd::Zero(n_rows, n_cols);
      cc.b = Eigen::VectorXd::Zero(n_rows);
      for (size_t r = 0; r < n_rows; ++r) {
        auto row = A_rows[r].as<std::vector<double>>();
        for (size_t c = 0; c < n_cols && c < row.size(); ++c) {
          cc.A(r, c) = row[c];
        }
        cc.b[r] = b_vec[r];
      }
      out.push_back(std::move(cc));
    }
  } catch (const std::exception & e) {
    RCLCPP_WARN(
      node->get_logger(),
      "%s Failed to parse coupled constraints: %s", log_tag.c_str(), e.what());
  }

  return out;
}

// Returns false on first violation: ``A·q_sub > b − margin`` for any row.
inline bool checkCoupledConstraints(
  const std::vector<CoupledConstraint> & constraints,
  const std::vector<double> & joint_values,
  double margin)
{
  for (const auto & cc : constraints) {
    const size_t n_cols = static_cast<size_t>(cc.A.cols());
    Eigen::VectorXd q_sub(n_cols);
    for (size_t c = 0; c < n_cols; ++c) {
      q_sub[c] = joint_values[cc.group_indices[c]];
    }
    Eigen::VectorXd s = cc.A * q_sub;
    for (Eigen::Index r = 0; r < s.size(); ++r) {
      if (s[r] > cc.b[r] - margin) {return false;}
    }
  }
  return true;
}

// Wrap a user IKCallbackFn so coupled-constraint violations short-circuit
// to NO_IK_SOLUTION before the user callback runs. Returns ``user_cb``
// unchanged when there are no constraints (avoids a needless allocation).
//
// ``constraints`` is captured by reference: the caller must keep the
// vector alive for as long as the returned callback is in use. In
// practice the plugin owns the vector and the wrapper is consumed
// within a single searchPositionIK call, so this is always safe.
inline kinematics::KinematicsBase::IKCallbackFn wrapWithCoupledCheck(
  const std::vector<CoupledConstraint> & constraints,
  double margin,
  kinematics::KinematicsBase::IKCallbackFn user_cb)
{
  if (constraints.empty()) {return user_cb;}

  return [&constraints, margin, user_cb = std::move(user_cb)](
    const geometry_msgs::msg::Pose & pose,
    const std::vector<double> & joints,
    moveit_msgs::msg::MoveItErrorCodes & error_code) {
           if (!checkCoupledConstraints(constraints, joints, margin)) {
             error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
             return;
           }
           if (user_cb) {
             user_cb(pose, joints, error_code);
           } else {
             error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
           }
         };
}

}  // namespace genie_sim_moveit_plugins
