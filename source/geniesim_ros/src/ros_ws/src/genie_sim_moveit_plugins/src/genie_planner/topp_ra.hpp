#pragma once

#include <Eigen/Dense>
#include <vector>

namespace genie_sim_moveit_plugins
{

struct ToppResult
{
  bool success{false};
  double duration{0.0};
  std::vector<Eigen::VectorXd> positions;
  std::vector<Eigen::VectorXd> velocities;
  std::vector<Eigen::VectorXd> accelerations;
  double dt{0.0};
  std::string error_msg;
};

ToppResult toppRA(
  const std::vector<Eigen::VectorXd> & waypoints,
  const Eigen::VectorXd & vel_limits,
  const Eigen::VectorXd & acc_limits,
  double output_dt);

std::vector<Eigen::VectorXd> butterworthFilter(
  const std::vector<Eigen::VectorXd> & positions,
  double cutoff_freq, double sample_freq, int buffer_size);

}  // namespace genie_sim_moveit_plugins
