#pragma once

#include <Eigen/Dense>
#include <cmath>

namespace genie_sim_moveit_plugins
{

inline Eigen::VectorXd grooveLoss(
  const Eigen::VectorXd & x, const Eigen::VectorXd & t,
  int d, double c, double f, int g)
{
  return -(-(x - t).array().pow(d) / (2.0 * c * c)).exp() +
         f * (x - t).array().pow(g);
}

inline Eigen::VectorXd grooveLossGrad(
  const Eigen::VectorXd & x, const Eigen::VectorXd & t,
  int d, double c, double f, int g)
{
  return -(-(x - t).array().pow(d) / (2.0 * c * c)).exp() *
         ((-d * (x - t)) / (2.0 * c * c)).array() +
         g * f * (x - t).array().pow(g - 1);
}

}  // namespace genie_sim_moveit_plugins
