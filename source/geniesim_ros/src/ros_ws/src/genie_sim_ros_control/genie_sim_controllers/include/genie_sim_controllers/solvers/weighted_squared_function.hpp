#pragma once

#include <memory>
#include <utility>
#include "genie_sim_controllers/solvers/quadratic_function.hpp"

namespace solvers
{

class WeightedSquaredFunction : public QuadraticFunction
{
public:
  WeightedSquaredFunction(const FunctionBase & general_function, const Eigen::VectorXd & weights)
  : WeightedSquaredFunction(general_function.deep_copy(), weights) {}

  WeightedSquaredFunction(
    const std::shared_ptr<FunctionBase> & general_function,
    const Eigen::VectorXd & weights)
  : QuadraticFunction(general_function, Eigen::VectorXd(weights * 2.0).asDiagonal()) {}

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<WeightedSquaredFunction>(general_function_->deep_copy(), Q_.diagonal());
  }
};

}  // namespace solvers
