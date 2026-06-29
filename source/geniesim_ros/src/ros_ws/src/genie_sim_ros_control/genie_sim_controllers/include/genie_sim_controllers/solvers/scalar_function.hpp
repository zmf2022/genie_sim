#pragma once

#include <utility>
#include "genie_sim_controllers/solvers/function_base.hpp"

namespace solvers
{

class ScalarFunction : public FunctionBase
{
public:
  explicit ScalarFunction(size_t input_dim, bool has_gradient = true, bool has_hessian = true)
  : FunctionBase(input_dim, 1, has_gradient, has_hessian) {}

  ~ScalarFunction() override = default;

  using GradientType = Eigen::Ref<Eigen::VectorXd>;
  using HessianType = Eigen::Ref<Eigen::MatrixXd>;

  using GradientTypeOpt = std::optional<GradientType>;
  using HessianTypeOpt = std::optional<HessianType>;

  void compute(
    FunctionBase::InputType input, std::optional<double *> output_opt,
    ScalarFunction::GradientTypeOpt gradient_opt,
    ScalarFunction::HessianTypeOpt hessian_opt)
  {
    if (gradient_opt) {
      gradient_opt->setZero();
    }
    if (hessian_opt) {
      hessian_opt->setZero();
    }
    doCompute(input, output_opt, std::move(gradient_opt), std::move(hessian_opt));
  }

  [[nodiscard]] double evaluate(FunctionBase::InputType input)
  {
    double cost = 0;
    compute(input, &cost, std::nullopt, std::nullopt);
    return cost;
  }

protected:
  virtual void doCompute(
    FunctionBase::InputType input, std::optional<double *> output_opt,
    ScalarFunction::GradientTypeOpt gradient_opt,
    ScalarFunction::HessianTypeOpt hessian_opt) = 0;

  void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt,
    FunctionBase::HessianArrayTypeOpt hessian_opt) override;
};

}  // namespace solvers
