#include "genie_sim_controllers/solvers/quadratic_function.hpp"

namespace solvers
{

void QuadraticFunction::doCompute(
  FunctionBase::InputType input, std::optional<double *> output_opt,
  ScalarFunction::GradientTypeOpt gradient_opt,
  ScalarFunction::HessianTypeOpt hessian_opt)
{
  int level = 0;
  if (hessian_opt.has_value()) {
    assert(gradient_opt.has_value());
    assert(output_opt.has_value());
    level = 3;
  } else if (gradient_opt.has_value()) {
    assert(output_opt.has_value());
    level = 2;
  } else if (output_opt.has_value()) {
    level = 1;
  }

  if (level == 0) {
    return;
  } else if (level == 1) {
    general_function_->compute(input, general_output_, std::nullopt, std::nullopt);
    (*output_opt.value()) = (general_output_.dot(Q_ * general_output_)) * 0.5;
  } else if (level == 2) {
    general_function_->compute(input, general_output_, general_jacobian_, std::nullopt);
    Eigen::VectorXd weighted_output = Q_ * general_output_;
    (*output_opt.value()) = (general_output_.dot(weighted_output)) * 0.5;
    gradient_opt.value() = general_jacobian_.transpose() * weighted_output;
  } else {
    if (!general_function_->has_hessian()) {
      general_function_->compute(input, general_output_, general_jacobian_, std::nullopt);
      Eigen::VectorXd weighted_output = Q_ * general_output_;
      (*output_opt.value()) = (general_output_.dot(weighted_output)) * 0.5;
      gradient_opt.value() = general_jacobian_.transpose() * weighted_output;
      hessian_opt.value() = general_jacobian_.transpose() * Q_ * general_jacobian_;
    } else {
      general_hessian_.setZero();
      general_function_->compute(input, general_output_, general_jacobian_, general_hessian_array_);
      Eigen::VectorXd weighted_output = Q_ * general_output_;
      (*output_opt.value()) = (general_output_.dot(weighted_output)) * 0.5;
      gradient_opt.value() = general_jacobian_.transpose() * weighted_output;
      hessian_opt.value() = general_jacobian_.transpose() * Q_ * general_jacobian_;
      for (size_t i = 0; i < general_function_->output_dim(); i++) {
        hessian_opt.value() += general_hessian_
          .block(
          0,
          (Eigen::Index)(i * general_function_->input_dim()),
          (Eigen::Index)general_function_->input_dim(),
          general_function_->input_dim())
          .matrix() *
          weighted_output((Eigen::Index)i);
      }
    }
  }
}

}  // namespace solvers
