#pragma once

#include <memory>
#include <utility>
#include "genie_sim_controllers/solvers/linear_relative_function.hpp"
#include "genie_sim_controllers/solvers/scalar_function.hpp"

namespace solvers
{

class QuadraticFunction : public ScalarFunction
{
public:
  QuadraticFunction(const FunctionBase & general_function, Eigen::MatrixXd Q)
  : QuadraticFunction(general_function.deep_copy(), std::move(Q)) {}

  QuadraticFunction(const std::shared_ptr<FunctionBase> & general_function, Eigen::MatrixXd Q)
  : ScalarFunction(general_function->input_dim(), true, true),
    general_function_(general_function),
    Q_(std::move(Q)),
    general_output_(static_cast<int>(general_function_->output_dim())),
    general_jacobian_(static_cast<int>(general_function_->output_dim()),
      static_cast<int>(general_function_->input_dim())),
    general_hessian_(static_cast<int>(general_function_->input_dim()),
      static_cast<int>(general_function_->input_dim()) *
      static_cast<int>(general_function_->output_dim()))
  {
    assert(general_function_->has_jacobian());
    general_hessian_array_.reserve(general_function_->output_dim());
    for (size_t i = 0; i < general_function_->output_dim(); i++) {
      general_hessian_array_.emplace_back(
        general_hessian_.block(
          0,
          static_cast<int>(i) * static_cast<int>(general_function_->input_dim()),
          general_function_->input_dim(), general_function_->input_dim()));
    }
  }

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    return std::make_shared<QuadraticFunction>(general_function_->deep_copy(), Q_);
  }

  template<typename Func>
  Func * general_function_as()
  {
    return dynamic_cast<Func *>(general_function_.get());
  }

  void set_Q(const Eigen::MatrixXd & Q) {Q_ = Q;}

  void doCompute(
    FunctionBase::InputType input, std::optional<double *> output_opt,
    ScalarFunction::GradientTypeOpt gradient_opt,
    ScalarFunction::HessianTypeOpt hessian_opt) override;

  std::shared_ptr<FunctionBase> general_function_;
  Eigen::MatrixXd Q_;
  Eigen::VectorXd general_output_;
  Eigen::MatrixXd general_jacobian_;
  Eigen::MatrixXd general_hessian_;
  std::vector<Eigen::Ref<Eigen::MatrixXd>> general_hessian_array_;
};

}  // namespace solvers
