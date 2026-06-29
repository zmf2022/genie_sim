#pragma once

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <utility>

namespace solvers
{

class FunctionBase
{
public:
  FunctionBase(
    size_t input_dim, size_t output_dim, bool has_jacobian = true,
    bool has_hessian = true)
  : INPUT_DIM_(input_dim), OUTPUT_DIM_(output_dim), HAS_JACOBIAN_(has_jacobian), HAS_HESSIAN_(
      has_hessian) {}

  virtual ~FunctionBase() = default;

  using InputType = Eigen::Ref<const Eigen::VectorXd>;
  using OutputType = Eigen::Ref<Eigen::VectorXd>;
  using JacobianType = Eigen::Ref<Eigen::MatrixXd>;
  using HessianType = Eigen::Ref<Eigen::MatrixXd>;
  using HessianArrayType = std::vector<HessianType>;

  using OutputTypeOpt = std::optional<OutputType>;
  using JacobianTypeOpt = std::optional<JacobianType>;
  using HessianArrayTypeOpt = std::optional<HessianArrayType>;

  void evaluate(FunctionBase::InputType input, FunctionBase::OutputType output)
  {
    compute(std::move(input), output, std::nullopt, std::nullopt);
  }

  void compute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt,
    FunctionBase::HessianArrayTypeOpt hessian_opt)
  {
    if (jacobian_opt.has_value()) {
      jacobian_opt->setZero();
    }
    if (hessian_opt.has_value()) {
      for (auto & hessian : *hessian_opt) {
        hessian.setZero();
      }
    }
    doCompute(
      std::move(input), std::move(output_opt), std::move(jacobian_opt),
      std::move(hessian_opt));
  }

  virtual void doCompute(
    FunctionBase::InputType input, FunctionBase::OutputTypeOpt output_opt,
    FunctionBase::JacobianTypeOpt jacobian_opt,
    FunctionBase::HessianArrayTypeOpt hessian_opt) = 0;

  [[nodiscard]] size_t input_dim() const {return INPUT_DIM_;}
  [[nodiscard]] size_t output_dim() const {return OUTPUT_DIM_;}
  [[nodiscard]] bool has_jacobian() const {return HAS_JACOBIAN_;}
  [[nodiscard]] bool has_hessian() const {return HAS_HESSIAN_;}

  [[nodiscard]] virtual std::shared_ptr<FunctionBase> deep_copy() const = 0;

  template<typename T>
  [[nodiscard]] std::shared_ptr<T> deep_copy_as() const
  {
    return std::dynamic_pointer_cast<T>(deep_copy());
  }

protected:
  size_t INPUT_DIM_;
  size_t OUTPUT_DIM_;
  bool HAS_JACOBIAN_;
  bool HAS_HESSIAN_;
};

}  // namespace solvers
