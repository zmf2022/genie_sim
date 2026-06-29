#pragma once

#include "genie_sim_controllers/solvers/function_base.hpp"

namespace solvers
{

class ConeBase
{
public:
  virtual ~ConeBase() = default;
  explicit ConeBase() = default;
  virtual void projection(
    const Eigen::Ref<const Eigen::VectorXd> & x,
    Eigen::Ref<Eigen::VectorXd> x_proj) const = 0;
  virtual void jacobian(
    const Eigen::Ref<const Eigen::VectorXd> & x,
    Eigen::Ref<Eigen::MatrixXd> jacobian) const = 0;
  [[nodiscard]] virtual const ConeBase & dual() const = 0;
};

class IdentityCone;

class ZeroCone final : public ConeBase
{
public:
  using ConeBase::ConeBase;
  void projection(
    const Eigen::Ref<const Eigen::VectorXd> & x,
    Eigen::Ref<Eigen::VectorXd> x_proj) const override
  {
    x_proj.setZero();
  }
  void jacobian(
    const Eigen::Ref<const Eigen::VectorXd> & x,
    Eigen::Ref<Eigen::MatrixXd> jacobian) const override
  {
    jacobian.setZero();
  }
  [[nodiscard]] const ConeBase & dual() const override;
};

class IdentityCone final : public ConeBase
{
public:
  using ConeBase::ConeBase;
  void projection(
    const Eigen::Ref<const Eigen::VectorXd> & x,
    Eigen::Ref<Eigen::VectorXd> x_proj) const override {x_proj = x;}
  void jacobian(
    const Eigen::Ref<const Eigen::VectorXd> & x,
    Eigen::Ref<Eigen::MatrixXd> jacobian) const override {jacobian.setIdentity();}
  [[nodiscard]] const ConeBase & dual() const override
  {
    static ZeroCone zero_cone;
    return zero_cone;
  }
};

inline const ConeBase & ZeroCone::dual() const
{
  static IdentityCone identity_cone;
  return identity_cone;
}

class NegativeOrthogonalCone final : public ConeBase
{
public:
  void projection(
    const Eigen::Ref<const Eigen::VectorXd> & x,
    Eigen::Ref<Eigen::VectorXd> x_proj) const override
  {
    x_proj = x.cwiseMin(0);
  }
  void jacobian(
    const Eigen::Ref<const Eigen::VectorXd> & x,
    Eigen::Ref<Eigen::MatrixXd> jacobian) const override
  {
    jacobian.setZero();
    for (Eigen::Index i = 0; i < jacobian.rows(); ++i) {
      jacobian(i, i) = (x(i) > 0) ? 0 : 1;
    }
  }
  [[nodiscard]] const ConeBase & dual() const override
  {
    static NegativeOrthogonalCone dual_cone;
    return dual_cone;
  }
};

}  // namespace solvers
