#pragma once

#include <osqp.h>
#include <Eigen/Sparse>
#include <cstdlib>
#include "genie_sim_controllers/solvers/solver.hpp"

namespace solvers::osqp_solver
{

class OSQPSolver : public solvers::Solver
{
public:
  struct CSCWrapper
  {
    CSCWrapper() = default;
    CSCWrapper(CSCWrapper &&) = default;
    CSCWrapper(const CSCWrapper &) = delete;
    CSCWrapper & operator=(CSCWrapper &&) = default;
    CSCWrapper & operator=(const CSCWrapper &) = delete;
    std::unique_ptr<csc, decltype(&c_free)> csc_matrix_{nullptr, c_free};
    std::vector<c_int> p_;
    std::vector<c_int> i_;
    std::vector<c_float> x_;
  };

  explicit OSQPSolver(const std::shared_ptr<solvers::Problem> & problem)
  : solvers::Solver(problem),
    q_(problem->N()),
    l_(problem->M()),
    u_(problem->M()),
    delta_x_(problem->N()),
    y_(problem->M()),
    upper_triangle_cost_hessian_((int)problem->N(), (int)problem->N()),
    constraint_jacobian_((int)problem->M(), (int)problem->N())
  {
    init();
  }

  void reset() override;
  const Eigen::VectorXd & delta_x() const {return delta_x_;}
  const Eigen::VectorXd & y() const {return y_;}
  const Eigen::VectorXd & l() const {return l_;}
  const Eigen::VectorXd & u() const {return u_;}
  const Eigen::SparseMatrix<double> & upper_triangle_cost_hessian() const
  {
    return upper_triangle_cost_hessian_;
  }
  const Eigen::SparseMatrix<double> & constraint_jacobian() const {return constraint_jacobian_;}
  const Eigen::VectorXd & q() const {return q_;}
  double cost() const {return cost_;}

protected:
  void doSolve() override;
  void init();

  [[nodiscard]] Eigen::Triplet<double> toTriplet(const std::pair<int, double> & pair) const
  {
    auto r = std::div(pair.first, (int)problem()->N());
    return {r.quot, r.rem, pair.second};
  }

  static CSCWrapper EigenSparseMatrixToOSQPCscMatrix(const Eigen::SparseMatrix<double> & matrix);
  void updateCostHessianMap(
    int start_row, int start_col,
    const Eigen::Ref<const Eigen::MatrixXd> & block);
  void updateConstraintJacobianMap(
    int start_row, int start_col,
    const Eigen::Ref<const Eigen::MatrixXd> & block);

protected:
  double cost_{0};
  std::unique_ptr<solvers::Problem::Context> problem_ctx_;
  Eigen::VectorXd q_;
  Eigen::VectorXd l_;
  Eigen::VectorXd u_;
  Eigen::VectorXd delta_x_;
  Eigen::VectorXd y_;

  std::unordered_map<int, double> cost_hessian_map_;
  std::unordered_map<int, double> constraint_jacobian_map_;

  std::vector<Eigen::Triplet<double>> cost_hessian_triplets_;
  std::vector<Eigen::Triplet<double>> constraint_jacobian_triplets_;

  Eigen::SparseMatrix<double> upper_triangle_cost_hessian_;
  Eigen::SparseMatrix<double> constraint_jacobian_;

  size_t cost_hessian_nnz_{0};
  size_t constraint_jacobian_nnz_{0};

  OSQPWorkspace * osqp_work_{nullptr};
  CSCWrapper osqp_solver_P_;
  CSCWrapper osqp_solver_A_;
  OSQPSettings osqp_settings_{};

  bool is_first_solve_{true};
};

}  // namespace solvers::osqp_solver
