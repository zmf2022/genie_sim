#include "genie_sim_controllers/solvers/osqp_solver.hpp"
#include <iostream>

namespace solvers::osqp_solver
{

void OSQPSolver::doSolve()
{
  upper_triangle_cost_hessian_.setZero();
  constraint_jacobian_.setZero();
  cost_ = 0;
  q_.setZero();
  cost_hessian_map_.clear();
  constraint_jacobian_map_.clear();
  cost_hessian_nnz_ = 0;
  constraint_jacobian_nnz_ = 0;

  for (size_t i = 0; i < problem()->getCostFunctions().size(); i++) {
    auto & cost = *problem()->getCostFunctions()[i].second;
    auto & cost_scalar = static_cast<solvers::ScalarFunction &>(cost);
    auto & relative_data = dynamic_cast<solvers::RelativeData &>(cost);
    auto & cost_block = problem_ctx_->getCostBlocks()[i];
    for (int j = 0; j < relative_data.getRelativeRangeSize(); j++) {
      relative_data.extractMutableInputFromLocal(
        j,
        cost_block.input) = relative_data.extractInputFromGlobal(j, x());
    }
    cost.compute(cost_block.input, &cost_block.value, cost_block.gradient, cost_block.hessian);
    cost_ += cost_block.value;
    for (int j = 0; j < relative_data.getRelativeRangeSize(); j++) {
      auto row_range = relative_data.getRelativeIndexRange(j);
      Eigen::Ref<Eigen::VectorXd> jacobian_segment = relative_data.extractGradientFromLocal(
        j,
        cost_block.gradient);
      q_.segment(
        static_cast<Eigen::Index>(row_range.first),
        jacobian_segment.size()) += jacobian_segment;
      for (int k = j; k < relative_data.getRelativeRangeSize(); k++) {
        auto col_range = relative_data.getRelativeIndexRange(k);
        Eigen::Ref<Eigen::MatrixXd> hessian_block = relative_data.extractHessianFromLocal(
          j, k,
          cost_block.hessian);
        updateCostHessianMap(
          static_cast<int>(row_range.first), static_cast<int>(col_range.first),
          hessian_block);
      }
    }
  }

  size_t constraints_offset = 0;
  for (size_t i = 0; i < problem()->getEqualityConstraints().size(); i++) {
    auto & constraint = *problem()->getEqualityConstraints()[i].second;
    const auto & relative_data = dynamic_cast<const solvers::RelativeData &>(constraint);
    auto & equality_block = problem_ctx_->getEqualityBlocks()[i];
    for (int j = 0; j < relative_data.getRelativeRangeSize(); j++) {
      relative_data.extractMutableInputFromLocal(
        j,
        equality_block.input) = relative_data.extractInputFromGlobal(j, x());
    }
    constraint.compute(equality_block.input, equality_block.value, equality_block.jacobian);
    u_.segment(static_cast<Eigen::Index>(constraints_offset), equality_block.value.size()) =
      l_.segment(
      static_cast<Eigen::Index>(constraints_offset),
      equality_block.value.size()) = -equality_block.value + constraint.value();
    for (int j = 0; j < relative_data.getRelativeRangeSize(); j++) {
      auto range = relative_data.getRelativeIndexRange(j);
      Eigen::Ref<Eigen::MatrixXd> jacobian_block = relative_data.extractJacobianFromLocal(
        j,
        equality_block.jacobian);
      updateConstraintJacobianMap(
        static_cast<int>(constraints_offset),
        static_cast<int>(range.first), jacobian_block);
    }
    constraints_offset += constraint.output_dim();
  }

  for (size_t i = 0; i < problem()->getInequalityConstraints().size(); i++) {
    auto & constraint = *problem()->getInequalityConstraints()[i].second;
    const auto & relative_data = dynamic_cast<const solvers::RelativeData &>(constraint);
    auto & inequality_block = problem_ctx_->getInequalityBlocks()[i];
    for (int j = 0; j < relative_data.getRelativeRangeSize(); j++) {
      relative_data.extractMutableInputFromLocal(
        j,
        inequality_block.input) = relative_data.extractInputFromGlobal(j, x());
    }
    constraint.compute(inequality_block.input, inequality_block.value, inequality_block.jacobian);
    u_.segment(
      static_cast<Eigen::Index>(constraints_offset),
      inequality_block.value.size()) = -inequality_block.value + constraint.upperBound();
    if (constraint.hasLowerBound()) {
      l_.segment(
        static_cast<Eigen::Index>(constraints_offset),
        inequality_block.value.size()) = -inequality_block.value + constraint.lowerBound();
    } else {
      l_.segment(static_cast<Eigen::Index>(constraints_offset), inequality_block.value.size()) =
        Eigen::VectorXd::Constant(inequality_block.value.size(), -OSQP_INFTY);
    }
    for (int j = 0; j < relative_data.getRelativeRangeSize(); j++) {
      auto range = relative_data.getRelativeIndexRange(j);
      Eigen::Ref<Eigen::MatrixXd> jacobian_block = relative_data.extractJacobianFromLocal(
        j,
        inequality_block.jacobian);
      updateConstraintJacobianMap(
        static_cast<int>(constraints_offset),
        static_cast<int>(range.first), jacobian_block);
    }
    constraints_offset += constraint.output_dim();
  }

  cost_hessian_triplets_.reserve(cost_hessian_nnz_);
  for (const auto & key_value : cost_hessian_map_) {
    cost_hessian_triplets_.emplace_back(toTriplet(key_value));
  }
  upper_triangle_cost_hessian_.setFromTriplets(
    cost_hessian_triplets_.begin(), cost_hessian_triplets_.end());
  cost_hessian_triplets_.clear();

  constraint_jacobian_triplets_.reserve(constraint_jacobian_nnz_);
  for (const auto & key_value : constraint_jacobian_map_) {
    constraint_jacobian_triplets_.emplace_back(toTriplet(key_value));
  }
  constraint_jacobian_.setFromTriplets(
    constraint_jacobian_triplets_.begin(), constraint_jacobian_triplets_.end());
  constraint_jacobian_triplets_.clear();

  OSQPWorkspace * work = nullptr;
  CSCWrapper P = EigenSparseMatrixToOSQPCscMatrix(upper_triangle_cost_hessian_);
  CSCWrapper A = EigenSparseMatrixToOSQPCscMatrix(constraint_jacobian_);
  c_float * q = q_.data();
  c_float * l = l_.data();
  c_float * u = u_.data();

  OSQPData data;
  data.n = (c_int)problem()->N();
  data.m = (c_int)problem()->M();
  data.P = P.csc_matrix_.get();
  data.q = q;
  data.A = A.csc_matrix_.get();
  data.l = l;
  data.u = u;

  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.rho = 0.1;
  settings.verbose = 0;

  auto exit_flag = osqp_setup(&work, &data, &settings);
  if (exit_flag != 0) {
    std::cerr << "OSQP setup failed with exit flag " << exit_flag << std::endl;
    status_ = Status::Failed;
    return;
  }

  exit_flag = osqp_solve(work);
  if (exit_flag != 0) {
    std::cerr << "OSQP solver failed with exit flag " << exit_flag << std::endl;
    std::cerr << "OSQP status message: " << work->info->status << std::endl;
    status_ = Status::Failed;
    osqp_cleanup(work);
    return;
  }

  if (work->info->status_val != OSQP_SOLVED) {
    std::cerr << "OSQP status message: " << work->info->status << std::endl;
    status_ = Status::Failed;
    osqp_cleanup(work);
    return;
  }

  delta_x_ = Eigen::Map<Eigen::VectorXd>(work->solution->x, (Eigen::Index)problem()->N());
  y_ = Eigen::Map<Eigen::VectorXd>(work->solution->y, (Eigen::Index)problem()->M());
  x_ += delta_x_;

  osqp_cleanup(work);
  status_ = Status::Solved;
}

void OSQPSolver::init()
{
  cost_ = 0;
  y_.setZero();
  problem_ctx_ = problem()->createContext();
}

void OSQPSolver::updateCostHessianMap(
  int start_row, int start_col,
  const Eigen::Ref<const Eigen::MatrixXd> & block)
{
  int index = start_row * (int)problem()->N() + start_col;
  for (int i = 0; i < block.rows(); i++) {
    for (int j = i; j < block.cols(); j++) {
      auto value = block(i, j);
      if (value == 0.0) {continue;}
      auto it = cost_hessian_map_.find(index + j);
      if (it == cost_hessian_map_.end()) {
        cost_hessian_map_[index + j] = value;
        cost_hessian_nnz_++;
      } else {
        it->second += value;
      }
    }
    index += (int)problem()->N();
  }
}

void OSQPSolver::updateConstraintJacobianMap(
  int start_row, int start_col,
  const Eigen::Ref<const Eigen::MatrixXd> & block)
{
  int index = start_row * (int)problem()->N() + start_col;
  for (int i = 0; i < block.rows(); i++) {
    for (int j = 0; j < block.cols(); j++) {
      auto value = block(i, j);
      if (value == 0.0) {continue;}
      auto it = constraint_jacobian_map_.find(index + j);
      if (it == constraint_jacobian_map_.end()) {
        constraint_jacobian_map_[index + j] = value;
        constraint_jacobian_nnz_++;
      } else {
        it->second += value;
      }
    }
    index += (int)problem()->N();
  }
}

OSQPSolver::CSCWrapper OSQPSolver::EigenSparseMatrixToOSQPCscMatrix(
  const Eigen::SparseMatrix<double> & matrix)
{
  CSCWrapper wrapper;

  c_int m = (c_int)matrix.rows();
  c_int n = (c_int)matrix.cols();
  c_int nzmax = (c_int)matrix.nonZeros();

  wrapper.p_.resize(n + 1);
  wrapper.i_.resize(nzmax);
  wrapper.x_.resize(nzmax);

  for (c_int i = 0; i < nzmax; i++) {
    wrapper.i_[i] = (c_int)matrix.innerIndexPtr()[i];
    wrapper.x_[i] = (c_float)matrix.valuePtr()[i];
  }
  for (c_int i = 0; i < n + 1; i++) {
    wrapper.p_[i] = (c_int)matrix.outerIndexPtr()[i];
  }

  wrapper.csc_matrix_.reset(
    csc_matrix(
      m, n, nzmax,
      wrapper.x_.data(),
      wrapper.i_.data(),
      wrapper.p_.data()));
  return wrapper;
}

void OSQPSolver::reset() {is_first_solve_ = true;}

}  // namespace solvers::osqp_solver
