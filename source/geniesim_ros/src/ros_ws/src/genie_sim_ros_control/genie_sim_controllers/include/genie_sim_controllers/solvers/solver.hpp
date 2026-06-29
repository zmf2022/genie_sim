#pragma once

#include <utility>
#include "genie_sim_controllers/solvers/problem.hpp"

namespace solvers
{

class Solver
{
public:
  virtual ~Solver() = default;
  enum class Status { Unsolved, Solved, Failed };

  explicit Solver(const std::shared_ptr<Problem> & problem)
  : problem_(problem), x_(problem->N()) {}

  void solve() {doSolve();}

  [[nodiscard]] Eigen::Ref<const Eigen::VectorXd> x() const {return x_;}

  virtual void setX(const Eigen::Ref<const Eigen::VectorXd> & x)
  {
    if (x.size() != problem_->N()) {
      throw std::invalid_argument("Size of x does not match the problem size.");
    }
    x_ = x;
  }

  [[nodiscard]] const std::shared_ptr<Problem> & problem() const {return problem_;}

  virtual void reset() {status_ = Status::Unsolved;}

  [[nodiscard]] Status status() const {return status_;}

protected:
  virtual void doSolve() = 0;

  std::shared_ptr<Problem> problem_;
  Eigen::VectorXd x_;
  Status status_ = Status::Unsolved;
};

}  // namespace solvers
