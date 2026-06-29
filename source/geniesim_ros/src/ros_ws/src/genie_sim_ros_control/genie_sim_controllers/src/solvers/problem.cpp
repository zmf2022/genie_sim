#include "genie_sim_controllers/solvers/problem.hpp"
#include <ranges>

namespace solvers
{

bool Problem::addCostFunctionImpl(
  const std::string & name,
  const std::shared_ptr<ScalarFunction> & cost_function)
{
  const auto [_, result] = cost_function_map_.insert(std::make_pair(name, cost_function));
  cost_functions_.emplace_back(name, cost_function);
  return result;
}

bool Problem::addEqualityConstraintImpl(
  const std::string & name,
  const std::shared_ptr<EqualityConstraint> & equality_constraint)
{
  const auto [_,
    result] = equality_constraint_map_.insert(std::make_pair(name, equality_constraint));
  equality_constraints_.push_back(std::make_pair(name, equality_constraint));
  return result;
}

bool Problem::addInequalityConstraintImpl(
  const std::string & name,
  const std::shared_ptr<InequalityConstraint> & inequality_constraint)
{
  const auto [_,
    result] = inequality_constraint_map_.insert(std::make_pair(name, inequality_constraint));
  inequality_constraints_.push_back(std::make_pair(name, inequality_constraint));
  return result;
}

bool Problem::makeProblem()
{
  auto result = doMakeProblem();
  if (!result) {return false;}
  result = postMakeProblem();
  if (!result) {return false;}
  equality_constraint_dim_ = 0;
  inequality_constraint_dim_ = 0;
  for (const auto & equality_constraint : equality_constraints_ | std::views::values) {
    equality_constraint_dim_ += equality_constraint->output_dim();
  }
  for (const auto & inequality_constraint : inequality_constraints_ | std::views::values) {
    inequality_constraint_dim_ += inequality_constraint->output_dim();
  }
  M_ = equality_constraint_dim_ + inequality_constraint_dim_;
  return result;
}

bool Problem::checkFunction(const RelativeData & function) const
{
  return function.getGlobalIndexUB() <= N_;
}

std::unique_ptr<Problem::Context> Problem::createContext() const
{
  auto context = std::make_unique<Context>();
  auto & cost_blocks = context->cost_blocks_;
  auto & equality_blocks = context->equality_blocks_;
  auto & inequality_blocks = context->inequality_blocks_;
  context->N_ = N_;
  context->M_ = 0;
  for (const auto & cost_function : cost_functions_ | std::views::values) {
    cost_blocks.emplace_back(cost_function->input_dim());
  }
  for (const auto & equality_constraint : equality_constraints_ | std::views::values) {
    equality_blocks.emplace_back(
      equality_constraint->input_dim(),
      equality_constraint->output_dim());
    context->M_ += equality_constraint->output_dim();
  }
  for (const auto & inequality_constraint : inequality_constraints_ | std::views::values) {
    inequality_blocks.emplace_back(
      inequality_constraint->input_dim(), inequality_constraint->output_dim());
    context->M_ += inequality_constraint->output_dim();
  }
  return context;
}

}  // namespace solvers
