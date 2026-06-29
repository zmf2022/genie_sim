#pragma once

#include "genie_sim_controllers/solvers/function_base.hpp"
#include "genie_sim_controllers/solvers/scalar_function.hpp"

namespace solvers
{

class RelativeData
{
public:
  RelativeData() {local_indices_.push_back(0);}
  RelativeData(const RelativeData & other) = default;

  explicit RelativeData(const std::vector<std::pair<size_t, size_t>> & relative_index_ranges)
  : relative_index_ranges_(relative_index_ranges)
  {
    if (!setRelativeIndexRanges(relative_index_ranges)) {
      throw std::invalid_argument("Invalid relative index ranges");
    }
  }

  [[nodiscard]] size_t getGlobalIndexLB() const {return GLOBAL_INDEX_LB_;}
  [[nodiscard]] size_t getGlobalIndexUB() const {return GLOBAL_INDEX_UB_;}

  bool setRelativeIndexRanges(const std::vector<std::pair<size_t, size_t>> & relative_index_ranges)
  {
    return updateRelativeIndexRanges(relative_index_ranges);
  }

  bool addRelativeIndexRange(const std::pair<size_t, size_t> & relative_index_range);

  [[nodiscard]] const std::vector<std::pair<size_t, size_t>> & getRelativeIndexRanges() const
  {
    return relative_index_ranges_;
  }
  [[nodiscard]] size_t getRelativeRangeSize() const {return relative_index_ranges_.size();}
  [[nodiscard]] std::pair<size_t, size_t> getRelativeIndexRange(size_t index) const
  {
    return relative_index_ranges_[index];
  }

  [[nodiscard]] std::pair<size_t, size_t> getLocalIndexRange(size_t index) const
  {
    return std::make_pair(local_indices_[index], local_indices_[index + 1]);
  }

  [[nodiscard]] Eigen::Ref<Eigen::VectorXd> extractMutableInputFromLocal(
    size_t index,
    Eigen::Ref<Eigen::VectorXd> local_input)
  const
  {
    return local_input.segment(
      (Eigen::Index)local_indices_[index],
      local_indices_[index + 1] - local_indices_[index]);
  }

  [[nodiscard]] Eigen::Ref<const Eigen::VectorXd> extractInputFromLocal(
    size_t index,
    Eigen::Ref<const Eigen::VectorXd> local_input)
  const
  {
    return local_input.segment(
      (Eigen::Index)local_indices_[index],
      local_indices_[index + 1] - local_indices_[index]);
  }

  [[nodiscard]] Eigen::Ref<Eigen::VectorXd> extractMutableInputFromGlobal(
    size_t index,
    Eigen::Ref<Eigen::VectorXd> global_input)
  const
  {
    return global_input.segment(
      (Eigen::Index)relative_index_ranges_[index].first,
      relative_index_ranges_[index].second - relative_index_ranges_[index].first);
  }

  [[nodiscard]] Eigen::Ref<const Eigen::VectorXd> extractInputFromGlobal(
    size_t index,
    Eigen::Ref<const Eigen::VectorXd> global_input)
  const
  {
    return global_input.segment(
      (Eigen::Index)relative_index_ranges_[index].first,
      relative_index_ranges_[index].second - relative_index_ranges_[index].first);
  }

  [[nodiscard]] Eigen::Ref<Eigen::MatrixXd> extractJacobianFromLocal(
    size_t index,
    Eigen::Ref<Eigen::MatrixXd> local_jacobian)
  const
  {
    return local_jacobian.middleCols(
      (Eigen::Index)local_indices_[index],
      local_indices_[index + 1] - local_indices_[index]);
  }

  [[nodiscard]] Eigen::Ref<Eigen::VectorXd> extractGradientFromLocal(
    size_t index,
    Eigen::Ref<Eigen::VectorXd> local_gradient)
  const
  {
    return local_gradient.segment(
      (Eigen::Index)local_indices_[index],
      local_indices_[index + 1] - local_indices_[index]);
  }

  [[nodiscard]] Eigen::Ref<Eigen::MatrixXd> extractJacobianFromGlobal(
    size_t index,
    Eigen::Ref<Eigen::MatrixXd> global_jacobian)
  const
  {
    return global_jacobian.middleCols(
      (Eigen::Index)relative_index_ranges_[index].first,
      relative_index_ranges_[index].second - relative_index_ranges_[index].first);
  }

  [[nodiscard]] Eigen::Ref<Eigen::VectorXd> extractGradientFromGlobal(
    size_t index,
    Eigen::Ref<Eigen::VectorXd> global_gradient)
  const
  {
    return global_gradient.segment(
      (Eigen::Index)relative_index_ranges_[index].first,
      relative_index_ranges_[index].second - relative_index_ranges_[index].first);
  }

  [[nodiscard]] Eigen::Ref<Eigen::MatrixXd> extractHessianFromLocal(
    size_t i, size_t j,
    Eigen::Ref<Eigen::MatrixXd> local_hessian)
  const
  {
    return local_hessian.block(
      (Eigen::Index)local_indices_[i], (Eigen::Index)local_indices_[j],
      local_indices_[i + 1] - local_indices_[i],
      local_indices_[j + 1] - local_indices_[j]);
  }

  [[nodiscard]] Eigen::Ref<Eigen::MatrixXd> extractHessianFromGlobal(
    size_t i, size_t j,
    Eigen::Ref<Eigen::MatrixXd> global_hessian)
  const
  {
    return global_hessian.block(
      (Eigen::Index)relative_index_ranges_[i].first, (Eigen::Index)relative_index_ranges_[j].first,
      relative_index_ranges_[i].second - relative_index_ranges_[i].first,
      relative_index_ranges_[j].second - relative_index_ranges_[j].first);
  }

protected:
  bool updateRelativeIndexRanges(const std::vector<std::pair<size_t, size_t>> & index_ranges);
  virtual void doUpdate() {}

  std::vector<std::pair<size_t, size_t>> relative_index_ranges_;
  std::vector<size_t> local_indices_;
  size_t GLOBAL_INDEX_LB_ = std::numeric_limits<size_t>::max();
  size_t GLOBAL_INDEX_UB_ = 0;
  size_t RELATIVE_INPUT_DIM_ = 0;
};

template<class FunctionType, typename = std::enable_if_t<std::is_base_of_v<FunctionBase,
  FunctionType>>>
class RelativeFunctionWrapper : public RelativeData, public FunctionType
{
public:
  RelativeFunctionWrapper(
    const FunctionType & function, const std::vector<std::pair<size_t,
    size_t>> & relative_index_ranges)
  : RelativeData(relative_index_ranges), FunctionType(function)
  {
    assert(function.input_dim() == RELATIVE_INPUT_DIM_);
  }

  RelativeFunctionWrapper(
    const std::shared_ptr<FunctionType> & function,
    const std::vector<std::pair<size_t, size_t>> & relative_index_ranges)
  : RelativeData(relative_index_ranges), FunctionType(*function)
  {
    assert(function->input_dim() == RELATIVE_INPUT_DIM_);
  }

  virtual ~RelativeFunctionWrapper() override = default;

  [[nodiscard]] std::shared_ptr<FunctionBase> deep_copy() const override
  {
    auto base_copy = std::dynamic_pointer_cast<FunctionType>(FunctionType::deep_copy());
    return std::make_shared<RelativeFunctionWrapper<FunctionType>>(
      base_copy,
      relative_index_ranges_);
  }
};

}  // namespace solvers
