#include "genie_sim_controllers/solvers/relative_function.hpp"

namespace solvers
{

bool RelativeData::updateRelativeIndexRanges(
  const std::vector<std::pair<size_t,
  size_t>> & index_ranges)
{
  if (index_ranges.empty()) {
    return false;
  }
  local_indices_.clear();
  local_indices_.reserve(index_ranges.size() + 1);
  size_t size = 0;
  for (const auto & cur_range : index_ranges) {
    if (cur_range.second <= cur_range.first) {
      return false;
    }
    GLOBAL_INDEX_UB_ = std::max(GLOBAL_INDEX_UB_, cur_range.second);
    GLOBAL_INDEX_LB_ = std::min(GLOBAL_INDEX_LB_, cur_range.first);
    local_indices_.push_back(size);
    size += cur_range.second - cur_range.first;
  }
  local_indices_.push_back(size);
  RELATIVE_INPUT_DIM_ = size;
  relative_index_ranges_ = index_ranges;
  doUpdate();
  return true;
}

bool RelativeData::addRelativeIndexRange(const std::pair<size_t, size_t> & relative_index_range)
{
  if (relative_index_range.first >= relative_index_range.second) {
    return false;
  }
  relative_index_ranges_.push_back(relative_index_range);
  RELATIVE_INPUT_DIM_ += relative_index_range.second - relative_index_range.first;
  local_indices_.push_back(RELATIVE_INPUT_DIM_);
  GLOBAL_INDEX_UB_ = std::max(GLOBAL_INDEX_UB_, relative_index_range.second);
  GLOBAL_INDEX_LB_ = std::min(GLOBAL_INDEX_LB_, relative_index_range.first);
  doUpdate();
  return true;
}

}  // namespace solvers
