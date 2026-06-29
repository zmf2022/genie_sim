#include "rrt_connect_planner.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace genie_sim_moveit_plugins
{

// ========================== StateSpace ==========================

StateSpace::StateSpace(
  const Eigen::VectorXd & lower,
  const Eigen::VectorXd & upper)
: lower_(lower), upper_(upper), range_(upper - lower),
  rng_(std::random_device{}())
{
}

Eigen::VectorXd StateSpace::sampleUniform()
{
  Eigen::VectorXd sample(lower_.size());
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (Eigen::Index i = 0; i < lower_.size(); ++i) {
    sample[i] = lower_[i] + dist(rng_) * range_[i];
  }
  return sample;
}

double StateSpace::distance(
  const Eigen::VectorXd & a,
  const Eigen::VectorXd & b) const
{
  return (a - b).norm();
}

double StateSpace::getMaxExtent() const
{
  return range_.norm();
}

bool StateSpace::isMotionValid(
  const Eigen::VectorXd & a,
  const Eigen::VectorXd & b) const
{
  if (!valid_checker_) {return true;}

  const int steps = std::max(2, static_cast<int>(std::ceil(distance(a, b) / 0.05)));
  for (int i = 0; i <= steps; ++i) {
    double t = static_cast<double>(i) / steps;
    Eigen::VectorXd q = a + t * (b - a);
    if (!valid_checker_(q)) {return false;}
  }
  return true;
}

void StateSpace::setValidChecker(
  std::function<bool(const Eigen::VectorXd &)> checker)
{
  valid_checker_ = std::move(checker);
}

// ========================== TreeData ==========================

void RRTConnect::TreeData::initRoot(const Eigen::VectorXd & state)
{
  tree_.clear();
  tree_.emplace_back(INVALID_INDEX, state);
}

void RRTConnect::TreeData::initRoots(
  const std::vector<Eigen::VectorXd> & states)
{
  tree_.clear();
  tree_.reserve(states.size());
  for (const auto & s : states) {
    tree_.emplace_back(INVALID_INDEX, s);
  }
}

size_t RRTConnect::TreeData::addNode(
  const Eigen::VectorXd & state,
  size_t parent_id)
{
  tree_.emplace_back(parent_id, state);
  return tree_.size() - 1;
}

size_t RRTConnect::TreeData::findNearestNode(
  const Eigen::VectorXd & state,
  const std::shared_ptr<StateSpace> & space) const
{
  double min_dist = std::numeric_limits<double>::max();
  size_t nearest = 0;
  for (size_t i = 0; i < tree_.size(); ++i) {
    double d = space->distance(tree_[i].second, state);
    if (d < min_dist) {min_dist = d; nearest = i;}
  }
  return nearest;
}

Eigen::VectorXd & RRTConnect::TreeData::state(size_t index)
{
  return tree_[index].second;
}

size_t RRTConnect::TreeData::size() const {return tree_.size();}

std::vector<Eigen::VectorXd> RRTConnect::TreeData::getPathToRoot(
  size_t node_id) const
{
  std::vector<Eigen::VectorXd> path;
  while (node_id != INVALID_INDEX) {
    path.push_back(tree_[node_id].second);
    node_id = tree_[node_id].first;
  }
  return path;
}

// ========================== RRTConnect ==========================

RRTConnect::RRTConnect(std::shared_ptr<StateSpace> space)
: space_(std::move(space))
{
  constexpr double EXTENT_FRACTION = 0.2;
  settings_.max_distance = EXTENT_FRACTION * space_->getMaxExtent();
}

bool RRTConnect::solve(
  const Eigen::VectorXd & start,
  const std::vector<Eigen::VectorXd> & goals)
{
  path_raw_.clear();
  path_simplified_.clear();
  start_ = std::make_shared<TreeData>();
  goal_ = std::make_shared<TreeData>();

  start_->initRoot(start);
  goal_->initRoots(goals);

  bool is_reverse = false;
  while (start_->size() + goal_->size() < settings_.max_iter) {
    auto tree1 = is_reverse ? goal_ : start_;
    auto tree2 = is_reverse ? start_ : goal_;

    Eigen::VectorXd random_state = space_->sampleUniform();
    size_t grow_node_1 = 0;
    GrowState gs = growTree(tree1, random_state, grow_node_1);
    if (gs != GrowState::TRAPPED) {
      size_t grow_node_2 = 0;
      GrowState gsc = growTree(tree2, tree1->state(grow_node_1), grow_node_2);
      while (gsc == GrowState::ADVANCED) {
        gsc = growTree(tree2, tree1->state(grow_node_1), grow_node_2);
      }
      if (gsc == GrowState::REACHED) {
        size_t start_id = (tree1 == start_) ? grow_node_1 : grow_node_2;
        size_t goal_id = (tree1 == start_) ? grow_node_2 : grow_node_1;
        auto path_s = start_->getPathToRoot(start_id);
        auto path_g = goal_->getPathToRoot(goal_id);
        std::reverse(path_s.begin(), path_s.end());
        path_raw_ = std::move(path_s);
        path_raw_.insert(path_raw_.end(), path_g.begin() + 1, path_g.end());
        return true;
      }
    }
    is_reverse = !is_reverse;
  }
  return false;
}

void RRTConnect::simplifyPath()
{
  if (path_raw_.size() <= 2) {path_simplified_ = path_raw_; return;}

  size_t n = path_raw_.size();
  std::vector<std::pair<size_t, double>> dp(n, {INVALID_INDEX, 0.0});
  for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
    dp[i] = {static_cast<size_t>(i + 1),
      space_->distance(path_raw_[i], path_raw_[i + 1]) + dp[i + 1].second};
  }

  for (int i = static_cast<int>(n) - 3; i >= 0; --i) {
    for (int j = static_cast<int>(n) - 1; j >= i + 2; --j) {
      double cost = space_->distance(path_raw_[i], path_raw_[j]) + dp[j].second;
      if (cost < dp[i].second &&
        space_->isMotionValid(path_raw_[i], path_raw_[j]))
      {
        dp[i] = {static_cast<size_t>(j), cost};
      }
    }
  }

  path_simplified_.clear();
  size_t cur = 0;
  while (cur != INVALID_INDEX) {
    path_simplified_.push_back(path_raw_[cur]);
    cur = dp[cur].first;
  }
}

std::vector<Eigen::VectorXd> RRTConnect::getPath() const {return path_raw_;}
std::vector<Eigen::VectorXd> RRTConnect::getPathSimplified() const {return path_simplified_;}

// ========================== private ==========================

RRTConnect::GrowState RRTConnect::growTree(
  TreeDataPtr tree, const Eigen::VectorXd & target, size_t & new_node_id)
{
  new_node_id = 0;
  size_t nearest = tree->findNearestNode(target, space_);
  if (space_->distance(tree->state(nearest), target) < NUMBER_TOLERANCE) {
    new_node_id = nearest;
    return GrowState::REACHED;
  }
  Eigen::VectorXd ns = steer(tree->state(nearest), target);
  if (space_->isMotionValid(tree->state(nearest), ns)) {
    new_node_id = tree->addNode(ns, nearest);
    return (space_->distance(ns, target) < NUMBER_TOLERANCE) ?
           GrowState::REACHED : GrowState::ADVANCED;
  }
  return GrowState::TRAPPED;
}

Eigen::VectorXd RRTConnect::steer(
  const Eigen::VectorXd & from,
  const Eigen::VectorXd & to)
{
  Eigen::VectorXd dir = to - from;
  double d = space_->distance(from, to);
  return from + dir * std::min(d, settings_.max_distance) / d;
}

}  // namespace genie_sim_moveit_plugins
