#pragma once

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <random>
#include <vector>

namespace genie_sim_moveit_plugins
{

class StateSpace
{
public:
  StateSpace(const Eigen::VectorXd & lower, const Eigen::VectorXd & upper);

  Eigen::VectorXd sampleUniform();
  double distance(const Eigen::VectorXd & a, const Eigen::VectorXd & b) const;
  double getMaxExtent() const;

  bool isMotionValid(const Eigen::VectorXd & a, const Eigen::VectorXd & b) const;
  void setValidChecker(std::function<bool(const Eigen::VectorXd &)> checker);

private:
  Eigen::VectorXd lower_, upper_, range_;
  std::function<bool(const Eigen::VectorXd &)> valid_checker_;
  std::mt19937 rng_;
};

class RRTConnect
{
public:
  struct Settings
  {
    double max_distance{0.5};
    size_t max_iter{5000};
  };

  explicit RRTConnect(std::shared_ptr<StateSpace> space);

  bool solve(
    const Eigen::VectorXd & start,
    const std::vector<Eigen::VectorXd> & goals);

  void simplifyPath();
  std::vector<Eigen::VectorXd> getPath() const;
  std::vector<Eigen::VectorXd> getPathSimplified() const;

  Settings & settings() {return settings_;}

private:
  static constexpr size_t INVALID_INDEX = static_cast<size_t>(-1);
  static constexpr double NUMBER_TOLERANCE = 1e-6;

  enum class GrowState { REACHED, ADVANCED, TRAPPED };

  struct TreeData
  {
    void initRoot(const Eigen::VectorXd & state);
    void initRoots(const std::vector<Eigen::VectorXd> & states);
    size_t addNode(const Eigen::VectorXd & state, size_t parent_id);
    size_t findNearestNode(
      const Eigen::VectorXd & state,
      const std::shared_ptr<StateSpace> & space) const;
    Eigen::VectorXd & state(size_t index);
    size_t size() const;
    std::vector<Eigen::VectorXd> getPathToRoot(size_t node_id) const;

private:
    std::vector<std::pair<size_t, Eigen::VectorXd>> tree_;
  };

  using TreeDataPtr = std::shared_ptr<TreeData>;

  GrowState growTree(
    TreeDataPtr tree, const Eigen::VectorXd & target,
    size_t & new_node_id);
  Eigen::VectorXd steer(
    const Eigen::VectorXd & from,
    const Eigen::VectorXd & to);

  std::shared_ptr<StateSpace> space_;
  Settings settings_;
  std::vector<Eigen::VectorXd> path_raw_, path_simplified_;
  TreeDataPtr start_, goal_;
};

}  // namespace genie_sim_moveit_plugins
