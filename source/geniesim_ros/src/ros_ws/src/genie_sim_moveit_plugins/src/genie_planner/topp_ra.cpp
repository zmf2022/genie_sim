#include "topp_ra.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace genie_sim_moveit_plugins
{

static double pathLength(const std::vector<Eigen::VectorXd> & wp)
{
  double len = 0.0;
  for (size_t i = 1; i < wp.size(); ++i) {
    len += (wp[i] - wp[i - 1]).norm();
  }
  return len;
}

static std::vector<double> cumulativeArcLength(
  const std::vector<Eigen::VectorXd> & wp)
{
  std::vector<double> s(wp.size(), 0.0);
  for (size_t i = 1; i < wp.size(); ++i) {
    s[i] = s[i - 1] + (wp[i] - wp[i - 1]).norm();
  }
  return s;
}

static Eigen::VectorXd interpLinear(
  const std::vector<Eigen::VectorXd> & wp,
  const std::vector<double> & s_vals,
  double s)
{
  if (s <= s_vals.front()) {return wp.front();}
  if (s >= s_vals.back()) {return wp.back();}

  auto it = std::upper_bound(s_vals.begin(), s_vals.end(), s);
  size_t idx = static_cast<size_t>(it - s_vals.begin());
  double ds = s_vals[idx] - s_vals[idx - 1];
  if (ds < 1e-12) {return wp[idx - 1];}
  double t = (s - s_vals[idx - 1]) / ds;
  return wp[idx - 1] + t * (wp[idx] - wp[idx - 1]);
}

ToppResult toppRA(
  const std::vector<Eigen::VectorXd> & waypoints,
  const Eigen::VectorXd & vel_limits,
  const Eigen::VectorXd & acc_limits,
  double output_dt)
{
  ToppResult result;
  if (waypoints.size() < 2) {
    result.error_msg = "Need at least 2 waypoints";
    return result;
  }

  double total_path = pathLength(waypoints);
  if (total_path < 1e-8) {
    result.error_msg = "Path length near zero";
    return result;
  }

  auto s_vals = cumulativeArcLength(waypoints);
  double S = s_vals.back();

  constexpr size_t GRID_N = 200;
  Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(GRID_N, 0.0, S);

  std::vector<Eigen::VectorXd> grid_q(GRID_N);
  std::vector<Eigen::VectorXd> grid_dqds(GRID_N);
  for (size_t i = 0; i < GRID_N; ++i) {
    grid_q[i] = interpLinear(waypoints, s_vals, grid[i]);
    double ds = (i + 1 < GRID_N) ? grid[i + 1] - grid[i] : grid[i] - grid[i - 1];
    Eigen::VectorXd q_next = interpLinear(
      waypoints, s_vals,
      std::min(grid[i] + ds, S));
    grid_dqds[i] = (q_next - grid_q[i]) / std::max(ds, 1e-12);
  }

  Eigen::VectorXd sd_max(GRID_N);
  for (size_t i = 0; i < GRID_N; ++i) {
    double sd = std::numeric_limits<double>::max();
    for (Eigen::Index j = 0; j < vel_limits.size(); ++j) {
      double abs_dq = std::abs(grid_dqds[i][j]);
      if (abs_dq > 1e-10) {
        sd = std::min(sd, vel_limits[j] / abs_dq);
      }
    }
    sd_max[i] = sd;
  }

  Eigen::VectorXd sd_forward(GRID_N), sd_backward(GRID_N);
  sd_forward[0] = 0.0;
  for (size_t i = 1; i < GRID_N; ++i) {
    double ds = grid[i] - grid[i - 1];
    double acc_max_sd = std::numeric_limits<double>::max();
    for (Eigen::Index j = 0; j < acc_limits.size(); ++j) {
      double abs_dq = std::abs(grid_dqds[i - 1][j]);
      if (abs_dq > 1e-10) {
        acc_max_sd = std::min(acc_max_sd, acc_limits[j] / abs_dq);
      }
    }
    double sd_new = std::sqrt(
      sd_forward[i - 1] * sd_forward[i - 1] +
      2.0 * acc_max_sd * ds);
    sd_forward[i] = std::min(sd_new, sd_max[i]);
  }

  sd_backward[GRID_N - 1] = 0.0;
  for (int i = static_cast<int>(GRID_N) - 2; i >= 0; --i) {
    double ds = grid[i + 1] - grid[i];
    double acc_max_sd = std::numeric_limits<double>::max();
    for (Eigen::Index j = 0; j < acc_limits.size(); ++j) {
      double abs_dq = std::abs(grid_dqds[i + 1][j]);
      if (abs_dq > 1e-10) {
        acc_max_sd = std::min(acc_max_sd, acc_limits[j] / abs_dq);
      }
    }
    double sd_new = std::sqrt(
      sd_backward[i + 1] * sd_backward[i + 1] +
      2.0 * acc_max_sd * ds);
    sd_backward[i] = std::min(sd_new, sd_max[i]);
  }

  Eigen::VectorXd sd_profile(GRID_N);
  for (size_t i = 0; i < GRID_N; ++i) {
    sd_profile[i] = std::min({sd_forward[i], sd_backward[i], sd_max[i]});
  }

  double total_time = 0.0;
  for (size_t i = 1; i < GRID_N; ++i) {
    double ds = grid[i] - grid[i - 1];
    double avg_sd = 0.5 * (sd_profile[i - 1] + sd_profile[i]);
    if (avg_sd < 1e-12) {avg_sd = 1e-12;}
    total_time += ds / avg_sd;
  }

  std::vector<double> t_grid(GRID_N, 0.0);
  for (size_t i = 1; i < GRID_N; ++i) {
    double ds = grid[i] - grid[i - 1];
    double avg_sd = 0.5 * (sd_profile[i - 1] + sd_profile[i]);
    if (avg_sd < 1e-12) {avg_sd = 1e-12;}
    t_grid[i] = t_grid[i - 1] + ds / avg_sd;
  }

  auto interpS = [&](double t) -> double {
      if (t <= 0.0) {return 0.0;}
      if (t >= total_time) {return S;}
      auto it = std::upper_bound(t_grid.begin(), t_grid.end(), t);
      size_t idx = static_cast<size_t>(it - t_grid.begin());
      double dt_seg = t_grid[idx] - t_grid[idx - 1];
      double frac = (dt_seg > 1e-12) ? (t - t_grid[idx - 1]) / dt_seg : 0.0;
      return grid[idx - 1] + frac * (grid[idx] - grid[idx - 1]);
    };

  size_t num_pts = static_cast<size_t>(std::ceil(total_time / output_dt)) + 1;
  result.positions.reserve(num_pts);
  result.velocities.reserve(num_pts);
  result.accelerations.reserve(num_pts);

  Eigen::VectorXd prev_vel = Eigen::VectorXd::Zero(vel_limits.size());
  for (size_t i = 0; i < num_pts; ++i) {
    double t = std::min(static_cast<double>(i) * output_dt, total_time);
    double s = interpS(t);
    Eigen::VectorXd q = interpLinear(waypoints, s_vals, s);
    result.positions.push_back(q);

    if (i > 0) {
      Eigen::VectorXd vel = (q - result.positions[i - 1]) / output_dt;
      result.velocities.push_back(vel);
      if (i > 1) {
        Eigen::VectorXd acc = (vel - prev_vel) / output_dt;
        result.accelerations.push_back(acc);
      } else {
        result.accelerations.push_back(Eigen::VectorXd::Zero(vel_limits.size()));
      }
      prev_vel = vel;
    } else {
      result.velocities.push_back(Eigen::VectorXd::Zero(vel_limits.size()));
      result.accelerations.push_back(Eigen::VectorXd::Zero(vel_limits.size()));
    }
  }

  if (!result.positions.empty()) {
    result.positions.back() = waypoints.back();
    result.velocities.back() = Eigen::VectorXd::Zero(vel_limits.size());
    result.accelerations.back() = Eigen::VectorXd::Zero(vel_limits.size());
  }

  result.success = true;
  result.duration = total_time;
  result.dt = output_dt;
  return result;
}

std::vector<Eigen::VectorXd> butterworthFilter(
  const std::vector<Eigen::VectorXd> & positions,
  double cutoff_freq, double sample_freq, int buffer_size)
{
  if (positions.size() < 2) {return positions;}

  buffer_size = std::max(buffer_size, 0);
  cutoff_freq = std::min(std::abs(cutoff_freq), sample_freq / 2.0);
  double wc = std::tan(M_PI * cutoff_freq / sample_freq);
  double k1 = std::sqrt(2.0) * wc;
  double k2 = wc * wc;
  double a0 = k2 / (1.0 + k1 + k2);
  double a1 = 2.0 * a0;
  double a2 = a0;
  double b1 = 2.0 * a0 * (1.0 / k2 - 1.0);
  double b2 = 1.0 - (a0 + a1 + a2 + b1);

  auto expand = [&](const std::vector<Eigen::VectorXd> & in)
    -> std::vector<Eigen::VectorXd> {
      std::vector<Eigen::VectorXd> out;
      out.reserve(in.size() + 2 * buffer_size);
      for (int i = 0; i < buffer_size; ++i) {
        out.push_back(in.front());
      }
      for (auto & v : in) {
        out.push_back(v);
      }
      for (int i = 0; i < buffer_size; ++i) {
        out.push_back(in.back());
      }
      return out;
    };

  auto applyFilter = [&](std::vector<Eigen::VectorXd> & data, bool reverse) {
      size_t n = data.size();
      if (n < 3) {return;}
      int start = reverse ? static_cast<int>(n) - 3 : 2;
      int end = reverse ? -1 : static_cast<int>(n);
      int step = reverse ? -1 : 1;
      for (int i = start; i != end; i += step) {
        int im1 = i - step;
        int im2 = i - 2 * step;
        data[i] = a0 * data[i] + a1 * data[im1] + a2 * data[im2] +
          b1 * data[im1] + b2 * data[im2];
      }
    };

  auto expanded = expand(positions);
  applyFilter(expanded, false);
  applyFilter(expanded, true);

  std::vector<Eigen::VectorXd> out(positions.size());
  for (size_t i = 0; i < positions.size(); ++i) {
    out[i] = expanded[i + buffer_size];
  }
  return out;
}

}  // namespace genie_sim_moveit_plugins
