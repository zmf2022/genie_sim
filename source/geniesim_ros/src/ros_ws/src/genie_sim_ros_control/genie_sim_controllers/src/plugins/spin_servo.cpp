#include "genie_sim_controllers/plugins/spin_servo.hpp"
#include <algorithm>  // std::clamp — explicit include; not guaranteed
                      // to be transitively pulled in by other headers.
#include <cmath>

namespace genie_sim_controllers
{

void SpinServo::initialize(const CommonParams & params)
{
  setCommonParams(params);
  cmd_vx_max_ = 1.0f;
  cmd_vy_max_ = 1.0f;
  min_abs_wz_ = 0.05f;
}

void SpinServo::reset() {inner_state_ = InnerState::INIT;}

bool SpinServo::ready() const {return inner_state_ == InnerState::IDLE;}

std::optional<WheelCommand> SpinServo::update(
  const std::optional<TwistCmd> & twist_opt,
  const std::optional<WheelState> & wheel_state_opt,
  const WheelCommand & wheel_cmd_prev)
{
  if (inner_state_ == InnerState::INIT) {
    inner_state_ = InnerState::IDLE;
  }

  if (!twist_opt.has_value()) {
    return std::nullopt;
  }

  const auto & twist = twist_opt.value();
  float wz = twist.wz;
  if (std::abs(wz) < min_abs_wz_) {
    WheelCommand wheel_cmd{};
    for (size_t i = 0; i < 4; ++i) {
      wheel_cmd.steer_angles[i] = wheel_cmd_prev.steer_angles[i];
    }
    return wheel_cmd;
  }

  float norm_x = (cmd_vx_max_ > 0.0f) ? std::clamp(twist.vx / cmd_vx_max_, -1.0f, 1.0f) : 0.0f;
  float norm_y = (cmd_vy_max_ > 0.0f) ? std::clamp(twist.vy / cmd_vy_max_, -1.0f, 1.0f) : 0.0f;

  const double rx = 0.5 * common_params_.axis_distance;
  const double ry = 0.5 * common_params_.wheel_distance;

  double cx = norm_x * rx;
  double cy = norm_y * ry;

  const std::array<double, 4> rxs = {rx, rx, -rx, -rx};
  const std::array<double, 4> rys = {ry, -ry, ry, -ry};

  WheelCommand wheel_cmd{};

  for (size_t i = 0; i < 4; ++i) {
    double dx = rxs[i] - cx;
    double dy = rys[i] - cy;
    double r = std::hypot(dx, dy);

    double theta_fwd = std::atan2(dx, -dy);
    double theta_bck = std::atan2(-dx, dy);
    double theta_prev = wheel_cmd_prev.steer_angles[i];
    double d_fwd = std::abs(theta_fwd - theta_prev);
    double d_bck = std::abs(theta_bck - theta_prev);

    double theta, dir;
    if (theta_bck > common_params_.max_steer_angle || theta_bck < -common_params_.max_steer_angle) {
      theta = theta_fwd;
      dir = 1.0;
    } else if (theta_fwd > common_params_.max_steer_angle ||
      theta_fwd < -common_params_.max_steer_angle)
    {
      theta = theta_bck;
      dir = -1.0;
    } else if (d_fwd <= d_bck) {
      theta = theta_fwd;
      dir = 1.0;
    } else {
      theta = theta_bck;
      dir = -1.0;
    }

    double v = dir * wz * r;
    v = std::clamp(
      v, -(double)common_params_.max_drive_speed,
      (double)common_params_.max_drive_speed);

    wheel_cmd.steer_angles[i] = static_cast<float>(theta);
    wheel_cmd.drive_speeds[i] = static_cast<float>(v / common_params_.wheel_radius);
  }

  return wheel_cmd;
}

}  // namespace genie_sim_controllers
