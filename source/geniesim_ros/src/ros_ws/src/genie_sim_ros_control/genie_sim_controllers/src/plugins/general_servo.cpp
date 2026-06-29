#include "genie_sim_controllers/plugins/general_servo.hpp"
#include <algorithm>  // std::clamp — explicit include; not guaranteed
                      // to be transitively pulled in by other headers.
#include <cmath>

namespace genie_sim_controllers
{

void GeneralServo::reset() {inner_state_ = InnerState::INIT;}

bool GeneralServo::ready() const {return inner_state_ == InnerState::IDLE;}

std::optional<WheelCommand> GeneralServo::update(
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
  const double vx = twist.vx;
  const double vy = twist.vy;
  const double wz = twist.wz;
  const double wz_abs = std::abs(wz);

  const double rx = 0.5 * common_params_.axis_distance;
  const double ry = 0.5 * common_params_.wheel_distance;
  const std::array<double, 4> rxs = {rx, rx, -rx, -rx};
  const std::array<double, 4> rys = {ry, -ry, ry, -ry};

  WheelCommand wheel_cmd{};

  for (size_t i = 0; i < 4; ++i) {
    double vxi = wz_abs < 1e-9 ? vx : vx - wz * rys[i];
    double vyi = wz_abs < 1e-9 ? vy : vy + wz * rxs[i];
    double vi = std::hypot(vxi, vyi);

    double theta_fwd = std::atan2(vyi, vxi);
    double theta_bck = std::atan2(-vyi, -vxi);
    double theta_prev = wheel_cmd_prev.steer_angles[i];
    double d_fwd = std::abs(theta_fwd - theta_prev);
    double d_bck = std::abs(theta_bck - theta_prev);

    double theta, v;
    if (theta_bck > common_params_.max_steer_angle || theta_bck < -common_params_.max_steer_angle) {
      theta = theta_fwd;
      v = vi;
    } else if (theta_fwd > common_params_.max_steer_angle ||
      theta_fwd < -common_params_.max_steer_angle)
    {
      theta = theta_bck;
      v = -vi;
    } else if (d_fwd <= d_bck) {
      theta = theta_fwd;
      v = vi;
    } else {
      theta = theta_bck;
      v = -vi;
    }

    v = std::clamp(
      v, -(double)common_params_.max_drive_speed,
      (double)common_params_.max_drive_speed);

    wheel_cmd.steer_angles[i] = static_cast<float>(theta);
    wheel_cmd.drive_speeds[i] = static_cast<float>(v / common_params_.wheel_radius);
  }

  return wheel_cmd;
}

}  // namespace genie_sim_controllers
