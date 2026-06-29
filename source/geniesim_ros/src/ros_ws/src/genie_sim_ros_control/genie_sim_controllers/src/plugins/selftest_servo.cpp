#include "genie_sim_controllers/plugins/selftest_servo.hpp"
#include <cmath>

namespace genie_sim_controllers
{

void SelftestServo::initialize(const CommonParams & params)
{
  setCommonParams(params);
  test_wheel_speed_ = 0.5f;
  test_spin_duration_ = 1.0f;
}

void SelftestServo::reset()
{
  inner_state_ = InnerState::INIT;
  flag_ = false;
  test_state_ = 0;
  steps_ = 0;
  spin_turn_steer_angle_ = std::atan2(common_params_.axis_distance, common_params_.wheel_distance);
}

bool SelftestServo::ready() const {return inner_state_ == InnerState::IDLE;}

std::optional<WheelCommand> SelftestServo::update(
  const std::optional<TwistCmd> &,
  const std::optional<WheelState> & wheel_state_opt,
  const WheelCommand & wheel_cmd_prev)
{
  if (inner_state_ == InnerState::INIT) {
    inner_state_ = InnerState::IDLE;
  }

  WheelCommand wheel_cmd{};
  float dir = flag_ ? -1.0f : 1.0f;

  int spin_steps = static_cast<int>(test_spin_duration_ / common_params_.dt);
  int decel_steps =
    static_cast<int>(test_wheel_speed_ / common_params_.max_drive_accel / common_params_.dt);

  if (test_state_ == 0) {
    wheel_cmd.steer_angles[FRONT_LEFT] = -spin_turn_steer_angle_ * dir;
    wheel_cmd.steer_angles[FRONT_RIGHT] = spin_turn_steer_angle_ * dir;
    wheel_cmd.steer_angles[REAR_LEFT] = spin_turn_steer_angle_ * dir;
    wheel_cmd.steer_angles[REAR_RIGHT] = -spin_turn_steer_angle_ * dir;

    bool reached = true;
    for (size_t i = 0; i < 4; ++i) {
      if (std::abs(wheel_cmd.steer_angles[i] - wheel_cmd_prev.steer_angles[i]) > 0.01f) {
        reached = false;
        break;
      }
    }
    if (reached) {
      test_state_ = 1;
      steps_ = 0;
    }
  } else if (test_state_ == 1) {
    for (size_t i = 0; i < 4; ++i) {
      wheel_cmd.steer_angles[i] = wheel_cmd_prev.steer_angles[i];
    }
    wheel_cmd.drive_speeds[FRONT_LEFT] = test_wheel_speed_;
    wheel_cmd.drive_speeds[FRONT_RIGHT] = test_wheel_speed_;
    wheel_cmd.drive_speeds[REAR_LEFT] = test_wheel_speed_;
    wheel_cmd.drive_speeds[REAR_RIGHT] = test_wheel_speed_;
    ++steps_;
    if (steps_ >= spin_steps) {
      test_state_ = 2;
      steps_ = 0;
    }
  } else if (test_state_ == 2) {
    for (size_t i = 0; i < 4; ++i) {
      wheel_cmd.steer_angles[i] = wheel_cmd_prev.steer_angles[i];
    }
    float frac = 1.0f - static_cast<float>(steps_) / static_cast<float>(std::max(1, decel_steps));
    frac = std::max(0.0f, frac);
    wheel_cmd.drive_speeds[FRONT_LEFT] = test_wheel_speed_ * frac;
    wheel_cmd.drive_speeds[FRONT_RIGHT] = test_wheel_speed_ * frac;
    wheel_cmd.drive_speeds[REAR_LEFT] = test_wheel_speed_ * frac;
    wheel_cmd.drive_speeds[REAR_RIGHT] = test_wheel_speed_ * frac;
    ++steps_;
    if (steps_ >= decel_steps) {
      test_state_ = 0;
      steps_ = 0;
      flag_ = !flag_;
    }
  }

  return wheel_cmd;
}

}  // namespace genie_sim_controllers
