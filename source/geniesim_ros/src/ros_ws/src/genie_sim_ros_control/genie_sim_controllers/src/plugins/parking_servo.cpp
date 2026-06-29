#include "genie_sim_controllers/plugins/parking_servo.hpp"
#include <cmath>

namespace genie_sim_controllers
{

void ParkingServo::reset()
{
  inner_state_ = InnerState::INIT;
  const auto spin_turn_steer_angle = std::atan2(
    common_params_.axis_distance,
    common_params_.wheel_distance);
  idle_steer_angle_ = spin_turn_steer_angle - static_cast<float>(M_PI_2);
  reached_count_ = 0;
}

bool ParkingServo::ready() const {return inner_state_ == InnerState::IDLE;}

std::optional<WheelCommand> ParkingServo::update(
  const std::optional<TwistCmd> &, const std::optional<WheelState> &,
  const WheelCommand & wheel_cmd_prev)
{

  while (true) {
    if (inner_state_ == InnerState::INIT) {
      WheelCommand wheel_cmd{};
      wheel_cmd.steer_angles[FRONT_LEFT] = -idle_steer_angle_;
      wheel_cmd.steer_angles[FRONT_RIGHT] = idle_steer_angle_;
      wheel_cmd.steer_angles[REAR_LEFT] = idle_steer_angle_;
      wheel_cmd.steer_angles[REAR_RIGHT] = -idle_steer_angle_;

      bool is_complete = true;
      for (size_t i = 0; i < 4; ++i) {
        if (std::abs(wheel_cmd.steer_angles[i] - wheel_cmd_prev.steer_angles[i]) > 0.001f) {
          is_complete = false;
          break;
        }
        if (std::abs(wheel_cmd.drive_speeds[i] - wheel_cmd_prev.drive_speeds[i]) > 0.001f) {
          is_complete = false;
          break;
        }
      }

      if (is_complete) {
        if (reached_count_ < 20) {
          ++reached_count_;
        } else {
          reached_count_ = 0;
          inner_state_ = InnerState::IDLE;
          continue;
        }
      } else {
        reached_count_ = 0;
      }

      return wheel_cmd;
    }

    if (inner_state_ == InnerState::IDLE) {
      WheelCommand wheel_cmd{};
      wheel_cmd.steer_angles[FRONT_LEFT] = -idle_steer_angle_;
      wheel_cmd.steer_angles[FRONT_RIGHT] = idle_steer_angle_;
      wheel_cmd.steer_angles[REAR_LEFT] = idle_steer_angle_;
      wheel_cmd.steer_angles[REAR_RIGHT] = -idle_steer_angle_;
      return wheel_cmd;
    }

    return std::nullopt;
  }
}

}  // namespace genie_sim_controllers
