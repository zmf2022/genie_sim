#pragma once

#include <array>
#include <optional>
#include <string>

namespace genie_sim_controllers
{

struct WheelState
{
  std::array<float, 4> steer_angles{};
  std::array<float, 4> steer_speeds{};
  std::array<float, 4> drive_speeds{};
};

struct WheelCommand
{
  std::array<float, 4> steer_angles{};
  std::array<float, 4> steer_speeds{};
  std::array<float, 4> steer_accels{};
  std::array<float, 4> drive_speeds{};
  std::array<float, 4> drive_accels{};
};

struct CommonParams
{
  float wheel_distance{1.0f};
  float axis_distance{1.0f};
  float wheel_radius{0.1f};
  float max_steer_angle{0.523599f};
  float max_steer_speed{0.5f};
  float max_drive_speed{1.0f};
  float max_drive_accel{20.0f};
  float dt{0.005f};
};

struct TwistCmd
{
  float vx{0.0f};
  float vy{0.0f};
  float wz{0.0f};
};

class ServoBase
{
public:
  virtual ~ServoBase() = default;

  virtual void initialize(const CommonParams & params) = 0;

  void setCommonParams(const CommonParams & params) {common_params_ = params;}

  virtual void reset() {}

  virtual bool ready() const {return false;}

  virtual std::optional<WheelCommand> update(
    const std::optional<TwistCmd> & twist_opt,
    const std::optional<WheelState> & wheel_state_opt,
    const WheelCommand & wheel_cmd_prev) = 0;

protected:
  static constexpr int FRONT_LEFT = 0;
  static constexpr int FRONT_RIGHT = 1;
  static constexpr int REAR_LEFT = 2;
  static constexpr int REAR_RIGHT = 3;

  CommonParams common_params_;
};

}  // namespace genie_sim_controllers
