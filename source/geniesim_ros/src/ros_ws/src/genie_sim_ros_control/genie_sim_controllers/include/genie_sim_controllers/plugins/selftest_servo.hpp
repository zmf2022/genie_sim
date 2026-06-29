#pragma once
#include "genie_sim_controllers/servo_base.hpp"

namespace genie_sim_controllers
{

class SelftestServo : public ServoBase
{
public:
  void initialize(const CommonParams & params) override;
  void reset() override;
  bool ready() const override;
  std::optional<WheelCommand> update(
    const std::optional<TwistCmd> & twist_opt,
    const std::optional<WheelState> & wheel_state_opt,
    const WheelCommand & wheel_cmd_prev) override;

private:
  enum class InnerState { INIT, IDLE };
  InnerState inner_state_{InnerState::INIT};
  float test_wheel_speed_{0.5f};
  float test_spin_duration_{1.0f};
  bool flag_{false};
  float spin_turn_steer_angle_{0.0f};
  int test_state_{0};
  int steps_{0};
};

}  // namespace genie_sim_controllers
