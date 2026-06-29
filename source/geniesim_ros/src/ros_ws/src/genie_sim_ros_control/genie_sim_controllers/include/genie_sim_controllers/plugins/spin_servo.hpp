#pragma once
#include "genie_sim_controllers/servo_base.hpp"

namespace genie_sim_controllers
{

class SpinServo : public ServoBase
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
  float cmd_vx_max_{1.0f};
  float cmd_vy_max_{1.0f};
  float min_abs_wz_{0.05f};
};

}  // namespace genie_sim_controllers
