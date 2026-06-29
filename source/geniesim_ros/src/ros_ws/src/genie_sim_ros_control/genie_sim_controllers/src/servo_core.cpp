#include "genie_sim_controllers/servo_core.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

#include "genie_sim_controllers/plugins/parking_servo.hpp"
#include "genie_sim_controllers/plugins/general_servo.hpp"
#include "genie_sim_controllers/plugins/optimal_servo.hpp"
#include "genie_sim_controllers/plugins/selftest_servo.hpp"
#include "genie_sim_controllers/plugins/spin_servo.hpp"

namespace genie_sim_controllers
{

namespace
{
template<size_t N>
void limit_array(
  std::array<float, N> & cmd,
  const std::array<float, N> & prev,
  float max_delta,
  bool is_sync)
{
  if (is_sync) {
    float max_ratio = 0.0f;
    for (size_t i = 0; i < N; ++i) {
      float delta = std::abs(cmd[i] - prev[i]);
      if (max_delta > 0.0f) {
        max_ratio = std::max(max_ratio, delta / max_delta);
      }
    }
    if (max_ratio > 1.0f) {
      for (size_t i = 0; i < N; ++i) {
        cmd[i] = prev[i] + (cmd[i] - prev[i]) / max_ratio;
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      float delta = cmd[i] - prev[i];
      delta = std::clamp(delta, -max_delta, max_delta);
      cmd[i] = prev[i] + delta;
    }
  }
}
}  // namespace

void ServoCore::configure(const ServoParams & params)
{
  params_ = params;

  servos_.clear();
  auto parking = std::make_shared<ParkingServo>(); parking->initialize(params_.common);
  servos_["ParkingServo"] = parking;
  auto general = std::make_shared<GeneralServo>(); general->initialize(params_.common);
  servos_["GeneralServo"] = general;
  auto optimal = std::make_shared<OptimalServo>(); optimal->initialize(params_.common);
  servos_["OptimalServo"] = optimal;
  auto selftest = std::make_shared<SelftestServo>(); selftest->initialize(params_.common);
  servos_["SelftestServo"] = selftest;
  auto spin = std::make_shared<SpinServo>(); spin->initialize(params_.common);
  servos_["SpinServo"] = spin;

  auto it = servos_.find(params_.default_servo);
  if (it != servos_.end()) {
    active_servo_ = it->second;
    active_servo_name_ = params_.default_servo;
  } else {
    active_servo_ = parking;
    active_servo_name_ = "ParkingServo";
  }
  active_servo_->reset();

  td_steer_.reset();
  td_drive_.reset();
  if (params_.use_steer_ltd) {
    td_steer_ = std::make_unique<td::TrackingDifferentiator<double, 4, 3>>(
      static_cast<double>(params_.common.dt), params_.steer_ltd_r);
  }
  if (params_.use_drive_ltd) {
    td_drive_ = std::make_unique<td::TrackingDifferentiator<double, 4, 3>>(
      static_cast<double>(params_.common.dt), params_.drive_ltd_r);
  }

  reset();
}

void ServoCore::reset()
{
  prev_cmd_ = WheelCommand{};
  state_ = State::INIT;
  init_started_ = false;
  {
    std::lock_guard<std::mutex> lock(mode_mtx_);
    pending_mode_.clear();
  }
  if (active_servo_) {active_servo_->reset();}
}

void ServoCore::setTwist(const TwistCmd & t, double stamp_s)
{
  std::lock_guard<std::mutex> lock(twist_mtx_);
  latest_twist_ = t;
  twist_stamp_s_ = stamp_s;
}

void ServoCore::setWheelState(const WheelState & s, double stamp_s)
{
  std::lock_guard<std::mutex> lock(state_mtx_);
  latest_state_ = s;
  state_stamp_s_ = stamp_s;
}

bool ServoCore::requestMode(const std::string & name)
{
  if (servos_.count(name) == 0) {return false;}
  std::ostringstream oss;
  bool same;
  {
    std::lock_guard<std::mutex> lock(mode_mtx_);
    same = (name == active_servo_name_ && pending_mode_.empty());
    if (!same) {
      pending_mode_ = name;
      oss << "Servo mode change to '" << name
          << "' queued (current: '" << active_servo_name_ << "')";
    }
  }
  if (!same) {logInfo(oss.str());}
  return true;
}

std::optional<WheelCommand> ServoCore::tick(double now_s)
{
  if (state_ == State::INIT) {
    if (!init_started_) {
      init_start_s_ = now_s;
      init_started_ = true;
    }
    const double elapsed_ms = (now_s - init_start_s_) * 1000.0;
    if (elapsed_ms >= params_.init_wait_time_ms) {
      state_ = State::SETTING;
      setting_start_s_ = now_s;
      active_servo_->reset();
      std::ostringstream oss;
      oss << "Init complete, setting servo: " << active_servo_name_;
      logInfo(oss.str());
    }
    return std::nullopt;
  }

  if (state_ == State::SETTING) {
    if (active_servo_->ready()) {
      state_ = State::IDLE;
      std::ostringstream oss;
      oss << "Servo " << active_servo_name_ << " is ready";
      logInfo(oss.str());
    } else {
      const double elapsed_ms = (now_s - setting_start_s_) * 1000.0;
      if (elapsed_ms > params_.set_cmd_timeout_ms) {
        std::ostringstream oss;
        oss << "Servo " << active_servo_name_ << " setting timeout, forcing IDLE";
        logWarn(oss.str());
        state_ = State::IDLE;
      }
    }
  }

  // Honour any queued mode switch once we land in IDLE.
  if (state_ == State::IDLE) {
    std::string pending;
    {
      std::lock_guard<std::mutex> lock(mode_mtx_);
      pending = std::move(pending_mode_);
      pending_mode_.clear();
    }
    if (!pending.empty()) {
      auto it = servos_.find(pending);
      if (it != servos_.end()) {
        active_servo_ = it->second;
        active_servo_name_ = pending;
        active_servo_->reset();
        state_ = State::SETTING;
        setting_start_s_ = now_s;
        std::ostringstream oss;
        oss << "Switching to servo: " << active_servo_name_;
        logInfo(oss.str());
      }
    }
  }

  // Snapshot inputs with timeout filtering.
  std::optional<TwistCmd> twist_opt;
  {
    std::lock_guard<std::mutex> lock(twist_mtx_);
    if (latest_twist_.has_value()) {
      const double age_ms = (now_s - twist_stamp_s_) * 1000.0;
      if (age_ms < params_.twist_timeout_ms) {twist_opt = latest_twist_;}
    }
  }

  std::optional<WheelState> state_opt;
  {
    std::lock_guard<std::mutex> lock(state_mtx_);
    if (latest_state_.has_value()) {
      const double age_ms = (now_s - state_stamp_s_) * 1000.0;
      if (age_ms < params_.state_timeout_ms) {state_opt = latest_state_;}
    }
  }

  auto cmd_opt = active_servo_->update(twist_opt, state_opt, prev_cmd_);
  if (!cmd_opt.has_value()) {return std::nullopt;}

  auto cmd = cmd_opt.value();
  limit(cmd, prev_cmd_);

  if (params_.check_state_cmd_diff && state_opt.has_value()) {
    if (cmdNotApplied(cmd, state_opt.value())) {
      logWarn("Steer cmd/state diff too large, resetting");
      cmd = WheelCommand{};
      for (size_t i = 0; i < 4; ++i) {
        cmd.steer_angles[i] = state_opt->steer_angles[i];
      }
    }
  }

  smooth(cmd);
  prev_cmd_ = cmd;
  return cmd;
}

std::string ServoCore::activeServoName() const
{
  return active_servo_name_;
}

std::vector<ServoCore::LogEvent> ServoCore::drainEvents()
{
  std::lock_guard<std::mutex> lock(events_mtx_);
  std::vector<LogEvent> out;
  out.swap(events_);
  return out;
}

void ServoCore::limit(WheelCommand & cmd, const WheelCommand & prev) const
{
  const float max_delta_angle = params_.common.max_steer_speed * params_.common.dt;
  const float max_delta_speed = params_.common.max_drive_accel * params_.common.dt;
  limit_array<4>(cmd.steer_angles, prev.steer_angles, max_delta_angle, true);
  limit_array<4>(cmd.drive_speeds, prev.drive_speeds, max_delta_speed, true);
}

void ServoCore::smooth(WheelCommand & cmd)
{
  if (params_.use_steer_ltd && td_steer_) {
    Eigen::Vector4d steer_in;
    for (size_t i = 0; i < 4; ++i) {
      steer_in(i) = cmd.steer_angles[i];
    }
    td_steer_->update(steer_in);
    auto x = td_steer_->x();
    auto dx = td_steer_->dx();
    auto ddx = td_steer_->ddx();
    for (size_t i = 0; i < 4; ++i) {
      cmd.steer_angles[i] = static_cast<float>(x(i));
      cmd.steer_speeds[i] = static_cast<float>(dx(i));
      cmd.steer_accels[i] = static_cast<float>(ddx(i));
    }
  }
  if (params_.use_drive_ltd && td_drive_) {
    Eigen::Vector4d drive_in;
    for (size_t i = 0; i < 4; ++i) {
      drive_in(i) = cmd.drive_speeds[i];
    }
    td_drive_->update(drive_in);
    auto x = td_drive_->x();
    auto dx = td_drive_->dx();
    for (size_t i = 0; i < 4; ++i) {
      cmd.drive_speeds[i] = static_cast<float>(x(i));
      cmd.drive_accels[i] = static_cast<float>(dx(i));
    }
  }
}

bool ServoCore::cmdNotApplied(const WheelCommand & cmd, const WheelState & state) const
{
  for (size_t i = 0; i < 4; ++i) {
    if (std::abs(cmd.steer_angles[i] - state.steer_angles[i]) >
      params_.max_steer_state_cmd_diff)
    {
      return true;
    }
  }
  return false;
}

void ServoCore::logInfo(const std::string & msg)
{
  std::lock_guard<std::mutex> lock(events_mtx_);
  events_.push_back({LogLevel::INFO, msg});
}

void ServoCore::logWarn(const std::string & msg)
{
  std::lock_guard<std::mutex> lock(events_mtx_);
  events_.push_back({LogLevel::WARN, msg});
}

}  // namespace genie_sim_controllers
