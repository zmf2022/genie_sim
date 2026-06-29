#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

#include "genie_sim_controllers/servo_base.hpp"
#include "genie_sim_controllers/tracking_differentiator.hpp"

namespace genie_sim_controllers
{

/// Snapshot of every parameter the servo loop reads. Filled by the
/// wrapper (controller-plugin or standalone Node) before calling
/// ``ServoCore::configure``.
struct ServoParams
{
  CommonParams common{};                    // wheel geometry + limits + dt
  std::string default_servo{"ParkingServo"};
  double twist_timeout_ms{1000.0};
  double state_timeout_ms{1000.0};
  double init_wait_time_ms{1000.0};
  double set_cmd_timeout_ms{5000.0};
  bool check_state_cmd_diff{false};
  double max_steer_state_cmd_diff{0.1};
  bool use_steer_ltd{false};
  double steer_ltd_r{30.0};
  bool use_drive_ltd{false};
  double drive_ltd_r{30.0};
};

/// Pure four-wheel-steering servo logic. No rclcpp dependency: the
/// state machine, plugin switching, limiter, and tracking
/// differentiator all live here. The owner (a controller plugin or a
/// standalone Node) is responsible for:
///
///   * timestamping inputs (in seconds — any monotonic source)
///   * driving ``tick(now_s)`` from a timer or controller_manager update
///   * publishing the returned WheelCommand (or skipping when nullopt)
///   * surfacing log messages: ServoCore exposes recent
///     ``drain_pending_events`` so the wrapper can route them through
///     its preferred logger.
///
/// All setters are thread-safe: setTwist / setWheelState / requestMode
/// can be called from subscription callbacks while ``tick`` runs on a
/// timer thread.
class ServoCore
{
public:
  ServoCore() = default;

  /// Build the five built-in servo plugins, optionally allocate the
  /// tracking differentiators, and select the default mode. Idempotent
  /// — calling configure again rebuilds from scratch.
  void configure(const ServoParams & params);

  /// Reset the state machine to INIT and clear any cached previous
  /// command. The active servo's own ``reset()`` is also called.
  void reset();

  /// Push a fresh velocity command (vx, vy, wz) with a wall-clock-ish
  /// stamp in seconds. Out-of-band: also used to age inputs against
  /// ``twist_timeout_ms``.
  void setTwist(const TwistCmd & t, double stamp_s);

  /// Push a fresh wheel-state snapshot.
  void setWheelState(const WheelState & s, double stamp_s);

  /// Request a mode switch by plugin name. Returns false if the name
  /// is not one of the configured plugins.
  bool requestMode(const std::string & name);

  /// Advance the state machine + run the control law one tick.
  /// Returns the wheel command to publish, or nullopt if the loop is
  /// still settling (INIT/SETTING) and shouldn't emit.
  std::optional<WheelCommand> tick(double now_s);

  /// Currently-active plugin name (for log/status display).
  std::string activeServoName() const;

  /// Drain pending log events so the owner can route them through its
  /// rclcpp logger. Events accumulate across calls; this empties the
  /// buffer. Used to avoid printing directly from ServoCore.
  enum class LogLevel { INFO, WARN };
  struct LogEvent
  {
    LogLevel level;
    std::string message;
  };
  std::vector<LogEvent> drainEvents();

private:
  enum class State { INIT, SETTING, IDLE };

  void limit(WheelCommand & cmd, const WheelCommand & prev) const;
  void smooth(WheelCommand & cmd);
  bool cmdNotApplied(const WheelCommand & cmd, const WheelState & state) const;

  void logInfo(const std::string & msg);
  void logWarn(const std::string & msg);

  ServoParams params_{};
  WheelCommand prev_cmd_{};

  std::map<std::string, std::shared_ptr<ServoBase>> servos_;
  std::shared_ptr<ServoBase> active_servo_;
  std::string active_servo_name_;

  mutable std::mutex twist_mtx_;
  std::optional<TwistCmd> latest_twist_;
  double twist_stamp_s_{0.0};

  mutable std::mutex state_mtx_;
  std::optional<WheelState> latest_state_;
  double state_stamp_s_{0.0};

  mutable std::mutex mode_mtx_;
  std::string pending_mode_;

  State state_{State::INIT};
  double setting_start_s_{0.0};
  double init_start_s_{0.0};
  bool init_started_{false};

  std::unique_ptr<td::TrackingDifferentiator<double, 4, 3>> td_steer_;
  std::unique_ptr<td::TrackingDifferentiator<double, 4, 3>> td_drive_;

  mutable std::mutex events_mtx_;
  std::vector<LogEvent> events_;
};

}  // namespace genie_sim_controllers
