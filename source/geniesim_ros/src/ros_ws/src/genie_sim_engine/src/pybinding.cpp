// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.
//
// pybind11 bridge: the ONLY symbol Python touches on the C++ side.
// Designed so Python retains ownership of Isaac Sim / USD / World, and only
// hands flat numeric snapshots down here; we hand back command dicts and
// own ROS2 publish/subscribe on rclcpp threads.

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "realtime_buffer.hpp"
#include "realtime_ros_node.hpp"
#include "realtime_scheduler.hpp"
#include "stats.hpp"

namespace py = pybind11;

namespace
{

struct Core
{
  std::unique_ptr<gsi::RosBridge> bridge;
  std::unique_ptr<gsi::RenderScheduler> rsched;
  std::unique_ptr<gsi::StatsRing> stats;
  std::int64_t step_count = 0;
  double physics_hz = 100.0;
  double rtf = 1.0;
  double last_stats_time_s = 0.0;
  double stats_interval_s = 1.0;
  std::mutex mu;
};

Core & core()
{
  static Core c;
  return c;
}

}  // namespace

static void init_ros(
  const std::string & node_name,
  const std::string & ns,
  bool fake_slam,
  int executor_threads,
  const std::string & base_frame)
{
  auto & c = core();
  std::lock_guard<std::mutex> lk(c.mu);
  if (c.bridge) {return;}
  gsi::NodeOptions opts;
  opts.node_name = node_name;
  opts.namespace_ = ns;
  opts.fake_slam = fake_slam;
  opts.executor_threads = executor_threads;
  if (!base_frame.empty()) {
    // Override the child_frame_id used by publish_odom (TF + nav_msgs).
    // Required when the URDF declares a base_footprint as the ground-
    // projection frame anchored by the SRDF planar virtual_joint — the
    // simulator then publishes odom -> base_footprint, RSP fills the
    // base_footprint -> base_link static edge from the URDF fixed joint.
    opts.base_frame = base_frame;
  }
  int argc = 0;
  char ** argv = nullptr;
  c.bridge.reset(new gsi::RosBridge(argc, argv, opts));
}

static void init_scheduler(
  double render_target_hz,
  double render_safety_ms,
  double physics_hz,
  double rtf)
{
  auto & c = core();
  std::lock_guard<std::mutex> lk(c.mu);
  c.physics_hz = physics_hz;
  c.rtf = rtf > 0.0 ? rtf : 1.0;
  // Wall-clock physics period = (1/physics_hz) / rtf — render scheduler uses
  // this for min_gap_s_ so renders don't fire faster than the wall-clock
  // physics cadence. RTX denoise typically reuses last-frame data; firing the
  // render below the physics wall-clock period means dropping work.
  double dt_wall = 1.0 / (physics_hz * c.rtf > 0 ? physics_hz * c.rtf : 1.0);
  c.rsched.reset(new gsi::RenderScheduler(render_target_hz * c.rtf, render_safety_ms, dt_wall));
  if (!c.stats) {
    std::size_t cap = static_cast<std::size_t>(physics_hz * c.rtf * 5.0);
    c.stats.reset(new gsi::StatsRing(cap == 0 ? 500 : cap));
  }
}

static void set_topology(
  const std::vector<std::string> & joint_names,
  const std::vector<std::string> & body_frames)
{
  auto & c = core();
  if (!c.bridge) {return;}
  c.bridge->set_topology(joint_names, body_frames);
}

static void publish_clock(double sim_time)
{
  auto & c = core();
  if (!c.bridge) {return;}
  py::gil_scoped_release unlock;
  c.bridge->publish_clock(sim_time);
}

static void publish_rtf(float rtf)
{
  auto & c = core();
  if (!c.bridge) {return;}
  py::gil_scoped_release unlock;
  c.bridge->publish_rtf(rtf);
}

static void publish_joint_states(
  double sim_time,
  py::array_t<double, py::array::c_style | py::array::forcecast> pos,
  py::array_t<double, py::array::c_style | py::array::forcecast> vel)
{
  auto & c = core();
  if (!c.bridge) {return;}
  auto pos_buf = pos.request();
  auto vel_buf = vel.request();
  std::size_t n = static_cast<std::size_t>(pos_buf.size);
  if (static_cast<std::size_t>(vel_buf.size) != n) {
    throw std::runtime_error("publish_joint_states: pos/vel size mismatch");
  }
  const double * p = static_cast<const double *>(pos_buf.ptr);
  const double * v = static_cast<const double *>(vel_buf.ptr);
  py::gil_scoped_release unlock;
  c.bridge->publish_joint_states(sim_time, p, v, n);
}

static void publish_body_tf_render(
  double sim_time,
  py::array_t<double, py::array::c_style | py::array::forcecast> xyzwxyz)
{
  auto & c = core();
  if (!c.bridge) {return;}
  auto buf = xyzwxyz.request();
  if (buf.ndim != 2 || buf.shape[1] != 7) {
    throw std::runtime_error("publish_body_tf_render: expected shape (N, 7)");
  }
  std::size_t n = static_cast<std::size_t>(buf.shape[0]);
  const double * p = static_cast<const double *>(buf.ptr);
  py::gil_scoped_release unlock;
  c.bridge->publish_body_tf_render(sim_time, p, n);
}

static void publish_odom(
  double sim_time,
  py::array_t<double, py::array::c_style | py::array::forcecast> xyzwxyz,
  py::object twist_obj)
{
  auto & c = core();
  if (!c.bridge) {return;}
  auto buf = xyzwxyz.request();
  if (buf.size != 7) {
    throw std::runtime_error("publish_odom: expected 7-vector (x,y,z,qw,qx,qy,qz)");
  }
  const double * p = static_cast<const double *>(buf.ptr);
  const double * t = nullptr;
  py::array_t<double, py::array::c_style | py::array::forcecast> twist_arr;
  if (!twist_obj.is_none()) {
    twist_arr = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(twist_obj);
    auto tbuf = twist_arr.request();
    if (tbuf.size != 6) {
      throw std::runtime_error("publish_odom: twist must be a 6-vector (vx,vy,vz,wx,wy,wz)");
    }
    t = static_cast<const double *>(tbuf.ptr);
  }
  py::gil_scoped_release unlock;
  c.bridge->publish_odom(sim_time, p, t);
}

// === Inline OVRtx visualizer support ========================================
// All called from the OVRtx render thread (separate Python thread).
// GIL is released around the C++ publish path so the physics thread is not
// gated on the OVRtx thread when both publish concurrently.

static double last_sim_time()
{
  auto & c = core();
  if (!c.bridge) {return 0.0;}
  return c.bridge->last_sim_time();
}

static void create_camera_publisher(
  const std::string & topic,
  std::uint32_t width,
  std::uint32_t height,
  bool is_depth)
{
  auto & c = core();
  if (!c.bridge) {return;}
  c.bridge->create_camera_publisher(topic, width, height, is_depth);
}

static bool has_image_subscribers(const std::string & topic)
{
  auto & c = core();
  if (!c.bridge) {return false;}
  return c.bridge->has_image_subscribers(topic);
}

static void publish_camera_image_rgba8(
  double sim_time,
  const std::string & topic,
  const std::string & frame_id,
  py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> data)
{
  auto & c = core();
  if (!c.bridge) {return;}
  auto buf = data.request();
  // Accept (H, W, 4) uint8.
  if (buf.ndim != 3 || buf.shape[2] != 4) {
    throw std::runtime_error(
            "publish_camera_image_rgba8: expected shape (H, W, 4) uint8");
  }
  std::uint32_t height = static_cast<std::uint32_t>(buf.shape[0]);
  std::uint32_t width = static_cast<std::uint32_t>(buf.shape[1]);
  const std::uint8_t * p = static_cast<const std::uint8_t *>(buf.ptr);
  py::gil_scoped_release unlock;
  c.bridge->publish_camera_image_rgba8(sim_time, topic, frame_id, height, width, p);
}

static void publish_camera_image_depth32f(
  double sim_time,
  const std::string & topic,
  const std::string & frame_id,
  py::array_t<float, py::array::c_style | py::array::forcecast> data)
{
  auto & c = core();
  if (!c.bridge) {return;}
  auto buf = data.request();
  // Accept (H, W) or (H, W, 1) float32.
  if (!(buf.ndim == 2 || (buf.ndim == 3 && buf.shape[2] == 1))) {
    throw std::runtime_error(
            "publish_camera_image_depth32f: expected shape (H, W) or (H, W, 1) float32");
  }
  std::uint32_t height = static_cast<std::uint32_t>(buf.shape[0]);
  std::uint32_t width = static_cast<std::uint32_t>(buf.shape[1]);
  const float * p = static_cast<const float *>(buf.ptr);
  py::gil_scoped_release unlock;
  c.bridge->publish_camera_image_depth32f(sim_time, topic, frame_id, height, width, p);
}

static void publish_camera_info(
  double sim_time,
  const std::string & topic,
  const std::string & frame_id,
  std::uint32_t height,
  std::uint32_t width,
  py::array_t<double, py::array::c_style | py::array::forcecast> K,
  py::array_t<double, py::array::c_style | py::array::forcecast> P,
  py::array_t<double, py::array::c_style | py::array::forcecast> R,
  std::vector<double> D,
  const std::string & distortion_model)
{
  auto & c = core();
  if (!c.bridge) {return;}
  auto K_buf = K.request();
  auto P_buf = P.request();
  auto R_buf = R.request();
  if (K_buf.size != 9) {
    throw std::runtime_error("publish_camera_info: K must be a 9-vector (row-major)");
  }
  if (P_buf.size != 12) {
    throw std::runtime_error("publish_camera_info: P must be a 12-vector (row-major)");
  }
  if (R_buf.size != 9) {
    throw std::runtime_error("publish_camera_info: R must be a 9-vector (row-major)");
  }
  std::array<double, 9> K_arr{};
  std::array<double, 12> P_arr{};
  std::array<double, 9> R_arr{};
  std::memcpy(K_arr.data(), K_buf.ptr, 9 * sizeof(double));
  std::memcpy(P_arr.data(), P_buf.ptr, 12 * sizeof(double));
  std::memcpy(R_arr.data(), R_buf.ptr, 9 * sizeof(double));
  py::gil_scoped_release unlock;
  c.bridge->publish_camera_info(
    sim_time, topic, frame_id, height, width,
    K_arr, P_arr, R_arr, D, distortion_model);
}

static void subscribe_camera_pose(const std::string & topic)
{
  auto & c = core();
  if (!c.bridge) {return;}
  c.bridge->subscribe_camera_pose(topic);
}

// Returns the latest free-cam transform as a (1, 4, 4) row-major
// float64 matrix when fresh, or None when no new pose has arrived
// since the last call.  Layout matches OVRtx's USD row-vector
// convention used by ``ovrtx_set_xform_mat`` (translation in the last
// row), ready to feed directly into
// ``ovrtx.AttributeBinding.write([m])`` for a single-prim
// XFORM_MAT4x4 binding.  The pose-to-matrix conversion is done in
// C++ — see RosBridge::subscribe_camera_pose's callback in
// realtime_ros_node.cpp — to match
// ``genie_sim_render::on_free_cam_pose`` byte-for-byte.
static py::object take_free_cam_xform()
{
  auto & c = core();
  if (!c.bridge) {return py::none();}
  std::array<double, 16> buf{};
  bool fresh = false;
  {
    py::gil_scoped_release unlock;
    fresh = c.bridge->take_free_cam_xform(buf.data());
  }
  if (!fresh) {return py::none();}
  py::array_t<double> arr({1, 4, 4});
  std::memcpy(arr.mutable_data(), buf.data(), 16 * sizeof(double));
  return arr;
}

// Return the accumulated command dicts to Python for apply_commands(...).
// Returns: (positions_dict, efforts_dict, steer_dict, drive_dict, stamp)
static py::tuple pop_commands()
{
  auto & c = core();
  std::unordered_map<std::string, double> pos, eff, steer, drive;
  double stamp = 0.0;
  if (c.bridge) {
    c.bridge->commands().swap_out(pos, eff, steer, drive, stamp);
  }
  py::dict dpos, deff, dsteer, ddrive;
  for (auto & kv : pos) {
    dpos[py::str(kv.first)] = kv.second;
  }
  for (auto & kv : eff) {
    deff[py::str(kv.first)] = kv.second;
  }
  for (auto & kv : steer) {
    dsteer[py::str(kv.first)] = kv.second;
  }
  for (auto & kv : drive) {
    ddrive[py::str(kv.first)] = kv.second;
  }
  return py::make_tuple(dpos, deff, dsteer, ddrive, stamp);
}

static bool should_render_decoupled(double now_s, double budget_s)
{
  auto & c = core();
  if (!c.rsched) {return false;}
  return c.rsched->should_render(now_s, budget_s);
}

static void mark_rendered_decoupled(double now_s)
{
  auto & c = core();
  if (!c.rsched) {return;}
  c.rsched->mark_rendered(now_s);
}

static void note_step_timing(
  double step_ms, double solver_ms, double render_ms,
  double publish_ms, double spin_ms, bool did_render, double now_s)
{
  auto & c = core();
  if (!c.stats) {return;}
  c.step_count += 1;
  c.stats->push(step_ms, solver_ms, render_ms, publish_ms, spin_ms, did_render, now_s);
}

static py::object log_stats_if_due(double now_s)
{
  auto & c = core();
  if (!c.stats) {return py::none();}
  if (c.last_stats_time_s == 0.0) {
    c.last_stats_time_s = now_s;
    return py::none();
  }
  if ((now_s - c.last_stats_time_s) < c.stats_interval_s) {
    return py::none();
  }
  c.last_stats_time_s = now_s;
  std::uint64_t rendered = c.rsched ? c.rsched->rendered() : 0;
  std::uint64_t skipped_period = c.rsched ? c.rsched->skipped_period() : 0;
  std::uint64_t skipped_budget = c.rsched ? c.rsched->skipped_budget() : 0;
  std::string s = c.stats->format(
    c.step_count, c.physics_hz, c.rtf,
    rendered, skipped_period, skipped_budget);
  if (s.empty()) {return py::none();}
  return py::cast(s);
}

static bool ok()
{
  auto & c = core();
  return c.bridge && c.bridge->ok();
}

static void shutdown()
{
  auto & c = core();
  std::lock_guard<std::mutex> lk(c.mu);
  if (c.bridge) {
    py::gil_scoped_release unlock;
    c.bridge->request_shutdown();
    c.bridge.reset();
  }
  c.rsched.reset();
  c.stats.reset();
  c.step_count = 0;
  c.last_stats_time_s = 0.0;
}

PYBIND11_MODULE(genie_sim_engine_py, m)
{
  m.doc() = "";  // intentionally empty

  m.def(
    "init_ros", &init_ros,
    py::arg("node_name"), py::arg("namespace") = std::string(""),
    py::arg("fake_slam") = false, py::arg("executor_threads") = 2,
    py::arg("base_frame") = std::string(""));
  m.def(
    "init_scheduler", &init_scheduler,
    py::arg("render_target_hz"), py::arg("render_safety_ms"),
    py::arg("physics_hz"), py::arg("rtf") = 1.0);
  m.def(
    "set_topology", &set_topology,
    py::arg("joint_names"), py::arg("body_frames"));

  m.def("publish_clock", &publish_clock, py::arg("sim_time"));
  m.def("publish_rtf", &publish_rtf, py::arg("rtf"));
  m.def(
    "publish_joint_states", &publish_joint_states,
    py::arg("sim_time"), py::arg("pos"), py::arg("vel"));
  m.def(
    "publish_body_tf_render", &publish_body_tf_render,
    py::arg("sim_time"), py::arg("xyzwxyz"));
  m.def(
    "publish_odom", &publish_odom,
    py::arg("sim_time"), py::arg("xyzwxyz"), py::arg("twist") = py::none());

  m.def("pop_commands", &pop_commands);

  // Inline OVRtx visualizer support.
  m.def("last_sim_time", &last_sim_time);
  m.def(
    "create_camera_publisher", &create_camera_publisher,
    py::arg("topic"), py::arg("width"), py::arg("height"),
    py::arg("is_depth") = false);
  m.def("has_image_subscribers", &has_image_subscribers, py::arg("topic"));
  m.def(
    "publish_camera_image_rgba8", &publish_camera_image_rgba8,
    py::arg("sim_time"), py::arg("topic"), py::arg("frame_id"),
    py::arg("data"));
  m.def(
    "publish_camera_image_depth32f", &publish_camera_image_depth32f,
    py::arg("sim_time"), py::arg("topic"), py::arg("frame_id"),
    py::arg("data"));
  m.def(
    "publish_camera_info", &publish_camera_info,
    py::arg("sim_time"), py::arg("topic"), py::arg("frame_id"),
    py::arg("height"), py::arg("width"),
    py::arg("K"), py::arg("P"), py::arg("R"), py::arg("D"),
    py::arg("distortion_model") = std::string("plumb_bob"));
  m.def("subscribe_camera_pose", &subscribe_camera_pose, py::arg("topic"));
  m.def("take_free_cam_xform", &take_free_cam_xform);

  m.def(
    "should_render_decoupled", &should_render_decoupled,
    py::arg("now_s"), py::arg("budget_s"));
  m.def("mark_rendered_decoupled", &mark_rendered_decoupled, py::arg("now_s"));

  m.def(
    "note_step_timing", &note_step_timing,
    py::arg("step_ms"), py::arg("solver_ms"), py::arg("render_ms"),
    py::arg("publish_ms"), py::arg("spin_ms"), py::arg("did_render"),
    py::arg("now_s"));
  m.def("log_stats_if_due", &log_stats_if_due, py::arg("now_s"));

  m.def("ok", &ok);
  m.def("shutdown", &shutdown);
}
