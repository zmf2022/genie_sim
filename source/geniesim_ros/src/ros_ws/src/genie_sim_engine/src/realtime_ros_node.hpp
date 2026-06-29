// Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
// Author: Genie Sim Team
// Proprietary and confidential.

#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"

#include "image_transport/image_transport.hpp"
#include "realtime_tools/realtime_publisher.hpp"

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rosgraph_msgs/msg/clock.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float32.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

#include "realtime_buffer.hpp"

namespace gsi
{

struct NodeOptions
{
  std::string node_name = "genie_sim_engine";
  std::string namespace_ = "";

  std::string topic_clock = "/clock";
  std::string topic_joint_states = "/joint_states";
  std::string topic_odom = "/odom";
  std::string topic_tf = "/tf";
  std::string topic_tf_render = "/tf_render";
  std::string topic_joint_command = "/joint_command";
  std::string topic_cmd_4ws = "/cmd_4ws";

  std::string odom_frame = "odom";
  std::string base_frame = "base_link";
  std::string map_frame = "map";
  std::string world_frame = "world";

  bool fake_slam = false;

  int executor_threads = 2;
};

class RosBridge
{
public:
  RosBridge(int argc, char ** argv, const NodeOptions & opts);
  ~RosBridge();

  // Configure the topology of the per-tick state vectors. Must be called
  // before the first publish_*. After this, joint_state and body_tf vectors
  // must arrive with sizes that match these names.
  void set_topology(
    const std::vector<std::string> & joint_names,
    const std::vector<std::string> & body_frames);

  // === Publish path (called from Python's GIL-released hot path) ===========
  // sim_time is seconds (float64). joint_pos/joint_vel are deg-converted-to-rad
  // arrays, length = joint_names_.size(). body_xyzwxyz is a flat row-major
  // array (px,py,pz,qw,qx,qy,qz) per body, length = 7*body_frames_.size().
  // base_xyzwxyz is the 7-vector for /odom + odom->base_link.
  // base_twist is the 6-vector (vx, vy, vz, wx, wy, wz) expressed in the
  // base_link (child) frame, matching nav_msgs/Odometry.twist convention.
  // Pass nullptr to leave twist all-zero (e.g. on the very first tick).
  void publish_clock(double sim_time);
  void publish_rtf(float rtf);

  void publish_joint_states(
    double sim_time,
    const double * joint_pos,
    const double * joint_vel,
    std::size_t n);

  void publish_body_tf_render(
    double sim_time,
    const double * body_xyzwxyz,
    std::size_t n_bodies);

  void publish_odom(
    double sim_time,
    const double * base_xyzwxyz,
    const double * base_twist);

  // === Inline OVRtx visualizer support =====================================
  // Methods below are called from the OVRtx render thread (separate from the
  // physics tick thread). All publish paths go through realtime_tools::
  // RealtimePublisher just like the physics-thread publishers, so the
  // try_lock + unlockAndPublish pattern keeps the OVRtx thread non-blocking
  // and the actual DDS publish hops onto each RealtimePublisher's dedicated
  // thread. Subscriber counts use rclcpp::Publisher::get_subscription_count
  // which is thread-safe.

  // Latest sim_time stamped by publish_clock(). Atomic so the OVRtx thread
  // can read without a mutex while the physics thread writes.
  double last_sim_time() const;

  // Lazily construct (or look up) RGB / depth publishers per topic. Called
  // once per camera at startup; idempotent. width/height are recorded for
  // logging only — image dimensions are taken from each frame's DLTensor.
  void create_camera_publisher(
    const std::string & topic,
    std::uint32_t width,
    std::uint32_t height,
    bool is_depth);

  // Subscriber-count gate (matches render_node.cpp:313 pattern).
  // Returns false when the publisher does not exist or has zero subscribers.
  bool has_image_subscribers(const std::string & topic) const;

  // Hot-path publishers. Called from the OVRtx render thread.
  // Image data layout is the OVRtx convention used by render_node.cpp:
  //   * RGB:   shape (H, W, 4), uint8, encoding "rgba8", step = W*4
  //   * depth: shape (H, W),    float32, encoding "32FC1", step = W*4
  // The pointer must remain valid for the duration of this call; the
  // RealtimePublisher copies into its message buffer.
  void publish_camera_image_rgba8(
    double sim_time,
    const std::string & topic,
    const std::string & frame_id,
    std::uint32_t height,
    std::uint32_t width,
    const std::uint8_t * data);

  void publish_camera_image_depth32f(
    double sim_time,
    const std::string & topic,
    const std::string & frame_id,
    std::uint32_t height,
    std::uint32_t width,
    const float * data);

  // Camera info (intrinsics). 9-vector K (row-major), 12-vector P,
  // 9-vector R, distortion vector D (typically 5 plumb_bob coefs).
  void publish_camera_info(
    double sim_time,
    const std::string & topic,
    const std::string & frame_id,
    std::uint32_t height,
    std::uint32_t width,
    const std::array<double, 9> & K,
    const std::array<double, 12> & P,
    const std::array<double, 9> & R,
    const std::vector<double> & D,
    const std::string & distortion_model);

  // Free-camera pose subscription used by the inline OVRtx visualizer.
  // The subscription runs on the existing executor thread (rclcpp); the
  // callback converts the incoming ``geometry_msgs::PoseStamped`` directly
  // into a 4x4 row-major ``mat44d`` (USD row-vector convention, translation
  // in the last row) using the same formula as
  // ``genie_sim_render::on_free_cam_pose`` (render_node.cpp:531-540).  The
  // OVRtx render thread calls ``take_free_cam_xform(...)`` once per frame
  // to drain the slot.  Idempotent — ``subscribe_camera_pose`` only creates
  // a subscription on first call.
  void subscribe_camera_pose(const std::string & topic);

  // Drain the latest free-cam transform.  Returns ``true`` and fills
  // ``out_mat44d`` (length 16, row-major float64) when a fresh pose has
  // arrived since the last call; returns ``false`` otherwise.  This
  // matches the layout the OVRtx ``omni:xform`` writer expects (USD
  // row-vector convention; same as ``ovrtx_xform_matrix44d_t.v[]`` used
  // by ``ovrtx_set_xform_mat`` in the cross-process renderer).
  bool take_free_cam_xform(double * out_mat44d);

  // === Command path =========================================================
  RealtimeCommandBuffer & commands() {return cmd_store_;}

  bool ok() const;
  void request_shutdown();

private:
  // rclcpp owned init/shutdown wrapper to be safe across reload.
  static std::atomic<bool> rcl_inited_;

  NodeOptions opts_;
  std::shared_ptr<rclcpp::Context> ctx_;
  std::shared_ptr<rclcpp::Node> node_;
  std::vector<std::string> joint_names_;
  std::vector<std::string> body_frames_;

  // Publishers are wrapped in realtime_tools::RealtimePublisher so that the
  // RT physics tick only does try_lock + msg_ assignment + unlockAndPublish;
  // the actual DDS publish happens on a dedicated non-RT thread owned by
  // each RealtimePublisher. This bounds the publish-side latency seen by
  // the physics loop and keeps allocations off the hot path.
  using RTPubClock = realtime_tools::RealtimePublisher<rosgraph_msgs::msg::Clock>;
  using RTPubJoint = realtime_tools::RealtimePublisher<sensor_msgs::msg::JointState>;
  using RTPubOdom = realtime_tools::RealtimePublisher<nav_msgs::msg::Odometry>;
  using RTPubTF = realtime_tools::RealtimePublisher<tf2_msgs::msg::TFMessage>;
  using RTPubCamInfo = realtime_tools::RealtimePublisher<sensor_msgs::msg::CameraInfo>;
  using RTPubFloat32 = realtime_tools::RealtimePublisher<std_msgs::msg::Float32>;

  std::shared_ptr<RTPubClock> pub_clock_;
  std::shared_ptr<RTPubJoint> pub_joint_;
  std::shared_ptr<RTPubOdom> pub_odom_;
  std::shared_ptr<RTPubTF> pub_tf_;
  std::shared_ptr<RTPubTF> pub_tf_render_;
  std::shared_ptr<RTPubFloat32> pub_rtf_;

  // Camera publisher map. Built lazily by create_camera_publisher().
  // Reads (publish + has_image_subscribers) from the OVRtx render thread are
  // protected by cam_pubs_mu_; writes happen at startup (single-threaded).
  //
  // We use ``image_transport::Publisher`` for the image side so that
  // ``<topic>/image_raw/compressed`` (and ``compressedDepth`` for depth
  // topics) are advertised automatically — same pattern as
  // genie_sim_render's RosImagePublisherPlugin.  Subscriber-count gating
  // uses ``image_transport::Publisher::getNumSubscribers()`` which sums
  // across all advertised transports (raw + compressed + theora + …),
  // so a subscriber on /compressed alone still keeps the gate open.
  //
  // CameraInfo retains the realtime_tools fast-path because it is small
  // and emitted alongside every image at the same rate.
  mutable std::mutex cam_pubs_mu_;
  std::unordered_map<std::string, image_transport::Publisher> pub_images_it_;
  std::unordered_map<std::string, std::shared_ptr<RTPubCamInfo>> pub_caminfos_;

  // Atomic stamp for cross-thread sim-time reads from the OVRtx thread.
  // Written by publish_clock(); read by last_sim_time().
  std::atomic<double> last_sim_time_{0.0};

  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_joint_cmd_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_cmd_4ws_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_cam_pose_;

  // Free-cam mat44d latch.  The subscription callback (executor thread)
  // converts the incoming PoseStamped into a row-major 4x4 double matrix
  // (USD row-vector convention) and stores it here.  The OVRtx render
  // thread calls take_free_cam_xform() once per frame to drain.
  // ``cam_pose_dirty_`` lets the reader skip the mutex on the common
  // no-update path.
  mutable std::mutex cam_pose_mu_;
  std::atomic<bool> cam_pose_dirty_{false};
  std::array<double, 16> cam_pose_mat44d_{};

  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread spin_thread_;

  RealtimeCommandBuffer cmd_store_;
};

}  // namespace gsi
