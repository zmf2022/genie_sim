#pragma once

#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <pluginlib/class_loader.hpp>

#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>
#include <ovrtx/ovrtx_types.h>

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <deque>
#include <unordered_map>

namespace genie_sim_render
{

class IImagePublisherPlugin;

struct CameraConfig
{
  std::string path;
  std::string topic;
  // Depth ROS topic (from manifest's depth_topic field; scene yaml's topic.depth).
  // If empty, depth is not published and DistanceToImagePlaneSD render var is not authored on the render layer.
  std::string depth_topic;
  // DDS target topic (from scene json's dds_topic field). If empty, the DDS plugin skips this camera.
  std::string dds_topic;
  // DDS target topic for depth (from scene json's dds_depth_topic field). If empty, depth is not mirrored to DDS.
  std::string dds_depth_topic;
  int width{1280};
  int height{800};
  double fx{610.0};
  double fy{610.0};
  double cx{640.0};
  double cy{400.0};
  double k1{0.0};
  double k2{0.0};
  double p1{0.0};
  double p2{0.0};
  double k3{0.0};
  // 4th fisheye coefficient (OpenCV fisheye / equidistant uses k1..k4, no p1/p2). Only meaningful
  // when distortion_model is the fisheye model; ignored for plumb_bob.
  double k4{0.0};
  // CameraInfo distortion model string published to ROS. "plumb_bob" (default, Brown-Conrady) or
  // "equidistant" (OpenCV fisheye). Derived from the manifest "model" field by load_manifest.
  std::string distortion_model{"plumb_bob"};
  // Raw lens model string from the manifest ("opencvFisheye" for fisheye cams, empty otherwise).
  // Used to route the DDS image side: fisheye -> JPEG vendor IMAGE packet, else CompressedImage.
  std::string model;
  double min_range{0.01};
  double max_range{10000.0};

  std::string render_product_path;
  bool is_free_cam{false};
};

struct LidarConfig
{
  std::string render_product_path;
  std::string topic;
  std::string frame_id;
  // Sensor frame the per-lidar cloud is published in (e.g. livox_front), derived from the topic
  // basename. Also the child frame of the static base_link->sensor TF. Matches the rmagine path.
  std::string sensor_frame;
  // Body link the lidar is parented to (e.g. base_link); its world TF is mirrored onto the
  // render-layer lidar copy so ovrtx raycasts from the right pose.
  std::string parent_body;
  // Static base_link <- sensor transform: points come out in the sensor frame and are
  // transformed into the body frame before publishing.
  double bl_T_s_xyz[3]{0.0, 0.0, 0.0};
  double bl_T_s_wxyz[4]{1.0, 0.0, 0.0, 0.0};
  // Wall-time of the last publish, for the 10 Hz safety throttle.
  std::chrono::steady_clock::time_point last_publish{};
};

struct FrameTiming
{
  double step_ms{0.0};
  double wait_ms{0.0};
  double fetch_ms{0.0};
  double map_ms{0.0};
  double publish_ms{0.0};
  double total_ms{0.0};
  double interval_ms{0.0};
};

class RenderNode : public rclcpp::Node
{
public:
  using Clock = std::chrono::steady_clock;
  using TimePoint = Clock::time_point;

  explicit RenderNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~RenderNode() override;

private:
  void load_manifest(const std::string & manifest_path);
  void init_renderer();
  void setup_cameras();
  void setup_lidars();
  void load_plugins();
  void shutdown_renderer();

  void on_tf_render(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  void on_free_cam_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  void render_timer_callback();
  void stats_timer_callback();

  void publish_camera_images(const ovrtx_render_product_set_outputs_t & outputs);
  void publish_lidar_pointclouds(const ovrtx_render_product_set_outputs_t & outputs);
  void publish_lidar_tf();
  void publish_timing(const FrameTiming & t);

  ovrtx_render_var_output_handle_t find_output_for_product(
    const ovrtx_render_product_set_outputs_t & outputs,
    size_t product_index,
    const char * render_var_name);

  ovrtx_render_var_output_handle_t find_output_by_path(
    const ovrtx_render_product_set_outputs_t & outputs,
    const std::string & product_path,
    const char * render_var_name);

  std::string usd_path_;
  std::string robot_usda_;
  std::string render_layer_usda_;
  std::string robot_prefix_;
  double render_fps_;
  std::vector<std::string> prim_paths_;
  std::vector<CameraConfig> cameras_;
  std::vector<LidarConfig> lidars_;
  // ROS param 'lidar' (default false): lidar is opt-in for the ovrtx renderer. When false the
  // manifest's lidars are ignored, so no lidar render products are stepped and no PointCloud2
  // is published. Set 'lidar: true' in the launcher yaml's render_ovrtx params to enable.
  bool lidar_enabled_{false};
  std::string merged_lidar_topic_;

  ovrtx_renderer_t * renderer_{nullptr};
  ovrtx_usd_handle_t scene_handle_{0};
  ovrtx_usd_handle_t robot_handle_{0};
  ovrtx_usd_handle_t camera_layer_handle_{0};

  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_render_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr free_cam_sub_;
  std::string free_cam_prim_path_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr timing_pub_;
  rclcpp::TimerBase::SharedPtr render_timer_;
  rclcpp::TimerBase::SharedPtr stats_timer_;

  std::vector<ovrtx_xform_matrix44d_t> transforms_;
  std::unordered_map<std::string, std::string> cam_body_to_render_path_;
  std::unordered_map<std::string, std::string> lidar_body_to_render_path_;
  bool scene_loaded_{false};
  // Latest /tf_render stamp (sim time). Published clouds/images use this so their timestamps
  // share the TF tree's time base, else RViz's tf2 message filter drops them.
  double latest_tf_time_{0.0};

  // base_link's latest world pose (from /tf_render), used to publish the dynamic base_link->world
  // TF (= its inverse) so consumers can place the base_link/sensor-frame clouds as the robot moves.
  bool has_base_link_world_{false};
  bool tf_static_published_{false};
  double base_link_world_q_[4]{0.0, 0.0, 0.0, 1.0};  // x, y, z, w
  double base_link_world_t_[3]{0.0, 0.0, 0.0};

  TimePoint last_frame_time_{};
  uint64_t frame_count_{0};
  uint64_t drop_count_{0};
  // First-frame compile / shader-cache warmup can take 30-90s. Operators
  // see blank images and assume the renderer is hung. Track the first
  // wait so render_timer_callback can spawn a 1Hz heartbeat thread.
  bool first_frame_done_{false};
  std::deque<FrameTiming> timing_history_;
  static constexpr size_t kHistorySize = 300;

  std::shared_ptr<pluginlib::ClassLoader<IImagePublisherPlugin>> plugin_loader_;
  std::vector<std::shared_ptr<IImagePublisherPlugin>> image_plugins_;
};

}  // namespace genie_sim_render
