#include "genie_sim_render/render_node.hpp"
#include "genie_sim_render/image_publisher_plugin.h"

#include <nlohmann/json.hpp>
#include <ovrtx/ovrtx_attributes.h>

#include <cstring>
#include <sstream>
#include <fstream>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>

namespace genie_sim_render
{

using json = nlohmann::json;
namespace fs = std::filesystem;

template<typename ResultT>
static bool check_error(ResultT const & result, const char * op, rclcpp::Logger logger)
{
  if (result.status == OVRTX_API_ERROR) {
    ovx_string_t err = ovrtx_get_last_error();
    if (err.ptr && err.length > 0) {
      RCLCPP_ERROR(
        logger, "%s failed: %.*s", op,
        static_cast<int>(err.length), err.ptr);
    } else {
      RCLCPP_ERROR(logger, "%s failed (unknown error)", op);
    }
    return true;
  }
  return false;
}

static double ms_between(RenderNode::TimePoint a, RenderNode::TimePoint b)
{
  return std::chrono::duration<double, std::milli>(b - a).count();
}

static bool wait_for_op(
  ovrtx_renderer_t * renderer, uint64_t op_index, rclcpp::Logger logger,
  const char * op_name, bool fatal = true)
{
  ovrtx_op_wait_result_t wait_result{};
  while (ovrtx_wait_op(
      renderer, op_index,
      ovrtx_timeout_t{0}, &wait_result).status == OVRTX_API_TIMEOUT)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (wait_result.num_error_ops > 0) {
    std::ostringstream details;
    for (size_t i = 0; i < wait_result.num_error_ops; ++i) {
      ovrtx_op_id_t op_id = wait_result.error_op_ids[i];
      ovx_string_t err = ovrtx_get_last_op_error(op_id);
      std::string err_str = (err.ptr && err.length > 0) ?
        std::string(err.ptr, err.length) : "(no detail)";
      if (i > 0) {details << "; ";}
      details << "op_id=" << op_id << ": " << err_str;
    }
    if (fatal) {
      RCLCPP_ERROR(
        logger, "%s had %zu errors: %s",
        op_name, wait_result.num_error_ops, details.str().c_str());
    } else {
      RCLCPP_WARN(
        logger, "%s had %zu errors (non-fatal): %s",
        op_name, wait_result.num_error_ops, details.str().c_str());
    }
    return false;
  }
  return true;
}

// A composite PointCloud render var exposes named per-channel DLTensors. Match by name
// (ovrtx strings carry ptr+length, not necessarily null-terminated).
static const ovrtx_render_var_tensor_t * find_pc_tensor(
  const ovrtx_render_var_output_t & output, const char * name)
{
  const size_t len = std::strlen(name);
  for (size_t i = 0; i < output.num_tensors; ++i) {
    const ovrtx_render_var_tensor_t & t = output.tensors[i];
    if (t.name && t.name->ptr && t.name->length == len &&
      std::strncmp(t.name->ptr, name, len) == 0)
    {
      return &t;
    }
  }
  return nullptr;
}

RenderNode::RenderNode(const rclcpp::NodeOptions & options)
: Node("genie_sim_render", options)
{
  declare_parameter<std::string>("stage_manifest", "");
  declare_parameter<double>("render_fps", 30.0);
  declare_parameter<std::string>("prim_paths", "");
  declare_parameter<std::string>("ovrtx_root", "");
  declare_parameter<bool>("lidar", false);
  declare_parameter<std::vector<std::string>>(
    "plugin", std::vector<std::string>{"genie_sim_render/RosImagePublisherPlugin"});

  std::string manifest_path;
  get_parameter("stage_manifest", manifest_path);
  get_parameter("render_fps", render_fps_);
  get_parameter("lidar", lidar_enabled_);

  std::string prim_paths_str;
  get_parameter("prim_paths", prim_paths_str);
  if (!prim_paths_str.empty()) {
    std::istringstream ss(prim_paths_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) {
        prim_paths_.push_back(token);
      }
    }
  }

  if (manifest_path.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter 'stage_manifest' is required");
    throw std::runtime_error("stage_manifest parameter is required");
  }

  load_manifest(manifest_path);

  timing_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("~/render_timing", 10);

  tf_render_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
    "/tf_render", rclcpp::SensorDataQoS(),
    std::bind(&RenderNode::on_tf_render, this, std::placeholders::_1));

  if (!free_cam_prim_path_.empty()) {
    // Align with the latched QoS of RViz's ViewCameraPosePublisherDisplay:
    // KeepLast(1) + Reliable + TransientLocal; new subscribers immediately receive the last pose.
    rclcpp::QoS free_cam_qos(rclcpp::KeepLast(1));
    free_cam_qos.reliable();
    free_cam_qos.transient_local();
    free_cam_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/free_cam_pose", free_cam_qos,
      std::bind(&RenderNode::on_free_cam_pose, this, std::placeholders::_1));
  }

  init_renderer();
  load_plugins();
  setup_cameras();
  setup_lidars();

  auto period = std::chrono::duration<double>(1.0 / render_fps_);
  render_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&RenderNode::render_timer_callback, this));

  stats_timer_ = create_wall_timer(
    std::chrono::seconds(1),
    std::bind(&RenderNode::stats_timer_callback, this));

  RCLCPP_INFO(
    get_logger(), "Render node initialized (%.1f fps, %zu cameras)",
    render_fps_, cameras_.size());
}

RenderNode::~RenderNode()
{
  shutdown_renderer();
}

void RenderNode::load_manifest(const std::string & manifest_path)
{
  fs::path mp(manifest_path);
  if (mp.is_relative()) {
    mp = fs::current_path() / mp;
  }
  if (!fs::exists(mp)) {
    throw std::runtime_error("Manifest not found: " + mp.string());
  }

  std::ifstream f(mp);
  json manifest = json::parse(f);

  // Resolve every manifest-supplied path against ``base_path`` (falling back
  // to the manifest's own directory when ``base_path`` is missing). The
  // assemble_scene producer stores paths *relative to* ``base_path``, so
  // resolving here lets the render node be launched from any cwd.
  fs::path base_dir;
  if (manifest.contains("base_path") && !manifest["base_path"].get<std::string>().empty()) {
    base_dir = fs::path(manifest["base_path"].get<std::string>());
  } else {
    base_dir = mp.parent_path();
  }
  auto resolve_rel = [&base_dir](const std::string & p) -> std::string {
      if (p.empty()) {return p;}
      fs::path candidate(p);
      if (candidate.is_absolute()) {return candidate.string();}
      return (base_dir / candidate).lexically_normal().string();
    };

  if (manifest.contains("scene_usda")) {
    usd_path_ = resolve_rel(manifest["scene_usda"].get<std::string>());
  } else if (manifest.contains("usd_path")) {
    usd_path_ = resolve_rel(manifest["usd_path"].get<std::string>());
  } else {
    throw std::runtime_error("Manifest missing both 'scene_usda' and 'usd_path'");
  }
  robot_usda_ = resolve_rel(manifest.value("robot_usda", ""));
  render_layer_usda_ = resolve_rel(manifest.value("render_layer_usda", ""));
  robot_prefix_ = manifest.value("robot_prefix", "");
  free_cam_prim_path_ = manifest.value("free_cam_prim_path", "");

  RCLCPP_INFO(
    get_logger(),
    "load_manifest: manifest=%s base_dir=%s scene=%s robot=%s render_layer=%s",
    mp.string().c_str(), base_dir.string().c_str(),
    usd_path_.c_str(),
    robot_usda_.empty() ? "(none)" : robot_usda_.c_str(),
    render_layer_usda_.empty() ? "(none)" : render_layer_usda_.c_str());

  if (!fs::exists(usd_path_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Scene USD not found after base_path resolution: '%s' (manifest=%s, base_dir=%s)",
      usd_path_.c_str(), mp.string().c_str(), base_dir.string().c_str());
    throw std::runtime_error("Scene USD not found: " + usd_path_);
  }
  if (!robot_usda_.empty() && !fs::exists(robot_usda_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Robot USD not found after base_path resolution: '%s' (base_dir=%s)",
      robot_usda_.c_str(), base_dir.string().c_str());
    throw std::runtime_error("Robot USD not found: " + robot_usda_);
  }
  if (!render_layer_usda_.empty() && !fs::exists(render_layer_usda_)) {
    RCLCPP_ERROR(
      get_logger(),
      "Render layer USDA not found after base_path resolution: '%s' (base_dir=%s)",
      render_layer_usda_.c_str(), base_dir.string().c_str());
    throw std::runtime_error("Render layer USDA not found: " + render_layer_usda_);
  }

  for (size_t cam_idx = 0;
    cam_idx < (manifest.contains("cameras") ? manifest["cameras"].size() : 0);
    ++cam_idx)
  {
    const auto & cam_json = manifest["cameras"][cam_idx];
    CameraConfig cam;
    cam.path = cam_json.at("path").get<std::string>();
    cam.topic = cam_json.at("topic").get<std::string>();
    cam.depth_topic = cam_json.value("depth_topic", std::string{});
    cam.dds_topic = cam_json.value("dds_topic", std::string{});
    cam.dds_depth_topic = cam_json.value("dds_depth_topic", std::string{});
    cam.render_product_path = cam_json.value(
      "render_product_path", "/RenderOVRTX/Cam_" + std::to_string(cam_idx));
    cam.width = cam_json.value("width", 1280);
    cam.height = cam_json.value("height", 800);
    cam.fx = cam_json.value("fx", 610.0);
    cam.fy = cam_json.value("fy", 610.0);
    cam.cx = cam_json.value("cx", static_cast<double>(cam.width) / 2.0);
    cam.cy = cam_json.value("cy", static_cast<double>(cam.height) / 2.0);
    cam.k1 = cam_json.value("k1", 0.0);
    cam.k2 = cam_json.value("k2", 0.0);
    cam.p1 = cam_json.value("p1", 0.0);
    cam.p2 = cam_json.value("p2", 0.0);
    cam.k3 = cam_json.value("k3", 0.0);
    cam.k4 = cam_json.value("k4", 0.0);
    // Map the assemble-time lens model to a ROS CameraInfo distortion_model. opencvFisheye
    // (OpenCV fisheye / Kannala-Brandt) -> "equidistant"; everything else -> "plumb_bob".
    {
      std::string model = cam_json.value("model", std::string{});
      cam.model = model;
      cam.distortion_model = (model == "opencvFisheye") ? "equidistant" : "plumb_bob";
    }
    cam.min_range = cam_json.value("min_range", 0.01);
    cam.max_range = cam_json.value("max_range", 10000.0);
    cam.is_free_cam = cam_json.value("is_free_cam", false);
    cameras_.push_back(std::move(cam));
  }

  if (!render_layer_usda_.empty()) {
    for (const auto & cam : cameras_) {
      if (cam.is_free_cam) {continue;}
      auto slash_pos = cam.path.find('/');
      if (slash_pos != std::string::npos) {
        std::string body_name = cam.path.substr(0, slash_pos);
        std::string render_path = "/RenderOVRTX/Cameras/" + body_name;
        cam_body_to_render_path_[body_name] = render_path;
      }
    }
  }

  for (const auto & ljson : (lidar_enabled_ && manifest.contains("lidars") ?
    manifest["lidars"] : json::array()))
  {
    LidarConfig l;
    l.render_product_path = ljson.value("render_product_path", std::string{});
    l.topic = ljson.value("topic", std::string{});
    l.frame_id = ljson.value("frame_id", std::string{"base_link"});
    l.parent_body = ljson.value("parent_body", std::string{});
    // Sensor frame = topic basename (e.g. /lidar/livox_front -> livox_front), matching the
    // rmagine path. The per-lidar cloud is published in this frame; merged stays base_link.
    l.sensor_frame = l.topic.substr(l.topic.find_last_of('/') + 1);
    if (ljson.contains("base_link_T_sensor")) {
      const auto & T = ljson["base_link_T_sensor"];
      if (T.contains("xyz") && T["xyz"].size() == 3) {
        for (int k = 0; k < 3; ++k) {
          l.bl_T_s_xyz[k] = T["xyz"][k].get<double>();
        }
      }
      if (T.contains("wxyz") && T["wxyz"].size() == 4) {
        for (int k = 0; k < 4; ++k) {
          l.bl_T_s_wxyz[k] = T["wxyz"][k].get<double>();
        }
      }
    }
    if (l.render_product_path.empty() || l.topic.empty()) {continue;}
    lidars_.push_back(std::move(l));
  }
  if (!lidars_.empty()) {
    merged_lidar_topic_ = "/lidar/merged";
    for (const auto & l : lidars_) {
      if (!l.parent_body.empty()) {
        lidar_body_to_render_path_[l.parent_body] = "/RenderOVRTX/Lidars/" + l.parent_body;
      }
    }
  }

  RCLCPP_INFO(
    get_logger(),
    "Manifest loaded: scene=%s, render_layer=%s, cameras=%zu, lidars=%zu",
    usd_path_.c_str(),
    render_layer_usda_.empty() ? "(none)" : render_layer_usda_.c_str(),
    cameras_.size(),
    lidars_.size());
}

void RenderNode::init_renderer()
{
  RCLCPP_INFO(
    get_logger(),
    "Creating ovrtx renderer (first run compiles shaders, may be slow)...");

  std::string ovrtx_root;
  get_parameter("ovrtx_root", ovrtx_root);

  std::vector<ovrtx_config_entry_t> entries;
  if (!ovrtx_root.empty()) {
    ovx_string_t root_str{ovrtx_root.c_str(), ovrtx_root.size()};
    entries.push_back(ovrtx_config_entry_binary_package_root_path(root_str));
  }

  ovx_string_t log_path{"/tmp/ovrtx.log", 14};
  entries.push_back(ovrtx_config_entry_log_file_path(log_path));
  ovx_string_t log_level{"verbose", 7};
  entries.push_back(ovrtx_config_entry_log_level(log_level));

  // Required by the lidar sensor pipeline; only enable it when lidar is active so
  // camera-only scenes don't pay the motion-BVH build cost.
  if (lidar_enabled_) {
    entries.push_back(ovrtx_config_entry_enable_motion_bvh(true));
  }

  ovrtx_config_t config{};
  config.entries = entries.empty() ? nullptr : entries.data();
  config.entry_count = entries.size();

  ovrtx_result_t result = ovrtx_create_renderer(&config, &renderer_);
  if (check_error(result, "create_renderer", get_logger())) {
    throw std::runtime_error("Failed to create ovrtx renderer");
  }
  RCLCPP_INFO(get_logger(), "Renderer created");

  auto open_root_usd = [&](const std::string & path, const char * label) {
      RCLCPP_INFO(get_logger(), "Opening %s as root: %s", label, path.c_str());
      ovx_string_t file_str{path.c_str(), path.size()};
      ovrtx_enqueue_result_t er = ovrtx_open_usd_from_file(renderer_, file_str);
      if (check_error(er, label, get_logger())) {
        throw std::runtime_error(std::string("Failed to enqueue ") + label);
      }
      if (!wait_for_op(renderer_, er.op_index, get_logger(), label, true)) {
        throw std::runtime_error(std::string(label) + " loading failed");
      }
      RCLCPP_INFO(get_logger(), "%s loaded", label);
    };

  auto add_usd_reference = [&](const std::string & path, const std::string & prefix,
      ovrtx_usd_handle_t & handle, const char * label, bool fatal) {
      ovx_string_t file_str{path.c_str(), path.size()};
      ovx_string_t pfx{prefix.c_str(), prefix.size()};
      RCLCPP_INFO(
        get_logger(), "Adding %s reference: %s (prefix=%s)", label, path.c_str(),
        prefix.c_str());
      ovrtx_enqueue_result_t er =
        ovrtx_add_usd_reference_from_file(renderer_, file_str, pfx, &handle);
      if (check_error(er, label, get_logger())) {
        if (fatal) {
          throw std::runtime_error(std::string("Failed to enqueue ") + label);
        }
        RCLCPP_WARN(get_logger(), "%s enqueue failed (non-fatal)", label);
        return;
      }
      if (!wait_for_op(renderer_, er.op_index, get_logger(), label, fatal)) {
        if (fatal) {
          throw std::runtime_error(std::string(label) + " loading failed");
        }
        RCLCPP_WARN(get_logger(), "%s loaded with errors (non-fatal)", label);
      }
      RCLCPP_INFO(get_logger(), "%s loaded", label);
    };

  open_root_usd(usd_path_, "scene");
  scene_handle_ = 0;  // Root layer; no add-reference handle.

  if (!robot_usda_.empty()) {
    std::string robot_pfx = "/" + robot_prefix_;
    add_usd_reference(robot_usda_, robot_pfx, robot_handle_, "robot", false);
  }

  if (!render_layer_usda_.empty()) {
    add_usd_reference(
      render_layer_usda_, "/RenderOVRTX", camera_layer_handle_,
      "render_layer", true);
  } else {
    RCLCPP_WARN(
      get_logger(),
      "No render_layer_usda in manifest. Run genie_sim_engine's assemble_scene first.");
  }

  scene_loaded_ = true;
}

void RenderNode::setup_cameras()
{
  if (cameras_.empty()) {
    RCLCPP_WARN(get_logger(), "No cameras configured");
    return;
  }

  for (size_t i = 0; i < cameras_.size(); ++i) {
    const auto & cam = cameras_[i];
    RCLCPP_INFO(
      get_logger(), "Camera[%zu]: product=%s -> %s (%dx%d)%s",
      i, cam.render_product_path.c_str(), cam.topic.c_str(),
      cam.width, cam.height,
      cam.depth_topic.empty() ? "" : " +depth");
    for (auto & plugin : image_plugins_) {
      // Register ROS->DDS mapping first, then create the publisher; if the mapping is empty the plugin skips this camera.
      if (!cam.dds_topic.empty()) {
        // Fisheye cams must match the real robot's DDS format: a JPEG-payload
        // vendor IMAGE packet (encoding=JPEG, color_format=RGB, bit_depth=8)
        // rather than the sensor_msgs CompressedImage that register_dds_mapping
        // builds. Routed by the lens model.
        if (cam.model == "opencvFisheye") {
          plugin->register_dds_image_packet_mapping(cam.topic, cam.dds_topic);
        } else {
          plugin->register_dds_mapping(cam.topic, cam.dds_topic);
        }
      }
      // Depth-side mapping is dispatched separately so the plugin can build a depth-typed
      // publisher (e.g. Z16 raw over the vendor IMAGE packet) instead of a JPEG one.
      if (!cam.depth_topic.empty() && !cam.dds_depth_topic.empty()) {
        plugin->register_dds_depth_mapping(cam.depth_topic, cam.dds_depth_topic);
      }
      plugin->create_camera_publisher(
        cam.topic, static_cast<uint32_t>(cam.width),
        static_cast<uint32_t>(cam.height));
      // Depth reuses the same plugin interface; the ROS side publishes <depth_topic>/image_raw + camera_info,
      // and the DDS side picks it up via the depth mapping registered above (when dds_depth_topic was set).
      if (!cam.depth_topic.empty()) {
        plugin->create_camera_publisher(
          cam.depth_topic, static_cast<uint32_t>(cam.width),
          static_cast<uint32_t>(cam.height));
      }
    }
  }
}

void RenderNode::setup_lidars()
{
  if (lidars_.empty()) {
    return;
  }
  for (const auto & l : lidars_) {
    RCLCPP_INFO(
      get_logger(), "Lidar: product=%s -> %s (frame=%s, body=%s)",
      l.render_product_path.c_str(), l.topic.c_str(),
      l.frame_id.c_str(), l.parent_body.c_str());
    for (auto & plugin : image_plugins_) {
      plugin->create_pointcloud_publisher(l.topic);
    }
  }
  if (!merged_lidar_topic_.empty()) {
    for (auto & plugin : image_plugins_) {
      plugin->create_pointcloud_publisher(merged_lidar_topic_);
    }
    RCLCPP_INFO(
      get_logger(), "Merged lidar topic: %s (base_link frame)",
      merged_lidar_topic_.c_str());
  }
  // Set up the /tf_static + /tf broadcasters; the actual transforms are published from
  // on_tf_render once base_link's world pose is known (parity with the rmagine path).
  for (auto & plugin : image_plugins_) {
    plugin->create_tf_publisher();
  }
}

void RenderNode::load_plugins()
{
  plugin_loader_ =
    std::make_shared<pluginlib::ClassLoader<IImagePublisherPlugin>>(
    "genie_sim_render", "genie_sim_render::IImagePublisherPlugin");

  const auto plugin_names = get_parameter("plugin").as_string_array();
  for (const auto & plugin_name : plugin_names) {
    if (plugin_name.empty()) {continue;}
    try {
      auto plugin = plugin_loader_->createSharedInstance(plugin_name);
      plugin->initialize(this);
      image_plugins_.push_back(plugin);
      RCLCPP_INFO(get_logger(), "Loaded image publisher plugin: %s", plugin_name.c_str());
    } catch (const pluginlib::PluginlibException & e) {
      RCLCPP_ERROR(
        get_logger(), "Failed to load plugin '%s': %s",
        plugin_name.c_str(), e.what());
    }
  }
}

void RenderNode::shutdown_renderer()
{
  if (renderer_) {
    ovrtx_destroy_renderer(renderer_);
    renderer_ = nullptr;
    RCLCPP_INFO(get_logger(), "Renderer destroyed");
  }
}

void RenderNode::on_tf_render(const tf2_msgs::msg::TFMessage::SharedPtr msg)
{
  // Track the sim-time stamp so published clouds/images share the TF tree's time base.
  if (!msg->transforms.empty()) {
    const auto & s = msg->transforms[0].header.stamp;
    latest_tf_time_ = static_cast<double>(s.sec) + static_cast<double>(s.nanosec) * 1e-9;
  }

  if (!scene_loaded_ || robot_prefix_.empty()) {
    return;
  }

  RCLCPP_INFO_ONCE(
    get_logger(), "tf_render received (%zu transforms), robot_prefix=/%s",
    msg->transforms.size(), robot_prefix_.c_str());

  size_t tf_count = msg->transforms.size();
  std::vector<std::string> paths;
  std::vector<ovrtx_xform_matrix44d_t> xforms;
  paths.reserve(tf_count + cam_body_to_render_path_.size());
  xforms.reserve(tf_count + cam_body_to_render_path_.size());

  for (size_t i = 0; i < tf_count; ++i) {
    const auto & tf = msg->transforms[i];
    const std::string & body_name = tf.child_frame_id;

    double qx = tf.transform.rotation.x;
    double qy = tf.transform.rotation.y;
    double qz = tf.transform.rotation.z;
    double qw = tf.transform.rotation.w;
    double tx = tf.transform.translation.x;
    double ty = tf.transform.translation.y;
    double tz = tf.transform.translation.z;

    // Capture base_link's world pose so we can publish the dynamic base_link->world TF below.
    if (body_name == "base_link") {
      base_link_world_q_[0] = qx; base_link_world_q_[1] = qy;
      base_link_world_q_[2] = qz; base_link_world_q_[3] = qw;
      base_link_world_t_[0] = tx; base_link_world_t_[1] = ty; base_link_world_t_[2] = tz;
      has_base_link_world_ = true;
    }

    double xx = qx * qx, yy = qy * qy, zz = qz * qz;
    double xy = qx * qy, xz = qx * qz, yz = qy * qz;
    double wx = qw * qx, wy = qw * qy, wz = qw * qz;

    ovrtx_xform_matrix44d_t xform{};
    auto & m = xform.v;
    m[0] = 1.0 - 2.0 * (yy + zz); m[1] = 2.0 * (xy + wz);       m[2] = 2.0 * (xz - wy);
    m[3] = 0.0;
    m[4] = 2.0 * (xy - wz);       m[5] = 1.0 - 2.0 * (xx + zz); m[6] = 2.0 * (yz + wx);
    m[7] = 0.0;
    m[8] = 2.0 * (xz + wy);       m[9] = 2.0 * (yz - wx);       m[10] = 1.0 - 2.0 * (xx + yy);
    m[11] = 0.0;
    m[12] = tx;                     m[13] = ty;                     m[14] = tz;
    m[15] = 1.0;

    std::string prim_path;
    if (body_name.rfind("/World", 0) == 0 || body_name.rfind("/", 0) == 0) {
      prim_path = body_name;
    } else {
      prim_path = "/" + robot_prefix_ + "/" + body_name;
    }

    paths.push_back(prim_path);
    xforms.push_back(xform);

    auto it = cam_body_to_render_path_.find(body_name);
    if (it != cam_body_to_render_path_.end()) {
      paths.push_back(it->second);
      xforms.push_back(xform);
    }

    auto lit = lidar_body_to_render_path_.find(body_name);
    if (lit != lidar_body_to_render_path_.end()) {
      paths.push_back(lit->second);
      xforms.push_back(xform);
    }
  }

  size_t count = paths.size();
  std::vector<ovx_string_t> ovx_paths(count);
  for (size_t i = 0; i < count; ++i) {
    ovx_paths[i] = {paths[i].c_str(), paths[i].size()};
  }

  ovrtx_enqueue_result_t wr = ovrtx_set_xform_mat(
    renderer_, ovx_paths.data(), count, xforms.data());
  if (wr.status != OVRTX_API_SUCCESS) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "tf_render set_xform_mat failed");
  }
  // base_link->world is published with the lidar cloud (~10 Hz, same stamp) in
  // publish_lidar_pointclouds, not here. Publishing it on every /tf_render (~85 Hz)
  // overran the /tf publish queue (queue_size 5) and dropped historical TF samples, so
  // a late cloud stamp had no matching transform and the cloud jittered around Z during
  // rotation. The rmagine path likewise emits base_link->world from its 10 Hz lidar loop.
}

// Publish the lidar TF tree (parity with the rmagine path): a one-shot static
// base_link->sensor edge per lidar, and a dynamic base_link->world edge (= inverse of base_link's
// world pose) every /tf_render so consumers can place the base_link/sensor-frame clouds as the
// robot moves. Only active when lidar publishing is enabled.
void RenderNode::publish_lidar_tf()
{
  if (!lidar_enabled_ || lidars_.empty() || !has_base_link_world_) {
    return;
  }

  // Static base_link -> sensor (the mount), published once.
  if (!tf_static_published_) {
    std::vector<TransformMsg> statics;
    statics.reserve(lidars_.size());
    for (const auto & l : lidars_) {
      if (l.sensor_frame.empty()) {continue;}
      TransformMsg t;
      t.parent_frame = "base_link";
      t.child_frame = l.sensor_frame;
      t.timestamp = latest_tf_time_;
      t.tx = l.bl_T_s_xyz[0]; t.ty = l.bl_T_s_xyz[1]; t.tz = l.bl_T_s_xyz[2];
      // manifest stores wxyz; TransformMsg holds xyzw.
      t.qw = l.bl_T_s_wxyz[0]; t.qx = l.bl_T_s_wxyz[1];
      t.qy = l.bl_T_s_wxyz[2]; t.qz = l.bl_T_s_wxyz[3];
      statics.push_back(std::move(t));
    }
    if (!statics.empty()) {
      for (auto & plugin : image_plugins_) {
        plugin->publish_transforms(statics, true);
      }
      tf_static_published_ = true;
    }
  }

  // Dynamic base_link -> world = inverse of base_link's world pose (world->base_link).
  const double qx = base_link_world_q_[0], qy = base_link_world_q_[1];
  const double qz = base_link_world_q_[2], qw = base_link_world_q_[3];
  const double tx = base_link_world_t_[0], ty = base_link_world_t_[1], tz = base_link_world_t_[2];
  // Inverse rotation = conjugate; inverse translation = -R(conj) * t.
  const double iqx = -qx, iqy = -qy, iqz = -qz, iqw = qw;
  const double xx = iqx * iqx, yy = iqy * iqy, zz = iqz * iqz;
  const double xy = iqx * iqy, xz = iqx * iqz, yz = iqy * iqz;
  const double wx = iqw * iqx, wy = iqw * iqy, wz = iqw * iqz;
  const double r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz), r02 = 2 * (xz + wy);
  const double r10 = 2 * (xy + wz), r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
  const double r20 = 2 * (xz - wy), r21 = 2 * (yz + wx), r22 = 1 - 2 * (xx + yy);
  TransformMsg dyn;
  dyn.parent_frame = "base_link";
  dyn.child_frame = "world";
  dyn.timestamp = latest_tf_time_;
  dyn.tx = -(r00 * tx + r01 * ty + r02 * tz);
  dyn.ty = -(r10 * tx + r11 * ty + r12 * tz);
  dyn.tz = -(r20 * tx + r21 * ty + r22 * tz);
  dyn.qx = iqx; dyn.qy = iqy; dyn.qz = iqz; dyn.qw = iqw;
  std::vector<TransformMsg> dynamics{dyn};
  for (auto & plugin : image_plugins_) {
    plugin->publish_transforms(dynamics, false);
  }
}

void RenderNode::on_free_cam_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!scene_loaded_ || free_cam_prim_path_.empty()) {
    return;
  }

  RCLCPP_INFO_ONCE(
    get_logger(), "free_cam_pose received, writing to %s",
    free_cam_prim_path_.c_str());

  const auto & pose = msg->pose;
  double qx = pose.orientation.x;
  double qy = pose.orientation.y;
  double qz = pose.orientation.z;
  double qw = pose.orientation.w;
  double tx = pose.position.x;
  double ty = pose.position.y;
  double tz = pose.position.z;

  double xx = qx * qx, yy = qy * qy, zz = qz * qz;
  double xy = qx * qy, xz = qx * qz, yz = qy * qz;
  double wx = qw * qx, wy = qw * qy, wz = qw * qz;

  ovrtx_xform_matrix44d_t xform{};
  auto & m = xform.v;
  m[0] = 1.0 - 2.0 * (yy + zz); m[1] = 2.0 * (xy + wz);       m[2] = 2.0 * (xz - wy);
  m[3] = 0.0;
  m[4] = 2.0 * (xy - wz);       m[5] = 1.0 - 2.0 * (xx + zz); m[6] = 2.0 * (yz + wx);
  m[7] = 0.0;
  m[8] = 2.0 * (xz + wy);       m[9] = 2.0 * (yz - wx);       m[10] = 1.0 - 2.0 * (xx + yy);
  m[11] = 0.0;
  m[12] = tx;                     m[13] = ty;                     m[14] = tz;
  m[15] = 1.0;

  ovx_string_t prim_path{free_cam_prim_path_.c_str(), free_cam_prim_path_.size()};
  ovrtx_enqueue_result_t wr = ovrtx_set_xform_mat(renderer_, &prim_path, 1, &xform);
  if (wr.status != OVRTX_API_SUCCESS) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "free_cam set_xform_mat failed: %d", wr.status);
  }
}

void RenderNode::render_timer_callback()
{
  if (!scene_loaded_ || cameras_.empty()) {
    return;
  }

  auto frame_start = Clock::now();
  FrameTiming timing{};

  if (last_frame_time_.time_since_epoch().count() > 0) {
    timing.interval_ms = ms_between(last_frame_time_, frame_start);
  }
  last_frame_time_ = frame_start;

  // --- step (all cameras in one call) ---
  auto t0 = Clock::now();

  std::vector<ovx_string_t> rp_strings;
  rp_strings.reserve(cameras_.size() + lidars_.size());
  for (const auto & cam : cameras_) {
    rp_strings.push_back({cam.render_product_path.c_str(), cam.render_product_path.size()});
  }
  for (const auto & l : lidars_) {
    rp_strings.push_back({l.render_product_path.c_str(), l.render_product_path.size()});
  }

  ovrtx_render_product_set_t render_products{};
  render_products.render_products = rp_strings.data();
  render_products.num_render_products = rp_strings.size();

  ovrtx_step_result_handle_t step_handle = 0;
  double dt = 1.0 / render_fps_;
  ovrtx_enqueue_result_t enqueue_result =
    ovrtx_step(renderer_, render_products, dt, &step_handle);
  if (check_error(enqueue_result, "step", get_logger())) {
    ++drop_count_;
    return;
  }

  auto t1 = Clock::now();
  timing.step_ms = ms_between(t0, t1);

  // --- wait ---
  // First call to wait_op also covers shader compilation + JIT warmup, which
  // can block for 30-90s on a fresh machine. Without a heartbeat the operator
  // sees blank images, no logs, and assumes the renderer is hung. Spawn a 1Hz
  // ticker that prints elapsed time until the wait returns; subsequent frames
  // skip this entirely.
  std::atomic<bool> heartbeat_stop{false};
  std::thread heartbeat_thread;
  if (!first_frame_done_) {
    RCLCPP_INFO(get_logger(), "================================================================");
    RCLCPP_INFO(get_logger(), "OVRTX first run on this machine: compiling shaders now.");
    RCLCPP_INFO(get_logger(), "This typically takes 30-90 seconds. Render output will stay");
    RCLCPP_INFO(get_logger(), "blank until compilation finishes; the cache will make");
    RCLCPP_INFO(get_logger(), "subsequent runs instant.");
    RCLCPP_INFO(get_logger(), "================================================================");
    auto warmup_t0 = Clock::now();
    auto logger = get_logger();
    heartbeat_thread = std::thread(
      [&heartbeat_stop, warmup_t0, logger]() {
        while (!heartbeat_stop.load(std::memory_order_acquire)) {
          std::this_thread::sleep_for(std::chrono::seconds(1));
          if (heartbeat_stop.load(std::memory_order_acquire)) {
            break;
          }
          double elapsed = ms_between(warmup_t0, Clock::now()) / 1000.0;
          RCLCPP_INFO(
            logger,
            "first frame: still compiling shaders / warming GPU "
            "(%.0fs elapsed) — normal on first run, subsequent runs hit the cache",
            elapsed);
        }
      });
  }

  ovrtx_op_wait_result_t wait_result{};
  ovrtx_result_t result = ovrtx_wait_op(
    renderer_, enqueue_result.op_index, ovrtx_timeout_infinite, &wait_result);

  if (heartbeat_thread.joinable()) {
    heartbeat_stop.store(true, std::memory_order_release);
    heartbeat_thread.join();
    first_frame_done_ = true;
    RCLCPP_INFO(get_logger(), "OVRTX first frame complete — renderer is live.");
  }

  if (check_error(result, "wait_op(step)", get_logger())) {
    ++drop_count_;
    ovrtx_destroy_results(renderer_, step_handle);
    return;
  }

  auto t2 = Clock::now();
  timing.wait_ms = ms_between(t1, t2);

  // --- fetch ---
  ovrtx_render_product_set_outputs_t outputs{};
  result = ovrtx_fetch_results(renderer_, step_handle, ovrtx_timeout_infinite, &outputs);
  if (check_error(result, "fetch_results", get_logger())) {
    ++drop_count_;
    ovrtx_destroy_results(renderer_, step_handle);
    return;
  }

  auto t3 = Clock::now();
  timing.fetch_ms = ms_between(t2, t3);

  // --- map + publish all cameras ---
  publish_camera_images(outputs);
  publish_lidar_pointclouds(outputs);

  auto t4 = Clock::now();
  timing.map_ms = ms_between(t3, t4);

  ovrtx_destroy_results(renderer_, step_handle);

  timing.total_ms = ms_between(frame_start, Clock::now());
  ++frame_count_;

  timing_history_.push_back(timing);
  if (timing_history_.size() > kHistorySize) {
    timing_history_.pop_front();
  }

  publish_timing(timing);
}

void RenderNode::publish_camera_images(
  const ovrtx_render_product_set_outputs_t & outputs)
{
  // Use the TF tree's time base (sim time from /tf_render) so RViz can transform the output;
  // fall back to wall clock only before the first /tf_render arrives.
  double timestamp = latest_tf_time_ > 0.0 ? latest_tf_time_ : now().seconds();

  // PROF (temporary, remove after diagnosing FPS): per-stage timing buckets accumulated over all
  // cameras this frame, printed once/sec at the end. Splits the coarse map_ms bucket into GPU
  // readback (ovrtx_map) vs memcpy vs per-plugin publish (plugin0=ROS, plugin1=DDS/JPEG).
  double prof_map_ms = 0.0;
  double prof_copy_ms = 0.0;
  std::vector<double> prof_plugin_ms(image_plugins_.size(), 0.0);
  std::vector<double> prof_cam_map_ms(cameras_.size(), 0.0);
  std::vector<double> prof_cam_pub_ms(cameras_.size(), 0.0);

  for (size_t cam_idx = 0; cam_idx < cameras_.size(); ++cam_idx) {
    const auto & cam = cameras_[cam_idx];

    ovrtx_render_var_output_handle_t ldr_handle =
      find_output_for_product(outputs, cam_idx, "LdrColor");
    if (ldr_handle == static_cast<ovrtx_render_var_output_handle_t>(-1)) {
      continue;
    }

    ovrtx_map_output_description_t map_desc{};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_render_var_output_t rendered_output{};
    auto _prof_t0 = Clock::now();
    ovrtx_result_t result = ovrtx_map_render_var_output(
      renderer_, ldr_handle, &map_desc, ovrtx_timeout_infinite, &rendered_output);
    {
      double _dt = ms_between(_prof_t0, Clock::now());
      prof_map_ms += _dt;
      prof_cam_map_ms[cam_idx] += _dt;
    }
    if (result.status != OVRTX_API_SUCCESS) {
      continue;
    }

    if (rendered_output.status == OVRTX_EVENT_COMPLETED &&
      rendered_output.num_tensors >= 1 && rendered_output.tensors[0].dl)
    {
      const DLTensor & tensor = *rendered_output.tensors[0].dl;
      int height = static_cast<int>(tensor.shape[0]);
      int width = static_cast<int>(tensor.shape[1]);
      int channels = 4;

      CameraImageMsg img_msg;
      img_msg.frame_id = cam.path;
      img_msg.timestamp = timestamp;
      img_msg.width = static_cast<uint32_t>(width);
      img_msg.height = static_cast<uint32_t>(height);
      img_msg.encoding = "rgba8";
      img_msg.step = static_cast<uint32_t>(width * channels);
      auto _prof_cp0 = Clock::now();
      img_msg.data.resize(height * width * channels);
      std::memcpy(img_msg.data.data(), tensor.data, img_msg.data.size());
      prof_copy_ms += ms_between(_prof_cp0, Clock::now());

      CameraInfoMsg info_msg;
      info_msg.frame_id = cam.path;
      info_msg.timestamp = timestamp;
      info_msg.width = static_cast<uint32_t>(cam.width);
      info_msg.height = static_cast<uint32_t>(cam.height);
      info_msg.distortion_model = cam.distortion_model;
      info_msg.d = (cam.distortion_model == "equidistant")
        ? std::vector<double>{cam.k1, cam.k2, cam.k3, cam.k4}
        : std::vector<double>{cam.k1, cam.k2, cam.p1, cam.p2, cam.k3};
      info_msg.k = {cam.fx, 0.0, cam.cx, 0.0, cam.fy, cam.cy, 0.0, 0.0, 1.0};
      info_msg.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      info_msg.p = {cam.fx, 0.0, cam.cx, 0.0, 0.0, cam.fy, cam.cy, 0.0, 0.0, 0.0, 1.0, 0.0};

      for (size_t _p = 0; _p < image_plugins_.size(); ++_p) {
        auto _prof_p0 = Clock::now();
        image_plugins_[_p]->publish_camera_image(cam.topic, img_msg);
        image_plugins_[_p]->publish_camera_info(cam.topic, info_msg);
        double _dt = ms_between(_prof_p0, Clock::now());
        prof_plugin_ms[_p] += _dt;
        prof_cam_pub_ms[cam_idx] += _dt;
      }
    }

    ovrtx_cuda_sync_t no_sync{};
    ovrtx_unmap_render_var_output(renderer_, rendered_output.map_handle, no_sync);

    // ----- Depth map (DistanceToImagePlaneSD, float32, unit: meters along optical axis) -----
    // Only when scene yaml has a non-empty topic.depth does assemble_scene.py author
    // DistanceToImagePlaneSD render var on the render layer; this mirrors that condition.
    if (cam.depth_topic.empty()) {
      continue;
    }

    ovrtx_render_var_output_handle_t depth_handle =
      find_output_for_product(outputs, cam_idx, "DistanceToImagePlaneSD");
    if (depth_handle == static_cast<ovrtx_render_var_output_handle_t>(-1)) {
      continue;
    }

    ovrtx_map_output_description_t depth_map_desc{};
    depth_map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_render_var_output_t depth_output{};
    auto _prof_td0 = Clock::now();
    ovrtx_result_t depth_result = ovrtx_map_render_var_output(
      renderer_, depth_handle, &depth_map_desc, ovrtx_timeout_infinite, &depth_output);
    {
      double _dt = ms_between(_prof_td0, Clock::now());
      prof_map_ms += _dt;
      prof_cam_map_ms[cam_idx] += _dt;
    }
    if (depth_result.status != OVRTX_API_SUCCESS) {
      continue;
    }

    if (depth_output.status == OVRTX_EVENT_COMPLETED &&
      depth_output.num_tensors >= 1 && depth_output.tensors[0].dl)
    {
      const DLTensor & dt = *depth_output.tensors[0].dl;
      // Shape (H, W, 1), dtype=float32 (OVRTX contract), maps directly to sensor_msgs/Image 32FC1.
      int dheight = static_cast<int>(dt.shape[0]);
      int dwidth = static_cast<int>(dt.shape[1]);

      CameraImageMsg depth_msg;
      depth_msg.frame_id = cam.path;
      depth_msg.timestamp = timestamp;
      depth_msg.width = static_cast<uint32_t>(dwidth);
      depth_msg.height = static_cast<uint32_t>(dheight);
      depth_msg.encoding = "32FC1";
      depth_msg.step = static_cast<uint32_t>(dwidth * sizeof(float));
      auto _prof_dcp0 = Clock::now();
      depth_msg.data.resize(static_cast<size_t>(dheight) * dwidth * sizeof(float));
      std::memcpy(depth_msg.data.data(), dt.data, depth_msg.data.size());
      prof_copy_ms += ms_between(_prof_dcp0, Clock::now());

      // CameraInfo shares the same path as RGB (same camera intrinsics); constructed separately to avoid coupling with the LDR scope above.
      CameraInfoMsg depth_info;
      depth_info.frame_id = cam.path;
      depth_info.timestamp = timestamp;
      depth_info.width = static_cast<uint32_t>(cam.width);
      depth_info.height = static_cast<uint32_t>(cam.height);
      depth_info.distortion_model = cam.distortion_model;
      depth_info.d = (cam.distortion_model == "equidistant")
        ? std::vector<double>{cam.k1, cam.k2, cam.k3, cam.k4}
        : std::vector<double>{cam.k1, cam.k2, cam.p1, cam.p2, cam.k3};
      depth_info.k = {cam.fx, 0.0, cam.cx, 0.0, cam.fy, cam.cy, 0.0, 0.0, 1.0};
      depth_info.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      depth_info.p = {cam.fx, 0.0, cam.cx, 0.0, 0.0, cam.fy, cam.cy, 0.0, 0.0, 0.0, 1.0, 0.0};

      for (size_t _p = 0; _p < image_plugins_.size(); ++_p) {
        auto _prof_dp0 = Clock::now();
        image_plugins_[_p]->publish_camera_image(cam.depth_topic, depth_msg);
        image_plugins_[_p]->publish_camera_info(cam.depth_topic, depth_info);
        double _dt = ms_between(_prof_dp0, Clock::now());
        prof_plugin_ms[_p] += _dt;
        prof_cam_pub_ms[cam_idx] += _dt;
      }
    }

    ovrtx_unmap_render_var_output(renderer_, depth_output.map_handle, no_sync);
  }

  // PROF (temporary): once/sec breakdown of the publish stage (= the coarse map_ms bucket).
  double _prof_pub_sum = 0.0;
  for (double v : prof_plugin_ms) {_prof_pub_sum += v;}
  std::ostringstream _prof_pl;
  for (size_t _p = 0; _p < prof_plugin_ms.size(); ++_p) {
    _prof_pl << " plugin" << _p << "=" << prof_plugin_ms[_p] << "ms";
  }
  std::ostringstream _prof_cams;
  for (size_t _c = 0; _c < cameras_.size(); ++_c) {
    _prof_cams << "\n    cam[" << _c << "] " << cameras_[_c].topic << " "
               << cameras_[_c].width << "x" << cameras_[_c].height
               << ": map=" << prof_cam_map_ms[_c] << "ms pub=" << prof_cam_pub_ms[_c] << "ms";
  }
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 1000,
    "[prof] cams=%zu gpu_readback(ovrtx_map)=%.1fms memcpy=%.1fms publish_total=%.1fms%s%s",
    cameras_.size(), prof_map_ms, prof_copy_ms, _prof_pub_sum, _prof_pl.str().c_str(),
    _prof_cams.str().c_str());
}

void RenderNode::publish_lidar_pointclouds(
  const ovrtx_render_product_set_outputs_t & outputs)
{
  if (lidars_.empty()) {
    return;
  }

  // Use the TF tree's time base (sim time from /tf_render) so RViz can transform the output;
  // fall back to wall clock only before the first /tf_render arrives.
  double timestamp = latest_tf_time_ > 0.0 ? latest_tf_time_ : now().seconds();
  auto wall_now = Clock::now();
  constexpr uint32_t point_step = 22;
  constexpr uint8_t reflectivity_default = 128;

  std::vector<uint8_t> merged;  // accumulated base_link-frame points across all lidars
  bool published_any = false;   // gate the (10 Hz) base_link->world TF publish to cloud cadence

  for (auto & lidar : lidars_) {
    ovrtx_render_var_output_handle_t handle =
      find_output_by_path(outputs, lidar.render_product_path, "PointCloud");
    if (handle == static_cast<ovrtx_render_var_output_handle_t>(-1)) {
      continue;
    }

    ovrtx_map_output_description_t map_desc{};
    map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
    ovrtx_render_var_output_t pc{};
    ovrtx_result_t result = ovrtx_map_render_var_output(
      renderer_, handle, &map_desc, ovrtx_timeout_infinite, &pc);
    if (result.status != OVRTX_API_SUCCESS) {
      continue;
    }

    ovrtx_cuda_sync_t no_sync{};
    if (pc.status != OVRTX_EVENT_COMPLETED) {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }

    const ovrtx_render_var_tensor_t * counts_t = find_pc_tensor(pc, "Counts");
    const ovrtx_render_var_tensor_t * coords_t = find_pc_tensor(pc, "Coordinates");
    const ovrtx_render_var_tensor_t * inten_t = find_pc_tensor(pc, "Intensity");
    const ovrtx_render_var_tensor_t * flags_t = find_pc_tensor(pc, "Flags");
    if (!counts_t || !counts_t->dl || !coords_t || !coords_t->dl) {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }

    const int32_t * counts = static_cast<const int32_t *>(counts_t->dl->data);
    const int32_t valid = counts ? counts[0] : 0;
    // partialOutputs=false => a full scan only materializes at the lidar's scan rate (~10 Hz);
    // empty/partial frames carry Counts==0, so this also throttles publish to that rate.
    if (valid <= 0) {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }
    // Safety throttle: never publish a given lidar faster than ~10 Hz.
    if (lidar.last_publish.time_since_epoch().count() != 0 &&
      std::chrono::duration<double, std::milli>(wall_now - lidar.last_publish).count() < 90.0)
    {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }
    lidar.last_publish = wall_now;

    const float * coords = static_cast<const float *>(coords_t->dl->data);
    const int64_t alloc = coords_t->dl->shape[1];  // 3 x alloc, row-major (x row, y row, z row)
    const float * inten = (inten_t && inten_t->dl) ?
      static_cast<const float *>(inten_t->dl->data) : nullptr;
    // VALID bit (1<<6). With skipDroppingInvalidPoints=false this is set on every delivered
    // entry; kept as a safety so stray no-return rays never reach the cloud.
    const uint8_t * flags = (flags_t && flags_t->dl) ?
      static_cast<const uint8_t *>(flags_t->dl->data) : nullptr;
    constexpr uint8_t kFlagValid = 0x40;

    // base_link <- sensor rotation matrix from the static quaternion.
    const double w = lidar.bl_T_s_wxyz[0], qx = lidar.bl_T_s_wxyz[1];
    const double qy = lidar.bl_T_s_wxyz[2], qz = lidar.bl_T_s_wxyz[3];
    const double xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const double xy = qx * qy, xz = qx * qz, yz = qy * qz;
    const double wx = w * qx, wy = w * qy, wz = w * qz;
    const double r00 = 1 - 2 * (yy + zz), r01 = 2 * (xy - wz), r02 = 2 * (xz + wy);
    const double r10 = 2 * (xy + wz), r11 = 1 - 2 * (xx + zz), r12 = 2 * (yz - wx);
    const double r20 = 2 * (xz - wy), r21 = 2 * (yz + wx), r22 = 1 - 2 * (xx + yy);
    const double tx = lidar.bl_T_s_xyz[0], ty = lidar.bl_T_s_xyz[1], tz = lidar.bl_T_s_xyz[2];

    PointCloudMsg msg;
    msg.frame_id = lidar.sensor_frame;  // per-lidar cloud is in the sensor frame (rmagine parity)
    msg.timestamp = timestamp;
    msg.point_step = point_step;
    msg.is_dense = true;
    msg.height = 1;
    msg.data.resize(static_cast<size_t>(valid) * point_step);
    const size_t merged_base = merged.size();
    if (!merged_lidar_topic_.empty()) {
      merged.resize(merged_base + static_cast<size_t>(valid) * point_step);
    }

    uint32_t out = 0;
    for (int32_t j = 0; j < valid; ++j) {
      if (flags && !(flags[j] & kFlagValid)) {
        continue;
      }
      // ovrtx emits points already in the original sensor (livox) frame; base_link_T_sensor
      // (the mount) maps them to base_link. Per-lidar cloud keeps the raw sensor-frame point;
      // the merged cloud accumulates the base_link-frame point.
      const double xs = coords[0 * alloc + j];
      const double ys = coords[1 * alloc + j];
      const double zs = coords[2 * alloc + j];
      const float sx = static_cast<float>(xs);
      const float sy = static_cast<float>(ys);
      const float sz = static_cast<float>(zs);
      uint8_t refl = reflectivity_default;
      if (inten) {
        const float v = inten[j] * 255.0f;
        refl = static_cast<uint8_t>(v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v));
      }
      uint8_t * row = msg.data.data() + static_cast<size_t>(out) * point_step;
      std::memcpy(row + 0, &sx, sizeof(float));
      std::memcpy(row + 4, &sy, sizeof(float));
      std::memcpy(row + 8, &sz, sizeof(float));
      row[12] = refl;
      row[13] = 0;
      std::memcpy(row + 14, &timestamp, sizeof(double));
      if (!merged_lidar_topic_.empty()) {
        const float bx = static_cast<float>(r00 * xs + r01 * ys + r02 * zs + tx);
        const float by = static_cast<float>(r10 * xs + r11 * ys + r12 * zs + ty);
        const float bz = static_cast<float>(r20 * xs + r21 * ys + r22 * zs + tz);
        uint8_t * mrow = merged.data() + merged_base + static_cast<size_t>(out) * point_step;
        std::memcpy(mrow + 0, &bx, sizeof(float));
        std::memcpy(mrow + 4, &by, sizeof(float));
        std::memcpy(mrow + 8, &bz, sizeof(float));
        mrow[12] = refl;
        mrow[13] = 0;
        std::memcpy(mrow + 14, &timestamp, sizeof(double));
      }
      ++out;
    }
    msg.data.resize(static_cast<size_t>(out) * point_step);
    msg.width = out;
    if (!merged_lidar_topic_.empty()) {
      merged.resize(merged_base + static_cast<size_t>(out) * point_step);
    }
    if (out == 0) {
      ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
      continue;
    }

    for (auto & plugin : image_plugins_) {
      plugin->publish_pointcloud(lidar.topic, msg);
    }
    published_any = true;

    ovrtx_unmap_render_var_output(renderer_, pc.map_handle, no_sync);
  }

  if (!merged_lidar_topic_.empty() && !merged.empty()) {
    PointCloudMsg merged_msg;
    merged_msg.frame_id = "base_link";
    merged_msg.timestamp = timestamp;
    merged_msg.point_step = point_step;
    merged_msg.is_dense = true;
    merged_msg.height = 1;
    merged_msg.width = static_cast<uint32_t>(merged.size() / point_step);
    merged_msg.data = std::move(merged);
    for (auto & plugin : image_plugins_) {
      plugin->publish_pointcloud(merged_lidar_topic_, merged_msg);
    }
    published_any = true;
  }

  // Emit base_link->world together with the cloud (same `timestamp`, ~10 Hz) so every
  // published cloud has an exactly-matching transform in the consumer's TF buffer.
  if (published_any) {
    publish_lidar_tf();
  }
}

void RenderNode::stats_timer_callback()
{
  if (timing_history_.empty()) {
    return;
  }

  auto extract = [&](auto field) {
      std::vector<double> v;
      v.reserve(timing_history_.size());
      for (const auto & t : timing_history_) {
        v.push_back(t.*field);
      }
      return v;
    };

  auto stats = [](const std::vector<double> & v)
    -> std::tuple<double, double, double, double> {
      double sum = std::accumulate(v.begin(), v.end(), 0.0);
      double mean = sum / static_cast<double>(v.size());
      double mn = *std::min_element(v.begin(), v.end());
      double mx = *std::max_element(v.begin(), v.end());
      double sq_sum = 0.0;
      for (auto x : v) {
        sq_sum += (x - mean) * (x - mean);
      }
      double stddev = std::sqrt(sq_sum / static_cast<double>(v.size()));
      return {mean, mn, mx, stddev};
    };

  auto [total_mean, total_min, total_max, total_std] =
    stats(extract(&FrameTiming::total_ms));
  auto [step_mean, step_min, step_max, step_std] =
    stats(extract(&FrameTiming::step_ms));
  auto [wait_mean, wait_min, wait_max, wait_std] =
    stats(extract(&FrameTiming::wait_ms));
  auto [interval_mean, interval_min, interval_max, interval_std] =
    stats(extract(&FrameTiming::interval_ms));

  double target_interval_ms = 1000.0 / render_fps_;
  double jitter_ms = interval_std;
  double actual_fps = (interval_mean > 0.0) ? 1000.0 / interval_mean : 0.0;

  RCLCPP_INFO(
    get_logger(),
    "--- Render Stats (last %zu frames, %zu cameras) ---\n"
    "  Frames: %lu  Drops: %lu  Actual FPS: %.1f (target %.1f)\n"
    "  Total:    mean=%.2f  min=%.2f  max=%.2f  std=%.2f ms\n"
    "  Step:     mean=%.2f  min=%.2f  max=%.2f  std=%.2f ms\n"
    "  Wait:     mean=%.2f  min=%.2f  max=%.2f  std=%.2f ms\n"
    "  Interval: mean=%.2f  min=%.2f  max=%.2f  jitter=%.2f ms (target=%.2f ms)",
    timing_history_.size(), cameras_.size(),
    frame_count_, drop_count_, actual_fps, render_fps_,
    total_mean, total_min, total_max, total_std,
    step_mean, step_min, step_max, step_std,
    wait_mean, wait_min, wait_max, wait_std,
    interval_mean, interval_min, interval_max, jitter_ms, target_interval_ms);
}

void RenderNode::publish_timing(const FrameTiming & t)
{
  auto msg = std::make_unique<std_msgs::msg::Float64MultiArray>();
  msg->layout.dim.resize(1);
  msg->layout.dim[0].label = "timing";
  msg->layout.dim[0].size = 7;
  msg->layout.dim[0].stride = 7;
  msg->data = {
    t.step_ms,
    t.wait_ms,
    t.fetch_ms,
    t.map_ms,
    t.publish_ms,
    t.total_ms,
    t.interval_ms,
  };
  timing_pub_->publish(std::move(msg));
}

ovrtx_render_var_output_handle_t RenderNode::find_output_for_product(
  const ovrtx_render_product_set_outputs_t & outputs,
  size_t product_index,
  const char * render_var_name)
{
  const auto & cam = cameras_[product_index];
  const std::string & target_path = cam.render_product_path;

  static bool logged_once = false;
  if (!logged_once) {
    logged_once = true;
    for (size_t p = 0; p < outputs.output_count; ++p) {
      const auto & product = outputs.outputs[p];
      std::string out_path(product.render_product_path.ptr,
        product.render_product_path.length);
      RCLCPP_INFO(
        get_logger(), "Output[%zu] product=%s frames=%zu",
        p, out_path.c_str(), product.output_frame_count);
      for (size_t f = 0; f < product.output_frame_count; ++f) {
        const auto & frame = product.output_frames[f];
        for (size_t v = 0; v < frame.render_var_count; ++v) {
          const auto & var = frame.output_render_vars[v];
          std::string vname(var.render_var_name.ptr ? var.render_var_name.ptr : "(null)",
            var.render_var_name.length);
          RCLCPP_INFO(get_logger(), "  var[%zu]: name='%s'", v, vname.c_str());
        }
      }
    }
  }

  for (size_t p = 0; p < outputs.output_count; ++p) {
    const auto & product = outputs.outputs[p];
    std::string out_path(product.render_product_path.ptr,
      product.render_product_path.length);
    if (out_path != target_path) {
      continue;
    }
    for (size_t f = 0; f < product.output_frame_count; ++f) {
      const auto & frame = product.output_frames[f];
      for (size_t v = 0; v < frame.render_var_count; ++v) {
        const auto & var = frame.output_render_vars[v];
        if (var.render_var_name.ptr &&
          strncmp(
            var.render_var_name.ptr, render_var_name,
            var.render_var_name.length) == 0)
        {
          return var.output_handle;
        }
      }
    }
    break;
  }
  return static_cast<ovrtx_render_var_output_handle_t>(-1);
}

ovrtx_render_var_output_handle_t RenderNode::find_output_by_path(
  const ovrtx_render_product_set_outputs_t & outputs,
  const std::string & product_path,
  const char * render_var_name)
{
  const size_t var_len = std::strlen(render_var_name);
  for (size_t p = 0; p < outputs.output_count; ++p) {
    const auto & product = outputs.outputs[p];
    std::string out_path(product.render_product_path.ptr, product.render_product_path.length);
    if (out_path != product_path) {
      continue;
    }
    for (size_t f = 0; f < product.output_frame_count; ++f) {
      const auto & frame = product.output_frames[f];
      for (size_t v = 0; v < frame.render_var_count; ++v) {
        const auto & var = frame.output_render_vars[v];
        if (var.render_var_name.ptr && var.render_var_name.length == var_len &&
          strncmp(var.render_var_name.ptr, render_var_name, var_len) == 0)
        {
          return var.output_handle;
        }
      }
    }
    break;
  }
  return static_cast<ovrtx_render_var_output_handle_t>(-1);
}

}  // namespace genie_sim_render
