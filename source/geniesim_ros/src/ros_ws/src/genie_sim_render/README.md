# genie_sim_render

GPU rendering for the GenieSim simulator. Produces photoreal camera
images via OVRtx and publishes them on ROS 2 image topics.

## Two render modes

| Mode | Where it runs | When you'd pick it |
|---|---|---|
| **Inline OVRtx** | Inside the Newton-standalone physics process (`InlineOvrtxVisualizer` in `genie_sim_engine`) | Newton-standalone runs (`launcher_newton_*`). Zero-copy `body_q → omni:xform` via Warp; no IPC. |
| **Standalone OVRtx node** | Separate ROS process (`render_ovrtx`, this package) | Isaac Sim runs (`launcher_ovrtx_isaac_*`). Physics and rendering get independent CPU/GPU budgets. |

Both modes consume the **same `manifest.json`** schema (cameras,
RenderProducts, robot prefix, free-cam prim path), so swapping
launcher_configs swaps render mode without scene-yaml edits.

A Python alternative — `render_isaacsim` — is also packaged for
Isaac-Sim-native rendering. Both `render_ovrtx` and `render_isaacsim`
publish the same topic shape.

## Topics

- `<camera_topic>/image_raw` — RGB (`sensor_msgs/Image`)
- `<camera_topic>/depth/image_raw` — depth in metres (32FC1)
- `<camera_topic>/camera_info` — intrinsics (`sensor_msgs/CameraInfo`)
- `<node_ns>/free_cam_pose` — interactive free-camera pose
  (`geometry_msgs/PoseStamped`)
- `/tf_render` — per-body local transforms from the engine

## First-run warmup

OVRtx compiles shaders on first run; this can take 30–90 s with no
output. Both render paths print a 1 Hz heartbeat
("first frame: still compiling shaders / warming GPU (Ns elapsed)")
so operators know warmup vs hang. Subsequent runs hit the cache and
start instantly.

## When you'd touch this package

- Adding a new render plugin (`pluginlib` — see
  `RosImagePublisherPlugin`).
- Changing topic conventions or `manifest.json` schema.
- Modifying the `/tf_render` protocol or free-cam pose handling.
- Working on the C++ OVRtx node (`src/render_node.cpp`) or the Python
  Isaac Sim node (`scripts/isaacsim_render.py`).

## Mechanics

See [AGENTS.md](AGENTS.md) for the manifest schema, plugin
registration, `/tf_render` protocol details, and routing rules.
