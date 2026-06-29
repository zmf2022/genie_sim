# genie_sim_render

Isaac Sim render node — publishes camera images from the USD stage over ROS 2.

Source: [source/geniesim_ros/src/ros_ws/src/genie_sim_render/](.)
License: [Mozilla Public License Version 2.0](../../../../LICENSE)

**Maintenance contract**: when you add a plugin, change the manifest schema
consumed here, or alter topic conventions, update this file in the same diff.

---

## Layout

```
genie_sim_render/
├── scripts/
│   └── isaacsim_render.py         ← Python Isaac Sim render node (render_isaacsim)
├── src/
│   ├── render_node.cpp            ← C++ OVRtx render node (render_ovrtx)
│   ├── main.cpp                   ← plugin registration
│   └── plugins/
│       └── ros_image_publisher_plugin.cpp ← RosImagePublisherPlugin
├── include/                       ← C++ headers
├── launch/
│   └── render.launch.py           ← standalone render launch
├── config/
│   └── cam_x4.perspective         ← default RViz2 camera perspective
└── plugins.xml                    ← pluginlib plugin description
```

---

## Two render backends

### `render_ovrtx` — C++ OVRtx node (`render_node.cpp`)

The production render backend. Runs as a separate process from the physics
node. Reads `manifest.json` (same schema as the physics node), opens the USD
stage, and publishes images via `RosImagePublisherPlugin`.

**First-run warmup.** OVRTX compiles shaders + JITs kernels on the first
`ovrtx_wait_op`, which can block for 30–90 s with no log output and blank
images. `render_timer_callback` prints a banner + a 1 Hz heartbeat
("first frame: still compiling shaders / warming GPU (Ns elapsed)") until
the wait returns, then a completion banner. Subsequent runs hit the
shader cache and start instantly. The same heartbeat pattern lives in
the inline OVRTX visualizer in `genie_sim_engine` so operators see the
same UX regardless of which render path is active.

Activated by listing `render_ovrtx` in the launcher YAML `renders:` block.
Constructed by `utils.py:make_render_ovrtx_node()` in the bringup package.

Key subscriptions / publications:
- Subscribes `/tf_render` — applies body transforms to robot prims each frame.
  `child_frame_id` is the **absolute USD prim path** (e.g.
  `/ur5/Geometry/base_link/arm_base_link/arm_link1`), not a basename, and
  the transform is the body's **local pose relative to its immediate USD
  parent**, not world. The renderer writes the received pose directly into
  the target prim's `xformOp:translate` / `xformOp:orient`, and USD's
  standard hierarchical composition reconstructs the world pose by walking
  the parent chain — every ancestor link is itself receiving a local
  transform on the same tick, so the full chain matches the simulator's
  state. This protocol is layout-agnostic across the three URDF→USD
  pipelines: 4.x/5.x flat (`/<prefix>/link`, parent at identity ⇒ local =
  world), `isaac_physx` AS3 (`/<prefix>/Geometry/.../link`, parent has
  non-trivial transform), and `isaac_newton` AS3 (same nesting). The
  publisher computes local via `_xform_to_xyzwxyz_local` in
  `runtime.stage.py` —
  `local = world * inverse(parent_world)` in USD row-vector convention.
  The path-resolution branch in `on_tf_render`/`_on_tf_render`
  (`if path.startswith("/"): use_directly`) handles every payload uniformly;
  the legacy `prefix + basename` fallback stays as defensive code but is
  unreachable for any payload published by the current
  `genie_sim_engine`.
- Subscribes `~/free_cam_pose` — moves the free-fly camera
- Publishes `<topic>/image_raw` + `<topic>/camera_info` per manifest camera

### `render_isaacsim` — Python Isaac Sim node (`isaacsim_render.py`)

Python replacement for the C++ OVRtx node. Uses the same `manifest.json`
schema and the same topic conventions. Activated by listing `render_isaacsim`
in the launcher YAML `renders:` block.

Constructed by `utils.py:make_render_isaacsim_node()` in the bringup package.

---

## Manifest schema consumed

Both backends read the same `manifest.json` fields:

| Field | Purpose |
|---|---|
| `scene_usda` / `usd_path` | World USD to open |
| `robot_usda` | Robot USD to reference in |
| `render_layer_usda` | Render-product layer (cameras + RenderProducts) |
| `robot_prefix` | Prim namespace for the robot (`/<robot_prefix>`) |
| `free_cam_prim_path` | Path to the free-fly camera prim |
| `base_path` | Anchor for resolving relative paths |
| `cameras[]` | Per-camera: `topic`, `depth_topic`, `render_product_path`, `path`, `frame_id`, `width`, `height`, intrinsics |

---

## Plugin: `RosImagePublisherPlugin`

C++ pluginlib plugin (`genie_sim_render/RosImagePublisherPlugin`). Registered
in `plugins.xml`. Activated via the launcher YAML params block:

```yaml
render_ovrtx:
  ros__parameters:
    plugin:
      - genie_sim_render/RosImagePublisherPlugin
```

---

## Topic conventions

- RGB: `<camera_topic>/image_raw` (`sensor_msgs/Image`)
- Depth: `<camera_topic>/depth/image_raw` (`sensor_msgs/Image`, 32FC1 metres)
- Camera info: `<camera_topic>/camera_info` (`sensor_msgs/CameraInfo`)
- Free cam pose: `<node_ns>/free_cam_pose` (`geometry_msgs/PoseStamped`)
- TF render: `/tf_render` (`tf2_msgs/TFMessage`) — `child_frame_id` is an
  absolute USD prim path; transform is **local relative to the immediate
  USD parent**, not world. Internal channel between `genie_sim_engine` and
  the render backends; not a substitute for `/tf` (which `robot_state_publisher`
  publishes with URDF-style basenames + world frames for general ROS
  consumers).

---

## Routing rules

- OVRtx C++ render node → `src/render_node.cpp`
- Image publisher plugin → `src/plugins/ros_image_publisher_plugin.cpp`
- Python Isaac Sim render node → `scripts/isaacsim_render.py`
- Standalone launch → `launch/render.launch.py`
- Node construction (from bringup) → `genie_sim_bringup/launch/utils.py:make_render_ovrtx_node` / `make_render_isaacsim_node`
