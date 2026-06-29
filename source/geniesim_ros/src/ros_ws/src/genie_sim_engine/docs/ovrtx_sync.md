# OVRtx as Newton's in-process visualizer — zero-copy GPU sync, parallel render thread

> **Status**: implemented. See `engine/newton/visualizers/ovrtx.py` (`InlineOvrtxVisualizer`).
> **Target OVRtx**: **0.3.0** (`OVRTX_VERSION_MAJOR=0`, `MINOR=3`, confirmed at `<OVRtx repo>/include/ovrtx/ovrtx_types.h:13-15`).
> **Sibling docs**: [engines.md](engines.md), [pipeline.md](pipeline.md), [perf.md](perf.md).
>
> **Design vs. implementation.** This document is the original design /
> archaeology note — pseudocode signatures, `__init__` parameter lists,
> and the "Sketch" code blocks reflect the *intent*, not the shipped
> shape. For ground-truth API (constructor signature, `bind_newton_bodies`
> parameters, `renderer.step` call pattern, `_publish_outputs` vs
> `_publish_camera_outputs`, `PrimMode.EXISTING_ONLY` vs the design-time
> `MUST_EXIST`, the `RuntimeError`-based version guard vs the design's
> `assert`), read [`engine/newton/visualizers/ovrtx.py`](../scripts/engine/newton/visualizers/ovrtx.py)
> and [`engine/newton/visualizers/ovrtx_kernels.py`](../scripts/engine/newton/visualizers/ovrtx_kernels.py).
> The architectural claims here (zero-copy hot path, threading contract,
> CUDA stream handshake, attribute layout / semantic) are still
> authoritative — those have not changed.

---

## Scope

This doc covers an **inline OVRtx visualizer** that lives **inside `genie_sim_engine`** as the `ovrtx` arm of `physics_engine_visualizer`. It is a separate path from — and does not replace or modify — the existing cross-process renderer in `genie_sim_render`:

- The **`genie_sim_render` package is out of scope** for this work. Its render node (`genie_sim_render/src/render_node.cpp`) and its `RosImagePublisherPlugin` continue to run as an independent ROS 2 process driven by `/tf_render`. Nothing in `genie_sim_render/` is touched by this design.
- The **only allowed interaction with `genie_sim_render`** is **borrowing code patterns** (camera-config parsing, OVRtx renderer init sequence, output extraction, `sensor_msgs::Image` packing, subscriber-count gating). New code lives entirely under `genie_sim_engine/`.
- The two paths are **mutually exclusive at runtime**: a launcher either spawns the cross-process `render_ovrtx` node (Lane B today) **or** sets `physics_engine_visualizer:=ovrtx` for inline OVRtx. The newton-standalone launchers (`launcher_newton_mjwarp.yaml`, `launcher_newton_fsvbd.yaml`) already keep `launcher.renders: []`, so they're the natural home for the inline path.
- The cross-process renderer keeps its `/tf_render`-driven workflow for users who want render and physics in different processes (e.g. for GPU isolation, separate failure domains, or to drive multiple physics sources from one renderer). This design intentionally does not deprecate it.

> **Anchor file boundary**: every "Modify" target in this doc lives under `source/geniesim_ros/src/ros_ws/src/genie_sim_engine/`. References to `genie_sim_render/...:line` are read-only — they exist so an implementer can lift the relevant pattern, not patch the file.

---

## Context

In `genie_sim_engine`'s newton-standalone path, body transforms today never leave the physics process — there is no separate-process renderer wired up (`launcher_newton_mjwarp.yaml` has `launcher.renders: []`). The visualizer surface in this entry point is currently `none` (headless), `newton` (Newton GL OpenGL window), or two TODO placeholders (`rerun`, `ovrtx`). This design fills in the `ovrtx` placeholder.

For reference, the `genie_sim_render` cross-process node has a different design problem entirely: body transforms travel from Newton GPU → physics CPU → `tf2_msgs::TFMessage` serialise → ROS middleware → deserialise → `ovrtx_set_xform_mat` (an inline wrapper around `ovrtx_write_attribute` with `OVRTX_DATA_ACCESS_SYNC` doing a CPU→GPU copy on every call — `ovrtx_attributes.h:191-217`). That cost is fundamental to a cross-process design and is not what this doc fixes.

What this doc does fix, for the in-process path:

1. **Zero CPU-side copy** of body transforms — Warp kernel writes directly into OVRtx's internal Fabric buffer.
2. **No physics-thread blocking** — OVRtx runs on its own Python thread + CUDA stream.
3. **Single-process delivery** of camera images — published via `gsi::RosBridge` from the OVRtx thread, not bridged across processes.

The design has three properties:

- **Zero-copy GPU sync**: Warp kernel writes directly into OVRtx's internal Fabric buffer via `binding.map(device=Device.CUDA)` — the pattern the upstream `mapping-attributes` skill (`<OVRtx repo>/skills/mapping-attributes/SKILL.md`) names as the recommended path for "Warp/CUDA kernel writes into mapped tensors".
- **Dedicated OVRtx render thread**: a Python `threading.Thread` owns the renderer, the per-frame sync kernel, the OVRtx step, and the ROS image publishes. Physics thread does no OVRtx work other than recording one CUDA event per step.
- **Cross-stream CUDA handshake**: physics records `physics_step_event` on Newton's stream; the OVRtx thread's Warp stream waits on it before each frame's sync kernel. Physics never CPU-blocks; physics and OVRtx GPU work overlap.

---

## OVRtx 0.3.0 skill references that anchor this design

The OVRtx repository ships authoritative skill docs under `<OVRtx repo>/skills/`. These are the API source-of-truth — every pattern below is lifted from a skill snippet, not invented.

| Skill | Why it applies |
|---|---|
| `<OVRtx repo>/skills/mapping-attributes/SKILL.md` | **The recommended path for this design.** "Zero-copy attribute map/unmap for direct memory access to ovrtx internal buffers. Use when user asks about zero-copy writes, map attribute, direct memory access, Warp/CUDA kernel writes into mapped tensors, or GPU attribute updates." Recommends `binding.map(device=Device.CUDA)` over `write_attribute` for repeated GPU writes. |
| `<OVRtx repo>/skills/cuda-interop/SKILL.md` | CUDA stream/event handshake patterns. Critical detail at line 155: when a CUDA Warp tensor is passed with `cuda_stream=`, OVRtx forwards the stream into the producer's DLPack sync — no manual `wp.synchronize_stream` needed. |
| `<OVRtx repo>/skills/writing-transforms/SKILL.md` | Confirms `"omni:xform"` is the canonical attribute name, `Semantic.XFORM_MAT4x4` is the semantic, **USD row-vector convention** (translation in last row `[3][0..2]`), `float64` only. |
| `<OVRtx repo>/skills/attribute-bindings/SKILL.md` | The non-zero-copy alternative — "repeated writes with caller-owned tensors when a copy is acceptable". Not chosen for the hot path because it requires us to allocate and own an intermediate `body_xforms` buffer. |
| `<OVRtx repo>/skills/reading-render-output/SKILL.md` and `<OVRtx repo>/skills/stepping-and-rendering/SKILL.md` | Patterns for `renderer.step(...)` and `render_var.map(device=Device.CPU)` to pull frame data for the ROS publish step. |
| `<OVRtx repo>/skills/update-0_2-0_3/SKILL.md` | Reference if any older `ovrtx` examples need translating. |

**Canonical zero-copy snippet** (from `<OVRtx repo>/tests/docs/python/test_attribute_bindings.py:171-189`, snippet `doc-map-attribute-cuda`):

```python
mapping = renderer.map_attribute(
    [prim_paths], "omni:xform",
    dtype="float64", shape=(4, 4),
    device=ovrtx.Device.CUDA,
)
tensor = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
stream = wp.Stream(device=tensor.device)
wp.launch(kernel, dim=N, inputs=[tensor, ...], stream=stream)
mapping.unmap(stream=stream.cuda_stream)
```

This is the hot path. **Zero copies**: OVRtx's internal Fabric buffer is exposed as a DLPack tensor, Warp wraps it without copying, the kernel writes directly into OVRtx's buffer, and `unmap(stream=...)` signals OVRtx via stream order — no intermediate `body_xforms` buffer.

---

## Goal

OVRtx is wired into the existing newton-standalone visualizer dispatch in `genie_sim_engine_newton.py` (alongside `newton` GL and the `rerun` TODO). It is selected at launch by:

```bash
ros2 launch genie_sim_bringup app.launch.py \
    launcher_config:=launcher_newton_mjwarp \
    headless:=false \
    physics_engine_visualizer:=ovrtx
```

**Selection rules** (unchanged from the existing scaffolding — we just fill in the `ovrtx` branch):

- `headless:=true` (the launcher default — argv lacks `--gui`): no visualizer runs, regardless of `physics_engine_visualizer`. The module-level guard at `genie_sim_engine_newton.py:64-70` downgrades to `none` and the run is pure headless.
- `headless:=false`: `physics_engine_visualizer` selects exactly one visualizer (`none` | `newton` | `ovrtx` | `rerun`). They are mutually exclusive.
- OVRtx is itself kitless and has no GUI window today, but it is **treated as a non-headless visualizer** because a GUI may be added later — and even without a GUI, treating it as "requires `headless:=false`" keeps the existing single rule simple.
- **No Kit involvement anywhere**: newton-standalone is Kit-free — it never imports `omni.*` and never constructs a `SimulationApp`.

**What the `ovrtx` branch does, once selected:**

1. OVRtx 0.3.0 renderer runs on a dedicated Python thread in the physics process.
2. Physics thread does no OVRtx work beyond `wp.record_event` after each Newton step.
3. OVRtx thread per frame: wait on physics event → `binding.map(device=CUDA)` → `wp.launch(sync_kernel)` (writes directly into mapped buffer) → `mapping.unmap(stream=...)` → `renderer.step(...)` → fetch outputs → publish `sensor_msgs::Image` from this thread via `gsi::RosBridge`.
4. Latest-state-wins: if OVRtx falls behind, intermediate physics ticks are skipped silently. Physics never waits.
5. `/tf_render` continues to publish on the physics thread regardless of visualizer mode — it's a free debug/record channel (RViz TF view, `ros2 bag record /tf_render`) with no inline-OVRtx-specific behaviour. Inline OVRtx does **not** consume the topic; it reads `body_q` directly from Newton state via the CUDA event handshake.
6. With `physics_engine_visualizer:=none` (or `headless:=true`), behaviour is bit-for-bit identical to today.

---

## Architecture

```
┌────────────────────────────── physics process ─────────────────────────────────┐
│                                                                                │
│  ┌────────────────────── Physics thread (main) ──────────────────────┐         │
│  │  EngineRunLoop.spin()                                             │         │
│  │   ├─ Newton step (wp.capture_launch) ─► writes state.body_q (GPU) │         │
│  │   ├─ wp.record_event(physics_step_event)  ◄── only NEW per-tick   │         │
│  │   │                                          OVRtx-related call   │         │
│  │   ├─ tick_extras (Fabric write — only if Kit viewport active)     │         │
│  │   ├─ publish_tick (clock, joints, odom, /tf_render — all unchanged)│         │
│  │   └─ sleep to next tick                                           │         │
│  └───────────────────────────────────────────────────────────────────┘         │
│                            │ (CUDA event handshake — non-blocking on CPU)      │
│                            ▼                                                   │
│  ┌──────────────────── OVRtx thread (threading.Thread) ─────────────┐          │
│  │  while not shutdown:                                             │          │
│  │   ├─ wait next render tick (timer at target FPS)                 │          │
│  │   ├─ ovrtx_stream.wait_event(physics_step_event)                 │          │
│  │   ├─ with binding.map(device=Device.CUDA) as mapping:            │          │
│  │   │     ovrtx_xforms = wp.from_dlpack(mapping.tensor,            │          │
│  │   │                                    dtype=wp.mat44d)          │          │
│  │   │     wp.launch(sync_body_q_to_ovrtx_mat44d,                   │          │
│  │   │               dim=N,                                         │          │
│  │   │               inputs=[ovrtx_xforms, body_indices, body_q],   │          │
│  │   │               stream=ovrtx_stream)                           │          │
│  │   │     # exit calls mapping.unmap(stream=ovrtx_stream.cuda_stream)│        │
│  │   ├─ step_h = renderer.step(period)                              │          │
│  │   ├─ outputs = renderer.fetch_results(step_h)                    │          │
│  │   └─ for cam, var in outputs:                                    │          │
│  │         with var.map(device=Device.CPU) as img:                  │          │
│  │             _core.publish_image(sim_time, cam, img.dl)           │          │
│  │     renderer.destroy_results(step_h)                             │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Synchronization primitives

| Primitive | Producer | Consumer | Purpose |
|---|---|---|---|
| `physics_step_event` (`wp.Event`) | Physics thread after each `step()` | OVRtx thread before each sync kernel | GPU-side cross-stream handshake — OVRtx's sync kernel sees a committed `body_q`. **Non-blocking on CPU.** |
| `binding.map(...).tensor` (DLPack view into OVRtx Fabric buffer) | OVRtx-internal | Our Warp kernel | Zero-copy write target. Lifetime is the `with` block. |
| `ovrtx_stream` (`wp.Stream`) | OVRtx thread | — | Dedicated CUDA stream so OVRtx work doesn't serialise behind physics work on Warp's default stream |
| `shutdown_event` (`threading.Event`) | Main (atexit) | OVRtx thread | Clean teardown |
| `last_sim_time` (`std::atomic<double>` in `RosBridge`) | Physics thread (set on `publish_clock`) | OVRtx thread (read on publish_image) | Timestamp images without a mutex |

**No CPU mutex on `body_q`.** Physics is the only writer; OVRtx waits on `physics_step_event` so its kernel observes a fully-committed `body_q`.

**No backpressure on physics.** OVRtx renders the latest `body_q` whenever it gets to it.

**Stream parallelism.** Physics kernels on Newton's default stream; OVRtx sync kernel + render on `ovrtx_stream`. CUDA scheduler overlaps them whenever device capacity allows.

---

## Implementation Plan

### 1. New Python module: `InlineOvrtxVisualizer`

**New file**: `scripts/engine/newton/visualizers/ovrtx.py`

Owns the renderer, the persistent binding, the dedicated render thread, and the per-frame zero-copy map/launch/unmap cycle.

```python
import threading, time, warp as wp, ovrtx
from .ovrtx_kernels import sync_body_q_to_ovrtx_mat44d

class InlineOvrtxVisualizer:
    def __init__(self, ros_bridge, scene_usda, robot_usda, render_layer_usda,
                 cameras_cfg, device="cuda:0", ovrtx_root=None, render_fps=30.0):
        # version guard — header/wheel skew is the most common setup bug
        major, minor, _ = ovrtx.get_version()
        assert (major, minor) == (0, 3), f"requires OVRtx 0.3.x, got {major}.{minor}"

        # renderer init mirrors render_node.cpp:281-326 in Python
        self._renderer = ovrtx.Renderer([...])  # config entries
        self._renderer.open_usd(scene_usda)
        self._renderer.add_usd_reference(robot_usda, prefix=...)
        self._render_products = self._build_render_products(cameras_cfg)

        self._binding = None            # set in bind_newton_bodies
        self._body_indices = None       # wp.array(int32) — newton body index map
        self._physics_event = None      # wp.Event from physics thread
        self._ovrtx_stream = wp.Stream(device=device)
        self._device = device
        self._render_period = 1.0 / max(render_fps, 1e-3)
        self._ros = ros_bridge
        self._thread = None
        self._shutdown = threading.Event()
        self._ready = threading.Event()

    def bind_newton_bodies(self, model):
        """Build the persistent attribute binding and the Newton-index map.
        Called from physics thread after Newton's start_simulation()."""
        body_paths = list(model.body_label)
        # Persistent binding pays the prim-resolution cost once and is used
        # via binding.map(...) every frame — see mapping-attributes skill.
        self._binding = self._renderer.bind_attribute(
            prim_paths=body_paths,
            attribute_name="omni:xform",
            dtype="float64",
            shape=(4, 4),
            prim_mode=ovrtx.PrimMode.EXISTING_ONLY,
            flags=ovrtx.BindingFlag.OPTIMIZE,
        )
        self._body_indices = wp.array(list(range(len(body_paths))),
                                      dtype=wp.int32, device=self._device)
        self._ready.set()

    def attach_physics_event(self, event):
        self._physics_event = event

    def start(self):
        assert self._ready.is_set()
        self._thread = threading.Thread(target=self._run, name="ovrtx-render",
                                        daemon=False)
        self._thread.start()

    def stop(self):
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._binding is not None:
            self._binding.unbind()
        self._renderer = None

    def _run(self):
        next_tick = time.monotonic()
        while not self._shutdown.is_set():
            now = time.monotonic()
            if now < next_tick:
                self._shutdown.wait(timeout=(next_tick - now))
                continue
            next_tick = now + self._render_period
            try:
                self._render_one_frame()
            except Exception:
                logger.exception("[ovrtx-thread] frame failed")

    def _render_one_frame(self):
        # 0. Subscriber-aware skip — port of render_node.cpp:313 pattern,
        # rewritten in-engine (no edits to genie_sim_render). If nobody
        # is subscribed to ANY camera output, skip the whole frame:
        # ovrtx_step + map + Warp kernels + publishes are all elided.
        if not self._any_subscribers():
            return

        # 1. Cross-stream sync: wait for the latest physics commit
        if self._physics_event is not None:
            self._ovrtx_stream.wait_event(self._physics_event)

        # 2. ZERO-COPY MAP — the heart of the design.
        # binding.map gives us a DLPack view into OVRtx's internal Fabric
        # buffer for "omni:xform". The Warp kernel writes directly into it.
        body_q = self._read_body_q()  # live wp.array, no copy
        with self._binding.map(device=ovrtx.Device.CUDA) as mapping:
            ovrtx_xforms = wp.from_dlpack(mapping.tensor, dtype=wp.mat44d)
            wp.launch(
                sync_body_q_to_ovrtx_mat44d,
                dim=ovrtx_xforms.shape[0],
                inputs=[ovrtx_xforms, self._body_indices, body_q],
                stream=self._ovrtx_stream,
            )
            # __exit__ does mapping.unmap(stream=self._ovrtx_stream.cuda_stream)
            # so OVRtx sees the writes are stream-ordered against our work.

        # 3. Render + publish (sensor publishes detailed in §1a below)
        step_h = self._renderer.step(self._render_period, self._render_products)
        outputs = self._renderer.fetch_results(step_h)
        sim_time = self._ros.last_sim_time()
        self._publish_camera_outputs(sim_time, outputs)
        self._renderer.destroy_results(step_h)

    def _any_subscribers(self) -> bool:
        """True if any camera RGB or depth topic has a live subscriber."""
        for cam in self._cameras:
            if self._ros.has_image_subscribers(cam.topic):
                return True
            if cam.depth_topic and self._ros.has_image_subscribers(cam.depth_topic):
                return True
        return False

    def _read_body_q(self):
        from isaaclab_newton.physics import NewtonManager  # or local equivalent
        return NewtonManager.get_state_0().body_q
```

> **Important on `with binding.map(...)` lifetime**: the `mapping-attributes` skill (line 135) is explicit — "the tensor from `mapping.tensor` is only valid while the mapping is active... Accessing it after the `with` block exits is undefined behavior." Our kernel launch is *inside* the `with` block; only the kernel reference into the buffer escapes via Warp's stream-ordered queue, which is well-defined.

> **Important on `unmap(stream=...)`**: also from `mapping-attributes` line 138-139 — "Data must be fully written before calling `unmap()`. For CUDA, pass `stream` or `event` so ovrtx knows when the GPU write is complete." Python's context-manager `__exit__` for `binding.map` calls `unmap` with the stream we passed in (verify the exact ctor signature — it may need `binding.map(device=Device.CUDA, stream=self._ovrtx_stream.cuda_stream)` to propagate the stream into __exit__'s unmap).

### 1a. Sensor publishing on the OVRtx thread

The OVRtx thread owns publishing of all render-derived sensor outputs. **No edits to `genie_sim_render`** — the patterns below are lifted from `genie_sim_render/src/render_node.cpp` and re-expressed in-engine.

**Sensors in scope** (newton-standalone today): camera RGB and camera depth, each with its own `sensor_msgs::Image` and `sensor_msgs::CameraInfo`. Lidar is reserved in scene YAML (`lidars: []`) but not yet implemented; out of scope here.

**Sensors NOT in scope of this thread** (already published by physics thread via the existing `RosBridge`, unaffected by inline OVRtx):

- `publish_clock` (`/clock`)
- `publish_joint_states` (`/joint_states`)
- `publish_odom` (`/odom`)

Those continue at physics cadence on the physics thread — see `realtime_ros_node.cpp:153-238`.

**New `RosBridge` methods** (added under `genie_sim_engine/src/realtime_ros_node.{hpp,cpp}` and pybinding shim under `src/pybinding.cpp`):

| Method | Mirrors `genie_sim_render` reference | Purpose |
|---|---|---|
| `publish_camera_image(sim_time, topic, dl_tensor)` | `RosImagePublisherPlugin::publish_camera_image` (`plugins/ros_image_publisher_plugin.cpp:30`) | One RGB or depth frame as `sensor_msgs::Image`. Topic name selects the image type (RGB → `sensor_msgs::image_encodings::RGBA8`, depth → `TYPE_32FC1`). |
| `publish_camera_info(sim_time, topic, intrinsics)` | `RosImagePublisherPlugin::publish_camera_info` (`plugins/ros_image_publisher_plugin.cpp:50`) | Per-frame `sensor_msgs::CameraInfo`. Intrinsics taken from scene YAML; RosBridge caches per topic. |
| `has_image_subscribers(topic) -> bool` | inline use of `Publisher::get_subscription_count()` at `render_node.cpp:313` | Subscriber-count gate for the §1 skip path. Returns `false` when no consumer is subscribed. |
| `last_sim_time() -> double` (atomic read) | n/a | Allows the OVRtx thread to stamp images with the most recent physics-thread `publish_clock` time. Set by physics thread (single writer), read by any thread (no mutex; `std::atomic<double>` is lock-free on x86_64). |

The pybind shim follows the existing pattern at `src/pybinding.cpp:94-100, 117-118, 133-134, 159-160` — release the GIL around the C++ publish call. `rclcpp::Publisher::publish` is documented thread-safe; no extra mutex.

**Camera config source.** The OVRtx thread reads the same per-camera fields the cross-process renderer does — borrowed from `render_node.cpp:233-235` (depth_topic, dds_depth_topic) — but the source of truth in this path is `manifest.json` (or the equivalent Python-loadable cameras_cfg from `EngineSession`), not a re-parse of scene YAML. `assemble_scene.py` already writes camera entries into the manifest. The OVRtx visualizer takes a `cameras_cfg` list at construction:

```python
@dataclass
class CameraCfg:
    prim_path: str            # for OVRtx render-product binding
    topic: str                # RGB topic (e.g., "/camera/wrist/color")
    depth_topic: str | None   # set when scene YAML topic.depth is non-empty
    dds_depth_topic: str | None
    width: int
    height: int
    intrinsics: CameraIntrinsics
```

**Render product setup** mirrors `render_node.cpp:281-326` (open_usd, add_usd_reference, render-product creation per camera). Implementer ports each call into Python via the OVRtx 0.3.0 Python bindings — see `<OVRtx repo>/examples/python/minimal/main.py` and the `stepping-and-rendering` skill for the canonical Python idioms. Do not reach into the C++ render node's compiled binary.

**Output extraction** mirrors `render_node.cpp:631-680`. The `_publish_camera_outputs` helper:

```python
def _publish_camera_outputs(self, sim_time, outputs):
    for cam in self._cameras:
        # RGB
        if self._ros.has_image_subscribers(cam.topic):
            with outputs.color_for(cam).map(device=ovrtx.Device.CPU) as img:
                self._ros.publish_camera_image(sim_time, cam.topic, img.dl)
            self._ros.publish_camera_info(sim_time, cam.topic, cam.intrinsics)
        # Depth (only when scene YAML set topic.depth)
        if cam.depth_topic and self._ros.has_image_subscribers(cam.depth_topic):
            with outputs.depth_for(cam).map(device=ovrtx.Device.CPU) as img:
                self._ros.publish_camera_image(sim_time, cam.depth_topic, img.dl)
            self._ros.publish_camera_info(sim_time, cam.depth_topic, cam.intrinsics)
```

The CPU-side `render_var.map(device=Device.CPU)` is the upstream-recommended path for "image data into a `sensor_msgs::Image`" (see `<OVRtx repo>/skills/reading-render-output/SKILL.md`). For zero-copy GPU forwarding to other GPU consumers (e.g. a future on-GPU image pipeline), `Device.CUDA` is the alternative — out of scope here since ROS messages need CPU-resident buffers.

**Stamping consistency.** All outputs from a single OVRtx frame are stamped with `sim_time = ros_bridge.last_sim_time()` captured *once* at the top of the frame. RGB and depth from the same frame share the stamp. The stamp may lag the most recent physics commit by one render-period (worst case ~1/30 s ≈ 33 ms at 30 FPS) — documented in `Failure modes & coupling` below.

**DDS depth mapping.** When a camera defines `dds_depth_topic`, the existing C++ plugin registers a name remap (`render_node.cpp:385-386`). For the inline path, this is a one-shot setup at `bind_newton_bodies` time — RosBridge gains a `register_dds_depth_mapping(rgb_topic, dds_depth_topic)` method that records the mapping and forwards it to the DDS layer if active. Implementer borrows from `RosImagePublisherPlugin::register_dds_depth_mapping` semantically; no shared compiled code.

### 2. The Warp kernel

**New file**: `scripts/engine/newton/visualizers/ovrtx_kernels.py`

```python
import warp as wp

@wp.kernel(enable_backward=False)
def sync_body_q_to_ovrtx_mat44d(
    ovrtx_xforms:   wp.array(dtype=wp.mat44d),
    body_indices:   wp.array(dtype=wp.int32),
    body_q:         wp.array(dtype=wp.transformf),
):
    """Newton state.body_q -> OVRtx row-major mat44d.

    OVRtx 0.3.0's "omni:xform" is USD row-vector convention (translation in
    last row at [3, 0..2]) per the writing-transforms skill. Warp's
    transform_to_matrix produces column-vector convention, so we transpose.
    Verified against <OVRtx repo>/tests/docs/python/test_attribute_bindings.py:21-39.
    """
    i = wp.tid()
    body_idx = body_indices[i]
    transform = body_q[body_idx]
    ovrtx_xforms[i] = wp.transpose(wp.mat44d(wp.transform_to_matrix(transform)))
```

The kernel writes directly into the OVRtx-mapped tensor — no intermediate buffer, no copy.

### 3. Engine integration

**Modify**: `scripts/engine/newton/engine.py`

Add a small hook so the engine can record a CUDA event after each step when an OVRtx visualizer is attached. No other changes to the engine class.

```python
class NewtonEngine(PhysicsEngine):
    def __init__(self, ...):
        ...
        self._physics_step_event = None  # set by attach_physics_event

    def attach_physics_event(self, event):
        """Called by the visualizer once at startup. The engine records
        `event` on Warp's current stream at the end of every tick_extras
        so a consumer thread can GPU-wait on the latest physics commit."""
        self._physics_step_event = event

    def tick_extras(self):
        super().tick_extras()  # existing Fabric write (no-op for newton-standalone)
        if self._physics_step_event is not None:
            wp.record_event(self._physics_step_event)  # one cheap call per tick
```

**Modify**: `scripts/genie_sim_engine_newton.py`

Replace the line-88 TODO branch with a real `ovrtx` dispatch in `run()`. The module-level headless guard at lines 64-70 is **unchanged** — it already does the right thing (downgrade to `none` when `headless:=true`).

```python
# === module level: just delete the TODO ovrtx branch at lines 88-94.
# The existing headless guard above already handles the `headless:=true` case.

# === run() ===
viewer = None      # Newton GL
ovrtx_viz = None   # InlineOvrtxVisualizer

if visualizer == "newton":
    viewer = _create_newton_viewer(session.sim, session.render_hz, logger)
elif visualizer == "ovrtx":
    from engine.newton.visualizers.ovrtx import InlineOvrtxVisualizer
    ovrtx_viz = InlineOvrtxVisualizer(
        ros_bridge=session.ros_bridge,
        scene_usda=session.scene_usda,
        robot_usda=session.robot_usda,
        render_layer_usda=session.render_layer_usda,
        cameras_cfg=session.cameras_cfg,
        device=str(wp.get_device()),
        render_fps=session.render_hz,
    )
    ovrtx_viz.bind_newton_bodies(session.sim._model)
    physics_step_event = wp.Event(device=wp.get_device())
    ovrtx_viz.attach_physics_event(physics_step_event)
    session.sim.attach_physics_event(physics_step_event)
    ovrtx_viz.start()

def _render_hook(now, next_tick, sim_time):
    if viewer is not None and now >= next_render[0]:
        # existing Newton GL path — unchanged
        ...
    # OVRtx path is a no-op here: rendering runs on its own thread,
    # gated by the physics CUDA event.
    return 0.0, 0.0, False

# _exit_check is unchanged. Newton-GL has window-close detection;
# OVRtx has no window (today), so default behaviour (always True)
# applies — shutdown is driven by Ctrl-C / ROS shutdown.

# After session.run() returns, clean up the OVRtx thread:
if ovrtx_viz is not None:
    ovrtx_viz.stop()
```

Key invariants:
- Only one new line on the physics hot path: `wp.record_event(...)` inside `tick_extras`. Cost ≈ microseconds.
- The OVRtx work happens entirely on the OVRtx thread.
- Only one visualizer runs at a time — enforced by the single-valued `physics_engine_visualizer` param. No fan-out logic is needed.
- The module-level `_HEADLESS` guard is unchanged; `ovrtx` is treated like `newton` for headless purposes (downgraded to `none` when no `--gui`).

### 4. Add `publish_image` / `publish_camera_image` / `publish_camera_info` / `has_image_subscribers` to `RosBridge`

**Modify** (within `genie_sim_engine` only): `genie_sim_engine/src/realtime_ros_node.{hpp,cpp}` and `src/pybinding.cpp`.

- Add the `RosBridge` methods listed in §1a:
  - `publish_camera_image(sim_time, topic, dl_tensor)`
  - `publish_camera_info(sim_time, topic, intrinsics)`
  - `has_image_subscribers(topic) -> bool`
  - `last_sim_time() -> double` (`std::atomic<double>`, set inside `publish_clock`)
  - `register_dds_depth_mapping(rgb_topic, dds_depth_topic)` (one-shot at startup)
- Add the matching `_core.publish_camera_image(...)` / `_core.publish_camera_info(...)` / `_core.has_image_subscribers(...)` Python bindings in `src/pybinding.cpp` with `py::gil_scoped_release unlock` — same idiom as `pybinding.cpp:98, 117, 133, 159`.
- `rclcpp::Publisher::publish` is documented thread-safe — no extra mutex needed.
- The publisher set is built lazily on first publish per topic (cache by topic name) or eagerly in `set_topology` if camera topics are known at startup (preferred — less GIL traffic in steady state).

The pattern is well-established in this codebase: `pybinding.cpp:121-135` (publish_body_tf_render) is the template.

> **Boundary reminder**: these methods are added under `genie_sim_engine/src/`. The cross-process `genie_sim_render/src/plugins/ros_image_publisher_plugin.cpp` is the *reference pattern* for the message construction (RGB encoding, depth encoding, info layout) but is **not** edited, linked into, or shared with `genie_sim_engine`. Re-implement equivalent serialisation in-engine — deduplication via a shared library is intentionally not done in v1.

### 5. Launcher / parameter wiring

**No new launcher YAML keys.** The existing `physics_engine_visualizer` parameter already supports `ovrtx` as a documented (but TODO) value in `launcher_newton_mjwarp.yaml:14-19` and `launcher_newton_fsvbd.yaml:11-16`. Filling in the implementation is the only change.

What to update:

- `launcher_newton_mjwarp.yaml:19` and `launcher_newton_fsvbd.yaml:16`: remove the `(falls back to none)` annotation on the `ovrtx` line in the visualizer comments.
- The existing `render_hz` (e.g. `launcher_newton_mjwarp.yaml:64`) drives OVRtx's render FPS — same field already used by Newton GL. Update its inline comment to reflect that it now also applies to `ovrtx`.
- `launcher.renders: []` (already empty in newton-standalone launchers) stays empty — the separate-process `render_ovrtx` C++ node is **not** used in this path. The inline OVRtx visualizer in the physics process is the renderer.
- Update `.agent/geniesim_ros.md` package-map row for `genie_sim_engine` to note that the `ovrtx` visualizer is now implemented for newton-standalone.

### 6. Dependency pinning

- `genie_sim_engine/package.xml` and pyproject: pin `ovrtx>=0.3.0,<0.4.0`.
- Reinstall reminder: a stale `site-packages/ovrtx/` may still be 0.2.x. Run `pip install -e <OVRtx-repo-root>[python]` (or the appropriate build target) before testing.

### 7. Stream scoping — the most-likely subtle bug

- Physics's Newton kernels run on Warp's default device stream.
- The OVRtx thread's `self._ovrtx_stream = wp.Stream(device=...)` is a separate, non-default stream.
- **Every `wp.launch` call in the OVRtx thread MUST pass `stream=self._ovrtx_stream` explicitly.** Without that, Warp falls back to its default stream and re-serialises with physics — silently losing all the parallelism this design is built around.
- The `mapping.unmap` and `render_var.map` calls must propagate the same stream so OVRtx queues against the right one.

### 8. Backpressure & frame-drop policy

- OVRtx renders at its target FPS. A slow frame doesn't queue up — `next_tick = now + period` so the next attempt happens immediately and aligns to the next slot.
- Physics never waits. If OVRtx is consistently slow, OVRtx falls further behind in wall time but always renders the *latest* `body_q` because the kernel reads live (gated only by the CUDA event handshake).
- Document this latest-state-wins semantic explicitly in the user-facing docs.

---

## Failure modes & coupling

The threading split eliminates contention on the **CPU thread axis** — physics never CPU-blocks on render finishing. But there are still four shared resources, and the design does *not* magic away contention on them:

| Resource | Shared? | Notes |
|---|---|---|
| GPU compute (SMs) | Yes — single device, two streams | CUDA scheduler fills idle SMs from either stream; truly parallel only when device has spare capacity |
| GPU memory bandwidth | Yes | Hardware-level; no software guard |
| Python GIL | Yes | Released during all C++ extension calls (Warp, OVRtx, rclcpp) |
| PCIe bandwidth | Yes | Mostly only OVRtx uses it (CPU image readback for ROS publish) |
| CPU cores | No | Physics has its own thread, OVRtx has its own thread |
| `body_q` / mapped buffer | No race | CUDA event + stream-ordered unmap |

### Scenario A: physics runs slower than its target (heavy contacts, complex scene)

Cause: Newton's MJWarp solver iterations grow on hard contact configurations; `wp.capture_launch` takes longer.

**Effect on render**:
- OVRtx thread waits on `physics_step_event` via `ovrtx_stream.wait_event(...)` — a GPU-side wait, enqueued onto the stream. CPU returns immediately, so the OVRtx Python loop is not blocked.
- The sync kernel can't actually start until physics signals the event, so OVRtx's GPU pipeline throttles to whatever rate physics is committing.
- **Render rate gets capped at physics tick rate.** If physics drops from 200 Hz to 50 Hz with render targeting 30 FPS, render is unaffected. If physics drops below render target, render drops with it.

**Latency stays good**: every rendered frame uses the most-recently-committed `body_q` (event handshake gates on the latest commit).

### Scenario B: render runs slower than its target (heavy RTX, high resolution)

Cause: 4K resolution, many lights, complex materials — render takes 50 ms when budget is 33 ms.

**Effect on render**: target FPS missed. Frame loop doesn't queue up — `next_tick = now + period` aligns to the next slot, dropping intermediate slots.

**Effect on physics — this is where coupling *does* show up**:

1. **GPU contention (the dominant effect).** OVRtx render kernels and Newton step kernels both run on the same device. Even with separate streams, CUDA schedules onto SMs based on availability. If OVRtx uses 80% of SMs, Newton gets the remaining 20% and `wp.capture_launch` takes longer to drain.
   - Symptom: `note_step_timing` reports growing tick wall time even though physics CPU work itself didn't slow down — the GPU just took longer.
   - If physics target has headroom (e.g., 200 Hz needs 5 ms/tick on a 4090), this is invisible. If physics is already GPU-saturated, render slowdowns directly slow physics.

2. **GIL contention (small).** OVRtx thread holds the GIL during Python orchestration (timer arithmetic, dict lookups, context-manager bookkeeping). Physics thread releases the GIL during `wp.capture_launch` but holds it during Python-level publish helpers. Net effect: a few hundred microseconds of GIL waiting per tick — negligible at 200 Hz physics, possibly measurable at 1000 Hz.

3. **PCIe contention (small).** OVRtx's `render_var.map(device=Device.CPU)` does a GPU→CPU image copy each frame. Physics's CPU↔GPU traffic is only the CUDA event signal and the small joint-state numpy fetch. Contention here is unlikely to matter unless camera resolution is very high.

**Quick math for joint saturation.** Physics 200 Hz × 5 ms/tick = 1000 ms/sec of GPU work. Render 30 FPS × 20 ms/frame = 600 ms/sec. Total demand 1600 ms/sec on a 1000 ms/sec budget → both run at ~62% of target. Without parallel streams (the old serial design) they'd be effectively 100% serialised, so the parallel design *helps* — but it doesn't conjure capacity that doesn't exist.

### Symptoms by failure mode

| Failure | `note_step_timing` | `/clock` rate | `/camera/.../image` rate | Mitigation |
|---|---|---|---|---|
| Physics-bound | tick wall time ↑ | drops below target | drops, capped at physics rate | Reduce contact complexity, raise solver tolerance, drop physics Hz |
| Render-bound (GPU) | tick wall time ↑ slightly | drops slightly | below `render_fps` | Lower render resolution, lower `inline_ovrtx_render_fps`, simpler materials |
| Render-bound (CPU/Python) | mostly unchanged | mostly unchanged | drops | Profile OVRtx thread, check ROS topic backlog |
| Both saturated | ↑ | drops | below target | Multi-GPU split, lower targets |

### What the design DOES guarantee

1. **Physics CPU thread never blocks on render finishing.** OVRtx work runs entirely on a different thread + different CUDA stream. No Python `.join()`, no `wait()`, no `wp.synchronize_device()` on the physics path.
2. **Render always uses fresh physics state.** Each frame uses the most recent committed `body_q` via the CUDA event handshake.
3. **No CPU-side mutex.** Physics never waits for OVRtx to release a buffer.
4. **Failure is graceful.** Slow render drops frames silently rather than queuing — OVRtx doesn't accumulate latency.

### What the design does NOT guarantee

1. **Independent throughput.** Single GPU is shared — saturating one will throttle the other.
2. **Hard real-time physics.** GPU contention from OVRtx adds variance to tick wall time. If physics needs hard real-time guarantees, OVRtx must be on a separate GPU.
3. **Render at exactly `render_fps`.** Latest-state-wins means slow frames get skipped, not queued.

### Mitigations available in the codebase / config

1. **Throttle render**: `inline_ovrtx_render_fps:=15` cuts GPU time roughly in half.
2. **Smaller render output**: lower-resolution camera config in `cameras_cfg`.
3. **Subscribe-aware rendering**: `render_node.cpp:313` already skips render when no subscriber exists. Port the same check into `InlineOvrtxVisualizer._render_one_frame` — if `_ros.has_image_subscribers(cam_name)` returns false for all cameras, skip the entire frame (sync kernel, render, publish). Saves all GPU + CPU + PCIe cost when nobody is looking.
4. **Realtime scheduling for physics**: `realtime_scheduler.cpp` pins the physics thread / sets SCHED_FIFO. The OVRtx thread runs at default priority, so the kernel preempts it for physics when CPU is the bottleneck. Do not raise OVRtx thread priority.
5. **Multi-GPU split** (out of v1 scope): physics on GPU 0, OVRtx on GPU 1 — eliminates GPU contention entirely. Newton supports per-device placement; OVRtx accepts a device id at renderer creation.

### Practical expectation

For typical robot-sim workloads (hundreds of bodies, 200 Hz physics, 30 FPS 720p render, single RTX 4090-class GPU) there's enough GPU headroom that scenarios A and B are rare, and when they happen the failure is graceful (one-sided throughput drop, no instability).

Verification step #3 (physics throughput within 5% of physics-only baseline) is the empirical guardrail — if it fails on real hardware, the scene is already over the GPU's capacity and no software change will help: reduce the workload or split GPUs.

---

## Critical files and existing utilities to reuse

> **Boundary**: every file under `genie_sim_engine/` is fair game to modify. Every file under `genie_sim_render/` is **read-only reference** — borrow patterns, do not edit.

### Modify (under `genie_sim_engine/`)

| File | What changes |
|---|---|
| `genie_sim_engine/scripts/genie_sim_engine_newton.py` | Replace TODO `ovrtx` branch (line 88-94) with real dispatch in `run()`; add `ovrtx_viz.stop()` cleanup |
| `genie_sim_engine/scripts/engine/newton/engine.py:458-498` | Add `attach_physics_event` method; `tick_extras` records the event when set |
| `genie_sim_engine/scripts/engine/newton/visualizers/ovrtx.py` | **NEW** — `InlineOvrtxVisualizer` class |
| `genie_sim_engine/scripts/engine/newton/visualizers/ovrtx_kernels.py` | **NEW** — `sync_body_q_to_ovrtx_mat44d` Warp kernel |
| `genie_sim_engine/src/realtime_ros_node.{hpp,cpp}:153-238` | Add `publish_camera_image`, `publish_camera_info`, `has_image_subscribers`, `last_sim_time`, `register_dds_depth_mapping` |
| `genie_sim_engine/src/pybinding.cpp:94-100, 117-118, 133-134, 159-160` | Add matching `_core.*` bindings using existing GIL-release template |
| `genie_sim_engine/scripts/common/loop.py:138-150` | **No changes** — `EngineRunLoop` contract unchanged; OVRtx runs on its own thread |
| `genie_sim_engine/package.xml` and pyproject pin | Pin `ovrtx>=0.3.0,<0.4.0` |
| `launcher_newton_mjwarp.yaml:14-19, 64`, `launcher_newton_fsvbd.yaml:11-16` | Drop `(falls back to none)` annotation from `ovrtx` line; extend `render_hz` comment to cover OVRtx |
| `.agent/geniesim_ros.md` (package-map row for `genie_sim_engine`) | Note the `ovrtx` visualizer is now implemented |

### Read-only references (do NOT edit)

| File | What to borrow conceptually |
|---|---|
| `genie_sim_render/src/render_node.cpp:281-326` | OVRtx renderer init sequence (config entries, USD load, USD references) — port to Python in `InlineOvrtxVisualizer.__init__` |
| `genie_sim_render/src/render_node.cpp:313` | Subscriber-count gate pattern (`get_subscription_count() > 0`) — replicate via `RosBridge::has_image_subscribers` in-engine |
| `genie_sim_render/src/render_node.cpp:631-680` | Output extraction + frame stamping — port to `_publish_camera_outputs` in Python |
| `genie_sim_render/src/plugins/ros_image_publisher_plugin.cpp:30-74` | `sensor_msgs::Image` / `CameraInfo` construction (encoding, intrinsics layout) — re-express in `RosBridge::publish_camera_image/info` |
| `genie_sim_render/src/render_node.cpp:233-235, 385-393` | Camera-config schema (`depth_topic`, `dds_depth_topic`, DDS depth mapping) — replicate in-engine `CameraCfg` dataclass |

### OVRtx 0.3.0 skill / test references (read-only)

| File | What to reuse |
|---|---|
| `<OVRtx repo>/skills/mapping-attributes/SKILL.md` | **Authoritative pattern for this design** — zero-copy `binding.map(device=Device.CUDA)` + Warp kernel + `unmap(stream=...)` |
| `<OVRtx repo>/skills/cuda-interop/SKILL.md` | Stream/event handshake rules; the "no manual `wp.synchronize_stream` needed" detail |
| `<OVRtx repo>/skills/writing-transforms/SKILL.md` | `"omni:xform"`, `XFORM_MAT4x4`, USD row-vector convention, `float64` |
| `<OVRtx repo>/skills/reading-render-output/SKILL.md` | `render_var.map(device=Device.CPU)` for sensor_msgs::Image extraction |
| `<OVRtx repo>/skills/stepping-and-rendering/SKILL.md` | `renderer.step(...)` / `fetch_results` lifecycle |
| `<OVRtx repo>/tests/docs/python/test_attribute_bindings.py:171-189` (snippet `doc-map-attribute-cuda`) | Verbatim source of the hot-path map/launch/unmap idiom |
| `<OVRtx repo>/tests/docs/python/test_attribute_bindings.py:21-39` | Matrix-layout reference (translation at `[3,0]`) for verification |
| `<OVRtx repo>/examples/python/minimal/main.py` | Renderer config + USD load + step loop |
| `IsaacLab/source/isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer.py:456-480` | Reference Python implementation of the same pattern (pre-0.3 API but logically identical) |
| `IsaacLab/source/isaaclab_ov/isaaclab_ov/renderers/ovrtx_renderer_kernels.py:312-322` | `sync_newton_transforms_kernel` — verbatim port |

---

## Verification

1. **OVRtx version guard at startup**: `ovrtx.get_version() == (0, 3, _)`. Refuse to start otherwise — header (`<OVRtx repo>`) vs installed wheel (`~/.local/.../ovrtx/`) skew is the most common setup bug. Check inside `InlineOvrtxVisualizer.__init__`.

2. **Smoke test (kit-free newton + inline OVRtx)**:
   ```bash
   ros2 launch genie_sim_bringup app.launch.py \
       scene:=<scene> \
       launcher_config:=launcher_newton_mjwarp \
       headless:=false \
       physics_engine_visualizer:=ovrtx
   ```
   Confirm: no `omni.*` modules loaded (`pmap` / no Kit in process); no `render_ovrtx` ROS node in `ros2 node list`; camera image topics publish at the configured `render_hz`; physics ticks at target Hz (`note_step_timing` in `common/loop.py:227`).

3. **The headline test — physics throughput is unchanged**:
   - Baseline A: `physics_engine_visualizer:=none` (or `headless:=true`). Measure mean physics tick wall time over 60 s.
   - Test B: `physics_engine_visualizer:=ovrtx` with `headless:=false`, 30 Hz render.
   - **Pass criterion**: B is within 5 % of A. A larger slowdown means physics is blocking on OVRtx — likely a stream-scoping bug (§7).

4. **Headless invariant**: with `headless:=true` and `physics_engine_visualizer:=ovrtx`, the module-level guard at `genie_sim_engine_newton.py:64-70` must downgrade to `none`. Verify the log line and confirm no OVRtx renderer is constructed (`ovrtx_viz is None` in `run()`).

4. **Matrix-layout sanity check** (catch row/col-major mistake early):
   - Place one body at translation `(5, 0, 0)`.
   - After one step + one OVRtx render: read back `renderer.read_attribute("omni:xform", [path])`.
   - Verify `5.0` at `[3, 0]` (USD row-vector). If at `[0, 3]`, drop the `wp.transpose` in the kernel.

5. **CUDA stream parallelism check** via `nsys profile -t cuda,nvtx`:
   - Physics's Newton-graph kernels on stream X.
   - OVRtx sync kernel + render on stream Y (`self._ovrtx_stream`).
   - Visible overlap on the timeline — i.e., physics and OVRtx kernels execute concurrently.

6. **Side-by-side correctness vs separate-process path**: SSIM > 0.99 on the same scripted trajectory at `t=5.0s`.

7. **Latency**: measure `physics_step_done_timestamp → first_pixel_published_timestamp`. Inline should beat the legacy path by the ROS-serialise + middleware + deserialise cost (0.5–2 ms typical).

8. **Stability**: 60+ min run with body count 1000+. `ovrtx.get_last_error()` stays empty; no Warp graph-capture failures; no rendered-frame tearing; no unbounded memory growth (verify `destroy_results` cleanup).

9. **Backward compatibility**: `physics_engine_visualizer:=none` and `headless:=true` produce the exact same behaviour as today. `colcon test` + existing render integration tests pass.

10. **Shutdown safety**: `Ctrl-C` cleanly stops the OVRtx thread (`ovrtx_viz.stop()` after `session.run()` returns); OVRtx thread joins within 5 s; `nvidia-smi` after exit shows no zombie context.

---

## Out-of-scope (explicitly)

- **No edits to `genie_sim_render/`**: the cross-process renderer continues to exist and operate as today. Code patterns are borrowed (read-only); files are not modified, recompiled into `genie_sim_engine`, or otherwise touched. The two paths are mutually exclusive at runtime — a launcher uses one or the other.
- **No Kit involvement, anywhere**: this entry point is newton-standalone. For the Kit viewport, use `physics_engine:=isaac_newton` from `genie_sim_engine_isaacsim.py` instead.
- **No Fabric writes**: newton-standalone is Kit-free, so there is no USDRT stage to write into; `_write_body_transforms()` is already a no-op here. OVRtx's mapping is into its own internal Fabric (different system; not Kit's USDRT Fabric).
- **Multi-visualizer fan-out**: only one of `none | newton | ovrtx | rerun` runs at a time. Enforced by the single-valued `physics_engine_visualizer` param.
- **OVRtx-with-window GUI**: a future GUI for OVRtx may be added. Treating `ovrtx` as a non-headless visualizer (requires `headless:=false`) keeps that path open without a rules change later.
- **Sensor types other than camera RGB / depth**: lidar, IMU, tactile, force/torque, GPS — none are wired into newton-standalone today (`lidars: []` is reserved schema). When added, GPU-derived sensors fit the OVRtx-thread model, pure-physics-derived sensors fit the physics-thread model.
- **Code deduplication between `genie_sim_engine` and `genie_sim_render`**: extracting a shared `geniesim_ovrtx_common` library is a future cleanup, not a v1 goal. The two paths intentionally re-implement the small overlap (renderer init, image serialisation) so each can evolve independently.
- **`rerun` implementation**: a parallel TODO, not addressed by this plan.
- **PhysX engine support**: PhysX doesn't expose body transforms as Warp arrays. Out of scope.
- **OVRtx ≤ 0.2.x compatibility**: 0.3.x only.
- **`OVRTX_MAP_DEVICE_TYPE_CUDA_ARRAY` / Vulkan interop**: documented in the `cuda-interop` skill but not needed here — ROS publish uses the standard `render_var.map(device=Device.CPU)` path.
