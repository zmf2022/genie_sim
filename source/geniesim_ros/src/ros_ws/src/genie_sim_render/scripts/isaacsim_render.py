#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
isaacsim_render.py

Isaac Sim replacement for ``genie_sim_render_node`` (ovrtx-based).

Mirrors the C++ node contract:
  * Reads the ``manifest.json`` produced by ``assemble_scene`` (same JSON
    schema as the C++ RenderNode):
      ``scene_usda`` / ``usd_path``, ``robot_usda``, ``render_layer_usda``,
      ``robot_prefix``, ``free_cam_prim_path``, ``base_path``,
      ``cameras[...]`` (each with ``path``, ``topic``, ``render_product_path``,
      intrinsics, optional ``depth_topic``, ``is_free_cam``).
  * Loads the scene + optional robot sublayer + optional render layer into
    the Isaac Sim USD stage.
  * Subscribes to ``/tf_render`` and applies every transform to the matching
    prim under ``/{robot_prefix}`` (mirrors ``RenderNode::on_tf_render``).
  * Subscribes to ``~/free_cam_pose`` and applies it to ``free_cam_prim_path``.
  * Publishes RGB images + CameraInfo per manifest camera through the Isaac
    Sim ROS2 bridge OmniGraph. Topics follow the same convention as the
    default ``RosImagePublisherPlugin``: ``<topic>/image_raw`` and
    ``<topic>/camera_info``.

Run with the system python3 that has ``isaacsim`` installed via pip:

    python3 isaacsim_render.py --ros-args \\
        -p stage_manifest:=/path/to/manifest.json \\
        -p render_fps:=30.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Launch SimulationApp FIRST. All omni.* imports must come after this.
# ---------------------------------------------------------------------------
import isaacsim  # noqa: F401  (registers extensions)
from isaacsim import SimulationApp


def _env_truthy(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off", "")


# Headless by default; override with ISAACSIM_HEADLESS=0 to show the viewport.
_DEFAULT_HEADLESS = _env_truthy("ISAACSIM_HEADLESS", True)
simulation_app = SimulationApp({"headless": _DEFAULT_HEADLESS})

# ---------------------------------------------------------------------------
# Omni / Isaac Sim imports (post-SimulationApp)
# ---------------------------------------------------------------------------
import omni  # noqa: E402
import omni.graph.core as og  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import (  # noqa: E402
    add_reference_to_stage,
    open_stage,
    get_current_stage,
)
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

# ROS2 bridge extension must be enabled before creating the camera graph
enable_extension("isaacsim.ros2.bridge")
enable_extension("omni.graph.core")

# ---------------------------------------------------------------------------
# ROS2
# ---------------------------------------------------------------------------
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from geometry_msgs.msg import PoseStamped  # noqa: E402
from std_msgs.msg import Float64MultiArray  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

_SENSOR_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
)


@dataclass
class CameraConfig:
    path: str
    topic: str
    render_product_path: str
    width: int = 1280
    height: int = 800
    fx: float = 610.0
    fy: float = 610.0
    cx: float = 640.0
    cy: float = 400.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    min_range: float = 0.01
    max_range: float = 10000.0
    is_free_cam: bool = False


@dataclass
class Manifest:
    scene_usda: str
    render_layer_usda: str
    robot_usda: str = ""
    robot_prefix: str = ""
    free_cam_prim_path: str = ""
    cameras: List[CameraConfig] = field(default_factory=list)

    @classmethod
    def load(cls, manifest_path: Path) -> "Manifest":
        """Load a ``manifest.json`` produced by ``assemble_scene``.

        Mirrors the C++ ``RenderNode::load_manifest`` logic exactly:
        paths stored relative to ``base_path`` (or the manifest's own
        directory when ``base_path`` is absent) are resolved to absolute.
        """
        manifest_path = manifest_path.resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path) as f:
            data = json.load(f)

        # Resolve relative paths against base_path (same logic as C++ node).
        if data.get("base_path"):
            base_dir = Path(data["base_path"])
        else:
            base_dir = manifest_path.parent

        def _resolve(p: str) -> str:
            if not p:
                return p
            pp = Path(p)
            return str(pp if pp.is_absolute() else (base_dir / pp).resolve())

        scene_usda = _resolve(data.get("scene_usda") or data.get("usd_path", ""))
        if not scene_usda:
            raise KeyError("Manifest missing both 'scene_usda' and 'usd_path'")
        robot_usda = _resolve(data.get("robot_usda", ""))
        render_layer_usda = _resolve(data.get("render_layer_usda", ""))
        robot_prefix = data.get("robot_prefix", "")
        free_cam_prim_path = data.get("free_cam_prim_path", "")

        if not Path(scene_usda).exists():
            raise FileNotFoundError(f"Scene USD not found: {scene_usda}")
        if robot_usda and not Path(robot_usda).exists():
            raise FileNotFoundError(f"Robot USD not found: {robot_usda}")
        if render_layer_usda and not Path(render_layer_usda).exists():
            raise FileNotFoundError(f"Render layer USDA not found: {render_layer_usda}")

        cams: List[CameraConfig] = []
        for i, cj in enumerate(data.get("cameras", [])):
            width = int(cj.get("width", 1280))
            height = int(cj.get("height", 800))
            render_product_path = cj.get("render_product_path") or f"/RenderOVRTX/Cam_{i}"
            cams.append(
                CameraConfig(
                    path=cj["path"],
                    topic=cj["topic"],
                    render_product_path=render_product_path,
                    width=width,
                    height=height,
                    fx=float(cj.get("fx", 610.0)),
                    fy=float(cj.get("fy", 610.0)),
                    cx=float(cj.get("cx", width / 2.0)),
                    cy=float(cj.get("cy", height / 2.0)),
                    k1=float(cj.get("k1", 0.0)),
                    k2=float(cj.get("k2", 0.0)),
                    p1=float(cj.get("p1", 0.0)),
                    p2=float(cj.get("p2", 0.0)),
                    k3=float(cj.get("k3", 0.0)),
                    min_range=float(cj.get("min_range", 0.01)),
                    max_range=float(cj.get("max_range", 10000.0)),
                    is_free_cam=bool(cj.get("is_free_cam", False)),
                )
            )

        return cls(
            scene_usda=scene_usda,
            render_layer_usda=render_layer_usda,
            robot_usda=robot_usda,
            robot_prefix=robot_prefix,
            free_cam_prim_path=free_cam_prim_path,
            cameras=cams,
        )


def _set_camera_intrinsics(prim, cam: CameraConfig, focal_length: float = 1.0) -> None:
    """Configure Isaac Sim OpenCV-pinhole intrinsics from a manifest CameraConfig.

    Same mapping used by isaacsim_helper.CameraConfig.set_intrinsics.
    """
    horizontal_aperture = cam.width * focal_length / cam.fx
    vertical_aperture = horizontal_aperture * (cam.height / cam.width)
    prim.GetAttribute("focalLength").Set(focal_length)
    prim.GetAttribute("horizontalAperture").Set(horizontal_aperture)
    prim.GetAttribute("verticalAperture").Set(vertical_aperture)
    cx_attr = prim.GetAttribute("omni:lensdistortion:opencvPinhole:cx")
    cy_attr = prim.GetAttribute("omni:lensdistortion:opencvPinhole:cy")
    fx_attr = prim.GetAttribute("omni:lensdistortion:opencvPinhole:fx")
    fy_attr = prim.GetAttribute("omni:lensdistortion:opencvPinhole:fy")
    if cx_attr and cx_attr.IsValid():
        cx_attr.Set(float(cam.cx))
    if cy_attr and cy_attr.IsValid():
        cy_attr.Set(float(cam.cy))
    if fx_attr and fx_attr.IsValid():
        fx_attr.Set(float(cam.fx))
    if fy_attr and fy_attr.IsValid():
        fy_attr.Set(float(cam.fy))


def _apply_xform(prim, translation: Gf.Vec3d, rotation: Gf.Quatd) -> None:
    """Write translate + orient onto an existing xform prim."""
    if not prim or not prim.IsValid():
        return
    t_attr = prim.GetAttribute("xformOp:translate")
    if not t_attr or not t_attr.IsValid():
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(translation)
        xformable.AddOrientOp().Set(Gf.Quatf(rotation))
        return
    t_attr.Set(translation)
    o_attr = prim.GetAttribute("xformOp:orient")
    if o_attr and o_attr.IsValid():
        if o_attr.GetTypeName() == Sdf.ValueTypeNames.Quatf:
            o_attr.Set(Gf.Quatf(rotation))
        else:
            o_attr.Set(rotation)


def _build_camera_omnigraph(graph_path: str, cameras: List[CameraConfig], fps_skip: int) -> None:
    """Create a single OmniGraph that runs a ROS2CameraHelper + CameraInfoHelper per camera."""
    nodes = [("Tick", "omni.graph.action.OnPlaybackTick")]
    edges = []
    for i, _cam in enumerate(cameras):
        nodes += [
            (f"CreateRP{i}", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
            (f"CamHelper{i}", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (f"CamInfo{i}", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
        ]
        edges += [
            ("Tick.outputs:tick", f"CreateRP{i}.inputs:execIn"),
            (f"CreateRP{i}.outputs:execOut", f"CamHelper{i}.inputs:execIn"),
            (f"CreateRP{i}.outputs:execOut", f"CamInfo{i}.inputs:execIn"),
            (f"CreateRP{i}.outputs:renderProductPath", f"CamHelper{i}.inputs:renderProductPath"),
            (f"CreateRP{i}.outputs:renderProductPath", f"CamInfo{i}.inputs:renderProductPath"),
        ]

    og.Controller.edit(
        {"graph_path": graph_path},
        {
            og.Controller.Keys.CREATE_NODES: nodes,
            og.Controller.Keys.CONNECT: edges,
        },
    )

    for i, cam in enumerate(cameras):
        og.Controller.set(f"{graph_path}/CreateRP{i}.inputs:cameraPrim", cam.path)
        og.Controller.set(f"{graph_path}/CreateRP{i}.inputs:width", int(cam.width))
        og.Controller.set(f"{graph_path}/CreateRP{i}.inputs:height", int(cam.height))
        og.Controller.set(f"{graph_path}/CamHelper{i}.inputs:topicName", cam.topic + "/image_raw")
        og.Controller.set(f"{graph_path}/CamHelper{i}.inputs:frameId", cam.path)
        og.Controller.set(f"{graph_path}/CamHelper{i}.inputs:frameSkipCount", int(fps_skip))
        og.Controller.set(f"{graph_path}/CamInfo{i}.inputs:topicName", cam.topic + "/camera_info")
        og.Controller.set(f"{graph_path}/CamInfo{i}.inputs:frameId", cam.path)
        og.Controller.set(f"{graph_path}/CamInfo{i}.inputs:queueSize", 1)
        og.Controller.set(f"{graph_path}/CamInfo{i}.inputs:frameSkipCount", int(fps_skip))


class IsaacSimRenderNode(Node):
    """ROS2 node that drives Isaac Sim rendering to mirror genie_sim_render_node."""

    def __init__(self) -> None:
        super().__init__("render_isaacsim")

        # Parameters (names mirror the C++ RenderNode)
        self.declare_parameter("stage_manifest", "")
        self.declare_parameter("render_fps", 30.0)
        self.declare_parameter("prim_paths", "")

        manifest_path = self.get_parameter("stage_manifest").get_parameter_value().string_value
        self._render_fps = float(self.get_parameter("render_fps").get_parameter_value().double_value or 30.0)

        if not manifest_path:
            raise RuntimeError("Parameter 'stage_manifest' is required")
        self._manifest = Manifest.load(Path(manifest_path))

        self._timing_pub = self.create_publisher(Float64MultiArray, "~/render_timing", 10)
        self._tf_sub = self.create_subscription(TFMessage, "/tf_render", self._on_tf_render, _SENSOR_QOS)
        if self._manifest.free_cam_prim_path:
            self._free_cam_sub = self.create_subscription(
                PoseStamped, "~/free_cam_pose", self._on_free_cam_pose, _SENSOR_QOS
            )

        self._world: Optional[World] = None
        self._stage = None
        self._prim_cache: Dict[str, object] = {}
        self._cam_body_to_render_path: Dict[str, str] = {}
        self._frame_count = 0
        self._drop_count = 0
        self._last_frame_time = 0.0

        self._init_stage()
        self._setup_cameras()

        self.get_logger().info(
            f"Isaac Sim render node ready ({self._render_fps:.1f} fps, "
            f"{len(self._manifest.cameras)} cameras, manifest={manifest_path})"
        )

    # -- stage / cameras --------------------------------------------------

    def _init_stage(self) -> None:
        self.get_logger().info(f"Opening scene USD: {self._manifest.scene_usda}")
        open_stage(self._manifest.scene_usda)
        for _ in range(10):
            simulation_app.update()
        self._world = World(stage_units_in_meters=1.0)
        self._stage = get_current_stage()

        if self._manifest.robot_usda:
            robot_root = "/" + self._manifest.robot_prefix if self._manifest.robot_prefix else "/Robot"
            self.get_logger().info(f"Attaching robot USD: {self._manifest.robot_usda} at {robot_root}")
            add_reference_to_stage(self._manifest.robot_usda, robot_root)

        # The render-layer USDA parallels the ovrtx "/RenderOVRTX" sublayer. It may
        # contain camera prim definitions; reference it under the same root so that
        # manifest paths starting with "/RenderOVRTX/..." resolve. When the
        # render_config is a plain scene JSON (no render_layer_usda), skip it —
        # camera prims are expected to live directly under the scene USD.
        if self._manifest.render_layer_usda:
            self.get_logger().info(f"Attaching render layer USD: {self._manifest.render_layer_usda}")
            add_reference_to_stage(self._manifest.render_layer_usda, "/RenderOVRTX")
        else:
            self.get_logger().info("No render_layer_usda configured; using scene USD camera prims directly")

        self._world.reset()

        # Build body-name -> /RenderOVRTX/Cameras/<body> mapping (matches C++ node).
        for cam in self._manifest.cameras:
            if cam.is_free_cam:
                continue
            slash = cam.path.find("/", 1) if cam.path.startswith("/") else cam.path.find("/")
            if slash != -1:
                body_name = cam.path[:slash].lstrip("/")
                self._cam_body_to_render_path[body_name] = f"/RenderOVRTX/Cameras/{body_name}"

    def _setup_cameras(self) -> None:
        if not self._manifest.cameras:
            self.get_logger().warn("No cameras configured in manifest")
            return

        for i, cam in enumerate(self._manifest.cameras):
            prim = self._stage.GetPrimAtPath(cam.path)
            if not prim or not prim.IsValid():
                self.get_logger().warn(f"Camera prim {cam.path} not in stage; skipping intrinsics")
                continue
            try:
                _set_camera_intrinsics(prim, cam)
            except Exception as exc:  # pragma: no cover - best-effort logging
                self.get_logger().warn(f"Intrinsics setup failed for {cam.path}: {exc}")
            self.get_logger().info(f"Camera[{i}]: {cam.render_product_path} -> {cam.topic} ({cam.width}x{cam.height})")

        # Frame-skip keeps the ROS publish rate near render_fps given Isaac's
        # default 60Hz playback. 0 means publish every tick.
        fps_skip = max(0, int(round(60.0 / max(self._render_fps, 1e-3))) - 1)
        _build_camera_omnigraph("/World/CameraGraph", self._manifest.cameras, fps_skip)

    # -- subscriptions ----------------------------------------------------

    def _get_prim(self, prim_path: str):
        prim = self._prim_cache.get(prim_path)
        if prim is None:
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim and prim.IsValid():
                self._prim_cache[prim_path] = prim
            else:
                return None
        return prim

    def _on_tf_render(self, msg: TFMessage) -> None:
        if self._stage is None:
            return
        prefix = self._manifest.robot_prefix
        for tf in msg.transforms:
            body = tf.child_frame_id
            if body.startswith("/World") or body.startswith("/"):
                prim_path = body
            elif prefix:
                prim_path = f"/{prefix}/{body}"
            else:
                prim_path = f"/{body}"

            t = tf.transform.translation
            r = tf.transform.rotation
            translation = Gf.Vec3d(float(t.x), float(t.y), float(t.z))
            rotation = Gf.Quatd(float(r.w), float(r.x), float(r.y), float(r.z))

            prim = self._get_prim(prim_path)
            if prim is not None:
                _apply_xform(prim, translation, rotation)

            render_path = self._cam_body_to_render_path.get(body)
            if render_path:
                render_prim = self._get_prim(render_path)
                if render_prim is not None:
                    _apply_xform(render_prim, translation, rotation)

    def _on_free_cam_pose(self, msg: PoseStamped) -> None:
        if self._stage is None or not self._manifest.free_cam_prim_path:
            return
        p = msg.pose.position
        o = msg.pose.orientation
        translation = Gf.Vec3d(float(p.x), float(p.y), float(p.z))
        rotation = Gf.Quatd(float(o.w), float(o.x), float(o.y), float(o.z))
        prim = self._get_prim(self._manifest.free_cam_prim_path)
        if prim is not None:
            _apply_xform(prim, translation, rotation)

    # -- main loop --------------------------------------------------------

    def _publish_timing(self, step_ms: float, total_ms: float, interval_ms: float) -> None:
        msg = Float64MultiArray()
        msg.layout.dim.append(msg.layout.dim.__class__())
        msg.layout.dim[0].label = "timing"
        msg.layout.dim[0].size = 7
        msg.layout.dim[0].stride = 7
        # [step, wait, fetch, map, publish, total, interval]
        msg.data = [step_ms, 0.0, 0.0, 0.0, 0.0, total_ms, interval_ms]
        self._timing_pub.publish(msg)

    def spin(self) -> None:
        period = 1.0 / max(self._render_fps, 1e-3)
        next_tick = time.monotonic()
        while simulation_app.is_running() and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.0)

            now = time.monotonic()
            if now < next_tick:
                time.sleep(min(next_tick - now, period))
                continue

            interval_ms = 0.0 if self._last_frame_time == 0.0 else (now - self._last_frame_time) * 1000.0
            self._last_frame_time = now

            t0 = time.monotonic()
            try:
                self._world.step(render=True)
            except Exception as exc:  # pragma: no cover
                self._drop_count += 1
                self.get_logger().warn(f"world.step failed: {exc}")
                next_tick = now + period
                continue
            step_ms = (time.monotonic() - t0) * 1000.0
            total_ms = step_ms  # Isaac Sim handles publishing asynchronously via OmniGraph

            self._frame_count += 1
            self._publish_timing(step_ms, total_ms, interval_ms)

            next_tick += period
            if next_tick < now:
                next_tick = now + period

    def shutdown(self) -> None:
        try:
            if self._world is not None:
                self._world.stop()
        except Exception:  # pragma: no cover
            pass


def _strip_ros_args(argv: List[str]) -> List[str]:
    """Return argv without the ROS2 '--ros-args ...' section."""
    if "--ros-args" not in argv:
        return argv
    idx = argv.index("--ros-args")
    return argv[:idx]


def main() -> int:
    argv = list(sys.argv)
    parser = argparse.ArgumentParser(description="Isaac Sim renderer (mirrors genie_sim_render_node).")
    parser.add_argument(
        "--manifest",
        default=os.environ.get("STAGE_MANIFEST", ""),
        help=(
            "Path to the stage manifest JSON. If omitted, the ROS parameter "
            "'stage_manifest' must be set (e.g. via --ros-args -p stage_manifest:=...)."
        ),
    )
    cli_args, _ = parser.parse_known_args(_strip_ros_args(argv)[1:])

    rclpy.init(args=argv)

    extra_params: List[rclpy.parameter.Parameter] = []
    if cli_args.manifest:
        extra_params.append(rclpy.parameter.Parameter("stage_manifest", value=cli_args.manifest))

    node: Optional[IsaacSimRenderNode] = None
    try:
        node = IsaacSimRenderNode()
        if extra_params:
            node.set_parameters(extra_params)
        node.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[isaacsim_render] fatal: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
