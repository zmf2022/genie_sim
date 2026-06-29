# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone visualizers (RViz markers + OVRtx viewport).

Two independent paths share this subpackage:

  * :mod:`markers` — sibling rclpy publishers (one shared node) that
    surface engine state to RViz: ``DeformableMarkerPublisher`` for
    cloth / FEM tet surfaces, ``DeformablePointCloudPublisher`` for raw
    ``particle_q`` as a PointCloud2, ``ObjectMarkerPublisher`` for
    FREE-joint rigid bodies.  Gated by ``newton.debug.pub_*`` scene-yaml
    flags.

  * :mod:`ovrtx` — in-process OVRtx (path-traced) viewport with its
    own GPU stream, camera management, and per-frame compositor.
    Companion :mod:`ovrtx_camera` holds ``CameraCfg`` and manifest
    loading; :mod:`ovrtx_kernels` holds the Warp kernels the
    compositor invokes per frame.

Both paths import ``rclpy`` / OVRtx lazily so processes that don't
enable either knob never pay the cost.
"""

from engine.newton.visualizers.markers import (
    DeformableMarkerPublisher,
    DeformablePointCloudPublisher,
    ObjectMarkerPublisher,
    shutdown as shutdown_marker_node,
)
from engine.newton.visualizers.ovrtx import InlineOvrtxVisualizer, get_render_stats

__all__ = [
    "DeformableMarkerPublisher",
    "DeformablePointCloudPublisher",
    "ObjectMarkerPublisher",
    "shutdown_marker_node",
    "InlineOvrtxVisualizer",
    "get_render_stats",
]
