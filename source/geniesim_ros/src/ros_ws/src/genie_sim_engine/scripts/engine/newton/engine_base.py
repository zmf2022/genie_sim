# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Direct Newton engine — cloth behaviour matches example_cloth_franka.py.

Franka demo params, in meter scale:
  particle_radius      0.008 m
  contact_margin       0.008 m
  self_contact_radius  0.002 m
  soft_contact_ke      1e4   (VBD stiffness is dimensionless)
  robot_contact_ke     5e4
  density              0.02
  tri_ke / tri_ka      1e4
  sim_substeps         10
  VBD iterations       5

Step structure is identical to the demo:
  for each substep:
    clear forces
    robot step  (particle_count=0, gravity=0, shape_contact_pair_count=0)
    restore particle_count + gravity
    cloth collision + VBD step
    swap states

Performance optimisations (vs. naive uncaptured + USD-Set writeback):
  * 10 substeps captured into a single CUDA graph (~0.4ms/frame total)
  * Kit-free: no Fabric / Hydra render kernels racing on the default
    CUDA stream.

ASSET REQUIREMENTS for cloth USDs:
  * metersPerUnit = 1.0 (raw vertex values are already meters)
  * No parent xformOp:scale or non-trivial xforms — pipeline reads raw
    points directly without applying transforms.
  * Hierarchy must include an Xform parent containing a Mesh child
    (e.g. /Root/shirt with Mesh as a child of /Root). Referencing a Mesh
    as the default prim under a destination Xform doesn't compose cleanly.

If a cloth USD doesn't meet these requirements, bake it into a clean form
(see tools/usd/bake_cloth_meters.py for a one-shot conversion script that
applies the parent xform stack + metersPerUnit into raw vertex data).
"""

from __future__ import annotations

import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp

from engine.base import PhysicsEngine

# Module-level kernels live in ``kernels.py``.
#
# Newton-standalone is Kit-free, so the only writeback kernel is
# ``_kernel_velocity_inject`` (no Fabric path).

from engine.newton.setup import (
    _DebugPubsMixin,
    _InitPoseMixin,
    _ModelMixin,
    _NormalizeMixin,
    _RuntimeMixin,
    _SolverMixin,
    _StageMixin,
)
from engine.newton.cloth import _ClothMixin
from engine.newton.control import _ControlMixin
from engine.newton.plugin import _PluginMixin
from engine.newton.state import _StateMixin
from engine.newton.stats import _StatsMixin
from engine.newton.topology import _TopologyMixin


class _NewtonStandaloneBase(
    # Build phases — fixed order mirrors execution order in ``_build``.
    _RuntimeMixin,  # _build TOC + _warmup + _capture_graph + _dump_runtime_usd
    _StageMixin,  # phase 1: stage open + USD overrides
    _ModelMixin,  # phases 2-3: ModelBuilder + add_usd + finalize
    _NormalizeMixin,  # phase 4: mass clamp + group/articulation unification
    _SolverMixin,  # phases 5-6: states + robot solver + cloth solver
    _DebugPubsMixin,  # phase 7: rclpy marker publishers
    _InitPoseMixin,  # phase 8: name maps + init_joint_pos + state sync
    # Non-build mixins (control, state I/O, etc.).
    _ClothMixin,
    _PluginMixin,
    _TopologyMixin,
    _ControlMixin,
    _StateMixin,
    _StatsMixin,
    PhysicsEngine,
):
    """Shared init, properties, and physics loop for Newton standalone engines.

    Concrete subclasses implement the three template methods:
      _open_stage(newton_scene)  — how to open/populate the USD stage
      _warmup_renders()          — no-op
      _configure_viewport()      — no-op

    Newton-standalone is Kit-free; the only concrete subclass is
    ``NewtonHeadlessEngine``.
    """

    # ---------------------------------------------------------------
    # Solver tuning — match example_cloth_franka.py exactly.
    #
    # franka demo (verbatim):
    #     self.fps          = 50            # ← main control variable
    #     self.sim_substeps = 10            # fixed
    #     self.iterations   = 5             # fixed
    #     self.frame_dt     = 1 / fps
    #     self.sim_dt       = frame_dt / sim_substeps
    #
    # Our engine maps:
    #     physics_hz                  ← franka's ``fps`` (user-controlled)
    #     physics_solver_substep      ← franka's ``sim_substeps`` (default 10)
    #     physics_solver_iterations   ← franka's ``iterations`` (default 5)
    #     sim_dt = (1 / physics_hz) / substeps                  (same formula)
    #
    # Just like franka, the user controls real-time speed via physics_hz.
    # Substeps + iterations stay at franka's tuned values unless explicitly
    # overridden. At physics_hz = 50  → sim_dt = 2.000 ms (franka exact).
    # At physics_hz = 60   → sim_dt = 1.667 ms (franka's other common rate).
    # At physics_hz = 100  → sim_dt = 1.000 ms (smaller, more accurate).
    # ---------------------------------------------------------------
    _DEFAULT_SUBSTEPS = 10  # franka self.sim_substeps
    _DEFAULT_ITERATIONS = 5  # franka self.iterations

    _PARTICLE_RADIUS = 0.008  # 0.8 cm
    _CLOTH_BODY_CONTACT_M = 0.008  # 0.8 cm — cloth↔rigid margin
    _SELF_CONTACT_RADIUS = 0.002  # 0.2 cm
    _SELF_CONTACT_MARGIN = 0.002  # 0.2 cm

    _SOFT_CONTACT_KE = 1e4
    _SOFT_CONTACT_KD = 1e-2
    _SOFT_CONTACT_MU = 0.25  # self_contact_friction in demo

    _ROBOT_KE = 5e4
    _ROBOT_KD = 1e-3
    _ROBOT_KU = 1.5

    _TRI_KE = 1e4
    _TRI_KA = 1e4
    _TRI_KD = 1.5e-6
    _EDGE_KE = 5.0
    _EDGE_KD = 1e-2
    _DENSITY = 0.02

    # ---------------------------------------------------------------

    def __init__(
        self,
        *,
        robot_prefix: str,
        scene_usda: str,
        robot_usda: str,
        render_layer_usda: str,
        physics_hz: float,
        render_hz: float,
        simulation_app: Any,
        logger: Any,
        params: Any,
        robot_from_urdf: bool,
        init_joint_pos: Any,
        runtime_usd_dump_path: str,
        pin_base_to_world: bool = False,
        convert_joints_to_fixed: list | None = None,
        newton_solvers_path: str = "",
        scene_cfg: dict | None = None,
        scene_yaml_path: str = "",
        physics_solver: str = "fsvbd",
        physics_solver_substep: int = 0,  # 0 = auto from physics_hz
        physics_solver_iterations: int = 0,  # 0 = auto from substeps
        physics_solver_mass_matrix_interval: int = 0,  # 0 = engine default (= sim_substeps); pass a large value to effectively disable auto rebuild
        render_mode: str = "raster",
        mujoco_pd_ke: float = 0.0,  # 0 = engine default (50000 N·m/rad)
        mujoco_pd_kd: float = 0.0,  # 0 = engine default (500 N·m·s/rad)
    ) -> None:
        self._robot_prefix_str = robot_prefix
        self._scene_usda = scene_usda
        self._robot_usda = robot_usda
        self._render_layer_usda = render_layer_usda
        self._physics_hz = physics_hz
        self._render_hz = render_hz
        self._simulation_app = simulation_app
        self._logger = logger
        self._params = params
        self._robot_from_urdf = robot_from_urdf
        self._init_joint_pos = init_joint_pos
        self._convert_joints_to_fixed = list(convert_joints_to_fixed or [])
        self._pin_base_to_world = bool(pin_base_to_world)
        # Path for the post-build USD snapshot of the live composed stage
        # (robot.usda + newton_scene.usda + any runtime authoring). Empty
        # disables the dump. Written once in startup() after _warmup so
        # the snapshot reflects exactly what ``add_usd`` parsed plus the
        # cloth-scene sublayer — see ``_dump_runtime_usd``.
        self._runtime_usd_dump_path = runtime_usd_dump_path
        self._newton_solvers_path = newton_solvers_path
        self._scene_cfg = scene_cfg or {}
        self._scene_yaml_path = scene_yaml_path
        # ``_scene_plugin`` is loaded lazily in ``_build`` from the scene
        # yaml's ``newton.scene_plugin`` field; ``None`` until then.
        self._scene_plugin: Any = None
        self._scene_plugin_path: str = ""
        # Sim-time accumulator the loop drives via ``step(dt, …)``; passed
        # to the scene plugin's on_post_step / on_render hooks so they can
        # phase a controller without re-reading the host clock.
        self._sim_time_acc: float = 0.0
        # Render-mode choice — forwarded to ``configure_viewport_for_debug``
        # after _build/_warmup so the live viewport agrees with the
        # bootstrap-time kit args. See ``runtime.bootstrap.RENDER_MODE_MAP``
        # for the canonical mode list.
        self._render_mode = (render_mode or "raster").strip().lower()
        self._physics_solver = physics_solver.strip().lower()
        # Solver-specific substep defaults.
        # mujoco-warp is an implicit integrator that converges in 1 pass per frame.
        # vbd/featherstone (explicit) need 10 substeps for cloth stability.
        if self._physics_solver in ("mujoco-warp", "mujoco_warp"):
            _substep_default = 1
            _iter_default = 0  # let SolverMuJoCo use its own default (100)
        else:
            _substep_default = self._DEFAULT_SUBSTEPS
            _iter_default = self._DEFAULT_ITERATIONS
        self._sim_substeps = int(physics_solver_substep) if int(physics_solver_substep) > 0 else _substep_default
        self._vbd_iterations = int(physics_solver_iterations) if int(physics_solver_iterations) > 0 else _iter_default
        # 0 sentinel → adapter falls back to ``sim_substeps`` (= current
        # behaviour, M rebuilt once per frame).  Any positive value
        # OVERRIDES the default; the user passes e.g. 100000 to
        # effectively cache M forever (suitable for slow-motion cloth
        # folding where joint velocities stay well below 1 rad/s and
        # M(q) drift is negligible across many frames).
        self._mass_matrix_interval = int(physics_solver_mass_matrix_interval)
        self._mujoco_pd_ke = float(mujoco_pd_ke)
        self._mujoco_pd_kd = float(mujoco_pd_kd)

        # Runtime state
        # Inline OVRtx visualizer (physics_engine_visualizer:=ovrtx) attaches
        # a wp.Event here.  When set, ``tick_extras`` records it on Warp's
        # current stream after each physics step so the OVRtx render thread
        # can GPU-wait on the latest physics commit without CPU sync.
        # See docs/ovrtx_sync.md.
        self._physics_step_event: Any = None
        self._stage: Any = None
        # Diagnostic: when GENIESIM_STEP_GPU_SYNC is truthy, ``step()``
        # blocks on a ``wp.synchronize_device`` after launching the
        # captured graph and emits a 1Hz log of the GPU completion time.
        # OFF by default — turning it on forces every frame to wait for
        # the full cloth+robot graph (~22 ms on the fr3+tshirt scene),
        # which strips the parallel overlap with the Hydra render and
        # drops render to 0 Hz. Useful for one-shot profiling to settle
        # whether physics is GPU-bound; keep off in normal use.
        self._step_gpu_sync = os.environ.get("GENIESIM_STEP_GPU_SYNC", "").strip().lower() in ("1", "true", "yes", "on")
        self._model: Any = None
        self._state_0: Any = None
        self._state_1: Any = None
        self._control: Any = None
        self._robot_solver: Any = None
        # Solver-specific behavior lives in a SolverAdapter.  The adapter
        # owns the position-target buffer (mjwarp -> control.joint_target_pos,
        # featherstone -> dedicated wp.array) and the per-substep robot
        # step.  apply_commands routes /joint_command writes through
        # self._adapter.target_buffer().
        from engine.newton.adapters import make_adapter

        # When ``runtime_usd_dump_path`` is set, default the MJCF dump
        # to a sibling file ``robot_runtime.xml`` in the same scene
        # directory.  Operator gets a matched pair of post-load
        # snapshots — USD on the Newton side, MJCF on the mjwarp side —
        # so the two model representations can be inspected and diffed.
        mjcf_dump = ""
        if self._runtime_usd_dump_path:
            mjcf_dump = os.path.splitext(self._runtime_usd_dump_path)[0] + ".xml"
        self._adapter = make_adapter(
            self._physics_solver,
            mujoco_pd_ke=self._mujoco_pd_ke,
            mujoco_pd_kd=self._mujoco_pd_kd,
            mujoco_save_to_mjcf=mjcf_dump,
            physics_params=self._params,
            physics_hz=self._physics_hz,
        )
        self._cloth_solver: Any = None
        self._contacts: Any = None
        self._gravity_zero: Any = None
        self._gravity_earth: Any = None
        # ``newton.debug.pub_deformable_marker`` — debug-only sibling
        # rclpy publisher (visualization_msgs/Marker TRIANGLE_LIST of the
        # cloth / FEM surface, particles indexed by tri_indices).
        # Lazily constructed in _build when the flag is true.  ``None``
        # otherwise; tick_extras skips the publish path when None.
        self._deformable_pub: Any = None
        # ``newton.debug.pub_deformable_pointcloud`` — debug-only sibling
        # rclpy publisher (sensor_msgs/PointCloud2 of every particle in
        # state.particle_q).  Same on/off semantics as the marker pub
        # above; the two are independent so an operator can run either,
        # both, or neither.
        self._deformable_pc_pub: Any = None
        # ``newton.debug.pub_object_marker`` — debug-only sibling rclpy
        # publisher (visualization_msgs/MarkerArray of every free-joint
        # body's pose).  Same on/off semantics.
        self._object_pub: Any = None
        # Sim-time of next debug-pub fire.  Both debug publishers share
        # this gate so they fire at ``render_hz`` (not physics_hz × subs).
        # See ``tick_extras`` for the advance step.
        self._next_debug_pub_time: float = 0.0
        self._joint_names: List[str] = []
        self._joint_prim_map: Dict[str, str] = {}
        # ``_joint_name_to_dof`` is the qd (velocity / DOF vector) index
        # — the right thing to use for ``control.joint_target_pos``,
        # ``control.joint_target_vel``, ``state.joint_qd``, and the
        # per-DOF controlled-mask in the featherstone / AVBD adapters.
        # ``_joint_name_to_q_idx`` is the q (configuration vector) index,
        # needed when reading ``state.joint_q`` for the joint position.
        # The two coincide for revolute / prismatic / fixed joints
        # (q_count == dof_count) and diverge by +1 for every preceding
        # FREE joint (q_count=7, dof_count=6 — quaternion vs angular
        # velocity).  Conflating them makes every commanded joint move
        # its neighbour once ``pin_base_to_world: false`` puts a FREE
        # joint at the head of the articulation, so the two maps are
        # kept strictly distinct.
        #
        # The two name maps are populated from a single ``JointIndex``
        # snapshot in ``_build_joint_map``.  ``self._jindex`` keeps the
        # full snapshot around so downstream consumers (adapters,
        # state.py, control.py) can reach richer
        # per-joint metadata — joint type, q/qd slice widths, the
        # ``copy_q_to_qd`` helper — without re-walking the Newton
        # ``model.joint_*`` arrays.  Stays ``None`` until
        # ``_build_joint_map`` runs (i.e. until ``add_usd`` has
        # populated those arrays); callers that pre-empt that order
        # should guard with ``if self._jindex is not None``.
        self._joint_name_to_dof: Dict[str, int] = {}
        self._joint_name_to_q_idx: Dict[str, int] = {}
        from engine.newton.joint_index import JointIndex  # noqa: PLC0415

        self._jindex: Optional[JointIndex] = None
        self._body_paths: List[str] = []

        # Joints that fix_base / fix_head / fix_body collapsed to FixedJoint
        # at runtime.  Featherstone has 0 DOFs for them; Newton's
        # ``model.joint_q`` doesn't carry their values.  But the URDF still
        # describes them as revolute, so robot_state_publisher / RViz /
        # downstream consumers need to see them in ``/joint_states`` to
        # build a consistent kinematic-chain TF tree.  This dict is the
        # single source of truth for "what value to publish" for those
        # joints — populated once at lifecycle build time, queried every
        # tick by ``state.get_joint_states``.  Empty when no fix_* flag
        # is on (then everything goes through the normal joint_q path).
        self._static_joint_q: Dict[str, float] = {}

        # CUDA graph
        self._cuda_graph: Any = None
        self._sim_dt_captured: float = 0.0

        # Pinned-host mirrors of state_0.joint_q / joint_qd / body_q.
        # ``AsyncMirror`` runs an async device→host wp.copy on the
        # captured graph's stream after every step (engine.step below);
        # ``get_joint_states`` / ``get_body_transforms`` / ``get_odom``
        # read the resulting host buffer with no GPU contact.  The
        # mirror lags by ONE physics tick — see async_mirror.py for the
        # full ping-pong design.  Trade: 10 ms of /joint_states lag at
        # 100 Hz vs ~6 ms of CPU stall every tick on the synchronous
        # path.
        from engine.newton.async_mirror import AsyncMirror  # noqa: PLC0415

        self._jq_mirror = AsyncMirror()
        self._jqd_mirror = AsyncMirror()
        self._bq_mirror = AsyncMirror()

        # Stats
        self._tick_count = 0
        self._step_ms_acc = 0.0
        self._step_ms_max = 0.0
        self._writeback_ms_acc = 0.0
        self._writeback_ms_max = 0.0
        self._render_ms_acc = 0.0
        self._render_ms_max = 0.0
        self._render_count = 0
        self._render_target_hz = 30.0  # set by run loop via note_render_target
        self._t_log_start = time.monotonic()
        self._t_loop_start = time.monotonic()
        self._frame_id = 0
        self._LOG_EVERY = int(physics_hz * 5)
        # Per-phase timings (run loop pushes via note_phase_timing)
        self._phase_step_ms_acc = 0.0
        self._phase_extras_ms_acc = 0.0
        self._phase_render_ms_acc = 0.0
        self._phase_render_sync_ms_acc = 0.0
        self._phase_render_sync_max = 0.0
        # Publish-phase timings (run loop pushes via note_publish_phase)
        self._publish_clock_ms_acc = 0.0
        self._publish_joints_ms_acc = 0.0
        self._publish_bodies_ms_acc = 0.0
        self._publish_odom_ms_acc = 0.0

        self._build()
        self._warmup()
        # One-shot USD snapshot of the live composed stage AFTER add_usd
        # parsing + cloth-scene sublayer resolve.
        self._dump_runtime_usd()
        self._capture_graph()

    # ------------------------------------------------------------------
    # PhysicsEngine interface
    # ------------------------------------------------------------------

    @property
    def stage(self) -> Any:
        return self._stage

    @property
    def robot_prefix(self) -> str:
        return self._robot_prefix_str

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def joint_prim_map(self) -> Dict[str, str]:
        return self._joint_prim_map

    @property
    def body_paths(self) -> List[str]:
        return self._body_paths

    def startup(self, headless: bool) -> None:
        # Match franka: fps → frame_dt → sim_dt = frame_dt / sim_substeps.
        frame_dt_ms = 1000.0 / self._physics_hz
        sim_dt_ms = frame_dt_ms / self._sim_substeps
        total_iters = self._sim_substeps * self._vbd_iterations
        self._logger.info(
            f"[newton-standalone] ready — physics_hz={self._physics_hz:.0f}  "
            f"frame_dt={frame_dt_ms:.3f}ms  (1× real time)"
        )
        self._logger.info(
            f"[newton-standalone]   sim_substeps={self._sim_substeps} (franka default 10)  "
            f"iterations={self._vbd_iterations} (franka default 5)"
        )
        self._logger.info(
            f"[newton-standalone]   sim_dt={sim_dt_ms:.3f}ms  "
            f"total_iters/frame={total_iters}  "
            f"(= sim_substeps × iterations, franka-style)"
        )
        self._logger.info(
            f"[newton-standalone] step GPU sync diagnostic: "
            f"{'ENABLED' if self._step_gpu_sync else 'disabled'} "
            f"(set GENIESIM_STEP_GPU_SYNC=0 to turn off)"
        )

    def step(self, dt: float, step_start: float) -> float:
        """One physics frame (10 substeps). Uses CUDA graph if captured.

        With ``GENIESIM_STEP_GPU_SYNC=1`` set, also issues a
        ``wp.synchronize_device("cuda:0")`` after the launch and reports
        the GPU completion time via the 1Hz log. This is a DIAGNOSTIC
        knob — it stalls the Python thread until every queued kernel
        finishes, so don't leave it on for normal runs (it strips the
        async overlap that lets render and physics share the GPU). Use
        it to settle whether ``get_joint_states`` sync time is bound by
        the captured graph itself or by something queued outside it
        (e.g. cloth post-step BVH rebuild, ROS publish-side numpy() syncs).
        """
        t0 = time.monotonic()

        if self._cloth_solver is not None and hasattr(self._cloth_solver, "rebuild_bvh"):
            # BVH must be rebuilt each frame; can't be inside the captured
            # graph. Only ``SolverVBD`` exposes ``rebuild_bvh`` — XPBD and
            # Style3D maintain their own broad-phase internally and the
            # method isn't part of their API. ``hasattr`` guard keeps the
            # call VBD-only without branching on the solver class name.
            self._cloth_solver.rebuild_bvh(self._state_0)

        if self._cuda_graph is not None:
            # Single GPU launch for all 10 substeps — eliminates 10× kernel-
            # launch overhead and Python loop costs. Rebuild_bvh is the only
            # required outside-graph step.
            wp.capture_launch(self._cuda_graph)
        else:
            # Fallback: uncaptured Python loop (used during warmup, or if
            # capture failed). State swap is Python-level here.
            self._simulate_substeps(dt, captured=False)

        # Scene-plugin per-frame callback.  Runs OUTSIDE the captured CUDA
        # graph (post the substep loop) so it can do host-side work like
        # state-machine updates and writing the next frame's kinematic
        # body poses.  See plugin.py for the contract.
        self._sim_time_acc += dt
        self._call_plugin(
            "on_post_step",
            self._state_0,
            self._sim_time_acc,
            dt,
            self._plugin_ctx(),
        )

        # Async device→host mirror refresh — see ``async_mirror.py``
        # for the ping-pong design.  Each mirror enqueues a wp.copy
        # onto the source-device's current stream (the same one the
        # captured graph just ran on) so the memcpy is FIFO-ordered
        # behind the kernel writes; the host returns immediately.
        # Readers in state.py see the OTHER slot, which the previous
        # tick wrote and which has long since completed.
        if self._state_0 is not None:
            jq_src = getattr(self._state_0, "joint_q", None)
            jqd_src = getattr(self._state_0, "joint_qd", None)
            bq_src = getattr(self._state_0, "body_q", None)
            primary = jq_src or jqd_src or bq_src
            if primary is not None:
                stream = wp.get_stream(primary.device)
                if jq_src is not None:
                    self._jq_mirror.refresh(jq_src, stream)
                if jqd_src is not None:
                    self._jqd_mirror.refresh(jqd_src, stream)
                if bq_src is not None:
                    self._bq_mirror.refresh(bq_src, stream)

        # Optional GPU completion timing (diagnostic; default off).
        if getattr(self, "_step_gpu_sync", False):
            t_sync0 = time.monotonic()
            try:
                wp.synchronize_device("cuda:0")
            except Exception:
                pass
            t_sync = (time.monotonic() - t_sync0) * 1000.0
            if not hasattr(self, "_step_gpu_sync_t0"):
                self._step_gpu_sync_t0 = time.monotonic()
                self._step_gpu_sync_acc = 0.0
                self._step_gpu_sync_max = 0.0
                self._step_gpu_sync_n = 0
            self._step_gpu_sync_acc += t_sync
            self._step_gpu_sync_max = max(self._step_gpu_sync_max, t_sync)
            self._step_gpu_sync_n += 1
            if (time.monotonic() - self._step_gpu_sync_t0) >= 1.0 and self._step_gpu_sync_n > 0:
                avg = self._step_gpu_sync_acc / self._step_gpu_sync_n
                self._logger.info(
                    f"[newton-standalone] step GPU sync (1Hz, {self._step_gpu_sync_n} calls): "
                    f"avg={avg:.3f}ms  max={self._step_gpu_sync_max:.3f}ms  "
                    f"(time for the captured cloth+robot graph to finish on the GPU)"
                )
                self._step_gpu_sync_t0 = time.monotonic()
                self._step_gpu_sync_acc = 0.0
                self._step_gpu_sync_max = 0.0
                self._step_gpu_sync_n = 0

        total_ms = (time.monotonic() - t0) * 1000.0
        self._step_ms_acc += total_ms
        if total_ms > self._step_ms_max:
            self._step_ms_max = total_ms
        self._frame_id += 1
        return total_ms

    def _simulate_substeps(self, dt: float = 0.0, captured: bool = True) -> None:
        """Inner substep body. Called either inside CUDA graph capture
        (``captured=True`` — uses ``.assign()`` to swap states, no Python ops)
        or uncaptured (``captured=False`` — uses Python state swap).

        When captured, ``dt`` is taken from ``self._sim_dt_captured`` set at
        capture time (graph baked-in constant).

        Per-substep work is delegated to ``self._substep_body``, a method
        reference bound once in ``_build`` based on the active scene
        composition.  Each branch (plain rigid, franka-VBD-cloth, etc.)
        owns its own model-state mutations so the orchestrator here
        stays minimal and a new solver mode just adds another
        ``_substep_body_*`` method without touching this loop.
        """
        sub_dt = self._sim_dt_captured if captured else (dt / self._sim_substeps)

        for _ in range(self._sim_substeps):
            # Clear BOTH state buffers — the franka cloth demo does this
            # and it matters because state_0 ↔ state_1 swap each substep.
            # After the swap, the new state_0 (= old state_1) carries
            # whatever forces the previous substep wrote; without
            # clearing we leak per-substep stale impulses into the next
            # integration.  Universal across substep bodies.
            self._state_0.clear_forces()
            self._state_1.clear_forces()

            self._substep_body(sub_dt)

            # Swap states. Inside CUDA graph capture we can't do a Python swap;
            # use .assign() (a GPU memcpy) so state_0 holds the latest result.
            if captured:
                # CRITICAL: copy the FULL state forward, not just body/particle.
                # Featherstone writes new joint_q / joint_qd to state_1 each
                # substep (see SolverFeatherstone — its kernels' outputs include
                # state_out.joint_q / state_out.joint_qd). Without rolling
                # those forward, every substep re-reads the same initial
                # joint_q and the captured 10-substep frame effectively does
                # only ONE step of motion (and even that gets thrown away
                # next frame because state_0.joint_q is still the init pose).
                # The user-visible symptom is "robot is frozen" / "commands
                # have no effect" even though apply_commands wrote the target.
                if self._state_0.joint_q is not None and self._state_1.joint_q is not None:
                    self._state_0.joint_q.assign(self._state_1.joint_q)
                if self._state_0.joint_qd is not None and self._state_1.joint_qd is not None:
                    self._state_0.joint_qd.assign(self._state_1.joint_qd)
                if self._state_0.particle_q is not None:
                    self._state_0.particle_q.assign(self._state_1.particle_q)
                    self._state_0.particle_qd.assign(self._state_1.particle_qd)
                if self._state_0.body_q is not None:
                    self._state_0.body_q.assign(self._state_1.body_q)
                    self._state_0.body_qd.assign(self._state_1.body_qd)
            else:
                self._state_0, self._state_1 = self._state_1, self._state_0

    # ------------------------------------------------------------------
    # Per-substep bodies — one branch per solver composition.
    #
    # Selected once in ``_build`` and stored as ``self._substep_body``;
    # the captured CUDA graph bakes whichever branch was active at
    # capture time.  Add a new solver here by writing a new
    # ``_substep_body_<name>`` method and extending ``_pick_substep_body``
    # — keeps assumptions about gravity / contacts / cloth interleave
    # local to each branch instead of compounding in a shared loop.
    # ------------------------------------------------------------------

    def _substep_body_kinematic_control(self, sub_dt) -> None:
        """One rigid-solver substep in *kinematic-control mode*:
        ``particle_count=0``, ``gravity=0``, ``shape_contact_pair_count=0``.

        Default for cloth-free scenes.  The rigid solver tracks
        JOINT_TARGET / velocity-injection set points cleanly without
        fighting an inertial / contact load — no body droop under
        gravity, no wheel-vs-floor contact bouncing.  IsaacSim's
        URDF→USD converter always emits ``PhysicsFixedJoint
        "root_joint"`` between ``/robot`` and ``base_link``, so the
        base is welded to world by the importer; zero gravity inside
        the rigid step is therefore safe (no drift possible).
        """
        pc = self._model.particle_count
        self._model.particle_count = 0
        self._model.gravity.assign(self._gravity_zero)
        self._model.shape_contact_pair_count = 0
        self._adapter.substep(self._model, self._state_0, self._state_1, self._control, sub_dt)
        if getattr(self._state_0, "particle_f", None) is not None:
            self._state_0.particle_f.zero_()
        self._model.particle_count = pc
        self._model.gravity.assign(self._gravity_earth)

    def _substep_body_franka_vbd_cloth(self, sub_dt) -> None:
        """One rigid-solver substep with REAL gravity and shape contacts,
        followed by ``model.collide`` + VBD cloth ``step``.

        Only ``particle_count`` is zeroed (the rigid step ignores
        cloth particles; cloth gets its own step below).  Gravity and
        rigid contacts run normally — Featherstone handles dynamic
        bodies the standard way.  The URDF importer welds
        ``base_link`` to world via ``PhysicsFixedJoint 'root_joint'``
        (``pin_base_to_world=True``), so the robot doesn't drift under
        gravity regardless.  Passive dynamic bodies in the scene
        (hanger, dropped objects) get gravity through the standard
        path.
        """
        pc = self._model.particle_count
        self._model.particle_count = 0
        self._adapter.substep(self._model, self._state_0, self._state_1, self._control, sub_dt)
        if getattr(self._state_0, "particle_f", None) is not None:
            self._state_0.particle_f.zero_()
        self._model.particle_count = pc

        # Cloth step (VBD + collision) — uses real gravity, the cloth
        # solver consults its own particle contacts.
        # NOTE: ``model.collide(state_0, ...)`` intentionally uses the
        # PRE-rigid-step state. With ``integrate_with_external_rigid_solver=
        # True`` VBD reads ``body_q_prev_for_particles = state_in.body_q``
        # (= state_0) and ``body_q_for_particles = state_out.body_q``
        # (= state_1) to infer rigid body velocity for cloth-rigid friction.
        # Forwarding state_1.body_q into state_0.body_q before collide
        # zeroes that inferred velocity and breaks cloth-rigid coupling on
        # any moving body (hanger drop, gripper close, etc.). Substep dt
        # is 1.67 ms — body displacement per substep is small enough that
        # contact normals computed against state_0 are still valid for the
        # cloth penalty force evaluated against state_1 inside VBD.
        self._model.collide(self._state_0, self._contacts)
        # SolverXPBD has no ``integrate_with_external_rigid_solver`` flag
        # (only SolverVBD / SolverStyle3D do).  When XPBD's step() runs
        # against a model with body_count > 0 it re-integrates rigid
        # bodies AND re-projects rigid joint / contact constraints — on
        # the MJW combo path this fights the implicit solve MJW just
        # finished, and the welded ``[base, head, body]`` chassis chain
        # accumulates drift each substep until the wheels detach.
        # Snapshot body state before XPBD's step and restore after; cloth
        # particles still see the pre-XPBD body_q via ``model.collide``
        # above (computed against state_0), so particle-vs-rigid contact
        # is unaffected.  No-op for VBD / Style3D — they don't touch
        # body_q in external-rigid mode.
        _xpbd_freeze = getattr(self, "_cloth_solver_pref", "") == "xpbd" and self._state_1.body_q is not None
        if _xpbd_freeze:
            _saved_body_q = wp.clone(self._state_1.body_q)
            _saved_body_qd = wp.clone(self._state_1.body_qd) if self._state_1.body_qd is not None else None
        self._cloth_solver.step(self._state_0, self._state_1, self._control, self._contacts, sub_dt)
        if _xpbd_freeze:
            self._state_1.body_q.assign(_saved_body_q)
            if _saved_body_qd is not None:
                self._state_1.body_qd.assign(_saved_body_qd)

    def _pick_substep_body(self):
        """Bind ``self._substep_body`` based on the active solver
        composition.  Called from ``_build`` after the cloth solver and
        adapter have been constructed; the choice is frozen for the
        lifetime of the engine (and the captured CUDA graph).

        Selection matrix (cloth present, pin_base_to_world):
          * cloth, *           → ``_substep_body_franka_vbd_cloth``
            (kinematic-control rigid + VBD cloth)
          * no cloth, true     → ``_substep_body_kinematic_control``
            (rigid step with gravity=0, contacts=0; URDF root_joint
            FixedJoint anchors the base — stationary humanoid / arm
            bench)
          * no cloth, false    → ``_substep_body_plain``
            (real gravity + shape contacts; root_joint has been
            deactivated so the base is free under physics — mobile
            robot, contact tuning becomes operator's responsibility)
        """
        cloth = self._cloth_solver is not None
        pinned = getattr(self, "_pin_base_to_world", False)
        adapter_name = getattr(self._adapter, "name", "")

        # AVBD path: SolverVBD integrates rigid + cloth in one step.
        # No separate cloth_solver, no model.collide between rigid and
        # cloth — adapter.substep does everything inside its solver.step.
        # (lifecycle.py sets self._cloth_solver = None on this path so
        # the ``cloth`` flag above is False; this branch fires before
        # the cloth/pinned/plain matrix.)
        if adapter_name == "avbd":
            self._substep_body = self._substep_body_avbd_unified
            self._logger.info(
                "[newton-standalone] substep body: avbd-unified "
                "(SolverVBD integrates rigid via AVBD + cloth in one step; "
                "no separate cloth solver, no inter-solver coupling)"
            )
        elif cloth:
            self._substep_body = self._substep_body_franka_vbd_cloth
            self._logger.info(
                "[newton-standalone] substep body: franka-VBD-cloth "
                "(rigid step under real gravity + shape contacts; cloth step "
                "under real gravity + particle contacts)"
            )
        elif pinned:
            self._substep_body = self._substep_body_kinematic_control
            self._logger.info(
                "[newton-standalone] substep body: kinematic-control "
                "(rigid step with gravity=0, shape_contacts=0; URDF root_joint "
                "FixedJoint anchors the base.)"
            )
        else:
            self._substep_body = self._substep_body_plain
            self._logger.info(
                "[newton-standalone] substep body: plain "
                "(real gravity + shape contacts; root_joint deactivated, "
                "base is mobile under physics — tune geom_solref/solimp on "
                "collision shapes if bouncing instability appears)"
            )

    def _substep_body_plain(self, sub_dt) -> None:
        """One rigid-solver substep under real gravity + shape contacts.
        Used for mobile-robot scenes (``pin_base_to_world: false``) where
        floor contact through the wheels keeps the chassis anchored.
        Pair with contact tuning (geom_solref/solimp) on collision
        shapes for the robot's mass distribution.
        """
        self._adapter.substep(self._model, self._state_0, self._state_1, self._control, sub_dt)

    def _substep_body_avbd_unified(self, sub_dt) -> None:
        """One unified AVBD step — rigid + cloth advance together.

        The AVBD adapter constructed a single ``SolverVBD`` instance with
        ``integrate_with_external_rigid_solver=False``, so its
        ``solver.step`` integrates rigid bodies (via Augmented VBD) and
        cloth particles (via VBD) in one pass.  No ``model.collide`` /
        ``cloth_solver.step`` plumbing is needed — that whole inter-solver
        coupling layer (which the Featherstone+VBD path uses) is replaced
        by the unified solver's internal contact pipeline.

        ``particle_count`` and ``shape_contact_pair_count`` stay at their
        finalize-time values; the solver's contact pass handles both
        cloth-vs-rigid and rigid-vs-rigid.
        """
        self._adapter.substep(self._model, self._state_0, self._state_1, self._control, sub_dt)

    def tick_extras(self) -> None:
        """Per-tick work that ALWAYS fires (every physics step).

        Kit-free newton-standalone has no Fabric writeback to drive.
        Kept as a stats heartbeat: increment the tick counter and emit
        the 1 Hz stats log.

        When the inline OVRtx visualizer is attached
        (``physics_engine_visualizer:=ovrtx``), this also records the
        per-tick CUDA event the OVRtx render thread waits on — see
        ``docs/ovrtx_sync.md``.
        """
        if self._model is None:
            return
        tw0 = time.monotonic()

        # Inline OVRtx handshake: cheap (~µs).  Records on Warp's current
        # stream — the same stream the Newton CUDA graph just launched on,
        # so the OVRtx thread's stream-side wait sees a committed body_q.
        if self._physics_step_event is not None:
            wp.record_event(self._physics_step_event)

        wb_ms = (time.monotonic() - tw0) * 1000.0
        self._writeback_ms_acc += wb_ms
        if wb_ms > self._writeback_ms_max:
            self._writeback_ms_max = wb_ms

        self._tick_count += 1
        # 1Hz wall-clock stats log
        if (time.monotonic() - self._t_log_start) >= 1.0:
            self._print_stats()

        # Debug-only publishers run at ``render_hz``, not physics_hz ×
        # substeps — RViz can't render any faster than render_hz anyway,
        # and publishing every physics tick wastes wire bandwidth + GIL
        # time inside the hot path.  Shared gate so both pubs land on the
        # same beat; advances in sim-time so it's not skewed by realtime
        # factor or wall-clock jitter.  ``render_hz <= 0`` (headless) =>
        # publish every tick (cheap fallback for the rare case where a
        # user wants raw cadence).
        _hz = float(self._render_hz)
        _due = _hz <= 0.0 or self._sim_time_acc >= self._next_debug_pub_time
        if _due and (
            self._deformable_pub is not None or self._deformable_pc_pub is not None or self._object_pub is not None
        ):
            # Debug-only: publish deformable / cloth surface as a
            # TRIANGLE_LIST Marker.  Gated by
            # ``newton.debug.pub_deformable_marker`` (default off); the
            # ``is None`` branch costs one attribute lookup per tick in
            # production.
            if self._deformable_pub is not None:
                try:
                    pq = getattr(self._state_0, "particle_q", None)
                    if pq is not None:
                        self._deformable_pub.publish(pq, self._sim_time_acc)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warn(f"[newton-standalone] deformable marker publish failed: {exc!r}")
                    self._deformable_pub = None  # disarm on first failure to avoid log spam

            # Debug-only: publish raw particle_q as a sensor_msgs/PointCloud2.
            # Gated by ``newton.debug.pub_deformable_pointcloud``; same
            # disarm-on-failure pattern as the marker pub above.
            if self._deformable_pc_pub is not None:
                try:
                    pq = getattr(self._state_0, "particle_q", None)
                    if pq is not None:
                        self._deformable_pc_pub.publish(pq, self._sim_time_acc)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warn(f"[newton-standalone] deformable pointcloud publish failed: {exc!r}")
                    self._deformable_pc_pub = None

            # Debug-only: publish free-joint rigid object markers.  Same
            # gating and disarm-on-failure pattern as the deformable pub.
            if self._object_pub is not None:
                try:
                    bq = getattr(self._state_0, "body_q", None)
                    if bq is not None:
                        self._object_pub.publish(bq, self._sim_time_acc)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warn(f"[newton-standalone] object marker publish failed: {exc!r}")
                    self._object_pub = None

            if _hz > 0.0:
                # Advance by a fixed period; tolerate large physics-side
                # gaps (e.g. after a long pause) by snapping forward
                # rather than letting the gate flood once it catches up.
                period = 1.0 / _hz
                self._next_debug_pub_time = max(self._next_debug_pub_time + period, self._sim_time_acc + period * 0.5)

    def attach_physics_event(self, event: Any) -> None:
        """Attach a ``wp.Event`` recorded after every physics step.

        Used by the inline OVRtx visualizer so its render thread can
        ``wp.Stream.wait_event(event)`` to gate the per-frame sync kernel
        on the latest physics commit, with no CPU-side wait on the physics
        thread.  Idempotent — pass ``None`` to detach.
        """
        self._physics_step_event = event

    def sync_visual_state(self) -> None:
        """Explicit force-sync hook.  No-op on Kit-free
        newton-standalone — there is no Fabric or USD-side mirror to
        push state into; the Newton GL viewer reads ``state_0`` GPU
        arrays directly via ``viewer.log_state()`` each render frame.
        Kept for ``PhysicsEngine`` interface compatibility.
        """
        return

    def shutdown(self) -> None:
        self._model = None
        self._stage = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
