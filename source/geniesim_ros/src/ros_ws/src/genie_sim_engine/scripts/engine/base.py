#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Abstract physics engine interface + factory.

All concrete engines implement ``PhysicsEngine``.  The run loop in
``genie_sim_engine.py`` depends only on this interface — no engine-specific
conditionals in the loop itself.

Supported engines (``physics_engine`` selector):

    ``isaac_physx``   → :class:`IsaacPhysXEngine`
    ``isaac_newton``  → :class:`IsaacNewtonEngine`
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PhysicsEngine(ABC):
    """Minimal interface the physics step loop depends on.

    Lifecycle::

        engine = PhysicsEngine.create(physics_engine, ...)
        engine.startup(headless)
        while running:
            engine.step(dt, step_start)          # simulate + fetch
            engine.tick_extras(usd_update_fn)    # e.g. cloth writeback
        engine.shutdown()
    """

    # ------------------------------------------------------------------
    # Properties the run loop reads every tick
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def stage(self) -> Any:
        """Live USD stage."""

    @property
    @abstractmethod
    def robot_prefix(self) -> str:
        """USD prim prefix for the robot root (e.g. ``"ur5"``)."""

    @property
    @abstractmethod
    def joint_names(self) -> List[str]:
        """Ordered list of joint DOF names used for state publish."""

    @property
    @abstractmethod
    def joint_prim_map(self) -> Dict[str, str]:
        """Joint name → USD prim path mapping."""

    @property
    @abstractmethod
    def body_paths(self) -> List[str]:
        """USD prim paths for all rigid bodies (for tf_render)."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def startup(self, headless: bool) -> None:
        """One-time setup just before entering the physics loop."""

    @abstractmethod
    def step(self, dt: float, step_start: float) -> float:
        """Run one physics tick.

        Returns the wall-clock seconds spent in the physics solve.
        Implementations should NOT call ``simulation_app.update()``
        here — the caller handles the render budget.
        """

    @abstractmethod
    def tick_extras(self) -> None:
        """Per-tick work that ALWAYS fires (every physics step).

        For engines that publish state to a parallel renderer (Newton-direct
        with C++ GIL-released render thread), this is where Fabric writeback
        happens — every tick, so the renderer always sees fresh data when
        it polls Fabric on its own cadence.

        For ``isaac_physx``: no-op (PhysX's own Fabric extension handles
        writeback automatically).
        For ``isaac_newton``: same — the wrapper extension does it.
        For ``newton``: cloth points + body matrices land on Fabric every tick.
        """

    def sync_visual_state(self) -> None:
        """Optional explicit force-sync hook.

        Default no-op. Engines that defer Fabric writeback to render-time
        (single-threaded inline-render use cases) can override this and the
        run loop will call it just before ``simulation_app.update()``. For
        engines that write every tick from ``tick_extras`` (the
        parallel-renderer case), this is a no-op since the data is already
        fresh.
        """
        return

    def note_render(self, render_ms: float, did_render: bool) -> None:
        """Called by the run loop after each potential render tick.

        Default no-op. Override to gate expensive per-render work (e.g.
        SelectPrims rebuild for FabricSceneDelegate) to render cadence rather
        than physics cadence.  ``did_render=True`` means ``simulation_app.update()``
        fired this tick.
        """
        return

    def note_render_target(self, render_hz: float) -> None:
        """Tell the engine the run loop's target render Hz. Default no-op."""
        return

    def note_phase_timing(
        self,
        step_ms: float = 0.0,
        extras_ms: float = 0.0,
        render_ms: float = 0.0,
        render_sync_ms: float = 0.0,
        did_render: bool = False,
    ) -> None:
        """Per-tick phase timings from the run loop. Default no-op."""
        return

    def note_publish_phase(
        self,
        clock_ms: float = 0.0,
        joints_ms: float = 0.0,
        bodies_ms: float = 0.0,
        odom_ms: float = 0.0,
    ) -> None:
        """Publish-phase timing breakdown from the run loop. Default no-op."""
        return

    def get_joint_states(self):
        """Return (positions, velocities) for joint_names.

        Default: reads from the live USD stage via snapshot_joint_states.
        Override for engines that keep state in non-USD buffers (e.g. Newton).
        """
        from kit.stage import snapshot_joint_states

        return snapshot_joint_states(self.stage, self.joint_names, self.joint_prim_map)

    def get_body_transforms(self):
        """Return (Nx7 array, frame_paths). Default: USD-stage snapshot."""
        from kit.stage import snapshot_body_transforms

        return snapshot_body_transforms(self.stage, self.body_paths)

    def get_odom(self, sim_time: float):
        """Return (pose7, twist6) for the base_link, or None. Default: USD-stage snapshot."""
        from kit.stage import snapshot_odom

        return snapshot_odom(self.stage, self.robot_prefix, sim_time)

    @abstractmethod
    def apply_commands(
        self,
        cmd_positions: Dict[str, float],
        cmd_4ws_steer_pos: Any,
        cmd_4ws_drive_vel: Any,
        cmd_4ws_stamp: Any,
    ) -> None:
        """Forward joint/drive commands to the active articulation."""

    @abstractmethod
    def shutdown(self) -> None:
        """Clean up resources."""

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def create(
        physics_engine: str,
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
        physics_solver: str = "mujoco-warp",
        physics_solver_substep: int = 0,
        physics_solver_iterations: int = 0,
        physics_solver_mass_matrix_interval: int = 0,
        render_mode: str = "raster",
        mujoco_pd_ke: float = 0.0,
        mujoco_pd_kd: float = 0.0,
    ) -> "PhysicsEngine":
        """Instantiate the engine for ``physics_engine``.

        ``physics_engine`` must already be normalized (``isaac_physx``,
        ``isaac_newton``, or ``newton``) — call
        ``runtime.bootstrap._validate_engine_id`` before this factory.

        Two scene-yaml runtime flags reach every engine:

          * ``pin_base_to_world`` — applies to ALL engines.  Kit
            engines toggle the world-weld FixedJoint in
            ``kit/stage.py:_apply_fix_base_policy``; newton-standalone
            deactivates the URDF root_joint in
            ``engine/newton/setup/stage.py:_deactivate_root_joint``.
          * ``convert_joints_to_fixed`` — newton-standalone only.
            Kit engines silently ignore this list since they manage
            articulation via PhysX schemas, not session-layer
            FixedJoint replacement.

        These two flags are INDEPENDENT — neither implies anything
        about the other.
        """
        kwargs = dict(
            robot_prefix=robot_prefix,
            scene_usda=scene_usda,
            robot_usda=robot_usda,
            render_layer_usda=render_layer_usda,
            physics_hz=physics_hz,
            render_hz=render_hz,
            simulation_app=simulation_app,
            logger=logger,
            params=params,
            robot_from_urdf=robot_from_urdf,
            init_joint_pos=init_joint_pos,
            runtime_usd_dump_path=runtime_usd_dump_path,
            pin_base_to_world=pin_base_to_world,
            newton_solvers_path=newton_solvers_path,
            scene_cfg=scene_cfg or {},
            scene_yaml_path=scene_yaml_path,
            physics_solver=physics_solver,
        )
        if physics_engine == "newton":
            # Newton-standalone is Kit-free; ``NewtonHeadlessEngine`` is
            # the only concrete class.  ``simulation_app`` is still
            # passed in via ``kwargs`` for ABC compatibility but the
            # headless engine ignores it.
            from engine.newton import NewtonHeadlessEngine

            return NewtonHeadlessEngine(
                **kwargs,
                # newton-standalone-only: sub-tree FixedJoint
                # replacement for Featherstone mass-matrix shrinkage.
                convert_joints_to_fixed=list(convert_joints_to_fixed or []),
                physics_solver_substep=physics_solver_substep,
                physics_solver_iterations=physics_solver_iterations,
                physics_solver_mass_matrix_interval=physics_solver_mass_matrix_interval,
                render_mode=render_mode,
                mujoco_pd_ke=mujoco_pd_ke,
                mujoco_pd_kd=mujoco_pd_kd,
            )
        if physics_engine == "isaac_newton":
            from kit.isaac_newton import IsaacNewtonEngine

            return IsaacNewtonEngine(
                **kwargs,
                physics_solver_substep=physics_solver_substep,
                physics_solver_iterations=physics_solver_iterations,
                render_mode=render_mode,
            )
        # default: isaac_physx
        from kit.isaac_physx import IsaacPhysXEngine

        return IsaacPhysXEngine(**kwargs, render_mode=render_mode)
