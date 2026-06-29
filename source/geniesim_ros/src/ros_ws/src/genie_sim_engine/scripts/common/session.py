# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared session bootstrap for both engine entry points.

SimpleLogger — parameterised prefix logger shared by both entry points.

EngineSession — owns everything from physics-params loading through set_topology.

Entry points do:
  1. Bootstrap (SimulationApp or kernel warmup).
  2. Construct EngineSession(node_name, params, simulation_app, physics_engine, logger).
  3. Log the engine-specific "ready" message using session attributes.
  4. Optionally append to session.post_step_hooks.
  5. Call session.startup(headless).
  6. Define a render_hook closure (entry-point-specific).
  7. Call session.run(render_hook, exit_check).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from common.loop import EngineRunLoop
from common.params import EngineNodeParams, load_physics_params, parse_init_joint_pos
from common.scene_config import parse_convert_joints_to_fixed, parse_pin_base_to_world
from engine.base import PhysicsEngine
from kit.stage import snapshot_body_transforms, wait_for_manifest


class SimpleLogger:
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def info(self, m: str) -> None:
        print(f"[{self._prefix}] {m}", flush=True)

    def warn(self, m: str) -> None:
        print(f"[{self._prefix}] WARN: {m}", flush=True)

    def error(self, m: str) -> None:
        print(f"[{self._prefix}] ERROR: {m}", flush=True)


class EngineSession:
    """Bootstrap a physics engine from a params dict + manifest.

    After construction: _core is initialised, sim is built, topology is set.
    Call startup(headless) then run(render_hook, exit_check) to enter the loop.

    Public attributes
    -----------------
    core            : genie_sim_engine_py module (pybind .so)
    logger          : the logger passed in
    sim             : PhysicsEngine instance
    physics_hz      : float
    render_hz       : float
    realtime_factor : float  # 1.0 = realtime; <1 = slower; >1 = faster
    dt              : 1.0 / physics_hz
    scene_usda      : resolved path, for log messages
    post_step_hooks : list of Callable[[float], None] — append to extend step
                      behaviour without subclassing (IsaacLab EventManager
                      pattern).  Each hook receives sim_time and is called
                      after tick_extras() every tick.
    """

    def __init__(
        self,
        node_name: str,
        params: dict,
        simulation_app: Any,
        physics_engine: str,
        logger: Any,
    ) -> None:
        import genie_sim_engine_py as _core  # deferred — .so requires ROS env

        self.core = _core
        self.logger = logger
        self.post_step_hooks: list = []

        # --- typed params (IsaacLab @configclass pattern) ---
        ep = EngineNodeParams.from_dict(params)

        physics_params_file = ep.physics_params_file
        if not physics_params_file:
            try:
                from ament_index_python.packages import get_package_share_directory

                physics_params_file = os.path.join(
                    get_package_share_directory("genie_sim_engine"), "config", "physics_params.yaml"
                )
            except Exception as exc:
                logger.warn(f"could not resolve default physics params file: {exc}")

        phys = load_physics_params(physics_params_file, logger)

        init_joint_pos = parse_init_joint_pos(ep.init_joint_pos_json, logger)
        if init_joint_pos:
            logger.info(
                f"init_joint_pos: {len(init_joint_pos)} joint(s) requested "
                f"(deg for revolute, m for prismatic) — will apply at articulation init"
            )

        render_hz = ep.render_hz if ep.render_hz > 0.0 else float(phys.render_target_hz)
        if render_hz <= 0.0:
            logger.warn(f"render_hz must be > 0; using {phys.render_target_hz}")
            render_hz = float(phys.render_target_hz)

        # --- manifest ---
        if not ep.stage_manifest:
            raise RuntimeError("parameter 'stage_manifest' is required")
        manifest_path = str(Path(ep.stage_manifest).resolve())

        wait_for_manifest(manifest_path, logger)
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        manifest_base = manifest.get("base_path") or os.getcwd()

        def _abs(p: str) -> str:
            if not p:
                return p
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(manifest_base, p))

        robot_prefix = manifest.get("robot_prefix", "genie")
        scene_usda = _abs(manifest.get("scene_usda", ""))
        robot_usda = _abs(manifest.get("robot_usda", ""))
        render_layer_usda = _abs(manifest.get("render_layer_usda", ""))
        robot_from_urdf = bool(manifest.get("robot_from_urdf", False))

        # --- scene yaml ---
        scene_yaml_path = manifest.get("scene_yaml", "")
        scene_cfg: dict = {}
        if scene_yaml_path and os.path.isfile(scene_yaml_path):
            try:
                import yaml as _yaml

                with open(scene_yaml_path, "r") as _f:
                    _loaded = _yaml.safe_load(_f) or {}
                if isinstance(_loaded, dict):
                    scene_cfg = _loaded
                else:
                    logger.warn(f"scene_yaml {scene_yaml_path} did not parse to a mapping; ignoring")
            except Exception as exc:
                logger.warn(f"failed to read scene_yaml {scene_yaml_path}: {exc} — using defaults")
        elif scene_yaml_path:
            logger.warn(f"scene_yaml {scene_yaml_path} not found — using defaults")
        else:
            logger.warn("manifest missing 'scene_yaml' field — using defaults for dynamic params")

        _robot_section = scene_cfg.get("robot") if isinstance(scene_cfg.get("robot"), dict) else {}
        # Two independent runtime-behavior flags from the scene yaml:
        #   * ``pin_base_to_world`` — applies to ALL engine paths.
        #     Kit engines (isaac_physx / isaac_newton) toggle the
        #     world-weld FixedJoint via
        #     ``kit/stage.py:_apply_fix_base_policy``; newton-standalone
        #     deactivates the URDF root_joint via
        #     ``engine/newton/setup/stage.py:_deactivate_root_joint``.
        #     Same semantic, different mechanism per engine.
        #   * ``convert_joints_to_fixed: [base, head, body]`` —
        #     newton-standalone only.  Replaces matching joints with
        #     UsdPhysics.FixedJoint to shrink Featherstone's mass
        #     matrix.  Kit engines silently ignore this list (they
        #     manage articulation via PhysX schemas, not
        #     session-layer FixedJoint replacement).
        # Independent — neither implies anything about the other.
        pin_base_to_world = parse_pin_base_to_world(_robot_section)
        convert_joints_to_fixed = parse_convert_joints_to_fixed(_robot_section)

        # --- pybind init ---
        # base_frame: "base_footprint" asks the C++ publish_odom to split
        # base_link's pose into ``odom -> base_footprint`` (ground-projected:
        # z=0, level orientation, yaw kept) plus ``base_footprint -> base_link``
        # (dynamic, carrying the residual height/tilt).  Done unconditionally
        # so every scene -- mobile or pinned, mobile-base or arm-only -- has
        # the same TF topology, which keeps downstream consumers (RViz,
        # MoveIt, the OVRtx render node) on a single contract.  Pinned arm-
        # only scenes simply have a constant residual transform; the cost is
        # one extra TF message per tick.  The empty string fallback exists
        # only for hosts pinning the single-edge TF for compatibility tests;
        # not used in production.
        _base_frame = "base_footprint"
        _core.init_ros(
            node_name=node_name,
            namespace="",
            fake_slam=ep.fake_slam,
            executor_threads=2,
            base_frame=_base_frame,
        )
        _core.init_scheduler(
            render_target_hz=render_hz,
            render_safety_ms=float(phys.render_safety_ms),
            physics_hz=ep.physics_hz,
            rtf=ep.realtime_factor,
        )

        # --- engine ---
        # physics_solver_substep / physics_solver_iterations use 0 as the
        # "engine picks its own default" sentinel.  Each engine substitutes its
        # own appropriate fallback when 0 is passed:
        #
        #   * newton-direct   — SolverFeatherstone/SolverMuJoCo defaults
        #                       (substeps=10, iters=5 for Featherstone VBD/XPBD)
        #   * isaac_newton    — leaves Newton's NewtonConfig defaults alone
        #                       (typically num_substeps=1, iterations=100).
        #                       Forcing newton-direct franka defaults onto the
        #                       wrapper drops solver iters from 100 → 5 and
        #                       under-converges cloth constraints.
        #   * isaac_physx     — ignores both knobs entirely.
        #
        # Explicit non-zero values from the CLI / launcher YAML override the
        # engine default and reach the engine constructor as-is.
        newton_solvers_path = os.path.join(os.path.dirname(manifest_path), "newton_solvers.json")
        sim = PhysicsEngine.create(
            physics_engine,
            robot_prefix=robot_prefix,
            scene_usda=scene_usda,
            robot_usda=robot_usda,
            render_layer_usda=render_layer_usda,
            physics_hz=ep.physics_hz,
            render_hz=render_hz,
            simulation_app=simulation_app,
            logger=logger,
            params=phys,
            robot_from_urdf=robot_from_urdf,
            init_joint_pos=init_joint_pos,
            runtime_usd_dump_path=os.path.join(os.path.dirname(manifest_path), "robot_runtime.usda"),
            pin_base_to_world=pin_base_to_world,
            convert_joints_to_fixed=convert_joints_to_fixed,
            newton_solvers_path=newton_solvers_path,
            scene_cfg=scene_cfg,
            scene_yaml_path=scene_yaml_path,
            physics_solver=ep.physics_solver,
            physics_solver_substep=ep.physics_solver_substep,
            physics_solver_iterations=ep.physics_solver_iterations,
            physics_solver_mass_matrix_interval=ep.physics_solver_mass_matrix_interval,
            render_mode=ep.render_mode,
            mujoco_pd_ke=ep.mujoco_pd_ke,
            mujoco_pd_kd=ep.mujoco_pd_kd,
        )

        _, body_frames = snapshot_body_transforms(sim.stage, sim.body_paths)
        _core.set_topology(list(sim.joint_names), list(body_frames))

        self.sim = sim
        self.physics_hz = ep.physics_hz
        self.render_hz = render_hz
        self.realtime_factor = ep.realtime_factor
        self.dt = 1.0 / ep.physics_hz
        self.scene_usda = scene_usda

    def startup(self, headless: bool) -> None:
        self.sim.startup(headless)
        self.sim.note_render_target(self.render_hz)

    def run(self, render_hook=None, exit_check=None) -> int:
        EngineRunLoop(
            self.core,
            self.sim,
            self.dt,
            self.logger,
            post_step_hooks=self.post_step_hooks,
            rtf=self.realtime_factor,
        ).spin(
            render_hook=render_hook,
            exit_check=exit_check,
        )
        return 0
