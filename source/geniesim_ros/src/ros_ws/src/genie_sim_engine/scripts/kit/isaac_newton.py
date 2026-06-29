#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac Sim Newton wrapper engine — MuJoCo-Warp rigid-body simulation.

Uses ``isaacsim.physics.newton`` (v0.6.0 at Isaac Sim 6.0) which exposes
exactly two solvers via ``newton_stage._get_solver``: ``SolverMuJoCo``
and ``SolverXPBD``. This engine relies on the wrapper's default
``solver_cfg`` (``MuJoCoSolverConfig``) — we never touch
``ns.cfg.solver_cfg``, so MuJoCo-Warp is what runs.

Why MuJoCo-Warp only:

  * **Crisp articulation.** MuJoCo-Warp's articulation solver gives
    rigid-body-quality joint dynamics. XPBD under the same wrapper
    treats joints as compliant position-based constraints — the
    FR3-class arms come out spongy at our 200 Hz cadence.

  * **No cloth.** ``SolverMuJoCo.step`` has no particle pipeline (verified
    by reading ``newton/_src/solvers/mujoco/solver_mujoco.py`` — zero
    references to ``particle_q``). Choosing XPBD to get cloth would
    apply XPBD to the robot too, which is the trade-off we wanted to
    avoid. For cloth, switch to ``physics_engine:=newton`` and use
    newton-standalone's full VBD / XPBD / Style3D menu.

The launch arg ``physics_solver`` is pinned to ``mujoco-warp`` at the
``DeclareLaunchArgument(choices=["mujoco-warp"])`` level, so the wrong
value can't even be typed — no runtime gate is needed in the launch
file or engine. The value reaches this engine purely cosmetically and
is unused (the wrapper's default solver_cfg already selects MuJoCo-Warp).

Cloth is a newton-standalone responsibility — this engine does not
author cloth particles in the wrapper's ``ModelBuilder.finalize``. The
wrapper's cloth support is too narrow to be reliable (API drift in
``Model.collide`` kwargs, silent solver downgrades for unsupported
``newton.solver.prefer`` values, forced ``num_substeps=10`` bumps that
override the launcher).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engine.base import PhysicsEngine


def _quat_mul(ax, ay, az, aw, bx, by, bz, bw):
    """Hamilton product (a * b) on unit quats. Components are (x, y, z, w)."""
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    rw = aw * bw - ax * bx - ay * by - az * bz
    return rx, ry, rz, rw


def _quat_rotate(qx, qy, qz, qw, vx, vy, vz):
    """Rotate vector ``v`` by unit quat ``q``. Returns ``(x', y', z')``."""
    # v' = q * (v,0) * q⁻¹  — expanded form, no scratch quats.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


_PATCHED_NEWTON_DEFAULTS = False


def _patch_isaac_newton_class_defaults(logger: Any) -> None:
    """Mutate Isaac ``NewtonConfig`` / ``MuJoCoSolverConfig`` CLASS-level
    defaults so every subsequent instance picks up newton-standalone-
    equivalent values.

    Why this exists
    ---------------
    Isaac's NewtonStage builds its live ``SolverMuJoCo`` inside
    ``world.reset()`` (called from ``IsaacSimStage.__init__``).
    Construction-time cfg fields (``cone``, ``njmax``, ``nconmax``,
    ``use_mujoco_contacts``, ``ls_parallel``, ``ls_iterations``) are
    snapshotted into the live ``mjw_model`` / dispatch flags right
    then.  Subsequent writes to ``ns.cfg.solver_cfg`` have no effect.

    Newton-standalone doesn't have this gap — it constructs
    ``SolverMuJoCo(model, njmax=1024, nconmax=512, …)`` explicitly
    after all model mutations.

    By mutating the dataclass class defaults here, the SAME values
    flow into the live solver at construction time without us having
    to intercept Isaac's call.

    Targets
    -------
    * ``MuJoCoSolverConfig``:
      - ``cone="pyramidal"``       (Isaac default ``"elliptic"`` —
        Newton-standalone uses MuJoCo's ``"pyramidal"`` default)
      - ``njmax=1024``             (Isaac 1200 — match standalone)
      - ``nconmax=512``            (Isaac 200 — Isaac is under-allocated
        for our 51-shape robot; mjwarp viewer reports ``nefc overflow``)
      - ``use_mujoco_contacts=True`` (Isaac False — Newton default;
        flipping to MuJoCo's contact code path matches the standalone
        path that produces stable behaviour in pure MuJoCo)
      - ``ls_parallel=False``      (Isaac True — Newton default)
      - ``ls_iterations=50``       (Isaac 15 — MuJoCo default 50;
        Isaac's 15 under-converges line search)
    * ``NewtonConfig``:
      - ``joint_limit_ke=1.0e4``   (Isaac 100 — Newton stock 1e4;
        Isaac's 100 produces ``solreflimit="-100 -1"`` which is too
        soft for kp=1e5 body PD at 1ms dt)
      - ``joint_limit_kd=10.0``    (Isaac 1 — Newton stock 10)

    Idempotent — runs once per process; subsequent calls are no-ops.
    """
    global _PATCHED_NEWTON_DEFAULTS
    if _PATCHED_NEWTON_DEFAULTS:
        return
    try:
        from isaacsim.physics.newton.impl.solver_config import MuJoCoSolverConfig
        from isaacsim.physics.newton.impl.newton_config import NewtonConfig
    except ImportError as exc:
        logger.warn(f"[isaac_newton] class-default patch skipped — import failed: {exc!r}")
        return

    def _set_field_default(cls: type, field_name: str, value: Any) -> tuple:
        """Update both the class attribute and the dataclass field's
        default. Returns (old, applied) — ``applied`` is False if the
        field doesn't exist on the class."""
        if not hasattr(cls, field_name):
            return (None, False)
        old = getattr(cls, field_name)
        # Update the class attribute (read by dataclass __init__ when
        # no explicit value is passed).
        setattr(cls, field_name, value)
        # Belt-and-suspenders: dataclasses keep a separate copy of the
        # field default in ``__dataclass_fields__`` (used by some code
        # paths). Update it if present.
        try:
            fld = cls.__dataclass_fields__.get(field_name) if hasattr(cls, "__dataclass_fields__") else None
            if fld is not None:
                fld.default = value
        except Exception:  # noqa: BLE001
            pass
        return (old, True)

    solver_overrides = {
        "cone": "pyramidal",
        "njmax": 1024,
        "nconmax": 512,
        "use_mujoco_contacts": True,
        "ls_parallel": False,
        "ls_iterations": 50,
    }
    for name, target in solver_overrides.items():
        old, applied = _set_field_default(MuJoCoSolverConfig, name, target)
        if applied:
            logger.info(f"[isaac_newton] MuJoCoSolverConfig.{name}: {old!r} -> {target!r} (class default).")
        else:
            logger.warn(f"[isaac_newton] MuJoCoSolverConfig has no field '{name}' — patch skipped.")

    newton_overrides = {
        "joint_limit_ke": 1.0e4,
        "joint_limit_kd": 10.0,
    }
    for name, target in newton_overrides.items():
        old, applied = _set_field_default(NewtonConfig, name, target)
        if applied:
            logger.info(f"[isaac_newton] NewtonConfig.{name}: {old!r} -> {target!r} (class default).")

    _PATCHED_NEWTON_DEFAULTS = True


class IsaacNewtonEngine(PhysicsEngine):
    """Isaac Sim Newton wrapper engine (MuJoCo-Warp rigid-body, no cloth).

    Delegates stage and articulation management to
    :class:`kit.stage.IsaacSimStage`; uses ``isaacsim.physics.newton``
    for the simulation loop. ``physics_solver`` is gated to ``mujoco``
    upstream in the launch — any other value is rejected before this
    engine is constructed.
    """

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
        pin_base_to_world: bool,
        newton_solvers_path: str = "",  # accepted for API parity; unused (no cloth)
        scene_cfg: dict | None = None,  # forwarded to IsaacSimStage so it can honour robot.init_base_pose
        scene_yaml_path: str = "",  # accepted for API parity; unused
        physics_solver: str = "mujoco-warp",  # accepted for API parity; the wrapper's default MuJoCoSolverConfig is what actually runs
        physics_solver_substep: int = 0,
        physics_solver_iterations: int = 0,
        render_mode: str = "raster",
    ) -> None:
        # IMPORTANT — patch Isaac's NewtonConfig CLASS defaults BEFORE
        # IsaacSimStage is instantiated below. ``IsaacSimStage.__init__``
        # calls ``World.reset()``, which triggers
        # ``NewtonStage.initialize_newton``, which constructs the live
        # ``SolverMuJoCo(model, **cfg.solver_cfg.__dict__)``. All
        # construction-time cfg fields (cone, njmax, nconmax,
        # use_mujoco_contacts, ls_parallel, ls_iterations) are frozen
        # into the live solver / mjw_model at that point. Writing them
        # later via ``_apply_newton_cfg`` is too late — that's the gap
        # the user identified vs. newton-standalone's design, which
        # constructs SolverMuJoCo with explicit kwargs AFTER all model
        # mutations.
        #
        # By patching ``MuJoCoSolverConfig`` and ``NewtonConfig`` class
        # defaults here, every subsequent ``NewtonConfig()`` /
        # ``MuJoCoSolverConfig()`` instantiation (including the one
        # NewtonStage creates) picks up our values without us having to
        # hook the construction itself.
        #
        # Only fields whose construction-time read is the one that
        # matters get patched here. ``num_substeps`` is read per-step
        # so it stays as a ``_apply_newton_cfg`` write.
        _patch_isaac_newton_class_defaults(logger)
        self._logger = logger
        # ``PhysicsParams`` from isaac_params/physics_params.yaml.
        # Kept on the wrapper so the MJCF post-process pipeline (and
        # any live-mjw_model mutation that wants gripper/equality
        # tuning) can read from the same single source of truth that
        # ``kit/stage.py:_configure_drives`` reads.  ``params`` is
        # always provided by the engine entry-point — there is no
        # ``None`` path in production.
        self._params = params
        # NewtonStage.cfg knobs — applied in ``startup`` once the stage
        # has been acquired. ``0`` means "trust the wrapper default"
        # (typically num_substeps=1, MuJoCoSolverConfig.iterations=100).
        self._sim_substeps = int(physics_solver_substep) if physics_solver_substep else 0
        self._solver_iterations = int(physics_solver_iterations) if physics_solver_iterations else 0
        # Outer-frame rate.  Kept on the wrapper so the post-step MJCF
        # dump in ``_force_solver_init`` can compute the live substep
        # dt and augment ``<option timestep>`` / ``<custom>`` numerics
        # via ``common/mjcf_postprocess.py``.
        self._physics_hz = float(physics_hz)
        self._render_hz = float(render_hz) if render_hz else 30.0
        self._render_mode = (render_mode or "raster").strip().lower()
        self._newton_stage_cache: Any = None  # None=unknown, False=absent, obj=found
        # ``robot_runtime.xml`` MJCF dump — written once after the
        # NewtonStage's ``SolverMuJoCo`` is built.  This mirrors the
        # newton-standalone path's ``MuJoCoWarpAdapter(save_to_mjcf=…)``
        # flow so operators get matched ``robot_runtime.usda`` +
        # ``robot_runtime.xml`` snapshots regardless of which engine
        # ran.  We derive the sibling MJCF path from
        # ``runtime_usd_dump_path`` so newton-standalone and
        # isaac_newton land on the EXACT same target — see
        # ``engine.newton.engine`` for the canonical derivation.
        # The dump itself fires from ``step()`` after the live solver
        # has been built (option A); see the block in ``step()`` for
        # the full rationale.
        import os as _os

        if runtime_usd_dump_path:
            self._mjcf_out_path = _os.path.splitext(runtime_usd_dump_path)[0] + ".xml"
        else:
            self._mjcf_out_path = ""
        # ``_mjcf_dump_done`` flips True once the post-first-step
        # ``_force_solver_init`` has fired (option A: the disposable
        # solver is constructed AFTER ``_init_articulation`` has
        # written runtime gains into ``model.joint_target_ke`` AND
        # the live solver has read them on its first step, so the
        # dumped MJCF reflects what the live runtime actually
        # executes).
        self._mjcf_dump_done = False

        # Live-equality stiffening fires after the wrapper's lazy
        # SolverMuJoCo build, not at startup. ``ns.solver`` is None
        # until the first ``step_sim`` triggers ``initialize_newton``,
        # so the patch waits for ``mjw_model`` to materialise. Flips
        # to True once successfully patched (or definitively skipped)
        # so we stop polling on every tick.
        self._eq_stiffen_done = False

        # Cached for ``tick_extras`` — see the docstring there for why
        # headless isaac_newton has to drive ``simulation_app.update()``
        # from the physics loop itself, and how we avoid paying for it
        # on every tick.
        #
        # We tried bypassing apply_action entirely and writing straight
        # to ``ns.control.joint_target_pos`` (mirroring
        # ``engine.newton.control._ControlMixin``). The mapping, sync,
        # biastype, and gains all check out and the writes are
        # observable in the log, but downstream actuation never fires —
        # the wrapper apparently completes the apply_action chain via
        # something only Kit's stage-update event drives, which is not
        # debuggable from the wrapper source alone. So we stay on the
        # known-working apply_action path and use ``simulation_app.update()``
        # as the chain driver, with a dict-compare freshness gate so
        # the ~9 ms Kit update only fires on actual command changes.
        self._simulation_app = simulation_app
        self._headless = False
        # Dirty flag set by :meth:`apply_commands` whenever ``cmd_positions``
        # or ``cmd_4ws_stamp`` differs from the last applied snapshot,
        # cleared by :meth:`tick_extras` after the Kit update flushes
        # the wrapper's apply_action chain through.
        self._cmd_pending = False
        self._last_cmd_positions: Dict[str, float] = {}
        self._last_cmd_4ws_stamp: float = 0.0

        # Lazily-built parent-index table used by ``get_body_transforms``
        # to convert Newton's world-space ``state_0.body_q`` into the
        # local-relative pose format the renderer's ``_apply_xform``
        # expects (matches what PhysX's USD writeback produces).
        # ``None`` = not built yet, ``False`` = build failed (fall back
        # to USD readback once and stop trying).
        self._body_xform_table: Any = None
        self._body_xform_warned: bool = False

        from kit.stage import IsaacSimStage

        self._stage_obj = IsaacSimStage(
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
            physics_engine="isaac_newton",
            # IsaacSimStage's internal arg is still ``fix_base`` —
            # canonical scene-yaml name is ``pin_base_to_world``; both
            # refer to "weld base_link → world".
            fix_base=pin_base_to_world,
            newton_solvers_path="",  # cloth disabled on this engine
            scene_cfg=scene_cfg,
        )

        # Viewport / Hydra render mode. IsaacSimStage's ``__init__``
        # already ran ``World.reset()`` + 5 ``simulation_app.update()``
        # calls so the editor viewport has rendered at least once.
        try:
            from kit.bootstrap import configure_viewport_for_debug

            configure_viewport_for_debug(render_mode=self._render_mode)
        except Exception as exc:
            self._logger.warn(f"[isaac_newton] viewport config failed: {exc}")

    # ------------------------------------------------------------------
    # PhysicsEngine interface
    # ------------------------------------------------------------------

    @property
    def stage(self) -> Any:
        return self._stage_obj.stage

    @property
    def robot_prefix(self) -> str:
        return self._stage_obj._robot_prefix

    @property
    def joint_names(self) -> List[str]:
        return self._stage_obj.joint_names

    @property
    def joint_prim_map(self) -> Dict[str, str]:
        return self._stage_obj.joint_prim_map

    @property
    def body_paths(self) -> List[str]:
        return self._stage_obj.body_paths

    def _apply_newton_cfg(self) -> None:
        """Push ``physics_solver_substep`` / ``physics_solver_iterations``
        into ``NewtonStage.cfg``.

        Both knobs are only applied when non-zero. ``0`` means "trust
        the wrapper default" — ``num_substeps=1`` (sufficient for
        MuJoCo-Warp on rigid bodies) and
        ``MuJoCoSolverConfig.iterations=100`` (MuJoCo's Newton solver
        iters; ``ls_iterations=15`` line-search defaults to 15).

        Each field is probed via ``hasattr`` so we don't break across
        Newton wrapper minor versions; if the live cfg doesn't expose
        a knob we log it and move on.

        Called from ``startup`` rather than ``__init__`` because the
        NewtonStage isn't always available at construction time —
        ``acquire_stage()`` only succeeds after ``World.reset()`` has
        actually built the Newton model.
        """
        if not self._sim_substeps and not self._solver_iterations:
            # Still proceed — we always want to inject per-class model
            # writes (passive-joint, contact, mode-override) regardless
            # of whether substep/iter CLI knobs were passed.
            pass
        ns = self._get_newton_stage()
        if ns is None or getattr(ns, "cfg", None) is None:
            self._logger.warn(
                "[isaac_newton] NewtonStage.cfg unavailable at startup; "
                "physics_solver_substep / physics_solver_iterations not applied"
            )
            return
        cfg = ns.cfg
        if self._sim_substeps and hasattr(cfg, "num_substeps"):
            old = int(getattr(cfg, "num_substeps", 0) or 0)
            cfg.num_substeps = self._sim_substeps
            self._logger.info(f"[isaac_newton] cfg.num_substeps: {old} -> {self._sim_substeps}")
        # MuJoCo solver iterations live on ``cfg.solver_cfg.iterations``
        # (``MuJoCoSolverConfig.iterations``). ``hasattr`` keeps this
        # tolerant of attribute renames in future wrapper versions.
        if self._solver_iterations:
            sc = getattr(cfg, "solver_cfg", None)
            if sc is not None and hasattr(sc, "iterations"):
                old = int(getattr(sc, "iterations", 0) or 0)
                sc.iterations = self._solver_iterations
                self._logger.info(f"[isaac_newton] cfg.solver_cfg.iterations: {old} -> {self._solver_iterations}")
            else:
                self._logger.info(
                    "[isaac_newton] solver_cfg has no ``iterations`` attr; " "physics_solver_iterations not applied"
                )
        # All other solver_cfg knobs (``cone``, ``njmax``, ``nconmax``,
        # ``use_mujoco_contacts``, ``ls_parallel``, ``ls_iterations``)
        # are CLASS-DEFAULT patched at the top of ``IsaacNewton.__init__``
        # via ``_patch_isaac_newton_class_defaults``. They have to land
        # before ``IsaacSimStage.__init__`` calls ``world.reset()`` (which
        # triggers ``NewtonStage.initialize_newton`` and constructs the
        # live ``SolverMuJoCo`` with all construction-time cfg fields
        # frozen into the live mjw_model). Writing them HERE — after
        # ``startup_loop_setup`` ran — is the gap that produced the
        # chassis-drift symptom: dump showed the right values (disposable
        # re-reads ns.model live) but the live solver kept Isaac's
        # defaults.
        # Joint-limit ke/kd are ALSO class-default patched (via
        # ``NewtonConfig.joint_limit_ke``); the builder reads
        # ``self.cfg.joint_limit_ke`` and stamps it onto
        # ``default_joint_cfg.limit_ke`` BEFORE the USD importer runs,
        # so every joint already has the stiff 1e4/10 by the time we
        # get here. No live mutation needed.
        # Per-class passive-joint authoring (armature, frictionloss).
        # Mirrors the newton-standalone adapter's per-class loop in
        # ``engine/newton/adapters/mujoco_warp.py:prepare_model``.
        # Without this, isaac_newton's dump shows ``armature=0.1`` and
        # missing ``frictionloss`` on every joint (Isaac's defaults),
        # while newton-standalone authors yaml values per-class
        # (chassis_drive frictionloss=0.05, body armature=0.1, head/arm
        # armature=0.05, wrist 0.02, etc.). The drive-wheel
        # frictionloss is the key one for chassis stability — without
        # it, mjwarp PGS can't pin the wheel against ground reaction
        # under no command.
        ns_for_model = self._get_newton_stage()
        model_for_limits = getattr(ns_for_model, "model", None) if ns_for_model is not None else None
        if model_for_limits is None:
            self._logger.info(
                "[isaac_newton] NewtonStage.model unavailable at _apply_newton_cfg; " "per-class model writes skipped."
            )
        self._apply_per_class_passive_joint(model_for_limits)
        # Per-class CONTACT overrides (shape_material_ke/kd/mu, geom_solimp).
        # Mirrors newton-standalone's lifecycle.py:_apply_mjc_contact_overrides
        # + the _ROBOT_KU friction broadcast. Without this, every
        # geom in the dumped MJCF carries Newton's defaults
        # (solref="0.02", friction="1") instead of the
        # MJC_CONTACT_DEFAULTS values (robot solref="0.005 5",
        # friction="1.5") that pure-MuJoCo proves stable on this
        # robot. Soft contacts at 1ms substep + kp=1e5 body PD =
        # contact-pop oscillation under any standing load.
        self._apply_per_class_contact(model_for_limits)
        # Force POSITION-only mode for PD-controlled joints.
        # Isaac's NewtonStage builder uses
        # ``force_position_velocity_actuation=True``
        # (newton_stage.py:413), so every joint with ke>0 AND kd>0
        # gets ``JointTargetMode.POSITION_VELOCITY``. Newton's mjwarp
        # converter then emits TWO actuators per such joint:
        #   * affine PD with biasprm=[0, -kp, 0]   (POSITION term)
        #   * velocity damper with biasprm=[0, 0, -kd]  (VELOCITY term)
        # Both fire every tick. With Isaac's apply_action writing
        # joint_target_vel=0 for chassis joints on cmd timeout, the
        # velocity damper actively brakes the steer joint while the
        # position actuator is also pulling. The standalone path
        # forces POSITION-only for everything except chassis_drive
        # (which is forced VELOCITY) — same convention is what
        # produces the stable working dump.
        self._apply_position_mode_override(model_for_limits)

    def _apply_per_class_passive_joint(self, model: Any) -> None:
        """Write per-class ``joint_armature`` and ``joint_friction`` into
        the live Newton model, sourced from
        ``physics_params.yaml::usd_drive_api.<class>``.

        Mirrors what the newton-standalone adapter does in
        ``engine/newton/adapters/mujoco_warp.py:prepare_model``. Isaac's
        wrapper has no equivalent path, so without this every joint
        carries the USD importer's defaults: armature=0.1 (heavy for
        wrist/head links) and no frictionloss (catastrophic for
        free-spin chassis_drive wheels — the velocity actuator alone
        can't hold them against ground reaction in mjwarp PGS).

        Skips JK_GRIPPER (assemble_robot.py's overlay already authored
        ``mjc:damping`` + ``physxJoint:armature`` per gripper joint —
        we must not stomp those) and JK_OTHER / JK_PASSIVE (no class
        tuning).

        Per-DOF write (joint_armature / joint_friction are qd-indexed
        in Newton's model).
        """
        if model is None:
            self._logger.info("[isaac_newton] passive-joint: model unavailable; skipped.")
            return
        try:
            from common.joint_classification import (  # noqa: PLC0415
                JK_ARM,
                JK_ARM_MID,
                JK_ARM_SHOULDER,
                JK_ARM_WRIST,
                JK_BODY,
                JK_CHASSIS_DRIVE,
                JK_CHASSIS_STEER,
                JK_CHASSIS_WHEEL,
                JK_GRIPPER,
                JK_HEAD,
                JK_PASSIVE,
                classify_joint_by_name,
                is_chassis_wheel_free,
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[isaac_newton] passive-joint: classifier import failed: {exc!r}")
            return
        p = self._params
        if p is None:
            self._logger.info("[isaac_newton] passive-joint: no PhysicsParams; skipped.")
            return
        # Per-class (frictionloss, armature). dof_damping is left at
        # whatever the USD-import path landed on — yaml has 0.0 across
        # the board for this robot, and writing 0 unconditionally
        # would clobber any per-joint mjc:damping the assemble overlay
        # authored on the gripper.
        try:
            class_table: Dict[str, tuple] = {
                JK_BODY: (p.drive_body.dof_frictionloss, p.drive_body.armature),
                JK_HEAD: (p.drive_head.dof_frictionloss, p.drive_head.armature),
                JK_ARM: (p.drive_default_revolute.dof_frictionloss, p.drive_default_revolute.armature),
                JK_ARM_SHOULDER: (p.drive_arm_shoulder.dof_frictionloss, p.drive_arm_shoulder.armature),
                JK_ARM_MID: (p.drive_arm_mid.dof_frictionloss, p.drive_arm_mid.armature),
                JK_ARM_WRIST: (p.drive_arm_wrist.dof_frictionloss, p.drive_arm_wrist.armature),
                JK_CHASSIS_DRIVE: (p.drive_chassis_drive_joint.dof_frictionloss, p.drive_chassis_drive_joint.armature),
                JK_CHASSIS_STEER: (p.drive_chassis_steer_joint.dof_frictionloss, p.drive_chassis_steer_joint.armature),
                JK_GRIPPER: (None, None),
                JK_PASSIVE: (None, None),
            }
        except AttributeError as exc:
            self._logger.warn(
                f"[isaac_newton] passive-joint: PhysicsParams missing expected drive blocks "
                f"({exc!r}); skipping per-class write."
            )
            return
        labels = list(getattr(model, "joint_label", []) or [])
        qd_start_arr = getattr(model, "joint_qd_start", None)
        dof_dim_arr = getattr(model, "joint_dof_dim", None)
        armature_arr = getattr(model, "joint_armature", None)
        friction_arr = getattr(model, "joint_friction", None)
        limit_lo_arr = getattr(model, "joint_limit_lower", None)
        limit_hi_arr = getattr(model, "joint_limit_upper", None)
        if not labels or qd_start_arr is None or dof_dim_arr is None:
            self._logger.info(
                f"[isaac_newton] passive-joint: model lacks expected fields "
                f"(labels={len(labels)}, qd_start={qd_start_arr is not None}, "
                f"dof_dim={dof_dim_arr is not None}); skipped."
            )
            return
        if armature_arr is None and friction_arr is None:
            self._logger.info("[isaac_newton] passive-joint: no joint_armature / joint_friction array; skipped.")
            return
        try:
            qd_start_np = qd_start_arr.numpy()
            dof_dim_np = dof_dim_arr.numpy()
            limit_lo_np = limit_lo_arr.numpy() if limit_lo_arr is not None else None
            limit_hi_np = limit_hi_arr.numpy() if limit_hi_arr is not None else None
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[isaac_newton] passive-joint: GPU readback failed: {exc!r}")
            return
        armature_np = armature_arr.numpy().copy() if armature_arr is not None else None
        friction_np = friction_arr.numpy().copy() if friction_arr is not None else None
        n_armature_set = 0
        n_friction_set = 0
        class_counts: Dict[str, int] = {}
        unmatched_samples: list = []
        for j, name in enumerate(labels):
            if j >= len(qd_start_np):
                break
            short = name.rsplit("/", 1)[-1] if "/" in name else name
            kind = classify_joint_by_name(short)
            qd_start = int(qd_start_np[j])
            try:
                dof_count = int(dof_dim_np[j, 0]) + int(dof_dim_np[j, 1])
            except Exception:  # noqa: BLE001
                dof_count = 1
            # Split chassis_wheel into drive vs steer using the joint's
            # actual limits (radians on Newton's model — pass the
            # rad-scale threshold of 12.0 to is_chassis_wheel_free).
            # Without this every chassis wheel hits the lookup as
            # ``JK_CHASSIS_WHEEL`` which our table doesn't know, so
            # nothing gets written.
            if (
                kind == JK_CHASSIS_WHEEL
                and limit_lo_np is not None
                and limit_hi_np is not None
                and qd_start < len(limit_lo_np)
            ):
                lo_v = float(limit_lo_np[qd_start])
                hi_v = float(limit_hi_np[qd_start])
                kind = JK_CHASSIS_DRIVE if is_chassis_wheel_free(lo_v, hi_v, threshold=12.0) else JK_CHASSIS_STEER
            class_counts[kind] = class_counts.get(kind, 0) + 1
            cls_passive = class_table.get(kind)
            if cls_passive is None:
                if len(unmatched_samples) < 5:
                    unmatched_samples.append((short, kind))
                continue
            target_friction, target_armature = cls_passive
            for k in range(dof_count):
                qd = qd_start + k
                if friction_np is not None and target_friction is not None and qd < len(friction_np):
                    friction_np[qd] = float(target_friction)
                    n_friction_set += 1
                if (
                    armature_np is not None
                    and target_armature is not None
                    and float(target_armature) > 0.0
                    and qd < len(armature_np)
                ):
                    armature_np[qd] = float(target_armature)
                    n_armature_set += 1
        try:
            if friction_np is not None and friction_arr is not None:
                friction_arr.assign(friction_np.astype("float32"))
            if armature_np is not None and armature_arr is not None:
                armature_arr.assign(armature_np.astype("float32"))
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[isaac_newton] passive-joint: GPU write failed: {exc!r}")
            return
        unmatched_str = f", unmatched_samples={unmatched_samples}" if unmatched_samples else ""
        self._logger.info(
            f"[isaac_newton] passive-joint: armature set on {n_armature_set} DOF(s), "
            f"frictionloss set on {n_friction_set} DOF(s); class counts={class_counts}{unmatched_str}."
        )

    def _apply_per_class_contact(self, model: Any) -> None:
        """Write per-class contact compliance into the live Newton
        model: ``shape_material_ke / shape_material_kd / shape_material_mu``
        (and ``geom_solimp`` if present).

        Mirrors the newton-standalone path:
          * lifecycle.py:_apply_mjc_contact_overrides — per-class solref
            inverted to ke/kd via ``convert_solref(ke, kd, 1, 1)``
          * lifecycle.py:671-675 — scene-wide friction broadcast
            (``_ROBOT_KU = 1.5``).

        Without this, every robot collider in the dumped MJCF (and
        more importantly the LIVE mjwarp model) carries Newton's stock
        defaults (solref=0.02, mu=1.0). At 1 ms substep + kp=1e5 body
        PD that's soft enough to oscillate the body against the
        ground; the chassis_steer joints carry the pitch reaction
        torque and the drive wheels see slip oscillation as a result.

        The pure-MuJoCo dump that's known stable shows solref="0.005 5"
        (robot/passive/static) and solref="0.02 1" (floor) — those are
        what we author here.
        """
        if model is None:
            self._logger.info("[isaac_newton] mjc_contact: model unavailable; skipped.")
            return
        try:
            from common.object_classification import (  # noqa: PLC0415
                ALL_OBJECT_KINDS,
                MJC_CONTACT_DEFAULTS,
                classify_shape,
            )
            import newton  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[isaac_newton] mjc_contact: import failed: {exc!r}")
            return

        ke_arr = getattr(model, "shape_material_ke", None)
        kd_arr = getattr(model, "shape_material_kd", None)
        mu_arr = getattr(model, "shape_material_mu", None)
        if ke_arr is None or kd_arr is None:
            self._logger.info("[isaac_newton] mjc_contact: shape_material_ke/kd unavailable; skipped.")
            return

        def _solref_to_ke_kd(solref: tuple) -> tuple:
            timeconst, dampratio = float(solref[0]), float(solref[1])
            if timeconst <= 0.0 or dampratio <= 0.0:
                return 2500.0, 100.0
            kd = 2.0 / timeconst
            ke = (kd / (2.0 * dampratio)) ** 2
            return ke, kd

        params: Dict[str, Dict[str, tuple]] = {k: dict(MJC_CONTACT_DEFAULTS[k]) for k in ALL_OBJECT_KINDS}

        try:
            n_shapes = int(getattr(model, "shape_count", 0) or 0)
            if n_shapes == 0:
                self._logger.info("[isaac_newton] mjc_contact: shape_count=0; skipped.")
                return

            shape_types = model.shape_type.numpy()
            shape_bodies = model.shape_body.numpy()
            shape_labels = list(getattr(model, "shape_label", []) or [])
            body_labels = list(getattr(model, "body_label", []) or [])
            JT_PLANE = int(newton.GeoType.PLANE)
            robot_prefix = self.robot_prefix or ""

            ke_np = ke_arr.numpy().copy()
            kd_np = kd_arr.numpy().copy()
            mu_np = mu_arr.numpy().copy() if mu_arr is not None else None
            solimp_arr = getattr(model, "geom_solimp", None)
            solimp_np = solimp_arr.numpy().copy() if solimp_arr is not None else None
            if solimp_np is not None and solimp_np.ndim == 1:
                solimp_np = solimp_np.reshape(-1, 5)

            class_counts: Dict[str, int] = {k: 0 for k in ALL_OBJECT_KINDS}
            class_ke_kd: Dict[str, tuple] = {}
            for i in range(n_shapes):
                b = int(shape_bodies[i]) if i < len(shape_bodies) else -1
                kind = classify_shape(
                    shape_label=shape_labels[i] if i < len(shape_labels) else "",
                    body_index=b,
                    body_label=body_labels[b] if 0 <= b < len(body_labels) else None,
                    shape_type_int=int(shape_types[i]),
                    robot_prefix=robot_prefix,
                    plane_geo_type=JT_PLANE,
                )
                if kind not in class_ke_kd:
                    class_ke_kd[kind] = _solref_to_ke_kd(params[kind]["solref"])
                ke, kd = class_ke_kd[kind]
                ke_np[i] = ke
                kd_np[i] = kd
                if solimp_np is not None:
                    if solimp_np.ndim == 2:
                        solimp_np[i] = params[kind]["solimp"]
                    elif solimp_np.ndim == 3:
                        solimp_np[:, i] = params[kind]["solimp"]
                class_counts[kind] += 1

            ke_arr.assign(ke_np.astype("float32"))
            kd_arr.assign(kd_np.astype("float32"))
            if solimp_np is not None and solimp_arr is not None:
                solimp_arr.assign(solimp_np.astype("float32"))

            # Scene-wide friction broadcast (matches standalone's
            # ``_ROBOT_KU = 1.5`` blanket write at lifecycle.py:671-675).
            # No per-class friction in MJC_CONTACT_DEFAULTS yet; if you
            # later need wheels at mu=2.0 or floor at mu=0.8, add it
            # there and switch this to per-class lookup.
            mu_target = 1.5
            if mu_np is not None and mu_arr is not None:
                mu_np[...] = mu_target
                mu_arr.assign(mu_np.astype("float32"))

            self._logger.info(
                f"[isaac_newton] mjc_contact: ke/kd written for {n_shapes} shape(s); "
                f"mu broadcast to {mu_target}; class counts={class_counts}."
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[isaac_newton] mjc_contact: write failed (continuing): {exc!r}")

    def _apply_position_mode_override(self, model: Any) -> None:
        """Force ``joint_target_mode`` to POSITION (or VELOCITY for
        chassis_drive, EFFORT for gripper followers) — overriding
        Isaac's ``force_position_velocity_actuation=True`` which
        emits TWO actuators per PD-controlled joint and creates
        a position-vs-velocity ctrl conflict on every tick.

        Mirrors the standalone adapter's mode-forcing block at
        ``mujoco_warp.py:641-672``.

        Mode constants:
          * 0 = NONE
          * 1 = POSITION
          * 2 = VELOCITY
          * 3 = POSITION_VELOCITY (the bad one)
          * 4 = EFFORT
        """
        if model is None:
            return
        mode_arr = getattr(model, "joint_target_mode", None)
        ke_arr = getattr(model, "joint_target_ke", None)
        if mode_arr is None or ke_arr is None:
            self._logger.info("[isaac_newton] mode-override: joint_target_mode / ke unavailable; skipped.")
            return
        try:
            from common.joint_classification import (  # noqa: PLC0415
                JK_CHASSIS_DRIVE,
                JK_CHASSIS_WHEEL,
                JK_GRIPPER,
                classify_joint_by_name,
                is_chassis_wheel_free,
            )
            import numpy as _np  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[isaac_newton] mode-override: import failed: {exc!r}")
            return
        labels = list(getattr(model, "joint_label", []) or [])
        qd_start_arr = getattr(model, "joint_qd_start", None)
        dof_dim_arr = getattr(model, "joint_dof_dim", None)
        limit_lo_arr = getattr(model, "joint_limit_lower", None)
        limit_hi_arr = getattr(model, "joint_limit_upper", None)
        if qd_start_arr is None or dof_dim_arr is None or not labels:
            self._logger.info("[isaac_newton] mode-override: model lacks fields; skipped.")
            return
        try:
            mode_np = mode_arr.numpy().copy()
            ke_np = ke_arr.numpy()
            qd_start_np = qd_start_arr.numpy()
            dof_dim_np = dof_dim_arr.numpy()
            limit_lo_np = limit_lo_arr.numpy() if limit_lo_arr is not None else None
            limit_hi_np = limit_hi_arr.numpy() if limit_hi_arr is not None else None
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[isaac_newton] mode-override: readback failed: {exc!r}")
            return

        MODE_POSITION = 1
        MODE_VELOCITY = 2
        MODE_EFFORT = 4
        n_to_pos = 0
        n_to_vel = 0
        n_to_effort = 0
        # Followers detection: any gripper joint we DIDN'T author with
        # non-zero ke is a follower (constraint-driven, ke=0 by yaml).
        # Master has non-zero ke; we leave it at POSITION.
        for j, name in enumerate(labels):
            if j >= len(qd_start_np):
                break
            short = name.rsplit("/", 1)[-1] if "/" in name else name
            kind = classify_joint_by_name(short)
            qd_start = int(qd_start_np[j])
            try:
                dof_count = int(dof_dim_np[j, 0]) + int(dof_dim_np[j, 1])
            except Exception:  # noqa: BLE001
                dof_count = 1
            # Refine chassis_wheel via limits (rad threshold).
            if (
                kind == JK_CHASSIS_WHEEL
                and limit_lo_np is not None
                and limit_hi_np is not None
                and qd_start < len(limit_lo_np)
            ):
                lo_v = float(limit_lo_np[qd_start])
                hi_v = float(limit_hi_np[qd_start])
                kind = JK_CHASSIS_DRIVE if is_chassis_wheel_free(lo_v, hi_v, threshold=12.0) else "chassis_steer"
            for k in range(dof_count):
                qd = qd_start + k
                if qd >= len(mode_np):
                    break
                if kind == JK_CHASSIS_DRIVE:
                    if mode_np[qd] != MODE_VELOCITY:
                        mode_np[qd] = MODE_VELOCITY
                        n_to_vel += 1
                elif kind == JK_GRIPPER:
                    # Gripper followers (ke=0) → EFFORT (no actuator).
                    # Gripper master (ke>0) → POSITION.
                    if qd < len(ke_np) and float(ke_np[qd]) > 0.0:
                        if mode_np[qd] != MODE_POSITION:
                            mode_np[qd] = MODE_POSITION
                            n_to_pos += 1
                    else:
                        if mode_np[qd] != MODE_EFFORT:
                            mode_np[qd] = MODE_EFFORT
                            n_to_effort += 1
                else:
                    # Body / head / arm / chassis_steer: ke>0 → POSITION.
                    # Anything with ke=0 we leave alone.
                    if qd < len(ke_np) and float(ke_np[qd]) > 0.0:
                        if mode_np[qd] != MODE_POSITION:
                            mode_np[qd] = MODE_POSITION
                            n_to_pos += 1
        try:
            mode_arr.assign(mode_np.astype(_np.int32))
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[isaac_newton] mode-override: GPU write failed: {exc!r}")
            return
        self._logger.info(
            f"[isaac_newton] mode-override: forced POSITION on {n_to_pos} DOF(s), "
            f"VELOCITY on {n_to_vel} (chassis_drive), EFFORT on {n_to_effort} (gripper followers). "
            f"Cancels Isaac's force_position_velocity_actuation=True."
        )

    def step(self, dt: float, step_start: float) -> float:
        t0 = time.monotonic()
        if not getattr(self, "_step_beacon", False):
            self._step_beacon = True
            print(f"[isaac_newton] step() first call: dt={dt} step_start={step_start}", flush=True)
        # Drive Newton's step directly.  Backstory:
        #
        # ``omni.physx.get_physx_simulation_interface()`` is the
        # *PhysX-specific* C++ simulator.  The Newton wrapper registers
        # with ``omni.physics.core`` (see ``register_simulation.py`` —
        # ``physics.register_simulation(self.simulation, "Newton")``),
        # which is a separate dispatcher.  Calling ``physx_sim.simulate()``
        # on an isaac_newton scene returns immediately — PhysX owns no
        # objects in this scene, so there is nothing for it to step.
        # Newton itself only advances when ``NewtonStage.on_update`` is
        # invoked, and that hook only fires off Kit's stage-update
        # event inside ``simulation_app.update()``.
        #
        # In GUI mode the run loop's render_hook calls
        # ``simulation_app.update()`` ~30 times/sec so Newton ticks on
        # the render cadence.  In headless mode the loop never calls
        # update, so without explicit stepping here Newton freezes and
        # the robot only moves at command-change events that other
        # paths might trigger.
        #
        # Calling ``ns.step_sim(dt)`` is the same code path the wrapper
        # ultimately reaches through ``on_update`` → ``simulation_functions.simulate``
        # but skips the Kit event bus, the substep loop, and the
        # ``update_fabric`` writeback (we read ``state_0.body_q``
        # directly in :meth:`get_body_transforms`, so the Fabric → USD
        # sync isn't needed here).  Solver lazy-init still happens on
        # the first call (``step_sim`` calls ``initialize_newton`` when
        # ``self.initialized`` is False), so the MJCF-dump trigger
        # below still lands at the same point in the lifecycle.
        ns = self._get_newton_stage()
        if ns is not None:
            # Newton's step_sim early-returns when ``playing`` is False
            # AND the timeline isn't playing.  Startup runs World.reset
            # + several simulation_app.update() ticks before we get
            # here, which fires on_resume → ``playing = True``.  Keep
            # it pinned defensively so a stray pause event can't freeze
            # us.
            if hasattr(ns, "playing") and not ns.playing:
                ns.playing = True
            ns.step_sim(dt)
        else:
            # NewtonStage not yet acquired — first call(s) before
            # ``extension.acquire_stage()`` succeeds.  Fall back to the
            # physx_sim path the wrapper's lazy initialization latches
            # onto.
            self._stage_obj.physx_sim.simulate(dt, step_start)
            self._stage_obj.physx_sim.fetch_results()
        if not self._mjcf_dump_done:
            self._mjcf_dump_done = True
            self._force_solver_init()
        if not self._eq_stiffen_done and ns is not None:
            solver = getattr(ns, "solver", None)
            if solver is not None and getattr(solver, "mjw_model", None) is not None:
                gripper = getattr(self._params, "drive_gripper", None)
                solref = tuple(gripper.mimic_eq_solref) if gripper is not None else (-10000.0, -100.0)
                solimp = tuple(gripper.mimic_eq_solimp) if gripper is not None else (0.95, 0.99, 0.001, 0.5, 2.0)
                try:
                    self._stiffen_live_mimic_equalities(ns, solref=solref, solimp=solimp)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warn(f"[isaac_newton] _stiffen_live_mimic_equalities failed (continuing): {exc!r}")
                # One-shot mjw_model audit: count actuators / DOFs / joints.
                # Restless behaviour with no commands + 52 dropped duplicate
                # actuators in the XML strongly suggests the live solver
                # has duplicates too — each one contributes torque per tick.
                try:
                    mjw = solver.mjw_model
                    nu = (
                        int(mjw.actuator_gainprm.shape[1])
                        if mjw.actuator_gainprm.ndim >= 2
                        else int(mjw.actuator_gainprm.shape[0])
                    )
                    njnt = int(mjw.jnt_type.shape[0]) if hasattr(mjw, "jnt_type") else -1
                    nv = int(mjw.dof_armature.shape[0]) if hasattr(mjw, "dof_armature") else -1
                    cone_val = int(getattr(mjw.opt, "cone", -1)) if hasattr(mjw, "opt") else -1
                    iter_val = int(getattr(mjw.opt, "iterations", -1)) if hasattr(mjw, "opt") else -1
                    timestep_val = None
                    try:
                        ts = mjw.opt.timestep
                        timestep_val = float(ts.numpy().flat[0]) if hasattr(ts, "numpy") else float(ts)
                    except Exception:
                        pass
                    print(
                        f"[isaac_newton] live-mjw audit: nu={nu} njnt={njnt} nv={nv} "
                        f"cone={cone_val} (0=pyramidal, 1=elliptic) "
                        f"iter={iter_val} timestep={timestep_val} "
                        f"(expect nu around 70 for this robot; ~122 indicates duplicate actuators).",
                        flush=True,
                    )
                    # Backend dispatch — confirm whether the LIVE solver
                    # is CPU (mj_step) or GPU (mjwarp). We default to
                    # the GPU mjwarp path (Isaac's default + Newton's
                    # default since use_mujoco_cpu=False is the
                    # class default after _patch_isaac_newton_class_defaults).
                    use_cpu_live = bool(getattr(solver, "use_mujoco_cpu", False))
                    print(
                        f"[isaac_newton] live-mjw backend: solver.use_mujoco_cpu={use_cpu_live} "
                        f"({'CPU mj_step' if use_cpu_live else 'GPU mjwarp'} per-tick).",
                        flush=True,
                    )
                    # mjwarp njmax/nconmax — under-allocation here causes
                    # silent constraint-row truncation → contact friction
                    # not fully applied → wheel slip drift. Standalone
                    # passes njmax=1024, nconmax=512 explicitly. Isaac
                    # uses MuJoCoSolverConfig defaults (1200, 200) before
                    # our cfg writes; need to verify our writes reached
                    # the live mjw_data.
                    try:
                        mjw_data = getattr(solver, "mjw_data", None)
                        njmax_live = getattr(mjw_data, "njmax", -1) if mjw_data is not None else -1
                        nconmax_live = getattr(mjw_data, "nconmax", -1) if mjw_data is not None else -1
                        nefc_live = getattr(mjw_data, "nefc", None) if mjw_data is not None else None
                        if hasattr(nefc_live, "numpy"):
                            nefc_live = int(nefc_live.numpy().flat[0])
                        ncon_live = getattr(mjw_data, "ncon", None) if mjw_data is not None else None
                        if hasattr(ncon_live, "numpy"):
                            ncon_live = int(ncon_live.numpy().flat[0])
                        print(
                            f"[isaac_newton] live-mjw constraint sizing: njmax={njmax_live} "
                            f"nconmax={nconmax_live} nefc(used)={nefc_live} ncon(used)={ncon_live}. "
                            f"If nefc >= njmax, mjwarp silently truncates and contacts "
                            f"under-converge (wheel drift signature).",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[isaac_newton] live-mjw njmax probe failed: {exc!r}", flush=True)
                    # Cross-check: do contact, joint-limit, and joint-friction
                    # writes actually appear in the LIVE mjwarp model? The
                    # dumped MJCF is written from a DISPOSABLE solver
                    # (force_solver_init). The live wrapper builds its own
                    # SolverMuJoCo lazily on first step_sim. They can diverge.
                    try:
                        n_print = 0
                        # Pick a chassis_drive wheel by joint name and audit its
                        # mjwarp counterparts.
                        import mujoco as _mj  # noqa: PLC0415

                        live_mj = getattr(solver, "mj_model", None)
                        if live_mj is not None:
                            wheel_jname = "idx112_chassis_lwheel_front_joint2"
                            jid = _mj.mj_name2id(live_mj, _mj.mjtObj.mjOBJ_JOINT, wheel_jname)
                            if jid >= 0:
                                jnt_solref_lim = live_mj.jnt_solref[jid].tolist()
                                jnt_friction = (
                                    float(live_mj.dof_frictionloss[jid])
                                    if hasattr(live_mj, "dof_frictionloss")
                                    else None
                                )
                                print(
                                    f"[isaac_newton] live-mjw chassis_drive joint '{wheel_jname}' "
                                    f"jid={jid} jnt_solref_lim={jnt_solref_lim} "
                                    f"dof_frictionloss={jnt_friction}",
                                    flush=True,
                                )
                                n_print += 1
                                # ALL actuators bound to this joint (inc. duplicates).
                                for aid in range(live_mj.nu):
                                    trnid = int(live_mj.actuator_trnid[aid][0])
                                    if trnid != jid:
                                        continue
                                    aname = _mj.mj_id2name(live_mj, _mj.mjtObj.mjOBJ_ACTUATOR, aid) or f"<aid{aid}>"
                                    gainprm = live_mj.actuator_gainprm[aid][:3].tolist()
                                    biasprm = live_mj.actuator_biasprm[aid][:3].tolist()
                                    forcerange = (
                                        live_mj.actuator_forcerange[aid].tolist()
                                        if hasattr(live_mj, "actuator_forcerange")
                                        else None
                                    )
                                    ctrl = (
                                        float(getattr(solver.mj_data, "ctrl", [0.0] * live_mj.nu)[aid])
                                        if hasattr(solver, "mj_data")
                                        else None
                                    )
                                    print(
                                        f"[isaac_newton] live-mjw wheel actuator aid={aid} name='{aname}' "
                                        f"gainprm={gainprm} biasprm={biasprm} forcerange={forcerange} ctrl={ctrl}",
                                        flush=True,
                                    )
                                    n_print += 1
                            # First geom under the chassis_lwheel_front_link2 body.
                            for gid in range(live_mj.ngeom):
                                gname = _mj.mj_id2name(live_mj, _mj.mjtObj.mjOBJ_GEOM, gid) or ""
                                if "chassis_lwheel_front_link2" in gname:
                                    geom_solref = live_mj.geom_solref[gid].tolist()
                                    geom_friction = live_mj.geom_friction[gid].tolist()
                                    print(
                                        f"[isaac_newton] live-mjw chassis_drive geom '{gname[-80:]}' "
                                        f"gid={gid} solref={geom_solref} friction={geom_friction}",
                                        flush=True,
                                    )
                                    n_print += 1
                                    break
                        if n_print == 0:
                            print(
                                "[isaac_newton] live-mjw chassis_drive audit: no mj_model; "
                                "writes may have only landed in mjw_model not mj_model.",
                                flush=True,
                            )
                    except Exception as exc:  # noqa: BLE001
                        print(f"[isaac_newton] live-mjw chassis audit failed: {exc!r}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[isaac_newton] live-mjw audit failed: {exc!r}", flush=True)
                # Live-patch contact compliance into mjw_model. Writes to
                # ns.model.shape_material_* only reach the DISPOSABLE
                # solver that writes the dump; the live mjwarp solver
                # snapshotted those arrays at construction and now reads
                # ``mjw_model.geom_solref / geom_solimp / geom_friction``
                # directly. Same pattern as the equality stiffening.
                try:
                    self._stiffen_live_geom_contacts(ns)
                except Exception as exc:  # noqa: BLE001
                    self._logger.warn(f"[isaac_newton] _stiffen_live_geom_contacts failed (continuing): {exc!r}")
                self._eq_stiffen_done = True
            else:
                # Per-tick beacon to confirm step() is even running and
                # show why the stiffen is being deferred. Throttled so it
                # only fires while the wrapper is still lazy-building.
                if not getattr(self, "_eq_stiffen_beacon", False):
                    self._eq_stiffen_beacon = True
                    print(
                        f"[isaac_newton] equality-stiffen LIVE: deferred — "
                        f"ns={'set' if ns is not None else 'None'} "
                        f"solver={'set' if solver is not None else 'None'} "
                        f"mjw_model={'set' if (solver is not None and getattr(solver, 'mjw_model', None) is not None) else 'None'}. "
                        f"Will retry on next tick.",
                        flush=True,
                    )
        elif not self._eq_stiffen_done and ns is None:
            if not getattr(self, "_eq_stiffen_ns_none_beacon", False):
                self._eq_stiffen_ns_none_beacon = True
                print(
                    "[isaac_newton] equality-stiffen LIVE: deferred — "
                    "NewtonStage not yet acquired in step(). Will retry.",
                    flush=True,
                )
        return (time.monotonic() - t0) * 1000.0

    def startup(self, headless: bool) -> None:
        """Standard startup.

        The MJCF dump used to live here too — see ``step()``'s
        ``_mjcf_dump_done`` block for why it now fires after the
        first ``simulate()`` instead.  Short version: the live
        ``SolverMuJoCo`` is lazily built inside ``step_sim`` on
        the first step, and we want our disposable's
        ``_convert_to_mjc`` to read the same post-runtime-override
        ``model.joint_target_ke`` that the live solver reads,
        rather than dumping before the live solver has even
        consumed the model.

        ``_inject_mjcf_dump_path_pre_reset`` is kept (does nothing
        in practice — the wrapper builds its solver lazily, not
        during ``world.reset()``, so the cfg-time hook never
        fires) so future Newton releases that move solver
        construction back to ``world.reset()`` get picked up
        automatically.

        ``_apply_newton_cfg`` runs after reset because the
        substeps / iterations knobs route through cfg fields the
        wrapper reads per-step (not at construction).
        """
        self._inject_mjcf_dump_path_pre_reset()
        self._headless = bool(headless)
        self._stage_obj.startup_loop_setup(headless)
        self._apply_newton_cfg()

    def _inject_mjcf_dump_path_pre_reset(self) -> None:
        """Set ``cfg.solver_cfg.save_to_mjcf`` so ``world.reset()``'s
        downstream ``SolverMuJoCo(...)`` call writes the XML at
        construction.

        This must run BEFORE ``startup_loop_setup`` (which fires
        ``world.reset()``).  Skipped quietly if:

          * NewtonStage isn't acquirable yet (extension not loaded /
            warm).  Operator gets a one-line warn but startup proceeds.
          * ``cfg.solver_cfg`` doesn't expose ``save_to_mjcf`` (older
            wrapper versions).  We log the attr list so the version
            skew is obvious.
          * No ``runtime_usd_dump_path`` was provided — nowhere to
            write to.
        """
        import os
        import sys

        if not self._mjcf_out_path:
            msg = (
                "[isaac_newton] MJCF dump skipped: runtime_usd_dump_path "
                "wasn't provided to IsaacNewtonEngine; nowhere to write "
                "robot_runtime.xml."
            )
            print(msg, file=sys.stderr, flush=True)
            self._logger.warn(msg)
            return

        scene_dir = os.path.dirname(self._mjcf_out_path)
        try:
            os.makedirs(scene_dir, exist_ok=True)
        except OSError as exc:
            msg = f"[isaac_newton] MJCF dump skipped: cannot create / " f"write scene dir {scene_dir!r}: {exc}."
            print(msg, file=sys.stderr, flush=True)
            self._logger.warn(msg)
            return

        ns = self._get_newton_stage()
        if ns is None:
            msg = (
                "[isaac_newton] MJCF dump pre-reset injection: NewtonStage "
                "not acquirable before world.reset(). "
                "robot_runtime.xml will be skipped this run."
            )
            print(msg, file=sys.stderr, flush=True)
            self._logger.warn(msg)
            return
        cfg = getattr(ns, "cfg", None)
        sc = getattr(cfg, "solver_cfg", None) if cfg is not None else None
        if sc is None or not hasattr(sc, "save_to_mjcf"):
            attrs = sorted(vars(sc).keys()) if sc is not None else "<none>"
            msg = (
                f"[isaac_newton] MJCF dump pre-reset injection: "
                f"cfg.solver_cfg has no ``save_to_mjcf`` attr — wrapper "
                f"version skew?  Available solver_cfg fields: {attrs}.  "
                f"Skipped."
            )
            print(msg, file=sys.stderr, flush=True)
            self._logger.warn(msg)
            return

        old = getattr(sc, "save_to_mjcf", None)
        sc.save_to_mjcf = self._mjcf_out_path
        msg = (
            f"[isaac_newton] MJCF dump: cfg.solver_cfg.save_to_mjcf "
            f"{old!r} -> {self._mjcf_out_path!r}.  Disposable "
            f"SolverMuJoCo at startup-end will write the XML; the "
            f"cfg setting is also kept so any future wrapper-side "
            f"lazy build picks it up too."
        )
        print(msg, flush=True)
        self._logger.info(msg)

    def _force_solver_init(self) -> None:
        """Write ``robot_runtime.xml`` by spawning a disposable
        ``SolverMuJoCo`` against the live ``ns.model``.

        Why this approach (after exhausting the others):

          * Cfg-injection (``ns.cfg.solver_cfg.save_to_mjcf = path``
            BEFORE ``world.reset()``) sets the field but the wrapper
            built its solver BEFORE our startup hook even runs — it
            happens during ``IsaacSimStage`` construction, well
            upstream of ``IsaacNewtonEngine.startup``.  Probes A
            (``SolverMuJoCo.__init__``) and B
            (``NewtonStage._get_solver``) installed in
            ``_inject_mjcf_dump_path_pre_reset`` BOTH stayed silent
            during a full run while ``ns.solver`` was already
            populated by then — confirming the construction is
            upstream.
          * Forcing rebuild via ``ns.initialized = False`` +
            ``initialize_newton`` works but reallocates the live
            mjwarp model on the same CUDA stream — earlier we saw
            this throw ``cudaErrorIllegalAddress`` in
            ``apply_mjc_control_kernel`` once the engine started
            stepping.
          * In-place ``solver._convert_to_mjc(model, target_filename
            =path)`` writes the XML but mutates the live solver's
            ``mj_model``/``mjw_data`` — same crash class.

        Disposable construction:
          * ``newton.solvers.SolverMuJoCo(model, save_to_mjcf=path,
            separate_worlds=False)`` runs the conversion on a fresh
            instance, writes the XML inside ``__init__``, and
            allocates ITS OWN mjwarp Warp arrays — entirely separate
            from the live solver's.  Drop the reference and Python's
            GC frees the disposable's arrays.
          * The live ``ns.solver`` is untouched.

        Cost: a few hundred ms once at startup, plus transient GPU
        memory for the second mj_model/mjw_data (released when the
        disposable is GC'd).  Newton-standalone pays the same cost
        in its native dump path.
        """
        import sys

        if not self._mjcf_out_path:
            return

        ns = self._get_newton_stage()
        if ns is None:
            msg = "[isaac_newton] force_solver_init: NewtonStage not " "acquirable; skipping MJCF dump."
            print(msg, file=sys.stderr, flush=True)
            self._logger.warn(msg)
            return

        model = getattr(ns, "model", None)
        if model is None:
            msg = (
                "[isaac_newton] force_solver_init: ns.model is None; "
                "the wrapper hasn't built the Newton model yet, so the "
                "disposable solver can't run.  Skipping MJCF dump."
            )
            print(msg, file=sys.stderr, flush=True)
            self._logger.warn(msg)
            return

        try:
            # Diagnostic: log a summary of ``model.joint_target_ke`` /
            # ``joint_target_kd`` / ``joint_effort_limit`` *right before*
            # the disposable solver runs so we can tell from the log
            # alone whether the runtime gains have actually landed in
            # the Newton model.  ``_convert_to_mjc`` reads from these
            # arrays per-DOF when it authors the actuator gainprm /
            # biasprm / forcerange — so if the values here are still
            # the URDF importer defaults, the dumped MJCF will show
            # the same defaults regardless of what ``physics_params.yaml``
            # says.
            try:
                import numpy as _np

                ke = model.joint_target_ke.numpy()
                kd = model.joint_target_kd.numpy()
                el = model.joint_effort_limit.numpy()
                jlabels = list(getattr(model, "joint_label", []) or [])
                # Pick a few representative joints to spot-check
                wanted = ("body_joint1", "arm_l_joint2", "arm_l_joint7", "gripper_l_inner_joint1")
                lines: List[str] = []
                for w in wanted:
                    for i, lbl in enumerate(jlabels):
                        if w in lbl:
                            lines.append(
                                f"      {lbl[-50:]:50}  ke={ke[i]:>10.2f}  kd={kd[i]:>8.2f}  effort={el[i]:>9.2f}"
                            )
                            break
                print(
                    "[isaac_newton] force_solver_init: "
                    "model.joint_target_ke summary BEFORE disposable build:\n"
                    f"      ke   min={ke.min():.2f}  max={ke.max():.2f}  mean={ke.mean():.2f}\n"
                    f"      kd   min={kd.min():.2f}  max={kd.max():.2f}  mean={kd.mean():.2f}\n"
                    f"      eff  min={el.min():.2f}  max={el.max():.2f}  mean={el.mean():.2f}\n" + "\n".join(lines),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[isaac_newton] force_solver_init: pre-dump model summary failed ({exc!r})",
                    flush=True,
                )
            print(
                f"[isaac_newton] force_solver_init: spawning disposable "
                f"SolverMuJoCo against ns.model to write "
                f"{self._mjcf_out_path}…",
                flush=True,
            )
            import newton.solvers

            disposable = newton.solvers.SolverMuJoCo(
                model,
                save_to_mjcf=self._mjcf_out_path,
                separate_worlds=False,
            )
            # Drop the reference immediately so the disposable's
            # mjwarp arrays are freed at GC time, not held for the
            # rest of the run.
            del disposable

            import os

            wrote = os.path.isfile(self._mjcf_out_path)
            size = os.path.getsize(self._mjcf_out_path) if wrote else 0
            msg = (
                f"[isaac_newton] force_solver_init: disposable solver "
                f"finished, file {'present' if wrote else 'MISSING'} "
                f"({size} bytes)."
            )
            print(msg, flush=True)
            self._logger.info(msg)
            if wrote:
                # Post-patch: stiffen the mimic equality constraints
                # that Newton's mjwarp converter emits with default
                # ``solref="0.02 1"``.  Newton's mimic-emit path at
                # ``solver_mujoco.py:4732`` doesn't read the
                # ``eq_solref`` custom attribute (line 621
                # registration is only consumed for non-mimic
                # equality types), so we can't author this through
                # the assemble pipeline.  Without the patch, gripper
                # followers swing 0.1+ rad under 0.5 rad arm-joint3
                # sweep — even though the master holds at 0.0001 rad
                # — because the constraint solver can't enforce
                # ``q_follower = ±q_master`` against the inertial
                # reaction torque from the arm chain.  ``solref
                # "-10000 -100"`` (direct stiffness=1e4 N·m/rad,
                # damping=100 N·m·s/rad on the equality) cuts
                # follower amplitude in half without numerical
                # instability at the 2ms timestep.  Going much
                # stiffer (1e6 / 1e4) explodes at this dt.
                #
                # We do this in TWO places so the dumped MJCF and
                # the LIVE mjwarp solver land at the same physical
                # behaviour — otherwise ``test_robot_xml_dynamic.py``
                # would predict one thing (against the patched
                # dump) and the runtime would do another (against
                # the un-patched live solver):
                #
                #   1. Text-edit the dumped MJCF file so anyone
                #      reading or testing against it sees the same
                #      values the runtime is using.
                #   2. Mutate the LIVE solver's ``mjw_model.eq_solref``
                #      / ``eq_solimp`` GPU arrays so the running
                #      simulation actually has the stiff constraints.
                #
                # If either path is skipped, the dynamic-tool
                # predictions stop matching the live runtime — a
                # real gotcha we hit before.
                # Wrapper-specific step: mutate the LIVE solver's
                # ``mjw_model.eq_solref`` / ``eq_solimp`` arrays so
                # the running simulation enforces whatever the
                # ``usd_drive_api.gripper.mimic_eq_*`` block in
                # ``physics_params.yaml`` configures.  This is the
                # GPU-array side of the equality stiffening — the
                # MJCF text side is handled by
                # ``augment_mjcf_gripper_from_params`` inside the
                # pipeline below, so live and dump stay in sync.
                # Standalone (mjwarp adapter) does the equivalent
                # via ``MuJoCoWarpAdapter._apply_mimic_eq_solref``;
                # this wrapper-side call exists because the wrapper
                # builds its solver lazily and never goes through
                # the adapter path.
                #
                # The actual GPU-array patch fires from ``step()``
                # once ``ns.solver.mjw_model`` materialises — the
                # disposable solver above is a separate object and
                # was just GC'd, and the wrapper's own solver is
                # built lazily on the first ``step_sim``. Calling
                # the live-patch here would silently no-op against
                # ``ns.solver is None``. See the
                # ``_eq_stiffen_done`` block in ``step()``.

                # Single unified MJCF post-process pipeline.  Every
                # step runs in its own ``try / except`` inside the
                # helper, so one step's failure no longer cascades
                # to the rest (which is what produced the
                # un-normalised body names on a recent regen).  Same
                # helper drives the standalone mjwarp adapter's
                # ``build_solver``; both engine paths converge on
                # this single source of truth for "what does a
                # well-formed dump look like?".
                from common.mjcf_postprocess import apply_mjcf_postprocess_pipeline

                apply_mjcf_postprocess_pipeline(
                    mjcf_path=self._mjcf_out_path,
                    sim_substeps=int(self._sim_substeps) if self._sim_substeps else 1,
                    physics_hz=float(self._physics_hz),
                    physics_params=self._params,
                    qpos=None,  # wrapper has no post-init-pose hook today
                    logger=self._logger,
                )
        except Exception as exc:  # noqa: BLE001
            import traceback

            tb = traceback.format_exc()
            msg = (
                f"[isaac_newton] force_solver_init: disposable "
                f"SolverMuJoCo construction raised {exc!r}.  MJCF dump "
                f"skipped this run.  Trace:\n{tb}"
            )
            print(msg, file=sys.stderr, flush=True)
            self._logger.warn(msg)

    def _stiffen_mimic_equalities(
        self,
        path: str,
        solref: str = "-10000 -100",
        solimp: str = "0.95 0.99 0.001 0.5 2",
        name_substring: str = "gripper",
    ) -> None:
        """Post-process a freshly-dumped MJCF to stiffen mimic
        equality constraints.

        Why this exists
        ---------------

        Newton's mjwarp converter
        (``newton/_src/solvers/mujoco/solver_mujoco.py:4732`` in
        ``newton==1.15.0.dev20260526``) emits ``mjEQ_JOINT`` mimic
        equalities without setting ``solref`` / ``solimp``, so they
        fall through to MuJoCo's default ``solref="0.02 1"``.  That's
        too soft for the swiftpicker grippers under reaction torque
        from arm motion: a 0.5 rad arm-elbow sweep drives gripper
        followers ±0.15 rad while the master holds at < 0.001 rad
        — verified via ``test_robot_xml_dynamic.py
        --cross-impact-mode sweep``.

        Newton does register an ``eq_solref`` custom attribute
        (``solver_mujoco.py:621``, USD ``mjc:solref`` on the equality
        prim) but the MIMIC emit path doesn't consume it.  So we
        can't author per-mimic-equality stiffness through the
        assemble pipeline; the only intervention point is a textual
        patch on the MJCF after the disposable solver writes it.

        Stiffness choice
        ----------------

        ``solref="-10000 -100"`` (direct stiffness=1e4 N·m/rad,
        damping=100 N·m·s/rad on the equality) cuts follower
        amplitude in half (0.15 rad → 0.06 rad under the same arm
        sweep) without numerical instability at the wrapper's 2 ms
        substep.  Going much stiffer (``-1e6 -1e4``) explodes the
        integrator at this dt.  If the wrapper ever lowers the
        physics step to 1 ms, we can re-tune toward ``-50000 -500``
        for further tightening.

        Scope
        -----

        Only equalities whose ``joint1`` name contains
        ``name_substring`` are patched (``"gripper"`` by default) —
        keeps the patch conservative so future loop joints on
        non-gripper sub-chains don't get accidentally over-
        stiffened.
        """
        import re
        import sys

        try:
            with open(path) as f:
                txt = f.read()
        except OSError as exc:
            print(f"[isaac_newton] equality-stiffen: read failed: {exc}", file=sys.stderr, flush=True)
            return

        pattern = re.compile(
            r'<joint\s+(?:name="[^"]*"\s+)?joint1="[^"]*'
            + re.escape(name_substring)
            + r'[^"]*"\s+joint2="[^"]*"\s+polycoef="[^"]*"\s*/>'
        )

        def _inject(m: "re.Match[str]") -> str:
            s = m.group(0)
            if "solref=" in s:
                return s
            return s.replace("/>", f' solref="{solref}" solimp="{solimp}"/>')

        new_txt, n_subs = pattern.subn(_inject, txt)
        if n_subs == 0:
            print(
                f"[isaac_newton] equality-stiffen: no mimic equalities matched "
                f"name_substring={name_substring!r}; nothing to patch.",
                flush=True,
            )
            return
        try:
            with open(path, "w") as f:
                f.write(new_txt)
        except OSError as exc:
            print(f"[isaac_newton] equality-stiffen: write failed: {exc}", file=sys.stderr, flush=True)
            return

        msg = (
            f"[isaac_newton] equality-stiffen: patched {n_subs} mimic "
            f"equality constraint(s) with solref={solref!r} "
            f"solimp={solimp!r} (name_substring={name_substring!r})."
        )
        print(msg, flush=True)
        self._logger.info(msg)

    def _stiffen_live_mimic_equalities(
        self,
        ns: Any,
        solref: Tuple[float, float] = (-10000.0, -100.0),
        solimp: Tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0),
        name_substring: str = "gripper",
    ) -> None:
        """Mutate the LIVE mjwarp solver's ``eq_solref`` / ``eq_solimp``
        wp.array values for gripper mimic equalities.

        Why both this AND ``_stiffen_mimic_equalities`` exist
        ----------------------------------------------------

        The text-patch version updates the DUMPED MJCF file — so
        anyone running ``test_robot_xml_dynamic.py`` against the
        dump sees the stiff equality.  This function updates the
        LIVE solver — so the running simulation actually feels the
        stiff equality.  Both have to run, in this order:

          1. Dispose solver writes raw MJCF (default soft solref).
          2. Text-patch the MJCF file.
          3. Live-patch the live solver's wp.arrays.

        If we only did (2), the dynamic-tool predictions would
        diverge from the runtime — a real gotcha the user hit:
        "dynamic tool says gripper follower swings 0.04 rad, but
        runtime is 0.11 rad".  The MJCF was lying because we
        edited it after the live solver had already taken its
        snapshot.

        Selection
        ---------

        We identify gripper equalities by looking at the
        equality's ``joint1`` id and checking whether the joint's
        name (from ``ns.model.joint_label``) contains
        ``name_substring``.  Same scoping as the text-patch
        version — only gripper mimics get stiffened.

        Implementation
        --------------

        ``ns.solver.mjw_model.eq_solref`` is a ``wp.array(dtype=
        wp.vec2)`` with shape (nworld * neq,) in flattened form.
        We copy to numpy, modify the gripper rows, copy back via
        ``.assign(...)`` which Warp uses for host→device upload.
        Same for ``eq_solimp`` (vec5).
        """
        import numpy as np
        import sys

        if ns is None:
            print("[isaac_newton] equality-stiffen LIVE: ns is None; skipping.", flush=True)
            return
        solver = getattr(ns, "solver", None)
        model = getattr(ns, "model", None)
        if solver is None or model is None:
            print(
                f"[isaac_newton] equality-stiffen LIVE: ns.solver={solver!r} "
                f"ns.model={'present' if model is not None else 'None'}; skipping.",
                flush=True,
            )
            return
        mjw = getattr(solver, "mjw_model", None)
        if mjw is None:
            print(
                "[isaac_newton] equality-stiffen LIVE: solver has no " "``mjw_model`` (XPBD solver?); skipping.",
                flush=True,
            )
            return

        # Pull the eq → joint1 mapping from the LIVE solver's mjw_model.
        # mjwarp lays out eq arrays per-world: ``eq_solref`` and
        # ``eq_solimp`` are shape ``(nworld, neq, NREF/NIMP)`` (see
        # mujoco_warp/_src/types.py: ``array("*", "neq", wp.vec2)``);
        # ``eq_obj1id`` is flat ``(neq,)``. We patch every world's
        # row for the matching equality so the change holds under
        # eventual multi-world replication too.
        try:
            eq_obj1id = mjw.eq_obj1id.numpy()
            eq_solref_np = mjw.eq_solref.numpy().copy()
            eq_solimp_np = mjw.eq_solimp.numpy().copy()
        except Exception as exc:  # noqa: BLE001
            print(
                f"[isaac_newton] equality-stiffen LIVE: could not read " f"mjw_model eq_* arrays ({exc}); skipping.",
                flush=True,
            )
            return
        # Defensive shape probe for the log so any future Newton layout
        # change announces itself instead of crashing silently.
        print(
            f"[isaac_newton] equality-stiffen LIVE: eq_obj1id.shape={eq_obj1id.shape} "
            f"eq_solref.shape={eq_solref_np.shape} eq_solimp.shape={eq_solimp_np.shape}",
            flush=True,
        )

        # Joint labels live on the Newton model (``model.joint_label``)
        # as a Python list of prim-path-like strings.  mjwarp's
        # eq_obj1id refers to MUJOCO joint ids — but Newton's converter
        # keeps the joint name list in mjwarp-emit order accessible
        # via ``mjw_model.joint`` (a wrapped MjModel-like view).  We
        # try ``mj_id2name`` on the mjwarp model first, then fall back
        # to Newton's label list with a direct index match.
        joint_names: List[str] = []
        try:
            import mujoco

            # The disposable solver kept its ``mj_model`` on
            # ``solver.mj_model`` after construction.  Reuse it for
            # name resolution since mjwarp keeps the same id space.
            mj_model = getattr(solver, "mj_model", None)
            if mj_model is not None:
                joint_names = [
                    mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(mj_model.njnt)
                ]
        except Exception:  # noqa: BLE001
            joint_names = []
        if not joint_names:
            # Fall back to Newton's label list.  Not always perfectly
            # aligned with mjwarp's joint ordering (Newton omits some
            # joints that don't make it to MuJoCo, e.g. fixed joints
            # collapsed during conversion) but close enough for our
            # gripper-substring check.
            try:
                joint_names = list(model.joint_label)
            except Exception:  # noqa: BLE001
                pass

        n_patched = 0
        # mjw eq_solref / eq_solimp are (nworld, neq, …). Patch every
        # world row for each matching equality so multi-world setups
        # (none yet, but cheap insurance) stay coherent.
        if eq_solref_np.ndim < 2 or eq_solimp_np.ndim < 2:
            print(
                f"[isaac_newton] equality-stiffen LIVE: unexpected layout "
                f"eq_solref.ndim={eq_solref_np.ndim} eq_solimp.ndim={eq_solimp_np.ndim} "
                f"(expected >=2: nworld × neq × ref). Skipping to avoid a wrong write.",
                flush=True,
            )
            return
        solref_arr = np.asarray(solref, dtype=eq_solref_np.dtype)
        solimp_arr = np.asarray(solimp, dtype=eq_solimp_np.dtype)
        for i, jid in enumerate(eq_obj1id):
            jid_int = int(jid)
            if jid_int < 0 or jid_int >= len(joint_names):
                continue
            jname = joint_names[jid_int]
            if name_substring not in jname:
                continue
            eq_solref_np[:, i] = solref_arr
            eq_solimp_np[:, i] = solimp_arr
            n_patched += 1

        if n_patched == 0:
            print(
                f"[isaac_newton] equality-stiffen LIVE: no equalities "
                f"matched name_substring={name_substring!r}; skipping.",
                flush=True,
            )
            return

        try:
            mjw.eq_solref.assign(eq_solref_np)
            mjw.eq_solimp.assign(eq_solimp_np)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[isaac_newton] equality-stiffen LIVE: wp.assign "
                f"failed ({exc}); equality values may not have "
                f"actually been updated on the GPU.  Check that the "
                f"simulation hasn't started stepping (live mutations "
                f"during a CUDA graph capture sometimes silently "
                f"no-op).",
                file=sys.stderr,
                flush=True,
            )
            return

        msg = (
            f"[isaac_newton] equality-stiffen LIVE: patched "
            f"{n_patched} mimic equality constraint(s) in the LIVE "
            f"mjwarp solver with solref={solref!r} solimp={solimp!r} "
            f"(name_substring={name_substring!r}).  Runtime behaviour "
            f"now matches the dumped MJCF."
        )
        print(msg, flush=True)
        self._logger.info(msg)

    def _stiffen_live_geom_contacts(self, ns: Any) -> None:
        """Mirror ``_apply_per_class_contact`` into the LIVE mjwarp solver.

        Why this exists separately from ``_apply_per_class_contact``
        ----------------------------------------------------------
        ``_apply_per_class_contact`` writes to ``ns.model.shape_material_*``
        and ``ns.model.geom_solimp``. Newton's mjwarp converter reads
        those at SolverMuJoCo construction time, then snapshots them
        into ``mjw_model.geom_solref / geom_solimp / geom_friction``.
        Subsequent writes to ``ns.model`` don't propagate.

        The disposable solver in ``_force_solver_init`` re-reads
        ``ns.model`` at MJCF dump time, so the dumped XML correctly
        shows ``solref="0.005 5"``. But the LIVE solver was built
        earlier and still has Newton's defaults — proven empirically
        by the live-mjw audit:
            dump:  geom solref="0.005 5"  friction="1.5"
            live:  geom_solref=[0.02, 1.0]  geom_friction=[1.0, …]

        This function patches the live mjw_model arrays directly,
        same pattern as ``_stiffen_live_mimic_equalities``.

        Per-shape, classified by ``classify_shape`` from the same
        table the dump-time path uses, so live and dump stay in sync.
        """
        if ns is None:
            return
        solver = getattr(ns, "solver", None)
        model = getattr(ns, "model", None)
        if solver is None or model is None:
            print(
                f"[isaac_newton] geom-contact LIVE: ns.solver={solver!r} "
                f"ns.model={'present' if model is not None else 'None'}; skipping.",
                flush=True,
            )
            return
        mjw = getattr(solver, "mjw_model", None)
        if mjw is None:
            print("[isaac_newton] geom-contact LIVE: solver has no mjw_model; skipping.", flush=True)
            return

        try:
            from common.object_classification import (  # noqa: PLC0415
                ALL_OBJECT_KINDS,
                MJC_CONTACT_DEFAULTS,
                classify_shape,
            )
            import newton  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            print(f"[isaac_newton] geom-contact LIVE: import failed: {exc!r}", flush=True)
            return

        def _solref_to_ke_kd(solref: tuple) -> tuple:
            timeconst, dampratio = float(solref[0]), float(solref[1])
            if timeconst <= 0.0 or dampratio <= 0.0:
                return 2500.0, 100.0
            kd = 2.0 / timeconst
            ke = (kd / (2.0 * dampratio)) ** 2
            return ke, kd

        params = {k: dict(MJC_CONTACT_DEFAULTS[k]) for k in ALL_OBJECT_KINDS}

        # Build the (kind per shape) list using the same classifier as
        # the dump-time writer. Newton's shape ordering matches mjwarp's
        # geom ordering one-to-one (Newton emits geoms in shape order).
        try:
            n_shapes = int(getattr(model, "shape_count", 0) or 0)
            if n_shapes == 0:
                print("[isaac_newton] geom-contact LIVE: shape_count=0; skipping.", flush=True)
                return
            shape_types = model.shape_type.numpy()
            shape_bodies = model.shape_body.numpy()
            shape_labels = list(getattr(model, "shape_label", []) or [])
            body_labels = list(getattr(model, "body_label", []) or [])
            JT_PLANE = int(newton.GeoType.PLANE)
            robot_prefix = self.robot_prefix or ""

            geom_solref_np = mjw.geom_solref.numpy().copy()
            geom_solimp_np = mjw.geom_solimp.numpy().copy() if hasattr(mjw, "geom_solimp") else None
            geom_friction_np = mjw.geom_friction.numpy().copy() if hasattr(mjw, "geom_friction") else None

            ngeom_live = geom_solref_np.shape[1] if geom_solref_np.ndim >= 2 else geom_solref_np.shape[0]
            n_to_iter = min(n_shapes, int(ngeom_live))

            class_counts: Dict[str, int] = {k: 0 for k in ALL_OBJECT_KINDS}
            class_ke_kd: Dict[str, tuple] = {}
            n_patched = 0

            mu_target = 1.5  # matches scene-wide _ROBOT_KU broadcast in standalone

            for i in range(n_to_iter):
                b = int(shape_bodies[i]) if i < len(shape_bodies) else -1
                kind = classify_shape(
                    shape_label=shape_labels[i] if i < len(shape_labels) else "",
                    body_index=b,
                    body_label=body_labels[b] if 0 <= b < len(body_labels) else None,
                    shape_type_int=int(shape_types[i]),
                    robot_prefix=robot_prefix,
                    plane_geo_type=JT_PLANE,
                )
                if kind not in class_ke_kd:
                    class_ke_kd[kind] = _solref_to_ke_kd(params[kind]["solref"])
                # mjwarp consumes solref directly (timeconst, dampratio)
                solref_v = params[kind]["solref"]
                if geom_solref_np.ndim == 3:
                    geom_solref_np[:, i] = solref_v
                elif geom_solref_np.ndim == 2:
                    geom_solref_np[i] = solref_v
                if geom_solimp_np is not None:
                    solimp_v = params[kind]["solimp"]
                    if geom_solimp_np.ndim == 3:
                        geom_solimp_np[:, i] = solimp_v
                    elif geom_solimp_np.ndim == 2:
                        geom_solimp_np[i] = solimp_v
                if geom_friction_np is not None:
                    fr = (mu_target, 0.005, 0.0001)
                    if geom_friction_np.ndim == 3:
                        geom_friction_np[:, i] = fr
                    elif geom_friction_np.ndim == 2:
                        geom_friction_np[i] = fr
                class_counts[kind] += 1
                n_patched += 1

            mjw.geom_solref.assign(geom_solref_np.astype(np.float32))
            if geom_solimp_np is not None:
                mjw.geom_solimp.assign(geom_solimp_np.astype(np.float32))
            if geom_friction_np is not None:
                mjw.geom_friction.assign(geom_friction_np.astype(np.float32))

            # Cone live-patch: setting cfg.solver_cfg.cone='pyramidal'
            # at startup configures the cfg, but the LIVE mjwarp model
            # snapshotted from cfg at SolverMuJoCo build time and now
            # reads ``mjw_model.opt.cone`` directly. The audit shows
            # cone=1 (elliptic) in the live model even after our cfg
            # write. Patch the live scalar here too.
            cone_set = False
            try:
                import mujoco as _mj_for_cone  # noqa: PLC0415

                target_cone = int(_mj_for_cone.mjtCone.mjCONE_PYRAMIDAL)
                opt = getattr(mjw, "opt", None)
                cone_attr = getattr(opt, "cone", None) if opt is not None else None
                if cone_attr is not None:
                    if hasattr(cone_attr, "fill_"):
                        cone_attr.fill_(target_cone)
                        cone_set = True
                    elif hasattr(cone_attr, "assign"):
                        cone_attr.assign(np.array([target_cone], dtype=np.int32))
                        cone_set = True
                    else:
                        # Plain scalar attribute
                        try:
                            opt.cone = target_cone
                            cone_set = True
                        except Exception:  # noqa: BLE001
                            pass
            except Exception as exc:  # noqa: BLE001
                print(f"[isaac_newton] geom-contact LIVE: cone patch failed: {exc!r}", flush=True)

            msg = (
                f"[isaac_newton] geom-contact LIVE: patched {n_patched} geom(s) in mjw_model "
                f"(class counts={class_counts}, mu broadcast to {mu_target}, cone_set={cone_set}). "
                f"Runtime contact compliance now matches dumped MJCF."
            )
            print(msg, flush=True)
            self._logger.info(msg)
        except Exception as exc:  # noqa: BLE001
            print(f"[isaac_newton] geom-contact LIVE: write failed: {exc!r}", flush=True)

    def tick_extras(self) -> None:
        """Drive the wrapper's apply_action chain in headless mode.

        The wrapper completes ``ArticulationView.apply_action`` writes
        via something Kit's stage-update event drives (we narrowed it
        down by elimination — pre_step callbacks aren't subscribed,
        ``physx_sim.simulate()`` already routes through
        ``NewtonSimulationFunctions.simulate``, yet direct writes to
        ``ns.control.joint_target_pos`` + ``sync_position_targets``
        don't produce actuation). ``simulation_app.update()`` is what
        wires the rest of the chain together, so in headless mode we
        call it here — gated by :attr:`_cmd_pending` so the ~9 ms Kit
        update fires only when :meth:`apply_commands` actually saw a
        new command.

        Body-transform writeback is handled by
        :meth:`get_body_transforms` reading ``state_0.body_q`` directly,
        so we don't need ``simulation_app.update()`` for that —
        rendering keeps following the robot even on ticks where the
        flag is clear.
        """
        if not self._headless or self._simulation_app is None:
            return
        if not self._cmd_pending:
            return
        self._cmd_pending = False
        self._simulation_app.update()

    def apply_commands(self, cmd_positions, cmd_4ws_steer_pos, cmd_4ws_drive_vel, cmd_4ws_stamp) -> None:
        """Delegate to the wrapper's apply_action chain.

        In headless mode the run loop never calls
        ``simulation_app.update()`` itself, so :meth:`tick_extras` does
        it on our behalf — but only on ticks where the dirty flag below
        is set, so the ~9 ms Kit update doesn't burn every tick.

        ``pop_commands`` in ``realtime_buffer.cpp`` does NOT clear the
        buffer on read — it just snapshots the current state — so the
        dicts handed in look identical tick after tick once the first
        command lands. We detect *change* via dict equality (microsecond
        cost for ~70 floats), not non-emptiness, so step-mode wbc
        publishing the same goal every 50 ms only triggers one Kit
        update per goal change, not 20/s.
        """
        if cmd_positions and cmd_positions != self._last_cmd_positions:
            self._cmd_pending = True
            # Copy to detach from the snapshot C++ hands us.
            self._last_cmd_positions = dict(cmd_positions)
        if cmd_4ws_stamp and cmd_4ws_stamp != self._last_cmd_4ws_stamp:
            self._cmd_pending = True
            self._last_cmd_4ws_stamp = cmd_4ws_stamp
        self._stage_obj.apply_commands(
            cmd_positions=cmd_positions,
            cmd_4ws_steer_pos=cmd_4ws_steer_pos,
            cmd_4ws_drive_vel=cmd_4ws_drive_vel,
            cmd_4ws_stamp=cmd_4ws_stamp,
        )

    def get_joint_states(self):
        """Bypass USD-attribute readback (Newton doesn't write those)
        and read joint state from the articulation handle. See
        ``IsaacSimStage.get_joint_states`` for the full rationale.

        The run loop checks for this method via
        ``hasattr(sim, "get_joint_states")`` and prefers it over the
        module-level ``snapshot_joint_states`` (which would silently
        return zeros under Newton).
        """
        return self._stage_obj.get_joint_states()

    def get_body_transforms(self) -> Tuple[np.ndarray, List[str]]:
        """Read body poses directly from Newton's ``state_0.body_q``.

        Why this override exists
        ------------------------
        The default :meth:`PhysicsEngine.get_body_transforms` reads each
        body's ``xformOp:translate`` / ``xformOp:orient`` from the USD
        stage (see ``kit/stage.py::snapshot_body_transforms``). PhysX
        writes those USD attributes back every tick, so the snapshot
        always sees fresh data. The Newton wrapper does NOT — it pushes
        state into Fabric (``NewtonStage.update_fabric_attrs``) via
        usdrt, not into the authoring ``pxr.Usd`` stage that
        ``snapshot_body_transforms`` reads. As a result the USD body
        attributes stay at their initial values forever, ``/tf_render``
        keeps publishing the init pose, and the OVRtx renderer (which
        applies the channel as local xform ops onto its own stage)
        shows a frozen robot even though physics is advancing and
        ``/joint_states`` updates correctly.

        Source of truth here is ``state_0.body_q`` (per-body world-space
        ``wp.transformf`` = ``(x, y, z, qx, qy, qz, qw)``); the mapping
        from Newton body index to USD prim path comes from
        ``model.body_label``.

        World → local conversion
        ------------------------
        Newton stores world-space body poses, but the renderer's
        ``_apply_xform`` (``isaacsim_render.py``) writes the received
        translation/orient as the prim's LOCAL ``xformOp:translate`` /
        ``xformOp:orient`` and lets USD's hierarchy compose them.
        So we have to deliver each body's pose relative to its USD
        parent (whatever the next-up entry in the chain is), exactly
        as PhysX's USD writeback would. For each body we precompute the
        parent's index in ``body_label`` once, then per-tick compose
        ``local = parent_world⁻¹ · body_world``. Bodies whose USD parent
        is not a Newton body (e.g. a static Xform container or the
        scene root) fall through to world == local — same convention as
        the existing newton-standalone path.
        """
        body_paths = self._stage_obj.body_paths
        if not body_paths:
            return np.zeros((0, 7), dtype=np.float64), []

        ns = self._get_newton_stage()
        if ns is None or getattr(ns, "state_0", None) is None or getattr(ns, "model", None) is None:
            return np.zeros((0, 7), dtype=np.float64), []

        body_q_buf = getattr(ns.state_0, "body_q", None)
        if body_q_buf is None:
            return np.zeros((0, 7), dtype=np.float64), []

        try:
            body_q = np.asarray(body_q_buf.numpy())
        except Exception as exc:
            if not self._body_xform_warned:
                self._logger.warn(f"[isaac_newton] get_body_transforms: body_q read failed: {exc}")
                self._body_xform_warned = True
            return np.zeros((0, 7), dtype=np.float64), []

        if body_q.size == 0:
            return np.zeros((0, 7), dtype=np.float64), []

        # Build the (body_path → newton_idx, parent_newton_idx) table on
        # first use. ``model.body_label`` is the canonical newton-index
        # → USD-path list; we invert it and walk each USD parent chain
        # to find the nearest ancestor that is also a Newton body.
        table = self._ensure_body_xform_table(ns)
        if table is None:
            return np.zeros((0, 7), dtype=np.float64), []
        idx_arr, parent_idx_arr, kept_paths = table

        n_q = int(body_q.shape[0])
        out = np.zeros((len(kept_paths), 7), dtype=np.float64)
        for row, (i, p) in enumerate(zip(idx_arr, parent_idx_arr)):
            if i >= n_q:
                continue
            # Newton: (px, py, pz, qx, qy, qz, qw)
            q_world = body_q[i]
            px, py, pz = float(q_world[0]), float(q_world[1]), float(q_world[2])
            qx, qy, qz, qw = float(q_world[3]), float(q_world[4]), float(q_world[5]), float(q_world[6])

            if p >= 0 and p < n_q:
                qp = body_q[p]
                ppx, ppy, ppz = float(qp[0]), float(qp[1]), float(qp[2])
                pqx, pqy, pqz, pqw = float(qp[3]), float(qp[4]), float(qp[5]), float(qp[6])
                # parent⁻¹ · world  (unit quats: inverse = conjugate)
                # local_p = R(parent⁻¹) · (p - p_parent)
                dx, dy, dz = px - ppx, py - ppy, pz - ppz
                lx, ly, lz = _quat_rotate(-pqx, -pqy, -pqz, pqw, dx, dy, dz)
                # local_q = parent⁻¹ · q
                lqx, lqy, lqz, lqw = _quat_mul(-pqx, -pqy, -pqz, pqw, qx, qy, qz, qw)
            else:
                lx, ly, lz = px, py, pz
                lqx, lqy, lqz, lqw = qx, qy, qz, qw

            out[row, 0] = lx
            out[row, 1] = ly
            out[row, 2] = lz
            out[row, 3] = lqw
            out[row, 4] = lqx
            out[row, 5] = lqy
            out[row, 6] = lqz
        return out, list(kept_paths)

    def _ensure_body_xform_table(self, ns) -> Optional[Tuple[List[int], List[int], List[str]]]:
        """Build the cached (newton_idx, parent_newton_idx, kept_path) table.

        Walked once on first call; populated from ``self._stage_obj.body_paths``
        intersected with ``ns.model.body_label``. Paths in our body_paths
        that Newton doesn't know about (e.g. fixed-frame children) drop
        out — they wouldn't have a body_q row anyway.
        """
        if self._body_xform_table is False:
            return None
        if self._body_xform_table is not None:
            return self._body_xform_table
        try:
            labels = list(getattr(ns.model, "body_label", []) or [])
        except Exception as exc:
            self._logger.warn(f"[isaac_newton] get_body_transforms: body_label unreadable: {exc}")
            self._body_xform_table = False
            return None
        if not labels:
            self._body_xform_table = False
            return None

        label_to_idx: Dict[str, int] = {str(p): i for i, p in enumerate(labels)}
        body_paths = list(self._stage_obj.body_paths)

        kept_paths: List[str] = []
        idx_arr: List[int] = []
        parent_idx_arr: List[int] = []
        missing: List[str] = []
        for path in body_paths:
            i = label_to_idx.get(path)
            if i is None:
                missing.append(path)
                continue
            # Walk USD ancestors to find the nearest one Newton also
            # knows about; that's our composition parent.
            parent_idx = -1
            cur = path.rsplit("/", 1)[0] if "/" in path else ""
            while cur:
                pi = label_to_idx.get(cur)
                if pi is not None:
                    parent_idx = pi
                    break
                cur = cur.rsplit("/", 1)[0] if "/" in cur else ""
            kept_paths.append(path)
            idx_arr.append(i)
            parent_idx_arr.append(parent_idx)

        if missing:
            self._logger.info(
                f"[isaac_newton] get_body_transforms: {len(missing)} body_paths "
                f"absent from Newton model.body_label (likely fixed-frame children); "
                f"first few: {missing[:5]}"
            )
        if not kept_paths:
            self._body_xform_table = False
            return None

        self._body_xform_table = (idx_arr, parent_idx_arr, kept_paths)
        self._logger.info(
            f"[isaac_newton] get_body_transforms: parent table built for {len(kept_paths)} "
            f"bodies (of {len(body_paths)} requested)"
        )
        return self._body_xform_table

    def shutdown(self) -> None:
        self._stage_obj.shutdown()

    # ------------------------------------------------------------------
    # Newton internals
    # ------------------------------------------------------------------

    def _get_newton_stage(self) -> Optional[Any]:
        """Return the live ``NewtonStage``, or ``None``.

        Cached after the first successful lookup; ``False`` means
        definitively absent. Used by ``_apply_newton_cfg`` to push the
        wrapper-cfg overrides.
        """
        if self._newton_stage_cache is False:
            return None
        if self._newton_stage_cache is not None:
            return self._newton_stage_cache
        try:
            from isaacsim.physics.newton.impl.extension import acquire_stage

            ns = acquire_stage()
        except Exception:
            ns = None
        if ns is not None:
            self._newton_stage_cache = ns
            print("[newton] NewtonStage acquired via extension.acquire_stage()", flush=True)
        else:
            print("[newton] NewtonStage not yet available; will retry next tick.", flush=True)
        return ns
