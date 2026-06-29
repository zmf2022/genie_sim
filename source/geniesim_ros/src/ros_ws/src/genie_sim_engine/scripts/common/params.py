# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Physics parameter loader for genie_sim_engine.

The node passes ``physics_params_file`` (a path to a YAML file) at construction
time. This module returns a fully-populated :class:`PhysicsParams` dataclass.
On any failure (missing file, parse error, missing key) it falls back to the
built-in defaults defined below.

In addition to ``PhysicsParams`` (user-tuning, sourced from
``physics_params.yaml``), this module also defines :class:`JointInitSpec` —
the typed shape used to push **scene-specific** initial joint poses
into the articulation. The pose dict is sourced from the scene YAML's
``robot.init_joint_pos`` block at launch time and forwarded as a
JSON-encoded node parameter (``init_joint_pos_json``); see
``parse_init_joint_pos``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


@dataclass(frozen=True)
class DriveParams:
    """Per-joint USD ``DriveAPI`` + passive-joint authoring values.

    Two unrelated families of fields live here together because both
    flow through the same YAML block (``usd_drive_api.<class>:``):

    Actuator / DriveAPI (Layer 1 PD on the joint actuator):
      * ``stiffness`` — kp written to ``UsdPhysics.DriveAPI.stiffness``
      * ``damping``   — kd written to ``UsdPhysics.DriveAPI.damping``
      * ``max_force`` — written to ``UsdPhysics.DriveAPI.maxForce``

    Passive joint (no actuator involvement — physics dissipation
    intrinsic to the joint, applied every step regardless of any
    drive):
      * ``dof_damping``     — joint viscous damping (N·m·s/rad);
                              written to ``mjc:damping`` on USD
                              prims, lands as MuJoCo ``dof_damping``
                              after Newton's converter.  Default 0
                              (kept off so scenes that don't author
                              passive damping aren't unexpectedly
                              stiffer).
      * ``dof_frictionloss`` — Coulomb-style joint friction (N·m);
                               written to ``mjc:frictionloss`` on
                               USD prims, lands as MuJoCo
                               ``dof_frictionloss``.  Default 0.

    The passive-joint fields exist on every class because the
    reference G2 MJCF authors per-joint damping/frictionloss values
    that are well above zero (arm shoulder: damping=3.0,
    frictionloss=1.0; arm mid: 1.5 / 0.5; arm wrist: 0.8 / 0.3).
    Without these the dumped MJCF has 0 across the board and the
    simulation feels less settled than the reference under the same
    control — verified empirically via the static / dynamic XML diff
    tools against the reference G2 MJCF.
    """

    stiffness: float
    damping: float
    max_force: float
    free_limits: bool = False
    # Passive joint damping (mjc:damping) — see class docstring.
    dof_damping: float = 0.0
    # Joint frictionloss (mjc:frictionloss) — see class docstring.
    dof_frictionloss: float = 0.0
    # Reflected rotor inertia (N·m·s²/rad).  Lands as MuJoCo
    # ``dof_armature`` via Newton's ``model.joint_armature`` array.
    # Default 0 = leave whatever the URDF importer wrote (usually 0
    # for non-gripper joints).  Per-class values matter for mjwarp
    # because the gripper-only authoring path skips body/head/arm
    # and the dumped MJCF then omits the ``armature`` attribute,
    # which makes the model under-damped in pure-MuJoCo replay and
    # less stable at larger outer timesteps.  No angular quirk on
    # this field — author the physical value verbatim.
    armature: float = 0.0


@dataclass(frozen=True)
class ArticulationGains:
    kp: float
    kd: float
    max_effort: float


@dataclass(frozen=True)
class GripperDriveParams:
    """URDF-route gripper tuning — single source of truth.

    This dataclass is the **only** place gripper tuning lives.  The
    mjwarp adapter
    (``engine/newton/adapters/mujoco_warp.py:MuJoCoWarpAdapter``)
    reads every gripper field from here.  The isaac_newton wrapper
    (``kit/stage.py:_configure_drives`` plus
    ``kit/isaac_newton.py:_stiffen_*_mimic_equalities``) reads the
    same fields, so PhysX and mjwarp see the same tuning.

    Applied only on the URDF route (``from_urdf=True``); hand-authored
    USDs are left as-authored.  The master joint is identified by
    ``HasAPI(DriveAPI)`` (Isaac path) or by name classification
    (``classify_joint_by_name`` returning ``JK_GRIPPER`` and the joint
    matching the master via ``mimic_followers`` set membership;
    standalone path).

    Field families (call out who reads what):

    Master actuator
    ---------------
    Used by BOTH the PhysX path (DriveAPI on the joint USD prim,
    pushed through ``UsdPhysics.DriveAPI``) and the mjwarp path
    (``model.joint_target_ke/kd`` → ``actuator gainprm/biasprm``
    in the dumped MJCF).

      * ``master_stiffness`` / ``master_damping`` — kp / kd of the
        master position actuator.  Reference G2 MJCF uses (5, 0);
        a stiffer master fights the equality constraint when the
        equality is left at MuJoCo's soft default solref.
      * ``master_max_force`` — actuator effort cap, MJCF
        ``forcerange="-max +max"``.  ``0`` means "preserve the
        URDF ``<limit effort>`` value" (recommended — matches
        hardware spec).  Set non-zero to override.
      * ``master_ctrl_range_from_joint_limit`` — when True, the
        master's actuator ``ctrlrange`` is authored from the
        joint's ``range`` attribute, clamping policy ctrl writes
        to within the hardware limits.  Default True.

    Joint passive physics (applied to master + every follower)
    ----------------------------------------------------------
    Both paths consume; mjwarp reads via ``model.mujoco
    .dof_passive_damping`` / ``model.joint_friction`` /
    ``model.joint_armature``; PhysX reads via ``physxJoint:armature``
    + ``physxJoint:jointDamping`` schemas.

      * ``armature`` — per-joint rotor inertia (N·m·s²/rad).
        Improves solver stability without changing the
        dynamics.  Reference: 0.001.
      * ``dof_damping`` — passive viscous damping (N·m·s/rad).
        Reference: 0.05.  Critical for damping out follower
        oscillations under a soft mimic equality.
      * ``dof_frictionloss`` — Coulomb-style joint friction
        (N·m).  Reference: 0.01.  Tames residual creep.

    Mimic equality solver tuning (mjwarp ONLY)
    ------------------------------------------
    PhysX uses ``PhysxMimicJointAPI`` (see ``mimic_natural_frequency``
    / ``mimic_damping_ratio`` below) and ignores these.  Mjwarp uses
    these to set ``mjw_model.eq_solref`` / ``eq_solimp`` for every
    gripper mimic equality.

      * ``mimic_eq_solref`` — equality solver impedance time-constant
        / damping-ratio pair.  MuJoCo's default ``(0.02, 1.0)`` is a
        soft spring with 20 ms time constant — works but lets
        followers swing under aggressive arm motion.  For stiff
        direct mode set negative values: ``(-stiffness, -damping)``
        e.g. ``(-10000, -100)``.  See MuJoCo docs on ``solref``.
      * ``mimic_eq_solimp`` — equality impedance schedule
        ``(d_min, d_max, width, midpoint, power)``.

    PhysX mimic API (Isaac path only — ignored by mjwarp)
    -----------------------------------------------------
      * ``mimic_natural_frequency`` / ``mimic_damping_ratio`` —
        ``PhysxMimicJoint:<axis>:naturalFrequency`` /
        ``dampingRatio``.  Both MUST be 0.0 for a rigid kinematic
        constraint on the PhysX side.  Non-zero turns the
        constraint into a soft spring/damper.
    """

    master_stiffness: float
    master_damping: float
    armature: float
    mimic_natural_frequency: float
    mimic_damping_ratio: float
    # New mjwarp-relevant fields with defaults matching the prior
    # adapter constants.  These propagate through ``_gripper()``
    # (the YAML loader merger) so a YAML without these keys still
    # loads at the documented defaults.
    master_max_force: float = 0.0
    master_ctrl_range_from_joint_limit: bool = True
    dof_damping: float = 0.05
    dof_frictionloss: float = 0.01
    mimic_eq_solref: tuple = (0.02, 1.0)
    mimic_eq_solimp: tuple = (0.9, 0.95, 0.001, 0.5, 2.0)


@dataclass(frozen=True)
class PhysicsParams:
    drive_chassis_drive_joint: DriveParams
    drive_chassis_steer_joint: DriveParams
    drive_default_revolute: DriveParams
    drive_default_prismatic: DriveParams
    # Per-sub-class passive-joint authoring.  The actuator gains
    # (``stiffness`` / ``damping`` / ``max_force``) on these blocks
    # are NOT applied — they fall back to ``drive_default_revolute``
    # for the adapter's PD lookup.  Only ``dof_damping`` and
    # ``dof_frictionloss`` are read per sub-class, matching the
    # reference G2 MJCF's per-joint-position profile (shoulder
    # 3.0/1.0, mid 1.5/0.5, wrist 0.8/0.3, body/head 0/0).  See
    # ``MuJoCoWarpAdapter._per_class_passive_joint`` for the route.
    drive_body: DriveParams
    drive_head: DriveParams
    drive_arm_shoulder: DriveParams
    drive_arm_mid: DriveParams
    drive_arm_wrist: DriveParams
    drive_gripper: GripperDriveParams
    art_default: ArticulationGains
    art_chassis_drive: ArticulationGains
    art_chassis_steer: ArticulationGains
    # Per-sub-class articulation-PD gains (Layer 2 actuator authority,
    # written via ArticulationView.set_dof_stiffnesses /
    # set_dof_dampings / set_dof_max_forces).  The reference G2 MJCF
    # tiers these by joint
    # inertia: heavy proximal joints carry high kp (40000) with
    # commensurate kd (600); light wrist joints run far lower
    # (8000 / 100); body uses 1e5 / 1e3; head uses 1e3 / 20.
    # Falling back to ``art_default`` (5e4 / 5e3) across body / head
    # / arm gives > 2× divergence on every joint vs the reference,
    # which the static-diff tool surfaces as suspicious flags.
    # Each sub-class block is independent — when a YAML omits a
    # block, the parser falls back to ``art_default`` so operator
    # configs without sub-class overrides keep working unchanged.
    art_body: ArticulationGains
    art_head: ArticulationGains
    art_arm_shoulder: ArticulationGains
    art_arm_mid: ArticulationGains
    art_arm_wrist: ArticulationGains
    cmd_4ws_timeout_s: float
    render_target_hz: float
    render_safety_ms: float


_DEFAULTS = PhysicsParams(
    drive_chassis_drive_joint=DriveParams(stiffness=0.0, damping=1.0e6, max_force=1.0e7, free_limits=True),
    # Steering joints: same shape as the regular position drive but stiffer
    # so the steering rack tracks setpoints fast and accurately. Mirrors the
    # ``art_chassis_steer`` articulation-level kp = 1e5.
    drive_chassis_steer_joint=DriveParams(stiffness=1.0e5, damping=5.0e3, max_force=5.0e3, armature=0.05),
    # Passive damping/frictionloss defaults (``dof_damping``, ``dof_frictionloss``)
    # for body / arm / head are the midpoints of the reference G2 MJCF's
    # per-joint-position tiers: arm shoulder uses 3.0 / 1.0, mid uses
    # 1.5 / 0.5, wrist uses 0.8 / 0.3.  We collapse those into a single
    # per-class value here (1.5 / 0.5) because the classifier in
    # ``common/joint_classification.py`` doesn't split arm by joint
    # number — if you need the tiered profile, add sub-class regexes
    # there.  Lifted from the reference G2 MJCF's arm joints.
    drive_default_revolute=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=1.5,
        dof_frictionloss=0.5,
        armature=0.05,
    ),
    drive_default_prismatic=DriveParams(stiffness=5.0e4, damping=5.0e3, max_force=5.0e4),
    # Per-sub-class passive-joint defaults — see ``PhysicsParams`` docstring.
    # The stiffness / damping / max_force fields are placeholder copies
    # of ``drive_default_revolute``'s actuator gains (the adapter never
    # reads those off the sub-class blocks; per-class gains stay
    # centralised in ``art_default``).  Only ``dof_damping`` and
    # ``dof_frictionloss`` matter here.
    drive_body=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=0.0,
        dof_frictionloss=0.0,
        armature=0.1,
    ),
    drive_head=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=0.0,
        dof_frictionloss=0.0,
        armature=0.05,
    ),
    drive_arm_shoulder=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=3.0,
        dof_frictionloss=1.0,
        armature=0.05,
    ),
    drive_arm_mid=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=1.5,
        dof_frictionloss=0.5,
        armature=0.05,
    ),
    drive_arm_wrist=DriveParams(
        stiffness=5.0e4,
        damping=5.0e3,
        max_force=5.0e3,
        dof_damping=0.8,
        dof_frictionloss=0.3,
        armature=0.02,
    ),
    # Gripper master/mimic tuning — see GripperDriveParams docstring. The
    # 1e4 / 10 / 0.001 / 0 / 0 defaults are hand-tuned values that
    # avoid gravity-induced droop on the Robotiq 2F-140 linkage while
    # keeping the 2F-85 within a working envelope.
    drive_gripper=GripperDriveParams(
        master_stiffness=1.0e4,
        master_damping=10.0,
        armature=0.001,
        mimic_natural_frequency=0.0,
        mimic_damping_ratio=0.0,
    ),
    art_default=ArticulationGains(kp=5.0e4, kd=5.0e3, max_effort=5.0e3),
    art_chassis_drive=ArticulationGains(kp=0.0, kd=1.0e4, max_effort=1.0e7),
    art_chassis_steer=ArticulationGains(kp=1.0e5, kd=5.0e3, max_effort=5.0e3),
    # Per-sub-class actuator-PD gains, lifted from the reference G2
    # MJCF's position-actuator gainprm /
    # biasprm[2].  ``max_effort`` for body/head matches the
    # reference's motor-actuator ``ctrlrange=±1200`` (the reference
    # leaves position-actuator forcerange unlimited and gates
    # input via ctrlrange — ours expresses the cap as max_effort
    # since the runtime PD path doesn't author ctrlrange).  Arm
    # tiers use the URDF effort caps (108 / 35 / 18 N·m).
    art_body=ArticulationGains(kp=1.0e5, kd=1.0e3, max_effort=1.2e3),
    art_head=ArticulationGains(kp=1.0e3, kd=20.0, max_effort=1.2e3),
    art_arm_shoulder=ArticulationGains(kp=4.0e4, kd=600.0, max_effort=108.0),
    art_arm_mid=ArticulationGains(kp=1.5e4, kd=220.0, max_effort=35.0),
    art_arm_wrist=ArticulationGains(kp=8.0e3, kd=100.0, max_effort=18.0),
    cmd_4ws_timeout_s=0.1,
    render_target_hz=30.0,
    render_safety_ms=2.0,
)


def default_physics_params() -> PhysicsParams:
    return _DEFAULTS


def _drive(d: Dict[str, Any], fallback: DriveParams) -> DriveParams:
    return DriveParams(
        stiffness=float(d.get("stiffness", fallback.stiffness)),
        damping=float(d.get("damping", fallback.damping)),
        max_force=float(d.get("max_force", fallback.max_force)),
        free_limits=bool(d.get("free_limits", fallback.free_limits)),
        dof_damping=float(d.get("dof_damping", fallback.dof_damping)),
        dof_frictionloss=float(d.get("dof_frictionloss", fallback.dof_frictionloss)),
        armature=float(d.get("armature", fallback.armature)),
    )


def _art(d: Dict[str, Any], fallback: ArticulationGains) -> ArticulationGains:
    return ArticulationGains(
        kp=float(d.get("kp", fallback.kp)),
        kd=float(d.get("kd", fallback.kd)),
        max_effort=float(d.get("max_effort", fallback.max_effort)),
    )


def _gripper(d: Dict[str, Any], fallback: GripperDriveParams) -> GripperDriveParams:
    return GripperDriveParams(
        master_stiffness=float(d.get("master_stiffness", fallback.master_stiffness)),
        master_damping=float(d.get("master_damping", fallback.master_damping)),
        armature=float(d.get("armature", fallback.armature)),
        mimic_natural_frequency=float(d.get("mimic_natural_frequency", fallback.mimic_natural_frequency)),
        mimic_damping_ratio=float(d.get("mimic_damping_ratio", fallback.mimic_damping_ratio)),
        master_max_force=float(d.get("master_max_force", fallback.master_max_force)),
        master_ctrl_range_from_joint_limit=bool(
            d.get("master_ctrl_range_from_joint_limit", fallback.master_ctrl_range_from_joint_limit)
        ),
        dof_damping=float(d.get("dof_damping", fallback.dof_damping)),
        dof_frictionloss=float(d.get("dof_frictionloss", fallback.dof_frictionloss)),
        mimic_eq_solref=tuple(d.get("mimic_eq_solref", fallback.mimic_eq_solref)),
        mimic_eq_solimp=tuple(d.get("mimic_eq_solimp", fallback.mimic_eq_solimp)),
    )


def load_physics_params(path: Optional[str], logger=None) -> PhysicsParams:
    """Load physics tuning from ``path`` (YAML).

    Empty / missing path, missing file, parse error, or bad shape all log a
    warning (when ``logger`` is provided) and return :func:`default_physics_params`.
    """
    if not path:
        if logger is not None:
            logger.info("physics_params_file empty — using built-in defaults")
        return _DEFAULTS

    try:
        import yaml  # ros2 already pulls PyYAML
    except ImportError:
        if logger is not None:
            logger.warn(f"PyYAML unavailable — using defaults, ignoring {path}")
        return _DEFAULTS

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
    except OSError as exc:
        if logger is not None:
            logger.warn(f"could not read {path}: {exc} — using defaults")
        return _DEFAULTS
    except yaml.YAMLError as exc:
        if logger is not None:
            logger.warn(f"could not parse {path}: {exc} — using defaults")
        return _DEFAULTS

    # Two-layer drive authoring (see top-of-file block comment in
    # ``config/physics_params.yaml`` for the full explanation).  The YAML
    # canonical block names are ``usd_drive_api:`` (Layer 1: USD-authored,
    # cross-engine) and ``articulation_view_runtime:`` (Layer 2:
    # ArticulationView tensor API, Isaac-only).  We also accept the
    # shorter ``drive:`` / ``articulation:`` block names for back-compat
    # and log a nudge so the rename eventually propagates.
    if "usd_drive_api" in data:
        drive = data.get("usd_drive_api", {}) or {}
    elif "drive" in data:
        if logger is not None:
            logger.warn(
                f"{path}: ``drive:`` block name accepted for back-compat — "
                f"rename to ``usd_drive_api:`` to match the canonical schema."
            )
        drive = data.get("drive", {}) or {}
    else:
        drive = {}
    if "articulation_view_runtime" in data:
        art = data.get("articulation_view_runtime", {}) or {}
    elif "articulation" in data:
        if logger is not None:
            logger.warn(
                f"{path}: ``articulation:`` block name accepted for "
                f"back-compat — rename to ``articulation_view_runtime:`` to "
                f"match the canonical schema."
            )
        art = data.get("articulation", {}) or {}
    else:
        art = {}
    cmd = data.get("command", {}) or {}
    step = data.get("stepping", {}) or {}

    params = PhysicsParams(
        drive_chassis_drive_joint=_drive(
            drive.get("chassis_drive_joint", {}) or {}, _DEFAULTS.drive_chassis_drive_joint
        ),
        drive_chassis_steer_joint=_drive(
            drive.get("chassis_steer_joint", {}) or {}, _DEFAULTS.drive_chassis_steer_joint
        ),
        drive_default_revolute=_drive(drive.get("default_revolute", {}) or {}, _DEFAULTS.drive_default_revolute),
        drive_default_prismatic=_drive(drive.get("default_prismatic", {}) or {}, _DEFAULTS.drive_default_prismatic),
        # Per-sub-class passive-joint blocks default the
        # actuator-gain fields from ``default_revolute`` (or from the
        # built-in sub-class default, whichever the YAML doesn't
        # author).  Means an operator only has to write
        # ``dof_damping`` / ``dof_frictionloss`` per sub-class and
        # the unrelated DriveAPI gains stay consistent.
        drive_body=_drive(drive.get("body", {}) or {}, _DEFAULTS.drive_body),
        drive_head=_drive(drive.get("head", {}) or {}, _DEFAULTS.drive_head),
        drive_arm_shoulder=_drive(drive.get("arm_shoulder", {}) or {}, _DEFAULTS.drive_arm_shoulder),
        drive_arm_mid=_drive(drive.get("arm_mid", {}) or {}, _DEFAULTS.drive_arm_mid),
        drive_arm_wrist=_drive(drive.get("arm_wrist", {}) or {}, _DEFAULTS.drive_arm_wrist),
        drive_gripper=_gripper(drive.get("gripper", {}) or {}, _DEFAULTS.drive_gripper),
        art_default=_art(art.get("default", {}) or {}, _DEFAULTS.art_default),
        art_chassis_drive=_art(art.get("chassis_drive", {}) or {}, _DEFAULTS.art_chassis_drive),
        art_chassis_steer=_art(art.get("chassis_steer", {}) or {}, _DEFAULTS.art_chassis_steer),
        # Per-sub-class actuator gains.  Each sub-class falls back to
        # its built-in tiered default (NOT to ``art_default``) — same
        # pattern as ``chassis_drive`` / ``chassis_steer`` above.
        # The tiered defaults match the reference G2 MJCF; an
        # operator who wants a uniform 5e4 / 5e3 across body / arm /
        # head must spell it out explicitly per sub-class in the
        # YAML.
        art_body=_art(art.get("body", {}) or {}, _DEFAULTS.art_body),
        art_head=_art(art.get("head", {}) or {}, _DEFAULTS.art_head),
        art_arm_shoulder=_art(art.get("arm_shoulder", {}) or {}, _DEFAULTS.art_arm_shoulder),
        art_arm_mid=_art(art.get("arm_mid", {}) or {}, _DEFAULTS.art_arm_mid),
        art_arm_wrist=_art(art.get("arm_wrist", {}) or {}, _DEFAULTS.art_arm_wrist),
        cmd_4ws_timeout_s=float(cmd.get("cmd_4ws_timeout_s", _DEFAULTS.cmd_4ws_timeout_s)),
        render_target_hz=float(step.get("render_target_hz", _DEFAULTS.render_target_hz)),
        render_safety_ms=float(step.get("render_safety_ms", _DEFAULTS.render_safety_ms)),
    )
    # A ``newton:`` block in the YAML is silently ignored — there is no
    # runtime consumer for it.  Issue a one-line nudge so an operator
    # with such a block knows it is dead weight.
    if logger is not None and data.get("newton"):
        logger.warn(
            f"{path}: ``newton:`` block found — no runtime consumer "
            f"exists.  Drop the block to silence this warning."
        )

    if logger is not None:
        logger.info(f"loaded physics params from {path}")
    return params


# ---------------------------------------------------------------------------
# Engine node parameters (ROS parameter dict → typed dataclass)
#
# Mirrors the IsaacLab @configclass pattern: one place owns all field
# definitions, types, defaults, and coercion logic.  EngineSession constructs
# this from the raw {str: str} dict that _parse_args() returns so that the
# rest of EngineSession.__init__ only touches typed attributes.
# ---------------------------------------------------------------------------


@dataclass
class EngineNodeParams:
    """Typed representation of the flat ROS-parameter dict for engine entry points.

    Construct with :meth:`from_dict`.  All coercion and sentinel semantics live here.

    ``render_hz == 0.0`` is the "not provided" sentinel; callers resolve it
    against ``PhysicsParams.render_target_hz`` after loading the YAML config.

    ``physics_solver == ""`` is the "engine picks its own default" sentinel;
    each engine normalises it internally.

    ``physics_solver_substep / physics_solver_iterations == 0`` is the
    "engine picks its own default" sentinel — see the comment in
    ``EngineSession.__init__`` for per-engine semantics.
    """

    physics_hz: float = 100.0
    render_hz: float = 0.0  # 0.0 → use PhysicsParams.render_target_hz
    realtime_factor: float = 1.0  # 1.0 = realtime; 0.1 = 10× slower; >1 = faster-than-real
    fake_slam: bool = False
    physics_params_file: str = ""
    physics_solver: str = ""  # "" → engine default
    render_mode: str = "raster"
    physics_solver_substep: int = 0  # 0 → engine default
    physics_solver_iterations: int = 0  # 0 → engine default
    physics_solver_mass_matrix_interval: int = (
        0  # 0 → engine default (= sim_substeps; large value disables auto rebuild)
    )
    stage_manifest: str = ""
    init_joint_pos_json: str = ""
    mujoco_pd_ke: float = 0.0  # 0 → engine default (50000 N·m/rad)
    mujoco_pd_kd: float = 0.0  # 0 → engine default (500 N·m·s/rad)

    @classmethod
    def from_dict(cls, params: dict) -> "EngineNodeParams":
        """Coerce a flat ``{str: str}`` params dict into typed fields."""

        def _bool(key: str, default: bool = False) -> bool:
            s = str(params.get(key, "")).strip().lower()
            if s in ("true", "1", "yes", "on"):
                return True
            if s in ("false", "0", "no", "off"):
                return False
            return default

        def _float(key: str, default: float) -> float:
            raw = params.get(key, "")
            try:
                return float(raw) if raw not in ("", None) else default
            except (TypeError, ValueError):
                return default

        def _int_nonneg(key: str) -> int:
            try:
                return max(0, int(params.get(key, 0) or 0))
            except (TypeError, ValueError):
                return 0

        return cls(
            physics_hz=_float("physics_hz", 100.0),
            render_hz=_float("render_hz", 0.0),
            realtime_factor=max(1e-6, _float("realtime_factor", 1.0)),
            fake_slam=_bool("fake_slam"),
            physics_params_file=str(params.get("physics_params_file", "") or ""),
            physics_solver=str(params.get("physics_solver", "") or "").strip().lower(),
            render_mode=str(params.get("render_mode", "raster") or "raster").strip().lower(),
            physics_solver_substep=_int_nonneg("physics_solver_substep"),
            physics_solver_iterations=_int_nonneg("physics_solver_iterations"),
            physics_solver_mass_matrix_interval=_int_nonneg("physics_solver_mass_matrix_interval"),
            stage_manifest=str(params.get("stage_manifest", "") or ""),
            init_joint_pos_json=str(params.get("init_joint_pos_json", "") or ""),
            mujoco_pd_ke=_float("mujoco_pd_ke", 0.0),
            mujoco_pd_kd=_float("mujoco_pd_kd", 0.0),
        )


# ---------------------------------------------------------------------------
# Scene-sourced initial joint pose
#
# Mirrors the typed-dataclass + override-logging style of DriveParams /
# ArticulationGains, but sourced from the **scene YAML** at launch time
# (forwarded as a JSON-encoded node parameter so it survives launch's
# flat-string parameter pipeline). Kept out of the manifest deliberately
# so re-launching with a tweaked ``init_joint_pos`` does NOT require a
# re-assemble step.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointInitSpec:
    """Requested initial pose for one joint, sourced from scene YAML.

    The numeric ``value`` is interpreted by the consumer based on the
    joint's USD type at apply time:

    * ``UsdPhysics.RevoluteJoint`` → degrees (converted to radians for
      ``set_joint_positions``).
    * ``UsdPhysics.PrismaticJoint`` → metres (passed through verbatim).

    This matches the convention already used in existing scene YAMLs
    (e.g. ``arm_joint2: -90`` is degrees, not radians), and avoids the
    error-prone hand-editing of radian values in YAML.
    """

    value: float


def parse_init_joint_pos(
    payload: Optional[Union[str, Dict[str, Any]]],
    logger=None,
) -> Dict[str, JointInitSpec]:
    """Parse the ``init_joint_pos`` payload into ``{joint_name: JointInitSpec}``.

    Accepts either:

    * a JSON-encoded string (the launch-file → node parameter path), or
    * a Python dict (programmatic / test path).

    Empty / missing payloads return an empty dict; malformed payloads
    log a warning (when ``logger`` is provided) and return empty.
    """
    if not payload:
        return {}

    data: Any
    if isinstance(payload, str):
        s = payload.strip()
        if not s or s in ("{}", "null"):
            return {}
        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            if logger is not None:
                logger.warn(f"init_joint_pos_json: invalid JSON ({exc}) — ignoring")
            return {}
    else:
        data = payload

    if not isinstance(data, dict):
        if logger is not None:
            logger.warn(f"init_joint_pos: expected mapping, got {type(data).__name__} — ignoring")
        return {}

    out: Dict[str, JointInitSpec] = {}
    for jname, raw in data.items():
        try:
            out[str(jname)] = JointInitSpec(value=float(raw))
        except (TypeError, ValueError):
            if logger is not None:
                logger.warn(f"init_joint_pos[{jname}]: non-numeric value {raw!r} — skipping")
    return out
