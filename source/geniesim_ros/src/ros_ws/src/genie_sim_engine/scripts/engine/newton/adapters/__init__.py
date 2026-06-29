# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Solver-adapter factory.

Every supported physics solver returns a ``SolverAdapter`` instance;
unknown solvers raise ``ValueError`` so a typo can't silently fall back
to inline default behavior.

Solver taxonomy
---------------
* **Pure solvers** integrate everything (rigid bodies, particles, etc.)
  inside a single ``solver.step`` call — no composition with another
  solver downstream.  Currently: ``avbd``, ``mujoco-warp``.
* **Combo solvers** delegate cloth/particle integration to a separate
  Newton solver (always ``SolverVBD`` in this codebase) that
  ``lifecycle.py`` constructs in addition to the rigid solver returned
  here.  Currently: ``fsvbd`` (Featherstone-rigid + VBD-cloth).

The factory does NOT build the cloth solver — that's
``lifecycle._build``'s job, gated on whether the scene yaml declares any
cloth/particle entry.  Whether the adapter is "pure" vs "combo" is
exposed via ``adapter.supports_cloth`` (pure cloth-supporting → True
forces lifecycle to skip a separate cloth solver; combo adapters report
False so lifecycle adds one).
"""

from __future__ import annotations

from typing import Any, Optional

from engine.newton.adapters.base import SolverAdapter


def make_adapter(
    physics_solver: str,
    *,
    mujoco_pd_ke: float = 0.0,
    mujoco_pd_kd: float = 0.0,
    mujoco_save_to_mjcf: str = "",
    physics_params: Optional[Any] = None,
    physics_hz: float = 0.0,
) -> SolverAdapter:
    """Return an adapter for ``physics_solver``.

    Supported solvers:
      * ``mujoco-warp`` / ``mujoco_warp``  -> :class:`MuJoCoWarpAdapter`   (pure)
      * ``mjvbd`` / ``mjxpbd``             -> :class:`MuJoCoWarpAdapter`   (combo)
      * ``fsvbd`` / ``featherstone``       -> :class:`FeatherstoneAdapter` (combo)
      * ``avbd``                           -> :class:`AVBDAdapter`         (pure)

    ``mjvbd`` is the MJW analogue of ``fsvbd``: MuJoCo-Warp drives the
    rigid articulation, and ``lifecycle.py`` builds a separate
    ``SolverVBD`` with ``integrate_with_external_rigid_solver=True`` for
    cloth particles.  The adapter object is identical to the pure
    ``mujoco-warp`` adapter — the only difference is that the scene yaml
    declares cloth entries, which triggers the cloth-solver build.
    Useful when you want MJW's implicit integrator for the robot while
    keeping VBD cloth (e.g., to compare against the Featherstone+VBD
    path on the same scene with a single-token swap).

    ``fsvbd`` is the pipeline-composition name: Featherstone-for-rigid
    combined with a separate ``SolverVBD`` instance constructed by
    ``lifecycle.py`` for cloth particles.  The two solvers run
    back-to-back inside each substep; rigid bodies are kinematic from
    SolverVBD's point of view (``integrate_with_external_rigid_solver=True``).

    ``avbd`` selects a single ``SolverVBD`` instance in unified mode
    (``integrate_with_external_rigid_solver=False``): one ``solver.step``
    advances both rigid (via Augmented VBD) and cloth (via VBD).
    Lifecycle skips the separate cloth-solver build for this path, and
    the engine binds ``_substep_body_avbd_unified`` instead of
    ``_substep_body_franka_vbd_cloth``.

    ``mujoco-warp`` is also pure but rigid-only — cloth scenes refuse
    to start under it (``supports_cloth == False``).

    ``mujoco_save_to_mjcf`` is mjwarp-only: when set to a path the
    SolverMuJoCo writes the compiled MJCF (joints, actuators, gainprm/
    biasprm, contact params, equality constraints, joint limits) so
    operators can diff against a reference MJCF, run ``mujoco.viewer``
    on it, or validate it with ``mjpython``-side tooling.

    ``physics_params`` (mjwarp-only): when set, the adapter classifies
    each DOF via ``common.joint_classification`` and pulls per-class
    ``ke / kd / max_effort`` from ``PhysicsParams.art_body`` /
    ``.art_head`` / ``.art_arm_shoulder`` / ``.art_arm_mid`` /
    ``.art_arm_wrist`` (sub-classed body/head/arm) /
    ``.art_chassis_drive`` / ``.art_chassis_steer`` /
    ``.drive_gripper.master_*``, with ``.art_default`` as the
    fallback for un-classified joints.  Brings newton-standalone
    into parity with the isaac_newton wrapper (which applies these
    values via ``ArticulationView.set_dof_stiffnesses`` at runtime).
    When unset, the adapter falls back to its unscaled ``effort × 10``
    heuristic — used for ``test_newton_solver.py`` invocations that
    don't load the YAML.
    """
    key = physics_solver.strip().lower()
    # ``mjxpbd`` is an alias for ``mjvbd`` at the adapter layer: both keep
    # MuJoCo-Warp as the rigid solver and let lifecycle build a separate
    # particle solver. Which particle solver (VBD vs XPBD vs Style3D) is
    # chosen by ``newton.solver.prefer`` in the scene yaml.
    if key in ("mujoco-warp", "mujoco_warp", "mjvbd", "mjxpbd"):
        from engine.newton.adapters.mujoco_warp import MuJoCoWarpAdapter

        return MuJoCoWarpAdapter(
            pd_ke=mujoco_pd_ke,
            pd_kd=mujoco_pd_kd,
            save_to_mjcf=mujoco_save_to_mjcf,
            physics_params=physics_params,
            physics_hz=physics_hz,
        )
    if key in ("fsvbd", "featherstone"):
        from engine.newton.adapters.featherstone import FeatherstoneAdapter

        return FeatherstoneAdapter()
    if key == "avbd":
        from engine.newton.adapters.avbd import AVBDAdapter

        return AVBDAdapter()
    raise ValueError(
        f"unknown physics_solver: {physics_solver!r}; "
        f"expected one of: mujoco-warp (pure), mjvbd | mjxpbd (combo), "
        f"fsvbd | featherstone (combo), avbd (pure)"
    )


__all__ = ["SolverAdapter", "make_adapter"]
