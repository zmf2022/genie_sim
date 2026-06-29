# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Solver-adapter abstract base class.

A ``SolverAdapter`` absorbs all solver-specific behavior — PD-drive prep,
solver construction, target-buffer storage — so that ``_LifecycleMixin``
in ``lifecycle.py`` can stay solver-agnostic.

The lifecycle calls hooks at well-defined points; see each method's
docstring for ordering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import warp as wp


class SolverAdapter(ABC):
    # ------------------------------------------------------------------
    # Capabilities (lifecycle queries these before build)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in log messages.

        e.g. ``"mujoco-warp"``, ``"featherstone"``.
        """

    @property
    @abstractmethod
    def supports_cloth(self) -> bool:
        """Whether this solver handles VBD cloth entries in the scene yaml.

        Lifecycle uses this to gate cloth scenes: an mjwarp solver with
        cloth entries in the scene yaml raises early instead of silently
        running rigid-only.
        """

    # ------------------------------------------------------------------
    # Pre-finalize: register custom attributes on the builder
    # ------------------------------------------------------------------

    @abstractmethod
    def register_custom_attributes(self, builder: Any) -> None:
        """Register solver-specific custom attributes on the ModelBuilder.

        Called BEFORE ``builder.add_usd()``.  The attributes a solver
        authors on the builder MUST match the solver instantiated after
        finalize — VBD reads ``edge_rest_angle`` etc that XPBD doesn't
        author, and vice versa.
        """

    # ------------------------------------------------------------------
    # Post-finalize, pre-solver: mutate model arrays
    # ------------------------------------------------------------------

    @abstractmethod
    def prepare_model(
        self,
        model: Any,
        logger: Any,
        mimic_followers: Optional[Dict[str, list]] = None,
    ) -> None:
        """Mutate model arrays BEFORE solver construction.

        For mjwarp: forces ``joint_target_mode := POSITION`` and overrides
        ``joint_target_ke/kd`` from the launcher fallback (because Isaac's
        URDF importer authors a vestigial DriveAPI that Newton interprets
        as VELOCITY mode and ke=625).

        For Featherstone: no-op (velocity injection bypasses PD torques).
        """

    # ------------------------------------------------------------------
    # Solver construction
    # ------------------------------------------------------------------

    @abstractmethod
    def build_solver(
        self,
        model: Any,
        sim_substeps: int,
        sim_iterations: int,
        logger: Any,
        mass_matrix_interval: int = 0,
    ) -> Any:
        """Construct and return the solver instance.

        Encapsulates solver-specific config (njmax/nconmax for mjwarp,
        update_mass_matrix_interval for Featherstone, etc.).

        ``mass_matrix_interval`` is the Featherstone-specific tunable for
        the M(q) rebuild cadence; ``0`` (default) means "use the
        solver-appropriate default" (Featherstone falls back to
        ``sim_substeps`` = one rebuild per frame).  Adapters that don't
        have a mass-matrix concept (mjwarp) ignore it.
        """

    # ------------------------------------------------------------------
    # Target-position buffer
    # ------------------------------------------------------------------

    @abstractmethod
    def init_target_buffer(
        self,
        model: Any,
        control: Any,
        logger: Any,
    ) -> None:
        """Allocate the position-target buffer and seed with ``model.joint_q``.

        Called AFTER ``_apply_init_joint_pos`` so the seed IS the init pose.
        Buffer must be allocated once; later writes use ``.assign()`` to
        preserve the pointer captured by the CUDA graph.
        """

    @abstractmethod
    def target_buffer(self) -> Optional[wp.array]:
        """The ``wp.array`` ``apply_commands`` scatters into.

        Returns ``None`` if ``init_target_buffer`` hasn't run yet (i.e.
        commands arriving before model build completes are silently
        dropped, which is the correct startup behavior).
        """

    # ------------------------------------------------------------------
    # Post-joint-map: solver-specific tweaks that need joint name → DOF
    # ------------------------------------------------------------------

    @abstractmethod
    def post_joint_map(
        self,
        model: Any,
        jindex: Any,
        control: Any,
        logger: Any,
    ) -> None:
        """Solver-specific setup that needs the resolved joint-index map.

        Called ONCE after ``_build_joint_map`` and ``_apply_init_joint_pos``
        in ``lifecycle.py``.

        ``jindex`` is the engine's ``JointIndex`` snapshot (see
        ``engine/newton/joint_index.py``). Adapters that want a
        ``{name: dof}`` dict shape can call
        ``jindex.name_to_dof()``; adapters that want full slice
        metadata (joint type, q-vs-qd widths, slice extents) iterate
        ``jindex.slices()``. Routing the JointIndex itself avoids a
        per-adapter copy and keeps the q/qd asymmetry resolved in one
        place.

        For mjwarp: no-op (PD already set in ``prepare_model``).

        For Featherstone: applies selective PD on PASSIVE joints (those
        not driven by ROS commands — body/head on the G2).  Velocity
        injection alone has O(dt²) drift under constant external torque
        and lets heavy root chains accumulate into their joint limits
        within one frame; selective PD adds a restoring force on those
        joints while command-driven DOFs (arms/grippers/wheels) stay
        velocity-injection only.
        """

    # ------------------------------------------------------------------
    # Per-substep robot step (called INSIDE the captured CUDA graph)
    # ------------------------------------------------------------------

    @abstractmethod
    def substep(
        self,
        model: Any,
        state_in: Any,
        state_out: Any,
        control: Any,
        sim_dt: float,
    ) -> None:
        """Run one rigid-body substep.

        Called from ``engine.py:_simulate_substeps`` inside the
        ``wp.ScopedCapture()`` block — implementations MUST do only GPU
        ops (``wp.launch``, ``solver.step``, ``arr.assign``); no Python
        branching beyond startup-time decisions, no allocations.

        For mjwarp: ``self._solver.step(state_in, state_out, control, None, sim_dt)``.

        For Featherstone: ``wp.launch(_kernel_velocity_inject, ...)``
        first to write ``joint_qd = target − q`` into ``state_in``,
        then ``self._solver.step(...)``.
        """
