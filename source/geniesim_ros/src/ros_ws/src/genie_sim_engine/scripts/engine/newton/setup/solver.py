# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phases 5-6: states / robot solver / cloth solver / mjc readback.

Allocates ``state_0/state_1/control/contacts``, sets up gravity,
runs ``eval_fk``, builds the adapter-provided robot solver, then
the cloth solver (VBD / XPBD / Style3D) for particle scenes.
"""

from __future__ import annotations

import json
import math
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp


class _SolverMixin:
    def _phase_states_and_robot_solver(self) -> None:
        """Phase 5: allocate states/control/contacts, build the robot solver.

        Includes gravity setup, ``eval_fk`` init, mimic-map first pass,
        adapter ``prepare_model``, and the adapter-provided solver
        construction.  After this phase ``self._robot_solver`` is live.
        """
        import newton  # noqa: PLC0415

        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        self._contacts = self._model.contacts()

        # AVBD-unified adapter needs the contacts buffer for its
        # explicit ``model.collide`` + ``solver.step`` substep pattern
        # (mirrors newton/examples/cloth/example_cloth_poker_cards.py).
        # Featherstone and mjwarp adapters don't define ``set_contacts``
        # — Newton's contact pipeline is wired internally on those paths.
        if hasattr(self._adapter, "set_contacts"):
            self._adapter.set_contacts(self._contacts)
            self._logger.info(
                "[newton-standalone] adapter.set_contacts: contacts buffer " "passed into adapter (AVBD-unified path)"
            )

        # Gravity arrays for robot-step trick.
        # model.gravity may be None if no world was added to ModelBuilder before
        # finalize (the attribute is only populated when add_usd parses a
        # physicsScene or add_ground_plane triggers world creation).
        # Fall back to a standard -9.81 m/s² Z-up array if absent.
        import warp as wp  # noqa: PLC0415

        if getattr(self._model, "gravity", None) is not None and len(self._model.gravity) > 0:
            self._gravity_earth = wp.clone(self._model.gravity)
        else:
            self._gravity_earth = wp.array([wp.vec3(0.0, 0.0, -9.81)], dtype=wp.vec3, device="cuda:0")
            if hasattr(self._model, "gravity") and self._model.gravity is not None:
                self._model.gravity.assign(self._gravity_earth)
        self._gravity_zero = wp.zeros(len(self._gravity_earth), dtype=wp.vec3, device=self._gravity_earth.device)

        # FK init — place robot bodies at their correct initial positions
        newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state_0)

        # Solvers — the adapter owns PD-drive setup (mjwarp: prepare_model;
        # featherstone: post_joint_map for selective passive PD) and the
        # solver construction itself.  Cloth solver setup is solver-
        # independent and stays inline below.
        #
        # Parse mimic relationships from USD HERE (before prepare_model)
        # so the adapter can suppress actuators on follower DOFs.  Our
        # mjwarp adapter uses this to match the reference G2 MJCF
        # topology: only the gripper master joint gets a PD actuator;
        # followers are slaved by the mjwarp equality constraints Newton's
        # importer already wrote from USD ``NewtonMimicAPI`` schemas.
        # Without this suppression we get 6 independent PD actuators per
        # gripper fighting the equality constraints → numerical chatter.
        # ``_build_mimic_map`` is re-called later in the topology stage to
        # populate ``self._mimic_followers`` (used by apply_commands for
        # target broadcast); calling it twice is cheap — pure USD walk.
        self._build_mimic_map()
        self._adapter.prepare_model(self._model, self._logger, mimic_followers=self._mimic_followers)
        self._robot_solver = self._adapter.build_solver(
            self._model,
            self._sim_substeps,
            self._vbd_iterations,
            self._logger,
            mass_matrix_interval=getattr(self, "_mass_matrix_interval", 0),
        )

    def _phase_cloth_solver_and_readback(
        self,
        cloth_entries: List[Any],
        solver_pref: str,
        plugin_has_build: bool,
    ) -> None:
        """Phase 6: mjc_contact readback + cloth solver construction.

        The readback reads ``model.geom_solimp`` which MJW populates
        inside ``SolverMuJoCo`` construction (Phase 5), so it must run
        after the robot solver build, not after finalize.  Then the
        particle solver branch picks VBD / XPBD / Style3D for cloth
        scenes (skipped when AVBD is unified — its solver already
        covers cloth).
        """
        import newton  # noqa: PLC0415

        # Diagnostic-only readback of the pre-finalize mjc_contact writes.
        # Runs here (not right after finalize) so ``model.geom_solimp`` is
        # populated by the SolverMuJoCo construction above.  Cheap and
        # safe to leave on; doesn't mutate anything.
        from engine.newton.mjc_contact import readback_post_finalize as _mjc_readback  # noqa: PLC0415

        _mjc_readback(
            self._model,
            scene_cfg=self._scene_cfg,
            robot_prefix=str(getattr(self, "_robot_prefix_str", "") or ""),
            adapter_name=getattr(self._adapter, "name", ""),
            logger=self._logger,
        )

        # Plugin-driven particle scenes (no cloth_entries but particles
        # were emitted by ``on_build`` via ``add_soft_grid``) also need
        # the particle solver branch.  Detect by particle_count > 0.
        has_particles = bool(cloth_entries) or (plugin_has_build and self._model.particle_count > 0)
        if has_particles:
            # edge_rest_angle.zero_() is cloth-mesh-only (cloth has edges
            # in builder.tri_indices); FEM tet bars from add_soft_grid
            # also produce surface tris and a corresponding edge_rest_angle
            # array, so zeroing it is safe (rest pose is the as-spawned
            # straight bar).  Only call when the array exists.
            if hasattr(self._model, "edge_rest_angle") and self._model.edge_rest_angle is not None:
                self._model.edge_rest_angle.zero_()
            # When the rigid adapter is AVBD, ``build_solver`` already
            # constructed a unified SolverVBD that integrates BOTH rigid
            # and cloth in one ``solver.step``.  Building a SECOND
            # SolverVBD here would double-init the cloth contact pipeline
            # and confuse the substep loop (engine.py:_substep_body_*
            # would call cloth_solver.step on top of the already-stepped
            # state).  Skip the separate cloth-solver build entirely on
            # the AVBD path; the engine's substep selector picks the
            # unified-AVBD substep accordingly.
            if getattr(self._adapter, "name", "") == "avbd":
                self._cloth_solver = None
                self._logger.info(
                    "[newton-standalone] cloth solver: skipped "
                    "(AVBD rigid adapter handles cloth inside its own "
                    "SolverVBD instance — see _substep_body_avbd_unified)"
                )
            # ``solver_pref`` was loaded once at the top of _build (so the
            # builder's register_custom_attributes matches the solver we
            # instantiate here). No silent fallback to a default solver —
            # _load_solver_preference raises on unrecognised values.
            elif solver_pref == "vbd":
                self._cloth_solver = newton.solvers.SolverVBD(
                    self._model,
                    iterations=self._vbd_iterations,
                    integrate_with_external_rigid_solver=True,
                    particle_self_contact_radius=self._SELF_CONTACT_RADIUS,
                    particle_self_contact_margin=self._SELF_CONTACT_MARGIN,
                    particle_topological_contact_filter_threshold=1,
                    particle_rest_shape_contact_exclusion_radius=self._SELF_CONTACT_RADIUS * 2.5,
                    particle_enable_self_contact=True,
                    particle_vertex_contact_buffer_size=16,
                    particle_edge_contact_buffer_size=20,
                    particle_collision_detection_interval=-1,
                    rigid_contact_k_start=self._SOFT_CONTACT_KE,
                )
                self._logger.info(
                    f"[newton-standalone] cloth solver: SolverVBD " f"(iterations={self._vbd_iterations})"
                )
            elif solver_pref == "xpbd":
                # XPBD has a different constructor signature than VBD —
                # iterations is positional, particle_*_contact_* aren't
                # accepted at all. Keeping the params we can pass and
                # logging the ones we drop, so it's obvious from the
                # log what tuning is and isn't in effect.
                # ``soft_body_relaxation`` / ``soft_contact_relaxation``
                # are surfaced from the scene yaml's ``newton.xpbd:``
                # block (newton-engine-wide; not chef-specific).  Defaults
                # match chow_mein_flip_fem.py: 0.9 / 0.3 (the 0.9 default
                # for soft_contact_relaxation is jittery on FEM rests).
                _xpbd = (self._scene_cfg.get("newton") or {}).get("xpbd") or {}
                _relax = float(_xpbd.get("soft_body_relaxation", 0.9))
                _contact_relax = float(_xpbd.get("soft_contact_relaxation", 0.3))
                self._cloth_solver = newton.solvers.SolverXPBD(
                    self._model,
                    iterations=self._vbd_iterations,
                    soft_body_relaxation=_relax,
                    soft_contact_relaxation=_contact_relax,
                )
                self._logger.info(
                    f"[newton-standalone] cloth solver: SolverXPBD "
                    f"(iterations={self._vbd_iterations}, "
                    f"soft_body_relaxation={_relax}, "
                    f"soft_contact_relaxation={_contact_relax}); "
                    "VBD-only self-contact tuning params were not applied."
                )
            elif solver_pref == "style3d":
                self._cloth_solver = newton.solvers.SolverStyle3D(
                    self._model,
                    iterations=self._vbd_iterations,
                )
                self._logger.info(
                    f"[newton-standalone] cloth solver: SolverStyle3D "
                    f"(iterations={self._vbd_iterations}); self-contact "
                    "tuning params are VBD-only and were not applied."
                )
            else:
                raise RuntimeError(
                    f"newton-standalone engine has cloth entries but solver "
                    f"preference resolved to {solver_pref!r}. Set "
                    f"newton.solver.prefer to one of: vbd, xpbd, style3d "
                    f"in the scene yaml, then re-launch (assemble_newton "
                    f"runs on every launch and re-writes the sidecar)."
                )
