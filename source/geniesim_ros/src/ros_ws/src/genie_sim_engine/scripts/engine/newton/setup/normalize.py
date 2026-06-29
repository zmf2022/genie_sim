# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phase 4: post-finalize model normalisations.

Mass clamp (Featherstone NaN guard), soft-contact material
constants, collision-group unification, and MJW articulation
unification.  Pure mutations on ``self._model``.
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


class _NormalizeMixin:
    def _phase_normalize_model(self) -> None:
        """Phase 4: post-finalize model normalisations.

        Clamps zero-mass bodies (Featherstone NaN guard), sets soft-contact
        material constants, then collapses collision groups so cross-source
        rigid bodies collide, and on MJW unifies articulations so passive
        free bodies actually step.  Pure mutations on ``self._model``.
        """
        # Clamp zero-mass bodies to prevent Featherstone NaN.
        # USD robots sometimes have massless intermediate links (spine
        # segments, structural bridges) that PhysX handles via virtual mass
        # injection, but Newton's Featherstone inverts the spatial inertia
        # matrix directly — zero mass → singular → NaN on frame 1.
        _MIN_MASS = 1e-4  # 0.1 gram
        _MIN_INERTIA = 1e-8  # kg·m²
        if self._model.body_mass is not None:
            body_mass_np = self._model.body_mass.numpy().copy()
            zero_idx = (body_mass_np < _MIN_MASS).nonzero()[0]
            if len(zero_idx) > 0:
                self._logger.info(
                    f"[newton-standalone] clamping {len(zero_idx)} zero-mass body/bodies "
                    f"to {_MIN_MASS} kg: idx {zero_idx[:8].tolist()}"
                    f"{'...' if len(zero_idx) > 8 else ''}"
                )
                body_mass_np[zero_idx] = _MIN_MASS
                self._model.body_mass = wp.array(body_mass_np, dtype=wp.float32, device=self._model.device)
            if self._model.body_inertia is not None:
                body_inertia_np = self._model.body_inertia.numpy().copy()
                for i in zero_idx:
                    if np.max(np.abs(body_inertia_np[i])) < _MIN_INERTIA:
                        # Solid sphere: I = 2/5·m·r², r = 5 mm
                        r = 0.005
                        I_val = float((2.0 / 5.0) * _MIN_MASS * r * r)
                        body_inertia_np[i] = np.eye(3, dtype=np.float32) * I_val
                self._model.body_inertia = wp.array(body_inertia_np, dtype=wp.mat33, device=self._model.device)

        # Contact material (soft_contact_ke/kd/mu + soft_contact_margin) is set
        # in _phase_build_model (model.py) right after finalize and BEFORE the
        # plugin's on_model_ready hook, so scene plugins can override it; do NOT
        # re-assert it here or the plugin's FEM tuning would be clobbered
        # (noodles tunnel the wok). on_model_ready runs in phase 3, this is
        # phase 4 — re-setting here would always win over the plugin.

        # Per-shape friction is now driven by the per-class
        # ``newton.mjc_contact.<kind>.friction`` table inside
        # ``engine.newton.mjc_contact.apply_pre_finalize``, which writes
        # ``shape_material_mu`` (tangential), ``shape_material_mu_torsional``
        # and ``shape_material_mu_rolling`` separately.  The previous
        # scene-wide ``_ROBOT_KU`` broadcast that lived here was
        # silently overwriting that per-class write — and only set the
        # tangential slot, leaving torsional / rolling at their MuJoCo
        # defaults (0.005 / 0.0001), which is why fingers couldn't grip
        # round objects (the rod would just roll out of the jaw).
        # Removed entirely; tune via the yaml block instead.

        # ----------------------------------------------------------------
        # Collision-group unification.
        # ----------------------------------------------------------------
        # Newton's ModelBuilder assigns a fresh collision group per
        # ``add_usd`` / ``add_urdf`` call by default — pairs ACROSS groups
        # are filtered out of the rigid-vs-rigid contact pass.  In a
        # multi-source scene (robot URDF + scene USDA + sidecar USDAs from
        # scene-yaml ``newton.entries``) this means scene rigid bodies
        # like the hanger or fold-box NEVER collide with the robot, even
        # though both have CollisionAPI.
        #
        # Fix: collapse every shape into group 0 so the per-group filter
        # is a no-op.  Newton's contact pass still respects:
        #   * static-vs-static filtering (those pairs stay culled)
        #   * articulation self-collision flags
        #   * per-shape ``shape_collision_filter_pairs`` (untouched)
        # so we don't open a free-for-all — we only stop INTER-source
        # exclusion that the per-source group assignment created.
        #
        # Keeps cloth-vs-rigid unaffected (cloth uses a separate
        # broadphase that ignores collision groups regardless).
        if (
            getattr(self._model, "shape_collision_group", None) is not None
            and self._model.shape_collision_group.size > 0
        ):
            groups = self._model.shape_collision_group.numpy().copy()
            unique_before = sorted(set(int(g) for g in groups.tolist()))
            # groups[:] = 0
            self._model.shape_collision_group.assign(groups)
            self._logger.info(
                f"[lifecycle] collision-group unification: collapsed "
                f"{len(unique_before)} groups {unique_before!r} -> [0] "
                f"on {len(groups)} shapes (so cross-source bodies — "
                f"robot/scene/yaml-entries — can collide)"
            )

        # ----------------------------------------------------------------
        # Articulation unification (mjwarp ONLY).
        # ----------------------------------------------------------------
        # Newton's ``add_usd`` puts each free-floating ``RigidBodyAPI``
        # prim into its OWN articulation (separate from the URDF robot's
        # articulation), and MuJoCo-Warp's solver only steps the
        # articulations it was constructed with.  Result: passive scene
        # rigid bodies (hangers, dropped objects) with auto-created
        # FREE joints stay frozen at their spawn pose — gravity never
        # accelerates them, contacts never resolve.
        #
        # Fix on mjwarp: merge all articulations into one by collapsing
        # ``articulation_start`` to ``[0, joint_count]``.
        #
        # IMPORTANT: this is a NO-OP on the Featherstone path.  Newton's
        # contact pipeline uses articulation membership for
        # SELF-COLLISION filtering (pairs within the same articulation
        # are excluded from rigid-vs-rigid contact).  If we unify on
        # Featherstone, the robot's arm shapes and the hanger's shapes
        # land in the same articulation and become self-collision-filtered;
        # any init-pose overlap then produces unbounded penetration
        # impulses and the integrator NaNs on substep 1.  Featherstone
        # already steps every articulation regardless of count, so it
        # doesn't need the merge in the first place.
        adapter_name = getattr(self._adapter, "name", "")
        if adapter_name in ("mujoco-warp", "mujoco_warp"):
            try:
                artic_start = getattr(self._model, "articulation_start", None)
                artic_end = getattr(self._model, "articulation_end", None)
                joint_count = int(getattr(self._model, "joint_count", 0) or 0)
                if artic_start is not None and joint_count > 0:
                    import warp as _wp  # noqa: PLC0415

                    start_arr = artic_start.numpy()
                    n_art = len(start_arr)
                    # The articulation-membership representation differs by newton
                    # version, and the FK readback (``eval_articulation_fk``) walks
                    # whichever one is in use:
                    #   * newton 1.2.0 — CSR single array: ``articulation_start`` has
                    #     N+1 entries with ``joint_count`` as the trailing sentinel,
                    #     and FK reads the end as ``articulation_start[id + 1]``.
                    #     One articulation spanning everything = ``[0, joint_count]``.
                    #   * newton 1.3.0 — split arrays: ``articulation_start`` and
                    #     ``articulation_end`` each have N entries (one per
                    #     articulation), and FK reads ``articulation_end[id]``.
                    #     One articulation spanning everything = start ``[0]`` +
                    #     end ``[joint_count]``.
                    # We DETECT the scheme by whether ``articulation_end`` exists.
                    # Collapsing only ``articulation_start`` (the original code) is
                    # correct for 1.2.0 but leaves a STALE ``articulation_end`` on
                    # 1.3.0 — FK then stops at the first articulation's end and any
                    # late-added passive body (the FREE-joint wok, hangers, dropped
                    # objects) stays frozen at spawn even though MuJoCo round-trips
                    # its joint_q. ``joint_articulation`` is left as-is: real joints
                    # keep a valid owner (>=0, so FK processes them) while
                    # loop-closure joints stay -1 (correctly skipped by FK).
                    split_repr = artic_end is not None
                    if split_repr:
                        end_arr = artic_end.numpy()
                        spans_all = (
                            n_art == 1
                            and int(start_arr[0]) == 0
                            and len(end_arr) >= 1
                            and int(end_arr[0]) == joint_count
                        )
                    else:
                        # CSR: already unified iff start == [0, joint_count].
                        spans_all = n_art == 2 and int(start_arr[0]) == 0 and int(start_arr[1]) == joint_count
                    if not spans_all:
                        dev = self._model.device
                        if split_repr:
                            self._model.articulation_start = _wp.array(
                                np.array([0], dtype=np.int32), dtype=artic_start.dtype, device=dev
                            )
                            self._model.articulation_end = _wp.array(
                                np.array([joint_count], dtype=np.int32), dtype=artic_end.dtype, device=dev
                            )
                        else:
                            self._model.articulation_start = _wp.array(
                                np.array([0, joint_count], dtype=np.int32), dtype=artic_start.dtype, device=dev
                            )
                        if hasattr(self._model, "articulation_count"):
                            self._model.articulation_count = 1
                        self._logger.info(
                            f"[lifecycle] articulation unification (mjwarp): "
                            f"collapsed {n_art} articulations -> 1 spanning all "
                            f"{joint_count} joints ({'split start+end' if split_repr else 'CSR start'}"
                            f"+count; so passive FREE-joint scene bodies are FK'd + "
                            f"stepped by the solver)"
                        )
                    else:
                        self._logger.info(
                            f"[lifecycle] articulation unification (mjwarp): "
                            f"already 1 articulation spanning {joint_count} "
                            f"joints (no-op)"
                        )
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(f"[lifecycle] articulation unification (mjwarp) failed: {exc!r}")
        else:
            self._logger.info(
                f"[lifecycle] articulation unification: skipped on "
                f"adapter={adapter_name!r} (Featherstone steps every "
                f"articulation; merging would collapse self-collision "
                f"filtering and NaN the integrator on init-pose overlap)"
            )
