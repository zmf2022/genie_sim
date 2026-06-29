# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone joint-topology mixin.

Provides the ``_TopologyMixin`` class composed into
``_NewtonStandaloneBase`` via multiple inheritance — see
``engine_base.py`` for the full mixin order.  ``self.X`` references
resolve through the engine's MRO.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import warp as wp


class _TopologyMixin:
    def _build_joint_map(self) -> None:
        """Populate ``self._joint_names`` plus the per-joint lookup maps.

        Two distinct maps — see ``engine/newton/joint_index.py`` for
        the canonical explanation of why q-index and qd-index diverge:

          * ``self._joint_name_to_dof``  — qd / DOF index. Used by
            ``/joint_command``, ``/cmd_4ws``, ``/joint_states``
            velocity, the BASE-RUNAWAY diagnostic against
            ``joint_target_pos``, and the featherstone / AVBD
            controlled-DOF mask.
          * ``self._joint_name_to_q_idx`` — q / configuration index.
            Used only by the ``/joint_states`` position read out of
            ``state.joint_q``.

        Filters out joints that aren't user-controllable on the
        ROS-topic surface:

          * ``JointType.FREE``  — floating base, 6 DOFs, world joint
          * ``JointType.FIXED`` — weld, 0 DOFs, no state to publish
          * Auto-generated names (``root_joint``, ``tn__*``) — Newton's
            USD parser emits these for body-pair welds inherited from
            the URDF structure; they're noise in ``/joint_states``.
        """
        self._joint_names = []
        self._joint_name_to_dof = {}
        self._joint_name_to_q_idx = {}
        self._joint_prim_map = {}
        self._jindex = None
        if self._model is None:
            return

        from engine.newton.joint_index import JointIndex, JT_FIXED, JT_FREE  # noqa: PLC0415

        try:
            jindex = JointIndex(self._model)
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] _build_joint_map: JointIndex build failed: {exc}")
            return

        if not len(jindex):
            self._logger.warn(
                "[newton-standalone] _build_joint_map: model has no joint_label / joint_type / "
                "joint_q_start data — /joint_states will be empty"
            )
            return

        # Stash the full snapshot for downstream consumers (adapters,
        # state.py, control.py, anyone needing per-joint metadata
        # beyond the two name → index dicts below).
        self._jindex = jindex

        # Names we always filter out of /joint_states (auto-generated welds /
        # the URDF root joint that PhysX/isaac_newton don't publish either).
        SKIP_PREFIXES = ("tn__",)
        SKIP_EXACT = {"root_joint"}

        skipped: List[Tuple[str, str]] = []
        for s in jindex.slices():
            if not s.name:
                continue
            if s.joint_type == JT_FREE:
                skipped.append((s.name, "FREE"))
                continue
            if s.joint_type == JT_FIXED:
                skipped.append((s.name, "FIXED"))
                continue
            if any(s.name.startswith(p) for p in SKIP_PREFIXES) or s.name in SKIP_EXACT:
                skipped.append((s.name, "auto-name"))
                continue

            self._joint_names.append(s.name)
            self._joint_name_to_dof[s.name] = s.qd_start
            self._joint_name_to_q_idx[s.name] = s.q_start
            self._joint_prim_map[s.name] = s.label

        self._logger.info(
            f"[newton-standalone] joint map: {len(self._joint_names)} controllable joint(s) "
            f"from {len(jindex)} total Newton joints"
        )
        self._logger.info(f"[newton-standalone]   names: {self._joint_names}")
        if skipped:
            self._logger.info(
                f"[newton-standalone]   skipped {len(skipped)} (free/fixed/auto-name): "
                f"{[f'{n}({why})' for n, why in skipped[:8]]}"
                + (f" +{len(skipped)-8} more" if len(skipped) > 8 else "")
            )

        # Append synthetic entries for joints that fix_base / fix_head /
        # fix_body collapsed to FixedJoint.  Newton's add_usd already
        # treats them as Fixed (zero DOFs in the mass matrix), but the
        # URDF still calls them revolute, so robot_state_publisher / RViz
        # need to see them in ``/joint_states`` to compute consistent TF
        # for everything downstream.  These names have NO entry in
        # ``self._joint_name_to_dof`` — that's the marker ``state.py``
        # uses to know it should pull the value from
        # ``self._static_joint_q`` instead of ``state.joint_q``.
        static_q = getattr(self, "_static_joint_q", None) or {}
        if static_q:
            n_active = len(self._joint_names)
            for name in static_q:
                if name not in self._joint_name_to_dof and name not in self._joint_names:
                    self._joint_names.append(name)
            n_static = len(self._joint_names) - n_active
            if n_static > 0:
                self._logger.info(
                    f"[newton-standalone]   plus {n_static} synthetic joint(s) "
                    f"(fix_*-collapsed; published to /joint_states at static "
                    f"value from self._static_joint_q so robot_state_publisher "
                    f"can compute TF for the URDF-revolute kinematic chain)"
                )

    def _apply_init_joint_pos(self) -> None:
        """Set initial joint positions on ``model.joint_q`` AND
        ``control.joint_target_pos`` so the Featherstone PD drive holds them.

        ``self._init_joint_pos`` is a dict like ``{'arm_joint1': 0.0,
        'arm_joint2': -90.0, ...}`` from the launcher's scene YAML.

        UNIT CONVENTION (matches the PhysX/isaac_newton engines):
          * revolute  → DEGREES in the yaml → converted to radians here.
          * prismatic → METRES in the yaml  → used as-is.

        This is the SAME convention the PhysX path uses (USD's
        ``drive:angular:physics:targetPosition`` is degrees natively, so the
        yaml carries degrees and PhysX writes them through to USD verbatim).

        Without this, ``model.joint_q`` is whatever Newton parsed from the
        URDF defaults (typically zeros) and the robot starts collapsed.

        Diagnostic logging
        ------------------
        The launch chain has several places where init_pose can fall
        on the floor (yaml not reaching the engine, name mismatch
        between yaml keys and Newton's joint_label rsplit, joint type
        misclassified so degrees aren't converted, ``model.joint_q``
        being None at apply time, the post-assign value being clobbered
        before the keyframe / target-buffer init reads it).  Each
        ``[diag-init]`` line below isolates one of those — if init pose
        ends up wrong, comparing the logs to the final ``/joint_states``
        snapshot points at the exact stage.
        """
        # [diag-init] Entry state — what came in.  If _init_joint_pos is
        # empty or missing keys you expect, the failure is upstream
        # (scene yaml not parsed, init_joint_pos_json not plumbed to
        # the engine node, parse_init_joint_pos returning {}).
        ip = self._init_joint_pos or {}
        sample_keys = list(ip.keys())[:8]
        self._logger.info(
            f"[diag-init] _apply_init_joint_pos entry: "
            f"_init_joint_pos={len(ip)} key(s) (sample: {sample_keys}); "
            f"_joint_name_to_dof={len(self._joint_name_to_dof)} key(s); "
            f"_model is {'set' if self._model is not None else 'None'}; "
            f"_control is {'set' if self._control is not None else 'None'}"
        )
        if not self._init_joint_pos or self._model is None:
            self._logger.info(
                "[diag-init] EARLY-RETURN: no init_joint_pos or no model — "
                "nothing to apply; joints start at URDF defaults (typically zeros)."
            )
            return
        try:
            jq_attr = self._model.joint_q
            tgt_attr_ctl = self._control.joint_target_pos if self._control is not None else None
            jq = jq_attr.numpy().copy() if jq_attr is not None else None
            tgt = tgt_attr_ctl.numpy().copy() if tgt_attr_ctl is not None else None

            # [diag-init] Buffer state at entry.
            self._logger.info(
                f"[diag-init] buffers at entry: "
                f"model.joint_q size={len(jq) if jq is not None else 'None'}, "
                f"control.joint_target_pos size={len(tgt) if tgt is not None else 'None'}, "
                f"jindex={'set' if self._jindex is not None else 'None'} "
                f"({len(self._jindex) if self._jindex is not None else 0} joint(s))"
            )

            # ``JointIndex`` carries everything we need per joint —
            # the name → JointSlice map plus the joint type for the
            # degrees-vs-radians decision.  No more parallel lookup
            # tables.
            jindex = self._jindex
            if jindex is None:
                self._logger.warn(
                    "[diag-init] EARLY-RETURN: _jindex is None — "
                    "_build_joint_map didn't run; init_joint_pos has no slice metadata to land on."
                )
                return

            applied = []
            unknown = []
            skipped_static: List[str] = []
            per_joint_trace: List[str] = []
            DEG2RAD = 3.14159265358979323846 / 180.0
            static_q = getattr(self, "_static_joint_q", None) or {}
            for name, raw in self._init_joint_pos.items():
                slc = jindex.get(name)
                if slc is None or not slc.is_actuated:
                    # If the joint was collapsed to FixedJoint by fix_*,
                    # the init pose is already baked into the FixedJoint's
                    # local frame and the value is published synthetically
                    # via /joint_states from ``self._static_joint_q``.
                    # Skip silently — this is the expected path, not an
                    # error.
                    if name in static_q:
                        skipped_static.append(name)
                        continue
                    unknown.append(name)
                    continue
                # ``raw`` is a JointInitSpec dataclass (frozen, value: float)
                # produced by parse_init_joint_pos.  Accept both the
                # dataclass and a bare float so the engine survives a
                # programmatic dict.
                if hasattr(raw, "value"):
                    raw_val = float(raw.value)
                else:
                    raw_val = float(raw)
                # Newton JointType.REVOLUTE = 1 → yaml is degrees → convert.
                # JointType.PRISMATIC = 0 → yaml is metres → use as-is.
                if slc.joint_type == 1:  # revolute — yaml is degrees
                    v = raw_val * DEG2RAD
                    unit_note = f"deg→rad ×{DEG2RAD:.6f}"
                else:
                    v = raw_val
                    unit_note = f"jtype={slc.joint_type} (no deg→rad conversion)"
                # ``joint_q`` is q-indexed (FREE joints carry an extra
                # coord vs the qd vector); ``joint_target_pos`` is
                # qd-indexed.  Use the right index for each.
                if jq is not None and slc.q_start < len(jq):
                    jq[slc.q_start] = v
                if tgt is not None and slc.qd_start < len(tgt):
                    tgt[slc.qd_start] = v
                applied.append(f"{name}={v:.4f}")
                per_joint_trace.append(
                    f"  {name}: q_idx={slc.q_start} qd_idx={slc.qd_start} "
                    f"ji={slc.joint_index} raw={raw_val} {unit_note} → {v:.6f}"
                )

            # [diag-init] Pre-assign trace — everything we resolved.
            if per_joint_trace:
                self._logger.info(
                    f"[diag-init] resolved {len(per_joint_trace)} entry/entries:\n" + "\n".join(per_joint_trace)
                )

            if jq is not None:
                self._model.joint_q.assign(jq)
            if tgt is not None and self._control is not None:
                self._control.joint_target_pos.assign(tgt)

            # [diag-init] Post-assign verification — read joint_q +
            # joint_target_pos BACK from the live Newton buffers and
            # confirm the values landed at the right indices.  If
            # ``post-assign`` shows zeros for joints we just wrote, the
            # ``.assign()`` no-oped (size mismatch, Warp-array type
            # mismatch, or stale buffer reference).  If it shows the
            # right values here but the eventual ``/joint_states``
            # publishes zeros, something downstream (init_target_buffer
            # size-slice path, a state reset, a runtime over-write) is
            # clobbering between this point and the publish stage.
            try:
                jq_post = self._model.joint_q.numpy()
                tgt_post = (
                    self._control.joint_target_pos.numpy()
                    if self._control is not None and self._control.joint_target_pos is not None
                    else None
                )
                check_lines: List[str] = []
                for name in ip.keys():
                    slc = jindex.get(name) if jindex is not None else None
                    if slc is None or slc.q_start >= len(jq_post):
                        continue
                    jq_v = float(jq_post[slc.q_start])
                    tgt_v = (
                        float(tgt_post[slc.qd_start]) if tgt_post is not None and slc.qd_start < len(tgt_post) else None
                    )
                    check_lines.append(
                        f"  {name}[q={slc.q_start}/qd={slc.qd_start}]: joint_q={jq_v:.6f}"
                        + (f"  joint_target_pos={tgt_v:.6f}" if tgt_v is not None else "  joint_target_pos=N/A")
                    )
                if check_lines:
                    self._logger.info(
                        f"[diag-init] post-assign readback for {len(check_lines)} init joint(s):\n"
                        + "\n".join(check_lines)
                    )
            except Exception as exc:  # noqa: BLE001
                self._logger.warn(f"[diag-init] post-assign readback failed: {exc}")

            if applied:
                self._logger.info(
                    f"[newton-standalone] init_joint_pos applied to {len(applied)} DOF(s) "
                    f"(yaml-degrees → radians for revolute): "
                    f"{', '.join(applied)}"
                )
            if skipped_static:
                self._logger.info(
                    f"[diag-init] skipped {len(skipped_static)} static (fix_*-collapsed) joint(s): " f"{skipped_static}"
                )
            if unknown:
                self._logger.warn(
                    f"[newton-standalone] init_joint_pos: unknown joint(s) ignored: "
                    f"{unknown}; known: {list(self._joint_name_to_dof.keys())}"
                )
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] _apply_init_joint_pos: {exc}")

    def _configure_pd_drives(self) -> None:
        """Force POSITION-mode PD drives on every controllable revolute /
        prismatic DOF.

        Featherstone needs three things to actually exert torque toward
        ``control.joint_target_pos``:

          * ``model.joint_target_mode[i] = JointTargetMode.POSITION`` (=1)
          * ``model.joint_target_ke[i]   > 0``  (proportional gain / stiffness)
          * ``model.joint_target_kd[i]   > 0``  (derivative gain / damping)

        When the source URDF doesn't author ``physxJoint:driveStiffness`` etc.
        — or when the asset pipeline strips them — Newton's ``add_usd`` leaves
        all three at zero. Setting target_pos then has no effect: the joint
        floats freely under gravity and inertia.

        We log what add_usd produced and force defaults wherever they're zero.
        Defaults are intentionally on the higher side (200 / 20) so a 7-kg
        robot arm tracks a step input within a fraction of a second; tune
        per-robot via URDF if you need softer behaviour.
        """
        if self._model is None:
            return
        try:
            mode_arr = self._model.joint_target_mode
            ke_arr = self._model.joint_target_ke
            kd_arr = self._model.joint_target_kd
            if mode_arr is None or ke_arr is None or kd_arr is None:
                self._logger.warn(
                    "[newton-standalone] joint_target_{mode,ke,kd} array is None; "
                    "Featherstone will not apply PD drives"
                )
                return

            mode = mode_arr.numpy().copy()
            ke = ke_arr.numpy().copy()
            kd = kd_arr.numpy().copy()

            DEFAULT_KE = 1000.0
            DEFAULT_KD = 100.0

            forced = []
            for name, idx in self._joint_name_to_dof.items():
                if idx is None or idx >= len(mode):
                    continue
                changed = []
                if int(mode[idx]) != 1:
                    mode[idx] = 1  # POSITION
                    changed.append(f"mode→POS")
                if float(ke[idx]) <= 0.0:
                    ke[idx] = DEFAULT_KE
                    changed.append(f"ke←{DEFAULT_KE:g}")
                if float(kd[idx]) <= 0.0:
                    kd[idx] = DEFAULT_KD
                    changed.append(f"kd←{DEFAULT_KD:g}")
                if changed:
                    forced.append(f"{name}({','.join(changed)})")

            mode_arr.assign(mode)
            ke_arr.assign(ke)
            kd_arr.assign(kd)

            n_pd = sum(
                1
                for n, i in self._joint_name_to_dof.items()
                if i is not None and i < len(mode) and mode[i] == 1 and ke[i] > 0
            )
            self._logger.info(
                f"[newton-standalone] PD drives: {n_pd}/{len(self._joint_name_to_dof)} "
                f"controllable DOF(s) have POSITION-mode + ke>0 + kd>0"
            )
            if forced:
                self._logger.info(
                    f"[newton-standalone]   forced defaults on {len(forced)} DOF(s) "
                    f"(URDF didn't author drive params): "
                    + ", ".join(forced[:8])
                    + (f", +{len(forced)-8} more" if len(forced) > 8 else "")
                )
            # Per-joint summary for the first few (helps diagnose tuning)
            for name in list(self._joint_name_to_dof.keys())[:6]:
                idx = self._joint_name_to_dof[name]
                if 0 <= idx < len(mode):
                    self._logger.info(
                        f"[newton-standalone]   {name:32s}  mode={int(mode[idx])}  "
                        f"ke={float(ke[idx]):8.1f}  kd={float(kd[idx]):8.1f}"
                    )
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] _configure_pd_drives: {exc}")

    def _build_mimic_map(self) -> None:
        """Read mimic relations from the staged USD and stash them for
        the runtime broadcast in :class:`_ControlMixin`.

        Why we have to do this manually under newton-standalone: Newton's
        Featherstone solver explicitly does NOT support mimic / equality
        constraints (see its own docstring). For the Robotiq-85 gripper
        ``robot.urdf`` declares 5 followers all driven by
        ``gripper_active_master_joint`` with multipliers ±1; if we only
        write the master's ``joint_target_pos`` every follower keeps its
        target at 0 and the inertial chain locks the master in place —
        the gripper looks completely unresponsive even though the
        command did arrive.

        The PhysX path side-steps this with ``PhysxMimicJointAPI`` (a
        hard kinematic constraint authored on the followers). For Newton
        we read ``NewtonMimicAPI`` from USD instead — same data, written
        by the same URDF importer, but already on the live stage so we
        don't need ``robot.urdf`` to be present at runtime. The shared
        parser lives in :mod:`engine._mimic` and is also used by the
        ``isaac_newton`` engine.

        Populates:
          * ``self._mimic_followers``: ``{master: [(name, coef1, coef0), …]}``
          * ``self._mimic_master_of``:  ``{follower: (master, coef1, coef0)}``
            (currently unused but cheap to maintain — useful if we ever
             want to read follower state and relate it back to the master)
        """
        from engine._mimic import parse_mimic

        self._mimic_followers: Dict[str, List[Tuple[str, float, float]]] = {}
        self._mimic_master_of: Dict[str, Tuple[str, float, float]] = {}
        if self._stage is None:
            return
        try:
            self._mimic_followers = parse_mimic(self._stage, self._logger)
            for master, followers in self._mimic_followers.items():
                for fname, mult, off in followers:
                    self._mimic_master_of[fname] = (master, mult, off)
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] _build_mimic_map: {exc}")

    def _build_body_map(self) -> None:
        self._body_paths = list(getattr(self._model, "body_label", []) or [])
