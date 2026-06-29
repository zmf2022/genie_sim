# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phase 1: USD stage open + pre-``add_usd`` session-layer overrides.

Owns: stage open template, runtime fix policies (``pin_base_to_world``,
``convert_joints_to_fixed``, ``init_base_pose``), and the
RenderProduct disable cleanup.  Every method here mutates the
composed USD stage; ``add_usd`` runs in the next phase (Model).
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


class _StageMixin:
    def _open_stage(self, newton_scene) -> None:
        """Open the USD stage.  MUST be overridden."""
        raise NotImplementedError("_open_stage must be implemented by the engine subclass")

    def _disable_local_render_products(self) -> None:
        """Deactivate UsdRender.Product prims in the session layer.

        Edits go to the session layer so they are scoped to this process only.
        Uses pure pxr APIs — safe in both headless and Kit-backed modes.
        In headless mode the render layer is never loaded so this is a no-op
        in practice; the call is retained as a defensive cleanup.
        """
        try:
            from pxr import Usd, UsdRender

            session_layer = self._stage.GetSessionLayer()
            disabled = []
            with Usd.EditContext(self._stage, session_layer):
                for prim in self._stage.Traverse():
                    if prim.IsA(UsdRender.Product):
                        prim.SetActive(False)
                        disabled.append(str(prim.GetPath()))
            if disabled:
                self._logger.info(
                    f"[newton-standalone] disabled {len(disabled)} RenderProduct(s) "
                    f"locally (session layer): {disabled}"
                )
            else:
                self._logger.info("[newton-standalone] no RenderProduct prims found to disable")
        except Exception as exc:
            self._logger.warn(f"[newton-standalone] _disable_local_render_products failed: {exc}")

    def _apply_init_base_pose(self) -> None:
        """Apply ``scene.robot.init_base_pose`` to the robot Xform.

        YAML shape::

            robot:
              init_base_pose:
                x: 0.0           # world meters
                y: 0.0
                z: 0.04
                theta: 0.0       # yaw radians about world Z

        Authors a translate + orient xform op on ``/<robot_prefix>`` at
        session-layer strength.  Both ``pin_base_to_world`` modes consume
        this transparently:

          * ``pin_base_to_world: true``  — the world-weld FixedJoint stays;
            it pins ``base_link`` to ``/<robot_prefix>`` (which is now
            translated/rotated to the requested pose), so the welded
            base sits at ``init_base_pose`` in world.
          * ``pin_base_to_world: false`` — ``_deactivate_root_joint``
            drops the weld; Newton's ``parse_usd`` then adds a FREE base
            joint initialized to ``base_link``'s composed world pose,
            which is ``/<robot_prefix>``'s pose (since base_link itself
            has no local offset).

        Existing xform ops authored on the cached ``robot.usda`` layer
        stay in their layer untouched; we append a session-layer
        ``xformOp:translate:init_base`` + ``xformOp:orient:init_base``
        with a fresh xformOpOrder that names ONLY our two ops, so the
        composed transform is exactly the requested pose (the cached
        layer's ops are excluded from the order rather than overwritten).

        No-op when the yaml block is absent or non-mapping — callers
        get the cached ``robot.usda`` pose unchanged.
        """
        if self._stage is None:
            return
        base_pose = ((self._scene_cfg or {}).get("robot") or {}).get("init_base_pose")
        if not isinstance(base_pose, dict):
            return
        try:
            x = float(base_pose.get("x", 0.0))
            y = float(base_pose.get("y", 0.0))
            z = float(base_pose.get("z", 0.0))
            theta = float(base_pose.get("theta", 0.0))
        except (TypeError, ValueError) as exc:
            self._logger.warn(f"[newton-standalone] init_base_pose: numeric parse failed ({exc!r}); skipping")
            return
        try:
            import math  # noqa: PLC0415
            from pxr import Gf, Sdf, UsdGeom  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] init_base_pose: pxr import failed ({exc!r}); skipping")
            return

        root_path = Sdf.Path(f"/{self._robot_prefix_str}")
        root_prim = self._stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            self._logger.warn(f"[newton-standalone] init_base_pose: robot root {root_path} " f"not on stage; skipping")
            return

        xformable = UsdGeom.Xformable(root_prim)
        if not xformable:
            self._logger.warn(f"[newton-standalone] init_base_pose: {root_path} is not " f"Xformable; skipping")
            return

        # AddTranslateOp / AddOrientOp with a unique suffix author attributes
        # named ``xformOp:translate:init_base`` and ``xformOp:orient:init_base``
        # so they coexist with any existing ``xformOp:translate`` on a lower
        # layer.  SetXformOpOrder([t_op, o_op]) then names ONLY our two ops
        # as participating in the composed transform — the cached layer's
        # ops drop out of the order and contribute nothing.
        t_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "init_base")
        t_op.Set(Gf.Vec3d(x, y, z))
        half = 0.5 * theta
        quat = Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, 0.0, math.sin(half)))
        o_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "init_base")
        o_op.Set(quat)
        xformable.SetXformOpOrder([t_op, o_op])

        self._logger.info(
            f"[newton-standalone] init_base_pose applied to {root_path}: "
            f"xyz=({x:.4f}, {y:.4f}, {z:.4f}) yaw={theta:.4f} rad"
        )

    def _apply_runtime_fix_policies(self) -> None:
        """Apply ``pin_base_to_world`` + ``convert_joints_to_fixed``
        overrides on the composed stage before ``add_usd`` parses it.

        ``pin_base_to_world``
          IsaacSim's URDF→USD converter always emits a
          ``PhysicsFixedJoint "root_joint"`` between ``/robot`` (the
          Xform — Newton resolves as world) and ``base_link``, welding
          the base.  When the flag is ``True`` we leave it alone; when
          ``False`` we deactivate the prim so Newton's importer adds a
          FREE joint instead and the base is mobile under physics.

        ``convert_joints_to_fixed: [base, head, body, arm, gripper, no_robot]``
          A list of sub-tree tokens.  Each present token triggers
          ``_replace_joints_with_fixed`` over a specific name-substring,
          except ``no_robot`` which deactivates the whole robot Xform:

            * ``"base"`` → joints whose name contains ``chassis``
              (chassis attach + every wheel steering / spin joint).
              Chassis link + wheel links stay rigid for visualization.
            * ``"head"`` → joints whose name contains ``head_joint``
              (3 head revolute joints).
            * ``"body"`` → joints whose name contains ``body_joint``
              (G2's 5-DOF torso chain).
            * ``"arm"`` → joints whose name contains ``arm_``
              (left + right arm revolute joints).  Welds the entire
              arm chain to its init pose — robot becomes a frozen
              statue from the shoulder down.
            * ``"gripper"`` → joints whose name contains ``gripper_``
              (gripper master + mimic followers).
            * ``"no_robot"`` → deactivates ``/<robot_prefix_str>``
              entirely on the session layer.  ``add_usd`` then never
              sees the robot at all — the model contains only scene
              entries (cloth, hanger, table, ground).  Useful for
              isolating "is the rigid solver even doing rigid contact
              correctly" tests, or for AVBD scenes where the robot
              would force unsupported joint types into the model.
              Other welding tokens become no-ops in this case (their
              substring matches return zero joints).

          Bodies on both sides stay in ``model.body_label`` so TF
          publishes them at their init pose; the joints contribute 0
          DOFs to Featherstone's M matrix / mjwarp's actuator count.

        Joints that are already typed ``PhysicsFixedJoint`` (e.g. the
        URDF-fixed ``idx100_chassis_base_joint``) get skipped by the
        replacement helper — they're already cost-free.

        All edits are USD session-layer overrides on the live composed
        stage; nothing touches the cached ``robot.usda`` payloads, so
        these flags toggle at zero rebuild cost.
        """
        if self._stage is None:
            return
        try:
            from pxr import UsdPhysics  # noqa: F401  (verify USD bindings available)
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] runtime fix policies skipped (pxr import failed): {exc}")
            return

        if not getattr(self, "_pin_base_to_world", False):
            self._deactivate_root_joint()

        # Map normalized sub-tree token -> joint-name substring used to
        # find matching joints in _replace_joints_with_fixed.  Add new
        # rows here when a robot has a new welded sub-chain (e.g. legs).
        #
        # ``arm`` / ``gripper`` exist primarily for the AVBD path: this
        # Newton version's SolverVBD doesn't support REVOLUTE rigid
        # joints (only CABLE/BALL/FIXED), so to test AVBD on a scene
        # that contains an articulated robot we have to weld every
        # revolute joint to FIXED.  The robot becomes a frozen statue
        # — visible in the viewer, present as a static collider, but
        # no actuation, no joint motion.  Use only when you specifically
        # need the AVBD comparison; the Featherstone+VBD path runs
        # revolute joints natively and shouldn't have these tokens set.
        _SUBSTRING_BY_TOKEN = {
            "base": "chassis",
            "head": "head_joint",
            "body": "body_joint",
            "arm": "arm_",
            "gripper": "gripper_",
        }
        tokens = list(getattr(self, "_convert_joints_to_fixed", []) or [])

        # ``no_robot`` is the only token that doesn't map to a joint-name
        # substring.  Handle it first — deactivating the robot's top-level
        # Xform makes the other tokens' joint substring matches return
        # nothing, which logs cleanly as "0 replaced".
        if "no_robot" in tokens:
            self._deactivate_robot_xform()
            tokens = [t for t in tokens if t != "no_robot"]

        for token in tokens:
            sub = _SUBSTRING_BY_TOKEN.get(token)
            if sub is None:
                self._logger.warn(
                    f"[newton-standalone] convert_joints_to_fixed: unknown "
                    f"token {token!r}; valid: {sorted(_SUBSTRING_BY_TOKEN) + ['no_robot']}.  "
                    f"Skipped."
                )
                continue
            self._replace_joints_with_fixed(name_substring=sub, label=f"convert_joints_to_fixed[{token}]")

    def _deactivate_robot_xform(self) -> None:
        """Deactivate ``/<robot_prefix_str>`` on the session layer so
        Newton's ``add_usd`` never registers any robot prim — bodies,
        joints, shapes — into the model.

        Used by ``convert_joints_to_fixed: [no_robot]`` to test
        cloth + scene-rigid pipelines in isolation, without the robot's
        joint types or contact pairs interfering.  Session-layer scoped
        — the cached ``robot.usda`` is untouched.
        """
        if self._stage is None:
            return
        path = f"/{self._robot_prefix_str}"
        prim = self._stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._logger.warn(
                f"[newton-standalone] convert_joints_to_fixed[no_robot]: "
                f"couldn't locate robot Xform at {path}; nothing to deactivate"
            )
            return
        prim.SetActive(False)
        self._logger.info(
            f"[newton-standalone] convert_joints_to_fixed[no_robot]: "
            f"deactivated {path} (robot will not appear in model — "
            f"scene runs with only cloth/hanger/static colliders)"
        )

    def _deactivate_root_joint(self) -> int:
        """Deactivate the URDF→USD-authored world-weld FixedJoint so
        Newton's ``parse_usd`` adds a FREE joint at the base instead,
        making the URDF root mobile under physics.

        Both URDF importers MAY author this weld but at different prim
        paths:

          * Isaac Sim 6.0 (``urdf_usd_converter`` + Asset Structure 3.0):
            ``/<robot_prefix>/Joints/root_joint``
          * Isaac Sim 4.x/5.x (``URDFParseAndImportFile``): the path is
            importer-named and varies per layout, but the topology is
            the same — a ``UsdPhysics.FixedJoint`` whose ``body0`` is the
            articulation root prim and ``body1`` is the topmost
            ``RigidBodyAPI``-bearing descendant (the URDF root link).

        Detect by topology rather than by name, matching the same
        heuristic used by ``kit/stage.py::_apply_fix_base_policy``.
        Setting ``prim.SetActive(False)`` on every match removes them
        from the composed-stage view ``add_usd`` sees; Newton then
        treats the root link as an articulation root with no parent
        joint and ``_add_base_joint(floating=None)`` (the default)
        creates a FREE joint for it.

        Returns the number of welds deactivated.
        """
        if self._stage is None:
            return 0
        try:
            from pxr import Sdf, Usd, UsdPhysics  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(
                f"[newton-standalone] pin_base_to_world=False: pxr import failed ({exc}); "
                f"cannot locate URDF world-weld joint."
            )
            return 0

        robot_root_path = Sdf.Path(f"/{self._robot_prefix_str}")
        root_prim = self._stage.GetPrimAtPath(robot_root_path)
        if not root_prim or not root_prim.IsValid():
            self._logger.warn(
                f"[newton-standalone] pin_base_to_world=False: robot root prim "
                f"{robot_root_path} not found; cannot identify world-weld joint."
            )
            return 0

        # Topmost RigidBodyAPI descendants — the URDF root link(s).
        urdf_root_link_paths = []
        for desc in Usd.PrimRange(root_prim):
            if desc == root_prim:
                continue
            if not desc.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            ancestor = desc.GetParent()
            is_topmost = True
            while ancestor and ancestor.IsValid() and ancestor.GetPath() != robot_root_path:
                if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                    is_topmost = False
                    break
                ancestor = ancestor.GetParent()
            if is_topmost:
                urdf_root_link_paths.append(desc.GetPath())

        if not urdf_root_link_paths:
            self._logger.warn(
                f"[newton-standalone] pin_base_to_world=False: no RigidBodyAPI "
                f"descendant under {robot_root_path}; cannot identify URDF root link. "
                f"Base will be free by Newton's default in that case."
            )
            return 0

        found = 0
        for prim in Usd.PrimRange(root_prim):
            if not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.FixedJoint(prim)
            b0 = joint.GetBody0Rel().GetTargets()
            b1 = joint.GetBody1Rel().GetTargets()
            if not b0 or not b1:
                continue
            if b0[0] != robot_root_path:
                continue
            if b1[0] not in urdf_root_link_paths:
                # Internal fixed joint (body1 is a deeper-nested rigid body) —
                # leave strictly alone, regardless of pin_base_to_world.
                continue
            prim.SetActive(False)
            found += 1
            self._logger.info(
                f"[newton-standalone] pin_base_to_world=False → deactivated "
                f"world-weld {prim.GetPath()} (body0={b0[0]}, body1={b1[0]}). "
                f"Newton will add a FREE base joint; pair with the plain "
                f"rigid-substep regime (real gravity + contacts)."
            )

        if found == 0:
            self._logger.warn(
                f"[newton-standalone] pin_base_to_world=False: no world-weld "
                f"FixedJoint detected under {robot_root_path} (body0=root, "
                f"body1=URDF root link). Likely a ghost-root URDF — see "
                f"diagnose_urdf.py. Base will be free by Newton's default."
            )
        return found

    def _replace_joints_with_fixed(self, *, name_substring: str, label: str) -> int:
        """Replace every joint whose name contains ``name_substring`` with
        a ``UsdPhysics.FixedJoint``.

        The original joint prim is deactivated; the new FixedJoint is
        authored as a sibling under the same ``/robot/Physics/...``
        scope.  ``add_usd`` only sees the new FixedJoint and treats the
        connection as a static weld:

          * bodies on both sides remain rigid bodies and STAY in
            ``model.body_label`` — TF publishing emits their poses every
            tick (poses are constant because the joint can't move).
          * the joint contributes ZERO DOFs to Featherstone's mass
            matrix.

        For revolute joints with a non-zero ``init_joint_pos`` entry,
        the init angle is BAKED INTO the FixedJoint's ``localRot0``.
        That preserves the user-requested pose without giving the joint
        a DOF.  The same value (in radians) is recorded into
        ``self._static_joint_q`` so ``/joint_states`` publishes it
        synthetically and ``robot_state_publisher``'s URDF FK matches
        Newton's body_q for everything downstream.

        Net effect: ``fix_*`` flags shrink Featherstone's M(q) and
        nothing else — URDF, /joint_states shape, TF tree, RViz, init
        pose semantics all behave as if the flag weren't there.

        Returns the count of joints replaced.
        """
        if self._stage is None:
            return 0
        try:
            from pxr import Gf, Sdf, UsdPhysics
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] {label}: USD import failed: {exc}")
            return 0

        # Build a name → init-radians lookup once.  YAML stores revolute
        # init poses in degrees (matches PhysX convention); convert here
        # so the bake math is unit-clean.
        DEG2RAD = math.pi / 180.0
        init_rad_by_name: Dict[str, float] = {}
        ip = getattr(self, "_init_joint_pos", None) or {}
        if isinstance(ip, dict):
            for nm, raw in ip.items():
                try:
                    v = float(raw.value if hasattr(raw, "value") else raw)
                except (TypeError, ValueError):
                    continue
                # Revolute → degrees.  Prismatic init poses (metres) get
                # baked verbatim below as a translation; non-zero values
                # there would need a different code path (translation of
                # localPos0), which we don't currently exercise — flagged
                # in the bake site below.
                init_rad_by_name[nm] = v * DEG2RAD

        _AXIS_VEC = {
            "X": Gf.Vec3f(1.0, 0.0, 0.0),
            "Y": Gf.Vec3f(0.0, 1.0, 0.0),
            "Z": Gf.Vec3f(0.0, 0.0, 1.0),
        }

        n = 0
        n_baked = 0
        sample = []
        baked_sample = []
        try:
            # Two-step: collect first, mutate after — mutating mid-traverse
            # can invalidate the prim iterator on some USD versions.
            candidates = []
            for prim in self._stage.Traverse():
                if not (prim.IsA(UsdPhysics.Joint) or prim.HasAPI(UsdPhysics.Joint)):
                    continue
                if name_substring not in prim.GetName():
                    continue
                # Skip if already a FixedJoint or if the prim is somehow inactive.
                if prim.GetTypeName() == "PhysicsFixedJoint":
                    continue
                if not prim.IsActive():
                    continue
                candidates.append(prim)

            for prim in candidates:
                joint_name = prim.GetName()

                joint = UsdPhysics.Joint(prim)
                body0_targets = list(joint.GetBody0Rel().GetTargets())
                body1_targets = list(joint.GetBody1Rel().GetTargets())
                if not body0_targets or not body1_targets:
                    # Defensive — a degenerate joint with no bodies isn't
                    # useful to replace.  Just deactivate so add_usd
                    # doesn't get confused later.
                    prim.SetActive(False)
                    continue

                lp0 = joint.GetLocalPos0Attr().Get()
                lr0 = joint.GetLocalRot0Attr().Get()
                lp1 = joint.GetLocalPos1Attr().Get()
                lr1 = joint.GetLocalRot1Attr().Get()

                # Bake the init pose into the FixedJoint's local frame
                # for revolute joints.  Math:
                #   At joint angle θ, body1's frame in body0's frame is
                #     localTransform0 * Rotate_in_joint_frame(axis, θ)
                #     * localTransform1.Inverse()
                #   So the equivalent FixedJoint has
                #     new_localRot0 = localRot0 * Quat(axis, θ)
                #   while localPos0 / localPos1 / localRot1 stay
                #   unchanged.  Pixar Gf.Quatf uses (q1 * q2)(v) = q1(q2(v))
                #   composition — see ``Gf.Quatf.__mul__`` — so the
                #   left-multiply applies the existing localRot0 last,
                #   matching the joint's "rotate around the joint axis
                #   in body0's joint frame" semantics.
                init_rad = init_rad_by_name.get(joint_name, 0.0)
                if init_rad != 0.0 and prim.GetTypeName() == "PhysicsRevoluteJoint":
                    rev = UsdPhysics.RevoluteJoint(prim)
                    axis_token = rev.GetAxisAttr().Get()
                    axis = _AXIS_VEC.get(axis_token) if axis_token else None
                    if axis is not None:
                        half = init_rad * 0.5
                        bake_q = Gf.Quatf(
                            math.cos(half),
                            axis * math.sin(half),
                        )
                        lr0 = (Gf.Quatf(lr0) * bake_q) if lr0 is not None else bake_q
                        n_baked += 1
                        if len(baked_sample) < 4:
                            baked_sample.append(f"{joint_name}@{math.degrees(init_rad):+.1f}°")

                # New FixedJoint as a sibling under the same parent scope
                # (typically /robot/Physics/).
                src_path = prim.GetPath()
                new_path = Sdf.Path(f"{src_path.GetParentPath().pathString}/{src_path.name}__fixed")
                new_joint = UsdPhysics.FixedJoint.Define(self._stage, new_path)
                new_joint.GetBody0Rel().SetTargets(body0_targets)
                new_joint.GetBody1Rel().SetTargets(body1_targets)
                if lp0 is not None:
                    new_joint.GetLocalPos0Attr().Set(lp0)
                if lr0 is not None:
                    new_joint.GetLocalRot0Attr().Set(lr0)
                if lp1 is not None:
                    new_joint.GetLocalPos1Attr().Set(lp1)
                if lr1 is not None:
                    new_joint.GetLocalRot1Attr().Set(lr1)

                prim.SetActive(False)
                n += 1
                # Synthetic /joint_states value.  Always the init pose
                # value (radians) — for joints without an init_pose entry,
                # this is 0 and matches the un-rotated FixedJoint we just
                # authored; for joints with one, this matches the angle
                # we baked into ``localRot0`` so the URDF FK in
                # robot_state_publisher reproduces Newton's body_q
                # transform exactly.
                self._static_joint_q[joint_name] = init_rad
                if len(sample) < 4:
                    sample.append(src_path.name)

            if n > 0:
                msg = (
                    f"[newton-standalone] {label}=True: replaced {n} joint(s) "
                    f"containing {name_substring!r} with UsdPhysics.FixedJoint "
                    f"(bodies retained for TF; {n} DOF(s) removed from M); "
                    f"sample={sample}"
                )
                if n_baked > 0:
                    msg += (
                        f"; baked init pose into {n_baked} FixedJoint local "
                        f"frame(s) (sample={baked_sample}) — published "
                        f"synthetically via /joint_states"
                    )
                self._logger.info(msg)
            else:
                self._logger.info(
                    f"[newton-standalone] {label}=True: no active joints containing "
                    f"{name_substring!r} on the stage (no-op)"
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] {label} policy failed: {exc}")
        return n

    def _phase_open_stage_and_overrides(self) -> None:
        """Phase 1: open the USD stage and apply pre-``add_usd`` overrides.

        Edits land on the session layer so the cached ``robot.usda``
        payloads stay scene-agnostic.  Order matters: fix policies first
        (drop / weld sub-trees), then ``init_base_pose`` (translate the
        whole robot Xform), then ``RenderProduct`` disable.
        """
        self._logger.info("[newton-standalone] opening stage…")
        newton_scene = Path(self._render_layer_usda).parent / "newton_scene.usda" if self._render_layer_usda else None

        self._open_stage(newton_scene)

        # Runtime fix_base / fix_head / fix_body policies — drop or freeze
        # sub-trees from the composed stage so Newton's ``add_usd`` either
        # skips them entirely (fix_base → chassis: zero presence) or sees
        # them as Fixed welds (fix_head / fix_body: bodies stay in
        # body_label for TF, joints contribute 0 DOFs to Featherstone's M).
        # Cached robot.usda stays scene-agnostic — these are pure
        # session-layer overrides on the live composed stage.
        self._apply_runtime_fix_policies()

        # Apply ``robot.init_base_pose`` to the robot Xform on the composed
        # stage BEFORE ``add_usd`` parses it.  Authoring at session-layer
        # strength so the cached ``robot.usda`` stays untouched.  Works
        # uniformly for pin_base_to_world true/false: the welded path
        # follows the Xform; the FREE-joint path initialises base_link at
        # the Xform's world pose.
        self._apply_init_base_pose()

        # NOTE: we deliberately do NOT add render_layer.usda to the engine's
        # stage. That layer carries Camera prims (Head_Camera, FreeCam) and
        # RenderProduct prims used by genie_sim_render_node (a separate
        # process running OVRTX, the canonical camera renderer in our
        # pipeline). The engine doesn't need them — Newton physics doesn't
        # consume cameras, and adding them only:
        #   * clutters Isaac's viewport camera dropdown
        #   * forces Hydra to walk those prims each tick
        #   * caused the "double-rendering" perf concern
        # The render_node opens its own stage view in its own process; our
        # not-loading-here is invisible to it.
        if self._render_layer_usda and Path(self._render_layer_usda).exists():
            self._logger.info(
                f"[newton-standalone] skipping render_layer.usda ({self._render_layer_usda}) — "
                f"OVRTX render_node loads it independently"
            )
        # _disable_local_render_products is now a no-op since we never loaded
        # the prims, but keep the call as a defensive cleanup if some other
        # code path adds them.
        self._disable_local_render_products()
