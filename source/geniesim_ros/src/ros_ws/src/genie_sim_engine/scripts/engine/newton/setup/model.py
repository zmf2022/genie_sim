# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build-phases 2-3: scene resolution + ModelBuilder + ``add_usd`` + finalize.

Owns scene-plugin / cloth / solver-preference resolution and the
ModelBuilder lifecycle through ``finalize``.  ``self._model`` is
populated by the end of this mixin's phase calls.
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


def _strip_disabled_physics_schemas(stage) -> None:
    """De-apply ``PhysicsRigidBodyAPI`` (and ``PhysicsMassAPI``) on prims
    whose composed ``physics:rigidBodyEnabled`` is ``False``, and clear the
    ``physics:approximation`` opinion on disabled colliders so Newton's GL
    viewer renders their actual mesh instead of the convex-decomposition
    proxy.

    Why this is necessary:

    1.  Newton's ``ModelBuilder.add_usd`` visual-loader pass
        (``newton/_src/utils/import_usd.py:_load_visual_shapes_impl``
        line 788) early-returns on any prim that has
        ``UsdPhysics.RigidBodyAPI`` applied, *regardless* of whether the
        rigid body is enabled.  Many of our scene props (e.g. the
        ``GDK/demo.usda`` shelves) compose ``RigidBodyAPI`` from the leaf
        USD and turn it off in the scene's per-instance ``over``;
        without the strip the visual is missing entirely.

    2.  When a mesh has ``PhysicsCollisionAPI`` with
        ``collisionEnabled=False`` *and* an ``physics:approximation``
        opinion (``convexDecomposition`` / ``convexHull`` / etc.), Newton
        still adds the shape to the builder (``COLLIDE_SHAPES`` flag
        cleared) and the GL viewer renders the *approximation* — which
        looks like the convex hull rather than the underlying mesh.
        Clearing the approximation token forces the viewer to render the
        mesh as authored.

    We deliberately do NOT strip ``PhysicsCollisionAPI`` itself: static
    props (like ``market_00``) that are NOT under a ``RigidBodyAPI``
    parent rely on the collision-shape pipeline to register their mesh
    as a (non-colliding) builder shape so the GL viewer renders it at
    all — Newton's visual-loader pass is only invoked from rigid-body
    parsing, so removing the schema would make those static props
    disappear from the viewport.

    The fix only touches the live stage in this process; on-disk USDs
    are unaffected, and other consumers (IsaacSim, OVRtx) read the
    canonical layers as authored.
    """
    try:
        from pxr import Sdf, UsdPhysics  # noqa: PLC0415
    except ImportError:
        return

    _RB_SCHEMAS = ("PhysicsRigidBodyAPI", "PhysicsMassAPI")
    stripped_rb = cleared_approx = 0
    for prim in stage.Traverse():
        applied = list(prim.GetAppliedSchemas())
        remove: list[str] = []

        rba = UsdPhysics.RigidBodyAPI(prim)
        if rba and rba.GetRigidBodyEnabledAttr().Get() is False:
            remove.extend(s for s in _RB_SCHEMAS if s in applied)

        col = UsdPhysics.CollisionAPI(prim)
        if col and col.GetCollisionEnabledAttr().Get() is False:
            approx_attr = prim.GetAttribute("physics:approximation")
            if approx_attr and approx_attr.HasAuthoredValue():
                approx_attr.Block()
                cleared_approx += 1

        if remove:
            cleaned = [s for s in applied if s not in remove]
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(cleaned))
            stripped_rb += sum(1 for s in remove if s in _RB_SCHEMAS)

    if stripped_rb or cleared_approx:
        print(
            f"[newton-standalone] reauthored disabled-physics opinions: "
            f"{stripped_rb} RigidBody/Mass schemas removed, "
            f"{cleared_approx} approximation tokens cleared "
            f"(restores visuals on rigidBodyEnabled=0 / collisionEnabled=0 props)",
            flush=True,
        )


class _ModelMixin:
    def _phase_resolve_scene(self) -> Tuple[List[Any], str, bool, bool]:
        """Phase 2: resolve cloth entries, scene plugin, and solver preference.

        Returns ``(cloth_entries, solver_pref, plugin_has_build,
        needs_particle_solver)`` — locals consumed by later build phases.
        Also stashes ``self._cloth_solver_pref`` for the substep selector.
        """
        self._logger.info("[newton-standalone] building Newton Model…")

        cloth_entries = self._load_cloth_entries()
        # Resolve scene-plugin early so its presence is part of the
        # particle-vs-rigid-only decision below.  A plugin-driven particle
        # scene (e.g. chow-mein FEM noodles) declares
        # ``newton.solver.prefer`` directly in the yaml even with no
        # ``newton.entries``; we treat that the same as a cloth entry for
        # adapter gating + solver-attribute registration.
        self._load_scene_plugin()
        plugin = getattr(self, "_scene_plugin", None)
        plugin_has_build = plugin is not None and callable(getattr(plugin, "on_build", None))

        # Cloth gate — an adapter that doesn't support cloth raises early
        # instead of silently running rigid-only on a cloth scene.
        if cloth_entries and not self._adapter.supports_cloth:
            raise RuntimeError(
                f"physics_solver={self._adapter.name} does not support cloth: found "
                f"{len(cloth_entries)} cloth entry/entries in the scene yaml. "
                f"Use physics_solver:=fsvbd for cloth tasks (launcher_newton_fsvbd.yaml)."
            )
        # Load the solver preference up-front so we can register the
        # matching custom attributes on the builder. The attributes
        # solvers register on the builder MUST match the solver we then
        # instantiate after finalize — VBD reads ``edge_rest_angle`` etc.
        # that XPBD doesn't author, and vice versa.  Plugin-driven scenes
        # also need this even with no cloth entries.
        needs_particle_solver = bool(cloth_entries) or plugin_has_build
        solver_pref = self._load_solver_preference() if needs_particle_solver else "none"
        # Stash for the substep — _substep_body_franka_vbd_cloth needs to
        # know whether to wrap cloth_solver.step() in a body-state freeze
        # (XPBD re-integrates rigid bodies; VBD/Style3D's
        # ``integrate_with_external_rigid_solver=True`` mode does not).
        self._cloth_solver_pref = solver_pref

        return cloth_entries, solver_pref, plugin_has_build, needs_particle_solver

    def _phase_build_model(
        self,
        cloth_entries: List[Any],
        solver_pref: str,
        plugin_has_build: bool,
        needs_particle_solver: bool,
    ) -> None:
        """Phase 3: ModelBuilder + ``add_usd`` + extras + ``finalize``.

        Builds the ModelBuilder, registers the matching custom-attribute
        set for the chosen solver, parses USD, adds cloth / ground /
        plugin bodies, applies pre-finalize mjc_contact tuning, then
        finalises onto cuda:0.  ``self._model`` is populated by the end
        of this phase; ``on_model_ready`` plugin hook also runs here.
        """
        import newton  # noqa: PLC0415
        from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx  # noqa: PLC0415

        # --- Build ModelBuilder (gravity from stage, default -9.81 m/s²) ---
        builder = newton.ModelBuilder()
        self._adapter.register_custom_attributes(builder)
        if needs_particle_solver:
            # AVBD's adapter already registered SolverVBD's custom
            # attributes above (the unified solver handles cloth too),
            # so the cloth-pref branch is a no-op on that path.
            if getattr(self._adapter, "name", "") == "avbd":
                pass
            elif solver_pref == "vbd":
                newton.solvers.SolverVBD.register_custom_attributes(builder)
            elif solver_pref == "xpbd":
                newton.solvers.SolverXPBD.register_custom_attributes(builder)
            elif solver_pref == "style3d":
                newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            else:
                # Defensive — _load_solver_preference already raises on
                # unknown values, but if it returns "none" with cloth /
                # plugin particles present that's a wiring bug we want
                # to surface here.
                raise RuntimeError(
                    f"particle entries present (cloth={bool(cloth_entries)}, "
                    f"plugin={plugin_has_build}) but solver preference is "
                    f"{solver_pref!r}. Set newton.solver.prefer in the "
                    f"scene yaml."
                )

        # Robot from USD.  Note: we do NOT override the importer's
        # ``floating`` flag here — IsaacSim's URDF→USD converter
        # always authors a ``PhysicsFixedJoint "root_joint"`` between
        # ``/robot`` and ``base_link``, which Newton parses as a
        # body-to-world FixedJoint and uses as the root of the
        # articulation.  Passing ``floating=False`` on top would tell
        # ``_add_base_joint`` to ALSO add an auto-fixed-root-joint,
        # producing two joints with the same ``child=base_link`` and
        # tripping mjwarp's directed ``topological_sort`` with
        # ``"Multiple joints lead to body 0"``.
        _strip_disabled_physics_schemas(self._stage)
        builder.add_usd(
            source=self._stage,
            verbose=False,
            collapse_fixed_joints=False,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],
        )

        # Ground plane — cloth needs it
        builder.add_ground_plane()

        # Cloth meshes
        for entry in cloth_entries:
            self._inject_cloth(builder, entry)

        # Scene-plugin: optional Python file referenced from the scene
        # yaml as ``newton.scene_plugin``.  Lets a scene author add bodies
        # / soft-grids / shapes / kinematic colliders that have no USD
        # schema (e.g. FEM tet bars, runtime-meshed wok colliders) without
        # extending the cloth-entry vocabulary.  Plugin was loaded above so
        # solver-attribute registration could see it; we run on_build now,
        # after USD parse + ground plane + cloth injection.
        self._call_plugin("on_build", builder, self._plugin_ctx())

        # VBD requires coloring; XPBD/Style3D do not.  Coloring is also
        # safe-but-wasteful on a non-VBD particle scene, so gate it.
        if cloth_entries or (plugin_has_build and solver_pref == "vbd"):
            builder.color()

        # MuJoCo-style per-class contact compliance + friction injection.
        # Runs PRE-FINALIZE so it writes into the builder's shape arrays
        # AND the ``mujoco:geom_solimp`` custom-attribute store — both of
        # which Newton then propagates into the runtime model AND MJW's
        # MJCF compile path.  Single source of truth: the MJCF dump and
        # the live contact kernel always agree.
        #
        # The previous post-finalize approach wrote only to runtime
        # buffers and missed the custom-attribute store, producing the
        # misleading "MJCF shows defaults, runtime uses overrides" split
        # that drove the gripper-vs-box tunneling debug.  Module's
        # docstring at ``engine/newton/mjc_contact.py`` records the
        # asymmetry and why this is now the right cut.
        from engine.newton.mjc_contact import apply_pre_finalize as _mjc_apply  # noqa: PLC0415

        _mjc_apply(
            builder,
            scene_cfg=self._scene_cfg,
            robot_prefix=str(getattr(self, "_robot_prefix_str", "") or ""),
            adapter_name=getattr(self._adapter, "name", ""),
            physics_hz=float(getattr(self, "_physics_hz", 0) or 0),
            sim_substeps=int(getattr(self, "_sim_substeps", 0) or 0),
            logger=self._logger,
        )

        self._model = builder.finalize(device="cuda:0")

        # Engine soft-contact defaults (franka cloth values) — set BEFORE the
        # plugin hook so a scene plugin's on_model_ready can OVERRIDE them.
        # These MUST precede _call_plugin("on_model_ready"): FEM scenes
        # (chow-mein noodles) set their own leak-proof ke/kd/mu/margin in
        # on_model_ready, and if the engine wrote these AFTER the plugin
        # (as _phase_normalize_model used to) it would clobber the plugin's
        # tuning and the particles would tunnel straight through the wok.
        self._model.soft_contact_ke = self._SOFT_CONTACT_KE
        self._model.soft_contact_kd = self._SOFT_CONTACT_KD
        self._model.soft_contact_mu = self._SOFT_CONTACT_MU
        # Cloth ↔ rigid narrow-phase margin.  Newton's default is sized for
        # cm-scale assets; at meter scale it generates many spurious close
        # pairs and pumps cloth-rigid narrow-phase cost.  The franka cloth
        # demo creates a CollisionPipeline with an explicit
        # ``soft_contact_margin = particle_radius`` (0.8 cm); we set the
        # same on ``model.soft_contact_margin`` so ``model.collide()`` uses
        # it without us having to switch to the explicit pipeline.
        if hasattr(self._model, "soft_contact_margin"):
            self._model.soft_contact_margin = self._CLOTH_BODY_CONTACT_M

        # Plugin's chance to write into model.particle_* / soft_contact_*
        # before the solver is constructed.  Common use: the chow-mein
        # plugin sets particle_mu / soft_contact_radius / soft_contact_margin
        # (and ke/kd/mu) for FEM noodles, OVERRIDING the engine defaults set
        # just above.  Solver is built after this in _build's tail.
        self._call_plugin("on_model_ready", self._model, self._plugin_ctx())
