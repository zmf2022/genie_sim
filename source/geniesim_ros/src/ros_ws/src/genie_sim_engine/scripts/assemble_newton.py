#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""assemble_newton.py — Newton-only scene extras (cloth, soft body, …).

Bridges the gap between Isaac Sim's Newton wrapper and Newton's native
particle-based features (cloth, soft body, particles).

The wrapper builds its ``newton.Model`` from the USD stage; rigid bodies and
articulations come through, but cloth/soft/particle authoring has no USD
schema in this Newton version. This script:

1. Validates the scene yaml's ``newton:`` block.
2. Writes a lightweight ``newton_solvers.json`` sidecar — prim paths, solver
   preference, contact tuning, and physics params only. No geometry. Geometry
   comes from the cloth USDs added as stage payloads, which the engine reads
   from the live stage.
3. The cloth USD files are referenced from the already-existing stage scene
   USDs by ``_open_scene_with_references`` in ``runtime.stage.py`` which
   reads ``newton_solvers.json`` at stage-open time.

The heavy geometry (vertices/indices) is NEVER serialized here — it lives in
the cloth USD files and is read once by the runtime from the live stage prim
via ``UsdGeom.Mesh.GetPointsAttr()`` / ``GetFaceVertexIndicesAttr()``. This
keeps ``newton_solvers.json`` small (< 2 KB regardless of mesh complexity) and
makes the cloth visual a real USD citizen that RViz / usdview can see.

Pipeline placement
------------------
Runs AFTER ``assemble_scene`` and BEFORE the engine starts. Gated on:

* ``--physics-engine == isaac_newton``  AND
* scene yaml has a non-empty ``newton.entries`` list.

Output
------
``<output-dir>/newton_solvers.json``::

    {
      "version": 2,
      "solver": {"prefer": "vbd" | "xpbd" | "style3d"},
      "contact": {"soft_ke": float, "soft_kd": float, "soft_mu": float},
      "entries": [
        {
          "kind":         "cloth",
          "name":         "<str>",
          "usd_prim_path": "/World/<name>",
          "mesh_usd_abs": "<absolute path to cloth USD file>",
          "pose":         [tx,ty,tz,qx,qy,qz,qw],
          "vel":          [vx,vy,vz],
          "params":       {<add_cloth_mesh kwargs>}
        }
      ]
    }

Note: ``mesh_usd_abs`` is stored so ``_open_scene_with_references`` can add
the payload. Params are stored here (not in the cloth USD) because they are
physics-only tuning values that should not pollute the mesh asset itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_KIND_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "cloth": (
        "density",
        "tri_ke",
        "tri_ka",
        "tri_kd",
        "edge_ke",
        "edge_kd",
        "particle_radius",
    ),
    # ``box`` is a static rigid collider authored as ``UsdGeom.Cube`` with
    # ``UsdPhysics.CollisionAPI``. Newton's ``add_usd`` picks it up as a
    # world-static shape; cloth and articulations collide against it.
    # Required: ``half_extents=[hx,hy,hz]`` (metres). No ``mesh_usd``.
    "box": ("half_extents",),
}
# Particle solvers supported by the newton-standalone engine. Picking one of
# these in ``newton.solver.prefer`` is REQUIRED when the scene declares
# any cloth entry — the engine refuses to start with ``auto`` or a
# missing/empty value. ``auto`` is rejected because the resolution
# differs across code paths (assemble_newton would write ``xpbd`` to the
# sidecar but the runtime engine hard-codes VBD), so what's in the yaml
# would have nothing to do with what actually simulates. Explicit only.
# Note: ``avbd`` is the marketing name for "use a single SolverVBD
# instance for both cloth particles AND rigid bodies (via Augmented VBD)
# instead of the Featherstone-rigid + SolverVBD-cloth split".  Newton's
# SolverVBD class handles both internally; the choice is which adapter
# (Featherstone vs unified-AVBD) the engine instantiates.
_PARTICLE_SOLVERS = {"vbd", "xpbd", "style3d", "avbd"}
_KIND_REQUIRES_PARTICLE_SOLVER = {"cloth"}
# Geometry kinds that come from an external mesh USD file (referenced into
# the staged ``newton_scene.usda``). Primitive kinds (``box`` and other
# inline shapes) are authored inline and don't need a mesh_usd.
_KIND_REQUIRES_MESH_USD = {"cloth"}


def _die(msg: str) -> None:
    print(f"[assemble_newton] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_scene_yaml(scene_path: str) -> tuple[Path, dict]:
    p = Path(scene_path).resolve()
    if p.suffix not in (".yaml", ".yml"):
        _die(f"--scene must be a YAML file, got {p}")
    with p.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        _die(f"scene YAML did not parse to a mapping: {p}")
    return p, cfg


def _validate_newton_block(newton_cfg: dict, scene_path: Path) -> None:
    if not isinstance(newton_cfg.get("entries"), list):
        _die(f"newton.entries must be a list in {scene_path}")
    seen: set[str] = set()
    for i, entry in enumerate(newton_cfg["entries"]):
        ctx = f"newton.entries[{i}]"
        if not isinstance(entry, dict):
            _die(f"{ctx}: must be a mapping")
        kind = entry.get("kind")
        if kind not in _KIND_REQUIRED_PARAMS:
            _die(f"{ctx}: unknown kind={kind!r}. Supported: {sorted(_KIND_REQUIRED_PARAMS)}")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            _die(f"{ctx}: 'name' required and must be a non-empty string")
        if name in seen:
            _die(f"{ctx}: duplicate name={name!r}")
        seen.add(name)
        if kind in _KIND_REQUIRES_MESH_USD and "mesh_usd" not in entry:
            _die(f"{ctx} ({name!r}): 'mesh_usd' is required for kind={kind!r}")
        pose = entry.get("pose")
        if pose is not None and (not isinstance(pose, list) or len(pose) != 7):
            _die(f"{ctx} ({name!r}): 'pose' must be 7 floats [tx,ty,tz,qx,qy,qz,qw]")
        vel = entry.get("vel")
        if vel is not None and (not isinstance(vel, list) or len(vel) != 3):
            _die(f"{ctx} ({name!r}): 'vel' must be 3 floats [vx,vy,vz]")
        params = entry.get("params") or {}
        missing = [k for k in _KIND_REQUIRED_PARAMS[kind] if k not in params]
        if missing:
            _die(f"{ctx} ({name!r}): params missing required keys: {sorted(missing)}")
        if kind == "box":
            he = params.get("half_extents")
            if (
                not isinstance(he, list)
                or len(he) != 3
                or not all(isinstance(x, (int, float)) and float(x) > 0 for x in he)
            ):
                _die(f"{ctx} ({name!r}): params.half_extents must be 3 positive floats")


def _resolve_solver(newton_cfg: dict, entries: list[dict]) -> str:
    """Return the explicit particle-solver name from the scene yaml.

    No defaults. No ``auto``. If any cloth entry is present,
    ``newton.solver.prefer`` MUST be set to one of ``_PARTICLE_SOLVERS``.
    If no cloth is present, the field is optional and ``"none"`` is
    returned so the sidecar carries a definite value.
    """
    solver_block = newton_cfg.get("solver")
    needs_particles = any(e["kind"] in _KIND_REQUIRES_PARTICLE_SOLVER for e in entries)

    raw = ""
    if isinstance(solver_block, dict):
        raw = str(solver_block.get("prefer") or "").strip().lower()

    if not needs_particles:
        # No cloth/particles → field is optional; reject obviously
        # malformed values but otherwise accept silence.
        if raw and raw not in _PARTICLE_SOLVERS:
            _die(
                f"newton.solver.prefer={raw!r} is not a valid particle solver. "
                f"Valid: {sorted(_PARTICLE_SOLVERS)}. (Also: the field is "
                f"optional when no cloth entry is declared — leave it out.)"
            )
        return raw or "none"

    # Cloth present → explicit choice required.
    if not raw:
        _die(
            "newton.solver.prefer is REQUIRED when newton.entries contains a "
            f"cloth/particle entry. Set it to one of: {sorted(_PARTICLE_SOLVERS)}. "
            "'auto' is not supported — every path must declare the solver "
            "explicitly so the yaml matches the runtime."
        )
    if raw == "auto":
        _die("newton.solver.prefer='auto' is not supported. " f"Pick one explicitly: {sorted(_PARTICLE_SOLVERS)}.")
    if raw not in _PARTICLE_SOLVERS:
        _die(f"newton.solver.prefer={raw!r} is not supported. " f"Valid options: {sorted(_PARTICLE_SOLVERS)}.")
    return raw


def _resolve_mesh_path(mesh_ref: str, base_path: str, scene_path: Path) -> Path:
    p = Path(mesh_ref)
    if p.is_absolute() and p.is_file():
        return p
    if base_path:
        cand = Path(base_path) / p
        if cand.is_file():
            return cand.resolve()
    cand = scene_path.parent / p
    if cand.is_file():
        return cand.resolve()
    _die(
        f"mesh_usd={mesh_ref!r} not found. Tried:\n"
        f"  {Path(base_path) / p if base_path else '(no --base-path)'}\n"
        f"  {scene_path.parent / p}"
    )


def _build_sidecar(newton_cfg: dict, scene_path: Path, base_path: str) -> dict[str, Any]:
    entries_in = newton_cfg["entries"]
    solver_str = _resolve_solver(newton_cfg, entries_in)

    contact_in = newton_cfg.get("contact") or {}
    contact_out = {
        "soft_ke": float(contact_in.get("soft_ke", 1.0e4)),
        "soft_kd": float(contact_in.get("soft_kd", 1.0e-2)),
        "soft_mu": float(contact_in.get("soft_mu", 0.25)),
    }

    entries_out: list[dict] = []
    for entry in entries_in:
        kind = entry["kind"]
        name = entry["name"]
        pose = [float(x) for x in (entry.get("pose") or [0, 0, 0, 0, 0, 0, 1])]
        vel = [float(x) for x in (entry.get("vel") or [0, 0, 0])]
        # params may contain nested lists (e.g. box.half_extents) — only
        # scalar values get coerced to float here. Downstream consumers
        # already accept the mixed shape.
        params_in = entry.get("params") or {}
        params: dict = {}
        for k, v in params_in.items():
            if isinstance(v, (int, float)):
                params[k] = float(v)
            elif isinstance(v, list):
                params[k] = [float(x) for x in v]
            else:
                params[k] = v

        out_entry: dict = {
            "kind": kind,
            "name": name,
            "usd_prim_path": f"/World/{name}",
            "pose": pose,
            "vel": vel,
            "params": params,
        }
        if kind in _KIND_REQUIRES_MESH_USD:
            mesh_p = _resolve_mesh_path(entry["mesh_usd"], base_path, scene_path)
            print(f"[assemble_newton]   {name}: {kind} @ {mesh_p.name}", flush=True)
            out_entry["mesh_usd_abs"] = str(mesh_p)
        else:
            print(
                f"[assemble_newton]   {name}: {kind} (primitive, " f"params={ {k: v for k, v in params.items()} })",
                flush=True,
            )
        entries_out.append(out_entry)

    return {
        "version": 2,
        "solver": {"prefer": solver_str},
        "contact": contact_out,
        "entries": entries_out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-path", default="")
    # ``--physics-engine`` is accepted for CLI compatibility but does
    # not gate writing the sidecar. The launch
    # pipeline always chains this script regardless of the physics
    # engine choice (see ``make_assemble_pipeline``); the gate is
    # purely on whether the scene yaml declares any newton entries.
    # Writing the sidecar for a non-newton engine is harmless — the
    # PhysX / isaac_newton runtimes don't read it.
    parser.add_argument("--physics-engine", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    scene_path, scene_cfg = _load_scene_yaml(args.scene)
    newton_cfg = scene_cfg.get("newton")
    if not isinstance(newton_cfg, dict):
        print(f"[assemble_newton] no newton block in {scene_path} → skipping.", flush=True)
        return 0

    # Validate scene_plugin (optional Python file the engine loads at
    # build time — see engine/newton/plugin.py).  We only check the type
    # here; the engine resolves the path against the scene yaml's dir
    # and raises if it's missing.  The sidecar carries no plugin data:
    # the engine reads ``newton.scene_plugin`` straight from the scene
    # yaml at runtime.
    plugin_ref = newton_cfg.get("scene_plugin")
    if plugin_ref is not None and not isinstance(plugin_ref, str):
        _die(f"newton.scene_plugin must be a string path (got {type(plugin_ref).__name__}) " f"in {scene_path}")

    if not newton_cfg.get("entries"):
        # No cloth/USD entries — sidecar isn't needed.  The engine still
        # reads ``newton.solver.prefer`` and ``newton.scene_plugin``
        # directly from the scene yaml.
        if plugin_ref:
            print(
                f"[assemble_newton] no newton.entries in {scene_path}; "
                f"scene_plugin={plugin_ref!r} will run at engine build time → skipping.",
                flush=True,
            )
        else:
            print(f"[assemble_newton] no newton.entries in {scene_path} → skipping.", flush=True)
        return 0

    print(f"[assemble_newton] scene:     {scene_path}", flush=True)
    print(f"[assemble_newton] output:    {args.output_dir}", flush=True)

    _validate_newton_block(newton_cfg, scene_path)
    sidecar = _build_sidecar(newton_cfg, scene_path, args.base_path)

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sidecar_path = out / "newton_solvers.json"
    with sidecar_path.open("w") as f:
        json.dump(sidecar, f, indent=2)

    n = len(sidecar["entries"])
    size = sidecar_path.stat().st_size
    print(
        f"[assemble_newton] OK: {n} entr(ies), solver={sidecar['solver']['prefer']}, " f"sidecar={size} bytes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
