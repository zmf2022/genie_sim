# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""MuJoCo-style per-class contact compliance + friction injection.

Single source of truth: write per-shape ``solref`` / ``solimp`` / friction
into the ``ModelBuilder``'s shape arrays + custom-attribute store BEFORE
``finalize()``.  After finalize, Newton propagates these values into:

  * ``model.shape_material_{ke,kd,mu,mu_torsional,mu_rolling}`` (runtime
    contact kernel + MJW MJCF compile path both read these)
  * the custom-attribute store ``mujoco:geom_solimp`` (frequency=SHAPE),
    which both ``SolverMuJoCo._compile`` (MJCF emit) AND ``mjw_model.
    geom_solimp`` (runtime contact kernel) sample.

That means a single pre-finalize write makes the MJCF dump and the live
simulation agree.  The previous post-finalize approach wrote only to
the runtime buffers and missed the custom-attribute store, producing
the misleading "MJCF shows defaults, runtime uses overrides" split.

Source of override values: the active scene's ``newton.mjc_contact:``
yaml block, merged onto the defaults in
``common/object_classification.py::MJC_CONTACT_DEFAULTS``.

Adapters that don't honour these (Featherstone, AVBD) skip the helper
without modifying any builder state.

Public surface
--------------
* :func:`apply_pre_finalize` — call right before ``builder.finalize()``.
* :func:`readback_post_finalize` — optional diagnostic; logs the
  per-class values that actually landed on the finalized model.  Useful
  for verifying overrides reached the GPU buffers without re-running
  the whole MJCF dump comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Adapter gate
# ---------------------------------------------------------------------------

_MJW_ADAPTER_NAMES = frozenset({"mujoco-warp", "mujoco_warp"})


def _is_mjw(adapter_name: str) -> bool:
    """Return True when the active rigid adapter routes through MuJoCo-Warp.

    Featherstone / AVBD use different contact compliance APIs and the
    custom-attribute store this module writes to doesn't apply there —
    skip cleanly rather than corrupt unrelated arrays.
    """
    return (adapter_name or "").strip().lower() in _MJW_ADAPTER_NAMES


# ---------------------------------------------------------------------------
# solref ↔ ke/kd inversion
# ---------------------------------------------------------------------------
#
# MJW's MJCF compile step + the runtime contact kernel both IGNORE any
# direct ``geom_solref`` write — they recompute solref each step from
# ``shape_material_ke / shape_material_kd`` via
# ``convert_solref(ke, kd, d_width=1, d_r=1)``.  Concretely (from
# ``solver_mujoco`` kernels):
#
#     timeconst = 2 / kd
#     dampratio = (kd / 2) * sqrt(1 / ke)
#
# Inverting:
#
#     kd = 2 / timeconst
#     ke = (kd / (2 * dampratio))^2 = (1 / (timeconst * dampratio))^2
#
# So we author solref by writing ke + kd onto ``shape_material_*``.  All
# call sites use the inverted form — never write ``geom_solref`` directly.


def _solref_to_ke_kd(solref: Tuple[float, float]) -> Tuple[float, float]:
    """Invert MJW's ``convert_solref`` so we can write ke/kd that produce
    the requested (timeconst, dampratio).

    Falls back to MJW's stock defaults (timeconst=0.02, dampratio=1.0 →
    ke=2500, kd=100) when the requested values are non-positive — same
    rule MJW's ``convert_solref`` else-branch uses internally.
    """
    timeconst, dampratio = float(solref[0]), float(solref[1])
    if timeconst <= 0.0 or dampratio <= 0.0:
        return 2500.0, 100.0
    kd = 2.0 / timeconst
    ke = (kd / (2.0 * dampratio)) ** 2
    return ke, kd


# ---------------------------------------------------------------------------
# Per-class param resolution
# ---------------------------------------------------------------------------


def _resolve_params(scene_cfg: dict, logger: Any) -> Dict[str, Dict[str, Tuple[float, ...]]]:
    """Merge ``newton.mjc_contact:`` yaml overrides onto defaults.

    Returns ``{kind: {solref, solimp, friction}}`` for every kind in
    ``ALL_OBJECT_KINDS``.  Validates lengths so a typo in the yaml fails
    loud at build time instead of silently keeping the default.
    """
    from common.object_classification import ALL_OBJECT_KINDS, MJC_CONTACT_DEFAULTS  # noqa: PLC0415

    scene_newton = (scene_cfg or {}).get("newton") or {}
    user = scene_newton.get("mjc_contact") or {}
    params: Dict[str, Dict[str, Tuple[float, ...]]] = {}
    for kind in ALL_OBJECT_KINDS:
        params[kind] = dict(MJC_CONTACT_DEFAULTS[kind])
        override = user.get(kind) or {}
        if not isinstance(override, dict):
            logger.warn(f"[mjc_contact] {kind}: expected mapping, got " f"{type(override).__name__}; ignoring")
            continue
        for key, expected_n in (("solref", 2), ("solimp", 5), ("friction", 3)):
            if key not in override:
                continue
            v = override[key]
            if not isinstance(v, (list, tuple)) or len(v) != expected_n:
                logger.warn(
                    f"[mjc_contact] {kind}.{key}: expected {expected_n}-tuple, "
                    f"got {v!r}; keeping default {params[kind][key]}"
                )
                continue
            params[kind][key] = tuple(float(x) for x in v)
    return params


# ---------------------------------------------------------------------------
# Stability check
# ---------------------------------------------------------------------------


def _stability_warn(
    params: Dict[str, Dict[str, Tuple[float, ...]]],
    class_counts: Dict[str, int],
    sub_dt: float,
) -> Optional[str]:
    """Return a warn string when ``sub_dt > min(timeconst)/2`` across the
    classes actually present in the scene, else ``None``.

    MJW's contact integrator is stable when the substep dt is below half
    the tightest solref timeconst it has to honour.  Above that the
    Baumgarte stabiliser silently softens contact every other substep,
    producing the "gripper walks through the box" / penetration symptoms
    no amount of yaml tuning can fix.
    """
    if sub_dt <= 0.0:
        return None
    present = [k for k, n in class_counts.items() if n > 0]
    if not present:
        return None
    tightest = min(params[k]["solref"][0] for k in present)
    threshold = tightest / 2.0
    if sub_dt > threshold and threshold > 0.0:
        needed_subs = max(1, int(round(1.0 / threshold)) + 1)
        return (
            f"sub_dt={sub_dt*1000:.2f} ms > min(timeconst)/2={threshold*1000:.2f} ms; "
            f"contact will silently soften.  Raise ``physics_solver_substep`` "
            f"to >= {needed_subs} or relax solref timeconsts."
        )
    return None


# ---------------------------------------------------------------------------
# Pre-finalize injection (THE entry point)
# ---------------------------------------------------------------------------


def apply_pre_finalize(
    builder: Any,
    *,
    scene_cfg: dict,
    robot_prefix: str,
    adapter_name: str,
    physics_hz: float,
    sim_substeps: int,
    logger: Any,
) -> None:
    """Inject per-class MJW contact compliance + friction into ``builder``.

    Call this **after** ``builder.add_usd(...)`` (so shape arrays are
    populated) and **before** ``builder.finalize()`` (so the custom-
    attribute writes reach both consumers).  No-op on non-MJW adapters.

    Writes per-shape:
      * ``shape_material_ke`` and ``shape_material_kd`` — drive ``solref``
        via MJW's compile-time inversion.
      * ``shape_material_mu`` / ``shape_material_mu_torsional`` /
        ``shape_material_mu_rolling`` — the three friction slots.
      * ``mujoco:geom_solimp`` custom attribute (frequency=SHAPE) — drives
        ``solimp`` for both MJCF emit and runtime kernel.

    Emits one summary log line per class with the chosen values, the
    source (override vs default), and an MJW stability check warning if
    ``sub_dt > min(timeconst)/2``.
    """
    if not _is_mjw(adapter_name):
        logger.info(f"[mjc_contact] adapter={adapter_name!r}: skipping (non-MJW path)")
        return

    import newton  # noqa: PLC0415
    from common.object_classification import ALL_OBJECT_KINDS, classify_shape  # noqa: PLC0415

    n_shapes = int(getattr(builder, "shape_count", 0))
    if n_shapes == 0:
        logger.info("[mjc_contact] no shapes registered yet; skipping")
        return

    params = _resolve_params(scene_cfg, logger)

    # Builder-side snapshots (lists, not warp arrays — pre-finalize).
    shape_types = list(builder.shape_type)
    shape_bodies = list(builder.shape_body)
    shape_labels = list(getattr(builder, "shape_label", []) or [])
    body_labels = list(getattr(builder, "body_label", []) or [])
    JT_PLANE = int(newton.GeoType.PLANE)
    SHAPE_FREQ = newton.Model.AttributeFrequency.SHAPE
    user_set = set(((scene_cfg or {}).get("newton") or {}).get("mjc_contact") or {})

    # Precompute ke/kd per class so we only invert once per kind.
    class_ke_kd: Dict[str, Tuple[float, float]] = {k: _solref_to_ke_kd(params[k]["solref"]) for k in ALL_OBJECT_KINDS}
    class_counts: Dict[str, int] = {k: 0 for k in ALL_OBJECT_KINDS}

    has_custom_solimp = hasattr(builder, "custom_attributes") and "mujoco:geom_solimp" in (
        builder.custom_attributes or {}
    )

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
        p = params[kind]
        ke, kd = class_ke_kd[kind]

        # 1. solref via material ke/kd inversion.
        builder.shape_material_ke[i] = float(ke)
        builder.shape_material_kd[i] = float(kd)

        # 2. friction triplet — three independent slots.
        mu_t, mu_tor, mu_rol = p["friction"]
        builder.shape_material_mu[i] = float(mu_t)
        if hasattr(builder, "shape_material_mu_torsional"):
            builder.shape_material_mu_torsional[i] = float(mu_tor)
        if hasattr(builder, "shape_material_mu_rolling"):
            builder.shape_material_mu_rolling[i] = float(mu_rol)

        # 3. solimp via the custom-attribute store (the asymmetric one
        #    that the post-finalize ``model.geom_solimp`` write doesn't
        #    reach — see this module's docstring).
        if has_custom_solimp:
            builder._process_custom_attributes(
                i,
                {"mujoco:geom_solimp": list(p["solimp"])},
                SHAPE_FREQ,
            )

        class_counts[kind] += 1

    # Summary log.
    sub_dt = 0.0
    if physics_hz > 0 and sim_substeps > 0:
        sub_dt = (1.0 / float(physics_hz)) / int(sim_substeps)
    warn = _stability_warn(params, class_counts, sub_dt)
    stab = "OK" if warn is None else f"WARN {warn}"

    applied_lines = []
    for kind in ALL_OBJECT_KINDS:
        if class_counts[kind] == 0:
            continue
        p = params[kind]
        ke, kd = class_ke_kd[kind]
        src = "override" if kind in user_set else "default"
        applied_lines.append(
            f"  {kind:<12} [{class_counts[kind]:>3} shape(s), {src}] "
            f"solref={p['solref']} -> ke={ke:.1f} kd={kd:.1f}, "
            f"solimp={p['solimp']}, friction={p['friction']}"
        )
    logger.info(
        f"[mjc_contact] pre-finalize injection complete -- "
        f"sub_dt={sub_dt*1000:.2f} ms, "
        f"min(timeconst)/2 stability check: {stab}\n" + "\n".join(applied_lines)
    )


# ---------------------------------------------------------------------------
# Optional post-finalize diagnostic
# ---------------------------------------------------------------------------


def readback_post_finalize(
    model: Any,
    *,
    scene_cfg: dict,
    robot_prefix: str,
    adapter_name: str,
    logger: Any,
) -> None:
    """Diagnostic-only: confirm the pre-finalize writes reached the GPU
    buffers + custom-attribute store after ``builder.finalize()``.

    Walks one representative shape per class and prints
    ``shape_material_ke / kd`` (drives solref) and the corresponding
    ``model.geom_solimp[i]`` (drives solimp).  Use this when you want to
    sanity-check the override path without diffing MJCF dumps.

    No state mutation -- safe to call repeatedly.
    """
    if not _is_mjw(adapter_name):
        return

    import newton  # noqa: PLC0415
    from common.object_classification import ALL_OBJECT_KINDS, classify_shape  # noqa: PLC0415

    try:
        n_shapes = int(getattr(model, "shape_count", 0))
        if n_shapes == 0:
            return
        ke = (
            model.shape_material_ke.numpy()
            if hasattr(model.shape_material_ke, "numpy")
            else np.asarray(model.shape_material_ke)
        )
        kd = (
            model.shape_material_kd.numpy()
            if hasattr(model.shape_material_kd, "numpy")
            else np.asarray(model.shape_material_kd)
        )
        mu = (
            model.shape_material_mu.numpy()
            if hasattr(model.shape_material_mu, "numpy")
            else np.asarray(model.shape_material_mu)
        )
        solimp = None
        if hasattr(model, "geom_solimp") and model.geom_solimp is not None:
            solimp = model.geom_solimp.numpy()
            if solimp.ndim == 3:
                solimp = solimp[0]
        shape_types = model.shape_type.numpy() if hasattr(model.shape_type, "numpy") else np.asarray(model.shape_type)
        shape_bodies = model.shape_body.numpy() if hasattr(model.shape_body, "numpy") else np.asarray(model.shape_body)
        shape_labels = list(getattr(model, "shape_label", []) or [])
        body_labels = list(getattr(model, "body_label", []) or [])
        JT_PLANE = int(newton.GeoType.PLANE)

        first_of: Dict[str, int] = {}
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
            if kind not in first_of:
                first_of[kind] = i

        lines = []
        for kind in ALL_OBJECT_KINDS:
            if kind not in first_of:
                continue
            i = first_of[kind]
            solimp_str = ", ".join(f"{v:.4g}" for v in solimp[i]) if solimp is not None else "n/a"
            lines.append(
                f"  {kind:<12} shape[{i}] ke={float(ke[i]):.1f} kd={float(kd[i]):.2f} "
                f"mu={float(mu[i]):.3f}  geom_solimp=[{solimp_str}]"
            )
        if lines:
            logger.info("[mjc_contact] post-finalize readback:\n" + "\n".join(lines))
    except Exception as exc:  # noqa: BLE001
        logger.warn(f"[mjc_contact] post-finalize readback failed: {exc!r}")
