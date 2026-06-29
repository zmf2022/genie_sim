#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lifecycle helpers for ``genie_sim_engine``.

``bootstrap_simulation_app`` MUST be called before any ``omni.*`` /
``isaacsim.*`` import; it constructs the ``SimulationApp`` (headless or
GUI), strips ``--gui`` from ``sys.argv``, and returns ``(app, headless)``.

``configure_carb_settings`` may import omni / isaacsim freely because it is
only invoked from the running node, well after bootstrap.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

_VALID_ENGINES = ("isaac_physx", "isaac_newton", "newton")

# ---------------------------------------------------------------------------
# Render-mode map: ``--render-mode`` choices → (hydra_engine, rtx_submode).
#
# A single CLI knob picks the RTX submode. We don't expose the Pixar Storm
# delegate because it isn't bundled in ``isaacsim.exp.base.python.kit``
# (the minimal kit our newton-standalone path uses). When we tried Storm,
# ``vp.set_hd_engine("Storm")`` spammed
# ``UsdContext::createViewport — unable to find suitable engine`` warnings
# at ~15 Hz and silently fell back to whatever RTX submode was active. If
# you really need Storm, you'd have to switch to the full Isaac Sim kit
# experience and pay the heavier startup — not worth it just for a debug
# viewport.
#
# Both layers (kit args at app create time + ``configure_viewport_for_debug``
# at runtime) target the same RTX submode so a late kit-defaults push
# doesn't override the launch arg.
#
# Rough cost at the fr3+cloth scene on a 5090:
#   raster     ~16 ms    RaytracedLighting: raster + RT shadows + RT denoise.
#   pathtrace  ~49 ms    RealTimePathTracing (kit default).
#   offline    100-300ms PathTracing (no denoiser, multi-sample).
# ---------------------------------------------------------------------------
RENDER_MODE_MAP: dict[str, tuple[str, str]] = {
    "raster": ("rtx", "RaytracedLighting"),
    "pathtrace": ("rtx", "RealTimePathTracing"),
    "offline": ("rtx", "PathTracing"),
}
VALID_RENDER_MODES = tuple(RENDER_MODE_MAP.keys())
DEFAULT_RENDER_MODE = "raster"


def _resolve_render_mode(render_mode: str) -> tuple[str, str, str]:
    """Return ``(canonical_name, hydra_engine, rtx_submode)`` for a mode.

    Falls back to ``DEFAULT_RENDER_MODE`` on unrecognised input rather than
    raising, so a malformed launch arg doesn't crash bringup.
    """
    name = (render_mode or "").strip().lower() or DEFAULT_RENDER_MODE
    if name not in RENDER_MODE_MAP:
        print(
            f"[genie_sim_engine] render_mode={render_mode!r} not recognised; "
            f"valid: {VALID_RENDER_MODES}. Falling back to {DEFAULT_RENDER_MODE!r}.",
            flush=True,
        )
        name = DEFAULT_RENDER_MODE
    engine, submode = RENDER_MODE_MAP[name]
    return name, engine, submode


def _validate_engine_id(physics_engine: str) -> str:
    """Lower-case + validate against the canonical engine set."""
    raw = physics_engine.strip().lower()
    if raw not in _VALID_ENGINES:
        raise ValueError(f"physics_engine={physics_engine!r} is not a valid engine id. " f"Valid: {_VALID_ENGINES}.")
    return raw


def _configure_isaac_physx(headless: bool, physics_solver: str, render_mode: str, cfg: dict) -> str:
    print("[genie_sim_engine] physics engine: isaac_physx", flush=True)
    return ""


def _configure_isaac_newton(headless: bool, physics_solver: str, render_mode: str, cfg: dict) -> str:
    solver = physics_solver.strip().lower()
    # Locate the isaacsim newton .kit experience. Search order:
    #   1. ``GENIESIM_ISAACSIM_KIT`` env override (operator-set path).
    #   2. Standard pip/distro install at ``/usr/local/lib/python3.X/dist-packages/``.
    #   3. Whatever ``isaacsim`` resolves to via ``importlib`` (catches conda /
    #      virtualenv installs without hard-coding any home directory).
    experience = os.environ.get("GENIESIM_ISAACSIM_KIT", "")
    if not experience or not os.path.isfile(experience):
        experience = "/usr/local/lib/python3.12/dist-packages/isaacsim/apps/isaacsim.exp.full.newton.kit"
    if not os.path.isfile(experience):
        try:
            import importlib.util as _iu  # local import keeps cold-path overhead off

            spec = _iu.find_spec("isaacsim")
            if spec is not None and spec.submodule_search_locations:
                experience = os.path.join(
                    list(spec.submodule_search_locations)[0], "apps", "isaacsim.exp.full.newton.kit"
                )
        except Exception:
            pass
    cfg["extra_args"] = [f"--/exts/isaacsim.physics.newton/solver_type={solver}"]
    print(f"[genie_sim_engine] physics engine: isaac_newton (solver={solver})", flush=True)
    return experience


def _configure_newton(headless: bool, physics_solver: str, render_mode: str, cfg: dict) -> str | None:
    # Newton-standalone is Kit-free.  This function unconditionally
    # returns the Kit-free sentinel so ``bootstrap_simulation_app``
    # early-returns ``(None, True)`` for callers that still reach it
    # (the supported entry point is ``genie_sim_engine_newton.py``,
    # which never calls it at all).
    _warmup_newton_kernels("vbd")
    print("[genie_sim_engine] physics engine: newton (Kit-free; SimulationApp suppressed)", flush=True)
    return None


_ENGINE_CONFIGURE = {
    "isaac_physx": _configure_isaac_physx,
    "isaac_newton": _configure_isaac_newton,
    "newton": _configure_newton,
}

_ENGINE_POST_WARMUP = {
    "isaac_newton": lambda solver: _warmup_newton_kernels(solver),
    "newton": lambda _: _warmup_newton_kernels("vbd"),
}


def bootstrap_simulation_app(
    physics_engine: str = "isaac_physx",
    physics_solver: str = "mujoco-warp",
    render_mode: str = DEFAULT_RENDER_MODE,
) -> Tuple[object, bool]:
    """Create the ``SimulationApp`` and return ``(app, headless)``.

    Honors ``ISAACSIM_HEADLESS`` (set to a falsy value to force GUI) and
    consumes a ``--gui`` flag from ``sys.argv``.

    Args:
        physics_engine: ``"isaac_physx"`` (default) or ``"isaac_newton"``.
            Both go through Isaac Sim's wrappers; the choice picks
            ``omni.physx`` (PhysX 5) vs ``isaacsim.physics.newton``
            (Newton solver, rigid-body bridge only — no cloth / soft).
        physics_solver: Solver for ``isaac_newton`` — ``"mujoco-warp"``
            (default, MuJoCo-Warp), ``"xpbd"``, ``"featherstone"``, or
            ``"semiImplicit"``. Ignored for ``isaac_physx``.
    """
    import isaacsim  # noqa: F401
    from isaacsim import SimulationApp

    physics_engine = _validate_engine_id(physics_engine)

    headless = True
    if os.environ.get("ISAACSIM_HEADLESS", "").strip().lower() in ("0", "false", "no", "off"):
        headless = False
    if "--gui" in sys.argv:
        headless = False
        sys.argv.remove("--gui")

    launch_config: dict = {"headless": headless}

    experience = _ENGINE_CONFIGURE[physics_engine](headless, physics_solver, render_mode, launch_config)
    if experience is None:
        return None, True  # newton + headless: Kit-free early return

    app = SimulationApp(launch_config, experience=experience)

    if physics_engine in _ENGINE_POST_WARMUP:
        _ENGINE_POST_WARMUP[physics_engine](physics_solver.strip().lower())

    return app, headless


def configure_viewport_for_debug(
    render_mode: str = DEFAULT_RENDER_MODE,
    updates_enabled: bool | None = None,
) -> None:
    """Configure the editor viewport for a debug-only use case.

    Call this AFTER the scene has loaded (i.e. after the engine has opened
    the stage and rendered a few frames). Otherwise the viewport never gets
    to render the scene and the GUI stays black.

    Args:
        render_mode: One of ``RENDER_MODE_MAP`` keys (``storm`` / ``raster``
            / ``pathtrace`` / ``offline``). Resolved to a ``(hydra_engine,
            rtx_submode)`` pair, then applied via BOTH ``carb.settings`` and
            ``vp.set_hd_engine`` / ``vp.render_mode``. Pinning at multiple
            layers is required because kit's RTX defaults will override a
            launch-arg-only ``/rtx/rendermode`` flip during composer setup.
        updates_enabled: If False, the viewport stops re-rendering and stays
            on the last frame (cheap: ~2ms / simulation_app.update()). If True,
            the viewport keeps re-rendering live. If None, read the
            ``ISAACSIM_VIEWPORT_UPDATES`` env var (default: **True**, i.e.
            live debug view).
    """
    rm_name, hydra_engine, rtx_submode = _resolve_render_mode(render_mode)

    try:
        import carb.settings
        import omni.kit.viewport.utility as vpu
    except Exception as exc:
        print(f"[genie_sim_engine] viewport configure: import failed ({exc})", flush=True)
        return

    vp = vpu.get_active_viewport()
    if vp is None:
        print("[genie_sim_engine] no active viewport; skipping config", flush=True)
        return

    # Diagnostic: what does /rtx/rendermode hold right now?
    try:
        s = carb.settings.get_settings()
        print(
            f"[genie_sim_engine] viewport configure: render_mode={rm_name!r} → "
            f"engine={hydra_engine}, submode={rtx_submode}  "
            f"| live /rtx/rendermode = {s.get('/rtx/rendermode')!r}",
            flush=True,
        )
    except Exception:
        pass

    # 0. Pin carb settings (live + defaults + transient) BEFORE flipping the
    # viewport. The kit's RTX defaults override the launch-arg
    # /rtx/rendermode when the renderer is first attached; flipping the
    # transient setting too guarantees the composer picks up the choice on
    # the next update.
    try:
        s.set("/rtx/rendermode", rtx_submode)
        s.set("/rtx-defaults/rendermode", rtx_submode)
        s.set("/rtx-transient/rendermode", rtx_submode)
    except Exception as exc:
        print(
            f"[genie_sim_engine] viewport: carb /rtx*/rendermode set failed: {exc}",
            flush=True,
        )

    # 1. Pin the Hydra engine on the live viewport. Always ``"rtx"`` now —
    # see RENDER_MODE_MAP for why Storm isn't supported.
    try:
        before_engine = getattr(vp, "hydra_engine", None)
        vp.set_hd_engine(hydra_engine)
        print(
            f"[genie_sim_engine] viewport hydra_engine: {before_engine} → {hydra_engine}",
            flush=True,
        )
    except Exception as exc:
        print(f"[genie_sim_engine] viewport.set_hd_engine failed: {exc}", flush=True)

    # 2. Set the submode on the live viewport. ``vp.render_mode = X`` is
    # what triggers the viewport to actually flip (carb settings alone
    # don't force a re-attach).
    try:
        before = vp.render_mode
        vp.render_mode = rtx_submode
        print(
            f"[genie_sim_engine] viewport render mode: {before} → {vp.render_mode}",
            flush=True,
        )
    except Exception as exc:
        print(f"[genie_sim_engine] viewport.render_mode setter failed: {exc}", flush=True)

    try:
        print(f"[genie_sim_engine] viewport hydra_engine (read-back) = {vp.hydra_engine}", flush=True)
    except Exception:
        pass

    # 2. Decide updates_enabled. Default is LIVE (True) — user wants the
    # viewport to refresh continuously by default for debug. Set
    # ISAACSIM_VIEWPORT_UPDATES=0 (or false/no/off) to freeze it instead,
    # which makes simulation_app.update() ~2ms instead of ~33ms.
    if updates_enabled is None:
        env = os.environ.get("ISAACSIM_VIEWPORT_UPDATES", "").strip().lower()
        if env in ("0", "false", "no", "off"):
            updates_enabled = False
        else:
            updates_enabled = True  # default: live

    try:
        vp.updates_enabled = bool(updates_enabled)
        if updates_enabled:
            print(
                "[genie_sim_engine] viewport updates_enabled = True  (live debug view; "
                "set ISAACSIM_VIEWPORT_UPDATES=0 to freeze for ~33ms→~2ms render savings)",
                flush=True,
            )
        else:
            print(
                "[genie_sim_engine] viewport updates_enabled = False  " "(GUI frozen at last frame; cheap render)",
                flush=True,
            )
    except Exception as exc:
        print(f"[genie_sim_engine] viewport.updates_enabled setter failed: {exc}", flush=True)


def _warmup_newton_kernels(solver: str) -> None:
    """Pre-compile Warp CUDA kernels for Newton before the scene loads.

    Newton's ``ModelBuilder.finalize()`` triggers Warp JIT compilation of
    every solver's CUDA kernels.  This is a one-time cost (~5–30 s on first
    run; subsequent launches use the Warp kernel cache and take < 1 s).
    The problem: finalize is called mid-scene-load inside Isaac's GUI event
    loop, which blocks the UI thread and makes the viewport appear frozen.

    The fix: build a tiny 1-particle dummy model here — immediately after
    ``SimulationApp`` starts but before any stage opens — to force kernel
    compilation now.  The user sees a slow startup line in the log instead
    of a mid-run UI freeze.  The kernel cache survives across sessions so
    this overhead only occurs on first launch or after a Warp upgrade.

    The dummy model is discarded immediately; it has no effect on the scene.
    """
    print(
        f"[genie_sim_engine] warming up Newton/{solver} Warp kernels "
        f"(first-run: may take up to 30s; cached on next launch)…",
        flush=True,
    )
    try:
        import newton
        import warp as wp

        b = newton.ModelBuilder()
        # Register custom attributes for the requested solver so its
        # kernel-specific per-particle storage is included in the dummy model.
        solver_map = {
            "xpbd": newton.solvers.SolverXPBD,
            "style3d": newton.solvers.SolverStyle3D,
            "vbd": newton.solvers.SolverVBD,
            "mujoco-warp": newton.solvers.SolverMuJoCo,
            "featherstone": newton.solvers.SolverFeatherstone,
            "semiimplicit": newton.solvers.SolverSemiImplicit,
        }
        cls = solver_map.get(solver.lower())
        if cls and hasattr(cls, "register_custom_attributes"):
            cls.register_custom_attributes(b)

        # Add a minimal cloth grid so finalize allocates particle buffers
        # and triggers the cloth/particle solver kernels.
        if solver.lower() in ("xpbd", "style3d", "vbd"):
            b.add_cloth_grid(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat(0.0, 0.0, 0.0, 1.0),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=2,
                dim_y=2,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.01,
            )

        model = b.finalize("cuda:0")
        del model, b
        print("[genie_sim_engine] Newton kernel warmup complete.", flush=True)
    except Exception as exc:
        # Warmup failure is non-fatal — kernels will compile on first real
        # finalize instead (with the associated UI freeze on first run only).
        print(
            f"[genie_sim_engine] Newton kernel warmup skipped: {exc}",
            flush=True,
        )


def configure_carb_settings(headless: bool, logger) -> None:
    """Apply the carb runtime settings used by the physics loop."""
    import carb

    settings = carb.settings.get_settings()
    try:
        if headless:
            settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
            settings.set_bool("/app/runLoopsGlobal/rateLimitEnabled", False)
            settings.set_int("/app/runLoops/main/rateLimitFrequency", 1000)
            settings.set_bool("/app/runLoops/present/rateLimitEnabled", False)
            settings.set_bool("/persistent/app/runLoops/main/rateLimitEnabled", False)
            settings.set_bool("/app/asyncRendering", True)
            settings.set_bool("/app/asyncRenderingLowLatency", True)
            settings.set_bool("/app/vsync", False)
        else:
            settings.set_bool("/app/runLoops/main/rateLimitEnabled", True)
            settings.set_int("/app/runLoops/main/rateLimitFrequency", 60)
            settings.set_bool("/app/asyncRendering", False)
            settings.set_bool("/app/vsync", False)
        settings.set_float("/physics/minFrameRate", 0.0)
        settings.set_int("/persistent/simulation/minFrameRate", 0)
        settings.set_bool("/rtx/ecoMode/enabled", False)
        settings.set_bool("/persistent/app/usd/muteUsdDiagnostics", False)
    except Exception as exc:
        logger.warn(f"failed to set carb settings: {exc}")
