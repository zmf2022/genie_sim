# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared physics-loop primitives for both engine entry points.

``_merge_params_file``, ``_parse_args`` — ROS-arg helpers.
``_publish_tick`` — clock/joints/bodies/odom publish; identical in both.
``EngineRunLoop`` — common physics-step scaffold; render is injected via
  a callable so Kit-specific and GL-specific code stays in the entry point.
"""

from __future__ import annotations

import time


def _merge_params_file(out: dict, path: str, prefix: str = "genie_sim_engine") -> None:
    try:
        import yaml
    except Exception as exc:
        print(f"[{prefix}] WARN: cannot import yaml to read {path}: {exc}", flush=True)
        return
    try:
        with open(path, "r") as f:
            doc = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[{prefix}] WARN: cannot read params file {path}: {exc}", flush=True)
        return
    if not isinstance(doc, dict):
        return
    for _node_key, node_block in doc.items():
        if not isinstance(node_block, dict):
            continue
        ros_params = node_block.get("ros__parameters")
        if not isinstance(ros_params, dict):
            continue
        for k, v in ros_params.items():
            if isinstance(v, bool):
                out[str(k)] = "true" if v else "false"
            else:
                out[str(k)] = str(v)


def _parse_args(argv, prefix: str = "genie_sim_engine") -> dict:
    """Parse ``--ros-args`` style argv into a flat ``{name: str}`` dict.

    Supports ``--params-file <path>`` and ``-p key:=value``. CLI ``-p``
    overrides take precedence over ``--params-file`` values (same as rclcpp).
    """
    out: dict = {}
    cli_overrides: dict = {}
    it = iter(argv)
    for a in it:
        if a == "--ros-args":
            continue
        if a == "-p":
            kv = next(it, "")
            if ":=" in kv:
                k, v = kv.split(":=", 1)
                cli_overrides[k.strip()] = v.strip()
        elif a.startswith("-p="):
            kv = a[3:]
            if ":=" in kv:
                k, v = kv.split(":=", 1)
                cli_overrides[k.strip()] = v.strip()
        elif a == "--params-file":
            path = next(it, "")
            if path:
                _merge_params_file(out, path, prefix)
        elif a.startswith("--params-file="):
            path = a[len("--params-file=") :]
            if path:
                _merge_params_file(out, path, prefix)
    out.update(cli_overrides)
    return out


def _publish_tick(sim, sim_time: float, _core) -> float:
    """Publish clock / joints / bodies / odom.  Returns total ms."""
    t0 = time.monotonic()
    _core.publish_clock(sim_time)
    t1 = time.monotonic()

    jpos, jvel = sim.get_joint_states()
    _core.publish_joint_states(sim_time, jpos, jvel)
    t2 = time.monotonic()

    body_xyzwxyz, _frames = sim.get_body_transforms()
    if body_xyzwxyz.shape[0] > 0:
        _core.publish_body_tf_render(sim_time, body_xyzwxyz)
    t3 = time.monotonic()

    base = sim.get_odom(sim_time)
    if base is not None:
        base_pose, base_twist = base
        _core.publish_odom(sim_time, base_pose, base_twist)

    publish_ms = (time.monotonic() - t0) * 1000.0
    sim.note_publish_phase(
        clock_ms=(t1 - t0) * 1000.0,
        joints_ms=(t2 - t1) * 1000.0,
        bodies_ms=(t3 - t2) * 1000.0,
        odom_ms=(time.monotonic() - t3) * 1000.0,
    )
    return publish_ms


class _Acc:
    """Lightweight mean+max accumulator for Python-side perf supplements."""

    __slots__ = ("_sum", "_count", "mx")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0
        self.mx = 0.0

    def push(self, v: float) -> None:
        self._sum += v
        self._count += 1
        if v > self.mx:
            self.mx = v

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0


class EngineRunLoop:
    """Shared physics-step scaffold.

    ``render_hook(now, next_tick, sim_time) -> (render_ms, render_sync_ms, did_render)``
        Called each tick after ``tick_extras()`` and any ``post_step_hooks``.
        Return ``(0.0, 0.0, False)`` when no render fires this tick.
        ``render_sync_ms`` is the total including any post-render GPU sync
        (equals ``render_ms`` when no explicit sync is performed).

    ``exit_check() -> bool``
        If provided, called once per tick BEFORE the timing wait.  Return
        ``False`` to break the loop (e.g. GL window closed).

    ``post_step_hooks``
        Optional list of ``Callable[[float], None]`` called after
        ``tick_extras()`` each tick, receiving ``sim_time``.  Timed together
        with ``tick_extras`` under ``t_phase_extras``.  Append to extend
        step behaviour without subclassing (IsaacLab EventManager pattern).

    Performance stats
    -----------------
    The C++ ring (``note_step_timing`` → ``log_stats_if_due``) reports
    mean/min/max/std/p95/p99 for: interval, step, physx, render (sync-
    inclusive), publish, spin, plus overrun count and scheduler counters.

    The Python supplement appended to each stats message adds dimensions
    the C++ ring doesn't capture:

    * ``Cmd``      — ``pop_commands + apply_commands`` latency.
    * ``Extras``   — ``tick_extras + post_step_hooks`` latency.
    * ``GPU-sync`` — ``render_sync_ms − render_ms`` on render ticks: time
                     the CPU waited for the GPU to finish the render frame
                     after dispatch (zero when GENIESIM_RENDER_SYNC is off).
    * ``Budget``   — avg and worst-case ``(step+publish) / dt`` utilisation.
    """

    def __init__(self, core, sim, dt: float, logger, post_step_hooks=None, rtf: float = 1.0):
        self._core = core
        self._sim = sim
        self._dt = dt
        self._wall_tick = dt / max(rtf, 1e-6)
        self._logger = logger
        self._post_step_hooks = list(post_step_hooks) if post_step_hooks else []
        # Call log_stats_if_due at most once per simulated second to avoid
        # paying a pybind round-trip every tick for a check that returns "" ~99%
        # of the time.  The C++ side still owns the actual log interval.
        self._stats_interval_ticks = max(1, round(1.0 / dt))

    def spin(self, render_hook=None, exit_check=None) -> None:
        core = self._core
        sim = self._sim
        dt = self._dt
        wall_tick = self._wall_tick
        logger = self._logger
        post_step_hooks = self._post_step_hooks
        stats_interval = self._stats_interval_ticks
        target_ms = wall_tick * 1000.0

        next_tick = time.monotonic()
        sim_time = 0.0
        behind_count = 0
        stats_tick = 0

        # spin_ms = idle time from end of previous tick to start of this tick.
        # Seeded to now so the first tick reports 0 spin (no previous tick).
        t_prev_tick_end = time.monotonic()

        # /rtf is measured and published every physics step. The value is
        # the instantaneous ratio dt_sim / wall_delta — the wall time
        # actually elapsed for this step's wall_tick. No smoothing: the
        # topic reflects each step's true factor.
        _rtf_value = 1.0
        _rtf_prev_wall = time.monotonic()

        # Python-side rolling accumulators — reset when C++ stats fire.
        _acc_cmd = _Acc()  # pop_commands + apply_commands
        _acc_extras = _Acc()  # tick_extras + post_step_hooks
        _acc_gpu_sync = _Acc()  # render_sync_ms − render_ms (render ticks only)
        _acc_budget = _Acc()  # step_ms + publish_ms (budget consumption per tick)

        try:
            while core.ok():
                if exit_check is not None and not exit_check():
                    break

                now = time.monotonic()
                if now < next_tick:
                    remaining = next_tick - now
                    if remaining > 0.0005:
                        time.sleep(remaining * 0.9)
                    continue
                # Idle time since last tick's work finished.
                spin_ms = (now - t_prev_tick_end) * 1000.0

                # --- commands ---
                _t_cmd = time.monotonic()
                pos_dict, _eff_dict, steer_dict, drive_dict, cmd_4ws_stamp = core.pop_commands()
                sim.apply_commands(
                    cmd_positions=pos_dict,
                    cmd_4ws_steer_pos=steer_dict,
                    cmd_4ws_drive_vel=drive_dict,
                    cmd_4ws_stamp=cmd_4ws_stamp,
                )
                cmd_ms = (time.monotonic() - _t_cmd) * 1000.0

                step_start = time.monotonic()
                # Time spent inside ``sim.step()`` — the physics solver call,
                # whichever backend is active (PhysX, Newton-in-Isaac, mjwarp,
                # Featherstone, ...).  Logged as the "Solver" line in the
                # 1 Hz physics-stats block.
                solver_ms = 0.0
                render_ms = 0.0
                render_sync_ms = 0.0
                did_render = False
                t_phase_step = 0.0
                t_phase_extras = 0.0
                ok = False

                try:
                    _t = time.monotonic()
                    solver_ms = sim.step(dt, step_start)
                    t_phase_step = (time.monotonic() - _t) * 1000.0

                    _t = time.monotonic()
                    sim.tick_extras()
                    for hook in post_step_hooks:
                        hook(sim_time)
                    t_phase_extras = (time.monotonic() - _t) * 1000.0

                    if render_hook is not None:
                        render_ms, render_sync_ms, did_render = render_hook(now, next_tick, sim_time)

                    sim.note_render(render_ms, did_render)
                    sim.note_phase_timing(
                        step_ms=t_phase_step,
                        extras_ms=t_phase_extras,
                        render_ms=render_ms,
                        render_sync_ms=render_sync_ms,
                        did_render=did_render,
                    )
                    ok = True
                except Exception as exc:
                    logger.warn(f"step failed: {exc}")

                step_ms = (time.monotonic() - step_start) * 1000.0

                if not ok:
                    t_prev_tick_end = time.monotonic()
                    next_tick += wall_tick
                    continue

                sim_time += dt
                publish_ms = _publish_tick(sim, sim_time, core)

                # /rtf — instantaneous dt_sim / wall_delta for this physics
                # step, published every step.
                _wall_delta = now - _rtf_prev_wall
                if _wall_delta > 1e-6:
                    _rtf_value = dt / _wall_delta
                _rtf_prev_wall = now
                core.publish_rtf(float(_rtf_value))

                # Mark end of all active work; next spin_ms starts here.
                t_prev_tick_end = time.monotonic()

                core.note_step_timing(step_ms, solver_ms, render_sync_ms, publish_ms, spin_ms, did_render, now)

                # Python-side accumulation.
                _acc_cmd.push(cmd_ms)
                _acc_extras.push(t_phase_extras)
                _acc_budget.push(step_ms + publish_ms)
                if did_render:
                    _acc_gpu_sync.push(render_sync_ms - render_ms)

                stats_tick += 1
                if stats_tick >= stats_interval:
                    stats_tick = 0
                    msg = core.log_stats_if_due(now)
                    if msg:
                        n_render = _acc_gpu_sync._count
                        py_lines = (
                            f"  Cmd:      mean={_acc_cmd.mean:.2f}  max={_acc_cmd.mx:.2f} ms"
                            f"  (pop_commands + apply_commands)\n"
                            f"  Extras:   mean={_acc_extras.mean:.2f}  max={_acc_extras.mx:.2f} ms"
                            f"  (tick_extras + hooks)\n"
                            f"  GPU-sync: mean={_acc_gpu_sync.mean:.2f}  max={_acc_gpu_sync.mx:.2f} ms"
                            f"  ({n_render} render ticks, sync_ms − dispatch_ms)\n"
                            f"  Budget:   mean={_acc_budget.mean:.2f}ms"
                            f" ({_acc_budget.mean / target_ms * 100:.0f}%)"
                            f"  worst={_acc_budget.mx:.2f}ms"
                            f" ({_acc_budget.mx / target_ms * 100:.0f}%)"
                            f"  of {target_ms:.1f}ms target"
                        )
                        logger.info(msg + "\n" + py_lines)
                        _acc_cmd.reset()
                        _acc_extras.reset()
                        _acc_gpu_sync.reset()
                        _acc_budget.reset()

                next_tick += wall_tick
                if next_tick < now - wall_tick:
                    behind_count += 1
                    if behind_count % 20 == 1:
                        logger.warn(
                            f"loop fell behind by >1 tick: clamped {behind_count} times "
                            f"(step={step_ms:.1f} ms, target={target_ms:.1f} ms)"
                        )
                    next_tick = now + wall_tick

        finally:
            try:
                sim.shutdown()
            except Exception:
                pass
            try:
                core.shutdown()
            except Exception:
                pass
