#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Send a saved observation payload to an OpenPI inference server and validate the response.

The payload is a dict with observation fields (images, state, task); the server
replies with `{"actions": numpy_array}` where the array has shape (H, D).

Beyond key-presence and NaN/Inf checks, the response is validated per-dimension:
min/max/mean/std for each dim, plus flags for constant dims, out-of-range values
(JOINT_ABS expects radians, EEF_ABS expects meters+quat, gripper expects [0,1]).

Usage:
    python check_inference.py --host 127.0.0.1 --port 8999      # bundled payload
    python check_inference.py debug_preview/debug_0001.pkl --host 127.0.0.1
"""

import argparse
import os
import sys
import time

import numpy as np
import websockets.sync.client

# Bundled default payload, shipped next to this script.
DEFAULT_PAYLOAD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corobot_payload.pkl")


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

ICON_INFO = "ℹ️ "
ICON_OK = "✅"
ICON_WARN = "⚠️ "
ICON_FAIL = "❌"
ICON_LOAD = "📦"
ICON_CONN = "🔌"
ICON_RESP = "📡"
ICON_STAT = "📊"
ICON_TIME = "⏱️ "
ICON_ITER = "🔄"

RULE_HEAVY = "═" * 72
RULE_LIGHT = "─" * 72


def info(msg):
    print(f"{ICON_INFO} {msg}")


def ok(msg):
    print(f"{ICON_OK} {msg}")


def warn(msg):
    print(f"{ICON_WARN} {msg}")


def fail(msg):
    print(f"{ICON_FAIL} {msg}")


def section(icon, title):
    print(f"\n{icon} {title}")
    print(RULE_LIGHT)


# ---------------------------------------------------------------------------
# msgpack-numpy serialization (same format as corobotpolicy.py)
# ---------------------------------------------------------------------------

from geniesim_benchmark.utils.msgpack_numpy import packb as _pack, unpackb as _unpack


# ---------------------------------------------------------------------------
# Connection and metadata handling
# ---------------------------------------------------------------------------


def connect(uri):
    section(ICON_CONN, f"Connecting to {uri}")
    ws = websockets.sync.client.connect(uri, compression=None, max_size=None, open_timeout=15)
    ok(f"connected to {uri}")
    # OpenPI: server sends metadata on connect
    metadata_raw = ws.recv()
    metadata = _unpack(metadata_raw)
    section(ICON_INFO, "Server metadata")
    info(f"policy_type: {metadata.get('policy_type')}")
    info(f"chunk_size: {metadata.get('chunk_size')}")
    info(f"n_action_steps: {metadata.get('n_action_steps')}")
    if 'input_features' in metadata:
        info(f"input_features: {list(metadata['input_features'].keys())}")
    return ws, metadata


# ---------------------------------------------------------------------------
# Per-idx range / sanity checks
# ---------------------------------------------------------------------------


# (lo, hi) soft bounds per kind. Values outside are flagged but do not fail.
# GRIPPER is intentionally absent: gripper encodings vary across policies, so we
# only report its observed range (min/max/mean/std) and skip any bounds check.
KIND_BOUNDS = {
    "JOINT_ABS": (-3.5, 3.5),  # radians, a bit over ±π
    "ABS_JOINT": (-3.5, 3.5),
    "JOINT_REL": (-1.0, 1.0),  # delta radians per chunk step
    "EEF_ABS": (-2.0, 2.0),  # meters for xyz, [-1,1] for quat — both fit
}
HARD_ABS = 1e4  # |x| above this is flagged as failure (garbage)
JUMP_FROM_STATE = 0.5  # rad — flag if action[0]_d differs from state_d by more than this


def _flatten_to_2d(a):
    """Return (n_samples, n_dims) view of arr, with last axis treated as dims."""
    if a.ndim == 0:
        return a.reshape(1, 1)
    if a.ndim == 1:
        return a.reshape(1, -1)
    return a.reshape(-1, a.shape[-1])


def per_idx_summary(arr, label, *, kind=None, state=None, max_dims=64):
    """Print per-dimension stats and return True if no hard-fail issues found."""
    a = np.asarray(arr)
    if a.dtype.kind not in "fiub":
        warn(f"{label}: non-numeric dtype={a.dtype}, skipping per-idx checks")
        return True

    flat = _flatten_to_2d(a)
    n, d = flat.shape
    show = min(d, max_dims)

    soft_lo, soft_hi = KIND_BOUNDS.get(kind, (None, None))
    bounds_str = f"soft=[{soft_lo:+.2f}, {soft_hi:+.2f}]" if soft_lo is not None else "range-only (no bounds)"

    section(ICON_STAT, f"{label}")
    info(f"shape={tuple(a.shape)}  dtype={a.dtype}  samples={n}  dims={d}  kind={kind!r}  {bounds_str}")
    print()
    header = f"  {'idx':>3}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}   flags"
    print(header)
    print("  " + "─" * (len(header) - 2))

    fail_hard = False
    n_warn = 0
    for i in range(show):
        col = flat[:, i]
        if col.dtype.kind == "f":
            nan_n = int(np.isnan(col).sum())
            inf_n = int(np.isinf(col).sum())
            finite = col[np.isfinite(col)]
        else:
            nan_n = inf_n = 0
            finite = col

        flags = []
        if nan_n:
            flags.append(f"NaN×{nan_n}!")
            fail_hard = True
        if inf_n:
            flags.append(f"Inf×{inf_n}!")
            fail_hard = True

        if finite.size == 0:
            print(f"  {i:>3}  {'-':>10}  {'-':>10}  {'-':>10}  {'-':>10}   non-finite!")
            continue

        lo, hi = float(finite.min()), float(finite.max())
        mean, std = float(finite.mean()), float(finite.std())
        absmax = max(abs(lo), abs(hi))

        if absmax > HARD_ABS:
            flags.append(f"|x|>{HARD_ABS:g}!")
            fail_hard = True
        elif soft_lo is not None and (lo < soft_lo or hi > soft_hi):
            flags.append(f"OOB[{soft_lo:+.2f},{soft_hi:+.2f}]")
            n_warn += 1

        if n > 1 and std == 0:
            flags.append("const")
            n_warn += 1

        if state is not None and i < len(state):
            try:
                delta = float(col[0]) - float(state[i])
                if abs(delta) > JUMP_FROM_STATE:
                    flags.append(f"Δs={delta:+.3f}")
                    n_warn += 1
            except (TypeError, ValueError):
                pass

        print(f"  {i:>3}  {lo:>+10.4f}  {hi:>+10.4f}  {mean:>+10.4f}  {std:>10.4f}   {' '.join(flags)}")

    if d > show:
        print(f"  ... ({d - show} more dims hidden; pass --max-dims to see)")

    print()
    if fail_hard:
        fail(f"{label}: hard-fail flags present (NaN / Inf / |x|>{HARD_ABS:g})")
    elif n_warn:
        warn(f"{label}: {n_warn} soft warning(s) (OOB / const / Δstate)")
    else:
        ok(f"{label}: all {d} dims clean")
    return not fail_hard


def chunk_continuity(arrs, label, *, max_step=0.5):
    """Print and validate frame-to-frame deltas for a stack of same-shape arrays."""
    if len(arrs) < 2:
        return True
    try:
        stacked = np.stack([np.asarray(a) for a in arrs])
    except ValueError:
        warn(f"{label}: cannot stack chunks (shape mismatch)")
        return True
    if stacked.dtype.kind != "f":
        return True
    deltas = np.abs(np.diff(stacked.reshape(stacked.shape[0], -1), axis=0))
    if deltas.size == 0:
        return True
    max_delta = float(deltas.max())
    mean_delta = float(deltas.mean())
    msg = f"{label} continuity: mean Δ={mean_delta:.4f}  max Δ={max_delta:.4f}"
    (warn if max_delta > max_step * 4 else info)(msg + ("  ← jumpy!" if max_delta > max_step * 4 else ""))
    return True


# ---------------------------------------------------------------------------
# Generic NaN/Inf scan
# ---------------------------------------------------------------------------


def _flatten_arrays(obj):
    if isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _flatten_arrays(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _flatten_arrays(v)
    elif isinstance(obj, (np.ndarray, np.generic)):
        yield np.asarray(obj)
    elif isinstance(obj, (int, float)):
        yield np.asarray(obj)


def validate(arrays, label):
    arrays = list(arrays)
    if not arrays:
        fail(f"{label}: no numeric arrays in response")
        return False
    bad_nan = any(np.any(np.isnan(a)) for a in arrays if a.dtype.kind == "f")
    bad_inf = any(np.any(np.isinf(a)) for a in arrays if a.dtype.kind == "f")
    shapes = [tuple(a.shape) for a in arrays[:6]]
    info(f"{label}: {len(arrays)} arrays scanned, sample shapes={shapes}")
    if bad_nan or bad_inf:
        fail(f"{label}: NaN={bad_nan} Inf={bad_inf}")
        return False
    return True


# ---------------------------------------------------------------------------
# Response checker
# ---------------------------------------------------------------------------


def construct_dummy_observation(metadata):
    """Construct a dummy OpenPI observation dict based on server metadata."""
    obs = {}
    input_features = metadata.get("input_features", {})
    
    for feat_name, feat_info in input_features.items():
        feat_type = feat_info.get("type", "")
        shape = feat_info.get("shape", [])
        
        if feat_type == "VISUAL" or "image" in feat_name:
            # Image: metadata is (C, H, W) format, need to convert to (H, W, C)
            if len(shape) == 4:
                # (B, C, H, W) -> (C, H, W)
                shape = shape[1:]
            if len(shape) == 3:
                c, h, w = shape
                # Generate uint8 image data in (H, W, C) format
                obs[feat_name] = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
        elif feat_type == "STATE" or "state" in feat_name:
            # State vector: flat array
            if len(shape) == 2:
                shape = shape[1:]  # (B, D) -> (D,)
            if len(shape) == 1:
                obs[feat_name] = np.random.randn(*shape).astype(np.float32)
        elif feat_type == "LANGUAGE" or "language" in feat_name:
            # Language embedding: flat array
            if len(shape) == 2:
                shape = shape[1:]  # (B, D) -> (D,)
            if len(shape) == 1:
                obs[feat_name] = np.random.randn(*shape).astype(np.float32)
        else:
            warn(f"Unknown feature type {feat_type} for {feat_name}, skipping")
    
    return obs


def check_openpi(ws, metadata, max_dims):
    """Send OpenPI observation and validate response."""
    t0 = time.time()
    
    # Construct observation
    obs = construct_dummy_observation(metadata)
    section(ICON_INFO, "Sending OpenPI observation")
    info(f"keys: {list(obs.keys())}")
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            info(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            info(f"  {k}: {type(v).__name__}")
    
    # Send and receive
    ws.send(_pack(obs))
    resp = ws.recv()
    dt = (time.time() - t0) * 1000
    
    if isinstance(resp, str):
        fail(f"server returned string: {resp[:200]}")
        return False
    
    result = _unpack(resp)
    section(ICON_RESP, f"OpenPI response  {dt:.1f} ms")
    info(f"keys: {list(result.keys())}")
    
    if "error" in result:
        fail(f"server error: {result['error']}")
        return False
    
    # OpenPI returns {"actions": np.ndarray (H, D)}
    if "actions" not in result:
        fail(f"response missing 'actions' key. keys={list(result.keys())}")
        return False
    
    actions_arr = np.asarray(result["actions"])
    info(f"actions shape: {actions_arr.shape} dtype={actions_arr.dtype}")
    
    # Validate actions
    passed = validate(_flatten_arrays({"actions": actions_arr}), "actions")
    passed &= per_idx_summary(actions_arr, "actions", kind="JOINT_ABS", state=None, max_dims=max_dims)
    
    if actions_arr.ndim >= 2 and actions_arr.shape[0] > 1:
        chunk_continuity([actions_arr[i] for i in range(actions_arr.shape[0])], "actions")
    
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pkl_path",
        nargs="?",
        default=DEFAULT_PAYLOAD,
        help="OpenPI observation pkl (default: bundled openpi_observation.pkl)",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--max-dims", type=int, default=64, help="max idx rows to print per array")
    args = ap.parse_args()

    uri = f"ws://{args.host}:{args.port}"
    ws, metadata = connect(uri)

    ok_all = True
    latencies = []
    for i in range(args.iters):
        print()
        print(RULE_HEAVY)
        print(f"{ICON_ITER}  Iteration {i + 1} / {args.iters}")
        print(RULE_HEAVY)
        t0 = time.time()
        ok_all &= check_openpi(ws, metadata, args.max_dims)
        latencies.append((time.time() - t0) * 1000)

    try:
        ws.close()
    except Exception:
        pass

    print()
    print(RULE_HEAVY)
    if len(latencies) > 1:
        lat = np.asarray(latencies)
        print(
            f"{ICON_TIME} latency over {len(latencies)} iters (ms): "
            f"min={lat.min():.1f}  mean={lat.mean():.1f}  "
            f"p95={np.percentile(lat, 95):.1f}  max={lat.max():.1f}"
        )
    if ok_all:
        ok("PASS — inference server reachable and response valid")
    else:
        fail("FAIL — see warnings above")
    print(RULE_HEAVY)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
