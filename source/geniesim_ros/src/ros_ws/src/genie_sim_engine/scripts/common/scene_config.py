#!/usr/bin/env python3

# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Scene-YAML readers shared between assemble_scene and genie_sim_engine.

The ``manifest.json`` written by ``assemble_scene`` is intentionally limited
to **static** fields — the ones that determine which assets get loaded and
where prims live in USD namespace (``scene_usda``, ``robot_usda``,
``robot_prefix``, etc.). Anything that's a runtime *behavior* —
``pin_base_to_world``, ``convert_joints_to_fixed``, init joint poses,
overrides — is re-read live from the scene YAML at engine startup so the
operator can tweak behavior without invalidating the bake cache.

This module gathers the shared parsers so both ``assemble_scene`` (for
debugging / one-shot logs) and ``genie_sim_engine`` (for live runtime use)
read the same shapes from the same YAML keys without code duplication.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

_CONVERT_TOKENS: Set[str] = {"base", "head", "body"}


def _parse_bool(source: Dict[str, Any], key: str, default: bool) -> bool:
    raw = source.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() not in {"false", "0", "no", "off"}
    return bool(raw)


def parse_pin_base_to_world(robot_section: Dict[str, Any]) -> bool:
    """Read ``robot.robot_source.pin_base_to_world``.  Defaults to ``False``.

    Controls whether the URDF→USD ``root_joint`` FixedJoint is kept
    (base_link welded to world) or deactivated at runtime (base_link
    free under physics).

    Default is **False** — most arm-only scenes want the base welded
    by IsaacSim's URDF importer; mobile-robot scenes remove the weld
    so wheel-on-floor contact drives the base.  We encode "the base
    is meant to move" as the default and require an explicit opt-in
    to anchor it for stationary scenes.
    The Newton engine reads this flag to:
      * deactivate ``/<robot>/Joints/root_joint`` before ``add_usd``
        when ``pin_base_to_world == False`` (mobile)
      * leave it alone when ``True`` (stationary), in which case the
        ``kinematic-control`` substep regime is correct and gravity /
        contacts can be zeroed during the rigid step.
    """
    source = robot_section.get("robot_source") if isinstance(robot_section, dict) else None
    return _parse_bool(source, "pin_base_to_world", False) if isinstance(source, dict) else False


def parse_convert_joints_to_fixed(robot_section: Dict[str, Any]) -> List[str]:
    """Read ``robot.robot_source.convert_joints_to_fixed``.

    Returns a normalized list of sub-tree names whose articulated joints
    should be REPLACED with ``UsdPhysics.FixedJoint`` at runtime to
    shrink Featherstone's mass matrix (or trim mjwarp's actuator
    count).  Supported tokens are:

      * ``"base"``  — joints whose name contains ``chassis``  (chassis
                      attach joint + wheel steering / spin joints)
      * ``"head"``  — joints whose name contains ``head_joint``
      * ``"body"``  — joints whose name contains ``body_joint`` (G2's
                      5-DOF torso chain)

    Bodies on both sides of each replaced joint STAY in
    ``model.body_label`` so TF publishes them at their init pose, but
    the joints contribute 0 DOFs.

    Accepts a YAML list or a single string.  Defaults to an empty list
    (no conversion) when the field is missing — mobile-robot configs
    Just Work, arm-only setups that want the welded fast-path author
    the list explicitly.

    YAML::

      robot_source:
        convert_joints_to_fixed: [base, head, body]
    """
    source = robot_section.get("robot_source") if isinstance(robot_section, dict) else None
    if not isinstance(source, dict) or "convert_joints_to_fixed" not in source:
        return []
    raw = source["convert_joints_to_fixed"]
    if raw is None:
        return []
    if isinstance(raw, str):
        # Allow a single token without listification — `convert_joints_to_fixed: base`
        raw_list = [raw]
    elif isinstance(raw, (list, tuple)):
        raw_list = list(raw)
    else:
        return []
    out: List[str] = []
    for item in raw_list:
        s = str(item).strip().lower()
        if not s:
            continue
        if s not in _CONVERT_TOKENS:
            # Unknown token — leave it in the list anyway so the lifecycle
            # logger can complain with the actual offending string and the
            # operator knows what to fix.  No silent drop.
            out.append(s)
            continue
        if s not in out:
            out.append(s)
    return out
