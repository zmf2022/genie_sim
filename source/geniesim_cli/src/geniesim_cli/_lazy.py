# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lazy-import helpers shared by every ``geniesim_cli`` verb module.

Why this lives in its own module: ``geniesim_cli`` advertises a
zero-heavy-dep install. Every subcommand that needs a sibling
distribution (``geniesim``, ``geniesim_assets``, ``geniesim_ros``) MUST
import it through :func:`import_or_die` so the failure mode is a friendly
hint instead of an opaque ``ModuleNotFoundError`` traceback. Keeping
this in a tiny standalone module also keeps ``cli.py`` free of any
heavyweight imports at module load time.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

from geniesim_cli._style import BOLD, CYAN, DIM, RED, RST


def import_or_die(module_path: str, hint_pkg: str):
    """Lazy-import ``module_path``; print a friendly hint on failure."""
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        print(f"{RED}❌ Cannot import {BOLD}{module_path}{RST}{RED}: {exc}{RST}")
        print()
        print(f"   {DIM}This subcommand requires the {BOLD}{hint_pkg}{RST}{DIM} distribution.{RST}")
        print(f"   {DIM}Install it from a local checkout, then run:{RST} {CYAN}geniesim bootstrap{RST}")
        sys.exit(1)


def distribution_source_root(dist_name: str) -> str:
    """Return the source-tree root of an installed distribution (for deploy)."""
    from pathlib import Path

    try:
        spec = importlib.util.find_spec(dist_name)
    except ModuleNotFoundError:
        spec = None
    if spec is None or spec.origin is None:
        print(f"{RED}❌ Distribution {BOLD}{dist_name}{RST}{RED} is not importable.{RST}")
        print(f"   {DIM}Install it from a local checkout first, e.g.{RST} {CYAN}geniesim bootstrap{RST}{DIM}.{RST}")
        sys.exit(1)
    pkg_dir = Path(spec.origin).resolve().parent
    return str(pkg_dir.parent.parent)


def probe_module(module_name: str) -> tuple[bool, str | None]:
    """Try to import a module; return (ok, error-string-or-None).

    Side-effect-free for ok=True; on failure swallows the exception and
    formats it as ``ClassName: message`` so the caller can render it.
    """
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:  # noqa: BLE001 — surface any failure verbatim
        return False, f"{type(exc).__name__}: {exc}"
