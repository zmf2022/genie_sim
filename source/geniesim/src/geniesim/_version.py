# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Resolve the package version from the repo-root ``VERSION`` file.

The repo-root ``VERSION`` file is the single source of truth across the
whole GenieSim stack — every distribution's ``pyproject.toml`` points
at it via ``[tool.setuptools.dynamic] version`` and every package
re-exports it via this helper. Edits go through
``geniesim version bump <NEW>`` (or hand-edit ``VERSION`` then
``geniesim version sync``); never hardcode a literal here.

Resolution order:

1. Walk up the filesystem from this file looking for a ``VERSION``
   alongside ``source/`` — hits in editable installs / source checkouts.
2. Fall back to ``importlib.metadata.version`` of the installed
   distribution — hits in wheel installs where ``source/`` isn't
   present but ``[tool.setuptools.dynamic]`` baked the version in.
3. Return ``"0.0.0+unknown"`` as a last resort.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _read_repo_version() -> str | None:
    here = Path(__file__).resolve()
    for ancestor in (here.parents[4] if len(here.parents) > 4 else here.parent, *here.parents):
        candidate = ancestor / "VERSION"
        if candidate.is_file():
            text = candidate.read_text().strip()
            if text:
                return text
    return None


def _resolve_version(dist_name: str) -> str:
    repo = _read_repo_version()
    if repo:
        return repo
    try:
        return _pkg_version(dist_name)
    except PackageNotFoundError:
        return "0.0.0+unknown"
