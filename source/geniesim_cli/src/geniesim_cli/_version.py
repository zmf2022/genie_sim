# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Single source of truth for the GenieSim version.

The repo-root ``VERSION`` file is the authoritative source. Every
distribution's ``pyproject.toml`` reads it via
``[tool.setuptools.dynamic] version = {file = "../../VERSION"}``, and
ROS ``package.xml`` files are kept in sync by the
``geniesim version bump`` subcommand (also exposed as a pre-commit
``--check`` hook). At runtime, this module re-exports the same string
so importers can do ``from geniesim_cli import __version__``.

To bump: ``geniesim version bump <NEW_VERSION>`` (or edit ``VERSION``
and run ``geniesim version sync``). Never edit a sibling
``pyproject.toml`` or a ``package.xml`` ``<version>`` by hand.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path


def _read_repo_version() -> str | None:
    """Walk up from this file looking for a sibling-of-source ``VERSION``.

    The repo layout is ``<repo>/source/geniesim_cli/src/geniesim_cli/_version.py``,
    so ``parents[4]`` is the repo root in a source checkout. Editable
    installs hit this branch; wheel installs fall through to the package
    metadata fallback below (the wheel-build step embeds the version via
    ``[tool.setuptools.dynamic]`` so ``importlib.metadata`` knows it).
    """
    here = Path(__file__).resolve()
    for ancestor in (here.parents[4] if len(here.parents) > 4 else here.parent, *here.parents):
        candidate = ancestor / "VERSION"
        if candidate.is_file():
            text = candidate.read_text().strip()
            if text:
                return text
    return None


def _resolve_version() -> str:
    repo = _read_repo_version()
    if repo:
        return repo
    try:
        return _pkg_version("geniesim_cli")
    except PackageNotFoundError:
        return "0.0.0+unknown"


__version__: str = _resolve_version()

__all__ = ["__version__"]
