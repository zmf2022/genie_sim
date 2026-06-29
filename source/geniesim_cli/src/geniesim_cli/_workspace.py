# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Locate the ROS 2 colcon workspace targeted by ``geniesim ros *``.

The resolution order is documented on :func:`ros_workspace_root` and
deliberately layered so that:

1. ``$GENIESIM_WORKSPACE`` always wins (explicit user intent).
2. The current directory wins if it already looks like a colcon
   workspace (in-tree flow: ``cd`` into the repo and build).
3. The bundled workspace inside the ``geniesim_ros`` distribution
   takes over when the user is anywhere outside a checkout. This is
   what makes ``geniesim ros build dev`` work out of the box for
   downstream users who only ``pip install geniesim_ros``.

The ``ros_ws`` lookup uses ``importlib.util.find_spec`` rather than
hard-coded paths so editable installs and wheel installs are both
discovered correctly.
"""

from __future__ import annotations

import importlib.util
import sys

from geniesim_cli import _env
from geniesim_cli._style import BOLD, CYAN, DIM, RED, RST, WHITE


def looks_like_colcon_workspace(path) -> bool:
    """A directory counts as a colcon workspace iff it has ``src/`` with at
    least one nested ``package.xml``. This matches colcon's own discovery
    heuristic and avoids false positives like an empty ``src/`` or a
    pip-only source layout (no ``package.xml`` anywhere).

    Implementation note: we deliberately avoid ``Path.rglob`` here.
    On Python <3.13, ``rglob`` does not descend into symlinked
    directories, which would break the geniesim_ros pattern of
    ``ros_ws/src/<pkg>`` being a symlink into another repo. Instead we
    iterate the immediate children (``iterdir`` does follow symlinks)
    and then ``rglob`` inside each one (where the descent is into a
    real, non-symlinked directory tree)."""
    from pathlib import Path

    src = Path(path) / "src"
    if not src.is_dir():
        return False
    try:
        children = list(src.iterdir())
    except OSError:
        return False
    for child in children:
        if not child.is_dir():
            continue
        if (child / "package.xml").is_file():
            return True
        try:
            next(child.rglob("package.xml"))
            return True
        except StopIteration:
            continue
    return False


def _bundled_ros_ws():
    """Return the bundled ``<geniesim_ros>/src/ros_ws`` path, or None."""
    from pathlib import Path

    try:
        spec = importlib.util.find_spec("geniesim_ros")
    except ModuleNotFoundError:
        spec = None
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent.parent / "ros_ws"


def geniesim_ros_dist_root():
    """Return the source root of the installed ``geniesim_ros`` distribution.

    Used by the release flow to know where to stage colcon's ``install/``
    so that ``setup.py`` / ``pyproject.toml`` can include it as
    package_data when a wheel is built.

    Returns ``None`` when ``geniesim_ros`` is not installed.
    """
    from pathlib import Path

    try:
        spec = importlib.util.find_spec("geniesim_ros")
    except ModuleNotFoundError:
        spec = None
    if spec is None or spec.origin is None:
        return None
    pkg_dir = Path(spec.origin).resolve().parent
    return pkg_dir.parent.parent


def ros_workspace_root() -> str:
    """Locate the colcon workspace root used by ``geniesim ros *``.

    Resolution order (first hit wins):

    1. ``$GENIESIM_WORKSPACE`` — explicit override, used as-is.
    2. Current working directory, if it looks like a colcon workspace
       (i.e. has ``src/<somewhere>/package.xml``). This preserves the
       in-tree flow where users ``cd`` into the workspace first.
    3. The bundled colcon workspace shipped by the ``geniesim_ros``
       distribution at ``<geniesim_ros source root>/src/ros_ws``.
       This is the default when the user is anywhere outside a colcon
       tree but has ``geniesim_ros`` installed (typically ``pip install
       -e source/geniesim_ros`` from a checkout).

    If none match, the function exits with a friendly error rather than
    silently driving colcon at a non-workspace path.
    """
    from pathlib import Path

    override = _env.workspace()
    if override:
        return str(Path(override).resolve())

    cwd = Path.cwd().resolve()
    if looks_like_colcon_workspace(cwd):
        return str(cwd)

    ros_ws = _bundled_ros_ws()
    if ros_ws is not None and looks_like_colcon_workspace(ros_ws):
        return str(ros_ws)

    print(f"{RED}❌ No colcon workspace found.{RST}")
    print()
    print(f"   {DIM}Checked, in order:{RST}")
    print(f"     {WHITE}$GENIESIM_WORKSPACE{RST}              {DIM}(unset){RST}")
    print(f"     {WHITE}{cwd}{RST}  {DIM}(no src/**/package.xml){RST}")
    if ros_ws is not None:
        print(f"     {WHITE}{ros_ws}{RST}  {DIM}(no src/**/package.xml){RST}")
    else:
        print(f"     {WHITE}geniesim_ros bundled ws{RST}    {DIM}(geniesim_ros not importable){RST}")
    print()
    print(f"   {DIM}Fix one of:{RST}")
    print(f"     {CYAN}cd{RST} into a colcon workspace, or")
    print(f"     {CYAN}export GENIESIM_WORKSPACE=/path/to/ws{RST}, or")
    print(f"     {CYAN}pip install -e <repo>/source/geniesim_ros{RST}")
    sys.exit(1)
