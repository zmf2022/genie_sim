# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Lazy extraction of the bundled colcon install tree.

Why this exists
---------------
The wheel ships ``_ros_install.tar.gz`` — a tar of the colcon install
tree built by ``BdistWheelWithColcon`` (see ``setup.py``). Tar is used
because it preserves ``st_mode`` natively, so ROS 2 entrypoints under
``lib/<pkg>/<script>.py`` keep colcon's ``0o755`` bit. The alternative
(``package_data`` glob) silently strips modes via setuptools'
``copy_file`` + ``bdist_wheel``'s canonical ``0o100664``, breaking
``ros2 run`` (it filters by ``os.access(.., os.X_OK)``).

When ``import geniesim_ros`` runs, ``__init__.py`` calls
:func:`ensure_ros_install`, which extracts the tarball into
``<package_dir>/_ros_install/`` if it isn't already extracted (or is
out-of-date relative to the tarball's content hash). Subsequent imports
short-circuit on the hash match and return immediately.

Modes of operation
------------------
* **Wheel install** (``pip install geniesim_ros``). The tarball is
  present at ``<package_dir>/_ros_install.tar.gz``; first import
  extracts it; the install path is ``<package_dir>/_ros_install``.
* **Editable install** (``pip install -e``). No tarball is bundled (the
  ``BdistWheelWithColcon`` hook never runs). If a dev built the colcon
  workspace via ``geniesim ros build dev`` and staged it next to the
  package, ``_ros_install/`` already exists on disk and we use it
  directly. Otherwise this is a no-op — the dev hasn't built yet, and
  callers that try to source ``setup.bash`` will get a clear "not
  found" from the shell.
* **Source checkout (no install)**. Same as editable — no tarball, may
  or may not have ``_ros_install/`` on disk.

Concurrency
-----------
First-extract is guarded by ``fcntl.flock`` on a sentinel file. Two
parallel ``import geniesim_ros`` calls (e.g. pytest workers) race to
acquire the lock; the loser sees the tree already extracted by the
winner and returns immediately. ``flock`` is best-effort — on
filesystems that don't support it (some NFS / overlay setups), we
degrade to a "first writer wins, others may briefly see partial state"
mode. The hash check on the next import re-extracts if a partial state
slipped through.

Public API
----------
``ensure_ros_install()`` returns the path to the extracted install tree
(or ``None`` if no tarball is bundled and no pre-staged tree exists).
Idempotent and thread-safe. Cheap on the hot path — a single hash
comparison.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional

# All paths anchored to this file's parent directory (the package root).
# Resolving here keeps the helper independent of CWD and importable
# before the package's ``__init__`` finishes executing.
_PKG_DIR = Path(__file__).resolve().parent
_TARBALL = _PKG_DIR / "_ros_install.tar.gz"
_INSTALL_DIR = _PKG_DIR / "_ros_install"
_HASH_SIDECAR = _INSTALL_DIR / ".bootstrap_hash"
_LOCK_FILE = _PKG_DIR / "_ros_install.lock"

_HASH_BLOCK_SIZE = 1 << 20  # 1 MiB chunks for SHA-256 streaming


def _hash_file(path: Path) -> str:
    """Stream-hash ``path`` with SHA-256. Returns hex digest.

    Streamed so a 50+ MB tarball doesn't allocate all-at-once. The hash
    becomes the cache key for "is the extracted tree current?".
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(_HASH_BLOCK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _is_extraction_current(tarball_hash: str) -> bool:
    """Return True iff the extracted tree matches the tarball's hash.

    Cheap-path check: read the sidecar file and compare. Missing
    sidecar, missing tree, or hash mismatch → False (re-extract).
    """
    if not _INSTALL_DIR.is_dir():
        return False
    try:
        return _HASH_SIDECAR.read_text().strip() == tarball_hash
    except OSError:
        return False


def _extract_tarball(tarball_hash: str) -> None:
    """Wipe ``_INSTALL_DIR`` and re-extract the tarball into it.

    Caller MUST hold the file lock. ``shutil.rmtree`` first because
    ``tarfile.extractall`` doesn't remove files that no longer exist in
    the new tarball — leftover files from a previous version would
    silently survive a force-reinstall.

    Uses ``filter='data'`` (Python 3.12+) for path-traversal safety. On
    older Pythons (3.11 and below) we fall back to plain ``extractall``;
    the tarball ships inside our own wheel so the security exposure is
    bounded by what we put in it ourselves.
    """
    print(
        f"[geniesim_ros] extracting {_TARBALL.name} → {_INSTALL_DIR} ...",
        file=sys.stderr,
    )
    if _INSTALL_DIR.exists():
        shutil.rmtree(_INSTALL_DIR, ignore_errors=True)
    _PKG_DIR.mkdir(parents=True, exist_ok=True)

    with tarfile.open(_TARBALL, "r:gz") as tar:
        # ``filter='data'`` is the safe-extraction filter, available from
        # Python 3.12. On 3.11 it's a deprecation warning; on older
        # interpreters the kwarg doesn't exist at all.
        if sys.version_info >= (3, 12):
            tar.extractall(path=_PKG_DIR, filter="data")
        else:
            tar.extractall(path=_PKG_DIR)  # noqa: S202 — see docstring

    # Stamp the hash AFTER extraction so a half-extracted tree never
    # claims to be current.
    _HASH_SIDECAR.parent.mkdir(parents=True, exist_ok=True)
    _HASH_SIDECAR.write_text(tarball_hash)
    print(f"[geniesim_ros] extraction complete", file=sys.stderr)


def _locked_extract(tarball_hash: str) -> None:
    """Acquire the file lock, re-check, extract if still needed.

    The double-check inside the lock matters: thread A may have already
    extracted while thread B was waiting. B should NOT re-extract on
    top.

    On filesystems without ``flock`` support (rare; some NFS/overlay
    configs), we silently fall through to unlocked extraction. Worst
    case: two processes extract concurrently and one's writes get
    overwritten by the other — the next import's hash check re-extracts
    if anything is inconsistent.
    """
    try:
        import fcntl
    except ImportError:
        # Non-POSIX — Windows, etc. Fall back to unlocked.
        if not _is_extraction_current(tarball_hash):
            _extract_tarball(tarball_hash)
        return

    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK_FILE.open("w") as lock_fh:
        try:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
        except OSError:
            # Filesystem doesn't support flock — proceed unlocked.
            if not _is_extraction_current(tarball_hash):
                _extract_tarball(tarball_hash)
            return
        try:
            if not _is_extraction_current(tarball_hash):
                _extract_tarball(tarball_hash)
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def ensure_ros_install() -> Optional[Path]:
    """Ensure the colcon install tree is on disk; return its path.

    Returns ``None`` when no tarball is bundled AND no pre-staged
    ``_ros_install/`` directory exists — the editable / source-checkout
    case where the dev hasn't run ``geniesim ros build dev`` yet. Callers
    that need to source ``setup.bash`` should treat ``None`` as "user
    must build the colcon workspace first" and surface a clear error.

    Hot-path cost: one file ``stat`` (sidecar read) plus a string
    compare when the tree is current. Tar extraction (a few seconds for
    50+ MB) only fires on the first import after install or after the
    wheel changes.
    """
    # Wheel install path: tarball is bundled.
    if _TARBALL.is_file():
        tarball_hash = _hash_file(_TARBALL)
        if not _is_extraction_current(tarball_hash):
            _locked_extract(tarball_hash)
        return _INSTALL_DIR

    # Editable / source checkout path: no tarball; use whatever the dev
    # built locally.
    if _INSTALL_DIR.is_dir():
        return _INSTALL_DIR

    return None


def install_root() -> Optional[Path]:
    """Public discovery helper: just call :func:`ensure_ros_install`.

    Exposed under a more obvious name for external callers (e.g.
    ``geniesim ros prepare`` if added later, or one-liner shell
    discovery: ``python3 -c 'from geniesim_ros import install_root;
    print(install_root())'``).
    """
    return ensure_ros_install()


def setup_bash_path() -> Optional[Path]:
    """Return the path to ``setup.bash`` in the install tree, or None.

    Convenience for the common shell workflow::

        source $(python3 -c 'from geniesim_ros import setup_bash_path;
                             print(setup_bash_path() or "")')

    Returns ``None`` if the install tree isn't available; callers should
    handle that as "user must build / install the workspace first".
    """
    root = ensure_ros_install()
    if root is None:
        return None
    candidate = root / "setup.bash"
    return candidate if candidate.is_file() else None
