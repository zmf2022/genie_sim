# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Custom setuptools entry point for ``geniesim_ros``.

The package ships a ROS 2 colcon workspace under ``src/ros_ws``. Its
install behavior depends on **how pip invokes the build backend**:

* ``pip install -e <path>``  →  PEP 660 editable install. setuptools
  routes this through :class:`setuptools.command.develop.develop` (or
  the new ``editable_wheel`` command on modern pip). We want this path
  to be **fast and side-effect-free** — devs are expected to drive
  colcon themselves with ``geniesim ros build dev`` from any cwd. The
  colcon workspace must NOT be built during editable install.
  ``ensure_ros_install()`` is a no-op in this mode (no bundled tarball
  next to the package).

* ``pip install <path>``     →  wheel build (``bdist_wheel`` /
  ``build_wheel``). This is the production path; the resulting wheel
  must contain a usable colcon ``install/`` tree so that consumers who
  ``pip install geniesim_ros`` get a ready-to-source ROS 2 workspace
  without ever needing colcon themselves. We hook ``bdist_wheel.run``
  to (a) invoke ``colcon build`` into ``src/geniesim_ros/_ros_install/``
  and (b) tar that tree into ``src/geniesim_ros/_ros_install.tar.gz``.
  Only the **tarball** is shipped in the wheel — extraction happens
  lazily on first ``import geniesim_ros``.

Why a tarball, not raw ``package_data``
----------------------------------------
Setuptools' ``package_data`` glob copies files through
``distutils.file_util.copy_file`` which strips file modes, and
``bdist_wheel`` then writes every entry with the canonical wheel mode
(``0o100664``). The result: ROS 2 entrypoints under
``lib/<pkg>/<script>.py`` lose their executable bit, and ``ros2 run
<pkg> <script>`` fails because it filters by ``os.access(.., X_OK)``.

A tarball preserves modes natively (``tarfile`` round-trips
``st_mode``), so when we extract it on first import the original
``0o755`` bits land back on disk. The wheel itself is also smaller and
faster to build (one tar + one zip vs. thousands of zip entries with
per-file SHA-256 records).

The split is implemented as a single ``BdistWheelWithColcon`` subclass
that runs the colcon stage, tars the staged tree, removes the original
directory, and then delegates to the upstream ``bdist_wheel.run``. We
do not override ``develop`` / ``build_py`` / ``editable_wheel`` — they
retain their stock behavior, which is exactly what the editable path
needs.

Override knobs:

* ``GENIESIM_ROS_SKIP_BUILD=1``  — skip the colcon stage even on wheel
  builds. Useful for CI smoke tests that only validate metadata, and
  for the ``geniesim deploy geniesim_ros`` flow when the artifacts
  have already been staged by ``geniesim ros build release``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from setuptools import find_packages, setup

try:
    # setuptools >= 70 ships its own bdist_wheel; older releases delegate
    # to the ``wheel`` distribution. Both expose the same ``run()`` method
    # we want to wrap, so import-with-fallback keeps the override portable.
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # pragma: no cover - older setuptools
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


HERE = Path(__file__).resolve().parent
ROS_WS = HERE / "src" / "ros_ws"
STAGED_INSTALL = HERE / "src" / "geniesim_ros" / "_ros_install"
STAGED_TARBALL = HERE / "src" / "geniesim_ros" / "_ros_install.tar.gz"
# Deploy-only build/log dirs, namespaced so they cannot collide with the
# interactive ``geniesim ros build dev|release`` trees (which use
# ``devel_build/devel_log`` and ``build/log`` respectively at the
# *workspace* root, not the dist root). Keeping the deploy path on its
# own pair of directories means:
#   * ``geniesim ros build dev`` keeps its own incremental cache.
#   * ``geniesim deploy geniesim_ros`` cannot poison or be poisoned by
#     that cache (different dir => different ``CMakeCache.txt``).
#   * Cleanup is unambiguous: blowing away ``.colcon_deploy_*`` resets
#     only the deploy artifacts.
STAGED_BUILD = HERE / ".colcon_deploy_build"
STAGED_LOG = HERE / ".colcon_deploy_log"


def _stage_colcon_into_package() -> None:
    """Run ``colcon build`` and stage outputs under ``src/geniesim_ros/_ros_install``.

    Mirrors the ``release`` profile of ``geniesim ros build release``:
    Release CMake type, merged install layout, install dir colocated
    with the importable package so setuptools' ``package_data`` glob
    can pick it up, build/log siblings under the dist root.

    Raises a clean ``SystemExit`` on failure rather than a Python
    traceback, so the pip install error message stays readable.
    """
    if os.environ.get("GENIESIM_ROS_SKIP_BUILD") == "1":
        # NOTE: env-var name is also registered in geniesim_cli._env
        # (GENIESIM_ROS_SKIP_BUILD). We read it via os.environ here
        # because this setup.py runs before geniesim_cli is importable
        # in the build env.
        print("[geniesim_ros setup] GENIESIM_ROS_SKIP_BUILD=1 → skipping colcon stage")
        return

    if not (ROS_WS / "src").is_dir():
        print(f"[geniesim_ros setup] {ROS_WS}/src missing; nothing to build")
        return

    if shutil.which("colcon") is None:
        print(
            "[geniesim_ros setup] ERROR: colcon is not on PATH.\n"
            "    The wheel build needs colcon to stage the ROS 2 workspace.\n"
            "    Install ROS 2 + colcon, or set GENIESIM_ROS_SKIP_BUILD=1 to bypass.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Wipe stale CMake caches before colcon. The cache is keyed by the
    # interpreter path (e.g. ``Python3_EXECUTABLE=...``), so a previous
    # *isolated* PEP 517 run leaves behind references to ephemeral
    # ``/tmp/build-env-*`` paths that no longer exist. CMake happily
    # re-reads them, then fails with "Cannot run the interpreter".
    # Re-staging from an empty build/log dir guarantees CMake re-resolves
    # interpreters against the *current* PATH (which, under
    # ``--no-isolation``, is the ambient ROS-sourced env we want).
    for stale in (STAGED_BUILD, STAGED_LOG):
        if stale.exists():
            if os.environ.get("GENIESIM_ROS_KEEP_BUILD_CACHE") == "1":
                print(f"[geniesim_ros setup] GENIESIM_ROS_KEEP_BUILD_CACHE=1 → reusing {stale}")
            else:
                print(f"[geniesim_ros setup] wiping stale {stale}")
                shutil.rmtree(stale, ignore_errors=True)

    STAGED_INSTALL.mkdir(parents=True, exist_ok=True)

    cmd = [
        "colcon",
        "--log-base",
        str(STAGED_LOG),
        "build",
        "--build-base",
        str(STAGED_BUILD),
        "--install-base",
        str(STAGED_INSTALL),
        "--merge-install",
        "--parallel-workers",
        "8",
        "--cmake-args",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -flto=auto",
        "-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG -flto=auto",
        "--no-warn-unused-cli",
    ]
    print(f"[geniesim_ros setup] $ {' '.join(cmd)}")
    rc = subprocess.run(cmd, cwd=str(ROS_WS)).returncode
    if rc != 0:
        print(
            f"[geniesim_ros setup] ERROR: colcon build failed (exit {rc}).\n" f"    Logs: {STAGED_LOG}",
            file=sys.stderr,
        )
        sys.exit(rc)

    print(f"[geniesim_ros setup] staged ROS install at {STAGED_INSTALL}")


def _prune_pycache(root: Path) -> None:
    """Remove all ``__pycache__`` directories under ``root``.

    setuptools treats any directory as an importable package and warns
    (or silently drops) ``__pycache__`` dirs it finds inside the staged
    install tree. Pruning them before the tarball is assembled keeps the
    archive clean.
    """
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache, ignore_errors=True)


def _tar_staged_install() -> None:
    """Tar ``STAGED_INSTALL/`` → ``STAGED_TARBALL`` and remove the dir.

    Tarfile preserves ``st_mode`` natively, so colcon's ``0o755`` survives
    the round-trip — that's the whole point of using a tar instead of
    setuptools ``package_data`` (which strips modes via ``copy_file`` and
    ``bdist_wheel``'s canonical ``0o100664``).

    The original directory is **removed** after tarring so setuptools
    doesn't also bundle it as raw ``package_data``. Only the tarball
    ships in the wheel; ``geniesim_ros._bootstrap.ensure_ros_install``
    extracts it lazily on first import.

    Uses ``tarfile`` with deterministic ``mtime=0`` and sorted entries so
    repeated builds against the same tree produce byte-identical
    tarballs (helps reproducible-build pipelines and wheel content
    hashing).
    """
    if not STAGED_INSTALL.is_dir():
        # No staged tree — either GENIESIM_ROS_SKIP_BUILD=1 or
        # _ros_install was never produced. Nothing to tar; the wheel
        # ships without a tarball and import-time extraction silently
        # no-ops (matches the editable / dev path).
        if STAGED_TARBALL.exists():
            STAGED_TARBALL.unlink()
        return

    print(f"[geniesim_ros setup] tarring {STAGED_INSTALL} → {STAGED_TARBALL}")
    if STAGED_TARBALL.exists():
        STAGED_TARBALL.unlink()

    # Sort entries for reproducibility; reset uid/gid/mtime so the
    # archive doesn't depend on the build host.
    def _sanitize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        return info

    entries = sorted(STAGED_INSTALL.rglob("*"))
    with tarfile.open(STAGED_TARBALL, "w:gz", compresslevel=6) as tar:
        for path in entries:
            arcname = path.relative_to(STAGED_INSTALL.parent).as_posix()
            tar.add(path, arcname=arcname, recursive=False, filter=_sanitize)

    print(f"[geniesim_ros setup] tarball size: {STAGED_TARBALL.stat().st_size:,} bytes")
    # Remove the staged directory so setuptools doesn't double-ship it.
    shutil.rmtree(STAGED_INSTALL, ignore_errors=True)


class BdistWheelWithColcon(_bdist_wheel):
    """Wheel-build override that stages the colcon workspace, then tars it.

    Only triggered by ``pip install <path>`` / ``python -m build``;
    editable installs (``pip install -e``) go through ``develop`` /
    ``editable_wheel`` and never reach this override. The wheel produced
    here ships ``_ros_install.tar.gz`` only — no raw ``_ros_install/``
    files in ``package_data`` — so file modes survive the install and
    ``ros2 run`` finds executable scripts.
    """

    def finalize_options(self):
        # The native ROS binaries (genie_sim_render_node etc.) ride inside
        # _ros_install.tar.gz as package_data, so setuptools sees no ext module
        # and would tag the wheel `py3-none-any` — a portability lie. The baked
        # binaries are interpreter- and distro-specific (jazzy/Noble links
        # libtinyxml2.so.10, humble/Jammy links .so.9). Marking the distribution
        # as having ext modules makes bdist_wheel (a) stamp a real
        # cp3XX-cp3XX-linux_x86_64 tag (distinct per distro -> no bucket clobber,
        # pip won't cross-install) and (b) classify the package as platlib so it
        # stays at the archive ROOT. NB: setting `root_is_pure = False` alone is
        # NOT enough — with no ext module the package is still purelib, so it
        # gets relocated under `<name>.data/purelib/`, which breaks consumers
        # (and _bootstrap) that read `geniesim_ros/_ros_install.tar.gz` from the
        # archive root. Must run before super() computes root_is_pure + the tag.
        self.distribution.has_ext_modules = lambda: True
        super().finalize_options()

    def run(self):
        _stage_colcon_into_package()
        if STAGED_INSTALL.exists():
            _prune_pycache(STAGED_INSTALL)
        _tar_staged_install()
        super().run()


setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Ship only the tarball. ``geniesim_ros._bootstrap.ensure_ros_install``
    # extracts it on first import — extraction preserves file modes (the
    # whole reason for the tar), so ``lib/<pkg>/<script>.py`` keeps the
    # ``0o755`` colcon gave it.
    package_data={
        "geniesim_ros": [
            "_ros_install.tar.gz",
        ],
    },
    include_package_data=True,
    cmdclass={"bdist_wheel": BdistWheelWithColcon},
)
