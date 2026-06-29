# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim ros {build,doctor}`` — drive the ROS 2 colcon workspace.

Three sub-verbs live here:

* ``geniesim ros build dev``     — iterative dev build. Outputs land in
  ``$PWD/devel``, ``$PWD/devel_build``, ``$PWD/devel_log``.
  Uses ``--symlink-install`` and ``RelWithDebInfo``.
* ``geniesim ros build release`` — release build. Outputs land in
  ``$PWD/install``, ``$PWD/build``, ``$PWD/log``.
  Uses ``Release`` build type, no symlink-install.
* ``geniesim ros build cleanup`` — remove all dev + release outputs from
  ``$PWD``, with a single confirmation prompt.

Both profiles write to ``$PWD``. The colcon source workspace is resolved
by ``ros_workspace_root()``: ``$GENIESIM_WORKSPACE`` > CWD (if it looks
like a colcon workspace) > bundled ``geniesim_ros`` workspace.
"""

from __future__ import annotations

import sys

from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW
from geniesim_cli._workspace import ros_workspace_root

# Common colcon args shared by both profiles. Per-profile cmake args are
# appended on top.
_COMMON_BUILD_ARGS: tuple[str, ...] = (
    "--merge-install",
    "--parallel-workers",
    "8",
)

_DEV_PROFILE: dict = {
    "label": "Development (RelWithDebInfo + symlink-install)",
    "extra_args": (
        "--symlink-install",
        "--cmake-args",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ),
    # Dev outputs live in $PWD with conventional dev-tree names.
    "install_dir_name": "devel",
    "build_dir_name": "devel_build",
    "log_dir_name": "devel_log",
}

_RELEASE_PROFILE: dict = {
    "label": "Release",
    "extra_args": (
        "--cmake-args",
        "-DCMAKE_BUILD_TYPE=Release",
    ),
    # Release outputs land in $PWD — same convention as dev, just without
    # --symlink-install and with Release build type.
    "install_dir_name": "install",
    "build_dir_name": "build",
    "log_dir_name": "log",
}


# ---------------------------------------------------------------------------
# /opt/ros sourcing
# ---------------------------------------------------------------------------


def _ros_already_sourced() -> bool:
    """Cheap signal that an ament/ROS overlay is already on the environment.

    We treat ``$AMENT_PREFIX_PATH`` containing ``/opt/ros/`` as proof:
    sourcing ``/opt/ros/<distro>/setup.bash`` always populates that var
    with at least the base prefix. ``$ROS_DISTRO`` alone is *not* enough
    — users sometimes export it manually without sourcing the setup.
    """
    import os

    ament = os.environ.get("AMENT_PREFIX_PATH", "")
    return any(part.startswith("/opt/ros/") for part in ament.split(":") if part)


def _discover_opt_ros_distro() -> "str | None":
    """Return a distro name found under ``/opt/ros/`` whose ``setup.bash`` exists.

    Preference order: ``$ROS_DISTRO`` env var (if its setup.bash exists),
    otherwise the lexicographically-greatest directory under
    ``/opt/ros/`` that has a ``setup.bash`` (so ``humble`` < ``jazzy``
    picks ``jazzy``). Returns ``None`` when nothing usable is found.
    """
    import os
    from pathlib import Path

    opt_ros = Path("/opt/ros")
    if not opt_ros.is_dir():
        return None

    env_distro = os.environ.get("ROS_DISTRO")
    if env_distro and (opt_ros / env_distro / "setup.bash").is_file():
        return env_distro

    candidates = sorted(
        (p.name for p in opt_ros.iterdir() if (p / "setup.bash").is_file()),
        reverse=True,
    )
    return candidates[0] if candidates else None


def _ensure_ros_sourced() -> None:
    """Make sure ``/opt/ros/<distro>/setup.bash`` is sourced for this process.

    Why a function and not "tell the user to source it themselves":
    ``geniesim ros`` is a Python CLI launched from arbitrary shells and
    contexts (IDE terminals, CI, ``systemd`` units). Forgetting to
    source ROS is the #1 footgun — colcon then either errors out with
    cryptic "ament_cmake not found" messages or, worse, silently builds
    against a stale overlay. We make the contract explicit here.

    Strategy:

    1. Fast-exit if ``_ros_already_sourced()`` says we're good.
    2. Otherwise locate a ``/opt/ros/<distro>/setup.bash`` (env preference
       first, then auto-discovery).
    3. Spawn ``bash -c 'set -e; source <setup>; env -0'`` — ``env -0``
       prints the post-source environment as NUL-separated ``KEY=VAL``
       records, which we parse without any shell-quoting hazards.
    4. Overlay the parsed dict onto ``os.environ`` so every subsequent
       ``subprocess.run`` inherits the ROS overlay.

    On failure we print a colored error and ``sys.exit(1)``; partial
    sourcing would leave colcon in a worse state than not sourcing at all.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path

    if _ros_already_sourced():
        return

    distro = _discover_opt_ros_distro()
    if distro is None:
        print(f"{RED}❌ ROS is not sourced and no ROS install was found under {BOLD}/opt/ros{RST}{RED}.{RST}")
        print(f"   {DIM}Install ROS 2 (e.g. {BOLD}sudo apt install ros-jazzy-desktop{RST}{DIM}) and retry.{RST}")
        sys.exit(1)

    setup_bash = Path("/opt/ros") / distro / "setup.bash"

    if shutil.which("bash") is None:
        print(f"{RED}❌ Cannot source {BOLD}{setup_bash}{RST}{RED}: bash is not on PATH.{RST}")
        sys.exit(1)

    print(f"{DIM}🔌 Sourcing {BOLD}{setup_bash}{RST}{DIM} ...{RST}")

    # ``env -0`` is GNU-specific but ships in coreutils on all supported
    # Ubuntu/Debian targets. We avoid the more portable ``env`` (newline
    # separated) because env vars can legitimately contain newlines.
    proc = subprocess.run(
        ["bash", "-c", f"set -e; source {setup_bash}; env -0"],
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"{RED}❌ Failed to source {BOLD}{setup_bash}{RST}{RED} (exit {proc.returncode}){RST}")
        if proc.stderr:
            sys.stderr.write(proc.stderr.decode(errors="replace"))
        sys.exit(proc.returncode)

    new_env: dict[str, str] = {}
    for record in proc.stdout.split(b"\x00"):
        if not record:
            continue
        try:
            key, _, value = record.decode("utf-8", errors="replace").partition("=")
        except Exception:
            continue
        if key:
            new_env[key] = value

    # Overlay every var the ROS setup script set or modified. We don't
    # try to be selective (PATH/PYTHONPATH/AMENT_*/CMAKE_PREFIX_PATH/...)
    # because the setup script is the source of truth for which vars
    # matter; copying the whole post-source dict guarantees parity with
    # what a user would get in an interactive shell.
    for k, v in new_env.items():
        os.environ[k] = v


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _dev_output_dirs():
    """Return the ``$PWD``-anchored (install, build, log) for the dev profile."""
    from pathlib import Path

    cwd = Path.cwd().resolve()
    return (
        cwd / _DEV_PROFILE["install_dir_name"],
        cwd / _DEV_PROFILE["build_dir_name"],
        cwd / _DEV_PROFILE["log_dir_name"],
    )


def _release_output_dirs():
    """Return the ``$PWD``-anchored (install, build, log) for the release profile."""
    from pathlib import Path

    cwd = Path.cwd().resolve()
    return (
        cwd / _RELEASE_PROFILE["install_dir_name"],
        cwd / _RELEASE_PROFILE["build_dir_name"],
        cwd / _RELEASE_PROFILE["log_dir_name"],
    )


# ---------------------------------------------------------------------------
# build dev / build release
# ---------------------------------------------------------------------------


def _run_colcon_build(profile_name: str, repo_root, install_dir, build_dir, log_dir, label: str, extra_args) -> None:
    import subprocess
    import time

    print(f"{BOLD}{MAGENTA}⚙️  geniesim ros build {profile_name}{RST}")
    print(f"   {DIM}Profile:{RST}     {CYAN}{label}{RST}")
    print(f"   {DIM}Workspace:{RST}   {CYAN}{repo_root}{RST}")
    print(f"   {DIM}Install dir:{RST} {CYAN}{install_dir}{RST}")
    print(f"   {DIM}Build dir:{RST}   {CYAN}{build_dir}{RST}")
    print(f"   {DIM}Log dir:{RST}     {CYAN}{log_dir}{RST}")
    print()

    cmd = [
        "colcon",
        "--log-base",
        str(log_dir),
        "build",
        "--build-base",
        str(build_dir),
        "--install-base",
        str(install_dir),
        *_COMMON_BUILD_ARGS,
        *extra_args,
    ]

    print(f"   {YELLOW}🔨 {' '.join(cmd)}{RST}")
    print()

    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(repo_root))
    elapsed = time.perf_counter() - t0

    print()
    if result.returncode != 0:
        print(f"   {RED}❌ Build failed (exit code {result.returncode}){RST}")
        sys.exit(result.returncode)

    print(f"   {GREEN}✅ Build succeeded in {elapsed:.1f}s{RST}")
    print(f"   {DIM}Source workspace (use the file for your shell):{RST}")
    print(f"   {DIM}  bash:{RST}  source {install_dir}/setup.bash")
    print(f"   {DIM}  zsh:{RST}   source {install_dir}/setup.zsh")
    print()
    print(f"   {BOLD}🎉 Done!{RST}")


def _cwd_inside_colcon_source_tree() -> "Path | None":
    """Return the offending workspace root if ``$PWD`` is *inside* a colcon
    source tree (i.e. a subdirectory of a workspace, not the root itself).

    Being the workspace root is fine — that is the traditional in-tree flow.
    Being *inside* the source tree (e.g. ``src/some_pkg``) is the footgun
    we guard against, because dev outputs would land next to package.xml files.
    """
    from pathlib import Path

    cwd = Path.cwd().resolve()
    # Start from parent — CWD itself being a workspace root is allowed.
    for candidate in cwd.parents:
        src = candidate / "src"
        if not src.is_dir():
            continue
        try:
            for child in src.iterdir():
                if (child / "package.xml").is_file():
                    return candidate
        except OSError:
            continue
    return None


def _build_dev() -> None:
    from pathlib import Path

    offender = _cwd_inside_colcon_source_tree()
    if offender is not None:
        print(f"{RED}❌ Refusing to run {BOLD}geniesim ros build dev{RST}{RED} from inside a colcon workspace.{RST}")
        print(f"   {DIM}Detected workspace root:{RST} {BOLD}{offender}{RST}")
        print(f"   {DIM}Current directory:{RST}       {BOLD}{Path.cwd()}{RST}")
        print()
        print(f"   {YELLOW}Dev outputs are written to {BOLD}$PWD/devel*{RST}{YELLOW};{RST}")
        print(f"   {YELLOW}running from here would pollute the workspace source tree.{RST}")
        print()
        print(f"   {DIM}Fix:{RST} {CYAN}cd{RST} to a scratch directory (e.g. {BOLD}~/ws{RST}{DIM}) and re-run.{RST}")
        sys.exit(1)

    repo_root = Path(ros_workspace_root())
    install_dir, build_dir, log_dir = _dev_output_dirs()
    _run_colcon_build(
        "dev",
        repo_root,
        install_dir,
        build_dir,
        log_dir,
        _DEV_PROFILE["label"],
        _DEV_PROFILE["extra_args"],
    )


def _build_release() -> None:
    from pathlib import Path

    offender = _cwd_inside_colcon_source_tree()
    if offender is not None:
        print(
            f"{RED}❌ Refusing to run {BOLD}geniesim ros build release{RST}{RED} from inside a colcon workspace.{RST}"
        )
        print(f"   {DIM}Detected workspace root:{RST} {BOLD}{offender}{RST}")
        print(f"   {DIM}Current directory:{RST}       {BOLD}{Path.cwd()}{RST}")
        print()
        print(
            f"   {YELLOW}Release outputs are written to {BOLD}$PWD/install{RST}{YELLOW}, {BOLD}$PWD/build{RST}{YELLOW}, {BOLD}$PWD/log{RST}{YELLOW};{RST}"
        )
        print(f"   {YELLOW}running from here would pollute the workspace source tree.{RST}")
        print()
        print(f"   {DIM}Fix:{RST} {CYAN}cd{RST} to the workspace root or a scratch directory and re-run.{RST}")
        sys.exit(1)

    repo_root = Path(ros_workspace_root())
    install_dir, build_dir, log_dir = _release_output_dirs()
    _run_colcon_build(
        "release",
        repo_root,
        install_dir,
        build_dir,
        log_dir,
        _RELEASE_PROFILE["label"],
        _RELEASE_PROFILE["extra_args"],
    )


def _build_help_and_exit() -> None:
    print(f"{BOLD}🔨 geniesim ros build{RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim ros build {CYAN}<profile>|cleanup{RST}")
    print()
    print(f"{BOLD}Profiles:{RST}")
    print(f"  {WHITE}dev{RST}      {DIM}— {_DEV_PROFILE['label']}; outputs under {BOLD}$PWD{RST}")
    print(f"  {WHITE}release{RST}  {DIM}— {_RELEASE_PROFILE['label']}{RST}")
    print()
    print(f"{BOLD}Other:{RST}")
    print(f"  {WHITE}cleanup{RST}  {DIM}— remove all dev+release outputs (with confirmation){RST}")
    sys.exit(0)


# ---------------------------------------------------------------------------
# build cleanup
# ---------------------------------------------------------------------------


def _cleanup_preview_lines(path, max_entries: int = 20) -> list[str]:
    from pathlib import Path

    path = Path(path)
    lines: list[str] = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        lines.append(f"  {DIM}(cannot read: {exc}){RST}")
        return lines
    if not entries:
        lines.append(f"  {DIM}(empty directory){RST}")
        return lines
    for entry in entries[:max_entries]:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"  {WHITE}{entry.name}{suffix}{RST}")
    if len(entries) > max_entries:
        lines.append(f"  {DIM}... and {len(entries) - max_entries} more top-level entries{RST}")
    return lines


def _build_cleanup() -> None:
    """Remove every dev + release output from the current directory.

    Two buckets are scanned:

    1. ``$PWD/devel{,_build,_log}``  — dev outputs.
    2. ``$PWD/{install,build,log}``  — release outputs.

    The function gathers every existing path, shows a single preview,
    prompts once, and deletes everything in one pass.
    """
    import shutil
    from pathlib import Path

    candidates: list[tuple[str, Path]] = []

    for p in _dev_output_dirs():
        candidates.append(("dev", p))
    for p in _release_output_dirs():
        candidates.append(("release", p))

    existing = [(label, p) for (label, p) in candidates if p.exists()]

    print(f"{BOLD}{MAGENTA}🧹 geniesim ros build cleanup{RST}")
    print()

    if not existing:
        print(f"   {GREEN}Nothing to remove.{RST}")
        print(f"   {DIM}Checked:{RST}")
        for label, p in candidates:
            print(f"      {DIM}{label}:{RST} {WHITE}{p}{RST}")
        print()
        return

    print(f"   {YELLOW}The following paths will be removed permanently:{RST}")
    print()
    for label, p in existing:
        print(f"   {DIM}[{label}]{RST} {RED}{p}{RST}")
        for line in _cleanup_preview_lines(p):
            print(line)
        print()

    try:
        answer = input(f"   {BOLD}Type {GREEN}y{RST}{BOLD} to delete the paths above, anything else to cancel: {RST}")
    except EOFError:
        print()
        print(f"   {RED}❌ No input (non-interactive); aborting.{RST}")
        sys.exit(1)

    if answer.strip().lower() != "y":
        print()
        print(f"   {DIM}Cancelled.{RST}")
        return

    for _label, p in existing:
        print(f"   {YELLOW}Removing {p} ...{RST}")
        shutil.rmtree(p, ignore_errors=False)

    print()
    print(f"   {GREEN}✅ Cleanup finished.{RST}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(args: list[str]) -> None:
    """Dispatcher for ``geniesim ros <subcommand>``.

    ``args`` is the slice *after* the leading ``ros`` token, i.e.
    ``["build", "dev"]`` for ``geniesim ros build dev``.
    """
    if not args:
        print(f"{BOLD}{MAGENTA}⚙️  geniesim ros{RST}")
        print()
        print(f"{BOLD}Usage:{RST} geniesim ros {CYAN}<subcommand>{RST} [args...]")
        print()
        print(f"{BOLD}Subcommands:{RST}")
        print(f"  {CYAN}build{RST} dev|release|cleanup  🔨 colcon build, or 🧹 clean output dirs")
        print(f"  {CYAN}doctor{RST}                   🩺 check & fix rosdep dependencies")
        sys.exit(0)

    # Every ``geniesim ros`` subcommand shells out to colcon, which
    # requires the ROS overlay (PATH, AMENT_PREFIX_PATH, CMAKE_PREFIX_PATH,
    # PYTHONPATH, ...). Doing this once at dispatch time means individual
    # subcommands never need to worry about it.
    _ensure_ros_sourced()

    sub = args[0]

    if sub == "build":
        if len(args) < 2:
            _build_help_and_exit()
        target = args[1]
        if target == "cleanup":
            _build_cleanup()
            return
        if target == "dev":
            _build_dev()
            return
        if target == "release":
            _build_release()
            return
        print(f"{RED}❌ Error: unknown build profile '{target}'{RST}")
        print()
        _build_help_and_exit()
        return

    if sub == "graph":
        print(
            f"{YELLOW}⚠️  `geniesim ros graph` has been replaced by {CYAN}geniesim tool ros-dag{RST}{YELLOW}.{RST}\n"
            f"   {DIM}The ROS package DAG now lives in {CYAN}source/geniesim_ros/README.md{RST}{DIM} as a CI-checkable Mermaid block.{RST}\n"
            f"   {DIM}Run {CYAN}geniesim tool ros-dag --fix{RST}{DIM} to regenerate it.{RST}\n"
        )
        sys.exit(1)

    if sub == "doctor":
        from geniesim_cli.commands.doctor import _check_rosdep

        status, lines, fix_label, fix_fn = _check_rosdep()
        for ln in lines:
            print(ln)
        if fix_fn is not None and status == "fail":
            try:
                answer = input(f"   {BOLD}Apply fix? [Y/n]: {RST}").strip().lower()
            except EOFError:
                print()
                answer = "y"
            if not answer or answer in ("y", "yes"):
                fix_fn()
        return

    print(f"{RED}❌ Error: unknown ros subcommand '{sub}'{RST}")
    sys.exit(1)
