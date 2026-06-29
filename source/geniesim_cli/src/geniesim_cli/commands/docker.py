# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Manage GenieSim Docker containers across Isaac Sim variants.

Dispatched by four CLI commands:

* ``geniesim docker``    — default alias for ``docker5.1``.
* ``geniesim docker6.0`` — Isaac Sim 6.0 variant (geniesim4) — **incoming, not implemented yet**.
* ``geniesim docker5.1`` — Isaac Sim 5.1 variant (geniesim3).
* ``geniesim docker4.5`` — Isaac Sim 4.5 variant (geniesim2 E.O.L.).

The verbs map onto the shell scripts that live next to the Dockerfile in
``<repo_root>/docker/`` (resolved at runtime via :func:`_docker_dir`).
We intentionally keep the heavy lifting in shell — the Python layer is
just a typed dispatcher with consistent styling and env-var injection
(notably ``HOST_UID`` / ``HOST_GID``, used by ``entrypoint.sh`` to remap
the canned ``isaac-sim`` user inside the NVIDIA Isaac Sim base image).

Subcommands:

* ``build``   — ``docker build`` with the variant's Dockerfile from the repo root.
* ``up``      — start the container (GUI by default; ``--headless`` to skip X11).
* ``down``    — stop and remove the container.
* ``into``    — ``docker exec -it <container> bash``.
* ``logs``    — ``docker logs [-f] <container>``.

Container name and image tag are variant-specific (see ``_VARIANTS``). Both can
be overridden via env vars ``GENIESIM_CONTAINER`` / ``GENIESIM_IMAGE`` for
parity with the shell scripts the verbs delegate to.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from geniesim_cli import _env
from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

# DaoCloud's nvcr.io mirror — prefix-replacement form. The image path
# becomes nvcr.m.daocloud.io/nvidia/isaac-sim:<tag>, so the registry
# build-arg in Dockerfile is "nvcr.m.daocloud.io/nvidia".
#
# Tsinghua TUNA serves the apt + pip mirrors. These three together
# are what `--china` activates.
_CHINA_NVCR_REGISTRY = "nvcr.m.daocloud.io/nvidia"
_CHINA_APT_MIRROR = "https://mirrors.tuna.tsinghua.edu.cn/ubuntu"
_CHINA_PIP_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"

_VARIANTS: dict[str, dict] = {
    "": {
        "dir": "docker",
        "dockerfile": "Dockerfile",
        "image": "registry.agibot.com/genie-sim/geniesim4:latest",
        "container": "geniesim4",
        # geniesim4 / Isaac Sim 6.0 is the incoming variant: Dockerfile is
        # a placeholder, image is not published, runtime is unverified.
        # Marked here so ``run()`` can refuse to dispatch it.
        "implemented": False,
        "env": {
            "ROS_DISTRO": "jazzy",
            "GENIESIM_PY_CMD": "python3",
            "GENIESIM_BREAK_SYSTEM_PKGS": "1",
            "GENIESIM_CHOWN_KIT_PATH": "/usr/local/lib/python3.12/dist-packages/isaacsim/kit",
            "GENIESIM_EXTRA_CACHE_DIRS": "kit pip numba",
            "GENIESIM_ISAACSIM_KIT_CACHE_PATH": "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache",
            "GENIESIM_OVRTX_CACHE_PATH": "/usr/local/lib/python3.12/dist-packages/ovrtx/bin/cache",
        },
    },
    "4.5": {
        "dir": "docker",
        "dockerfile": "Dockerfile.4.5",
        "image": "registry.agibot.com/genie-sim/geniesim2:latest",
        "container": "geniesim2",
        "env": {
            "ROS_DISTRO": "humble",
            "GENIESIM_PY_CMD": "python3",
            "GENIESIM_BREAK_SYSTEM_PKGS": "1",
            "GENIESIM_CHOWN_KIT_PATH": "/usr/local/lib/python3.10/dist-packages/isaacsim/kit",
            "GENIESIM_EXTRA_CACHE_DIRS": "",
            "GENIESIM_ISAACSIM_KIT_CACHE_PATH": "/usr/local/lib/python3.10/dist-packages/isaacsim/kit/cache",
            "GENIESIM_OVRTX_CACHE_PATH": "/usr/local/lib/python3.10/dist-packages/ovrtx/bin/cache",
        },
    },
    "5.1": {
        "dir": "docker",
        "dockerfile": "Dockerfile.5.1",
        "image": "registry.agibot.com/genie-sim/geniesim3:latest",
        "container": "geniesim3",
        "env": {
            "ROS_DISTRO": "jazzy",
            "GENIESIM_PY_CMD": "omni_python",
            "GENIESIM_BREAK_SYSTEM_PKGS": "0",
            "GENIESIM_CHOWN_KIT_PATH": "/usr/local/lib/python3.12/dist-packages/isaacsim/kit",
            "GENIESIM_EXTRA_CACHE_DIRS": "kit pip numba",
            "GENIESIM_ISAACSIM_KIT_CACHE_PATH": "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache",
            "GENIESIM_OVRTX_CACHE_PATH": "/usr/local/lib/python3.12/dist-packages/ovrtx/bin/cache",
        },
    },
}

_active_variant: str = ""
# Command word the user typed at the CLI ("docker", "docker5.1", ...).
# Status/help banners echo this so the user sees the form they invoked,
# not the resolved variant alias.
_invoked_as: str = "docker"


def _cmd_label() -> str:
    return _invoked_as


def _image() -> str:
    v = _VARIANTS[_active_variant]
    return os.environ.get("GENIESIM_IMAGE", v["image"])


def _container() -> str:
    v = _VARIANTS[_active_variant]
    return os.environ.get("GENIESIM_CONTAINER", v["container"])


def _repo_root() -> Path:
    """Locate the repo root that contains the docker/ dir."""
    override = _env.repo_root()
    if override:
        return Path(override).resolve()

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "docker" / "Dockerfile").is_file():
            return candidate

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "docker" / "Dockerfile").is_file():
            return ancestor

    print(f"{RED}❌ Could not locate the repo root (no docker/Dockerfile found).{RST}")
    print(f"   {DIM}Set {BOLD}$GENIESIM_REPO_ROOT{RST}{DIM} or run from inside the repo.{RST}")
    sys.exit(1)


def _docker_dir() -> Path:
    v = _VARIANTS[_active_variant]
    return _repo_root() / v["dir"]


def _find_assets_src() -> str:
    import importlib.util

    spec = importlib.util.find_spec("geniesim_assets")
    if spec and spec.submodule_search_locations:
        loc = Path(list(spec.submodule_search_locations)[0])
    elif spec and spec.origin:
        loc = Path(spec.origin).parent
    else:
        return ""
    for p in [loc, *loc.parents]:
        if (p / "pyproject.toml").is_file():
            return str(p)
    return ""


def _ensure_docker_cli() -> None:
    if os.environ.get("GENIESIM_IN_CONTAINER") == "1":
        print(f"{RED}❌ 'geniesim {_cmd_label()}' cannot be used inside the container.{RST}")
        print(f"   {DIM}Run this command on the host machine.{RST}")
        sys.exit(1)
    if shutil.which("docker") is None:
        print(f"{RED}❌ docker is not on PATH.{RST}")
        print(f"   {DIM}Install Docker Engine first: https://docs.docker.com/engine/install/{RST}")
        sys.exit(1)


def _container_state() -> str | None:
    """Return ``'running'`` / ``'exited'`` / ``...`` or ``None`` if absent."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", _container()],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _print_usage() -> None:
    label = _cmd_label()
    print(f"{BOLD}{MAGENTA}🐳 geniesim {label}{RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim {label} {CYAN}<subcommand>{RST} [args...]")
    print()
    print(f"{BOLD}Subcommands:{RST}")
    print(f"  {CYAN}build{RST} [--china] [docker-build args...]  🔨 Build the image from {BOLD}docker/Dockerfile{RST}")
    print(f"  {CYAN}up{RST} [--headless]                  🚀 Start the container (GUI by default)")
    print(f"  {CYAN}down{RST}                              🛑 Stop and remove the container")
    print(f"  {CYAN}into{RST}                              📖 Drop into a shell in the running container")
    print(f"  {CYAN}logs{RST} [-f]                         📋 Show container logs ({DIM}-f to follow{RST})")
    print()
    print(f"{BOLD}Flags:{RST}")
    print(f"  {WHITE}--china{RST}  {DIM}use daocloud (nvcr) + tuna (apt/pip) mirrors during build{RST}")
    print()
    print(f"{BOLD}Environment overrides:{RST}")
    v = _VARIANTS[_active_variant]
    print(f"  {WHITE}GENIESIM_IMAGE{RST}      {DIM}image tag (default: {v['image']}){RST}")
    print(f"  {WHITE}GENIESIM_CONTAINER{RST}  {DIM}container name (default: {v['container']}){RST}")
    print(f"  {WHITE}GENIESIM_WORKSPACE{RST}  {DIM}host path bind-mounted at /workspace (default: cwd){RST}")
    print(f"  {WHITE}GENIESIM_CACHE_ROOT{RST} {DIM}host path for Isaac Sim cache (default: ~/docker/isaac-sim){RST}")


def _do_build(extra: list[str]) -> None:
    repo_root = _repo_root()
    v = _VARIANTS[_active_variant]
    dockerfile = _docker_dir() / v["dockerfile"]
    image = _image()

    # Pull --china out of the user-supplied extras. Anything else is
    # forwarded verbatim to `docker build` (e.g. --no-cache, --progress).
    china = False
    forwarded: list[str] = []
    for tok in extra:
        if tok == "--china":
            china = True
        else:
            forwarded.append(tok)

    print(f"{BOLD}{MAGENTA}🔨 geniesim {_cmd_label()} build{RST}")
    print(f"   {DIM}Repo root:{RST}  {CYAN}{repo_root}{RST}")
    print(f"   {DIM}Dockerfile:{RST} {CYAN}{dockerfile}{RST}")
    print(f"   {DIM}Tag:{RST}        {CYAN}{image}{RST}")
    if china:
        print(f"   {DIM}Mirrors:{RST}    {YELLOW}🇨🇳 china (daocloud + tuna){RST}")
    print()

    build_args: list[str] = []
    if china:
        build_args += [
            "--build-arg",
            f"ISAACSIM_REGISTRY={_CHINA_NVCR_REGISTRY}",
            "--build-arg",
            f"APT_MIRROR={_CHINA_APT_MIRROR}",
            "--build-arg",
            f"PIP_INDEX_URL={_CHINA_PIP_INDEX}",
        ]

    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "-t",
        image,
        *build_args,
        *forwarded,
        str(repo_root),
    ]
    print(f"   {YELLOW}$ {' '.join(cmd)}{RST}")
    print()
    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "1"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"   {RED}❌ docker build failed (exit {result.returncode}){RST}")
        sys.exit(result.returncode)
    print(f"   {GREEN}✅ Built {image}{RST}")


def _do_up(headless: bool) -> None:
    state = _container_state()
    if state == "running":
        print(f"{YELLOW}⚠️  Container {BOLD}{_container()}{RST}{YELLOW} is already running.{RST}")
        print(
            f"   {DIM}Use{RST} {CYAN}geniesim {_cmd_label()} into{RST} {DIM}to enter it, or{RST} {CYAN}geniesim {_cmd_label()} down{RST} {DIM}to recreate.{RST}"
        )
        return
    if state is not None:
        print(f"{YELLOW}⚠️  Container {BOLD}{_container()}{RST}{YELLOW} exists (state: {state}).{RST}")
        print(f"   {DIM}Run{RST} {CYAN}geniesim {_cmd_label()} down{RST} {DIM}first.{RST}")
        sys.exit(1)

    script = _docker_dir() / "start.sh"
    if not script.is_file():
        print(f"{RED}❌ Missing {script}{RST}")
        sys.exit(1)

    print(f"{BOLD}{MAGENTA}🚀 geniesim {_cmd_label()} up{RST} {DIM}({'headless' if headless else 'gui'}){RST}")
    print(f"   {DIM}Image:{RST}     {CYAN}{_image()}{RST}")
    print(f"   {DIM}Container:{RST} {CYAN}{_container()}{RST}")
    print()

    env = os.environ.copy()
    env.setdefault(_env.GENIESIM_IMAGE.name, _image())
    env.setdefault(_env.GENIESIM_CONTAINER.name, _container())
    env.setdefault(_env.GENIESIM_WORKSPACE.name, str(_repo_root()))
    env.setdefault("GENIESIM_ASSETS_SRC", _find_assets_src())
    # Inject variant-specific values for start.sh and entrypoint.sh.
    # Use direct assignment (not setdefault) so the variant always wins over
    # any same-named vars the user may have exported on the host (e.g. ROS_DISTRO).
    env.setdefault("GENIESIM_VARIANT_LABEL", _cmd_label())
    for key, val in _VARIANTS[_active_variant]["env"].items():
        env[key] = val
    cmd = ["bash", str(script)]
    if headless:
        cmd.append("--headless")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"   {RED}❌ up failed (exit {result.returncode}){RST}")
        sys.exit(result.returncode)


def _do_down() -> None:
    name = _container()
    state = _container_state()
    if state is None:
        print(f"{DIM}No container named {BOLD}{name}{RST}{DIM}; nothing to do.{RST}")
        return
    print(f"{BOLD}{MAGENTA}🛑 geniesim {_cmd_label()} down{RST}")
    print(f"   {DIM}Container:{RST} {CYAN}{name}{RST} {DIM}(state: {state}){RST}")
    if state == "running":
        subprocess.run(["docker", "stop", name], check=False)
    subprocess.run(["docker", "rm", "-f", name], check=False)
    print(f"   {GREEN}✅ Removed{RST}")


def _do_into() -> None:
    name = _container()
    state = _container_state()
    if state != "running":
        print(f"{RED}❌ Container {BOLD}{name}{RST}{RED} is not running (state: {state}).{RST}")
        print(f"   {DIM}Start it with{RST} {CYAN}geniesim {_cmd_label()} up{RST}{DIM}.{RST}")
        sys.exit(1)
    uid = os.getuid()
    gid = os.getgid()
    os.execvp(
        "docker",
        [
            "docker",
            "exec",
            "-it",
            "-u",
            f"{uid}:{gid}",
            "-e",
            "HOME=/home/isaac-sim",
            "-w",
            "/workspace",
            name,
            "bash",
            "-l",
        ],
    )


def _do_logs(follow: bool) -> None:
    name = _container()
    state = _container_state()
    if state is None:
        print(f"{RED}❌ No container named {BOLD}{name}{RST}{RED}.{RST}")
        sys.exit(1)
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append(name)
    os.execvp(cmd[0], cmd)


def run(args: list[str], variant: str = "", invoked_as: str = "docker") -> None:
    global _active_variant, _invoked_as
    _active_variant = variant
    _invoked_as = invoked_as
    _ensure_docker_cli()

    # ---- experimental variant warning -----------------------------------
    # geniesim4 / Isaac Sim 6.0 is incoming — Dockerfile is a placeholder,
    # image is not published, runtime is unverified. Let developers who
    # explicitly want to try it through, but make them acknowledge first
    # so failures aren't mistaken for bugs in their setup.
    if not _VARIANTS[_active_variant].get("implemented", True):
        print(
            f"\n{YELLOW}"
            f"╔════════════════════════════════════════════════════════════════╗\n"
            f"║  🚧  INCOMING variant: {invoked_as + ' (geniesim4)':<40}║\n"
            f"╠════════════════════════════════════════════════════════════════╣\n"
            f"║  This variant targets Isaac Sim 6.0 / geniesim4 and is not     ║\n"
            f"║  fully implemented yet. The Dockerfile is a placeholder, the   ║\n"
            f"║  image is not published, and runtime is unverified.            ║\n"
            f"║                                                                ║\n"
            f"║  For a stable container use geniesim docker (→ docker5.1).     ║\n"
            f"╚════════════════════════════════════════════════════════════════╝"
            f"{RST}\n"
        )
        try:
            input(f"{YELLOW}Press [ENTER] to continue, or Ctrl+C to abort...{RST}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{RED}Aborted by user.{RST}\n")
            sys.exit(130)

    if not args or args[0] in ("-h", "--help", "help"):
        _print_usage()
        return

    sub = args[0]
    rest = args[1:]

    if sub == "build":
        _do_build(rest)
        return
    if sub == "up":
        headless = "--headless" in rest
        _do_up(headless)
        return
    if sub == "down":
        _do_down()
        return
    if sub == "into":
        _do_into()
        return
    if sub == "logs":
        follow = "-f" in rest or "--follow" in rest
        _do_logs(follow)
        return

    print(f"{RED}❌ Error: unknown docker subcommand '{sub}'{RST}")
    print()
    _print_usage()
    sys.exit(1)
