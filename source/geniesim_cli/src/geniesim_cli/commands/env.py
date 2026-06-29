# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim env`` — inspect the GENIESIM_* environment registry.

This subcommand is the user-facing window into :mod:`geniesim_cli._env`.
It walks :data:`geniesim_cli._env.REGISTRY` and prints, for every
registered variable, the canonical name, current value (or default),
category, and which files in the repo consume it.

By default only variables with a current value are shown. ``--all``
includes unset entries; ``--unset`` shows only the unset ones (useful
when bootstrapping a new environment).
"""

from __future__ import annotations

import sys

from geniesim_cli import _env
from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RST, WHITE, YELLOW

_USAGE = f"""{BOLD}{MAGENTA}🧞 geniesim env{RST} {DIM}— inspect GENIESIM_* environment variables{RST}

Usage:
    {CYAN}geniesim env{RST}             Show set variables grouped by category
    {CYAN}geniesim env --all{RST}       Include unset variables (show defaults)
    {CYAN}geniesim env --unset{RST}     Show only unset variables
"""


def _print_var(var, value):
    """Render a single ``(EnvVar, current_value)`` record."""
    set_marker = f"{GREEN}✅{RST}" if value is not None else f"{DIM}⏭️ {RST}"
    value_str = f"{BOLD}{value}{RST}" if value is not None else f"{DIM}{var.default}{RST}"
    print(f"  {set_marker} {WHITE}{var.name}{RST} = {value_str}")
    print(f"     {DIM}{var.description}{RST}")
    if var.consumers:
        print(f"     {DIM}consumers:{RST} {DIM}{', '.join(var.consumers)}{RST}")


def run(argv: list[str]) -> None:
    show_all = "--all" in argv
    only_unset = "--unset" in argv
    if "--help" in argv or "-h" in argv:
        print(_USAGE)
        return

    groups = _env.by_category()
    print(f"{BOLD}{MAGENTA}🧞 geniesim env{RST} {DIM}— registry view{RST}\n")

    shown = 0
    for category, vars_in_cat in groups.items():
        rendered = []
        for var in vars_in_cat:
            value = var.get()
            if only_unset and value is not None:
                continue
            if not show_all and not only_unset and value is None:
                continue
            rendered.append((var, value))
        if not rendered:
            continue
        print(f"{BOLD}{CYAN}📦 {category}{RST}")
        for var, value in rendered:
            _print_var(var, value)
            shown += 1
        print()

    if shown == 0:
        if only_unset:
            print(f"{GREEN}✅ all registered GENIESIM_* variables are set{RST}")
        else:
            print(f"{YELLOW}⚠️  no GENIESIM_* variables currently set{RST}")
            print(f"   {DIM}pass {CYAN}--all{RST}{DIM} to view defaults{RST}")
    sys.exit(0)
