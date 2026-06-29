# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim dataset`` — dataset conversion / inspection utilities.

The ``dataset`` verb is the noun-space for operations that read or
write robot-dataset artifacts (parquet, h5, mp4). All actual logic
lives in ``geniesim_benchmark.dataset.*`` — keeping it there means the
heavy deps (``h5py``, ``pyarrow``, ``numpy``, ``ffmpeg``) stay paired
with the benchmark stack and only get imported when this verb is
actually invoked. ``geniesim_cli`` itself remains pure-stdlib at
import time.

Sub-commands:
    convert <FROM>-to-<TO>    Convert one dataset format to another.
                              Currently supported:
                                  agibot-to-lerobot   agibot v1 → LeRobot v2.1
"""

from __future__ import annotations

import sys

from geniesim_cli._style import BOLD, CYAN, DIM, MAGENTA, RED, RST, YELLOW

_USAGE = f"""{BOLD}{MAGENTA}🧞 geniesim dataset{RST} {DIM}— dataset conversion / inspection{RST}

Usage:
    {CYAN}geniesim dataset convert agibot-to-lerobot{RST} [...]   Convert agibot v1 → LeRobot v2.1

Run any sub-command with {CYAN}--help{RST} for its own argument list.
"""


def _convert_run(argv: list[str]) -> int:
    """``geniesim dataset convert <FORMAT-PAIR> ...`` dispatcher."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(f"{BOLD}Usage:{RST} geniesim dataset convert {CYAN}<FORMAT-PAIR>{RST} [...]")
        print()
        print(f"{BOLD}Supported pairs:{RST}")
        print(f"  {CYAN}agibot-to-lerobot{RST}   agibot v1 → LeRobot v2.1")
        return 0

    pair = argv[0]
    rest = argv[1:]

    if pair == "agibot-to-lerobot":
        # Heavy deps live in geniesim_benchmark.dataset.convert.* — keep the
        # import lazy so importing geniesim_cli stays pure-stdlib.
        try:
            from geniesim_benchmark.dataset.convert.agibot_to_lerobot import convert_cli
        except ImportError as exc:
            print(
                f"{RED}❌ Cannot import the converter: {exc}{RST}\n"
                f"   {DIM}geniesim_benchmark must be installed "
                f"(it is a tier-1 peer of geniesim).{RST}",
                file=sys.stderr,
            )
            return 1
        return convert_cli(rest)

    print(f"{RED}❌ Error: unknown convert pair '{pair}'{RST}", file=sys.stderr)
    print(f"   {DIM}Supported: agibot-to-lerobot{RST}", file=sys.stderr)
    return 1


def run(argv: list[str]) -> None:
    """Entry point called from ``geniesim_cli.cli.main``."""
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        sys.exit(0)

    sub = argv[0]
    rest = argv[1:]

    if sub == "convert":
        sys.exit(_convert_run(rest))

    print(f"{RED}❌ Error: unknown sub-command 'dataset {sub}'{RST}", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    sys.exit(1)
