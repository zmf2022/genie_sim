# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Shared ANSI color constants and style helpers for the geniesim CLI.

All CLI-facing modules across every ``geniesim_*`` distribution should
import colors from here to ensure consistent style and proper TTY /
NO_COLOR detection. ``geniesim_cli`` is the root of the dependency DAG
(every other dist depends on it), so this import is always safe and
never creates a cycle::

    from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, RED, RST, YELLOW

Theme-safety contract
---------------------
This palette is **terminal-theme-agnostic**: it must render legibly on
both light and dark terminal backgrounds. To honour that contract we
follow two rules:

1. Only use the **standard 8 ANSI foreground colors** (30-37). Every
   modern terminal (iTerm2, Alacritty, GNOME Terminal, Windows Terminal,
   VS Code, etc.) remaps these to a theme-appropriate palette so the
   user's chosen scheme decides the actual rendered hue. **Never** use
   the bright variants 90-97 — in particular ``\\033[97m`` (bright
   white) renders invisibly on light terminals, and ``\\033[30m``
   (black) renders invisibly on dark terminals.
2. Use the **default-foreground reset** ``\\033[39m`` for "emphasize
   this token without choosing a colour" (our ``WHITE`` constant —
   which is now a misnomer kept for source-compat; it does not paint
   the text white). This restores the terminal's default fg so the
   text inherits whatever colour the user configured for plain output.

Bold / dim / reset are pure SGR *attributes* with no colour, so they
are theme-agnostic by construction.
"""

from __future__ import annotations

import os
import sys

BOLD = "\033[1m"
DIM = "\033[2m"
RST = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[39m"

if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
    BOLD = DIM = RST = ""
    CYAN = GREEN = YELLOW = RED = MAGENTA = BLUE = WHITE = ""
