# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""geniesim_cli — lightweight command-line frontend for geniesim.

This distribution intentionally has zero heavy runtime dependencies. It
ships only the dispatcher and the shared ANSI style helpers so that
`pip install geniesim_cli` is fast and works in minimal environments
(e.g. a CI runner that only needs to call `geniesim deploy upload`).

Heavy work — Isaac Sim, USD, MuJoCo, ROS 2, asset conversion — lives in
the sibling distributions ``geniesim`` and ``geniesim_assets``. The CLI
imports them lazily, on demand, so a missing optional dependency does
not break unrelated subcommands.
"""

from __future__ import annotations

from geniesim_cli._version import __version__

__all__ = ["__version__"]
