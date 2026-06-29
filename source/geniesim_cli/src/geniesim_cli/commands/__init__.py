# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Verb modules for the ``geniesim`` CLI.

Each module in this package implements a single top-level verb (or a
small family of related sub-verbs, e.g. ``ros build`` / ``ros graph``)
and exports a single ``run(args: list[str]) -> None`` entry point.

The dispatcher in :mod:`geniesim_cli.cli` is intentionally a thin router
that maps verb strings to the matching ``run`` function. This keeps the
top-level file short, makes each verb independently testable, and lets
new verbs be added by dropping a single new module here plus one line
in the dispatch table.

Discovery is intentionally explicit (a static dict in ``cli.py``) rather
than dynamic / pkgutil-based: we want a deterministic command list, and
we want each verb's module to be lazy-loaded only when invoked, so that
``geniesim --help`` and ``geniesim version`` stay fast even when some
optional sibling distributions are missing or broken.
"""
