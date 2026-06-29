# geniesim_cli — repository-level dispatcher

> 🧭 **Canonical source**: [`source/geniesim_cli/AGENTS.md`](../source/geniesim_cli/AGENTS.md) — the per-package guide is the source of truth for command dispatch, lazy-import rules, and the `_STATUS_DISTRIBUTIONS` / `_INIT_TARGETS` tables.

This file is a 30-second pointer. Do not duplicate content here that lives at the canonical source — duplication is what makes dispatchers rot.

---

## What it is

`geniesim_cli` is the **standalone PEP 517 / PEP 621 wheel** that owns the `geniesim` console script. Sole CLI front-end for the platform: docker lifecycle, ROS 2 workspace builds, health probes, deploy, asset / scene bootstrap, benchmark dispatch, teleop dispatch.

Lightweight by contract: no heavy runtime deps at import time (no USD, no Isaac Sim, no MuJoCo). Operators can run `geniesim status` / `geniesim deploy` on a control node where only the CLI is installed.

## Where to look

| Topic | File |
|---|---|
| Canonical command dispatch + lazy-import rules | [`source/geniesim_cli/AGENTS.md`](../source/geniesim_cli/AGENTS.md) |
| User-facing intro + command table | [`source/geniesim_cli/README.md`](../source/geniesim_cli/README.md) |
| Fresh-machine setup (install order, bootstrap, status, doctor) | [`source/geniesim_cli/AGENTS.md` § 0](../source/geniesim_cli/AGENTS.md) |
| Implementation | [`source/geniesim_cli/src/geniesim_cli/cli.py`](../source/geniesim_cli/src/geniesim_cli/cli.py) + [`commands/`](../source/geniesim_cli/src/geniesim_cli/commands/) |
| ANSI / style constants | [`source/geniesim_cli/src/geniesim_cli/_style.py`](../source/geniesim_cli/src/geniesim_cli/_style.py) |

## Invariants the rest of the repo relies on

- The `geniesim` console script is owned **only** by `geniesim_cli`. Any sibling distribution declaring `[project.scripts] geniesim = …` is a packaging bug.
- `import geniesim_cli.cli` must succeed in a venv with **only** `geniesim_cli` installed — every heavy import (USD, Isaac, MuJoCo, `geniesim.*`, `geniesim_assets.*`, …) is lazy, inside the function that uses it.
- `geniesim status` must never raise on missing siblings — treat absent distributions as a finding, not an error.
- Adding a new peer distribution means updating `_STATUS_DISTRIBUTIONS`, `_DEPLOY_MODULES` (or `_SKIP_DEPLOY`), and `_INIT_TARGETS` in the same diff.
- Hard rule: never hint `pip install geniesim` or `pip install geniesim_assets` — not on PyPI. Always redirect through `geniesim bootstrap`.
