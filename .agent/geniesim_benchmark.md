# geniesim_benchmark — repository-level dispatcher

> 🧭 **Canonical source**: [`source/geniesim_benchmark/README.md`](../source/geniesim_benchmark/README.md) is the source of truth for the task runtime, config schema, CLI verbs, and scoring contract.

This file is a 30-second pointer. Do not duplicate content here that lives at the canonical source — duplication is what makes dispatchers rot.

---

## What it is

The **legacy** benchmark task runtime: declarative `.yaml` configs in `config/` describe what to load, which robot to drive, and what to evaluate. Tasks are launched through the `geniesim benchmark` CLI verb (owned by [`geniesim_cli`](geniesim_cli.md)) and run against a user-provided inference server (`--infer-host=IP:PORT`).

> 🚧 **Architectural status — legacy, refactor planned.**
> `geniesim_benchmark` talks to **Isaac Sim directly** (its own physics + scene loading); it is **not** layered on top of [`geniesim_ros`](geniesim_ros.md). The two stacks are **independent and parallel today**: choose the benchmark stack for headless / scoring runs, the RT Engine for interactive / closed-loop work.
>
> The roadmap is to refactor the benchmark runtime into a **benchmark layer on top of `geniesim_ros`** — a plugin or app inside the RT Engine — so the two stacks share one physics path. Until that lands, do not assume shared invariants between them.

## Where to look

| Topic | File |
|---|---|
| Canonical CLI surface + config naming convention | [`source/geniesim_benchmark/README.md`](../source/geniesim_benchmark/README.md) |
| User-facing intro + leaderboard pointer | [`source/geniesim_benchmark/README.md`](../source/geniesim_benchmark/README.md) |
| Agent skills (run, check-inference) | [`source/geniesim_benchmark/skills/`](../source/geniesim_benchmark/skills/) |
| Task configs (the work-list) | [`source/geniesim_benchmark/src/geniesim_benchmark/config/`](../source/geniesim_benchmark/src/geniesim_benchmark/config/) |
| App entry | [`source/geniesim_benchmark/src/geniesim_benchmark/app/app.py`](../source/geniesim_benchmark/src/geniesim_benchmark/app/app.py) |
| Dataset utilities (converters) | [`source/geniesim_benchmark/src/geniesim_benchmark/dataset/`](../source/geniesim_benchmark/src/geniesim_benchmark/dataset/) — `agibot → LeRobot v2.1` lives in `dataset/convert/agibot_to_lerobot.py`; CLI entry is `geniesim dataset convert agibot-to-lerobot` |

## Invariants the rest of the repo relies on

- **Independent stack from `geniesim_ros`.** No shared scene loader, no shared launch graph. A scene yaml here is **not** a scene yaml there. Don't cross-reference; don't unify prematurely — the refactor is the right place for that.
- **Config naming is `<robot>_<category>_<task>.yaml`.** The CLI splits on the **second** token as category; deeper tags (e.g. `g2op_probe_if_pick_*`) belong to the task name. Robot prefixes (`g2op`, `g290d`, `arxone`, `g1op`, `aloha`) are **internal to this package** today — they're not the same vocabulary as `geniesim_ros` scene yamls.
- **The `--infer-host=H:P` shorthand** is rewritten to `--benchmark.infer_host=…` (run/batch) or `--host …  --port …` (check-inference). Don't pin a host inside a `.yaml` — keep it at invocation time.
- **`config.yaml` / `template.yaml` / `teleop.yaml`** are templates / defaults, **not** runnable tasks. `geniesim benchmark list` filters them out.
- **Unknown `--key=value` flags are forwarded verbatim** to `app/app.py`'s `ParameterServer`. The full key space is the `@dataclass` tree in `config/params.py`. Don't strip "unknown" flags in the CLI dispatch.
- **`batch` semantics:** sequential per-config `subprocess.run`; exits non-zero if **any** config failed. Long jobs need `tmux` / log pipe.
- **Leaderboard tables in the root README (`GenieSim-Instruction` / `-Robust` / `-Manipulation` / `-Sim2Real`)** are part of the published contract. Don't rename these strings — they survive the refactor.
