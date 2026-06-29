# geniesim_cli — Agent Development Guide

Lightweight command-line front-end for the **geniesim** stack. Published as the
[`geniesim_cli`](./pyproject.toml) PyPI distribution; sole owner of the
`geniesim` console script.

**Maintenance contract** (read before editing): whenever you change command
dispatch, add/remove a subcommand, change colour/emoji conventions, or alter
how a sibling distribution is detected, **update this file in the same diff**.
This document is the agent's source of truth for what `geniesim_cli` does and
how it talks to its siblings.

---

## 0. Fresh-machine setup

The CLI is the **first thing** installed on a new machine — it ships the
`geniesim bootstrap` verb that lays down every other peer.

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
pip install -e source/geniesim_cli/      # 1. CLI first (no heavy deps)
geniesim bootstrap                       # 2. installs siblings + assets
geniesim status                          # 3. all-green check
geniesim doctor                          # 4. repair if not
```

**Install-order contract** (do not violate):

1. **`geniesim_cli` first.** Bootstrap walks the dependency graph; if the
   CLI itself isn't installed, the `geniesim` console script doesn't exist
   and you can't call `bootstrap` at all. This is why `geniesim_cli` is
   **deliberately excluded** from `_INIT_TARGETS` — re-installing the
   running CLI mid-execution is a footgun (see §7a).
2. **Topological install order is the CLI's job.** `_INIT_TARGETS` in
   `cli.py` enforces leaves-first, umbrella last. Bypassing the CLI and
   running `pip install` by hand in the wrong order leaves dangling deps.
3. **`bootstrap` is idempotent.** Re-running on a working machine is a
   cheap no-op (each step skips if the distribution imports). Use it as
   a "fix whatever's broken" verb.
4. **Asset pack is a separate download.** Python distributions are tiny;
   the 3D assets are several GB on HuggingFace + ModelScope. `bootstrap`
   fetches them via `_INIT_DOWNLOAD_SOURCES`.
5. **`geniesim` and `geniesim_assets` are NOT on PyPI.** Always go through
   `geniesim bootstrap`; never suggest `pip install geniesim` to the user.

---

## 1. Why this distribution exists

`comfy_cli` ↔ `comfy` was the model: ship the CLI as a separate, **dependency-light**
wheel so that:

- `pip install geniesim_cli` works in any environment, even one without USD,
  MuJoCo, or Isaac Sim.
- The console script does not collide with — and is not blocked by — the heavy
  SDK installations.
- Operators can run `geniesim status`, `geniesim version`, `geniesim deploy`
  even on a control node where only the CLI is installed.

There are now **three distributions** in the stack:

| Distribution | Role | Heavy deps? |
|---|---|---|
| `geniesim_cli` (this) | Dispatcher + ops commands | **No** — `dependencies = []` |
| `geniesim` | SDK: experimental tools, ROS2 surface, Isaac Sim glue | Yes |
| `geniesim_assets` | USD↔MJCF converters, asset compile | Medium (USD, trimesh, Pillow) |

The `[full]` extra in [pyproject.toml](./pyproject.toml) pins exact versions of
the other two for one-shot installs (`pip install "geniesim_cli[full]"`).

---

## 2. Layout

```
source/geniesim_cli/
├── AGENTS.md              ← this file
├── pyproject.toml         ← `geniesim` console script entry point lives here
├── setup.py               ← thin shim, find_packages(where="src")
└── src/geniesim_cli/
    ├── __init__.py        ← `__version__`
    ├── __main__.py        ← `python -m geniesim_cli`
    ├── _style.py          ← canonical ANSI / NO_COLOR / TTY-detect constants
    └── cli.py             ← single-file dispatcher (~650 lines)
```

`_style.py` is the **canonical home** of the ANSI helpers. Other distributions
(`geniesim`, `geniesim_assets`) may re-export from here once they take a
runtime dep on `geniesim_cli`; until they do, they keep their own copy.
**Never** redefine ANSI escapes inline in this package — always import from
[`geniesim_cli._style`](./src/geniesim_cli/_style.py).

---

## 3. Command surface

All commands route through [`main()`](./src/geniesim_cli/cli.py) at the bottom
of `cli.py`. Keep the top-of-file usage docstring, [`_print_usage()`](./src/geniesim_cli/cli.py),
and the dispatcher in `main()` in sync.

| Command | Purpose | Heavy deps required? |
|---|---|---|
| `geniesim version` | Print versions of all 3 distributions + Python | No |
| `geniesim status` | Per-distribution health probe + console-script + env | No |
| `geniesim bootstrap` | Bootstrap / re-initialize `geniesim` + `geniesim_assets` (interactive `pip install`) | No (but invokes `pip`) |
| `geniesim ros build dev` | `colcon build` with RelWithDebInfo + `--symlink-install` | Needs colcon on PATH |
| `geniesim ros build release` | `colcon build` with Release | Needs colcon on PATH |
| `geniesim ros build cleanup` | Remove `devel*`, `build/`, `install/`, `log/` (interactive prompt) | No |
| `geniesim ros graph` | `colcon graph` text → `dot -Tpng` PNG | Needs colcon + graphviz |
| `geniesim experimental TOOL …` | Run an experimental tool (lazy import from `geniesim`) | Yes (`geniesim`) |
| `geniesim deploy [MODULE]` | Build wheel(s) for every tier-1 peer (or one named MODULE) into `./deploy/` | Needs `python -m build` |
| `geniesim deploy upload` | `curl -F` upload of `./deploy/*.whl` to file server | Needs `curl` |
| `geniesim doctor` | Diagnose & repair (status + rosdep + more) | No (probes tools) |
| `geniesim docker[6.0\|5.1\|4.5] <build\|up\|down\|into\|logs>` | Manage the GenieSim container across Isaac Sim variants | Needs docker on host |
| `geniesim env [--all\|--unset]` | Show `GENIESIM_*` env vars | No |
| `geniesim completion bash\|zsh` | Generate shell completion script | No |
| `geniesim benchmark <run\|list\|batch\|categories\|robots\|check-inference>` | Run / inspect benchmark tasks (see [geniesim_benchmark/README.md](../geniesim_benchmark/README.md)) | `run`/`batch`: Yes (Isaac Sim); rest: No |
| `geniesim autocollect <list\|tasks\|robots\|run\|build\|up\|into\|down>` | List / run / build / manage data_collection tasks & container (see [data_collection/AGENTS.md](../data_collection/AGENTS.md)) | `run`/`up`: Docker + GPU; `build`: Docker; rest: No |
| `geniesim tool deps-dag` | Verify the module-dependency DAG in `source/README.md` matches every peer's `pyproject.toml` (CI hook) | No |
| `geniesim tool deps-dag --fix` | Regenerate the DAG in-place between `<!-- AUTOGEN:deps-dag -->` markers | No |
| `geniesim tool ros-dag` | Verify the ROS-package DAG in `source/geniesim_ros/README.md` matches every workspace `package.xml` (CI hook). **Replaces the retired `geniesim ros graph` PNG verb.** | No |
| `geniesim tool ros-dag --fix` | Regenerate the ROS-package DAG in-place between `<!-- AUTOGEN:ros-dag -->` markers | No |
| `geniesim tool docs` | Repo-wide doc-coverage audit: coverage × index × links, across cli + ros scopes (CI hook) | No |
| `geniesim tool docs --scope ros` | ROS-workspace-only audit (replaces the old `audit_package_docs.py` script) | No |
| `geniesim tool docs --check links` | Only run the broken-link audit across every AGENTS/README/SKILL | No |
| `geniesim dataset convert agibot-to-lerobot --agibot-dir … --output-dir …` | Convert agibot v1 → LeRobot v2.1. Dispatcher only — logic lives in [`geniesim_benchmark.dataset.convert.agibot_to_lerobot`](../geniesim_benchmark/src/geniesim_benchmark/dataset/convert/agibot_to_lerobot.py). | Yes (`geniesim_benchmark` + ffmpeg on PATH) |
| `geniesim update` | (not implemented) | — |
| `geniesim upgrade` | (not implemented) | — |

---

## 4. Architectural rules (do not violate)

1. **No top-level heavy imports.** `usd-core`, `mujoco`, `trimesh`, `Pillow`,
   `coacd`, anything from `isaacsim.*`, anything from `geniesim.*`,
   `geniesim_assets.*` — **all** must be imported lazily, inside the function
   that uses them, via [`_import_or_die(module_path, hint_pkg)`](./src/geniesim_cli/cli.py).
   Verify by:
   ```bash
   python -c "import geniesim_cli.cli"   # must succeed in a venv with no extras
   ```
2. **Every coloured f-string ends with `{RST}`.** No exceptions.
3. **No inline ANSI codes.** Import from [`_style`](./src/geniesim_cli/_style.py).
4. **Unknown subcommands print usage and exit non-zero.** See the bottom of
   `main()`. Don't silently fall through.
5. **Lazy-import errors must be friendly.** `_import_or_die` already handles
   this — surface the helpful `pip install <pkg>` hint, never let an
   `ImportError` traceback leak.
6. **The `geniesim` console script is owned here.** If a sibling distribution
   ever declares `[project.scripts] geniesim = …` again, that's a packaging
   bug; resolve in their `pyproject.toml`, not here.
7. **`status` must never raise on missing siblings.** Treat absent
   distributions as a *finding*, not an error. The probe lives in
   [`_status_inspect()`](./src/geniesim_cli/cli.py) and uses
   `importlib.metadata.distribution()` + `importlib.import_module()` wrapped
   in `try/except`.

---

## 5. Locating sibling source trees

[`_distribution_source_root(name)`](./src/geniesim_cli/cli.py) resolves the
on-disk source tree of *any* of the three distributions via
`importlib.util.find_spec`. This is what `geniesim deploy` uses to feed
`python -m build`. Two consequences:

- The CLI **does not** rely on `__file__`-relative path math to find sibling
  source — that broke when `geniesim_cli` moved out of `geniesim`.
- A distribution must be **installed (editable counts)** for `deploy` to find
  it. If you want `geniesim deploy geniesim_assets` to work from a fresh
  checkout, run `pip install -e source_assets` (or the `main` equivalent)
  first.

The ROS workspace root for `geniesim ros build *` is resolved by
[`_ros_workspace_root()`](./src/geniesim_cli/cli.py). It honours
`$GENIESIM_WORKSPACE`, otherwise falls back to `cwd`. **Never** infer the
workspace from `__file__`: the CLI now lives in a different repo from the
ROS engine.

---

## 6. Style rules

- Format every changed `.py` with `black --line-length 120` before finishing.
- Colour conventions:
  - `GREEN` + ✅ — success
  - `RED` + ❌ — error
  - `YELLOW` + ⚠️ — warning / not-installed
  - `CYAN` + relevant emoji — informational / progress
  - `DIM` — debug, secondary detail, metadata
  - `BOLD` — file paths, tool names, key values
  - `MAGENTA` + 🧞 — geniesim brand / banner
  - `WHITE` — list items
- Do **not** add comments unless the user asked for them. Use docstrings on
  non-trivial functions instead (see `_status_inspect`, `_status`).

---

## 7. How `status` decides "healthy"

[`_status()`](./src/geniesim_cli/cli.py) calls `_status_inspect` for each of
the three distributions and reports the following facts (probe spec lives in
[`_STATUS_DISTRIBUTIONS`](./src/geniesim_cli/cli.py)):

1. **installed** — `importlib.metadata.distribution(name)` resolves.
2. **version** — `dist.version`.
3. **editable** — read PEP 610 `direct_url.json` from the dist-info; check
   `dir_info.editable`. Returns `None` (→ "unknown mode") when the file is
   absent (e.g. `PYTHONPATH` shims, system packages without direct_url).
4. **import_ok** — actually `importlib.import_module(top_module)`. Captures
   any exception and surfaces `type(exc).__name__: exc`.
5. **runtime deps** — for each `(pip_name, import_name)` pair declared on
   the spec, attempt `importlib.import_module(import_name)` via
   [`_probe_module`](./src/geniesim_cli/cli.py). A failed dep flips the
   overall verdict to ⚠️ and prints `fix with: geniesim bootstrap`.
6. **submodules** — for each declared sub-package (e.g. `geniesim.assets`,
   `geniesim.experimental`), import-probe it. Useful for catching partial
   installs or namespace collisions.
7. **optional extras** — for each declared extra (e.g. `[coacd]`,
   `[texture]`), probe its import names. Missing extras are reported as
   skipped (⏭️ `DIM`), **not** as failures — they don't affect the verdict.
8. **ROS 2 sourcing** — separate `🤖 ROS 2` block. Considered "sourced"
   iff both `ROS_DISTRO` and `ROS_VERSION` are present in `os.environ`.
   Each of `ROS_DISTRO` / `ROS_VERSION` / `ROS_PYTHON_VERSION` /
   `ROS_LOCALHOST_ONLY` is rendered individually (✅ with value, or ⏭️
   unset). `AMENT_PREFIX_PATH` is rendered as the first colon-separated
   entry plus a `(+N more)` suffix. Not-sourced state prints a
   ⚠️ banner with the actionable
   `source /opt/ros/<distro>/setup.bash` hint. ROS state is a
   *finding*, not a hard failure: it does **not** flip the overall
   verdict — the engine builds need it but `assets` / `experimental`
   commands often don't.

Verdict: `🎉 All systems go!` only when **every** distribution is installed,
imports cleanly, all runtime deps import, and all declared submodules
import. Missing siblings or failed deps flip the verdict to
`⚠️ Some components need attention.` In that case, the report ends with
`Run geniesim bootstrap to install the missing distributions.` — the redirect is
intentional: `status` reports, `init` fixes.

When extending `status` (e.g. probing optional features inside an installed
package), keep it cheap and non-fatal. Heavy probes belong behind a future
`status --verbose` flag, not in the default path. When you add a new
runtime dep / submodule / extra to `geniesim` or `geniesim_assets`, **also
add it to `_STATUS_DISTRIBUTIONS`** in the same diff — that table is the
contract `init` consults to decide what's healthy.

---

## 7a. How `init` decides what to install

[`_init()`](./src/geniesim_cli/cli.py) reuses `_status_inspect` for each
target in [`_INIT_TARGETS`](./src/geniesim_cli/cli.py) (currently
`("geniesim", "geniesim_assets")` — `geniesim_cli` is **deliberately
excluded**: re-installing the running CLI mid-execution is a footgun).

- **Healthy stack** → prompt `Re-initialize anyway? [y/N]` (default
  **No**, per the user contract "ask to re-init or do nothing(default)").
- **Unhealthy stack** → enumerate findings (missing dist / failed import /
  bad runtime dep / missing submodule), build an install plan, prompt
  `Proceed with the install plan above? [Y/n]` (default **Yes** — the user
  reached this branch by invoking `init` to fix things).

The install command per dist is chosen by
[`_init_install_command`](./src/geniesim_cli/cli.py): if the dist is
importable from a local source tree (i.e. editable-installed somewhere or
on `PYTHONPATH`), the command becomes `pip install -e <source_root>`.
**There is no PyPI fallback** — `geniesim` and `geniesim_assets` are
private distributions and must be installed from a local checkout. When
no local source tree can be located, `_init_install_command` returns
`None`; `_init()` then surfaces a friendly ⚠️ block listing the
unresolvable targets and asks the user to clone the source repo (or
place it on `PYTHONPATH`) and re-run `geniesim bootstrap`.

**Hard rule for any new code in this CLI**: do **not** add hints of the
form `pip install geniesim` or `pip install geniesim_assets` (with or
without extras like `[coacd]`). Those distributions are not on PyPI.
Always redirect to `geniesim bootstrap`, optionally with the extra context
"installs from local checkout" — see [`_status()`](./src/geniesim_cli/cli.py)
for the canonical phrasing.

When a private distribution has a public **dataset / asset bundle**
mirror (e.g. `geniesim_assets` is published on HuggingFace + ModelScope),
register the download URLs in
[`_INIT_DOWNLOAD_SOURCES`](./src/geniesim_cli/cli.py). `_init()` reads
that table when it can't find a local source tree and prints a
two-step recipe (`1. Download from <URL>` → `2. pip install -e`) per
unresolved target. The canonical source of these URLs is
[`main/README.md`](../../README.md); whenever the README's badges /
links change, update `_INIT_DOWNLOAD_SOURCES` in the same diff.

---

## 8. Verification recipes

| Goal | Command |
|---|---|
| Lint check this package | `black --line-length 120 --check src/ setup.py` |
| Smoke-test `python -m geniesim_cli` | `PYTHONPATH=src python3 -m geniesim_cli` |
| Smoke-test `version` | `PYTHONPATH=src python3 -m geniesim_cli version` |
| Smoke-test `status` | `PYTHONPATH=src python3 -m geniesim_cli status` |
| Smoke-test `init` (interactive) | `PYTHONPATH=src python3 -m geniesim_cli init` |
| Smoke-test `init` non-interactive (decline) | `printf 'n\n' \| PYTHONPATH=src python3 -m geniesim_cli init` |
| Confirm zero heavy imports | `PYTHONPATH=src python3 -c "import geniesim_cli.cli"` (in a venv with **only** `geniesim_cli`) |
| Editable install | `pip install -e .` from this directory |
| Build wheel | `python -m build --wheel --outdir ../../deploy .` |

---

## 9. Canonical home

This directory is the **canonical home** of `geniesim_cli`. The package
lives here, gets edited here, and is built from here. Older mirror /
symlink setups in adjacent checkouts are no longer maintained — treat
this tree as the only source of truth.
