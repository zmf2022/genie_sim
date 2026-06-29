# scene_reconstruction — Agent Development Guide

Scene-level 3D reconstruction pipeline. Ingests multi-view images +
LiDAR point cloud, produces a PGSR Gaussian-splatting asset for
downstream Genie Sim / Isaac Sim consumption.

Source: [source/scene_reconstruction/](.)
License: [LICENSE](LICENSE) (multi-license — see § 4 below)
Human-readable docs: [README.md](README.md) — every operational
detail (build, run, parameters, troubleshooting, GPU arch matrix,
weights cache, …) lives there. **Read it first** if you need to
*use* the pipeline; this file only governs how to *change* it.
Module map: [../README.md](../README.md) · [../AGENTS.md](../AGENTS.md)

> **Maintenance contract** — when you re-pin a third-party commit,
> change a patch's MD5, alter the conda env matrix, change the
> entrypoint script's argument shape, or modify the input/output
> layout, **update the README in the same diff**. Agents read this
> file as the contract for *what may not change without explicit
> sign-off*.

> **Separately-maintained module** — not part of the `geniesim_*`
> peer set, not installed by `geniesim bootstrap`, not in the
> umbrella's dep tree. Has its own Docker build and release cadence.

---

## 1. Canonical source of truth

Refer to the `Data Format` and `Run Scene Reconstruction` sections in
the [README](./README.md) as **the** source of truth.

`real2sim_environment_entrypoint.sh` is the main entrypoint — every
operational path eventually runs through it.

---

## 2. Restricted scope

```text
┌─────────────────────────────────────────────────────────────┐
│  Work ONLY in:  source/scene_reconstruction                 │
│                                                             │
│  Do NOT modify:                                             │
│    ✗ source/geniesim        ✗ source/data_collection        │
│    ✗ benchmark scripts      ✗ robot-control code            │
└─────────────────────────────────────────────────────────────┘
```

Changes that touch any sibling module are out of scope for this
guide — coordinate with the owning module's AGENTS.md.

---

## 3. Architectural invariants (do not violate)

1. **Pinned commits are the contract.** Every third-party dependency
   in the README's "Pinned dependencies" table is locked to a
   specific commit / version. Do **not** upgrade casually — each has
   been smoke-tested against the full pipeline. To re-pin, regenerate
   the corresponding patch, update its MD5 in the README, and run an
   end-to-end pipeline run before merging.

2. **Patch checksums are integrity gates.** The three patches under
   `patch/` apply to specific upstream commits; their MD5s are
   recorded in the README. If an MD5 changes, the upstream commit
   has drifted — re-derive the patch, update the MD5, document the
   reason.

3. **Docker `COPY . .` requires `source/scene_reconstruction` as
   build context.** Don't break this by adding builds that COPY from
   elsewhere.

4. **Conda envs are isolated** — `base` (Python 3.12, gsplat),
   `pgsr` (Python 3.8, COLMAP/PGSR/HLoc), `difix` (Python 3.8,
   Difix3D). Each script declares its env in the README's
   script-to-stage map. Don't cross the streams: a script running in
   the wrong env will silently mis-import.

5. **Don't mount host paths over `/root`** at runtime. It hides the
   built third-party deps. Mount only `/data/*` and
   `/root/.cache/torch`.

6. **Input/output layout is the cross-module contract.** The input
   schema (`camera/left/*.png`, `info/calibration.json`,
   `transforms.json`, `colorized.las`) and the output structure
   (`gs-asset/` as the downstream consumable) are referenced by
   upstream capture tooling and downstream Genie Sim loading. Don't
   silently rename, flatten, or add required fields.

7. **`TORCH_CUDA_ARCH_LIST` defaults to Ada (`8.9`).** Other GPU
   architectures need the README's arch matrix knob — but the
   default stays Ada because that's what CI builds and tests against.

---

## 4. License preservation rules

Multi-license module — every component carries its own terms. See
the README's "Licensing" table for the breakdown.

### Rules

- ✅ Preserve `LICENSE` file
- ✅ Preserve all `patch/` files
- ✅ Preserve `third_party/` notices
- ✅ Check each dependency license before redistributing
- ❌ Do NOT remove copyright headers
- ❌ Do NOT claim unified licensing

---

## 5. Agent rules

> 🛑 **This module contains pinned third-party reconstruction code.**
> Keep changes small, explicit, and reproducible.

### ✅ DO

| # | Rule |
|---|---|
| 1 | Work only inside `source/scene_reconstruction` |
| 2 | Make small, targeted changes |
| 3 | Keep Dockerfile changes minimal and reproducible |
| 4 | Use variable-based, copy-pasteable commands |
| 5 | Follow the [README](./README.md) as source of truth |
| 6 | Verify script behavior before updating documentation |
| 7 | Preserve all patch files and licenses |

### ❌ DO NOT

| # | Rule |
|---|---|
| 1 | Modify unrelated Genie Sim modules |
| 2 | Remove patch files or license notices |
| 3 | Change pinned commits without explicit user request |
| 4 | Upgrade CUDA / PyTorch / pycolmap / PGSR / gsplat / HLoc casually |
| 5 | Hard-code private local paths |
| 6 | Mount host directories over `/root` |
| 7 | Reinstall pre-built packages inside running container |
| 8 | Invent a new pipeline if an official one exists |
| 9 | Move user-readable content into this file — that lives in [README.md](README.md) |

---

## 6. 📎 Final notes

- This `AGENTS.md` is a practical operating contract for coding agents; the [README.md](README.md) is the practical operating guide for users.
- When official commands differ from this guide, **follow the official docs** and update this file.
- The main entrypoint script is `real2sim_environment_entrypoint.sh` — start there when tracing how a stage is invoked.
