# geniesim_generator — repository-level dispatcher

> 🧭 **Canonical source**: [`source/geniesim_generator/AGENTS.md`](../source/geniesim_generator/AGENTS.md) — the per-package guide is the source of truth for the Scene Language DSL, the `LLM_RESULT.py` import contract, the MCP tool surface, and the Open WebUI config.

This file is a 30-second pointer. Do not duplicate content here that lives at the canonical source — duplication is what makes dispatchers rot.

---

## What it is

LLM-driven scene generation: an LLM (hosted in Open WebUI) writes a small Python program in the **Scene Language DSL**; that program builds a `Shape` tree from the shared asset library, and the package compiles it into a `scene.usda` + layout graph that the benchmark / sim stack consumes.

Heavy ML deps live here — gated behind the `[generator]` / `[full]` extras of the umbrella; `geniesim bootstrap` asks before installing.

## Where to look

| Topic | File |
|---|---|
| Canonical DSL primitives, MCP tools, output layout | [`source/geniesim_generator/AGENTS.md`](../source/geniesim_generator/AGENTS.md) |
| User-facing intro | [`source/geniesim_generator/README.md`](../source/geniesim_generator/README.md) |
| Agent skills (generate-scene, search-assets, deploy-generator) | [`source/geniesim_generator/skills/`](../source/geniesim_generator/skills/) |
| Open WebUI / MCP server compose file | [`source/geniesim_generator/`](../source/geniesim_generator/) |

## Invariants the rest of the repo relies on

- **Not a CLI verb.** Unlike `geniesim benchmark` / `geniesim teleop`, there is **no** `geniesim generator …` subcommand. The CLI treats this package as a pip-installable peer for `bootstrap` / `deploy` / `status` only; end-to-end use is through the skills.
- **`LLM_RESULT.py` is the LLM↔compiler contract.** The DSL program file must be importable as `LLM_RESULT` from the generator's app entry. Don't rename or relocate it; downstream consumers (benchmark `llm_task` integration) read this path.
- **Output layout: `benchmark/config/llm_task/<output_name>/`.** Generated scenes land here so the benchmark stack can pick them up as ordinary task configs. Don't write outputs elsewhere.
- **Asset library is shared.** Asset IDs returned by `search_assets` / `get_interactions` come from `ASSETS_INDEX`; the same IDs are referenced by hand-authored benchmark configs and by `geniesim_assets`. Renaming or removing an asset ID breaks downstream configs.
- **Two paths, one DSL.** Scenes can be authored via Open WebUI (LLM-in-the-loop, needs `deploy-generator`) **or** by Claude/agent writing the DSL program directly (no WebUI / no MCP server needed). Both paths compile through the same `geniesim_generator.app` — don't fork the compile pipeline.
