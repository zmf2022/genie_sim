# geniesim_generator — LLM-driven scene generation 🏗️

A scene-language DSL + LLM tooling that lets a model write a small
Python program that builds a `Shape` tree from a shared asset library,
which the package compiles into a `scene.usda` + layout graph the
Genie Sim stack can load.

License: [Mozilla Public License Version 2.0](LICENSE)
Agent doc: [`AGENTS.md`](AGENTS.md)
Skills: [`skills/`](skills/)

---

## 📦 Install

`geniesim_generator` is **gated behind extras** because its deps are
heavy (LLM / ML stacks). Install via:

```bash
pip install -e "source/geniesim/[generator]"
# or directly:
pip install -e source/geniesim_generator/
```

`geniesim bootstrap` asks before installing — default is skip.

---

## 🛠️ What it does

- **Scene Language DSL** — a small Python surface (`Shape`, primitives,
  `@register` helpers) that an LLM can program against.
- **LLM in the loop** — Open WebUI hosts the LLM, prompts ship an
  `LLM_RESULT.py` script back, the package executes it and emits a
  USD + layout graph.
- **Asset search** — VL-based queries into the shared asset library so
  the DSL can name real objects, not abstract placeholders.

> ⚠️ **Not a CLI verb.** Unlike `geniesim benchmark`, there's no
> `geniesim generator …` subcommand. `geniesim_cli` treats this
> package as a pip-installable peer (`bootstrap` / `deploy
> geniesim_generator` / `status`); end-to-end use happens via the
> skills below.

---

## 🤖 Skills

| Skill | Purpose |
|---|---|
| [generate-scene](skills/generate-scene/SKILL.md) | Drive the LLM to emit a new scene from a natural-language description |
| [search-assets](skills/search-assets/SKILL.md) | Query the asset library for objects matching a description |
| [deploy-generator](skills/deploy-generator/SKILL.md) | Build / ship the generator stack (Open WebUI + MCP servers) |

---

## 🔗 Pointers

- 🗺️ Module map: [`../README.md`](../README.md)
- 🏠 Repo root: [`../../README.md`](../../README.md)
- 🤖 Agent guide: [`AGENTS.md`](AGENTS.md)
