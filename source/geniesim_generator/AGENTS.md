# geniesim_generator — Agent Development Guide

LLM-driven scene generation for Genie Sim. An LLM (hosted in Open WebUI)
writes a small Python program in the **Scene Language DSL**; that program
builds a `Shape` tree from the shared asset library, and the package compiles
it into a `scene.usda` + layout graph that the benchmark/sim stack consumes.

Source: [source/geniesim_generator/](.)
License: [Mozilla Public License Version 2.0](LICENSE)
Skills (Claude Code / opencode): [skills/](skills/)
Repo-wide routing: [../AGENTS.md](../AGENTS.md) · package table: [../source/AGENTS.md](../AGENTS.md)

> **Maintenance contract** — when you add/rename a DSL primitive or `@register`
> helper, change the `LLM_RESULT.py` import contract, add an MCP tool/server,
> change the output directory layout, or touch the Open WebUI config exports,
> **update this file in the same diff.** Agents read this as the source of truth.

> **Not a CLI verb.** Unlike `geniesim benchmark`, there is **no** `geniesim
> generator …` subcommand. `geniesim_cli` only treats this package as a
> pip-installable peer (`bootstrap` / `deploy geniesim_generator` / `status`,
> gated behind the `[generator]` / `[full]` extras). All runtime entry points
> are plain `python` invocations and `docker compose` — see §4.

---

## 1. What this package is

A four-stage pipeline. The LLM produces DSL code; the package executes and
compiles it:

```
Open WebUI agent (geniesimscenegen)
   │  calls MCP tools on :8765 to discover assets + interactions
   ▼
LLM emits Python:  from helper import *  →  @register()… root_scene()
   │  "save" action writes it to → generator/LLM_RESULT.py
   ▼
app.py:  import LLM_RESULT.root_scene → run it → Shape tree
   │  helper.gen_scene_layout_info → (scene_info dict, networkx DiGraph)
   │  utils.usd.gen_scene_usda → scene.usda
   ▼
benchmark/config/llm_task/<scene_id>/<n>/{scene.usda, scene_info.json, graph.svg, LLM_RESULT.py}
   ▲
scene_viewer.py file-watches LLM_RESULT.py and live-reloads scene.usda into Isaac Sim
```

The DSL (`scene_language/`) is adapted from the public **Scene Language**
project (Zhang et al., CVPR 2025) — see [`scene_language/README.md`](src/geniesim_generator/scene_language/README.md)
and [`scene_language/LICENSE.md`](src/geniesim_generator/scene_language/LICENSE.md).
It has been specialized to drive Genie Sim's `ASSETS_INDEX` (from the
`geniesim_assets` package).

---

## 2. Path map

| Artifact | Location |
|---|---|
| Generator entry (`main`) | [`src/geniesim_generator/app.py`](src/geniesim_generator/app.py) |
| DSL execution surface (`from helper import *`) | [`src/geniesim_generator/helper.py`](src/geniesim_generator/helper.py) |
| LLM program slot / template | [`src/geniesim_generator/LLM_RESULT.py`](src/geniesim_generator/LLM_RESULT.py) |
| Isaac Sim live preview | [`src/geniesim_generator/scene_viewer.py`](src/geniesim_generator/scene_viewer.py) |
| Scene-Language DSL | [`src/geniesim_generator/scene_language/`](src/geniesim_generator/scene_language/) |
| USD serializer (`gen_scene_usda`) | [`src/geniesim_generator/utils/`](src/geniesim_generator/utils/) |
| MCP servers + RAG | [`src/geniesim_generator/server/`](src/geniesim_generator/server/) |
| Open WebUI config exports | [`src/geniesim_generator/config/`](src/geniesim_generator/config/) |
| Docker stack (MCP + Open WebUI) | [`src/geniesim_generator/compose.yaml`](src/geniesim_generator/compose.yaml) |
| Deploy / status integration | [`../geniesim_cli/src/geniesim_cli/commands/`](../geniesim_cli/src/geniesim_cli/commands/) (`deploy.py`, `status.py`, `bootstrap.py`) |

---

## 3. Routing map

- **Add/change a generator output** (USD, graph, json) → `app.py:main` + `helper.gen_scene_layout_info` + `utils/usd.py`
- **Add/change a DSL surface the LLM can call** → `helper.py` (re-exports + `usd()`, `get_*_info`) and `scene_language/`
- **DSL registration / lookup** (`register`, `library_call`, the `info["stack"]` frame) → `scene_language/dsl_utils.py`
- **Shape composition** (`concat_shapes`, `transform_shape`, bbox math) → `scene_language/shape_utils.py` + `_shape_utils.py`
- **Primitives** (`cube`/`sphere`/`cylinder`) → `scene_language/engine_utils.py` → `_engine_utils_exposed.py` (the only supported `ENGINE_MODE="exposed"`)
- **Spatial helpers** (`attach`, `align_with_*`) → `scene_language/calc_utils.py`; assertions → `assert_utils.py`
- **Offline Mitsuba rendering** → `scene_language/mi_helper.py` + `engine/utils/mitsuba_utils.py`
- **Asset search (RAG)** → `server/mcp_assets_server.py` → `server/assets_searcher/` (`AssetVectorDB` text, `AssetVectorDBVL` vision-language)
- **Asset interaction metadata** → `server/mcp_assets_info.py` (reads `ASSETS_INDEX.interaction.{active,passive}`)
- **LLM file writes** → `server/mcp_file_server.py` (MCP tool) and `server/save_to_local.py` / `config/save_data_gen.py` (Open WebUI actions)

---

## 4. Runtime entry points

```bash
# Generate one scene from the current LLM_RESULT.py
# (script-relative imports — run from the package dir, NOT via `python -m`)
cd src/geniesim_generator && PYTHONPATH=../.. python app.py --scene_id <id> [--task_gen]

# Live preview in Isaac Sim (watches LLM_RESULT.py, reloads on save)
python src/geniesim_generator/scene_viewer.py [--auto-play]

# MCP servers (stdio JSON-RPC; wired by server/mcp_config.json)
python src/geniesim_generator/server/mcp_assets_server.py
python src/geniesim_generator/server/mcp_assets_info.py
python src/geniesim_generator/server/mcp_file_server.py

# Full stack: MCP gateway (:8765) + Open WebUI, via Docker — pick ONE embedding profile (see §6)
# Prereq: pip install geniesim_assets on host, then derive GENIESIM_ASSETS_DIR
# from the running Python so compose can mount the host's installed copy.
export GENIESIM_ASSETS_DIR=$(python -c \
    "import geniesim_assets, os; print(os.path.dirname(geniesim_assets.__file__))")
docker compose --profile text up --build   # Qwen embedding API (no GPU; needs API key)
docker compose --profile vl   up --build   # Qwen3-VL embedding (local GPU + NVIDIA Container Toolkit)
```

`app.py` uses script-relative imports (`from helper import *`,
`from LLM_RESULT import root_scene`), so it must run with the package dir on
the path / as cwd. Outputs land under
`benchmark/config/llm_task/<scene_id>/<n>/`. See
[`server_readme.txt`](src/geniesim_generator/server_readme.txt) for text-vs-VL
deployment details.

---

## 5. Optional extras → what needs them

Install with `pip install -e source/geniesim_generator[<extra>]`.

| Extra | Pulls in | Required by |
|---|---|---|
| (core) | mitsuba, networkx, numpy, pillow, scipy, usd-core, geniesim_assets, … | `app.py`, `helper.py`, `scene_viewer.py`, `scene_language/*`, `utils/usd.py` |
| `mcp` | fastmcp, pydantic, aiofiles | all `server/mcp_*.py`, `server/save_to_local.py`, `config/save_data_gen.py` |
| `rag` | chromadb, openai, dashscope | `server/assets_searcher/assets_searcher.py`, `embeddings/text_embedding.py` (text mode, `EMBEDDING_MODE=text`) |
| `vl` | torch, typing-extensions | `server/assets_searcher/assets_searcher_vl.py`, `embeddings/vl_embedding.py` (default `EMBEDDING_MODE=vl`) |
| `full` | mcp + rag + vl | the entire `server/` stack |

Isaac Sim (`isaacsim.SimulationApp`, used by `scene_viewer.py`) is **not** a
pyproject dependency — it is provided by the external sim runtime.

---

## 6. Embedding backend — pick ONE before deploying

Asset RAG search (`server/mcp_assets_server.py`) needs an embedding backend.
The two `docker compose` profiles are mutually exclusive — both publish the
MCP gateway on `:8765`, so run only one at a time. Choose by what hardware /
credentials you have. Authoritative source: [`server_readme.txt`](src/geniesim_generator/server_readme.txt).

**Both profiles** need `geniesim_assets` `pip install`-ed on the host. Before
bringing the stack up, derive `GENIESIM_ASSETS_DIR` from the running Python so
compose mounts the host's installed copy into `/opt/geniesim_assets` (read-only;
the multi-GB asset library is **not** baked into the image). `entrypoint.sh`
adds `/opt` to `PYTHONPATH` to make the mount importable. `compose.yaml` errors
out fast if the variable is unset. No host paths are baked into the image.

```bash
export GENIESIM_ASSETS_DIR=$(python -c \
    "import geniesim_assets, os; print(os.path.dirname(geniesim_assets.__file__))")
```

### Option A — `text` profile (Qwen embedding **API**, no GPU)

Calls a remote embedding API (Dashscope's OpenAI-compatible endpoint). Use
this when you **don't have a GPU** but **can provide an API key**.

```bash
# 1. Put your key + endpoint in the text config
#    server/mcp_text_embedding/text_embedding_config.json
#      { "api_key": "<YOUR_DASHSCOPE_KEY>",
#        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
#        "dashscope_mode": true, "dimension": 2048, "model": "text-embedding-v4" }
# 2. From the generator package dir (GENIESIM_ASSETS_DIR already exported, see above):
docker compose --profile text up --build
```

- Pros: no GPU, fast to stand up. Cons: needs network + a paid API key; text-only (no image/video understanding of assets).
- Backend class: `AssetVectorDB`; embedder: `DashscopeTextEmbeddings` / `TextEmbeddings`.

### Option B — `vl` profile (Qwen **vision-language** embedding, **GPU required**)

Runs Qwen3-VL-Embedding locally on the GPU — no external API, and it embeds
asset **images** as well as text (better retrieval quality). Requires an
NVIDIA GPU **and the NVIDIA Container Toolkit**.

```bash
# 1. Download weights + upstream helper scripts (ModelScope by default; --huggingface to switch)
cd server/assets_searcher
bash download_model.sh            # 2B by default; also accepts 8B | ALL
#   → fetches Qwen3-VL-Embedding-2B + Qwen3-VL-Reranker-2B into ./models/
# 2. Tune services/mcp-server_vl/environment in compose.yaml for your card
#      BATCH_SIZE   (default 10 ≈ RTX 4090; larger = more VRAM, smaller = slower indexing)
#      USE_RERANKER (True/False; reranker improves quality but costs latency + VRAM)
# 3. From the generator package dir (GENIESIM_ASSETS_DIR already exported, see above):
docker compose --profile vl up --build
```

- Pros: fully local, image-aware, optional reranker. Cons: needs a CUDA GPU + container toolkit, VRAM for the model, weight download.
- Backend class: `AssetVectorDBVL`; embedder: `QwenVLEmbeddings` (+ optional Qwen3-VL-Reranker). The image (`nvcr.io/nvidia/pytorch`) already bundles torch/transformers/qwen-vl-utils — only the weights are downloaded.

### Server environment variables (set on the `vl`/`text` service in `compose.yaml`)

| Var | Profile | Effect |
|---|---|---|
| `EMBEDDING_MODE` | both | `vl` → `AssetVectorDBVL`; `text` → `AssetVectorDB` (set per profile, don't override) |
| `VL_EMBEDDING_MODEL` | vl | VL embedder weights, e.g. `Qwen3-VL-Embedding-2B` |
| `USE_RERANKER` | vl | Enable the Qwen3-VL reranker on top of VL search (`True`/`False`) |
| `PERMANENT_MODEL_IN_GPU` | vl | Keep the VL model resident (else evicted after ~5s idle) |
| `BATCH_SIZE` | vl | Index-build batch size; trade VRAM vs. speed |

ChromaDB cache is bind-mounted under `server_chromadb*` → `/tmp/chromadb_cache*`
(see `compose.yaml`). The `open-webui` service runs on host networking
regardless of profile and reads the tool servers at `http://localhost:8765`.

**vl cold-start caveats** (details + fixes in the [deploy-generator](skills/deploy-generator/SKILL.md) skill):
the first `vl` launch builds the full vector index on the GPU (can take tens of
minutes for the full library), during which `mcpo` times out connecting to
`assets-agent` and `/assets-agent/*` returns 404 — **expected; restart once
`Sync completed` is logged** and the cached index makes startup instant. The 2B model peaks
~16 GB VRAM; on a shared card lower `BATCH_SIZE` and set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid CUDA OOM. The
weights `models/` dir must sit inside `server/assets_searcher/` (covered by the
mount) — an out-of-tree symlink dangles in-container → `ModuleNotFoundError`.

---

## 7. Open WebUI config bundles (`config/`)

These are **exported Open WebUI artifacts**, not auto-loaded by the package —
import them into your Open WebUI instance.

| File | What it is |
|---|---|
| [`config/geniesimscenegen.json`](src/geniesim_generator/config/geniesimscenegen.json) | The scene-generator model (emits DSL programs) |
| [`config/geniesimassets.json`](src/geniesim_generator/config/geniesimassets.json) | The asset-search model (uses the MCP tools) |
| [`config/openwebui.json`](src/geniesim_generator/config/openwebui.json) | Workspace config: Dashscope endpoint + the three tool servers on `:8765` |
| [`config/function-save_code_to_file.json`](src/geniesim_generator/config/function-save_code_to_file.json) | Open WebUI function export of `server/save_to_local.py` |
| [`config/save_data_gen.py`](src/geniesim_generator/config/save_data_gen.py) | Open WebUI action: parse `root_scene()` from chat → write benchmark `data_gen.py` |

---

## 8. Key invariants

1. `app.py` always imports `root_scene` from `LLM_RESULT.py`. Keep that name
   and the `from helper import *` contract stable — the LLM and the templates
   depend on it.
2. The DSL stack frame is the bridge to layout: every `@register`-decorated
   function pushes `(func_name, uuid)` onto each returned shape's
   `info["stack"]`, and `helper.gen_scene_layout_info` walks exactly that. Do
   not change the frame shape without updating the walker.
3. `helper.usd(oid, keywords)` is the canonical way DSL code instantiates an
   `ASSETS_INDEX` entry. New asset-driven primitives go through it, carrying
   the asset id + keywords in `info`.
4. Only `ENGINE_MODE="exposed"` is implemented (`engine_utils.py` raises
   `NotImplementedError` otherwise). Don't reference other engine modes.
5. `scene_language.primitive_call` is an **unimplemented `Hole`** until
   `import geniesim_generator.scene_language.mi_helper` runs (`app.py` line 18 does
   this). Without that import it degrades to a placeholder and drops
   `info["stack"]` → `KeyError: 'stack'`. Always compile DSL programs through
   `app.py`; if you execute one elsewhere, import `mi_helper` first.
6. Asset metadata is owned by the `geniesim_assets` package (`ASSETS_INDEX`),
   not by this package — read it, don't fork it.

---

## 9. Skills

The [`skills/`](skills/) directory holds opencode/Claude-Code-style playbooks
for the common generator workflows. They're tool-agnostic markdown with YAML
frontmatter; symlink or copy them into `~/.claude/skills/` (or
`.opencode/skills/`) to enable auto-trigger.

| Skill | Trigger |
|---|---|
| [deploy-generator](skills/deploy-generator/SKILL.md) | "部署 generator", "deploy the scene generator", "启动资产检索服务", "start the MCP assets server" |
| [search-assets](skills/search-assets/SKILL.md) | "搜索资产", "find an asset", "search the asset library", "look up asset_id X" |
| [generate-scene](skills/generate-scene/SKILL.md) | "生成一个场景", "按需求生成场景", "绕过 webui 生成场景", "generate a scene", "make a scene with `<objects>`" |

The three chain naturally: **deploy-generator** (bring up `:8765` + Open WebUI)
→ **search-assets** (discover real asset ids) → **generate-scene** (write the
DSL program and compile `scene.usda`).

---

## 10. Do not

- Don't add a `geniesim generator` CLI verb to fake a runtime command — the
  package's contract is "pip peer + python entry points + docker." If a verb is
  genuinely wanted, add it deliberately in `geniesim_cli` and document it here.
- Don't hardcode remote hosts, API keys, or ports inside DSL code or configs —
  endpoints/keys come from env vars and the Open WebUI workspace config.
- Don't pin asset ids/bboxes by hand in DSL code — resolve them through
  `helper.usd()` / `get_*_info()` against `ASSETS_INDEX`.
- Don't put MCP/RAG/VL-only imports (`fastmcp`, `chromadb`, `torch`, …) into the
  core import path (`app.py` / `helper.py`) — keep them inside `server/` so the
  core generator installs without the heavy extras.
