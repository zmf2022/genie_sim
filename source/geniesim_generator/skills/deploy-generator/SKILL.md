---
name: deploy-generator
description: >
  Stand up the geniesim_generator scene-generation stack — the MCP asset
  servers + Open WebUI — via `docker compose`, picking one embedding backend.
  Trigger: When the user asks to "部署 generator", "deploy the scene generator",
  "启动资产检索服务", "start the MCP assets server", "run the generator stack",
  "set up open-webui for scene gen", or otherwise wants the generator's
  Docker services (`compose.yaml`, profiles `text` / `vl`) running.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites: []
inputs:
  - name: profile
    desc: "Embedding backend profile (`text` or `vl`)"
    required: true
  - name: host
    desc: Host to bind WebUI / MCP servers
    required: false
    default: "0.0.0.0"
outputs:
  - desc: "Open WebUI + MCP assets servers running (default ports); `docker compose ps` shows them healthy"
---

## When to Use

- User wants the scene-generation backend running: the MCP asset/file servers
  (gateway on `:8765`) plus the Open WebUI front-end.
- User asks which embedding backend to pick, or how to configure the API key /
  GPU model for asset retrieval.

Do **not** use for:
- Actually searching assets once the server is up → `search-assets` skill.
- Driving the LLM to produce a scene → `generate-scene` skill.

## The ONE decision: which embedding backend

Asset RAG search needs an embedding backend. The two `docker compose` profiles
both bind the gateway to `:8765`, so **only one runs at a time**. Choose by
hardware / credentials:

| | `text` profile | `vl` profile |
|---|---|---|
| Embedder | Qwen embedding **API** (Dashscope `text-embedding-v4`) | Qwen3-VL-Embedding **local model** |
| Hardware | **No GPU** | **NVIDIA GPU + Container Toolkit** |
| Credentials | **Needs an API key** | None (runs offline after weight download) |
| Modality | Text only | Image + text (better retrieval) |
| Extras | reranker N/A | optional Qwen3-VL reranker |

Ask the user (via `AskUserQuestion`) which they want if it isn't obvious from
context (do they have a GPU? do they have a Dashscope key?).

## Workflow

All commands run from the generator package dir
(`source/geniesim_generator/src/geniesim_generator/`, where `compose.yaml` lives).

### Prerequisites (both profiles)

The MCP servers `import geniesim_assets` (the multi-GB object library, shipped
as a **separate** package — not bundled in this image). It is now installed
on the **host** via `pip install geniesim_assets`; the compose stack mounts
the host's installed copy into each MCP container read-only (and
`entrypoint.sh` adds `/opt` to `PYTHONPATH` so the mount is importable).

So before bringing the stack up, point `GENIESIM_ASSETS_DIR` at the package
directory on the host — derive it from the running Python rather than hard-coding:

```bash
export GENIESIM_ASSETS_DIR=$(python -c \
    "import geniesim_assets, os; print(os.path.dirname(geniesim_assets.__file__))")
```

If unset (or pointing somewhere bogus), compose fails fast with a clear message
(no silent half-broken start). No paths are baked into the image.

### Option A — `text` (API, no GPU)

1. Edit `server/mcp_text_embedding/text_embedding_config.json` — fill in `api_key`
   (and confirm `base_url` / `model` / `dimension`):

   ```json
   { "api_key": "<YOUR_DASHSCOPE_KEY>",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "dashscope_mode": true, "dimension": 2048, "model": "text-embedding-v4" }
   ```

2. Launch (assumes `GENIESIM_ASSETS_DIR` is already exported — see Prerequisites):

   ```bash
   docker compose --profile text up --build
   ```

### Option B — `vl` (local GPU model)

1. Download the model weights + upstream helper scripts into the package
   (`server/assets_searcher/models/`); the container reads them via the mount:

   ```bash
   cd server/assets_searcher
   bash download_model.sh            # 2B by default; also accepts: 8B | ALL
   #   --huggingface to fetch from Hugging Face instead of ModelScope
   ```

   The `vl` image is based on `nvcr.io/nvidia/pytorch` and already bundles
   torch / transformers / qwen-vl-utils — you do **not** pip-install those
   yourself; only the weights are fetched here.

2. Tune `services/mcp-server_vl/environment` in `compose.yaml` for your card:
   - `BATCH_SIZE` — index-build batch (default `10` ≈ RTX 4090; larger = more VRAM, smaller = slower).
   - `USE_RERANKER` — `True`/`False`; reranker improves quality, costs latency + VRAM.
   - `PERMANENT_MODEL_IN_GPU` — keep model resident vs. evict after idle.

3. Launch (assumes `GENIESIM_ASSETS_DIR` is already exported — see Prerequisites):

   ```bash
   docker compose --profile vl up --build
   ```

### Verify it's up

- MCP gateway answers on `http://localhost:8765` with three tool routes:
  `/assets-agent` (`search_assets`), `/assets-info-agent` (`get_interactions`),
  `/file-agent` (`save_file`). Check a route is live:
  `curl -s localhost:8765/assets-agent/openapi.json | python3 -m json.tool | grep paths`
  — non-empty `paths` means `assets-agent` registered.
- Open WebUI is on host networking (`WEBUI_AUTH=False`) — open it in a browser
  and import the configs from `config/` (see `generate-scene` skill).

## vl troubleshooting (read before first launch)

The `vl` profile has two startup gotchas — both verified in practice:

1. **First launch builds the full vector index, and it's slow.** The VL backend
   decodes each asset's preview video + embeds it on the GPU — this can take
   **on the order of tens of minutes** for the full library (vs. ~minute-scale
   for `text`). During this window
   `mcpo`'s handshake to `assets-agent` **times out**, so the startup summary
   logs `Failed to connect to: assets-agent` and `/assets-agent/search_assets`
   returns `404`. **This is expected on the cold run.** The subprocess keeps
   writing `server_chromadb_vl/chroma.sqlite3` to completion; once you see
   `Sync completed, current asset count: N`, **restart the stack**. The second
   start finds the index unchanged, `sync` returns instantly, and `assets-agent`
   registers cleanly. (`assets-info-agent` / `file-agent` are light and always
   register on the first try — only `assets-agent` is gated by index build.)

2. **CUDA out of memory.** The 2B model + embedding peaks around **~16 GB VRAM**.
   On a 24 GB card shared with other GPU work you'll hit
   `torch.OutOfMemoryError`. Mitigations (set in `compose.yaml`
   `mcp-server_vl.environment`):
   - lower `BATCH_SIZE` (e.g. `4`) — smaller VRAM peak, slower indexing;
   - add `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation;
   - keep `USE_RERANKER=False` (reranker loads a second model);
   - or free other GPU processes first (`nvidia-smi` to see what's resident).

## Notes

- First `vl` startup builds the ChromaDB index (slow — see troubleshooting);
  subsequent runs reuse the bind-mounted cache
  (`server_chromadb_vl/` ↔ `/tmp/chromadb_cache_vl`). Delete that dir to force a rebuild.
- The weights `models/` dir must live **inside** `server/assets_searcher/`
  (where `download_model.sh` puts it) so it's covered by the `../..` mount and
  visible in-container. A symlink pointing outside the mounted tree will be
  **dangling inside the container** → `ModuleNotFoundError: …assets_searcher.models`.
- `text` mode needs outbound network to Dashscope; `vl` mode needs the weights
  present under `server/assets_searcher/models/` before launch.
- Don't run both profiles at once — they collide on port `8765`.

## Resources

- Compose stack: [compose.yaml](../../src/geniesim_generator/compose.yaml)
- Deployment notes: [server_readme.txt](../../src/geniesim_generator/server_readme.txt)
- Weight downloader: [server/assets_searcher/download_model.sh](../../src/geniesim_generator/server/assets_searcher/download_model.sh)
- Module guide: [../../AGENTS.md](../../AGENTS.md) §6 (embedding backend)
