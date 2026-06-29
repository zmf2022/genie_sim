---
name: search-assets
description: >
  Search the Genie Sim asset library by natural-language keyword via the
  generator's `search_assets` MCP tool (RAG over ASSETS_INDEX), and look up
  asset interaction metadata via `get_interactions`.
  Trigger: When the user asks to "搜索资产", "find an asset", "查一下有没有...的模型",
  "search the asset library", "what assets match <description>", "look up
  asset_id X", or wants to know which assets exist before generating a scene.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_generator:deploy-generator
inputs:
  - name: query
    desc: Natural-language description of the asset
    required: true
  - name: top_k
    desc: Max number of matches to return
    required: false
    default: "10"
  - name: asset_id
    desc: Skip search; just fetch interaction metadata for this id (uses `get_interactions`)
    required: false
outputs:
  - desc: Ranked list of `asset_id` candidates with their interaction metadata
---

## When to Use

- User wants to discover assets that match a description / keyword.
- User has an exact `asset_id` and wants its record.
- User wants the interaction structure (active/passive affordances) of one or
  more assets before placing them in a scene.

Prerequisite: the generator MCP stack is running (`:8765`). If it isn't, run
the `deploy-generator` skill first.

Do **not** use for:
- Bringing the server up → `deploy-generator` skill.
- Producing a full scene program → `generate-scene` skill.

## The tools (served on `http://localhost:8765`)

| Route | Tool | Purpose |
|---|---|---|
| `/assets-agent` | `search_assets` | RAG keyword search over the asset library |
| `/assets-agent` | `search_asset_by_asset_id` | Exact lookup by `asset_id` |
| `/assets-info-agent` | `get_interactions` | Active/passive interaction metadata for given asset ids |

### `search_assets` arguments

| Arg | Meaning |
|---|---|
| `keyword` (required) | What to search for; **the more detailed, the better the match**. |
| `topk` | Number of results (default 10). |
| `exclude_regex` | Drop assets whose `asset_id` matches (e.g. `omni6D.*`, `omni6D.*|sft.*`). |
| `include_regex` | Keep only assets whose `asset_id` matches. |
| `scene_description` | One-line scene summary to bias retrieval (e.g. "bottles of soda on a table"). |

Each result row carries `asset_id`, `semantic_name`, `full_description`, etc.

### `get_interactions` arguments

| Arg | Meaning |
|---|---|
| `assets_ids` (required) | List of asset ids to introspect. |
| `just_structure` | `True` (default) collapses leaf values to placeholders — use it to see the *shape* of the affordances without the full payload. |

## How to use

Normal path: drive these through the Open WebUI **`geniesimassets`** agent —
import `config/geniesimassets.json`, then chat ("find me a few ceramic mugs,
exclude omni6D"). The agent calls `search_assets` and shows the rows.

Programmatic / smoke-test path (an MCP client, or a quick HTTP call against the
mcpo gateway):

```bash
# search by keyword
curl -s http://localhost:8765/assets-agent/search_assets \
  -H 'Content-Type: application/json' \
  -d '{"keyword": "ceramic coffee mug with handle", "topk": 5,
       "exclude_regex": "omni6D.*", "scene_description": "mugs on a kitchen table"}'

# exact id lookup
curl -s http://localhost:8765/assets-agent/search_asset_by_asset_id \
  -H 'Content-Type: application/json' -d '{"asset_id": "<ASSET_ID>"}'

# interaction structure
curl -s http://localhost:8765/assets-info-agent/get_interactions \
  -H 'Content-Type: application/json' \
  -d '{"assets_ids": ["<ID_A>", "<ID_B>"], "just_structure": true}'
```

(The mcpo gateway publishes each MCP tool as an HTTP endpoint; exact request
schema is visible at `http://localhost:8765/docs`.)

## Tips

- **Be specific in `keyword`.** "red ceramic mug with a handle" beats "cup".
- Use `exclude_regex` / `include_regex` to constrain to (or away from) an asset
  family by id prefix — e.g. `include_regex: "objaverse.*"`.
- `scene_description` is a soft hint, not a filter; it nudges ranking.
- The backend (text API vs. VL model) is whatever profile `deploy-generator`
  brought up; the tool surface is identical either way, but VL search also
  understands asset images.

## Resources

- Search server: [server/mcp_assets_server.py](../../src/geniesim_generator/server/mcp_assets_server.py)
- Interaction server: [server/mcp_assets_info.py](../../src/geniesim_generator/server/mcp_assets_info.py)
- RAG backends: [server/assets_searcher/](../../src/geniesim_generator/server/assets_searcher/)
- Open WebUI agent: [config/geniesimassets.json](../../src/geniesim_generator/config/geniesimassets.json)
