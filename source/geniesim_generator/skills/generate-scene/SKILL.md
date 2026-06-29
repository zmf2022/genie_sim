---
name: generate-scene
description: >
  Turn a natural-language scene request into a Genie Sim scene — an LLM writes
  a Scene-Language DSL program (`LLM_RESULT.py`), and `geniesim_generator.app`
  compiles it into `scene.usda` + a layout graph under benchmark/config/llm_task/.
  Works either through the Open WebUI agent, OR by having Claude write the DSL
  program directly and run the compiler (no WebUI / no MCP server needed).
  Trigger: When the user asks to "生成一个场景", "按需求生成场景", "generate a scene",
  "make a scene with <objects>", "build a tabletop layout", "create scene.usda
  from a description", "直接写脚本生成场景", "绕过 webui 生成场景", or wants the
  generator to produce a scene from a prompt.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_generator:search-assets    # ground asset_ids before composing
  - geniesim_generator:deploy-generator # only if going through Open WebUI; skippable for direct DSL
inputs:
  - name: prompt
    desc: Natural-language description of the scene to generate
    required: true
  - name: output_name
    desc: Stem for the produced `LLM_RESULT.py` + `scene.usda`
    required: false
  - name: use_webui
    desc: Route through Open WebUI (true) vs author DSL directly (false)
    required: false
    default: "true"
outputs:
  - desc: "`scene.usda` + layout graph under `benchmark/config/llm_task/<output_name>/`, plus the `LLM_RESULT.py` source"
---

## When to Use

- User describes a scene in words and wants the generator to produce it
  (`scene.usda` + `scene_info.json` + layout graph).
- User has an `LLM_RESULT.py` (hand-written or LLM-produced) and wants to
  compile / preview it.
- User wants a quick one-shot scene **without** standing up Open WebUI / MCP —
  Claude writes the DSL program directly (Path B below).

Prerequisite **only for Path A** (the Open WebUI agent loop): the MCP stack +
Open WebUI are running (`deploy-generator` first). Path B needs just the package
importable + `ASSETS_INDEX` available — no servers.

Do **not** use for:
- Standing up the servers → `deploy-generator` skill.
- Just browsing assets → `search-assets` skill.

## The pipeline (what actually happens)

```
NL request
  │  ── Path A: Open WebUI "geniesimscenegen" agent (uses search_assets/get_interactions)
  │  ── Path B: Claude writes the DSL program directly (no WebUI / no MCP server)
  ▼
Python program:  from helper import *  →  @register()… def root_scene() -> Shape
  │  written to → src/geniesim_generator/LLM_RESULT.py
  ▼
cd src/geniesim_generator && python app.py  (imports LLM_RESULT.root_scene, runs it)
  │  gen_scene_layout_info → (scene_info, networkx graph)
  │  gen_scene_usda        → scene.usda
  ▼
benchmark/config/llm_task/<scene_id>/<n>/{scene.usda, scene_info.json, graph.dot, graph.svg, LLM_RESULT.py}
```

The program is the only handoff between "write" and "compile" — so **Path A and
Path B differ only in who writes it.** Path B (Claude writes it directly) needs
neither Open WebUI nor the MCP servers running; it only needs the package
importable and `ASSETS_INDEX` available.

## Workflow

### Step 1 — Produce `LLM_RESULT.py` from the request

Two ways; pick by whether the WebUI/MCP stack is up.

#### Path A — via the Open WebUI agent (the deployed loop)

Drive the **`geniesimscenegen`** agent (import `config/geniesimscenegen.json`;
MCP tools wired via `config/openwebui.json`). Describe the scene; the agent
searches the asset library, writes a DSL program, and its "save to file" action
drops it at `generator/LLM_RESULT.py`. Requires `deploy-generator` first.

#### Path B — Claude writes the program directly (no WebUI, no MCP)

When the servers aren't up (or you just want a one-shot scene), **write
`LLM_RESULT.py` yourself** and run the compiler. This is the lightweight path.

1. **Get real asset ids.** The program must reference ids that exist in
   `ASSETS_INDEX` — guessed ids raise `KeyError` in `helper.usd()`. Options:
   - If the MCP server is up, use the `search-assets` skill.
   - Otherwise query the index directly in Python:

     ```bash
     python -c "from geniesim_assets import ASSETS_INDEX; \
       import re; pat=re.compile('bottle', re.I); \
       print([k for k in ASSETS_INDEX if pat.search(k)][:20])"
     ```

2. **Write the program** to `src/geniesim_generator/LLM_RESULT.py` following the
   contract below. Build every object through `usd(oid, keywords)` / `library_call("usd", …)`
   and place with the DSL helpers (`transform_shape`, `translation_matrix`,
   `rotation_matrix`, `attach`, `align_with_*`, `concat_shapes`).

   Minimal real example (mirrors the shipped template):

   ```python
   from helper import *

   @register()
   def place_bottle(oid: str, position) -> Shape:
       shape = library_call("usd", oid=oid, keywords=["bottle", "drink"])
       # drop it so its center lands on `position`
       center = get_object_info(shape)["center"]
       return transform_shape(shape, translation_matrix(np.array(position) - center))

   @register()
   def root_scene() -> Shape:                 # REQUIRED entry point — app.py imports this name
       a = place_bottle("genie_beverage_bottle_007", (-0.32, -0.96, 1.11))
       b = place_bottle("genie_beverage_bottle_008", (-0.32, -0.70, 1.11))
       return concat_shapes(a, b)
   ```

   Keep one `@register()` on each builder (the decorator pushes the layout
   stack frame `gen_scene_layout_info` walks) and exactly one `root_scene()`.

3. Proceed to Step 2 to compile.

The program must follow the contract (see the shipped `LLM_RESULT.py` template):

```python
from helper import *

@register()
def place_mug() -> Shape:
    ...                       # build from usd(asset_id) + transform/concat helpers

@register()
def root_scene() -> Shape:    # REQUIRED entry point — app.py imports this name
    return place_mug()
```

If the user supplies their own program, **overwrite the live slot**
`src/geniesim_generator/LLM_RESULT.py` with it (back up the original first).
This is the one reliable way to feed a program in — see the `--template_path`
caveat in Step 2.

### Step 2 — Compile the scene

```bash
# Run from the package dir — app.py uses script-relative imports
# (`from helper import *`, `from LLM_RESULT import root_scene`), so it is NOT
# launchable as `python -m geniesim_generator.app`.
cd source/geniesim_generator/src/geniesim_generator
PYTHONPATH=../.. python app.py --scene_id <my_scene>
```

Flags:

| Flag | Effect |
|---|---|
| `--scene_id <id>` | Output dir name under `benchmark/config/llm_task/`. If omitted, derived from the scene graph root. |
| `--template_path <py>` | Copy this file into `<repo-layout>/generator/LLM_RESULT.py` before running. **Caveat:** the target is `dirname(dirname(app.py))/generator/LLM_RESULT.py`, which only exists in the *deployed* layout (`…/geniesim/generator/app.py`). In an editable **source** checkout (`…/src/geniesim_generator/app.py`) that path is `…/src/generator/` and does **not** exist → `FileNotFoundError`. In a source checkout, **don't** use this flag; just overwrite `LLM_RESULT.py` directly (Step 1). |
| `--task_gen` | Also run task generation. |

Outputs land in `benchmark/config/llm_task/<scene_id>/<n>/` (`<n>` auto-increments
per run): `scene.usda`, `scene_info.json`, `graph.dot`, `graph.svg`, and a
snapshot of the `LLM_RESULT.py` that produced it. On success it prints
`step3: save scene to <path>...`.

### Step 3 — Preview live in Isaac Sim (optional)

```bash
python src/geniesim_generator/scene_viewer.py [--auto-play]
```

`scene_viewer` watches `LLM_RESULT.py`; on every save it re-runs the generator
(via `run_generator.sh`, alongside `app.py`), parses the printed scene path, and
reloads `scene.usda` under `/World`. Edit the program → save → watch it update.
Needs Isaac Sim available in the environment.

## Tips

- `root_scene()` is the hard entry point — `app.py` always imports that exact
  name. Keep it.
- **Always compile via `app.py`, never `python LLM_RESULT.py` directly.**
  `primitive_call` is an unimplemented `Hole` until `app.py` runs
  `import geniesim_generator.scene_language.mi_helper` (its line 18) — that call
  is what implements the primitives. Run the program any other way and
  `primitive_call` silently degrades to a placeholder that drops `info["stack"]`,
  giving `KeyError: 'stack'`. If you ever execute a DSL program outside `app.py`
  (e.g. a quick unit check), `import geniesim_generator.scene_language.mi_helper`
  first.
- Build objects through `usd(asset_id, keywords)` so positions/bboxes resolve
  against `ASSETS_INDEX` — don't hand-pin coordinates. Use `attach` /
  `align_with_*` (in `scene_language/calc_utils.py`) for relative placement.
- Get real `asset_id`s from the `search-assets` skill before writing the program;
  guessed ids won't resolve in `ASSETS_INDEX`.
- Only `ENGINE_MODE="exposed"` primitives exist (`cube` / `sphere` / `cylinder`);
  everything else is composed from those + asset USDs.
- Inspect `graph.svg` to sanity-check the object relationship DAG the layout
  produced.

## Resources

- App entry: [src/geniesim_generator/app.py](../../src/geniesim_generator/app.py)
- DSL surface (`from helper import *`): [src/geniesim_generator/helper.py](../../src/geniesim_generator/helper.py)
- Program template / slot: [src/geniesim_generator/LLM_RESULT.py](../../src/geniesim_generator/LLM_RESULT.py)
- Live preview: [src/geniesim_generator/scene_viewer.py](../../src/geniesim_generator/scene_viewer.py)
- Scene-Language DSL: [src/geniesim_generator/scene_language/](../../src/geniesim_generator/scene_language/)
- Scene-gen agent: [config/geniesimscenegen.json](../../src/geniesim_generator/config/geniesimscenegen.json)
