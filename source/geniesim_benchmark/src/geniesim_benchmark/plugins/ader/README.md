# 🧠 Benchmark Evaluation Framework

Use ADER (Action Domain Evaluation Rule) for evaluation configuration

---

## 📦 What is ader?

ADER (Action Domain Evaluation Rule) is a declarative framework for expressing a
task's success criteria. An evaluation is described as a tree of *actions* —
sequencing primitives such as `ActionList`, `ActionSetWaitAny` and
`ActionSetWaitAll`, combined with domain predicates such as `Ontop`, `Inside` and
`PickUpOnGripper`. The tree is loaded from human-readable JSON and evaluated each
step against the simulator state to decide whether a task has succeeded, failed,
or is still in progress.

---

## 🚀 How it is used

Each scene instance carries a `problems.json` (the eval config). At runtime the
parser ([`action/action_parsing.py`](action/action_parsing.py) → `parse_action`)
turns `problems["problem<i>"]["Acts"][0]` into the action tree; that file is the
**single source of truth** for which checkers exist and how their arguments are
parsed.

Two ways to author the config:

1. **Hand-write** `problems.json` for a scene instance (the `add-benchmark-task`
   skill walks through the full task layout).
2. **Auto-generate** with the LLM evaluator:
   ```bash
   python -m geniesim_benchmark.evaluator.generators.eval_gen --scene_dir <llm_task/<task>/<i>>
   ```
   This feeds [`../../evaluator/prompts/ADER_SYSTEM_PROMPT.md`](../../evaluator/prompts/ADER_SYSTEM_PROMPT.md)
   (the checker spec) + `PROMPT_EVAL.txt` + the scene's `instructions.json` to a
   model and writes `problems.json`. Keep that spec in sync with the checkers below.

> When you add a checker, register its key in `action_parsing.py`, document it in
> this table, **and** add it to `ADER_SYSTEM_PROMPT.md` (otherwise the generator
> never emits it).

---

## ⚙️ Example Configuration
```json
{
  "Acts": [
    {
      "ActionList": [
        {
          "ActionSetWaitAny": [
            { "Follow": "beverage_bottle_002|[0.2,0.2,0.2]|right" },
            { "Onfloor": "beverage_bottle_002|0.0" }
          ]
        },
        {
          "ActionSetWaitAny": [
            { "PickUpOnGripper": "beverage_bottle_002|right" },
            { "Timeout": 120 },
            { "Onfloor": "beverage_bottle_002|0.0" }
          ]
        },
        {
          "ActionSetWaitAny": [
            { "Inside": "beverage_bottle_002|handbag_000|1" },
            { "StepOut": 1000 }
          ]
        }
      ]
    }
  ],
  "Init": [],
  "Objects": [],
  "Problem": "pack_in_the_supermarket"
}
```

---

## 🧱 Control & composite actions

These build the tree. `ActionList` sequences children; the `WaitAny/All/Some`
sets run children in parallel and decide completion. `Timeout`/`StepOut` are
**cancel actions** — when they fire they set `SCORE = 0` and cancel the whole
evaluation, so they are the standard "give up" guards inside a `WaitAny`.

| Action | Description | Value form |
|--------|-------------|------------|
| ActionList | Run children strictly in order; finish when the last one finishes. | `[ {…}, {…} ]` |
| ActionSetWaitAny | Finish as soon as **any** child finishes (others are stopped). | `[ {…}, {…} ]` |
| ActionSetWaitAll | Finish only when **all** children finish. | `[ {…}, {…} ]` |
| ActionSetWaitSome_N | Finish when at least **N** children finish (key suffix is the count, default 1). | `[ {…}, {…} ]` |
| ActionWaitForTime | Non-blocking sleep for N seconds, then finish. | raw number, e.g. `3.0` |
| Timeout | **Cancel** when accumulated time exceeds N seconds (`SCORE=0`). | raw number, e.g. `120` |
| StepOut | **Cancel** when step count since start exceeds N steps (`SCORE=0`). | raw number, e.g. `1000` |

---

## 🧩 Evaluation predicates

Pipe-separated arguments unless stated otherwise. `[opt]` = optional with the default shown.
"Frames" = consecutive simulation frames that must hold before the check passes.

### Spatial relationship

| Predicate | Syntax | Details |
|-----------|--------|-------|
| Ontop | `"active_obj\|passive_obj"` | Active sits on passive: vertical gap ≤ 0.02 m and XY overlap ≥ 50% (1 frame). |
| Inside | `"active_obj\|passive_obj\|scale"` | Active center inside passive AABB scaled by `scale` (2 frames). |
| InBBox | `"obj_id\|cx,cy,cz\|lx,ly,lz"` | Object inside a fixed world AABB (center + full side lengths) (2 frames). |
| Cover | `"active_obj\|passive_obj"` | Active covers passive: active CoG ≤ passive top + 0.002 m and XY overlap ≥ 50% (1 frame). |
| Stack | `"[a,b,c]\|[x,y]"` | ≥2 ids; every object's XY center within `(x,y)` of the first (default 0.05 m each) (2 frames). |
| OnShelf | `"obj\|target\|[xmin,xmax,ymin,ymax,zmin,zmax]\|height"` | Object pose relative to `target` (z minus `height`) inside the offset box (>5 frames). |
| Onfloor | `"obj\|ref_z"` | **Cancel**: object dropped to within 0.3 m of `ref_z` (`SCORE=0`). |
| Upright | `"obj\|tilt_deg[\|allow_flipped]"` | Object +Y axis within `tilt_deg` of world +Z; `allow_flipped=true` also accepts −Z. Default tilt 15°, allow_flipped false (1 frame). |
| RelativePosition | `"objA\|objB\|relation[\|threshold]"` | A vs B in robot body frame. `relation` ∈ `leftof,rightof,topof,bottomof,aligned_x,aligned_y,aligned_z`. `threshold` (default 0.05) only used by `aligned_*` (2 frames). |

### Grasping & manipulation

| Predicate | Syntax | Details |
|-----------|--------|-------|
| PickUpOnGripper | `"object\|gripper_id"` | Object lifted ≥0.02 m above start and within 0.2 m of the gripper. `object` may be comma-separated (any). `gripper_id` contains `right`→right else left. |
| StableGrasp | `"object\|gripper_id[\|dist[\|pos_diff[\|rot_diff_rad]]]"` | Object stays near the gripper and moves rigidly with it. Defaults: dist 0.1 m, pos_diff 0.02 m, rot_diff 0.1 rad (2 frames). |
| Follow | `"object\|[x,y,z]\|gripper_id"` | Gripper center inside a box of size `[x,y,z]` around the object. `object` may be comma-separated (any) (1 frame). |
| Approach | `"x\|y\|z"` | Right gripper center within 0.01 m (per axis) of target point (1 frame). |
| GripperPassing | `"obj_prim\|reverse"` | Handover detected (left→right, or right→left if `reverse=true`) with the object on the line between grippers. |
| LiftUp | `"object\|lift_threshold"` | Object Z risen ≥ `lift_threshold` above its first Z (default 0.05 m) (1 frame). |

### Joint, state & navigation

| Predicate | Syntax | Details |
|-----------|--------|-------|
| PushPull | `"obj\|min\|max[\|joint_index]"` | Prismatic joint `joint_index` (default 0) position within `[min,max]` — drawer open/closed (2 frames). |
| TriggerAction | `"prim_path\|expected"` | `get_trigger_action(prim_path)` equals `expected` (1 frame). |
| ChassisAtTarget | `"[x,y,yaw]\|[x_th,y_th,yaw_th]"` | Chassis within XY box and yaw threshold of target (yaw in degrees) (1 frame). |

### Special scenarios

| Predicate | Syntax | Details |
|-----------|--------|-------|
| FluidInside | `"container\|object_info_dir"` | >50 fluid particles inside the container's AABB (size read from `object_info_dir`). |
| CheckParticleInBBox | `"threshold\|xmin,ymin,zmin,xmax,ymax,zmax"` | Particle count inside the box drops **below** `threshold`. |
| CheckStainClean | `"stain_prim\|threshold"` | Visible stain meshes drop **below** `threshold`. |
| PlaceOnRivet | `"active\|passive\|rx,ry,rz\|qw,qx,qy,qz\|[xy_tol]\|[z_tol]\|[orient_tol]\|[still_thresh]\|[still_steps]"` | Workpiece reaches target relative pose vs workspace and holds still. Defaults: xy_tol 0.02 m, z_tol 0.01 m, orient_tol 0.15 rad, still_thresh 0.02, still_steps 15. |
| VLM | `"task_id\|interval"` | A VLM scorer judges the `task_id` description every `interval` updates; emits a fractional score (passes at ≥1.0). |

### Meta

| Predicate | Syntax | Details |
|-----------|--------|-------|
| MixedRules | JSON object (below) | Aggregate several sub-checkers with OR (partial credit) or AND (all required). |

```json
{
  "MixedRules": {
    "rules": [
      { "name": "Inside", "params": "cup|tray|1.0" },
      { "name": "Upright", "params": "cup|15" }
    ],
    "check_interval": 1,
    "mode": "or"
  }
}
```
`mode: "or"` scores `passed/total` (e.g. 0.5); `mode: "and"` scores 1.0 only when
all pass. Supported sub-checker `name`s: `Inside, PushPull, Upright,
RelativePosition, Ontop, Onfloor, Cover, LiftUp, InBBox, Stack, StableGrasp`.

---

## 📝 Remarks

- **Argument format**: object ids are used directly (`beverage_bottle_002`);
  bbox/list args are list literals (`[0.2,0.2,0.2]`); numbers and booleans
  (`true`/`false`) are plain strings inside the pipe value.
- **Placeholders**: many object args accept `{@placeholder_name}`, resolved at
  runtime (e.g. propagating the object a previous `PickUpOnGripper` grabbed).
- **Cancel vs success**: `Timeout`, `StepOut`, and `Onfloor` are cancel actions —
  pair them with success predicates inside `ActionSetWaitAny` to express
  "succeed, or give up on timeout/drop".
- **Key vs class names**: JSON key `Timeout` → class `TimeOut`; key
  `RelativePosition` → class `RelativePositionChecker`.
</content>
</invoke>
