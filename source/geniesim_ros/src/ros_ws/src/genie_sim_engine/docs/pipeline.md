# Prepare-stage pipeline

Before any engine script is launched, a two-stage **assemble pipeline**
prepares the robot USD and the scene manifest. The physics node itself is a
pure consumer — it only reads `manifest.json` and references the
already-prepared `robot.usda` into the stage.

## Pipeline overview

The launcher orchestrates three stages chained by `on_exit`, gated by a
stage cache (see `make_assemble_pipeline` in
[genie_sim_bringup/launch/utils.py](../../genie_sim_bringup/launch/utils.py),
called from `physics_isaacsim.launch.py` / `physics_mujoco.launch.py`,
which are composed by [app.launch.py](../../genie_sim_bringup/launch/app.launch.py)):

```
[assemble_robot]  --on_exit-->  [assemble_scene]  --on_exit-->  [genie_sim_engine]
   (URDF -> USD)                  (manifest.json)                 (physics loop)
```

All three stages share a `stage_dir` derived from the scene YAML's filename:

```
./assets/scenes/<scene_yaml_stem>/
    robot.urdf         <- intermediate (assemble_robot output)
    robot.usda         <- physics-grade robot USD
    render_layer.usda  <- cameras / RenderProducts
    manifest.json      <- what the engine scripts actually read
    pkg/<pkg>/...      <- symlinks for package:// URI rewriting
```

## Two routes: scene YAML decides

The branch is selected by a **single** signal — presence of
`robot.robot_source.urdf` in the scene YAML
(see `needs_assemble_robot()` in
[utils.py](../../genie_sim_bringup/launch/utils.py)):

| Condition in `robot.robot_source` | Path taken | Where `robot.usda` comes from |
| --- | --- | --- |
| `urdf:` key **absent** | **Legacy** | Pre-baked asset at `./assets/robot/<robot_name>/robot.usda`. `assemble_robot` is a no-op. |
| `urdf:` key **present** (value may be `{}`) | **URDF -> USD** | Generated from xacro by `assemble_robot.py`. |

Reserved keys on `robot.robot_source`:

- `package` (default: `genie_sim_robot_model`) — ROS 2 package providing the
  xacro (URDF route) or the legacy pre-baked USD assets (legacy route).
- `fix_base` (default: `true`) — whether the URDF root link is welded to world.
  Decided at **runtime**, not at bake time — see [fix_base](#fix_base-runtime-toggle).

The `urdf:` block accepts these reserved nested keys (URDF route only):

- `xacro_relpath` (default: `xacro/robot.xacro`) — entry path inside the package share dir.
- `mimic` — propagated to the manifest for `PhysxMimicJointAPI`, **not** forwarded to xacro.

Any other string keys under `urdf:` are forwarded as additional xacro mappings.

## Stage cache

Before either assemble stage runs, the launcher inspects `manifest.json`:

- **HIT** — skip both assemble stages. Reuse the existing `robot.usda`.
- **MISS** — run the assemble stages (and `mkdir -p stage_dir`).
- **OVERRIDE** — `always_regenerate_robot_usd:=true` forces re-run even on a hit.

```bash
# force a fresh build
rm -rf assets/scenes/<scene_stem>/
# or
ros2 launch genie_sim_bringup app.launch.py always_regenerate_robot_usd:=true
```

## Debug snapshots written every launch

- **`<stage_dir>/scene.yaml`** — verbatim copy of the input scene yaml, written before
  cache hit/miss decision. Use `diff` against the source yaml to reveal operator-edited drift.
- **`<stage_dir>/robot_runtime.usda`** — thin override layer exported by `IsaacSimStage`
  after `_init_articulation`. Debug-only; `assemble_scene.py` still references `robot.usda`.

## `assemble_robot.py` — URDF → USD

[assemble_robot.py](../scripts/assemble_robot.py) does this in five steps:

1. **Read scene YAML.** If `robot.robot_source.urdf` is absent, log and exit (no-op).
2. **Build xacro mappings** from `robot_source`: `robot_model`, `arm`, `body`, plus
   `gripper` → `variant`. Extra string keys under `urdf:` are forwarded as-is (except
   reserved `xacro_relpath`, `mimic`).
3. **Process xacro** → writes `robot.urdf` into `stage_dir`.
4. **Rewrite `package://` URIs** → symlinks under `<stage_dir>/pkg/<pkg>`, rewrites all URIs.
5. **Drive Isaac Sim's URDF importer** (`_convert_urdf_to_usd`):
   - **6.0+** → `urdf_usd_converter.Converter` + `importer_utils` post-passes + AS3 transformer.
   - **4.x / 5.x** → boots `SimulationApp({"headless": True})`, uses `URDFParseAndImportFile`.
   - Both converge on `<stage_dir>/robot.usda`.

### Cross-version invariants

Three design rules keep all three pipelines working with the same runtime code:

1. **Body discovery is hierarchy-walked.** `_collect_body_paths` in
   [`kit/stage.py`](../scripts/kit/stage.py) walks the full subtree via `Usd.PrimRange`
   and filters by `HasAPI(RigidBodyAPI)` — layout-agnostic.

2. **`/tf_render` carries absolute prim paths and local transforms.** `snapshot_body_transforms`
   emits `prim.GetPath().pathString` as each `child_frame_id`, and the pose is local relative
   to the USD parent. Sending world transforms worked for flat 4.x/5.x but composes twice on AS3.

3. **`snapshot_odom` resolves `base_link` lazily.** Walks the subtree on first call, caches the
   resolved path; re-resolves only if the prefix changes.

### URDF/xacro authoring as the source of truth

The 6.0 converter branches on `is_ghost_link` (root link with no inertial/visual/collision →
`body0 = default_prim` for every joint → PhysX can't build the articulation). Fix: author
`<inertial>` on every URDF link. Use `genie_sim_robot_model/scripts/diagnose_urdf.py` to
detect and auto-inject placeholder `<inertial>` blocks. The runtime no longer patches this.

Newton tolerates massless links via fixed-joint collapse — the URDF fix is PhysX-driven but
harmless to Newton.

### Default selective collision policy

Found in [`assemble_robot.py:_apply_post_transformer_collision_policy`](../scripts/assemble_robot.py).
Runs after the AS3 transformer on the final `robot.usda`. The baked `robot.usda` is the single
source of truth for all three engines.

| Link kind | Authored `<collision>`? | Action |
|---|---|---|
| Gripper (`gripper`/`finger`/`jaw`/`knuckle`) | yes | `MeshCollisionAPI` approx `sdf` |
| Gripper | no | Promote visual mesh with `CollisionAPI` + `MeshCollisionAPI` approx `sdf` |
| Wheel (`wheel`/`caster`/`castor`) | yes | `MeshCollisionAPI` approx `convexHull` |
| Wheel | no | No collider |
| Anything else | yes | **DISABLED** — `physics:collisionEnabled = False` |
| Anything else | no | No-op |

SDF for gripper (preserves concave grip geometry), convex hull for wheel (stable rolling contact).

To add contact on a specific link, add its name pattern to `_COLL_GRIPPER_RE` /
`_COLL_WHEEL_RE` in `assemble_robot.py`.

```
[collision] post-transformer policy applied: gripper-authored(SDF)=N, ...
```

### `<material_override>`: roughness / metallic on embedded materials

Authored as a child of `<visual>` to patch PBR scalars the DAE pipeline can't carry:

```xml
<link name="chassis_link">
  <visual>
    <geometry><mesh filename="${mesh_dir}/tracer_base.dae"/></geometry>
    <material_override>
      <roughness>0.30</roughness>
      <metallic>0.85</metallic>
    </material_override>
  </visual>
</link>
```

Silently ignored by urdfdom, RSP, MoveIt, RViz, and Isaac's converter. `assemble_robot.py`
strips it before passing the URDF to the converter, then re-applies it against the intermediate
USD stage so the AS3 transformer picks up the values. Targeting uses `UsdPhysics.RigidBodyAPI`
to distinguish link prims from same-named mesh wrappers.

## fix_base: runtime toggle {#fix_base-runtime-toggle}

`robot.robot_source.fix_base` is a **runtime** flag, not a bake-time one. The cached
`robot.usda` always carries a world-weld joint; the runtime decides per scene whether it is active.

```
scene.yaml
    fix_base: true|false
        │
        ▼
assemble_robot.py
    converter always authors a world-weld (PhysicsFixedJoint "root_joint")
        │
        ▼
assemble_scene.py
    writes manifest["scene_yaml"] = <original yaml path>
        │
        ▼
genie_sim_engine_isaacsim.py
    reads manifest["scene_yaml"] live (via common.session.EngineSession),
    parses fix_base from the loaded yaml mapping,
    forwards to IsaacSimStage
        │
        ▼
kit.stage._apply_fix_base_policy
    walks robot subtree for the world-weld (FixedJoint with body0=articulation root)
    sets physics:jointEnabled = fix_base
```

Detection is name-agnostic (works for both 6.0 AS3 and flat 4.x/5.x naming).
Default when `scene_yaml` is missing: `fix_base=True`.

## Mimic joints

URDF mimic joints are authored as schemas by the URDF→USD importer:

| Pipeline | Schema | Enforced by |
|---|---|---|
| `isaac_physx` | `PhysxMimicJointAPI:<axis>` | PhysX solver — rigid kinematic constraint |
| `isaac_newton` | `NewtonMimicAPI` | software broadcast via `engine/_mimic.expand_targets` |
| `newton` | `NewtonMimicAPI` | same shared helper |
| 4.x / 5.x | neither | not enforced |

The broadcast lives in [`scripts/engine/_mimic.py`](../scripts/engine/_mimic.py) and is
shared by both Newton paths. `kit.stage._configure_drives` discriminates gripper master vs
mimic by the **presence of a mimic schema** (not `HasAPI(DriveAPI)`, which is authored on all
revolute joints including mimics).

There is **no scene-yaml mimic block** — a mimic relation is robot geometry, lives in USD only.

## `assemble_scene.py` — bind the robot USD into the manifest

Picks **which** USD to advertise (`urdf:` present → staged `robot.usda`; absent → legacy asset).
Writes `manifest.json` with paths **relative to `base_path`** so the cache survives workspace moves.

The only artifacts it produces:

1. **`render_layer.usda`** — mirrors camera prims from the robot stage under `/RenderOVRTX`,
   defines `RenderProduct` nodes, appends a synthetic `FreeCam`.

2. **`manifest.json`** — the engine's contract:

   | Field | Source | Purpose |
   | --- | --- | --- |
   | `base_path` | CLI `--base-path` | Anchor for relative paths. |
   | `scene_usda` | `config["scene"]` | World USD opened at runtime. |
   | `robot_usda` | resolved staged/legacy path | Robot USD referenced into scene. |
   | `render_layer_usda` | path to just-written layer | Optional render-product layer. |
   | `robot_prefix` | `robot.robot_prefix` | Stage prim namespace (`/<robot_prefix>`). |
   | `robot_from_urdf` | derived from `urdf:` presence | Classification path hint for `kit.stage`. |
   | `scene_yaml` | absolute path to **original** input yaml | Engine reads this live for `fix_base` etc. |
   | `cameras[]` | per-camera fan-out | Per-camera intrinsics + render product paths. |

### Where the scene+robot bond happens

At runtime, inside `IsaacSimStage.__init__` → `_open_scene_with_references`
([`kit/stage.py`](../scripts/kit/stage.py)):

1. `open_stage(scene_usda)`
2. `add_reference_to_stage(robot_usda, "/<robot_prefix>")`
3. `add_reference_to_stage(render_layer_usda, "/RenderOVRTX")`
4. `simulation_app.update()` pumped after each reference to let composition settle.

## The engine is a pure consumer

The physics node only:

1. `wait_for_manifest(manifest_path)` — blocks until JSON exists.
2. Re-anchors paths against `base_path` via a local `_abs()` helper.
3. Opens stage + references robot via `_open_scene_with_references`, configures drives +
   articulation gains from `config/physics_params.yaml`.

It does **not** touch the URDF, run xacro, or invoke the URDF importer.

## TL;DR

```
scene_*.yaml --> assemble_robot.py --> robot.urdf --> (Isaac URDF importer) --> robot.usda
   |                                                                                |
   +----------> assemble_scene.py --> render_layer.usda + manifest.json <-----------+
                                                                |
                                                                v
                                            genie_sim_engine_*.py reads manifest.json
                                            and references robot.usda into the stage
```

- Force a rebuild: `always_regenerate_robot_usd:=true` or `rm -rf assets/scenes/<scene_stem>/`.
- Switch routes: add or remove `robot.robot_source.urdf` in the scene YAML.
