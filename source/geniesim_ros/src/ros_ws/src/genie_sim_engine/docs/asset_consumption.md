# Asset consumption: how each engine path uses `assemble_robot.py`'s output

This is the missing complement to [pipeline.md](pipeline.md) (which covers
the build) and [engines.md](engines.md) (which covers runtime behavior).
Here we trace how the **single shared build output** — `robot.usda` and
its `payloads/` tree — is *consumed* differently by each of the three
physics engine paths, plus what each path injects on top.

## The build output (recap)

`assemble_robot.py` produces an Asset Structure 3.0 package:

```
robot.usda                            ← top-level interface, sets default variant
payloads/
  base.usda                           ← link xforms + meshes (variant-independent)
  geometries.usd                      ← shared mesh data (variant-independent)
  instances.usda, materials.usda      ← shared
  Physics/
    physx.usda                        ← +PhysxArticulationAPI, +PhysxJointAPI, −NewtonArticulationRootAPI
    physics.usda                      ← +PhysicsRigidBodyAPI, +PhysicsArticulationRootAPI, +NewtonArticulationRootAPI
    mujoco.usda                       ← −PhysicsDriveAPI:angular, −PhysicsJointStateAPI:angular
```

`robot.usda` opens with:

```usda
def Xform "robot" (
    prepend references = @./payloads/base.usda@
    variants = { string Physics = "physx" }    ← default selection
    append variantSets = "Physics"
)
```

So **the default consumer gets the `physx` variant**.  Other engines
have to switch the variant if they want different schemas.

## Comparison matrix

| Aspect | `isaac_physx` | `isaac_newton` | `newton` (newton-direct) |
|---|---|---|---|
| Stage loader | Isaac Sim `omni.usd.open_stage` | Isaac Sim `omni.usd.open_stage` | `pxr.Usd.Stage.Open` (headless) or Isaac (kit-mode) |
| Stage entry | `kit/stage.py:_open_scene_with_references` | same as left | `engine/newton/engine.py:NewtonHeadlessEngine._open_stage` (concrete impl; abstract slot in `engine/newton/setup/stage.py`) |
| Default Physics variant | `physx` (matches `robot.usda` default) | `physx` (inherits default) | `physx` (inherits default) |
| Solver(s) | `omni.physx` (PhysX 5) | `isaacsim.physics.newton` wrapping mjwarp | `newton` Python directly (`SolverMuJoCo` or `SolverFeatherstone`) |
| Cloth support | no (PhysX rigid only here) | no (mjwarp wrapper drops particles) | yes (cloth solver wired in `engine/newton/setup/solver.py`; per-entry injection in `engine/newton/cloth.py`) |
| URDF importer used | Isaac Sim's | Isaac Sim's | Isaac's at build, but `add_usd(...)` re-parses the stage at runtime |
| Mimic joint handling | PhysxMimicJointAPI authored by Isaac importer | same | parses `NewtonMimicAPI` from stage; mjwarp adapter mutes follower actuators |

## Runtime injections per path

Everything below is what the engine does to the *already-built* USD, on
top of what `assemble_robot.py` baked in.

### `isaac_physx` — minimal post-load

`kit/stage.py` runs against the live composed stage:

| Injection | When | Where | What |
|---|---|---|---|
| `_apply_fix_base_policy` | post-`open_stage` | `kit/stage.py:449` | Toggles `physics:enabled` on the world-weld FixedJoint (the one assemble_robot always authors at the URDF root) per scene yaml's `fix_base`. |
| Manifest-driven scene composition | one-shot | `kit/stage.py:_open_scene_with_references` | Adds the robot reference and any render-layer reference under the scene root. |

That's it. PhysX reads everything else (drives, limits, masses,
`<mimic>` via `PhysxMimicJointAPI`) directly from the schemas the AS3
transformer + post-process emitted.  No per-DOF gain or armature
override at runtime.

### `isaac_newton` — same as physx + solver pin

| Injection | When | What |
|---|---|---|
| Identical to `isaac_physx` for stage open + fix_base | — | Inherits the same `kit/stage.py` machinery. |
| `physics_solver` pinned to `mujoco-warp` | engine `__init__` | Hardcoded in `kit/isaac_newton.py`; ignores any other `physics_solver` value. |

The Isaac Newton wrapper handles the URDF→Newton-model translation
internally.  No code path here reads our `physxJoint:armature` overlay
directly — it goes through `omni.isaac.newton`'s own reader.  In
practice this means: properties baked into `payloads/Physics/physx.usda`
(the variant Isaac Newton inherits) are what reach mjwarp; the Python-
side overlay attributes flow through Isaac's USD→Newton converter.

### `newton` (newton-direct) — heaviest runtime injection

The newton-direct path doesn't go through `omni.isaac.newton` — it
constructs the Newton model itself by re-parsing the USD via
`builder.add_usd(...)`.  That gives the path the most control AND the
most responsibility for runtime injection.

`engine/newton/setup/runtime.py:_RuntimeMixin._build` (and the
phase mixins it composes — see `engine/newton/setup/__init__.py`):

```python
builder.add_usd(
    source=self._stage,
    collapse_fixed_joints=False,
    schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()],   # ← reads physxJoint:armature etc.
)
```

| Injection | When | What |
|---|---|---|
| Stage-level `_apply_runtime_fix_policies` | before `add_usd` | Applies the `convert_joints_to_fixed: [base, head, body]` list — replaces matching joints with `UsdPhysics.FixedJoint` to shrink Featherstone's mass matrix.  USD session-layer overrides; payloads stay untouched. |
| `_deactivate_root_joint` | before `add_usd` | When `pin_base_to_world: false`, deactivates the URDF-importer-authored `root_joint` so Newton adds a free base. |
| `_build_mimic_map` | before `prepare_model` | Parses `NewtonMimicAPI` from the live stage to discover gripper masters/followers. |
| `MuJoCoWarpAdapter.prepare_model` | before `build_solver` | (a) Force `joint_target_mode = POSITION`. (b) Floor `joint_effort_limit` at 1 N·m up to 500 N·m. (c) Per-joint `kp = effort × 10` / `kd = 2·√kp` for unauthored DOFs. (d) Mimic policy: mute follower actuators (mode=NONE), soft-tune master kp ≤ 100 to 5 / 0. (e) Armature + frictionloss FALLBACK injection — only fires where the asset didn't author. |
| `MuJoCoWarpAdapter.build_solver` | post-construction | `mjw_model.dof_damping = 0.05` write on mimic-touched joints (no URDF/USD route reaches muted-actuator joints for joint-level damping). |
| `MuJoCoWarpAdapter.substep` | every step | Calls `solver._update_model_properties()` to propagate `model.gravity` → `mjw_model.opt.gravity` (mjwarp keeps a separate copy). |
| `_pick_substep_body` | once at build | Selects kinematic-control vs plain rigid vs franka-VBD-cloth substep regime based on `(cloth_solver, pin_base_to_world)`. |

## What the new USD overlay (`_apply_mimic_joint_overlay`) shifts

Three of the runtime injections in the table above become **fallbacks
that don't fire** when the build runs through the new overlay:

| Property | Pre-overlay (adapter does the work) | Post-overlay (asset does the work) |
|---|---|---|
| Hand-joint armature (12 DOFs) | adapter writes 0.001 | overlay authors `physxJoint:armature=0.001` → `joint_armature` populated by `add_usd` → adapter sees nonzero → fallback skipped |
| Arm shoulder/mid/wrist armature (14 DOFs) | adapter wouldn't write (only mimic-touched joints get fallback) | overlay authors per-class values (0.15 / 0.08 / 0.04) — pure win |
| Hand master `kp` / `kd` | adapter overrides to 5 / 0 | overlay authors `drive:angular:physics:stiffness = 0.0873` (= 5 × π/180 in PhysX per-degree convention) → adapter detects authored value via snapshot and skips |

Everything else stays adapter-side because URDF / USD can't express it:
- Mimic-follower actuator suppression (URDF's `<mimic>` is kinematic only)
- `dof_damping=0.05` on muted-actuator joints (Newton's
  `dof_passive_damping` route doesn't apply to mode=NONE joints)
- The franka kinematic-control substep regime (gravity=0, contacts=0
  during rigid step)

## When variant selection actually matters

All three engines currently use the **default `physx` variant**.  Two
scenarios where you'd switch:

- **`mujoco` variant** — strips `PhysicsDriveAPI:angular` and
  `PhysicsJointStateAPI:angular`.  Useful if your downstream tooling
  needs a "no-drive" view of the robot (pure kinematics + joint
  limits).  Mostly for direct mujoco / mjpython tooling outside our
  engine paths; not currently selected by any of our engines.

- **`physics` variant** — adds `NewtonArticulationRootAPI` and the
  generic `PhysicsArticulationRootAPI`, *without* the PhysX-specific
  schemas.  This would be the natural variant for newton-direct since
  it reads `NewtonArticulationRootAPI`, but at the cost of losing the
  `physxJoint:armature` attributes our overlay authors.  Today
  `SchemaResolverPhysx` reads `physxJoint:*` from any prim (the schema
  resolver doesn't require the API to be applied), so the `physx`
  variant works fine for newton-direct too.

If you ever need to run the newton-direct path against the `physics`
variant, the overlay would have to author into the `newton:armature`
attribute (read by `SchemaResolverNewton`) instead — small change in
`assemble_robot._apply_mimic_joint_overlay`.

## Validating each path's consumption

Three tools, each at a different point in the pipeline:

| Tool | Path | What it validates |
|---|---|---|
| `tools/test_assemble_robot.py` | build-time | Per-stage snapshots + structural validators on the assembled USD itself. |
| `tools/test_newton_solver.py --validate-pd` | runtime, newton-direct | Three-layer dump of `joint_target_*` (Newton arrays → mjwarp gainprm/biasprm → jnt_actfrcrange) so you can confirm the asset's authored values made it through `add_usd` + `prepare_model` + `_convert_to_mjc` to the solver. |
| `tools/test_newton_solver.py --save-mjcf` | runtime, newton-direct | Compiled-MJCF dump from `mjwarp.MjSpec.to_xml()`.  Diff against a reference MJCF (e.g. `G2.xml`) to spot structural differences — actuator topology, equality constraints, joint defaults. |

For `isaac_physx` / `isaac_newton`, the equivalent validation is
opening `robot.usda` in `isaaclab.viewer` or `mujoco.viewer` (for the
mjcf dump) and inspecting via the GUI.  No standalone tool wraps that
yet because Isaac's stage open is heavyweight enough that the launcher
is already the fastest reproduce.
