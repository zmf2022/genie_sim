# Param-injection evaluation: `isaac_newton` vs `newton` (newton-standalone)

**TL;DR — they do NOT share the MuJoCo-Warp param injection.** Two
parallel implementations exist, in different files, with different
strategies for the same problems.  The new USD overlay
(`assemble_robot._apply_mimic_joint_overlay`) is the *only* source of
unified configuration that both consume identically.

## Where each engine's injection lives

| Layer | newton-standalone (`physics_engine:=newton`) | isaac_newton (`physics_engine:=isaac_newton`) |
|---|---|---|
| Adapter / driver class | `engine/newton/adapters/mujoco_warp.py:MuJoCoWarpAdapter` | none — uses `isaacsim.physics.newton` wrapper directly |
| Pre-solver injection | `MuJoCoWarpAdapter.prepare_model` (called from `engine/newton/setup/solver.py`) | `kit/stage.py:_configure_drives` (called from `IsaacSimStage.__init__`) |
| Post-solver injection | `MuJoCoWarpAdapter.build_solver._apply_mimic_dof_damping` | none |
| Per-substep mutation | `MuJoCoWarpAdapter.substep` (`_update_model_properties`) | none — Isaac wrapper handles internally |
| Stage-level overrides | `_apply_runtime_fix_policies` (lifecycle) | `_apply_fix_base_policy` (kit/stage) |
| Imports `engine/newton/adapters/`? | yes | **no** — verified by grep, zero references |

## Side-by-side: same problem, different solutions

| Concern | newton-standalone (`MuJoCoWarpAdapter`) | isaac_newton (`kit/stage.py:_configure_drives`) | Same? |
|---|---|---|---|
| **Force POSITION mode** on every DOF | `mode[:] = 1` (overwrites all) | Per-joint via `UsdPhysics.DriveAPI` authoring of nonzero `stiffness` (Newton wrapper auto-creates POSITION actuator when stiffness > 0 at finalize-time) | NO — different mechanism, same effect |
| **`ke` / `kd` defaults** for joints with vestigial / zero authored | per-joint `kp = effort × 10`, `kd = 2·√kp` (auto-scaled critical damping) | per-joint by `JointKind` classification: `drv_chassis`, `drv_steer`, `drv_revolute`, `drv_prismatic` (constants from `EngineNodeParams`) | NO — different tuning entirely |
| **`effort_limit` floor** (URDF effort=0 → ?) | clamped to `_DEFAULT_EFFORT_LIMIT=500` N·m for unauthored DOFs | not handled — URDF effort=0 stays at 0, joint is jelly | NO |
| **Mimic-follower** strategy | mute actuator entirely (`mode = NONE`); rely on mjwarp equality constraints | author non-zero stiffness/damping on followers so Newton creates POSITION actuators (only the DriveAPI authoring inside `_apply_gripper_mimic` is gated to `isaac_newton`); software-broadcast values via `apply_action` | OPPOSITE strategy |
| **Mimic-master** gain | `kp=5, kd=0` (matches reference G2 MJCF `gainprm="5"`) | `kp=1e4, kd=10` (per `GripperDriveParams`) | DIFFERENT — 2000× stiffer |
| **Master joint armature** | overlay-authored `physxJoint:armature=0.001` if present, else fallback | overlay-authored if present, else `_apply_gripper_master_drive` writes `armature=0.001` | converged via overlay |
| **Arm joint armature** (shoulder=0.15, mid=0.08, wrist=0.04) | overlay reads → propagates via `joint_armature` array | overlay reads → propagates via `joint_armature` array (Newton wrapper reads same schema) | YES (via overlay) |
| **`<dynamics friction>` from URDF** | reads through `joint_friction` → mjwarp `frictionloss` | reads through (Isaac's URDF importer respects URDF) | YES (build-time) |
| **`<dynamics damping>` from URDF** | reads through `joint_target_kd` (actuator-side) | reads through (same) | YES (build-time) |
| **`dof_damping` (joint-level passive damping)** | post-build write to `mjw_model.dof_damping=0.05` for muted-actuator joints | not authored anywhere — Isaac wrapper has no analogous passive-damping path that we expose | NO |
| **Gravity propagation** to mjwarp `mjw_model.opt.gravity` | `_update_model_properties()` called in every captured substep | Isaac wrapper handles gravity sync internally (closed source to us) | NO (different mechanism, similar effect) |

## What the new USD overlay does for both

`assemble_robot._apply_mimic_joint_overlay` writes USD attributes that
**both** engines' importers consume:

| Authored attr | newton-direct path | isaac_newton path |
|---|---|---|
| `physxJoint:armature = 0.001` (12 hand joints) | read via `SchemaResolverPhysx` → `joint_armature` populated → adapter armature fallback skipped | read by Isaac's URDF→Newton importer → same `joint_armature` populated |
| `physxJoint:armature = 0.15 / 0.08 / 0.04` (14 arm joints) | same as above | same as above |
| `drive:angular:physics:stiffness = 0.0873` (= 5 × π/180) on hand masters | read via `SchemaResolverPhysx` → `joint_target_ke = 5` → adapter master soft-tune detects asset value, skips | read by Isaac's importer → same `joint_target_ke = 5` |
| `drive:angular:physics:damping = 0` on hand masters | read → `joint_target_kd = 0` | read → `joint_target_kd = 0` |

So the overlay is what made the **gripper master + arm armature** consistent
across both backends. Everything ELSE remains divergent.

## What's still divergent (and why it matters)

When you run the same scene through both engines, today these
behaviors will **NOT** match:

1. **`effort_limit = 0`** joints — newton-direct floors them to 500 N·m;
   isaac_newton leaves at 0, joint can't deliver torque (jelly under
   `/joint_command`).  Hand-authored URDFs with proper `<limit effort>`
   values are unaffected; URDFs that left effort blank are.
2. **Joints with no authored drive** — newton-direct's
   `_KE_TO_EFFORT_FACTOR=10` rule sets sane per-joint `kp`/`kd` from
   the effort cap.  isaac_newton uses `_configure_drives`'s static
   `drv_revolute` / `drv_chassis` constants from `EngineNodeParams`.
   Two different tunings → different motion characteristics.
3. **Mimic followers** — opposite strategies:
   - newton-direct: muted (no actuator), constraint-driven only
   - isaac_newton: actuated (so Newton emits the POSITION actuator),
     software-broadcasted via `apply_action`
   Both work, but the gripper "feel" differs.  isaac_newton's followers
   have a separate spring fighting the equality constraint at every
   step; newton-direct's followers ride the constraint cleanly.
4. **Joint-level passive damping (`dof_damping`)** — only
   newton-direct writes 0.05 on muted joints; isaac_newton has no
   equivalent.  Limited impact since isaac_newton doesn't mute
   followers in the first place (they have actuator-side `kd` instead).
5. **`fix_base` semantics** — different code paths (`_apply_fix_base_policy`
   in kit/stage.py for isaac_*, `_apply_runtime_fix_policies` in
   `engine/newton/setup/stage.py` for newton-direct).  Both toggle a FixedJoint to
   world; behavior should match.  `convert_joints_to_fixed: [base, head,
   body]` is **newton-direct only** — isaac_newton silently ignores it.

## Should they converge?

Three options:

1. **Status quo** — accept that each path has tuning suited to its
   physics backend.  Document the gap.  This is what the codebase has
   been doing.

2. **Lift everything to the USD overlay** — author every parameter in
   `assemble_robot._apply_mimic_joint_overlay` so both engines'
   importers read identical values.  Specifically: drive stiffness on
   every controllable joint (not just gripper masters), drive damping
   per joint, effort_limit floor (write `physics:joint:maxJointForce`
   or similar), follower drive=0 to coerce isaac_newton into
   no-actuator mode (matching newton-direct's mute).  Adapters become
   pure fallbacks for non-overlaid scenes.  **Highest fidelity, most
   work.**  Estimated 100-150 lines added to the overlay + a careful
   audit of which USD attributes Isaac Sim's Newton importer actually
   reads.

3. **Have isaac_newton reach into MuJoCoWarpAdapter** — after
   `acquire_stage()`, mutate the live Newton model the same way
   `MuJoCoWarpAdapter.prepare_model` does (joint_target_ke, kd,
   effort_limit, dof_damping, etc.).  The injection logic stays in
   `MuJoCoWarpAdapter`; isaac_newton imports and calls it post-stage
   acquisition.  **Mid-effort, requires reaching into Isaac wrapper
   internals which may break across versions.**

## Recommendation

Two principles, decided per layer:

1. **Per-class PD tuning (arm / body / head / chassis) lives in
   ``config/physics_params.yaml``.**  ``kit/stage.py:_configure_drives``
   reads the yaml at every launch and writes USD ``DriveAPI``
   attributes to the live stage — no rebake required to iterate on
   tuning, and the same yaml drives both ``isaac_physx`` and
   ``isaac_newton``.  This is the established pattern; do NOT bake
   per-joint kp/kd into ``robot.usda`` at build time.  The values in
   ``physics_params.yaml`` (``usd_drive_api.default_revolute``,
   ``usd_drive_api.chassis_drive_joint``, etc.) are tuned for PhysX; that's
   the engine's primary loop.

2. **Build-time USD overlay handles ONLY what ``physics_params.yaml`` can't
   express AND what doesn't exist in URDF.**  Today that's:

   - ``physxJoint:armature`` per joint class — URDF has no tag and
     ``physics_params.yaml`` doesn't carry per-joint armature either.
   - Gripper master drive (``stiffness=5, damping=0``) — matches the
     reference G2 MJCF; ``physics_params.yaml::usd_drive_api.gripper.master_stiffness``
     exists for the Kit path but only fires there, so authoring the
     USD attribute lets newton-standalone see the same value.
   - Gripper follower drive (``stiffness=0, damping=0``) —
     constraint-only behavior.  Newton's
     ``JointTargetMode.from_gains(0, 0, has_drive=True)`` resolves
     to EFFORT mode → no POSITION actuator emitted → equality
     constraint drives the follower cleanly.  Both engine paths
     read this identically.

   That's it — three USD-only authoring tasks, all gripper-related.

3. **Newton-standalone's ``MuJoCoWarpAdapter`` keeps its
   per-effort-scaled fallback** for arm / body / head / chassis,
   since ``physics_params.yaml``'s PhysX-tuned values (5e4 N·m/rad) are
   wildly too stiff for mjwarp's solver (saturates to jelly at
   any small error).  Newton-standalone's tuning loop is the
   adapter constants ``_KE_TO_EFFORT_FACTOR=10`` and
   ``_KD_TO_KE_FRACTION=2``.

## Net result

| Joint group | Source of PD on Kit (isaac_physx / isaac_newton) | Source of PD on newton-standalone | Source of armature |
|---|---|---|---|
| Hand master | USD overlay (kp=5) | USD overlay (kp=5) | USD overlay (0.001) |
| Hand follower | USD overlay (kp=0 → no actuator) | USD overlay (kp=0 → no actuator) | USD overlay (0.001) |
| Arm shoulder/mid/wrist | physics_params.yaml `usd_drive_api.default_revolute` | adapter `effort × 10` | USD overlay (0.15/0.08/0.04) |
| Body | physics_params.yaml `usd_drive_api.default_revolute` | adapter `effort × 10` | (none — bare per reference) |
| Head | physics_params.yaml `usd_drive_api.default_revolute` | adapter `effort × 10` | (none) |
| Chassis drive | physics_params.yaml `usd_drive_api.chassis_drive_joint` | adapter `effort × 10` | (none) |
| Chassis steer | physics_params.yaml `usd_drive_api.chassis_steer_joint` | adapter `effort × 10` | (none) |

The gripper is the **only** case where build-time overlay authoring is
the right answer (reference MJCF specific, can't express in URDF, and
``physics_params.yaml`` couldn't reach newton-standalone before).  Everything
else stays runtime-tuned via physics_params.yaml (Kit) or adapter constants
(newton-standalone).  Both engines reach functional parity for a
G2-class robot with this split.
