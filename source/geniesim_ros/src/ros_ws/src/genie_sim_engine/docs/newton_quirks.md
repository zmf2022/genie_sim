# Newton 1.15.0.dev20260526 quirks

The newton package shipped inside `geniesim docker5.1`
(`/usr/local/lib/python3.12/dist-packages/newton/`, version
`1.15.0.dev20260526`) has a couple of behavioural quirks that
matter for the assemble pipeline and any tool that round-trips
USD ↔ MJCF.  This page is the breadcrumb for the next person
who hits one of them.

## 1. `mjc:damping` for angular joints is divided by 180/π at MJCF write

### Where

`/usr/local/lib/python3.12/dist-packages/newton/_src/solvers/mujoco/solver_mujoco.py:4531`:

```python
# angular DOF branch of _convert_to_mjc
if joint_damping is not None:
    joint_params["damping"] = joint_damping[ai] * (np.pi / 180)
```

The linear-DOF branch on line **4433** writes `joint_damping[ai]`
verbatim with NO conversion.  Only angular DOFs get the
`× π/180` baked in.

### What happens

If you author `mjc:damping = 0.05` on a `hinge` joint in USD,
Newton's USD parser correctly stores `dof_passive_damping = 0.05`
(verified via `m.mujoco.dof_passive_damping.numpy()`).  But at
`SolverMuJoCo._convert_to_mjc` write time, the value emitted into
the MJCF is `0.05 × π/180 ≈ 0.000872665`.  Newton ALSO writes
`<compiler angle="radian"/>` at the top of the file, so MuJoCo
reads `damping="0.000872665"` per-radian — making the physical
damping **60× weaker** than authored.

There is NO compensating transform at read time (the linear
branch passes through; the angular branch is asymmetric).

### Why

Probably a leftover from an older MuJoCo-MJCF authoring
convention where `<compiler angle="degree">` was the default, in
which case the `× π/180` would correctly convert per-radian
Newton-internal to per-degree MJCF-authored.  But Newton emits
`angle="radian"` unconditionally, so the conversion is
inconsistent with its own compiler tag.

### Workaround (this codebase)

`assemble_robot.py:_apply_mimic_joint_overlay` pre-divides by
`π/180` when authoring `mjc:damping` so the converter's hardcoded
multiplication lands at the intended value:

```python
_HAND_JOINT_DAMPING_FINAL = 0.05                                # target per-rad value
_HAND_JOINT_DAMPING_USD   = _HAND_JOINT_DAMPING_FINAL / _DEG_TO_RAD  # ≈ 2.8648
```

Verified end-to-end inside the container: USD = `2.8648` →
`dof_passive_damping = 2.8648` → MJCF `damping="0.05"`.

### If Newton ever fixes this

Drop the `/ _DEG_TO_RAD` from `_HAND_JOINT_DAMPING_USD`.  The
overlay docstring carries a self-flagging comment so the next
person updating Newton spots the dependency.

The signal that Newton has fixed it: `grep -n 'np.pi / 180'
solver_mujoco.py` will no longer show line 4531's branch, and
`test_robot_xml_dynamic.py --xml <fresh dump>` will show
`damping="2.8648"` (the raw authored value) instead of `0.05`.

### Bug to file upstream

For a future Newton release the fix should be one of:

  * Remove line 4531's `× π/180` so both linear and angular
    branches treat `joint_damping` as per-radian (matches
    `<compiler angle="radian"/>`).
  * Or convert the per-degree MJCF damping back when reading
    so the round-trip is consistent.

Either way, the asymmetry between the linear and angular write
paths is the smell.

### Related field

`joint_params["stiffness"]` for angular DOFs (line 4530, same
function) has the SAME `× π/180` baked in.  We don't author
`mjc:stiffness` directly so it doesn't bite us today, but if a
future overlay does, it will hit the same trap and need the same
`/ _DEG_TO_RAD` compensation.

## 2. Force-cap inconsistency between joint and actuator

A separate sharp edge: when Newton's converter writes a joint
with both `actfrclimited="true"` and `forcerange="-5 5"` on the
actuator side, the joint-level `actfrcrange` IS authored from
`joint_effort_limit` (URDF `<limit effort>`) but the **actuator
side carries its own `forcerange`** derived independently.  In
the current behavior they happen to match, but reading code
shouldn't assume that — see `kit/stage.py:_apply_gripper_master_drive`
for how the runtime applies the master_stiffness override
without touching either cap.

## 3. SolverMuJoCo schema-resolver asymmetry

`newton-standalone` invokes `add_usd` with
`[SchemaResolverNewton(), SchemaResolverPhysx()]` —
**no `SchemaResolverMjc`**.

`isaac_newton` (wrapper) invokes with
`[SchemaResolverNewton(), SchemaResolverMjc(), SchemaResolverPhysx()]`.

In practice `SchemaResolverMjc` only changes things if `mjc:*`
attributes are authored on body / scene / actuator prims (not
the joint-level `mjc:damping` we use — that flows through the
custom-attribute registration system regardless).  But the
asymmetry is real and may matter for future overlays that touch
other `mjc:*` attributes.

If you see a USD attribute that lands cleanly in `MJCF` via the
wrapper but not in standalone (or vice versa), check whether
the attribute is registered as a `SchemaResolverMjc` mapping or
as a `ModelBuilder.CustomAttribute(namespace="mujoco", ...)` —
the latter works regardless of which resolvers are loaded.

## 4. Mimic equalities ignore `eq_solref` / `eq_solimp` custom attributes

### Where

`solver_mujoco.py:4732` — the loop that converts Newton's mimic
constraints into `mjEQ_JOINT` equalities:

```python
eq = spec.add_equality(objtype=mujoco.mjtObj.mjOBJ_JOINT)
eq.type = mujoco.mjtEq.mjEQ_JOINT
eq.active = bool(mimic_enabled[i])
eq.name1 = j0_name          # follower
eq.name2 = j1_name          # leader
eq.data[0] = float(mimic_coef0[i])
eq.data[1] = float(mimic_coef1[i])
# ... polycoef set, but eq.solref / eq.solimp are NEVER set
```

`eq_solref` IS registered as a custom attribute (line 621, USD
attribute `mjc:solref` on equality prims) but only the non-mimic
equality types consume it.  Mimic equalities always inherit the
MuJoCo default `solref="0.02 1"`.

### What happens

The default `solref="0.02 1"` makes the mimic equality constraint
behave like a 20-ms-time-constant spring.  Under inertial reaction
torque from arm motion, gripper followers swing 0.1+ rad against
master positions that are essentially fixed.  Verified via
`test_robot_xml_dynamic.py --cross-impact-mode sweep`: at default
solref, sweeping `arm_l_joint3` by 0.5 rad drives
`gripper_l_outer_joint2` by 0.15 rad while the master
`gripper_l_inner_joint1` holds at < 0.001 rad — the equality is
just not stiff enough.

### Workaround (this codebase)

Two coordinated patches in `kit/isaac_newton.py`:

  * `_stiffen_mimic_equalities` post-processes the dumped MJCF
    file, regex-injecting `solref="-10000 -100" solimp="0.95
    0.99 0.001 0.5 2"` on every `<joint joint1="...gripper..."
    joint2="..." polycoef="..."/>` equality.  Affects the file
    only.
  * `_stiffen_live_mimic_equalities` mutates the LIVE solver's
    `mjw_model.eq_solref` / `eq_solimp` wp.arrays for the same
    set of gripper equalities.  Affects the running solver only.

**Both must run** so the dumped MJCF and the live runtime land
at the same physical behaviour.  An earlier version of this code
ran only the first — the dumped MJCF showed `solref="-10000 -100"`
but the live solver kept Newton's default `solref="0.02 1"`, and
`test_robot_xml_dynamic.py` predictions diverged from the live
runtime by ~3× on follower amplitude.  The user spotted this
when same-param-tuning between CPU MuJoCo (against the dump) and
isaac_newton runtime stopped agreeing.

Equality scoping in both functions matches the same substring
(`"gripper"` on the joint name) so unrelated future loop joints
don't get caught.

Cuts follower amplitude from 0.15 rad to 0.06 rad under the same
arm sweep at the wrapper's 2 ms substep.  Going stiffer than
`-10000 -100` (e.g. `-1e6 -1e4`) explodes the integrator at this
dt; if the wrapper ever drops to 1 ms substeps we can re-tune
toward `-50000 -500`.

### Bug to file upstream

The mimic emit path should respect `eq_solref` / `eq_solimp`
custom attributes the same way other equality types do.  A
two-line change in `solver_mujoco.py` around line 4733:

```python
if eq_constraint_solref is not None:
    eq.solref = wp.vec2(*eq_constraint_solref[i])
if eq_constraint_solimp is not None:
    eq.solimp = vec5(*eq_constraint_solimp[i])
```

When that lands, this workaround in `_stiffen_mimic_equalities`
becomes redundant — we'd switch to authoring `mjc:solref` on the
follower joint prim in `_apply_mimic_joint_overlay` instead.

## 5. Joint range write does `rad2deg` against `compiler angle="radian"`

### Where

`solver_mujoco.py:5133` — the `mjJNT_HINGE` (angular DOF) branch of
`_convert_to_mjc`:

```python
# angular DOF branch
joint_params["range"] = (np.rad2deg(lower), np.rad2deg(upper))
```

vs. the linear branch on line 5035 which writes verbatim:

```python
# linear DOF branch
joint_params["range"] = (lower, upper)
```

The same asymmetry pattern as quirk #1: angular DOFs get a `× 180/π`
conversion to degrees at MJCF write time, while the compiler tag
Newton emits at the top of every dump is `<compiler angle="radian"/>`
unconditionally.

`joint_springref` and `joint_ref` on the same branch (lines 5149–5152)
have the same `rad2deg` baked in.

### What happens (empirically: nothing — for now)

If you author `physics:lowerLimit = -176` on a USD revolute joint
(-3.0718 rad), the chain through Newton 1.2.0's `MjSpec` produces a
final MJCF with `range="-3.0718 3.0718"` and `<compiler angle="radian"/>`
— the URDF-original radians, internally consistent with the compiler
tag. Verified inside the container:

```python
import newton, newton.solvers as ns, tempfile, os
b = newton.ModelBuilder()
b.add_usd("scene_flat_g2_sp_vbd/robot.usda")
m = b.finalize()
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "dump.xml")
    ns.SolverMuJoCo(m, save_to_mjcf=p)
    # range="-3.0718 3.0718"  ← radians, matches URDF
```

How: `joint_params["range"]` lands on `mujoco.MjSpec.add_joint(range=…)`
in **degree units** because that's the unit `MjSpec` expects regardless
of what the user wires to the compiler tag. `MjSpec.write()` then
converts to the unit the compiler tag declares on the way out, so the
emitted MJCF carries radians. The `× 180/π` on line 5133 is therefore a
"pre-bake into MjSpec's expected input units" operation, not a unit
bug.

### Why it's still listed

Three failure modes turn this from harmless into a quirk:

1. **A future Newton version drops the `rad2deg`** — at the same time
   `MjSpec` still expects degrees-in, the MJCF will then carry
   radian-magnitude values labelled as degrees by `MjSpec` and
   converted to "radians × 180/π" on write. Joint limits would
   become ~57× too tight (you'd see e.g. `range="-0.0536 0.0536"` on
   what used to be a 3.07 rad joint).

2. **A future Newton version changes the compiler to
   `angle="degree"`** — then everything matching the URDF in radians
   becomes wrong in the dump. `_stiffen_mimic_equalities` in
   `kit/isaac_newton.py` would also need re-checking since it
   regex-injects solref/solimp values that are unit-agnostic but
   sit next to the now-degree range field.

3. **A consumer skips MuJoCo's `MjSpec` and reads the raw mid-stage
   `joint_params["range"]` directly** (e.g., a custom dumper that
   uses Newton's internal JSON-style spec instead of going through
   MuJoCo's write path) — that consumer will see degree-magnitude
   values for angular joints and unit-verbatim values for linear,
   without any compensation. The legacy `test_robot_xml_dynamic.py`
   doesn't trip this because it loads the already-dumped MJCF, but
   future tools that inspect Newton's intermediate spec might.

### Workaround (this codebase)

None needed today. If failure mode #1 or #2 lands, the workaround is
the same pattern as `mjc:damping`: pre-divide the value we author in
USD by `_DEG_TO_RAD` so the converter's hardcoded transform lands at
the intended radian value. For joint limits specifically that would
mean intercepting the URDF→USD converter's output and rewriting
`physics:lowerLimit` / `physics:upperLimit` in our post-transformer
chain. We don't currently author joint limits directly — they come
from the URDF — so the rewrite would have to read each joint's
current value, multiply, write back.

The signal that #1 has landed: `grep -n 'rad2deg' solver_mujoco.py`
no longer shows the line 5133 / 5149 / 5152 transforms, AND the
container-side empirical test above produces `range="-176.00 176.00"`
on a joint whose USD value is `-176`.

### Related fields

The springref / ref / target_pos / target_vel angular handling in
`_convert_to_mjc` shares the same `rad2deg` pre-bake. The associated
`target_ke` / `target_kd` divide instead — see lines 5145–5146:

```python
joint_params["springref"] = np.rad2deg(joint_springref[ai])
joint_params["ref"]       = np.rad2deg(joint_ref[ai])
```

Plus the input-side mirror in `import_usd.py:1024–1030`:

```python
joint_params["target_pos"] *= DegreesToRadian   # USD-deg → Newton-rad
joint_params["target_vel"] *= DegreesToRadian
joint_params["target_kd"] /= DegreesToRadian / joint_drive_gains_scaling
joint_params["target_ke"] /= DegreesToRadian / joint_drive_gains_scaling
joint_params["limit_lower"] *= DegreesToRadian
joint_params["limit_upper"] *= DegreesToRadian
joint_params["limit_ke"]   /= DegreesToRadian
joint_params["limit_kd"]   /= DegreesToRadian
```

That import-side conversion is real and load-bearing — it's why our
USD-deg values land as URDF-rad values inside Newton's model. The
matching write-side `rad2deg` at line 5133 is what makes the
round-trip back through `MjSpec` come out at URDF-radian values in
the final MJCF.

### Bug to file upstream

The write-side `rad2deg` would be clearer as a no-op (drop the
conversion, document that MjSpec is the authoritative converter
based on the compiler tag) — or move all unit conversions into one
place (an explicit "MJCF write angle convention" knob) instead of
the current "in two halves, hope they cancel" pattern.

## 6. USD→MJCF inertia rotation uses the opposite convention from USD spec

### Where

`newton/_src/solvers/mujoco/solver_mujoco.py`,
inside `SolverMuJoCo.save_to_mjcf`'s inertia writer. The converter
reads USD's `PhysicsMassAPI` (`diagonalInertia` D, `principalAxes` R)
and emits MJCF `<inertial>` with `pos`, `quat`, `diaginertia`. The
back-projection to body-frame inertia uses

```python
I_body = R @ diag(D) @ R.T
```

but OpenUSD's `PhysicsMassAPI` spec says `principalAxes` rotates from
the body frame TO the principal frame, so the correct
body-frame inertia is

```python
I_body = R.T @ diag(D) @ R
```

### What happens

For most R the two formulas coincide (any rotation whose matrix
representation is involutory — eg. 90° about a single axis, which is
what Isaac picks for well-separated eigenvalues — has `R == R.T` on
the relevant block). The two **only differ when R is a 3-cycle
permutation** (rotation by 120° about (1,1,1)/√3, written as quat
`(0.5, 0.5, 0.5, 0.5)` or sign variants).

Isaac picks the 3-cycle quaternion when:

* the URDF `<inertia>` is diagonal in the link frame (`ixy=ixz=iyz=0`,
  `<origin rpy="0 0 0">`),
* AND the sorted-ascending order of (ixx, iyy, izz) cyclically
  permutes the link axes (eg. body_link5's `izz < ixx < iyy` — Z<X<Y
  is a 3-cycle, whereas Z<Y<X would be a transposition).

When all three conditions land, Newton's `R · diag · R.T` produces a
cyclic permutation of the URDF diagonal: `(ixx, iyy, izz) →
(iyy, izz, ixx)`. The MJCF then has `<inertial>` with the cyclically
permuted `diaginertia` and `quat="1 0 0 0"` (identity), and MuJoCo
reads it as the wrong inertia tensor in body frame — visible in the
inertia ellipsoid display as a 90° rotation relative to RViz / URDF.

### Why it's hard to see

* The bug only fires for the 3-cycle permutation case. Most links
  have eigenvalues well-separated enough that Isaac picks a
  transposition (`R == R.T`), and the convention mismatch cancels.
* The bug only fires when URDF has `rpy=0` and diagonal `<inertia>`.
  Any non-zero rpy or non-zero off-diagonal in the URDF
  pushes Isaac into a non-shortcut path that picks an arbitrary
  rotation matrix — usually NOT a 3-cycle — so the bug is masked.

Empirically: G2's `body_link5` (`izz=0.072 < ixx=0.240 < iyy=0.245`)
hits this exact bug; `body_link4` (`rpy=-0.664` from real CAD tilt)
does not.

### Workaround (this codebase)

`genie_sim_robot_model/scripts/recompute_g2_inertia.py` does two things
to dodge the bug without changing physics:

1. `_spread_near_degenerate` pushes near-equal eigenvalue pairs to
   ≥6% relative gap (preserving their average), so Isaac doesn't
   land in the degenerate-pair shortcut.
2. `_avoid_3cycle_sort` checks whether the resulting sorted-ascending
   order of `(ixx, iyy, izz)` is a 3-cycle of link axes, and if so,
   swaps two diagonal entries so the order becomes a transposition.
   Since both swapped entries are near-equal after step (1), the
   physical change is below 3% — well under the uniform-density
   modelling error.

The diagnose tool's `[inertia-degenerate]` rule warns when an
authored URDF would land in the bug regime (diagonal + rpy=0 + pair
within 5%).

### Why "snap-to-exact-equality" doesn't work

A natural first attempt: when `ixx ≈ iyy`, set them bit-identical.
We tried this — it makes things WORSE. With exactly-equal eigenvalues
Isaac STILL picks the 3-cycle quat in the degenerate subspace
(there's no "natural" rotation to fall back to), and the bug fires
on additional links that were previously well-separated. The actual
fix has to enforce a non-zero gap AND a non-cyclic sort order.

### Why "tiny rpy injection" doesn't work

A natural second attempt: write `<origin rpy="1e-3 0 0">` to nudge
Isaac off the diagonal-shortcut path. We tested this up to
`rpy=0.1` (5.7°). Isaac's choice of 3-cycle vs transposition is a
function of eigenvalue ordering, NOT rpy magnitude — the rpy
modulates the resulting quaternion slightly but Isaac still picks
the cyclic permutation. Comparing `body_link5` USD with `rpy=0.001`
vs `rpy=0.1`: the principalAxes quat is `(0.4997, 0.5002, 0.4997,
0.5002)` and `(0.4744, 0.5244, 0.4744, 0.5244)` respectively — both
still cyclic at the (0.5, 0.5, 0.5, 0.5) singularity.

### Bug to file upstream

The fix is one of:

* Change `save_to_mjcf` to use `I_body = R.T @ diag(D) @ R` (the
  USD-spec convention).
* OR change the read-side interpretation of `principalAxes` to match
  Newton's current convention (treat it as principal-to-body
  instead of body-to-principal) — though USD spec is clear on the
  direction.

Either way, the asymmetry between Newton's interpretation and USD
spec is the smell.
