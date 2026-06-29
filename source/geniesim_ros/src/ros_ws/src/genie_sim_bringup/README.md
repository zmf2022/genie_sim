# genie_sim_bringup

Launch orchestration for the GenieSim ROS 2 stack. This is the package
you land on first — `app.launch.py` is the top-level entry point that
reads a launcher YAML, picks the physics + render combination, and
wires up every other ROS node in the workspace.

## What's in here

- **`launch/app.launch.py`** — top-level launch. Requires
  `launcher_config:=…`; delegates to the right physics sub-launch.
- **`config/launcher_*.yaml`** — the supported (physics engine, renderer)
  combinations. Pick one with `launcher_config:=`.
- **`config/scene_*.yaml`** — per-robot, per-task scene definitions
  (cameras, init pose, optional cloth / soft-body blocks, debug
  publishers). Pick one with `scene:=`.
- **`rviz/*.rviz`** — canonical RViz layouts.

## Launching

```bash
# Genie G2 pick-and-place on Isaac Sim PhysX (stable default)
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_pnp_g2_op \
  launcher_config:=launcher_ovrtx_isaac_physx
```

See the workspace [README](../../../../README.md) for the full
quick-start (4-step flow + scene/launcher combination matrix).

## When you'd touch this package

- Adding a new scene → drop a `config/scene_<robot>_<task>.yaml`.
- Adding a new physics / renderer combination → drop a
  `config/launcher_<name>.yaml`.
- Changing how nodes are composed at launch → edit
  `launch/utils.py` (shared helpers) or one of the
  `physics_*.launch.py` sub-launches.
- Anything to do with what topics get published, what container the
  engine runs in, or which renderer fires — start here.

## Mechanics

See [AGENTS.md](AGENTS.md) for the full launch architecture, launcher
YAML schema, override precedence (CLI > YAML > defaults), and the rules
the sub-launches follow.
