# geniesim_teleop — Agent Development Guide

The teleoperation runtime: a VR / Pico-driven teleop loop that streams
device poses into the simulator over ROS 2, plus device drivers and a
data-recording pipeline that turns recorded rosbags into HDF5 episodes.
The canonical way to launch it is the **`geniesim teleop` CLI verb**,
owned by [`geniesim_cli`](../geniesim_cli/).

Source: [source/geniesim_teleop/](.)
License: [Mozilla Public License Version 2.0](LICENSE)
Skills (Claude Code / opencode): [skills/](skills/)

> **Maintenance contract** — when you add a subcommand, rename a module
> entry point, change the package layout, or touch the intra-package
> import paths, **update this file in the same diff**. Agents read this
> as the source of truth.

---

## 1. The CLI surface

Every interaction goes through `geniesim teleop <subcommand>`. If the
`geniesim` binary isn't on `$PATH`, substitute
`python3 -m geniesim_cli teleop …` — same dispatch, same args.

| Subcommand | What it does |
|---|---|
| `run [flags...]` | Launch the VR / Pico teleop loop (`python -m geniesim_teleop.teleop`) |
| `bridge [--mode inprocess]` | Launch the in-process image pub/sub bridge (`python -m geniesim_teleop.bridge`) |

`geniesim teleop` (no subcommand) prints the usage table — keep that
output and this file in sync.

### Forwarded flags (run)

Anything not consumed by the verb is forwarded verbatim to
`geniesim_teleop.teleop`'s `argparse`:

| Flag | Effect |
|---|---|
| `--client_host=H:P` | gRPC client host:port (default `localhost:50051`) |
| `--host_ip=IP` | VR host IP (auto-detected if omitted) |
| `--port=N` | VR server port (default `8080`) |
| `--robot_cfg=F` | Robot config json (default `G2_omnipicker.json`) |
| `--device_type=T` | Teleop device (default `pico`) |

### Interpreter selection

Same order as `geniesim benchmark`, so the verb works in every
environment:

1. `$GENIESIM_PY_CMD` — explicit override, set by `geniesim docker`
2. `omni_python` if on `$PATH` (canonical Isaac Sim wrapper)
3. `sys.executable` (or `python3` as last resort)

---

## 2. Package layout

```
src/geniesim_teleop/
  teleop.py           # entry: TeleOp loop + main()  (geniesim-teleop console script)
  bridge.py           # entry: in-process image pub/sub bridge
  utils/              # logger, ros_utils, ros_nodes, vr_server, name/transform utils
  devices/            # teleop_device (base), pico_device (Pico VR driver)
  config/             # robot_interface.py — robot descriptors (G2)
  data_recording/     # rosbag → HDF5 pipeline (extract_ros_bag, process_data, sim_data_converter)
  app/                # prebuilt motion-control runtime (binaries, robot cfg, vendored msgs) — shipped as package-data
```

All intra-package imports use the **absolute** `geniesim_teleop.…`
form (e.g. `from geniesim_teleop.utils.logger import Logger`). Do **not**
reintroduce bare top-level imports (`from utils.logger import …`) or
`sys.path` hacks — they break once the package is installed.

---

## 3. Path map

| Artifact | Location |
|---|---|
| Teleop entry | [`src/geniesim_teleop/teleop.py`](src/geniesim_teleop/teleop.py) |
| Bridge entry | [`src/geniesim_teleop/bridge.py`](src/geniesim_teleop/bridge.py) |
| Device drivers | [`src/geniesim_teleop/devices/`](src/geniesim_teleop/devices/) |
| Robot descriptors | [`src/geniesim_teleop/config/robot_interface.py`](src/geniesim_teleop/config/robot_interface.py) |
| Recording pipeline | [`src/geniesim_teleop/data_recording/`](src/geniesim_teleop/data_recording/) |
| Motion-control runtime | [`src/geniesim_teleop/app/`](src/geniesim_teleop/app/) |
| CLI dispatcher | [`source/geniesim_cli/src/geniesim_cli/commands/teleop.py`](../geniesim_cli/src/geniesim_cli/commands/teleop.py) |

### Cross-package dependency

`teleop.py`'s `_load_robot_init_states` reads
`config/teleop.yaml` and `benchmark/config/robot_init_states.py` from the
**`geniesim_benchmark`** package, located at runtime via
`importlib.util.find_spec("geniesim_benchmark")`. It degrades gracefully
(logs a warning, skips init-state loading) when the benchmark package
isn't installed.

---

## 4. Environment variables

| Var | Effect |
|---|---|
| `GENIESIM_PY_CMD` | Override the interpreter that launches the teleop modules. Set automatically by `geniesim docker`. |

---

## 5. Common workflows

### Launch the teleop loop (inside the GUI container)

```bash
geniesim teleop run --device_type=pico --port=8080
```

### Launch the in-process bridge

```bash
geniesim teleop bridge --mode inprocess
```

### Run without the CLI on PATH

```bash
python3 -m geniesim_cli teleop run --device_type=pico
# or directly:
python3 -m geniesim_teleop.teleop --device_type=pico
```

---

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `geniesim: command not found` | Use `python3 -m geniesim_cli teleop …` or reinstall `geniesim_cli`. |
| `❌ geniesim_teleop is not importable.` | Editable install missing. Run `pip install -e source/geniesim_teleop`. |
| `ModuleNotFoundError: rclpy` / `geometry_msgs` | ROS 2 isn't sourced. Run inside the Genie Sim container or `source` your ROS overlay. |
| `teleop.yaml not found … skip loading robot init states` | `geniesim_benchmark` isn't installed; init states are optional and skipped. |

---

## 7. Skills

| Skill | Trigger |
|---|---|
| [run-teleop](skills/run-teleop/SKILL.md) | "start teleop", "启动遥操作", "VR 采集", "run the teleop loop" |

---

## 8. Do not

- Don't reintroduce bare `from utils/devices/config …` imports or
  `sys.path.insert` hacks — use absolute `geniesim_teleop.*` imports.
- Don't pin a host/IP inside the code — pass `--host_ip` / `--client_host`
  at invocation time.
- Don't add a hard import of `geniesim_benchmark` at module top level;
  keep the `find_spec` lazy lookup so teleop runs without the benchmark
  package installed.
