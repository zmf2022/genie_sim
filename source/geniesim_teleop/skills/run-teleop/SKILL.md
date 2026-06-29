---
name: run-teleop
description: >
  Launch the geniesim_teleop VR / Pico teleoperation loop (or the
  in-process image bridge) using the `geniesim teleop` CLI verb, typically
  inside the Genie Sim GUI Docker container.
  Trigger: When the user asks to "start teleop", "run teleop", "启动遥操作",
  "VR 采集", "遥操作采集", "drive the robot with the VR headset", "launch the
  teleop loop", or wants to run anything under `geniesim_teleop`.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites:
  - geniesim_cli:fresh-machine-setup   # see source/geniesim_cli/AGENTS.md § 0
inputs:
  - name: device_type
    desc: VR device family
    required: false
    default: pico
  - name: port
    desc: VR server port (where the headset connects)
    required: false
    default: "8080"
  - name: robot_config
    desc: Robot config JSON (gripper variant)
    required: false
    default: G2_omnipicker.json
  - name: grpc_host
    desc: gRPC client host
    required: false
    default: "localhost:50051"
  - name: record_dir
    desc: Per-episode artifact output dir
    required: false
outputs:
  - desc: Live teleop loop streaming device poses to `/joint_command`; per-episode artifacts under `record_dir` if set
---

## When to Use

- User wants to teleoperate the simulated robot with a VR device (Pico) and
  optionally record episodes.
- User references `geniesim_teleop`, `teleop.py`, or the teleop bridge.

Do **not** use for:
- Running a benchmark task → `run-benchmark`.
- Verifying an inference server → `check-inference`.

## Critical Patterns

1. **The runtime needs ROS 2 + Isaac Sim on the host.** Inside the
   Genie Sim Docker image (`geniesim docker up` → `geniesim docker into`)
   that's already set up. Outside the container, source your ROS overlay
   and have Isaac Sim available.
2. **A VR device must be reachable.** The teleop loop opens a VR server
   (default port `8080`) and waits for the Pico headset to connect.
3. **Working directory**: anywhere under the repo works — the CLI uses
   `find_spec` to locate the `geniesim_teleop` package.
4. **Confirm before launching.** Teleop holds a GPU and a live device
   connection; ask before kicking it off if there's any ambiguity.

## Workflow

### Step 1 — Collect inputs

Ask via `AskUserQuestion` (all optional — sensible defaults exist):
- **Device type** (default `pico`).
- **VR port** (default `8080`).
- **Robot config** (default `G2_omnipicker.json`).
- **gRPC client host** (default `localhost:50051`).

### Step 2 — Launch the teleop loop

Inside the GUI container (`geniesim docker into`):

```bash
geniesim teleop run --device_type=pico --port=8080
```

With explicit overrides:

```bash
geniesim teleop run \
    --client_host=localhost:50051 \
    --port=8080 \
    --robot_cfg=G2_omnipicker.json \
    --device_type=pico
```

### Step 3 — (optional) Image bridge

If the user needs the in-process image pub/sub bridge:

```bash
geniesim teleop bridge --mode inprocess
```

## Commands (copy-paste summary for the user)

```bash
# Terminal A — host (start GUI container)
cd /path/to/main
./scripts/start_gui.sh

# Terminal B — host, then container
cd /path/to/main
./scripts/into.sh
# inside container:
geniesim teleop run --device_type=pico --port=8080
```

## Notes

- If `geniesim` isn't on `$PATH`, substitute `python3 -m geniesim_cli teleop …` — same args.
- Any unknown `--flag` after the subcommand is forwarded verbatim to
  `geniesim_teleop.teleop` / `geniesim_teleop.bridge`.
- Robot init states are loaded from the `geniesim_benchmark` package when
  it's installed; if it isn't, teleop still runs (init-state loading is
  skipped with a warning).
