# geniesim_teleop — repository-level dispatcher

> 🧭 **Canonical source**: [`source/geniesim_teleop/AGENTS.md`](../source/geniesim_teleop/AGENTS.md) — the per-package guide is the source of truth for the device-driver layer, gRPC client, robot configs, and rosbag → HDF5 pipeline.

This file is a 30-second pointer. Do not duplicate content here that lives at the canonical source — duplication is what makes dispatchers rot.

---

## What it is

VR / Pico teleoperation: streams device poses into the simulator over ROS 2 via the `geniesim teleop` CLI verb (owned by [`geniesim_cli`](geniesim_cli.md)). The teleop loop publishes `/joint_command` directly to the running RT Engine scene; per-episode recording artifacts land under `--record-dir`.

## Where to look

| Topic | File |
|---|---|
| Canonical CLI surface + device protocol + robot configs | [`source/geniesim_teleop/AGENTS.md`](../source/geniesim_teleop/AGENTS.md) |
| User-facing intro | [`source/geniesim_teleop/README.md`](../source/geniesim_teleop/README.md) |
| Agent skill (run-teleop) | [`source/geniesim_teleop/skills/run-teleop/SKILL.md`](../source/geniesim_teleop/skills/run-teleop/SKILL.md) |
| Bridging into the RT Engine | [`source/geniesim_ros/skills/teleop-bridge/SKILL.md`](../source/geniesim_ros/skills/teleop-bridge/SKILL.md) |

## Invariants the rest of the repo relies on

- **One publisher on `/joint_command`.** When teleop is driving, MoveIt must launch with `use_ros2_control:=false` so move_group only serves `/compute_ik` / `/compute_fk` — never spawn the ros2_control controllers in this mode, or they fight the teleop publisher at the CM update rate (arm jitters, payloads fly). See the [`teleop-bridge`](../source/geniesim_ros/skills/teleop-bridge/SKILL.md) skill.
- **Robot config ↔ scene gripper must match.** `--robot_config=G2_omnipicker.json` requires the scene to be built with the omnipicker gripper (e.g. `scene_pnp_g2_op`). Mismatched grippers produce a robot MoveIt can plan against but the engine refuses to drive.
- **Default VR port is 8080.** The teleop loop opens a VR server and waits for the Pico headset. Changing the default port means updating downstream firewall / docker port-publish rules.
- **Runs inside the container.** Needs ROS 2 + Isaac Sim — i.e. `geniesim docker into`. Outside the container, source the overlay first.
