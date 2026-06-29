# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
physics_mujoco.launch.py

🚧 **W.I.P. — the MuJoCo backend (``mujoco_geniesim``) is not yet
production-ready.** Scene loading, plugin params, render hookup, and
the assemble pipeline all work end-to-end on simple flat scenes, but
contact-rich + cloth + multi-camera flows have known gaps and are
actively being shaken out. Expect signature changes (CLI args, scene
YAML keys, plugin layout) between commits without notice. Pin a
specific revision if you depend on this launch file in CI.

Sub-launch owning the MuJoCo physics backend (``mujoco_geniesim``).

Spawns:
  * ``assemble_scene`` (no robot pipeline — MuJoCo XML is consumed directly)
  * The MuJoCo physics ``Node`` (gated behind ``assemble_scene`` exit)
  * Optional ``interaction_tools`` node when ``interaction_tools:=true``

This launch file is normally included by the composer ``minimal.launch.py``
but is also runnable standalone for backend-isolated debugging.

State-sharing contract: re-resolves the scene YAML, assets folder, stage
dir, launcher config and robot params from the same ``LaunchConfiguration``
namespace as the composer (no IPC, no global context). Helpers are pure
I/O — re-parsing scene YAML 2-3 times per run is negligible cost.
"""

import os, sys
import importlib.util
from pathlib import Path

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import OpaqueFunction, DeclareLaunchArgument, TimerAction

# Load sibling utils.py via importlib (rather than a plain ``import utils``)
# so an unrelated ``utils`` package elsewhere on sys.path can't shadow it.
_lu_spec = importlib.util.spec_from_file_location(
    "_launch_utils", os.path.join(os.path.dirname(os.path.realpath(__file__)), "utils.py")
)
lu = importlib.util.module_from_spec(_lu_spec)
_lu_spec.loader.exec_module(lu)
MSG_COLOR = lu.MSG_COLOR
ERR_COLOR = lu.ERR_COLOR
RESET = lu.RESET


def generate_launch_description():
    # Runtime W.I.P. banner — printed at launch-file parse time so the user
    # sees it before any node starts. Set ``GENIESIM_WIP_ACK=1`` to silence.
    if os.environ.get("GENIESIM_WIP_ACK") != "1":
        bar = "─" * 64
        print(f"{ERR_COLOR}{bar}{RESET}")
        print(f"{ERR_COLOR}🚧 physics_mujoco.launch.py is W.I.P. —" f" expect rough edges + breaking changes.{RESET}")
        print(f"{ERR_COLOR}   (set GENIESIM_WIP_ACK=1 to silence this banner){RESET}")
        print(f"{ERR_COLOR}{bar}{RESET}")

    declared_arguments = [
        DeclareLaunchArgument(
            name="mujoco_scene",
            default_value="",
            description=(
                "MuJoCo XML scene file (required for mujoco_geniesim). "
                "Resolved against cwd and share/genie_sim_bringup/config/."
            ),
        ),
        DeclareLaunchArgument(
            name="headless",
            default_value="true",
            choices=["true", "false"],
            description="Run MuJoCo physics in headless mode (no GUI window)",
        ),
        DeclareLaunchArgument(
            name="always_regenerate_robot_usd",
            default_value="false",
            choices=["true", "false"],
            description=("If true, force assemble_scene to run even when manifest.json is cached."),
        ),
    ]

    def parse_args(context):
        # ---- pull all CLI args from the parent (composer) context -------
        launch_robot_model = lu.perform(context, "robot_model")
        launch_body = lu.perform(context, "body")
        launch_arm = lu.perform(context, "arm")
        launch_gripper = lu.perform(context, "gripper")
        remap_tf = lu.perform(context, "remap_tf")
        use_sim_time = lu.perform(context, "use_sim_time")
        common_param = {"use_sim_time": True if "true" == use_sim_time else False}
        log_level = lu.perform(context, "log_level").strip()
        ros_log_args = ["--ros-args", "--log-level", log_level]

        nodes: list = []  # gated behind assemble pipeline

        # ---- scene YAML (optional) → robot_source ---------------------
        scene_info = lu.resolve_scene_yaml_robot_params(
            context,
            required=True,
            robot_model=launch_robot_model,
            body=launch_body,
            arm=launch_arm,
            gripper=launch_gripper,
        )
        scene_resolved = scene_info["scene_resolved"]
        resolved = scene_info["resolved"]
        gripper = resolved["gripper"]
        lgripper = gripper

        # ---- MuJoCo XML scene (required) -------------------------------
        mujoco_scene_arg = lu.perform(context, "mujoco_scene").strip()
        mujoco_scene_resolved = lu.resolve_bringup_config_file(mujoco_scene_arg) if mujoco_scene_arg else ""
        if not mujoco_scene_resolved:
            print(f"{ERR_COLOR}mujoco_scene is required for physics_mujoco.launch.py{RESET}")
            sys.exit(1)
        print(f"{MSG_COLOR}[physics_mujoco] mujoco_scene: {mujoco_scene_resolved}{RESET}")

        # ---- assets folder + stage dir ---------------------------------
        assets_folder = lu.resolve_assets_folder()
        always_regenerate_robot_usd = lu.perform(context, "always_regenerate_robot_usd").strip().lower() == "true"
        lu.discover_mujoco_plugin_dir()

        # ---- launcher YAML (physics engine binding + plugin params) ----
        launcher_config_arg = lu.perform(context, "launcher_config").strip()
        plugins_cfg_path = lu.resolve_bringup_config_file(launcher_config_arg) if launcher_config_arg else ""
        plugins_params_path = ""
        launcher_section: dict = {}
        if plugins_cfg_path:
            launcher_section, _, plugins_params_path = lu.load_launcher_yaml(plugins_cfg_path)

        try:
            physics_engine, physics_package, physics_executable, physics_node_name = lu.resolve_physics_engine(
                launcher_section,
                default_engine="mujoco_geniesim",
            )
        except ValueError as exc:
            print(f"{ERR_COLOR}{exc}{RESET}")
            sys.exit(1)
        print(
            f"{MSG_COLOR}[physics_mujoco] engine: {physics_engine} "
            f"(package={physics_package}, executable={physics_executable}, name={physics_node_name}){RESET}"
        )

        scene_stem = Path(mujoco_scene_resolved).stem
        stage_dir = os.path.abspath(os.path.join(assets_folder, "scenes", scene_stem))

        if scene_resolved:
            lu.stage_yaml_snapshot(stage_dir, scene_resolved)

        # ---- MuJoCo physics node ---------------------------------------
        fake_slam = lu.perform(context, "fake_slam")
        headless = lu.perform(context, "headless")

        remapping_mujoco = [
            ("/imu", "/imu/livox_back"),
            ("/wheel_command", "/pnc/chassis_joint_cmd"),
            ("/wheel_states", "/hal/chassis_joint_state"),
        ]
        if "true" == remap_tf:
            remapping_mujoco.append(("/tf", "/tf_sim"))

        mujoco_parameters: list = []
        if plugins_params_path:
            mujoco_parameters.append(plugins_params_path)
        mujoco_parameters.append(
            {
                **common_param,
                "fake_slam": "true" == fake_slam,
                "lgripper": lgripper,
                "headless": "true" == headless,
            }
        )

        # --gui must precede --ros-args; placing it after causes ROS2 to treat
        # it as an unknown ROS argument and abort.
        pre_ros_args = ["--gui"] if headless != "true" else []

        nodes.append(
            TimerAction(
                period=2.0,
                actions=[
                    Node(
                        package=physics_package,
                        executable=physics_executable,
                        name="genie_sim_engine",
                        output="both",
                        parameters=mujoco_parameters,
                        arguments=[mujoco_scene_resolved, *pre_ros_args, *ros_log_args],
                        remappings=remapping_mujoco,
                    )
                ],
            )
        )

        # ---- interaction_tools ------------------------------------------
        interaction_tools = lu.perform(context, "interaction_tools")
        if interaction_tools == "true":
            nodes.append(
                Node(
                    package="genie_sim_tools",
                    executable="interaction_tools.py",
                    name="interaction_tools",
                    output="both",
                    parameters=[
                        {
                            "mujoco_scene": mujoco_scene_resolved,
                            **common_param,
                        }
                    ],
                    arguments=ros_log_args,
                )
            )

        # ---- render nodes -----------------------------------------------
        stage_manifest = os.path.join(stage_dir, "manifest.json")
        renders_from_yaml: list = []
        raw_renders = launcher_section.get("renders") or []
        if isinstance(raw_renders, list):
            renders_from_yaml = [str(r).strip() for r in raw_renders if str(r).strip()]
        active_renders = lu.resolve_active_renders(renders_from_yaml)
        print(f"{MSG_COLOR}[physics_mujoco] render (active={sorted(active_renders)}){RESET}")

        if "render_ovrtx" in active_renders:
            nodes.append(
                lu.make_render_ovrtx_node(
                    stage_manifest=stage_manifest,
                    physics_node_name="genie_sim_engine",
                    plugins_params_path=plugins_params_path,
                    common_param=common_param,
                    ros_log_args=ros_log_args,
                )
            )
        if "render_isaacsim" in active_renders:
            nodes.append(
                lu.make_render_isaacsim_node(
                    stage_manifest=stage_manifest,
                    plugins_params_path=plugins_params_path,
                    common_param=common_param,
                    ros_log_args=ros_log_args,
                )
            )

        # ---------------------------------------------------------------
        # Assemble pipeline MUST be built AFTER all ``nodes.append(...)``
        # because ``make_assemble_pipeline`` snapshots ``runtime_nodes``
        # via ``list(...)`` at call time.
        # ---------------------------------------------------------------
        assemble_scene_proc, _, _, gate_runtime = lu.make_assemble_pipeline(
            scene_resolved=scene_resolved,
            stage_dir=stage_dir,
            assets_folder=assets_folder,
            always_regenerate_robot_usd=always_regenerate_robot_usd,
            runtime_nodes=nodes,
            assemble_robot_artefacts=None,
        )
        return [assemble_scene_proc, gate_runtime]

    return LaunchDescription(
        [
            *declared_arguments,
            OpaqueFunction(function=parse_args),
        ]
    )
