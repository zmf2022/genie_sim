# Genie Sim 部署笔记

---

## Step 1:主机装 CLI

安装轻量级 CLI 工具,后续所有命令(`docker` / `ros` / `benchmark` / `teleop`)都通过它调度。

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
pip install -e source/geniesim_cli/
```

验证:`geniesim --help` 打出帮助即可。

---

## Step 2:编译 Docker 镜像

基于 `docker/Dockerfile.5.1` 构建 Isaac Sim 5.1 + 6.0 组合镜像,首次会拉取 NVIDIA 基础镜像(约 30GB),总耗时 1-2 小时。

```bash
geniesim docker build --china    # 国内环境(走 DaoCloud + tuna)
# 或
geniesim docker build            # 海外直连
```

---

## Step 3:启动并进入容器

启动容器并把宿主机仓库 bind-mount 进去,entrypoint 会自动 editable-install 所有 tier-1 包。

```bash
geniesim docker up
geniesim docker into
```

进容器后跑一次健康检查:

```bash
geniesim status
geniesim doctor
```

---

## Step 4:构建 ROS 工作空间

colcon 编译 `genie_sim_*` 这套 ROS 2 包,产出 `devel/` overlay。每次开新 shell 都要重新 `source devel/setup.bash`。

```bash
geniesim ros build dev
source devel/setup.bash
ros2 pkg list | grep genie_sim     # 应列出 ~15 个包
```

---

## Step 5:启动仿真场景

把"场景 + 物理引擎 + 渲染器"组合启动。`scene` 选机器人和环境(`pnp_g2_op` = G2 + omnipicker 的 pick-and-place),`launcher_config` 选物理后端(`ovrtx_isaac_physx` 最稳定),`headless:=false` 要 GUI。

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_pnp_g2_op \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false
```

首次启动 shader 编译要 5-10 分钟。如果 GUI 卡顿,加 `physics_hz:=50 render_hz:=15` 降频(默认 100/30 对 48GB 卡压力大)。

---

## Step 6(可选):跑 Benchmark

前提是你自己起了一个推理服务器,监听 `IP:PORT`。这是 AgiBot World Challenge 的评测流程。

```bash
geniesim benchmark check-inference --infer-host=<IP>:<PORT>
geniesim benchmark run g2op_if_pick_block_color --infer-host=<IP>:<PORT>
geniesim benchmark categories    # 看可用任务类别
```

---

## Step 7(可选):VR 遥操作

用 Pico VR 头显控制仿真里的机器人,同时录制 episodes。前提是有 VR 设备且能访问容器。

```bash
geniesim teleop run --device_type=pico --port=8080
```

---

## Step 8(可选):geniesim_world(主机 conda)

把一张全景照片转成可探索的 3D 高斯世界,给仿真提供真实环境背景。**不在容器里跑**,独立在主机装 conda。前提:主机有 GPU,`source/external/` 下已 clone `ml-sharp` 和 `DA360`(含 checkpoint)。

```bash
conda create -n geniesim_world python=3.11 -y
conda activate geniesim_world

cd source/geniesim_world
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  -r requirements-cu128.txt
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .

geniesim_world create --panorama path/to/pano.png --work-dir runs/demo
```

---

## 常见坑

- **主机只装 CLI**,别装其他 tier-1 包,那些都在容器里。
- **每次进容器都要 source setup.bash**,不然 ros2 launch 找不到包。
- **assemble_robot 失败 + `SimulationApp = None`** 重新编译镜像就好,Dockerfile 已经修了 NVIDIA 基础镜像的遮蔽 bug。
