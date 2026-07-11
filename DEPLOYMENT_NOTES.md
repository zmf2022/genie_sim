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
ros2 pkg list | grep genie_sim     # 应列出 ~11 个包
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

需要先在容器内安装 teleop 包:

```bash
pip install -e source/geniesim_teleop/
```

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

## Step 9(可选):自动化数据采集(data_collection)

基于 Isaac Sim 5.1 + cuRobo 的自动化轨迹采集,产出 agibot 格式的 episode 数据(`aligned_joints*.h5` + `camera/` + `observations/` + `data_info.json`)。

**前提**:主机需要 editable-install `geniesim_assets` 并激活 conda 环境。

```bash
# 1. 安装 geniesim_assets(只需一次)
pip install -e geniesim_assets

# 2. 激活 conda 环境
conda activate genie_sim

# 3. 查看可用任务(~295 个)
geniesim autocollect list                        # 全部 task
geniesim autocollect list --robot=g2 sort_fruit  # 按机器人/任务过滤
geniesim autocollect tasks                       # 查看任务族
geniesim autocollect robots                      # 查看机器人(g1/g2)

# 4. 预览任务(不启动容器)
geniesim autocollect run sort_the_fruit_into_the_box_apple_g2 --dry-run

# 5. 无人值守采集(推荐)
geniesim autocollect run sort_the_fruit_into_the_box_apple_g2 --headless --standalone
```

**常用 run 参数**:

| 参数 | 说明 |
|---|---|
| `--headless` | 无 GUI,无 X11 主机必须 |
| `--standalone` | 仅写文件日志,不输出终端 |
| `--no-record` | 禁用录制 |
| `--dry-run` | 只打印命令,不启动容器 |
| `--container-name=N` | 自定义容器名(默认 `data_collection_open_source`) |

**数据产出**:

- 每个 episode 约 **1.5–2.2 GB**,位于 `source/data_collection/recording_data/[<task>_<index>]/`
- 包含:`camera/`(原始 mcap,~2G) + `observations/`(视频帧,~120M) + `aligned_joints.h5`(关节轨迹) + `data_info.json`(动作标签) + `state.json`
- 日志位于 `source/data_collection/logs/<task>/`

**关键细节**:

- 运行方式:主机 CLI 自动 `docker run -d` 拉起专用镜像(`geniesim3-data-collection:latest`),容器内跑两个进程(Isaac Sim server + task client)通过 gRPC 通信。**不是在已有容器里 exec**,而是独立拉起一个临时容器,退出后自动清理。
- 录制数据通过 bind-mount 直接落盘到主机,不需要从容器拷贝。
- cuRobo motion planning 偶发失败(`Stage X place fail at first step, try next sequence`)是正常现象,自动重试多个 sequence;每个 episode 约 1–3 分钟。
- **cuRobo 世界缓存会在单次容器里慢慢累积**,跑 6-8 个 task 后成功率会断崖式下跌(0% 失败)。官方 task JSON 里 `num_of_episode=8` 就是为了匹配这个窗口。**大批量采集时,要循环重启容器**,比如采 3000 条 = 375 批 × 8 条/批(`for i in $(seq 1 375); do geniesim autocollect run <TASK> --headless --standalone; docker stop/rm data_collection_open_source; sleep 5; done`)。
- 首次构建镜像:`geniesim autocollect build`(依赖 `geniesim3:latest` 基础镜像已就绪)。

---

## 常见坑

- **主机只装 CLI**,别装其他 tier-1 包,那些都在容器里。
- **每次进容器都要 source setup.bash**,不然 ros2 launch 找不到包。
- **assemble_robot 失败 + `SimulationApp = None`** 重新编译镜像就好,Dockerfile 已经修了 NVIDIA 基础镜像的遮蔽 bug。
