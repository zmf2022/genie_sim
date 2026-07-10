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
- 首次构建镜像:`geniesim autocollect build`(依赖 `geniesim3:latest` 基础镜像已就绪)。

### GPU 配置（建议单卡）

瓶颈在 CPU(agent ~1800%、curobo ~700% CPU),两张卡各仅 ~30% util,远未吃满。**单卡不会变快**,但默认双卡会被 Omniverse 跨卡分摊、引入同步开销还白占一张卡。限制单卡把另一张腾出来:

```bash
export CUDA_VISIBLE_DEVICES=0   # 容器内临时;持久改宿主机 run_data_collection.sh 的 --gpus all → --gpus "device=0"
```

> 真提速靠降 CPU:候选数 300→150、去 grasp_upper_percentile、降 MAX_ATTEMPTIONS。

---

## 常见坑

- **主机只装 CLI**,别装其他 tier-1 包,那些都在容器里。
- **每次进容器都要 source setup.bash**,不然 ros2 launch 找不到包。
- **assemble_robot 失败 + `SimulationApp = None`** 重新编译镜像就好,Dockerfile 已经修了 NVIDIA 基础镜像的遮蔽 bug。

---

## 数据采集调优经验（lab_dual_arm_sample_transfer_g2）

参考模板 `sort_the_fruit_into_the_box_apple_g2`,改完验证有效,记录如下。

### 现象
原配置跑 500 条几乎 0 成功:98 个已结束任务全 `failed`,0 `success`;抓取的 `pick` 第一步就 `pick fail at first step`,反复重试 4 个候选都失败。

### 根因
任务 pick 阶段设了 `grasp_upper_percentile: 75`(官方范例 `TASK_CONFIG_GUIDE.md` 就这么写,但默认值其实是 100)。它按抓取点高度过滤,砍掉最高的 25% 候选——而苹果在桌面上最稳的正是最竖直/top-down 的抓取。剩下的侧/低位抓取在 `AvoidObs` 避障规划下接近路径必撞桌子,planner 偏移 → 夹爪到达位姿偏离规划 grasp pose → `simple_check_completion`(pos<0.1 & angle<70°)判失败。

### 改动(均已验证生效)
1. 任务 json(`tasks/geniesim_2025/lab_sample_transfer/g2/...` 及参考模板)去掉 `grasp_upper_percentile`,`grasp_offset` 由 0.01 改 0.0,对齐官方成熟任务(`pick_specific_object` 等不设这个字段=默认 100)。
2. `client/agent/omniagent.py:846` 遍历 `saved_task/*.json` 加 `sorted(..., key=数字)`,避免每次重启从乱序索引(曾出现从 296 开始)跑,保证从 `_0` 顺序执行。

> 注意:这两处是未提交改动,`git reset --hard` 会冲掉(已踩过一次)。建议提交。

### 验证结果
- 重启后首个任务 `_0` 即 `TASK SUCCESS`(之前 0/98),证明修复有效。
- 排序生效:`_0 → _1 → …` 顺序跑。
- 耗时 ~90s/episode(成功任务含 pick+place+reset 全套,略长于失败任务),`num_of_episode=800` 时 ≈20h。
- 日志无 error。

### GPU / 性能结论
- 双 RTX5880 各仅 ~30% util、显存 2.7~2.9G/48G;瓶颈在 CPU(agent ~1800%、curobo server ~700% CPU)。**单卡不提速**,但默认 `--gpus all` 双卡会被 Omniverse 跨卡分摊、增同步开销且白占一张卡 → 限制单卡腾卡(见上「GPU 配置」)。
- 重活(curobo CUDA / PhysX GPU / RTX 录制)已在 GPU,喂不饱是因单机器人串行执行 + Python 编排。提速靠减 CPU 工作量,不是加 GPU。

### 待做优化(未实施)
- **奇偶并行两卡**:实例 A 跑偶数索引、B 跑奇数,各绑一张卡(`CUDA_VISIBLE_DEVICES=0/1`),吞吐 ~2x(≈10h)。需在 `agent.run` 加奇偶/stride 过滤。并行不影响渲染结果(episode 间独立仿真)。
- 减抓取候选:`client/planner/action/grasp.py` 的 `random_downsample(grasp_poses, 300, ...)` → 150,IK/规划量减半。
- 降重试:`client/agent/omniagent.py` 的 `MAX_ATTEMPTIONS`(当前约 4 次),成功率上来后调小。
- 批量规划:把所有抓取候选一次性 batch 进一个 curobo CUDA 调用,挑最优执行一次(执行仍串行)。

