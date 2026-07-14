# Genie Sim 部署笔记

## 1. 主机装 CLI

```bash
git clone https://github.com/AgibotTech/genie_sim.git
cd genie_sim
pip install -e source/geniesim_cli/
```

## 2. 编译 Docker 镜像

```bash
geniesim docker build --china    # 国内
geniesim docker build            # 海外
```

## 3. 启动容器

```bash
geniesim docker up
geniesim docker into
geniesim status
```

## 4. 构建 ROS 工作空间

```bash
geniesim ros build dev
source devel/setup.bash          # 每次开新 shell 都要
```

## 5. 启动仿真

> **注意**: 以下命令需要在容器内执行（先完成第 3 节进入容器，第 4 节 source 环境）

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_pnp_g2_op \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false
```

## 6. 跑 Benchmark(可选)

> **注意**: 以下所有命令都需要在 Docker 容器内执行（确保已执行第 3 节进入容器，并完成第 4 节的 `source devel/setup.bash`）

> **参数命名规则**: `ParameterServer` 只接受带前缀的参数格式（如 `--benchmark.preview`），不支持简写（如 `--preview`）。未知参数会被静默忽略，不会报错。

### 6.1 协议连通性检查

```bash
geniesim benchmark check-inference --infer-host=127.0.0.1:8000
# 期望输出: ✅ PASS — inference server reachable and response valid
```

### 6.2 多轮推理测试

```bash
geniesim benchmark check-inference --infer-host=127.0.0.1:8000 --iters 10
# 检查延迟统计: min/mean/p95/max 是否稳定
```

### 6.3 数据格式验证(debug 模式)

```bash
geniesim benchmark run g2op_if_pick_block_color \
  --infer-host=127.0.0.1:8000 --benchmark.debug
```

### 6.4 图像可视化(preview 模式)

```bash
geniesim benchmark run g2op_if_pick_block_color \
  --infer-host=127.0.0.1:8000 --benchmark.preview

# 生成 preview_<step>_<timestamp>_<camera>.png
# 肉眼检查图像是否清晰、无花屏/黑屏
```

### 6.5 端到端完整测试

```bash
geniesim benchmark run g2op_if_pick_block_color \
  --infer-host=127.0.0.1:8000

# 观察:
# - 机器人手臂是否跟随模型输出运动
# - 夹爪是否在正确时机开合
# - 任务是否完成(如抓取成功率)
```

### 6.6 查看可用任务

```bash
geniesim benchmark categories
```

### 验证清单

| 步骤 | 检查项 | 预期 |
|---|---|---|
| check-inference | 协议连通 | ✅ PASS |
| check-inference --iters | 延迟稳定 | p95 < 200ms |
| debug pkl | state shape | (16,) |
| debug pkl | image shape | (3, 480, 640) |
| preview 图像 | 图像质量 | 清晰无花屏 |
| 完整 run | 机器人响应 | 手臂跟随运动 |
| 完整 run | 夹爪动作 | 正确时机开合 |
| 完整 run | 任务完成 | 抓取成功率 > 0% |

## 7. VR 遥操作(可选)

> **注意**: 以下命令在主机执行（不需要进入容器）

```bash
pip install -e source/geniesim_teleop/
geniesim teleop run --device_type=pico --port=8080
```

## 8. geniesim_world(可选,主机 conda)

> **注意**: 以下命令在主机执行（不需要进入容器），使用主机 conda 环境

```bash
conda create -n geniesim_world python=3.11 -y && conda activate geniesim_world
cd source/geniesim_world
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-cu128.txt
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .

geniesim_world create --panorama path/to/pano.png --work-dir runs/demo
```

## 9. 数据采集(可选)

> **注意**: 以下命令在主机执行（不需要进入容器），采集框架会自动启动临时 Docker 容器进行仿真

### 9.1 环境准备

```bash
pip install -e geniesim_assets    # 只需一次
conda activate genie_sim
```

### 9.2 查看可用任务

```bash
geniesim autocollect list                        # 所有任务
geniesim autocollect list --task g2              # 筛选 G2 机器人任务
geniesim autocollect tasks                       # 任务类别汇总
```

### 9.3 预览采集命令（dry-run）

```bash
geniesim autocollect run <TASK> --dry-run
# Example: geniesim autocollect run lab_dual_arm_sample_transfer_g2 --dry-run
```

### 9.4 单任务采集

```bash
geniesim autocollect run <TASK> --headless --standalone
```

**参数说明**:

| 参数 | 作用 |
|---|---|
| `--headless` | 无 GUI,适合服务器环境 |
| `--standalone` | 只写文件日志,不输出到终端 |
| `--no-record` | 不录制数据(只跑任务) |
| `--container-name=N` | 自定义容器名 |

产物在 `source/data_collection/recording_data/[<task>_<index>]/`,每条 ~1.5–2.2 GB。

### 9.5 批处理采集（循环重启）

> **重要**: cuRobo 缓存在单次容器运行 6–8 个 task 后失败率飙升,大批量必须循环重启容器清理状态

使用仓库自带的批处理脚本:

```bash
bash source/data_collection/scripts/run_batch_collect.sh <TASK> <REPEATS>

# Example: 采集 lab_dual_arm_sample_transfer_g2 任务 100 批
bash source/data_collection/scripts/run_batch_collect.sh lab_dual_arm_sample_transfer_g2 100
```

脚本会自动:
- 循环运行 `<TASK>` 指定次数
- 每批结束后重启容器清理 cuRobo 缓存
- 实时显示已采集的 recording 数量
- 每批间隔 5 秒,避免容器状态未完全释放

### 9.6 验证采集结果

```bash
# 查看总采集数
ls -d source/data_collection/recording_data/<task>_* | wc -l

# 随机检查一条数据完整性
TASK="lab_dual_arm_sample_transfer_g2"
INDEX=0  # 检查第 0 条
DIR="source/data_collection/recording_data/${TASK}_${INDEX}"

# 必需文件
ls "$DIR/aligned_joints_all.h5"          # HDF5 关节数据
ls "$DIR/aligned_extrinsics.json"        # 外参配置
ls "$DIR/images/" | head -5              # 图像文件
```

**验证清单**:

| 检查项 | 命令 | 预期 |
|---|---|---|
| 总采集数 | `ls -d recording_data/<task>_* \| wc -l` | ≥ 预期数量 |
| HDF5 完整 | `h5py <DIR>/aligned_joints_all.h5` | 可读,含 joint/image 数据 |
| 外参 JSON | `cat <DIR>/aligned_extrinsics.json` | 有效的 JSON 格式 |
| 图像目录 | `ls <DIR>/images/ \| wc -l` | ≥ 数百张图 |
| 单条大小 | `du -sh <DIR>` | 1.5–2.2 GB |

### 9.7 常见问题

**错误: "cuRobo plan failed" 或任务成功率突降**
* 原因: cuRobo 缓存在单次容器 6–8 个 task 后失败率飙升
* 方案: 使用 9.5 节的批处理脚本,每批只跑 1 task 并重启容器

**错误: "docker exec: container not running"**
* 原因: 容器意外终止
* 方案: `docker stop data_collection_open_source && docker rm data_collection_open_source`,然后重试

**产物目录为空**
* 原因: 没有添加 `--standalone`,日志只打印到终端没落盘
* 方案: 加上 `--standalone` 参数

---

## 10. 转 LeRobot(可选)

> **注意**: 以下命令在主机执行（不需要进入容器）

```bash
pip install -e source/geniesim_benchmark   # 只需一次

# 默认 VLA 格式:state/action 各 16 维,3 路 RGB
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir source/data_collection/recording_data/ \
  --output-dir lerobot_out/<my_dataset> \
  --task "Use your closest hand to sort the apple into the corresponding storage box"

# 完整 agibot 格式:state 159 维 / action 40 维,6 路视频
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir source/data_collection/recording_data/ \
  --output-dir lerobot_out/<my_dataset> \
  --format agibot \
  --task "Use your closest hand to sort the apple into the corresponding storage box"
```

**格式对比**:

| `--format` | state | action | 视频 |
|---|---|---|---|
| `vla`(默认) | 16 维 | 16 维 | 3 路 RGB |
| `agibot` | 159 维 | 40 维 | 3 RGB + 3 depth |

**`--task` 参数**:VLA 模型(gr00t、pi0.5 等)将任务指令作为语言条件输入。
写入 `meta/tasks.jsonl` 和 `meta/info.json` 的 `high_level_instruction` 字段。
不填则语言字段为空。应根据实际任务填写自然语言描述。

**验证转换结果**:

```bash
# 查看元数据
cat lerobot_out/<my_dataset>/meta/info.json

# 检查数据集条数
python -c "from datasets import load_from_disk; ds = load_from_disk('lerobot_out/<my_dataset>'); print(f'Episodes: {len(ds)}')"
```

---

## 常见坑

- 主机只装 CLI,不装其他 tier-1 包(那些在容器里)
- 每次进容器都要 `source devel/setup.bash`
- `assemble_robot` 失败 + `SimulationApp = None` → 重新编译镜像
