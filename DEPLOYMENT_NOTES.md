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

```bash
ros2 launch genie_sim_bringup app.launch.py \
  scene:=scene_pnp_g2_op \
  launcher_config:=launcher_ovrtx_isaac_physx \
  headless:=false
```

## 6. 跑 Benchmark(可选)

```bash
geniesim benchmark check-inference --infer-host=<IP>:<PORT>
geniesim benchmark run g2op_if_pick_block_color --infer-host=<IP>:<PORT>
geniesim benchmark categories
```

## 7. VR 遥操作(可选)

```bash
pip install -e source/geniesim_teleop/
geniesim teleop run --device_type=pico --port=8080
```

## 8. geniesim_world(可选,主机 conda)

```bash
conda create -n geniesim_world python=3.11 -y && conda activate geniesim_world
cd source/geniesim_world
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-cu128.txt
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .

geniesim_world create --panorama path/to/pano.png --work-dir runs/demo
```

## 9. 数据采集(可选)

```bash
pip install -e geniesim_assets    # 只需一次
conda activate genie_sim

geniesim autocollect list                                      # 查看任务
geniesim autocollect run <TASK> --headless --standalone        # 采集
geniesim autocollect run <TASK> --dry-run                      # 只预览命令
```

| 参数 | 作用 |
|---|---|
| `--headless` | 无 GUI |
| `--standalone` | 只写文件日志 |
| `--no-record` | 不录制 |
| `--dry-run` | 只打印不执行 |
| `--container-name=N` | 自定义容器名 |

产物在 `source/data_collection/recording_data/[<task>_<index>]/`,每条 ~1.5–2.2 GB。

**注意**:cuRobo 缓存在单次容器 6–8 task 后失败率飙升,大批量需循环重启容器:

```bash
for i in $(seq 1 375); do
  geniesim autocollect run <TASK> --headless --standalone
  docker stop data_collection_open_source && docker rm data_collection_open_source
  sleep 5
done
```

## 10. 转 LeRobot(可选)

```bash
pip install -e source/geniesim_benchmark   # 只需一次

# 默认 VLA 格式:state/action 各 16 维,3 路 RGB
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir source/data_collection/recording_data/ \
  --output-dir lerobot_out/<my_dataset>

# 完整 agibot 格式:state 159 维 / action 40 维,6 路视频
geniesim dataset convert agibot-to-lerobot \
  --agibot-dir source/data_collection/recording_data/ \
  --output-dir lerobot_out/<my_dataset> \
  --format agibot
```

| `--format` | state | action | 视频 |
|---|---|---|---|
| `vla`(默认) | 16 维 | 16 维 | 3 路 RGB |
| `agibot` | 159 维 | 40 维 | 3 RGB + 3 depth |

---

## 常见坑

- 主机只装 CLI,不装其他 tier-1 包(那些在容器里)
- 每次进容器都要 `source devel/setup.bash`
- `assemble_robot` 失败 + `SimulationApp = None` → 重新编译镜像
