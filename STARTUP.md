# STARTUP：在另一台开发机启动 Dynamic 实验

本文对应 GitHub 仓库 `wuhao0120/Risk-aware-RL` 的 `DCA` 分支。目标是在新机器上使用相同科学源码和协议启动独立实验，再把结果与原机器汇总。导出时间和原分支 commit 记录在 `provenance/export.json`。

## 1. 拉取分支

第一次到新机器用 clone；已有这份导出的 DCA checkout 才用 pull。

```bash
git clone --single-branch --branch DCA --depth 1 \
  https://github.com/wuhao0120/Risk-aware-RL.git
cd Risk-aware-RL
git status --short --branch
```

此 DCA 是只含当前源码的新快照历史。不要在原机器那份有数百 GB 历史和未提交实验的旧 DCA 工作区直接 pull 或 merge；需要同步时另建上述 checkout。

## 2. 平台与镜像选择

已有 NVIDIA 驱动时无需重装驱动。先执行：

```bash
nvidia-smi
uname -m
```

当前已运行环境：Linux x86_64、Python 3.10.20、PyTorch 2.3.1+cu121、MuJoCo 3.12.0、mujoco-warp 3.12.0、warp-lang 1.16.0、NumPy 1.23.3；原机器 GPU 是 A100 80 GB，驱动 535.129.03。其他 GPU 的显存需求和速度需通过本机 smoke 检查；从单任务/单 GPU 开始。

可以直接选择 **PyTorch 2.3.1 + CUDA 12.1 + Python 3.10** 的 Linux 镜像。镜像名称不够，进入环境后检查实际值。CUDA 是 PyTorch 编译的 runtime 版本，`nvidia-smi` 显示的 CUDA Version 是驱动能力，不要求该数字恰好为 12.1。

不要把 PyTorch 2.6.x 当成已经验证的等价实验环境。当前训练/评估源码对 checkpoint 加载已显式使用 `weights_only=False`，避开了 [PyTorch 2.6 默认加载行为的变化](https://docs.pytorch.org/docs/2.6/notes/serialization.html#torch-load-with-weights-only-true)，但尚未验证 2.6 下全部训练与数值行为。为可比较的正式结果使用 2.3.1；原来的 2.6 环境可以保留。

Ubuntu/Debian 缺少系统库时安装以下依赖；有这些包的镜像可以跳过。没有 sudo 时请让平台管理员补装，或选择包含这些库的镜像。

```bash
sudo apt-get update
sudo apt-get install -y git curl ca-certificates build-essential pkg-config \
  libgl1 libegl1 libglfw3 libosmesa6 libsdl2-dev libsdl2-image-dev \
  libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev
```

实验使用 headless GPU 物理，`MUJOCO_GL=disable` 由启动器设置，无需桌面或浏览器。预留依赖下载/安装空间以及 checkpoint 空间，初次至少准备 20 GB 可用空间；正式实验根据 run 数持续检查 `df -h` 和 `du -sh runs`。

## 3A. 已有符合版本的 PyTorch 镜像：保留 PyTorch

推荐用于本次迁移的新机器。进入专用于该实验的镜像环境，检查：

```bash
python -c 'import sys,torch; print(sys.version); print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

期望 Python 3.10、torch `2.3.1` 或 `2.3.1+cu121`、CUDA `12.1`、`True`。然后：

```bash
bash tools/install.sh --image-torch
python tools/doctor.py --image-torch --gpu 0
```

此模式不会安装/替换 `torch`、`triton` 或 `nvidia-*` runtime 包，只补装锁定的项目依赖。它适用于镜像由 pip 或 conda 提供的 PyTorch/CUDA runtime；因此诊断记录实际 runtime 版本，而不要求镜像中的 CUDA 组件都以 pip 包存在。其他项目依赖会按锁文件调整版本，请使用专用实验镜像。

## 3B. 从零建立隔离环境

已有 conda 时，直接创建环境。没有 conda 时，可按 [Miniforge 官方安装说明](https://github.com/conda-forge/miniforge) 安装 Linux x86_64 版本：

```bash
mkdir -p "$HOME/downloads"
cd "$HOME/downloads"
curl -fL -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
curl -fL -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh.sha256
sha256sum -c Miniforge3-Linux-x86_64.sh.sha256
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

在仓库目录执行：

```bash
conda create -y -n risk-dynamic python=3.10.20 pip
conda activate risk-dynamic
bash tools/install.sh
python tools/doctor.py --gpu 0
```

也可使用已有 Python 3.10：`python3.10 -m venv .venv`，再 `source .venv/bin/activate`、`bash tools/install.sh`。不要在 Python 3.12 中安装这套锁文件。

安装脚本显式使用官方 PyPI 和 PyTorch cu121 索引。原机器默认镜像曾缺少锁定的包版本，出现 `No matching distribution` 时先检查索引，不要随意升级/降低科学计算包。安装下载数 GB 的 CUDA wheel 可能需要较长时间。

`--no-deps` 是有意的：Safety-Gymnasium 1.0.0 和 Gymnasium-Robotics 1.2.2 的旧 metadata 要求 MuJoCo 2.3.3，但当前代码使用 MuJoCo/Warp 3.12.0。根目录锁文件给出完整依赖闭包，额外补上旧 snapshot 锁文件遗漏的 `importlib-resources` 和 `zipp`。`pip check` 对这两个包报告的 MuJoCo 版本冲突属于已知情况；不要让自动 resolver 把 MuJoCo 降回 2.3.3。其他缺包或冲突要逐一排查。

## 4. 安装验收

```bash
python tools/experiment.py verify
python -m unittest discover -s tests -v
```

上述 `doctor.py` 会校验包版本和原始源码 SHA-256，并实际构建 SimpleButton、Dynamic 两个 GPU 环境，每个环境执行 2 worlds × 10 steps，检查 observation/reward/cost 有限。成功写出 `runs/doctor.json`，其中 `passed` 应为 `true`，`gpu_tests` 应包含两个环境。`--versions-only` 只检查版本，不代表 GPU 已验收。

首次 Warp 编译会比后续运行慢。缓存位于仓库 `.cache/`，可以在运行前设 `export RISK_CACHE_DIR=/你的大容量磁盘/risk-cache`。不要复用另一台机器的编译缓存。

随后做一次真正的短训练与 checkpoint 评估：

```bash
python tools/experiment.py campaign \
  --experiments dynamic-dca_div25 dynamic-dca_div200 dynamic-tqpo_b12 \
  --seeds 71 --gpus 0 --phase smoke --output runs/dynamic-preflight
```

有多张空闲 GPU 可用 `--gpus 0 1 2`。只填你被分配且空闲的 GPU。启动器使用物理 GPU 编号设置 `CUDA_VISIBLE_DEVICES`，每个 worker 内部使用 `cuda:0`。不要直接套用原机器固定的 4 卡调度器。

DCA smoke 保留完整 20 批 warmup + 2 批主训练，B25 对应 550,000 steps，并非几秒钟的小测试；TQPO B12 smoke 是 2 批、24,000 steps。每组独立评估 16 episodes。三组输出目录均出现 `COMPLETE.json` 才通过。镜像模式也使用完全相同的训练命令。

## 5. 开始当前 Dynamic 正式实验

最新协议：horizon=1000、cost threshold=15、epsilon=.2；reward gamma、actor cost gamma、dual cost gamma 都是 1；评估使用原始未折扣回报与 `C > 15` 的违反概率。

| launcher ID | 方法 | 每批 episodes | 正式预算 | 评估 checkpoint |
| --- | --- | ---: | ---: | --- |
| `dynamic-dca_div25` | DCA reward divisor=25 | 25 | 10,000,000 | 5M、10M |
| `dynamic-dca_div200` | DCA reward divisor=200 | 25 | 10,000,000 | 5M、10M |
| `dynamic-tqpo_b12` | 修复 CDF guard 的 TQPO | 12 | 10,008,000 | 5.004M、10.008M |

原机器本次 followup 正在跑 seeds 0、1。新机器推荐补跑 **seeds 2、3**，避免重复；若目的是复现，可显式选择 0、1。新机器使用自己的 run 目录。

先打印计划（不会启动训练）：

```bash
python tools/experiment.py campaign \
  --experiments dynamic-dca_div25 dynamic-dca_div200 dynamic-tqpo_b12 \
  --seeds 2 3 --gpus 0 --output runs/dynamic-machine2 --dry-run
```

确认后执行同一命令去掉 `--dry-run`。有两张空闲 GPU 时：

```bash
python tools/experiment.py campaign \
  --experiments dynamic-dca_div25 dynamic-dca_div200 dynamic-tqpo_b12 \
  --seeds 2 3 --gpus 0 1 --output runs/dynamic-machine2
```

启动器将 6 个任务分配到指定 GPU，每卡同时一个任务。无需 W&B 登录。训练结束后自动评估两个 checkpoint，每个 2048 episodes，独立评估 seed=92000000；验收训练步数、checkpoint roundtrip、源码签名和评估 checkpoint SHA-256，再写 `COMPLETE.json`。

只跑一个任务：

```bash
python tools/experiment.py run --experiment dynamic-dca_div25 \
  --seed 2 --gpu 0 --output runs/dynamic-machine2/dynamic-dca_div25/seed2
```

DCA 400 批中包含 20 批 warmup，正式参数实际为 `--main-batches 380`。TQPO B12 是 834 批。不要为了凑整数 10M 改 batch size 或裁掉 episode。

若终端可能断开，可把上述 campaign 放到 tmux 中，或用：

```bash
mkdir -p runs/dynamic-machine2
nohup python -u tools/experiment.py campaign \
  --experiments dynamic-dca_div25 dynamic-dca_div200 dynamic-tqpo_b12 \
  --seeds 2 3 --gpus 0 1 --output runs/dynamic-machine2 \
  > runs/dynamic-machine2/campaign.log 2>&1 &
echo $! > runs/dynamic-machine2/campaign.pid
```

停止本次 campaign 用 `kill -TERM "$(cat runs/dynamic-machine2/campaign.pid)"`，等待当前完整 batch 保存；不要 kill 其他人的实验。重启同一条命令会读取相同目录中的 checkpoint，已完成训练或评估会跳过。不要用 smoke 目录继续正式预算，或在同一输出目录更改 seed、算法、参数和源码；源码身份检查会拒绝。

## 6. 对照组和 SimpleButton

```bash
python tools/experiment.py list
```

Dynamic B25 对照入口为 `dynamic-baseline-dca`（divisor=100）、`dynamic-baseline-qcpo`、`dynamic-baseline-tqpo`，均为 all-gamma=1、epsilon=.2、10M steps。它们与 B12 followup 的算法/预算差异在 registry 中明确记录，不能混成同一实验设置。

当前有效 SimpleButton 矩阵是 reward gamma=.99、cost/dual gamma=1、B12，epsilon=.1/.2，DCA/QCPO/修复后的 TQPO，共 6 个配置，每组原 seeds 为 0–3。例：

```bash
python tools/experiment.py run \
  --experiment simplebutton-main_r099_c1_p20-dca \
  --seed 4 --gpu 0 --output runs/simplebutton-machine2/p20-dca-seed4
```

全套配置见 `experiments/registry.json`。不要启动 `provenance/` 中保存的原机调度脚本，它们包含历史绝对路径、旧任务依赖及旧 GPU 调度；新的运行入口是 `tools/experiment.py`。

## 7. 监控、结果和跨机同步

每个 run 保留以下文件：

- `training.log`、`status.json`：训练日志、最新状态和环境步数。
- `metrics.jsonl`、`episodes.jsonl`：训练曲线和逐批 episode 数据。
- `config.json`、`launch.json`：完整科学配置、源码签名和 launcher 配置。
- `latest.pt`、`checkpoint_*.pt`、`final.pt`：恢复和评估 checkpoint。
- `training_receipt.json`：完整预算和 checkpoint 验收回执。
- `validation_*/result.json`、`samples.npz`：独立评估统计和原始样本。
- `COMPLETE.json`：训练与全部规定评估均通过。

```bash
tail -f runs/dynamic-machine2/dynamic-dca_div25/seed2/training.log
cat runs/dynamic-machine2/dynamic-dca_div25/seed2/COMPLETE.json
```

Git 只同步源码。结果/checkpoint 被 `.gitignore` 排除，不会随 push/pull 传输。完成后通过 rsync/共享磁盘迁移整个 run 目录，保留 config、源码签名、评估样本和 checkpoint；不要覆盖另一台机器的同名 run。汇总时按方法/参数/seed 对齐，保留 PyTorch/CUDA/GPU 版本和 `runs/doctor.json`。

在这份导出的 checkout 上同步代码：

```bash
git status --short
git pull --ff-only origin DCA
```

有本地修改时先保存到明确的提交再同步，不要自动 reset。不要在训练进行中修改冻结源码或 pull 新的科学实现；完成当前 run 后再更新并启用新的输出目录。若要调整实验实现，请新建有明确身份的源码 snapshot 和 registry 配置，并更新来源 manifest，不要把新源码伪装成原实验。

## 8. 常见故障

- `CUDA unavailable`：检查镜像是否有 GPU 透传、驱动是否可见、所选 GPU 是否存在；不要靠换算法参数解决。
- Python dataclass、旧 Gymnasium 导入异常：先检查是否误用了 Python 3.11/3.12，以及 NumPy 是否被升级到了 2.x。
- `libGL` / EGL / SDL 相关导入失败：补齐第 2 节系统库。无需启动显示服务。
- 下载找不到版本：查看是否用了平台私有旧镜像；安装脚本显式使用官方索引。网络不可达时修复网络或准备完整 wheel 缓存。
- `configuration/scientific source drift`：输出目录与当前源码/预算不一致；使用对应原源码续跑，或另建 run 目录开始新实验。不要删除配置强行加载。
- `Resource temporarily unavailable`：相同输出目录已被 worker/launcher 锁定；检查是否重复启动了相同任务。
- OOM：减少同一 GPU 上的并发任务，或换更大显存 GPU。不要静默改变 B12/B25、horizon 或 critic 参数。
- `pip check` 的 MuJoCo 两项旧 metadata 冲突：见第 3B 节；GPU doctor 和短训练验收比旧包 metadata 更直接反映本实验能否运行。

## 9. 直接交给另一台机器 Codex 的任务

> 请按本仓库 STARTUP.md 配置和运行当前 Dynamic 实验。机器已有 NVIDIA 驱动；先检查 Python、PyTorch 和 CUDA。若镜像为 Python 3.10、PyTorch 2.3.1/CUDA 12.1，使用 `bash tools/install.sh --image-torch` 保留镜像的 PyTorch，然后运行带 `--image-torch` 的 GPU doctor。否则新建隔离的 Python 3.10 环境按锁文件安装，不覆盖原有 PyTorch 2.6 环境。校验源码、运行 launcher 测试及三组 smoke，全部通过后启动 dynamic-dca_div25、dynamic-dca_div200、dynamic-tqpo_b12 的 seeds 2、3。根据本机空闲 GPU 数设置 --gpus，每卡一个任务，输出到 runs/dynamic-machine2。保持 registry 中的科学参数不变，保存日志与 doctor 报告；正式完成以每个 run 的 COMPLETE.json 和两个 2048-episode 评估结果为准。不要自动启动旧 provenance 调度脚本，不要重复原机 seeds 0、1，不要把 checkpoint 提交到 Git。
