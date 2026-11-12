# Risk-aware RL — 当前 SimpleButton / Dynamic 实验快照

从 [STARTUP.md](STARTUP.md) 开始：新机器或 PyTorch 2.3.1/CUDA 12.1 镜像安装、GPU 检查、训练、独立评估和多机同步。

```bash
python tools/experiment.py list
python tools/experiment.py verify
```

当前 Dynamic 默认三组：`dynamic-dca_div25`、`dynamic-dca_div200`、`dynamic-tqpo_b12`。另外保留 B25 的 DCA /100、QCPO、TQPO 对照，以及 SimpleButton 最新有效矩阵（reward gamma=.99，cost gamma=1，epsilon=.1/.2，B12）。

`experiments/registry.json` 是可运行配置清单；`experiments/**/source/` 是原实验源码快照，1248 个文件逐一保留原内容。每个目录可能带有继承的其他算法入口，请通过统一 launcher 使用登记的方法。`provenance/` 保存原始协议和来源，里面的旧调度脚本仅供追溯；其中绝对路径不作为新机器的启动入口。`Quantile-based-Policy-Optimization/` 保留本地 DCA 工作区的基础项目代码。

远程分支 `DCA` 是从当前工作区导出的新源码快照分支；不包含约 450 GB 的原 Git 对象库、其他分支、训练 checkpoint、运行结果、环境缓存或登录凭据。原实验机器的分支、暂存区和运行任务没有改动。新机器从头训练额外 seed；若要从旧训练进度继续，需另行迁移完整 run 目录。

第三方代码的许可和来源见各 snapshot 的 `third_party/` 和 `official_reference/`。不要运行继承的旧 `smoke.py` 来判断当前 Dynamic 协议；统一检查入口见 STARTUP。
