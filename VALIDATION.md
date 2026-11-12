# 本次导出验证记录

截至本次导出，已经完成：

- 1248 个实验 snapshot 文件与原始源码 SHA-256 一致；打包前再次核对，原实验源码没有变化。
- 全部导出 Python 文件可解析；12 个登记的训练入口可导入并支持生成的 CLI 参数。
- 5 项 launcher 测试通过，包括预算、DCA warmup、dry-run 和输出目录保护。
- 在原实验环境中，通过新的 doctor 实际运行 SimpleButton 和 Dynamic 的 GPU 物理，各为 2 worlds × 10 steps，观测/奖励/代价均有限。
- 新建 Python 3.10.20 venv，从官方索引安装历史冻结依赖成功；其中 PyTorch 2.3.1+cu121 的 CUDA 可用性检查通过。根锁文件另外补齐 etils 的 importlib-resources、zipp 依赖。
- 推送内容检查不含训练 checkpoint、日志、缓存、本机登录凭据或其他 Git 分支。

用户要求停止后续测试并直接推送，因此没有在新环境中完成全部依赖与 GPU doctor 的联合验收，也没有重跑完整短训练、完整 10M 实验或目标机器上的镜像。`--image-torch` 路径按保留镜像 torch/triton/nvidia runtime 的方式实现；目标机器仍需按 STARTUP.md 完成 doctor 和 smoke，不能把本记录视作所有新机器均已验证。

原始科学源码保持不变。迁移新增的修改集中在统一 launcher、安装/检查脚本、根依赖锁与说明文档。源代码中继承的历史注释、旧测试和旧本地锁文件只用于追溯，当前有效协议以 registry 和对应方法的入口为准。
