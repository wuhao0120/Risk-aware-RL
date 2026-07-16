# DQCACBeta · Safety-Gym 调试实验账本

日期：2026-07-16（UTC）  
固定环境：`DynamicButton`  
固定种子：`seed=0`  
目标：先让 DQCACBeta 在无活跃约束惩罚时学会任务奖励，再逐层恢复风险约束；不以“跑满预算”代替机制排查。

## 1. 版本与数据基线

- 排查前源码快照：`40929db`（`chore(safety-gym): checkpoint pre-debug baseline`）。
- 持久化后台启动器：`caf5021`（`chore(safety-gym): add persistent experiment launcher`）。
- 未执行 `git push`；仓库其他目录的既有脏改动未纳入上述提交。
- 完整 W&B 基线下载：`_runs/wandb_export/baseline_dynamicbutton_2026-07-16/`。
- 500 万步 profile：`_runs/profiles/baseline_dynamicbutton_2026-07-16/`。
- 150 万步严格对齐 profile：`_runs/profiles/baseline_dynamicbutton_at_1500k_2026-07-16/`。

完整 500 万步训练后统一评估如下：

| 算法 | eval mean reward | reward std | outage `P(C>=15)` | λ final |
|---|---:|---:|---:|---:|
| DQCACBeta（旧基线） | -0.0811 | 0.4073 | 0.0615 | 0.0000 |
| QCPO_refs | 1.6696 | 0.4942 | 0.1538 | 6.9500 |

150 万步共同预算内，最后 20% 窗口为：

| 算法 | reward late mean | reward slope / 1M steps | reward quantile | outage late mean |
|---|---:|---:|---:|---:|
| DQCACBeta（旧基线） | -0.0616 | -0.0327 | -8.827 | 0.1167 |
| QCPO_refs | 1.4330 | +0.4405 | -13.810 | 0.1967 |

关键判断：旧 DQCACBeta 最终 `λ=0` 且策略奖励仍不增长，所以第一故障域是 reward actor-critic 链路。PID dual、cost CDF 精度或 β 风险权重可能是后续问题，但不能解释这个环境上的首要失败。

## 2. 实验纪律

1. 一次只运行一个训练 job，避免 CPU MuJoCo worker 与 GPU critic 更新互相争抢。
2. 所有训练通过 `launch_background.sh` 启动；使用 `nohup + setsid`，SSH 断开不影响训练。
3. 每个 job 在 `_runs/jobs/<job>/` 保存 PID、完整命令、Git commit、dirty 状态、开始/结束时间和退出码。
4. 每次训练结束后，用 W&B API 的 `scan_history()` 下载全部指标，不以控制台抽样点代替 history。
5. 用 `profile_wandb_metrics.py` 在共同 env-step 预算下统计 early/middle/late、趋势斜率、非有限值并画对齐图。
6. 修改在早期窗口没有形成方向一致的提升时提前停止；只对通过短预算门的版本扩大预算/seed。
7. 一次性数据、PID、日志与图放入 Git 已忽略的 _runs；TMPDIR、W&B cache 和 Matplotlib cache 固定到 /vepfs-mlp2/c20250510/251204033/.tmp/safety_gym_env，不占 20G 根分区。
8. 阶段结束清理失败的空 job、重复导出和不再使用的一次性探针；保留支撑结论的原始 W&B 导出/profile。

## 3. 当前实验

### E1：只扩大固定探索尺度

- 状态：已完成（exit code 0）。
- job：`DQCAC_DynamicButton_dbg_e1_std1_fixed_1500k_s0`。
- W&B run id：`hqf29nyd`。
- PID：`23149`（已退出；训练期间为独立 session）。
- 代码 commit：`caf5021`；算法代码与旧基线一致。
- 唯一算法变量：`init_std: 0.5 -> 1.0`；`learn_std=False` 保持不变。
- 固定项：`B=10`、`T=1000`、`warmup_iters=30`、`updates_per_episode=10`、`seed=0`。
- 预算：150 iterations = 150 万 env steps，为旧全量的 30%。
- 实际训练耗时：644.5 秒（10.7 分钟）；训练后完成 130 条统一评估轨迹。
- 75 万步检查点通过：同预算 reward late mean 和 slope 均优于旧 DQCAC；因此完成 150 万步短预算。
- 判定重点：探索变化是否显著增加 reward advantage / actor weight 的有效尺度，并让 reward 窗口均值与趋势同时上升。

命令：

```bash
./launch_background.sh DQCAC_DynamicButton_dbg_e1_std1_fixed_1500k_s0 -- \
  /vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python -u run_experiment.py \
  --algo DQCAC --env DynamicButton --seed 0 --device 0 --num_eval 128 \
  --wandb_mode online --tag dbg_e1_std1_fixed_1500k \
  --set num_iterations=150 num_envs=10 log_interval=5 init_std=1.0 \
  wandb_name=DQCAC_DynamicButton_dbg_e1_std1_fixed_1500k_s0 \
  wandb_group=dqcac_debug_dynamicbutton wandb_tags=debug,phase1,reward-gate,std1
```

### E1 最终结论

- 训练耗时：`644.5s`（10.7 分钟），150 iterations 实测约 4.30 秒/iteration；退出码 0。
- W&B 完整导出：`_runs/wandb_export/final_e1_dynamicbutton_2026-07-16/`。
- 同 150 万步 profile：`_runs/profiles/final_e1_dynamicbutton_2026-07-16/`。
- 统一评估：mean reward `-0.0144`，outage `0.0846`，critic CDF `0.0558`，λ final `0.9205`。
- 后 20% 训练窗口：reward mean `-0.0315`、slope `+0.0266/百万步`、λ mean `2.215`、actor weight std `0.224`。

结论：固定 `std=1.0` 在 73 万步 checkpoint 的确把 reward late mean 从旧基线 `0.0238` 提到 `0.1192`，说明探索不足是一个因素；但 90 万步后 critic-dual 把 λ 推高，reward 又回落。E1 只获得“探索有帮助”的证据，没有通过“正常训练”门，也说明 reward 主干实验必须暂时固定 `λ=0`。

### Smoke 验证（候选 GAE/PPO 实现）

2026-07-16 分别对 `distributional`、`gae`、`gae_ppo` 三条路径执行 `B=2,T=32,iters=2` 后台 smoke。三者训练、评估、JSON 保存和退出码均正常；训练分别约 8 秒，无 NaN/Traceback。scalar reward value 在 episodic 模式下与 distributional critic 一样接收 `t/T` step feature；GAE target 每个 rollout 只计算一次并在 value/PPO epochs 间冻结。

### 关键修正：actor 多次更新的策略新鲜度与 importance ratio

- 排查发现旧 DQCACBeta 的 `updates_per_episode=10` 同时表示 critic 和 actor 都在同一 rollout 上更新 10 次；第 2~10 次 actor step 已不再来自当前策略，但旧 `distributional` loss 没有 importance ratio，因此属于未修正的 off-policy rollout reuse。这是 reward 均值/quantile 长期大幅波动的高优先级原因之一。
- 现拆分为 `updates_per_episode`（critic/value epochs，默认 10）与 `actor_updates_per_episode`（actor epochs，非 PPO 默认 1）。`distributional` 和 `gae` 若设置 actor epochs > 1 会直接报错，防止静默重现旧问题。
- `gae_ppo` 在采样时只保存一次行为策略 `old_log_probs=log π_behavior(a|s)`；每次 actor 更新重新前向计算当前 `log π_θ(a|s)`，ratio 始终为 `exp(logπ_current-logπ_behavior)`，分母在整个 rollout 的所有 PPO epochs 中保持固定。不能在每次更新后覆盖 `old_log_probs`，否则 ratio 会被人为重置到 1，失去对 rollout-policy drift 的检测与裁剪。
- reward surrogate 使用 PPO clipped minimum；需要最小化的 cost-risk surrogate 使用 conservative maximum。同步记录 ratio mean/std、clip fraction 和 approximate KL。
- 额外发现 warmup 中旧代码在 optimizer 没有 step 时仍推进 actor/dual LR scheduler，并触发 PyTorch `scheduler.step() before optimizer.step()` 警告。现仅在对应 optimizer 真正更新后推进 scheduler，使学习率时间轴与参数更新时间一致。
- 持久化验证：`smoke_ppo_multiupdate` 令 critic/actor epochs 均为 2，训练 7.9 秒、exit code 0，证明固定行为 log-prob 的多 epoch PPO 路径可执行；`smoke_gae_warmup_scheduler` 令 warmup=1、critic epochs=2、actor epochs=1，训练 8.2 秒、exit code 0，且无 scheduler-before-optimizer 警告，证明单次 on-policy GAE 与调度顺序均生效。

### E2 首次启动中止记录（不计入算法结果）

- job：`DQCAC_DynamicButton_dbg_e2_gae_rewardonly_800k_s0`；W&B run id：`jud8j38v`。
- 代码：`58ace0b`；日志只到 iteration 4 / 约 5 万 env steps，仍处于 30-iteration critic warmup，actor 尚未执行，因此不能用于判断 GAE 效果。
- 人工中止原因：启动后确认该版本仍把同一 rollout 用于 10 次无 ratio 的 GAE actor 更新。为避免在已知不正确的策略新鲜度设置上浪费 80 万步，终止整个后台进程组；这属于开发过程的 aborted run，不与 E1/基线比较。
- 修正后 E2 将使用 critic/value epochs=10、actor epochs=1；E3 才在固定 old log-prob 的 PPO clip 下使用多个 actor epochs。

### E2：纯奖励门——scalar value + GAE（不启用 PPO）

- 状态：已完成（exit code 0）；通过持久化后台运行。
- job：`DQCAC_DynamicButton_dbg_e2b_gae_onpolicy_rewardonly_800k_s0`；PID：`43188`。
- W&B run id：`tayn9ex2`；源码 commit：`b91405b`（未 push）。
- 设置：`reward_actor_mode=gae`、`lambda_max=0`、`init_std=1.0`、`B=10`、`T=1000`、`warmup_iters=30`、critic/value epochs=10、actor epochs=1。
- 实际训练耗时 `343.4s`；80 万步完成后评估 130 条轨迹，总 job 约 6 分钟。
- 60 万步 W&B 门槛：late reward mean `-0.0087`、slope `+0.0616/百万步`，均低于同预算 E1 的 `0.0451`、`+0.1728`。虽在约 73 万步尝试优雅中止，但多进程 rollout 没有及时响应 SIGINT，已接近预算终点，故保留正常完成与终点评估。
- 80 万步最终 profile：late reward mean `0.00635`、slope `+0.0719/百万步`；E1 分别为 `0.1125`、`+0.2397`，QCPO_refs 为 `1.451`、`+1.606`。
- GAE 确实把 reward advantage std 从 E1 的 `0.0184` 放大到 `0.0543`，actor grad norm late mean 为 `0.0113`；value explained variance 中段 `0.174`、后段 `0.060`。信号尺度变大但单次 actor step 没转化为有效策略改进。
- 统一评估：mean reward `-0.00177`、reward std `0.1271`、outage `0`。本配置未通过 reward 门，不扩展预算或 seed。
- 最终导出：`_runs/wandb_export/final_e2b_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e2b_dynamicbutton_2026-07-16/`。

### E3：纯奖励门——GAE + PPO clip 多 actor epochs

- 状态：已完成（exit code 0）；只在 E2 上增加固定行为策略 ratio、PPO clip 与多 actor epochs。
- job：`DQCAC_DynamicButton_dbg_e3_gaeppo8_rewardonly_600k_s0`；W&B run id：`d2rfqck2`；启动 commit：`1970fa5`。
- 设置：`reward_actor_mode=gae_ppo`、actor epochs=8（对齐 QCPO_refs）、critic/value epochs=10、`ppo_ratio_clip=0.1`；`lambda_max=0`、std、warmup、B/T 与 E2 保持一致。
- 实际训练耗时 `272.3s`；评估 70 条轨迹 mean reward `0.2179`、outage `0.0714`。
- 60 万步严格对齐：late reward mean `0.1291`、slope `+0.3058/百万步`，超过 E1 的 `0.0451/+0.1728` 和旧 DQC 的 `0.0943/+0.2828`，但仍远低于 QCPO_refs 的 `1.574/+2.711`。
- PPO late health：ratio std `0.0449`、clip fraction `0.0425`、approx KL `0.00106`、actor grad norm `0.00992`；无 NaN/Inf。reward value explained variance late mean `0.257`、终点 `0.375`。
- 结论：固定 old log-prob 的 importance ratio + PPO clip 是关键改进，能够把更大的 GAE 信号转化为 reward 增长；但 QCPO_refs 的剩余优势还来自无 warmup 的样本效率、obs normalization、可学习 std/策略结构等候选因素。
- 最终导出：`_runs/wandb_export/final_e3_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e3_dynamicbutton_2026-07-16/`。

### E4：相同 PPO 主干，移除 reward-only warmup

- 状态：已完成（exit code 0）；不改 loss、网络或学习率，只令 `warmup_iters=0`。
- job：`DQCAC_DynamicButton_dbg_e4_gaeppo8_nowarmup_300k_s0`；W&B run id：`6y5i04cr`；启动 commit：`81af9de`。
- 设置保持 E3：GAE+PPO、actor epochs=8、critic/value epochs=10、`lambda_max=0`、固定 std=1.0。
- 实际训练耗时 `140.3s`；评估 70 条轨迹 mean reward `0.2577`、outage `0.10`。
- 30 万步 profile：late reward mean `0.2146`、slope `+0.9341/百万步`；同预算 QCPO_refs 为 `1.355/+5.431`，有 warmup 的 DQC 变体仍约 `-0.051`。
- actor-rollout 对齐：E4 的前 30 个 actor rollouts late mean `0.2146`，高于 E3 第 31~60 个 rollout 的 `0.1550`；去掉 warmup 不只节省 env steps，也略改善相同更新次数下的结果。
- 终点 PPO：ratio std `0.0738`、clip fraction `0.1653`、approx KL `0.00271`；仍有限且无 NaN/Inf。
- 结论：旧 30-iteration warmup 在 reward 主干上是明确的样本浪费；恢复约束时应只延迟 dual/risk 分支，不应冻结 reward PPO actor。
- 最终导出：`_runs/wandb_export/final_e4_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e4_dynamicbutton_2026-07-16/`。

### E5：在 E4 上只增加逐维 observation normalization

- 状态：已完成（exit code 0）；保持 MLP、固定 std=1.0、GAE/PPO、epochs、学习率和 `lambda_max=0` 不变。
- job：`DQCAC_DynamicButton_dbg_e5_obsnorm_gaeppo8_300k_s0`；W&B run id：`xjc6hrke`；启动 commit：`7dc2bb1`。
- 选择顺序依据：QCPO_refs 与 E4 都从 `σ=1` 开始，30 万步 entropy 仅为 `2.809` vs `2.838`，可学习 std 变化很小；而 QCPO_refs 明确使用 running mean/variance、clip 到 [-10,10]。
- 实现要求：actor、reward value、reward/cost critics 共用同一份逐维统计；rollout 期间统计冻结，rollout 后更新；首次仅刷统计、不更新 actor，避免新旧归一化导致虚假 PPO ratio。
- 实现验证：Chan 合并公式与 QCPO_refs `RunningMeanStdModel` 数值逐项对拍，`mean/var/output` 最大绝对误差均为 `0`；持久化后台端到端烟测正常退出，覆盖首次统计 warmup、GAE、PPO 两轮 actor update 和训练后评估。
- 实际训练耗时 `147.8s`；评估 70 条轨迹 mean reward `1.690`、outage `0.514`。
- 30 万步 profile：late reward mean `1.662`、slope `+6.816/百万步`，显著超过 E4 的 `0.2146/+0.9341`，也超过 QCPO_refs 同预算的 `1.355/+5.431`；reward 学习门通过，不扩跑 reward-only。
- PPO/拟合健康：终点 ratio std `0.0490`、clip fraction `0.0603`、approx KL `0.00121`；reward value explained variance `0.7368`，无 NaN/Inf。
- 约束诊断：训练后段 empirical outage `0.667`，统一评估 truth `0.514`；cost critic CDF `0.413`，低估 `0.101`。观测归一化解决了 reward 停滞，但 reward-only 策略明确违反 `ω=0.2`，下一阶段必须恢复 dual/risk 分支。
- 结论：旧 DQCACBeta 不涨的核心工程原因是高维异尺度 observation 未归一化；配合前面已经验证的 GAE、固定行为策略 old log-prob、importance ratio 和 PPO clip 后，DQCAC reward 主干不弱于 QCPO_refs。
- 最终导出：`_runs/wandb_export/final_e5_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e5_dynamicbutton_2026-07-16/`。

### E6：恢复经验 PID dual 与 sum normalization

- 状态：已完成（exit code 0）；只恢复约束，不改 E5 的 reward 主干、网络、观测归一化或 std。
- job：`DQCAC_DynamicButton_dbg_e6_empiricalpid_sum_norm_300k_s0`；W&B run id：`43ftc79n`；启动 commit：`e6b443c`。
- 必修一致性：同一 rollout 的 cost advantage 必须在 PPO epochs 之前冻结；不能一边更新 cost critic，一边让 8 个 actor epoch 使用不断移动的 risk advantage。
- dual 输入优先使用 rollout/最近窗口的经验 outage 或 `(1-ω)` cost quantile；当前 cost critic CDF 仍有 `-0.101` 偏差，只作为校准指标，不直接全权驱动 λ。
- 采用 QCPO_refs 风格积分 PID 与 `(J_r+λJ_c)/(1+λ)`；分别记录经验 gap、cost quantile gap、积分状态、effective reward/risk coefficient 与 CDF calibration error。
- 实现保持旧 `critic_adam`/无 sum-norm 为默认；实验开关使用最近 100 条轨迹 outage。risk advantage 在首个 actor epoch 计算后缓存，后续 7 个 epoch 不再随 critic 移动。
- 验证：outage/quantile 两种 PID 确定性单测均与手算完全一致；持久化后台烟测 exit 0，强制正误差时 λ 按 `0→0.08→0.16` 增长，并覆盖 observation warmup、两轮 PPO 和 sum normalization；另一次旧 `critic_adam + distributional` 默认路径回归烟测也 exit 0。
- 实际训练耗时 `142.7s`；评估 70 条轨迹 mean reward `1.640`、outage `0.600`，λ 终点 `0.374`。
- 30 万步 profile：late reward `1.499`、slope `+6.450/百万步`、empirical outage `0.550`；reward 仍高于 QCPO_refs 同预算的 `1.355`，但约束没有向 `0.2` 收敛。
- PID 确实生效：late window outage `0.528`、λ `0.287`、effective risk coefficient `0.221`；终点分别为 `0.570/0.374/0.272`。失败不是 λ 不更新，而是 risk gradient 控制力不足。
- PPO/拟合健康：late clip fraction `0.0274`、approx KL `0.00072`、reward value explained variance `0.662`，无 NaN/Inf。
- 约束根因：late raw risk advantage std 仅 `0.0137`；cost critic CDF 评估为 `0.307` 对真实 `0.600`，偏差扩大到 `-0.293`。此外 `β=0.95` 使 90% 累计 risk 权重落在前 45 步，与 T=1000 未折扣整段 cost 不匹配。
- 最终导出：`_runs/wandb_export/final_e6_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e6_dynamicbutton_2026-07-16/`。

### E7：只提高 Abel risk discount 到 β=0.99

- 状态：已完成（exit code 0）；完全复用 E6，只把 `beta=0.95` 改为 `0.99`。
- job：`DQCAC_DynamicButton_dbg_e7_beta099_empiricalpid_300k_s0`；W&B run id：`6xm1k6n3`；启动 commit：`bb25576`。
- 算法依据：β 是无限期 Abel 可和性带来的偏差—方差旋钮，β→1 恢复精确约束梯度；本实验是有限 T=1000，`0.99` 将 90% 累计权重覆盖从前 45 步扩到前 229 步，同时比直接 `β=1` 更保守。
- 实际训练耗时 `147.4s`；评估 reward `1.025`、outage `0.400`、λ `0.375`，cost critic CDF `0.261`，偏差 `-0.139`。
- 30 万步 profile：late reward `1.182`、slope `+5.005/百万步`、outage `0.433`；相比 E6 的 `1.499/+6.450/0.550`，约束明显改善但付出 reward 代价。QCPO_refs 同预算为 `1.355/+5.431/0.517`。
- 数值健康：late PPO clip `0.0582`、reward value explained variance `0.721`，无 NaN/Inf；终点训练批 outage 已到 `0.2`，但独立 70 轨迹评估仍为 `0.4`。
- 结论：β 是有效且此前过度短视的风险—收益旋钮；`0.99` 已优于 `0.95` 的约束控制，但短预算尚未可行。
- 最终导出：`_runs/wandb_export/final_e7_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e7_dynamicbutton_2026-07-16/`。

### E8：有限时域 β=1.0 诊断

- 状态：已完成（exit code 0）；只把 E7 的 `beta=0.99` 改为 `1.0`。
- job：`DQCAC_DynamicButton_dbg_e8_beta100_empiricalpid_300k_s0`；W&B run id：`o23rzgg8`；启动 commit：`0ad877e`。
- 目的：在有限 T=1000 下去掉 Abel 偏差，测试现有 risk advantage 的控制上限；这不是建议无限时域使用 β=1。
- 实际训练耗时 `142.0s`；评估 reward `0.1355`、outage `0.000`、λ `0.113`。
- 30 万步 profile：late reward `0.238`、slope `+1.351/百万步`、outage `0.0167`、λ `0.1375`；与 E7 的 `1.182/+5.005/0.433` 相比，已经跨过风险—收益最优区间并过度保守。
- 控制动态：训练 reward 在 190k 达 `1.185`，随后 outage 先降而 100 条窗口仍使 λ 上升，reward 到 290k 回落至 `0.174`；这是 PID 窗口滞后与 β=1 强风险梯度共同造成的过度校正。
- 数值健康：late PPO clip `0.122`、approx KL `0.00227`，仍无 NaN/Inf；critic CDF `0.062` 对 truth `0`，此时方向转为轻微高估。
- 结论：risk critic/advantage 有能力控制约束，问题不是风险梯度无效；`β=1` 不作为最终配置，工作点位于 `0.99~1.0`。
- 最终导出：`_runs/wandb_export/final_e8_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e8_dynamicbutton_2026-07-16/`。

### E9：β=0.995 中间工作点

- seed 0 状态：已完成（exit code 0）；只把 E7 的 `beta=0.99` 改为 `0.995`，其余保持不变。job 为 `DQCAC_DynamicButton_dbg_e9_beta0995_empiricalpid_300k_s0`，W&B run id 为 `nyypklpy`，启动 commit 为 `f282b48`。
- 覆盖解释：`β=0.995` 的 90% 累计 risk 权重约覆盖前 459 步，介于 E7 的 229 步和 E8 的整段等权之间。
- 实际训练耗时 `143.8s`；独立评估 70 条轨迹得到 reward `0.6576`、outage `0.200`、cost-return 0.8 quantile `14.2`、λ `0.357`，在 seed 0 上恰好达到 `ω=0.2`。
- cost critic 查询 CDF 为 `0.211`，相对经验 truth `0.200` 只偏高 `0.011`；reward value explained variance 为 `0.841`。因此这个工作点的主要不确定性已经不是“查询处 critic 严重失准”，而是短预算与单 seed 的策略/约束方差。
- 终点 PPO ratio std `0.0638`、clip fraction `0.1108`、approx KL `0.00202`，无 NaN/Inf；importance ratio 与 clip 工作在合理量级。
- 30 万步训练最后 20% 窗口 reward `0.9967`、slope `+4.432/百万步`、outage `0.400`、λ `0.294`。训练窗口与独立终评存在明显差异，不能只凭 seed 0 的 `0.200` 宣称稳定可行。
- 完整导出：`_runs/wandb_export/final_e9_dynamicbutton_2026-07-16/`；同 30 万步 profile：`_runs/profiles/final_e9_dynamicbutton_2026-07-16/`。
- seed 1/2 均通过持久化后台正常完成，训练耗时分别 `201.1s/193.4s`；W&B run id 为 `tuo3ehve/uq7crzsu`。并行共享 GPU 使单 run 比 seed 0 慢，但总墙钟约 4 分钟。
- 三 seed 终评（每 seed 70 条、共 210 条）：reward `0.889±0.229`、outage `0.262±0.054`、cost 0.8 quantile `17.2±2.69`、λ 均值 `0.317`；这里 `±` 是跨 seed 样本标准差。seed 0/1/2 outage 分别为 `0.200/0.286/0.300`，E9 尚不能判为多 seed 可行。
- critic CDF 三 seed 为 `0.211/0.244/0.218`，平均 `0.224`；相对各自经验 outage 的偏差为 `+0.011/-0.042/-0.082`，平均低估 `0.0375`。critic 查询精度已进入约 0.04 量级但仍有 seed 相关偏差，暂不足以取代经验 dual。
- 三 seed 的训练后 20% reward 为 `0.997/1.268/1.174`，outage 为 `0.400/0.400/0.417`；同 30 万步 QCPO_refs seed 0 为 `1.355/0.517`。DQCAC 已不再“reward 不涨”，且更早施加风险控制，但当前控制仍有滞后。
- 多 seed 导出：`_runs/wandb_export/final_e9_multiseed_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e9_multiseed_dynamicbutton_2026-07-16/`。
- 决策：不把 E9 直接扩至 500 万步，也不继续盲目细分 β。先验证 QCPO_refs 原式的 cost-quantile PID；由于 E9 查询处 CDF 平均偏差已约 0.04，局部 quantile 加密/IQN 的优先级低于 dual 响应与多 seed 验证。

### E10：QCPO_refs 风格 cost-quantile PID

- 状态：已完成（exit code 0）；job 为 `DQCAC_DynamicButton_dbg_e10_quantilepid_beta0995_300k_s0`，W&B run id 为 `kivy3xs4`。
- 只把 E9 的 `dual_pid_signal=outage` 改为 `cost_quantile`，并令 `pid_cost_scale=10`；其余 β、Ki、窗口、reward/PPO/obs-norm 和预算不变。
- 更新式为 `I←clip(I+0.1·(Q_0.8(C)-15)/10)`，逐式对应 QCPO_refs 的最近 100 条轨迹 quantile PID。它不仅判断是否超限，还利用超限严重度。
- 训练耗时 `142.4s`；独立评估 reward `-0.1749`、outage `0`、mean cost `0.057`、λ 终点 `0.880`。它满足约束但完全丢失任务性能，未通过短预算门。
- 训练后 20% reward `-0.0937`、slope `-0.5105/百万步`、outage `0`、λ `1.088`；数值无 NaN/Inf，失败来自控制过强而非数值崩溃。
- 控制动态给出直接证据：190k 时 reward/outage/λ 为 `0.871/0.20/0.34`；250k 时变为 `0.364/0/1.16`；290k 时为 `-0.379/0/1.01`。约束已经安全后，积分状态仍长期维持高惩罚，形成 windup/滞后。
- critic CDF `0.0549` 对 truth `0`，此时略高估，但它不是 dual 输入；因此不能把 E10 的过保守归咎于 critic CDF。
- 完整导出：`_runs/wandb_export/final_e10_dynamicbutton_2026-07-16/`；profile：`_runs/profiles/final_e10_dynamicbutton_2026-07-16/`。
- 结论：不能把 QCPO_refs 的 quantile-PID 原参数机械移植到 DQCAC。DQC risk advantage 与 β=0.995 的控制增益更强；下一步若继续 PID，应优先使用更小 quantile `Ki`、leaky-I/anti-windup 或基于 outage 的 PI/PID，而不是为 E10 扩预算/seed。

## 4. 分阶段改进路线

### A. 先修 reward 学习骨架

按成本从低到高执行：

1. E1：固定 `std=1.0`，验证旧实现是否只是探索不足。
2. 若 E1 不通过，加入 scalar reward value critic + GAE；先让 actor 使用低方差 GAE reward advantage，distributional cost critic 暂不承担 reward baseline。
3. 在同一 on-policy rollout 上加入 PPO ratio clip、old log-prob、minibatch/epoch；记录 ratio、clip fraction、approx KL、value explained variance 与 gradient norm。
4. PPO/GAE 通过短预算门后，再考虑 observation normalization、可学习 log-std 和 entropy 调度；这些分别做消融，不一次全部打开。

选择理由：QCPO_refs 的 reward 主干是 value + GAE + PPO，而旧 DQCACBeta 用两个同策略动作样本之差构造 `Q(s,a)-E_a Q(s,a)`。旧日志中 reward advantage raw std 只有约 `0.005~0.012`，容易让 actor 只接收到 critic/动作采样噪声。

### B. reward 正常后修 dual

1. 用完成轨迹的经验 outage 或滑动窗口/EMA 驱动 dual，替代校准不足时的初态 cost-critic CDF。
2. 移植 QCPO_refs 的 PID（至少先做积分项；再判断是否需要 P/D），记录 raw gap、EMA gap、P/I/D 分量与 λ 更新幅度。
3. 策略目标采用 `(J_r - λ J_c)/(1+λ)` 或等价 sum normalization 作为独立消融，避免 λ 大时完全吞没 reward 梯度。
4. 只有经验 outage 与 critic CDF 已校准到可接受误差后，才比较 critic-dual 是否能带来更低方差/更快响应。

### C. 提高查询阈值附近的 distributional critic 精度

候选顺序：

1. **平滑 CDF 查询**：把硬指示 `1{z_i>=b}` 改为带温度的 sigmoid；温度由查询点附近 quantile spacing 或固定小网格选择，同时保留硬 CDF 用作无偏诊断。
2. **均匀骨架 + 局部加密 quantiles**：目标阈值对应 cost CDF 约 `F_C(d)=1-alpha=0.8`，可在 `tau≈0.8` 附近增加点，但保留全域均匀点以维持 Bellman target 覆盖。
3. 非均匀 τ 不能继续简单平均 indicator。CDF 查询必须使用与 τ 网格对应的 quadrature 权重，或训练时采用 uniform/local mixture 并做重要性修正；否则“点更多”会直接改变所估计的分布。
4. budget `b_t` 随状态和历史 cost 改变，局部 CDF 水平不总是 0.8；因此需要记录查询处 crossing τ 的分布，判断固定局部加密是否真的覆盖常见查询区域。
5. **IQN**：若固定网格仍有明显 CDF 校准误差，再用 IQN 对 uniform τ 采样，并在查询时增加 τ 样本、排序后反演/平滑 CDF。IQN估计的是 quantile function，不是天然直接输出 CDF；计算成本与反演误差必须和加密 QR 基线公平比较。

### D. β 与长时域

旧 `beta=0.95` 在 T=1000 时把绝大多数风险梯度质量放在前段。reward、dual、CDF 三条链路稳定后，再比较 `beta∈{0.95,0.99,1.0}`；提前做会把多个故障源混在一起。

## 5. 短预算通过标准

候选版本至少同时满足：

- 在相同步数下，reward 后 20% 窗口均值明显高于旧 DQCACBeta，且全程 slope 为正；
- 改善不是单个迭代尖峰，rolling mean 能保持；
- actor 的 approx KL/clip fraction/梯度范数有限，无 NaN/Inf；
- 约束尚未开启或 λ≈0 时，先不要求 outage 精确贴边，但不能把任务奖励提升错误归因于 PID/CDF；
- 恢复约束后，经验 outage、critic CDF、λ 三者方向一致，CDF calibration error 不持续扩大。

只有通过上述门槛的配置才扩展到 300~500 万步和多个 seed。

### E11：QCPO 主干校准（Q-A0/Q-A1/Q-A2）

- 状态：实现与持久化 smoke 已完成，短预算对比待运行。
- 2026-07-16 定位到两个确定性工程错误：
  1. 旧 `QCPOGPU` 默认把同一 rollout 连续用于 5 次 actor update，但没有 behavior `old_log_prob`、importance ratio 或 PPO clip；第 2～5 次是未修正的 off-policy reuse。
  2. 基类虽已提供 `ObservationNormalizer`，QCPO 训练循环从未调用 `obs_normalizer.update()`；打开开关也只会一直使用初始 mean=0/var=1，等价于没有归一化。
- 修正后的安全默认是 `qcpo_actor_update_mode=on_policy, updates_per_episode=1`；如果 on-policy 模式配置多次更新会直接抛错，避免静默重现旧 bug。
- 新增 `qcpo_actor_update_mode=ppo`：rollout 采样时只保存一次 `old_log_probs=log π_behavior(a|s)`，所有 actor epochs 的分母固定；每个 epoch 只重新计算当前策略分子，并用 `clip=0.1` 的 PPO surrogate。记录 ratio mean/std、clip fraction、approx KL、actor grad norm 与 log-std。
- observation RMS 在预热 rollout 更新；正式 rollout 中，moments 在采样和全部 actor epochs 内保持冻结，本批训练完成后才合并统计供下一次 rollout 使用。因此 on-policy log-prob 输入变换严格一致，PPO 的 ratio 也不会混入本批 RMS 突变。
- 新增可学习 `log_std` 的有限区间保护 `[-5,2]`；默认仍关闭以复现旧基线，Q-A1/Q-A2 显式使用 `init_std=1, learn_std=true`。
- 持久化 smoke：
  - `QCPO_DynamicButton_smoke_onpolicy_fix_s0`：`B=2,T=32,iters=2`，训练 `7.5s`，exit code 0。
  - `QCPO_DynamicButton_smoke_ppo_obsnorm_fix_s0`：在同预算启用 obs RMS、PPO 8 epochs、可学习 σ，训练 `6.9s`，exit code 0。
- 接下来的轻量门控固定 `DynamicButton/seed0/λ=0/B=10/T=1000`：
  - Q-A0：单次 on-policy、无 obs norm、固定 σ=0.5，先跑 100k。
  - Q-A1：单次 on-policy + obs norm + σ=1/learnable，先跑 100k。
  - Q-A2：Q-A1 + 固定-old PPO 8 epochs，先跑 100k。
  - 100k 仅用于排除明显不学习配置；胜者扩到独立 300k run，不从 100k checkpoint 续跑，保证比较协议一致。

#### E11 Q-A0/Q-A1/Q-A2 结果

- 三个 100k run 均为 `B=10,T=1000,seed=0,lambda_max=0`，通过持久化后台完成并 exit code 0：
  - Q-A0 raw observation、固定 sigma=0.5、单次 on-policy：W&B `57t0ijye`，训练 `55.7s`，终评 reward `-0.0204`。
  - Q-A1 obs RMS、sigma=1/learnable、单次 on-policy：W&B `x3fwd2ao`，训练 `56.1s`，终评 reward `0.0275`。
  - Q-A2 Q-A1 + fixed-old PPO 8 epochs：W&B `gq2mukc4`，训练 `57.9s`，终评 reward `0.0209`。
- 共同 100k 后 20%：Q-A0/Q-A1/Q-A2 reward 分别为 `-0.0419/-0.0151/0.0385`，趋势分别为 `-0.400/+0.029/+0.649` 每百万步。完整导出与 profile：
  - `_runs/wandb_export/qcpo_qa012_100k_2026-07-16/`
  - `_runs/profiles/qcpo_qa012_100k_2026-07-16/`
- 只有 Q-A2 形成方向一致的正趋势，因此独立扩到 300k：W&B `6xp94twc`，训练 `146.0s`，130 条终评 reward `0.3496`、outage `0.1692`；lambda 固定为 0，所以 outage 只描述自然策略，不能归因于约束控制。
- 共同 300k 后 20%：Q-A2 reward `0.2297`、趋势 `+1.248/百万步`，相对旧 QCPO 的 `-0.1139/-0.345` 已完成定性修复；但仍明显低于 DQCAC E5 的 `1.662/+6.816` 和 QCPO_refs 的 `1.355/+5.431`。对齐 profile：
  - `_runs/wandb_export/qcpo_a2_vs_key_300k_2026-07-16/`
  - `_runs/profiles/qcpo_a2_vs_key_300k_2026-07-16/`
- 结论：observation RMS、sigma=1/learnable 与正确 PPO reuse 让 QCPO 从“不学习”恢复为稳定上升，但轨迹级 MC reward credit assignment 仍是剩余主瓶颈。

#### Q-B：QCPO-GAE/PPO hybrid 实现

- 新增 `qcpo_reward_mode=mc|gae`。`mc` 保持原始轨迹级 QCPO；`gae` 只将 reward 分支改成 scalar `V_r(s,t)` + 冻结 GAE lambda-return，constraint 仍是轨迹级 `I{C>=d}`，因此明确标为 hybrid。
- `V_r` 与 actor 使用相同 MLP 宽度、共享 observation RMS，并默认追加 `t/T`；GAE advantage/value target 每个 rollout 只计算一次，在 8 个 value/PPO epochs 中保持冻结。
- PPO surrogate 将 reward 和 risk 分开：reward 用 clipped minimum，risk 用 conservative maximum；这修正了“先组合正负权重再 clip”在 lambda>0 时可能不保守的问题。
- 持久化 smoke `QCPO_DynamicButton_smoke_gaeppo_obsnorm_s0`：`B=2,T=32,iters=2`，训练 `7.1s`，exit code 0；覆盖 obs RMS、V、GAE、固定 old-prob PPO 和终评。
- 下一实验：Q-B reward-only 100k；若明显超过 Q-A2 的 `0.0385` 后段均值并保持 PPO/value 数值健康，再扩独立 300k。

### E12：统一 MLP+LSTM 骨干（实现阶段）

- 已在 `utils/model.py` 新增可复用 `RecurrentActorValue`，严格采用 QCPO_refs 输入协议：
  - MLP 输入 `[raw_observation, previous_cost]`；
  - LSTM 输入 `[MLP feature, previous_action, previous_reward]`；
  - 默认 `[512,512] tanh + LSTM512`，并使用 `MLP feature + LSTM output` 残差；
  - 输出 tanh Gaussian mean、可学习 log-std、scalar reward value 和下一 recurrent state。
- augmented observation（含 previous cost）使用独立 Chan RMS 并 clip 到 `[-10,10]`，与 QCPO_refs 一致。
- 为保持 DQCAC 算法语义，通用骨干没有复制 QCPO_refs 的 state-value cost head；DQCAC 后续仍使用 action-conditioned `Z_c(history,a)` critic。网络公平性对齐 policy/reward-V backbone，constraint head 按各算法必要输出区分。
- 数值对拍：将 `QcpoRefModel(constraint=False)` 的同名权重和 RMS buffer 加载到新适配器，在随机 `T=13,B=4` 序列及非零初始 hidden state 上，mu/log_std/reward value/h/c/obs mean/var 的最大绝对误差全部为 `0.0`。
- 当前状态：适配器已验证但尚未接入训练 rollout/BPTT；下一步先接 Q-B，复现 MLP Q-B 的 100k/300k 结果，再复用同一 rollout adapter 接 DQCAC。

#### Q-B 100k/300k 结果

- Q-B 100k：job `QCPO_DynamicButton_qcal_b_gaeppo8_obsnorm_100k_s0`，W&B `pkyr3dub`，训练 `62.3s`，exit code 0。
  - 共同 100k 后 20% reward `0.5574`、趋势 `+6.950/百万步`；Q-A2 仅 `0.0385/+0.649`。
  - 同预算 DQCAC E5 为 `0.4441/+6.291`，QCPO_refs 为 `0.4776/+7.008`；Q-B 已进入相同样本效率量级。
  - 70 条终评 reward `0.5410`、outage `0.2143`；lambda 固定 0。value explained variance `0.703`，PPO KL `0.00274`、clip fraction `0.1597`。
- Q-B 独立 300k：job `QCPO_DynamicButton_qcal_b_gaeppo8_obsnorm_300k_s0`，W&B `ue1ye4kn`，训练 `148.3s`，exit code 0。
  - 后 20% reward `1.317`、趋势 `+5.059/百万步`；与 QCPO_refs 同预算 `1.355/+5.431` 接近，显著优于 Q-A2 `0.230/+1.248`。
  - 130 条终评 reward `1.621`，接近 DQCAC E5 `1.690` 与 QCPO_refs 全量终评 `1.670`；reward 主干可判为“表现正常”。
  - reward-only 的自然 outage 为 `0.546`，不满足约束是预期现象，下一阶段再恢复统一 dual/PID，不能以此否定 reward 校准。
  - value explained variance `0.641`；PPO ratio std `0.0497`、clip fraction `0.0587`、KL `0.00124`；无 NaN/Inf。
- 完整 profile：
  - `_runs/profiles/qcpo_qb_vs_key_100k_2026-07-16/`
  - `_runs/profiles/qcpo_qb_vs_key_300k_2026-07-16/`
- 最重要结论：QCPO 旧失败由两个层次叠加。未修正 rollout reuse 和未生效 obs RMS 是工程 bug；修完后 Q-A2 能学但慢。把 reward MC 换成 V+GAE 后才追平 QCPO_refs，说明低方差、逐时刻 reward credit assignment 是剩余主因。

#### E12 QCPO 接入状态

- `QCPOGPU` 新增 `policy_arch=mlp|mlp_lstm`，默认仍为 `mlp`，旧 Q-B 路径不变。
- recurrent rollout 逐步保存 augmented observation、previous action/reward、采样时 old log-prob、reward value，以及每步进入前 `h0/c0`。
- 更新时按 `recurrent_seq_len=100` 将每条 T=1000 轨迹切块，块维 shuffle 语义与 QCPO_refs 一致；每块使用 rollout 保存的初始 hidden state，8 epochs 中 behavior old log-prob 固定。
- policy PPO 与 reward-value loss 使用同一个 recurrent model、同一次 backward/optimizer step，与 QCPO_refs 的共享骨干优化方式一致。reward/risk PPO 仍分别使用 min/max。
- recurrent 专用评估器在每轮 episode batch 开始重置 previous cost/action/reward 与 hidden state，避免 stateless 统一评估器丢失历史。
- 持久化验证：
  - `QCPO_DynamicButton_smoke_recurrent_qb_s0`：小型 `[32,32]+LSTM32`、seq=16，训练 `7.1s`，exit code 0，覆盖 rollout/BPTT/联合 loss/终评。
  - `QCPO_DynamicButton_smoke_mlp_qb_regression_s0`：默认 MLP 分支训练 `7.0s`，exit code 0，确认 recurrent 分支未破坏已校准 Q-B。
- 下一门：全尺寸 `[512,512]+LSTM512`、seq=100、Q-B reward-only，先 100k seed0；目标是在相同 env steps 下至少复现 MLP Q-B 的明确正趋势，再决定扩 300k。
