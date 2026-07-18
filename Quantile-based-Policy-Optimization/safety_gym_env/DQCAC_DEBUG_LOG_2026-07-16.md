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


#### E12 全尺寸 recurrent R0 结果

- QCPO recurrent 100k：job `QCPO_DynamicButton_recur_qb_r0_qbhyper_100k_s0`，W&B `6tfy9aeo`，训练 `63.7s`，exit code 0；后段 reward `0.4006`、slope `+5.006/百万步`，终评 `0.3169`。
- QCPO recurrent 独立 300k：job `QCPO_DynamicButton_recur_qb_r0_qbhyper_300k_s0`，W&B `312u1kr3`，训练 `159.8s`，exit code 0；后段 reward `1.083`、slope `+4.114/百万步`，终评 reward `1.251`、outage `0.557`。
- 同预算 MLP Q-B 后段/终评为 `1.317/1.621`；recurrent 能学且数值健康，但当前 Q-B 超参下尚未复现 MLP，不能把 LSTM 当成无条件增益。
- 对齐 profile：`_runs/profiles/qcpo_recurrent_r0_100k_2026-07-16/`、`_runs/profiles/qcpo_recurrent_r0_300k_2026-07-16/`。
- 决策路线：保留 Q-R0；轻量验证 Q-R1（ref lr/clip）、Q-R2（value loss 系数）、Q-R3（LSTM256）；MLP Q-B 作为 Q-R4 强控制。100k 不超过 R0 则不扩预算。

### E13：DQCACBeta recurrent actor/reward-V 接入

- 状态：D-R0 已实现，默认 `policy_arch=mlp` 不变；循环分支要求 `reward_actor_mode=gae_ppo`。
- 历史输入和网络与 QCPO_refs policy/reward-V 对齐：`raw obs+previous cost` 经 RMS/MLP，拼 `previous action/reward` 后进入 LSTM+skip。
- 行为策略一致性：每个 rollout 固定 old log-prob、GAE/value target、risk advantage 和 chunk 初始 hidden；每个更新后只重算当前策略分子，绝不覆盖 behavior probability。
- critic target 一致性：循环模式的 N-step boot action 来自相同 rollout 的完整历史，而非对中间 state 使用零 hidden；terminal bootstrap 单独生成 `a_T`。
- cost baseline：从每个历史位置的行为高斯参数采 K 个动作，保持 DQC action-conditioned cost CDF advantage；分布 critics 首轮仍为 MLP/Markov 路线，不假装已完成 full recurrent critic。
- RMS：actor 的 augmented RMS 延迟到全部 PPO epoch 后更新；critic 使用独立 raw-state RMS。循环评估器重置完整历史并输出 cost CDF 校准。
- 持久化 smoke `DQCAC_DynamicButton_smoke_recurrent_dr0_s0`：`B=2,T=32,N=8,[32,32]+LSTM32`，训练 `4.7s`、exit code 0；训练与循环评估全链路通过。
- 默认路径回归 smoke `DQCAC_DynamicButton_smoke_mlp_regression_after_recurrent_s0`：相同小预算，训练 `4.5s`、exit code 0，确认循环分支没有破坏 MLP 构造、GAE/PPO、critic 或统一评估。
- 新增 `ppo/first_epoch_ratio_max_error`：首个循环 actor epoch 前 actor/RMS 未改变，理论上 ratio 必须为 1；该指标用于直接发现 history/chunk h0/old probability 接线偏差。
- 并行保留路线：D-R0 actor/V recurrent；D-R1 独立 recurrent cost critic；D-R2 full-shared recurrent hybrid；D-R3 explicit-budget-fair；D-RC 512 MLP 公平控制。
- 下一实验：D-R0/E5 reward-only 100k，预计含评估 `2–3min`；前段不形成正 slope 或 PPO/value/critic 数值异常就停止，不直接跑满。


#### E13 D-R0 100k/300k 结果

- 100k：job `DQCAC_DynamicButton_recur_dr0_e5hyper_100k_s0`，W&B `j57xp70k`，训练 `55.4s`；后段 reward `0.4278/+6.164/百万步`，终评 reward `0.4965`、outage `0.1286`。
- 同 100k E5 MLP 为 `0.4441/+6.291`，QCPO_refs `0.4776/+7.008`，循环 QCPO `0.4006/+5.006`；D-R0 基本复现旧 E5 前段，门控通过。
- 首 epoch `max|ratio-1|≈0.7e-5～1.1e-5`，证明 history/chunk h0/old probability 对齐；100k 终点 value EV `0.594`、KL `0.00338`、clip `0.192`。
- 独立 300k：job `DQCAC_DynamicButton_recur_dr0_e5hyper_300k_s0`，W&B `tavefru8`，训练 `153.1s`；后段 reward `1.381/+5.672`，终评 reward `1.442`。
- 同 300k DQC E5 MLP `1.662`、QCPO_refs `1.355`、MLP Q-B `1.317`、循环 QCPO `1.083`。D-R0 已超过同 recurrent/QCPO 路线，但仍低于最强 MLP DQC；不扩 seed，先做 LR/value coefficient 轻量消融。
- 300k 后段 KL `0.00103`、clip `0.0456`、value EV `0.722`，更像后期更新不足而非过强或 hidden bug。
- 约束警报：cost critic CDF `0.322` 对 truth `0.586`，bias `-0.263`；pred mean cost `12.47` 对真实 `23.56`。恢复 lambda 时必须先用 empirical dual，D-R1 cost-history 与 critic 校准单独验证。
- profiles：`_runs/profiles/dqc_recurrent_dr0_100k_2026-07-16/`、`_runs/profiles/dqc_recurrent_dr0_300k_2026-07-16/`。

### E14：QR transition chunking

- 新开关 `critic_minibatch_size`，默认 0 完全保留整批路径；正数按 transition 分块构造 N² TD-error。
- 每个 chunk loss 按样本比例加权，所有块只执行一次 optimizer step；不是增加 critic update 次数。
- 合成 uneven chunk 对拍 `M=11,N=7,chunk=4`：loss 误差 `4.33e-8`、prediction gradient 最大误差 `0.0`。
- 持久化集成 smoke `DQCAC_DynamicButton_smoke_chunked_qr_s0`：`T·B=64,N=16,chunk=17`，训练 `4.4s`、exit code 0。
- 用途：先解除 B=32 与 N=64/128 的峰值显存限制；性能、wall time 和 peak memory 后续作为独立实验报告。

### E15：D-R0 actor LR / shared value coefficient 轻量消融

- 公平口径：全部 `DynamicButton/seed0/B=10/T=1000/100k/lambda=0`，其余沿用 D-R0 的 LSTM512、GAE+PPO8、N=32。完整 profile：`_runs/profiles/dqc_recurrent_actor_ablation_100k_2026-07-16/`。
- 基线 `lr=2e-4,value_coef=1`：W&B `j57xp70k`，后段 reward `0.4278`、趋势 `+6.164/M`。
- 提高 actor LR 到 `3e-4`：job `DQCAC_DynamicButton_recur_dr0_lr3e4_100k_s0`，W&B `tmckgcax`，训练 `56.7s`；后段 `0.4766/+6.716/M`，终评 reward `0.5527`。后段 KL `0.00310`、clip fraction `0.178`，未出现更新过猛；该路线保留为新的 reward-only recurrent 候选。
- 仅把 shared value coefficient 降到 `0.5`：job `DQCAC_DynamicButton_recur_dr0_vcoef05_100k_s0`，W&B `x37vpl83`，训练 `59.8s`；后段 `0.3755/+5.797/M`，低于基线，故不扩 300k。它保留为负消融，说明当前证据不支持“value 梯度压制 policy”这个解释。
- 同预算 QCPO_refs 为 `0.4776/+7.008/M`，MLP Q-B 为 `0.5574/+6.950/M`。`lr=3e-4` 已追平 refs 前段，但单 seed/100k 不能宣称稳定胜出。
- 100k 的 cost critic 仍严重低估：lr 路线后段 predicted mean cost `1.13`，rollout cost mean `7.40`；终评 critic CDF `0.00045` 对 empirical outage `0.229`。因此停止 reward 小网格，转向 cost target/历史表示诊断。

### E16：cost distribution target 的分歧路线与 MC 实现

- **C-T0（保留基线）**：`cost_target_mode=nstep`，100-step QR-TD + target critic；默认值不变，历史实验可复现。
- **C-T1（当前轻量验证）**：`cost_target_mode=mc`，对完整 episodic rollout 反向计算 `G^c_t=c_t+gamma_c G^c_{t+1}`，用真实 return-to-go 直接监督 N 个 quantiles。只替换 cost target，不改变 reward critic/GAE/PPO。
- **C-T2（候选）**：若 MC 校准明显改善但方差大，做 n-step/MC anchor mixture 或 cost lambda-return；单列为消融，不能与 C-T1 混写。
- **C-H1（候选）**：独立 recurrent cost encoder + action-conditioned quantile head，显式处理 online/target history；用于检验部分可观测性。
- **C-H0.5（轻量候选）**：复用冻结 actor-history feature 接 cost quantile head；工程风险低但表示受 reward actor 漂移影响，必须与独立 encoder 分开报告。
- **C-Q（后续）**：N=64/128、smooth CDF、局部 tau/IQN；只有 mean/return target 已校准后才进入，避免用更多 quantile 掩盖整体传播偏差。
- 实现保护：MC 只允许 `episodic=True`，continuing 截断 rollout 会直接报错；scalar MC sample 重复为 N 列以保持现有 QR target-sample 求和的 loss/梯度尺度，避免同时重调 critic LR。
- 验证：语法与手算 discounted return 通过；持久化 smoke `DQCAC_DynamicButton_smoke_recur_mc_cost_s0` 使用 recurrent+chunked QR，训练 `5.8s`、exit code 0，JSON/评估全链路通过。
- C-T1 100k 已完成：job `DQCAC_DynamicButton_recur_mc_cost_lr3e4_100k_s0`，W&B `wmlzu8la`，训练 `56.9s`、exit code 0。reward 序列与 n-step 对照逐点完全一致，证明 lambda=0 时只改变了 cost 学习。
- 终评 predicted mean cost 从 n-step 的 `1.49` 提到 `4.02`，真实值 `8.96`；CDF 从 `0.00045` 提到 `0.0384`，真实 outage `0.2286`。MC 明确缓解传播低估，但 mean 仍低估 55%、CDF 仍低估 0.190，不扩 300k。profile：`_runs/profiles/dqc_cost_target_mc_100k_2026-07-16/`。
- 新增 C-O1 优化诊断：记录 reward/cost/joint 裁剪前 gradient norm 和 clip indicator；profile 工具同步纳入 cost target/gradient 指标。持久化 recurrent+chunk smoke `DQCAC_DynamicButton_smoke_mc_grad_diag_s0` 训练 `4.6s`、exit code 0。
- 下一门 C-O1：保持 MC 与 actor epochs=8，只把 critic epochs `10→20`。若校准仍无实质改善，再进入 C-H1；最终保留 `nstep/MC × Markov/recurrent-cost` 的 2×2 消融。


### E17：MC cost critic 优化与 bounded leaky-I

- C-O1 `critic epochs=20, lr=1e-3`：job `DQCAC_DynamicButton_recur_mc_cost_c20_lr3e4_100k_s0`，W&B `059boy42`，训练 `59.4s`。终评 predicted mean cost `6.97` 对 truth `8.96`（低估 22%），CDF `0.1308` 对 outage `0.2286`（bias `-0.0978`）；训练末 s0 prediction `8.24` 对当批 cost mean `10.0`。
- C-LR2 `critic epochs=10, lr=2e-3`：job `DQCAC_DynamicButton_recur_mc_cost_clr2e3_lr3e4_100k_s0`，W&B `73yevpqm`，训练 `58.2s`。终评 mean/CDF 只有 `5.91/0.0987`，低于 C-O1；故选择 C20，不用高 LR 替代。
- C20 的 joint grad norm 为 `1.37～7.86`，始终低于 clip=10；cost grad 大于 reward grad，但两 critic 参数独立且没有触发联合裁剪，排除“grad clip 导致追不上”。
- C10/C20/C-LR2 的 reward/PPO 逐点相同；cost 优化没有污染 lambda=0 的 actor 结论。C20 比 C10 只多约 `2.5s` wall time，当前 A100 采样中峰值实测约 `1.3GB`、GPU 利用率低，说明后续 B/N 有充足余量。
- profiles：`_runs/profiles/dqc_cost_optimizer_c20_100k_2026-07-16/`；C-LR2 作为日志/W&B 负消融保留。
- 新增默认兼容的 bounded leaky-I：`pid_integral_leak`、`pid_deadband`、`pid_delta_max`、`pid_reference_episodes`。默认 `1/0/inf/0` 精确复现旧 `lambda<-clip(lambda+Ki*error)`。
- episode scaling 使用几何积分和：常值误差下，一次 B=20 update 与两次 B=10 update 数值相同；手算断言覆盖 legacy exact、episode-scale exact 和 deadband。持久化 smoke `DQCAC_DynamicButton_smoke_leaky_pid_s0` 训练 `6.0s`、exit code 0。
- 首条 constrained 门控采用 C20+MC、recurrent actor、`beta=.995,sum_norm=true,outage PID`，并设 `Ki=.1,leak=.97,deadband=.02,delta_max=.05,reference=10`。这是 P-B1，不覆盖 E9 legacy-I；若 outage 偏高，路线是 leak=.98/Ki=.15；若过保守，路线是 leak=.95 或更大 deadband。


### E18：P-B1 纯 leaky-I 结果与 P-B2 PI

- P-B1：job `DQCAC_DynamicButton_recur_mc_c20_leakpid097_300k_s0`，W&B `9pxv0bmd`，启动 commit `02efba9`，训练 `158.2s`、exit code 0；配置为 C20+MC+recurrent、`beta=.995,sum_norm=true,Ki=.1,leak=.97,deadband=.02,delta_max=.05,reference=10,Kp=0`。
- 控制启动过晚：170k 以前 lambda=0；200k/250k/300k 约为 `0.012/0.056/0.130`，同期 batch outage 多为 `0.3～0.7`。130 条终评 reward `1.332`、outage `0.462`，明确不可行。
- 终评 critic CDF `0.295` 对 truth `0.462`（bias `-0.167`），pred mean cost `12.85` 对 truth `16.92`。C20+MC 在高 cost 策略分布上仍低估，后续 C-H1 必须保留。
- P-B1 证明 leak 解决过保守不等于闭环合格；纯 I + 100 episode 窗口在 300k 内响应太慢。它作为负消融保留，不通过 seed/长预算扩展门。
- P-B2 新增 `pid_Kp`，默认 0 精确保持旧 bounded-I。PI 输出为 `lambda=clip(I_state+Kp*filtered_error)`；P 项随当前窗口 error 立即出现/撤回，leaky-I 只消除稳态误差。
- 手算断言覆盖 Kp=0 legacy、Kp=1 和 lambda 下界；持久化 smoke `DQCAC_DynamicButton_smoke_pi_kp1_s0` 用 `cost_limit=0` 强制正误差，实际得到 `lambda=0.7999`，训练/评估 exit code 0。
- P-B2 保持 P-B1 所有参数，只设 `Kp=1`。按 P-B1 末 error 约 0.23 推算 lambda 会从 I≈0.13 立即升到约 0.34；若仍偏弱，后续路线是 Kp=2 或 window=50，二者不在同一 run 同时改变。


### E19：P-B2 Kp=1 结果与最后一个窗口消融

- P-B2：job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_300k_s0`，W&B `rgt6qukp`，启动 commit `1dfa3e1`，训练 `158.7s`、exit code 0；与 P-B1 唯一差异是 `pid_Kp=1`。
- P 项真实加速响应：180k/190k lambda `0.011/0.045`，P-B1 为 `0.001/0.005`；到 250k/280k/300k 为 `0.203/0.310/0.356`。I state 最终 `0.146`，P 输出约 `0.210`。
- 但控制到达工作区间太晚：130 条终评 reward `1.303`、outage `0.508`，训练最后 batch outage `0.3`。相对 P-B1 的 outage `0.462` 未改善，单 seed 波动下不能宣称 PI 更差，但明确仍不可行。
- critic CDF `0.365` 对 truth `0.508`（bias `-0.142`），pred mean `14.75` 对 truth `22.33`；较大的 lambda 不能补偿 actor risk advantage 的系统低估/表示误差。
- 只再保留 P-B3：window `100→50`，其余完全不变，让 PI 约早 5 iterations 响应。若仍不可行，停止 Kp/Ki/window 小网格，进入 C-H1 recurrent action-conditioned cost critic；Kp=2 只留作后续消融表选项，不立即运行。


### E20：P-B3 window=50 结果与 PID 网格停止门

- P-B3：job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_300k_s0`，W&B `o8nx0elk`，启动 commit `45697a5`，训练 `157.6s`、exit code 0；相对 P-B2 唯一变化是 `pid_window_episodes=50`。
- λ 在 150k 已到 `0.088`，并能在 batch outage 回落后降至 170k 的 `0.0095`；最终 `lambda=0.307=I 0.147+P 0.160`。短窗口解决一部分迟滞，但后期仍随经验概率振荡。
- 300k 后段 reward `0.9806/+3.994/M`；130 条终评 reward `1.132`、outage `0.315`。约束明显优于 P-B1/P-B2 的 `0.462/0.508`，但没有达到 `0.2`，不扩 seed。
- 终评 CDF `0.294` 对 truth `0.315`，pred mean `12.35` 对 truth `13.07`。旧 E9 seed0 可达 outage `0.2`，但 reward 仅 `0.658`；P-B3 仍只是未可行的高 reward 候选。
- 对齐 profile：`_runs/profiles/dqc_recurrent_pid_p_b123_300k_2026-07-16/`；原始导出：`_runs/wandb_export/dqc_recurrent_pid_p_b123_300k_2026-07-16/`。
- 决策：停止 Kp/Ki/window 小网格。Kp=2、Ki=.15/leak=.98、cost-quantile PI、双信号控制保留为后续消融选项；先修 cost-history/risk advantage 表示。

### E21：C-H0.5 actor-history feature 实现与分歧路线

- 根因假设：recurrent policy 下未来动作依赖 hidden，正确 cost value 是 `Z_c(s,h,a)`；D-R0 的 `Z_c(s,a)` 对策略诱导过程并非充分 Markov。
- 新开关 `cost_history_mode=raw|actor_feature`，默认 raw。actor_feature 保存 rollout 时真正产生动作的 `MLP+LSTM` feature，detach 后送入 action-conditioned cost quantile head；cost loss 不反传 actor。
- 完整接线：当前/boot/terminal feature、critic target/prediction、risk CDF、K-action baseline、constraint RMS、critic dual、s0 logging 与 recurrent eval。reward critic 与 PPO 路径不变。
- 接口回归：旧四元 actor forward 与 feature 五元接口数值逐元素相同。持久化 smoke `dqc_ch05_actor_feature_smoke_20260716` 与 `dqc_ch05_raw_regression_smoke_20260716` 均约 `6s`、exit code 0。
- C-H0.5A 的混杂因素是 feature 容量更大且随 actor 漂移。若 100k 有效，补 C-H0.5B（MLP-only feature/容量匹配 raw projection）；若无效，进入 C-H1 独立 online/target recurrent cost encoder。raw+history concat（C-H2）和允许 cost 梯度进 actor 的 full-shared（C-H3）只作为后续独立消融。
- 100k 门：C20+MC、actor lr3e-4、lambda0，只改 history mode。相对 raw C20 的 CDF bias `0.0978` 至少降 25%，或 mean-cost 相对误差从 22% 进入 15%，且另一校准量不恶化 >10%；否则不扩 300k。


### E22：C-H0.5A 100k 负结果

- job `DQCAC_DynamicButton_recur_mc_c20_ch05_actorfeature_100k_s0`，W&B `ooy3dlxs`，commit `6780296`，训练 `58.7s`、exit code 0；与 raw C20 唯一差异是 `cost_history_mode=actor_feature`。
- 两条 reward/真实 cost/PPO 序列逐点相同；终评都为 reward `0.5527`、outage `0.2286`、mean cost `8.957`，单变量隔离成立。
- raw 的 CDF/mean 为 `0.1308/6.974`；actor feature 为 `0.1129/6.634`。CDF bias 从 `0.0978` 恶化到 `0.1156`，mean 相对低估从 22.1% 恶化到 25.9%，门控失败，不扩 300k。
- profile：`_runs/profiles/dqc_cost_history_ch05_100k_2026-07-16/`；export：`_runs/wandb_export/dqc_cost_history_ch05_100k_2026-07-16/`。
- C-H1 独立 cost RNN、actor `(h,c)` C-H0.6、MLP-only 容量控制仍保留；H0.5A 无正信号后，先处理证据更直接的 hard-CDF 查询稀疏性。

### E23：C-Q1 sigmoid 查询 CDF 实现与 P-S1 门

- 新开关 `cost_cdf_mode=hard|sigmoid`、`cost_cdf_temperature`，默认 `hard/1`。hard 手算逐元素等于旧 quantile count。
- sigmoid surrogate 为 `(1/N)Σsigmoid((z_i-b)/T)`，只进入 actor actual/baseline risk CDF 与 constraint RMS；QR loss、hard calibration、empirical outage/PID 均不改。
- 新日志：smooth s0 CDF、risk advantage absolute mean/nonzero fraction、mode/temperature。它们用于判断 N=32 hard count 是否让多数 action difference 精确为 0。
- 持久化 smoke `dqc_cq1_sigmoid_cdf_smoke_20260716`：训练 `6.3s`、exit code 0，hard/smooth eval 和 JSON 全链路通过。
- P-S1 只在 P-B3 上打开 `sigmoid,T=1`；预计训练约 160s、总计约 3～3.5min。门：outage `≤0.22` 且 reward `>0.658`；outage>0.3 或前半段 reward 明显崩坏则停止 temperature 网格。
- 备选消融：T=.5/2、自适应 local spacing、N=64/128、local-τ importance weighting、uniform/query-mixture IQN；C-H1 与大 B 不在 P-S1 同时打开。

### E24：P-S1 `sigmoid,T=1` 结果与温度分歧路线

- P-S1：job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_smoothT1_300k_s0`，W&B `iisf2230`，启动 commit `6926fb8`，训练 `161.4s`、exit code 0；相对 P-B3 唯一变化为 actor risk CDF 从 hard count 改成 `sigmoid,T=1`。
- 130 条终评：reward `0.8296`、empirical outage `0.2692`、hard critic CDF `0.3474`、smooth critic CDF `0.3509`；predicted mean cost `15.63` 对 truth `11.08`，此时 critic 已由 P-B3 的轻微低估转为偏保守高估。
- 与 hard P-B3 的 reward/outage `1.132/0.315` 相比，平滑以约 `0.302` reward 换来 `0.046` outage 改善；与旧 E9 可行边界 `0.658/0.200` 相比，reward 仍高 `0.172`，但约束尚差 `0.069`。因此它是明确正信号，不是最终可行解，也不扩 seed。
- smooth 路线末段 `risk_adv_nonzero_fraction` 为 `0.865～0.999`、多数点高于 `0.92`，risk-adv std 约 `0.008～0.021`；这直接支持“hard N=32 查询导致动作差值稀疏/量化”的诊断。终评 hard/smooth s0 CDF 很接近，说明收益主要来自候选动作之间的局部连续排序，而不是简单把初始状态风险整体抬高。
- 最终 `lambda=0.256`，低于 hard P-B3 的 `0.307`，却取得更低 outage；说明 smooth actor risk advantage 对单位 λ 更有效。末期 PPO KL `0.00549`、clip fraction `0.248`，无 NaN/Inf，不能用策略更新崩坏解释 reward 损失。
- profile：`_runs/profiles/dqc_smooth_cdf_ps1_300k_2026-07-16/`；export：`_runs/wandb_export/dqc_smooth_cdf_ps1_300k_2026-07-16/`。
- **当前单变量路线 P-S2**：只把温度 `1→2`。通过门为 outage `≤0.22` 且 reward `>0.658`；若 outage 不优于 T=1，或 reward 降到 `≤0.658`，立即停止固定温度网格。
- **保留但不混跑的分歧路线**：`T=.5` 用于检验 T=1 是否过度平滑；自适应温度按查询附近 quantile spacing 定标；N=64/128 直接提高 hard-CDF 分辨率；uniform+local τ mixture 用 importance weight 保持目标分布；IQN 分 uniform-IQN 与 query-mixture-IQN 两条。若 T=2 失败，优先在 C-Q2/C-Q3 与 C-H1 中按诊断证据选择，不把多个变化塞进同一 run。

### E25：P-S2 `sigmoid,T=2` 结果、统计不确定性与 C-Q3A 门

- P-S2：job `DQCAC_DynamicButton_recur_mc_c20_pi_kp1_w50_smoothT2_300k_s0`，W&B `agmbfhlz`，启动 commit `87548a8`，训练 `159.6s`、exit code 0；相对 P-S1 唯一变化为 temperature `1→2`。
- 130 条终评 reward `0.9725`、outage `0.2385`、hard/smooth critic CDF `0.1834/0.2013`；pred mean cost `8.97` 对 truth `12.12`，最终 `lambda=0.1212`。它同时优于 T=1 的 `0.8296/0.2692`，但没过预设 `outage≤0.22` 硬门，不扩 multi-seed。
- 训练后段 reward `0.9715`、empirical probability `0.2833`；末点 risk-adv nonzero fraction `0.9808`、std `0.0145`，PPO KL `0.00187`、clip fraction `0.0957`，没有数值或更新异常。
- 统计保留意见：outage 是 `31/130`，Wilson 95% interval 约 `[0.173,0.319]`；T=1 的 `35/130` interval 约 `[0.200,0.351]`。区间高度重叠，故“Pareto 改善”只能作为单 seed 正信号，不能当显著结论。
- 固定温度主网格在 T=2 暂停；`T=.5`（是否过平滑）与 `T=4`（更宽核是否反而稀释局部差异）均记录为温度曲线消融。下一主门 C-Q3A 为 `N=64,lambda=0,100k`，相对 N=32 C20/MC 只改 N，并用 `critic_minibatch_size=2500` 保持 pairwise QR 峰值规模近似相同。
- C-Q3A 通过门：CDF bias 相对 N32 的 `0.0978` 至少下降 25%，或 mean relative error 进入 15%，且另一指标不恶化超过 10%、reward 轨迹正常。通过后才做 `N64+T2,300k`；失败则转 C-Q2 自适应 spacing/local-τ 或 C-H1。
- 工程路线 C-E1：增加 opt-in final checkpoint + eval-only 恢复，候选策略用 512/1024 episodes 复评；当前入口没有 checkpoint，P-S2 无法事后无训练复评。该缺口记录但不与 C-Q3A 同时改算法。
- 对齐 profile：`_runs/profiles/dqc_smooth_cdf_ps12_300k_2026-07-16/`；完整导出：`_runs/wandb_export/dqc_smooth_cdf_ps12_300k_2026-07-16/`。

### E26：C-Q3A N=64 直接扩展失败，定位 QR loss 尺度混杂

- C-Q3A-legacy：job `DQCAC_DynamicButton_recur_mc_c20_n64_chunk2500_100k_s0`，W&B `wqjcropf`，训练 `60.9s`、exit code 0。N=64 并未明显拖慢当前 A100：相对 N32 的 `59.4s` 仅增加约 1.5s。
- 终评 truth 与 N32 都为 reward `0.5527`、outage `0.2286`、mean cost `8.957`；N64 critic CDF/mean 为 `0.1223/6.808`，比 N32 的 `0.1308/6.974` 略差。CDF bias `0.1063`、mean relative error 24.0%，原 C-Q3A 门失败，不扩 300k。
- 根因混杂：QR loss 使用 `.sum(dim=target_quantile)`，所以 N=64 的 loss/梯度约为 N=32 的 2 倍。末点 cost grad norm `17.03`、joint clip fraction `1`；N32 C20 的 joint norm 约 `1.37～7.86` 且不触发 clip。直接扩大 N 同时改变了分辨率和优化尺度，负结果不能归因于 quantile 数。
- 新增 `quantile_target_reduction=legacy_sum|reference_mean` 与 `quantile_loss_reference_samples=32`。默认 legacy_sum 的 loss/gradient 与旧公式逐元素相同；reference_mean 乘 `32/N_target`，使 N64 scale 为 0.5、N32 保持参考尺度。
- 日志/profile 新增 target scale/reference；最终 JSON 保存 reduction 配置。手算对拍验证 legacy loss/gradient exact、N64 reference loss/gradient 精确为 legacy 的 0.5。
- 持久化 smoke `dqc_n64_reference_qr_smoke_20260716`：N64+chunk2500+reference，训练 `14.4s`、exit code 0，rollout/critic/PPO/eval/JSON 全链路通过。
- 下一门 C-Q3B：同一个 N64 100k run 只切 reference_mean。与 C-Q3A-legacy 网络形状相同，随机初始化/采样流应一致；要求真实 reward/cost 逐点相同，且 grad clip 消失、校准达到原门，才组合 N64+T2。
- 公平性分歧 C-RNG1：N32/N64 初始化会消耗不同数量的全局 RNG，跨网络宽度单 seed 的轨迹不保证配对。可选路线是 agent 初始化后统一 reseed 或分离 policy-action/critic RNG；当前先用同形 C-Q3A/B 配对，不把 RNG 修正混入本轮。

### E27：C-Q3B 尺度修正有效，但 uniform N64 仍无校准收益

- C-Q3B：job `DQCAC_DynamicButton_recur_mc_c20_n64_ref32_chunk2500_100k_s0`，W&B `61s441ku`，commit `f44bac6`，训练 `59.9s`、exit code 0。
- 与 N64 legacy 的 reward/empirical cost/终评 truth 逐点相同：reward `0.5527`、outage `0.2286`、mean cost `8.957`；单变量配对成立。
- reference scale 把末点 cost grad norm `17.03→7.67`、clip fraction `1→0`，CDF `0.1223→0.1268`、pred mean `6.808→6.994`，证明优化尺度修正正确。
- 但相对 N32 C20 的 CDF `0.1308`、pred mean `6.974` 没有实质提升；N64-ref 的 CDF bias `0.1018`、mean relative error 21.9%，未达到 25%/15% 门。停止 N64+T2 300k，N128 降为后续分辨率消融。
- profile：`_runs/profiles/dqc_quantile_resolution_n32_n64_100k_2026-07-16/`；export：`_runs/wandb_export/dqc_quantile_resolution_n32_n64_100k_2026-07-16/`。
- 下一主线 C-Q4A：只对 cost critic 使用 query-mixture τ，中心 `1-alpha=0.8`、窗口 `[0.7,0.9]`、local fraction `0.5`；CDF 用 importance/quadrature weight，prediction loss 先用 query-focused 均值。
- 分歧保留：C-Q4B 对 prediction loss 也做 importance weighting 以保持全局 W1；C-Q2 adaptive sigmoid bandwidth；C-H1 独立 cost RNN；N128；uniform-IQN 与 query-mixture-IQN。任何 local grid 都不能用未加权 quantile count 冒充 CDF。

### E28：C-Q4 cost-only query-mixture τ 实现

- 新开关 'cost_quantile_grid_mode=uniform|query_mixture'，默认 uniform；query center 默认 '1-alpha'，另有 half-width/local-fraction。reward critic 始终使用原 uniform τ。
- mixture 用解析 inverse CDF 的 deterministic stratified grid，不消耗随机数；N32、center .8、half-width .1、fraction .5 时 [.7,.9] 有 19 个 heads，uniform 只有 7 个。
- CDF/分布 mean/std/critic dual/评估和 n-step target sample 均使用归一化 '1/g(τ)' weight；prediction loss 独立支持 'query_focused|importance'。
- 合成检验：weights sum=1，min/max '0.01031/0.06186'；在 q_i=τ_i 的可解析例子，τ=.8 上尾加权值 '0.2165'，未加权会错误为 '0.34375'。默认 uniform loss/CDF/mean/std 与旧公式逐元素 exact。
- 持久化回归 smoke 'dqc_cost_grid_uniform_regression_smoke_20260716' 训练 '8.6s'、exit 0；query smoke 'dqc_cost_grid_query_mixture_smoke_20260716' 训练 '13.8s'、exit 0，均覆盖 recurrent rollout、QR/PPO、评估和 JSON。
- C-Q4A 门：N32+C20+MC+lambda0 100k，只打开 query_mixture/query_focused 与 reference/ref32。要求 truth 轨迹与 N32 baseline 一致；CDF bias 从 '0.0978' 降到 '≤0.0734'，或 mean relative error 进入 15%，另一指标不恶化 >10%。
- 若 A 通过，组合 query grid+smooth T2 做 300k；若 A 接近但 mean 明显失真，跑 C-Q4B importance prediction；若 A 全面无效，优先 C-H1，不继续调 local fraction/window 小网格。

### E29：C-Q4A/B 100k 结果，停止 local-grid 小网格

- C-Q4A query-focused：W&B 'tvgqpcip'，训练 '58.4s'、exit 0；终评 CDF '0.1386'、pred mean '6.934'，truth '0.2286/8.957'。CDF bias '0.0900'，只比 uniform '0.0978' 改善约 8%。
- A 的 cost grad norm '14.04'、clip fraction '1'，query-focused 同时放大了优化尺度；因此补预设 C-Q4B，而不把 A 的小改善当 local τ 结论。
- C-Q4B importance-prediction：W&B 'fw5276fj'，终评 CDF '0.1374'、pred mean '7.099'；cost grad '9.07'、clip '0'。CDF bias约 '0.0912'、mean relative error 20.7%，仍未过门。
- 三条 run 的 reward/真实 cost/PPO 逐点相同；local grid 两种 loss 都只有小幅校准收益。停止 fraction/window/N 小网格，不跑 local+T2 300k。
- profile：'_runs/profiles/dqc_local_quantiles_q4ab_100k_2026-07-16/'；export：'_runs/wandb_export/dqc_local_quantiles_q4ab_100k_2026-07-16/'。
- local τ 实现保留为正消融（查询步长更细但校准收益有限）；下一核心路线 C-H1 独立 recurrent cost encoder。C-Q2 adaptive bandwidth、IQN、P-M1 safety setpoint 与 checkpoint/eval-only 仍保留。

### E30：C-H1 独立 recurrent cost encoder 实现与 100k 校准门

- 新增默认关闭的 cost_history_mode=cost_lstm；raw 与 actor_feature 的默认/历史语义不变。C-H1 输入与 QCPO_refs recurrent policy 相同：augmented observation=[state, previous_cost]，MLP feature 再拼 previous_action/previous_reward，经过独立 LSTM。
- 关键区别：它不复用漂移的 actor feature。cost MLP+LSTM 只由 cost quantile regression loss 更新；与 actor 只共享无参数的 augmented-observation running mean/variance，保证输入尺度一致但不共享可学习表示。
- cost quantile head 继续显式接 action，因此模型是 Z_c(history,a)，没有退化成 QCPO_refs 的 state-value cost head。reward critic、GAE、PPO old probability、actor 和 PID 均未改。
- 训练使用 recurrent_seq_len=100 的 truncated BPTT：每个 chunk 包含连续 100 步和全部并行环境；hidden/cell 数值向后传递、chunk 边界 detach。各 chunk loss 按 transition 数占比累积，一次 joint gradient clip 和 optimizer step，避免隐式放大 critic 学习率。
- 每次 encoder optimizer step 后都重算并 detach 当前 rollout 的 cost feature；constraint RMS、risk actual/baseline、actor 与日志由同一版本 feature 查询。s0 critic-dual、训练日志和 recurrent eval 使用 previous cost/action/reward 全零的独立 cost feature，不会误用 actor feature。
- online/target cost encoder 同步创建并 Polyak 更新。首轮 C-H1 刻意限制为 finite-horizon MC target；recurrent n-step target 需要单独构造 t+N history，是另一条算法变量，不能混入本轮。
- 静态检查与独立梯度测试通过：feature shape=(20,3,32)，encoder gradient L1=5.48。持久化全链路 smoke dqc_ch1_cost_lstm_smoke_20260716 训练 6.4s、exit 0；旧 actor_feature 回归 dqc_ch05_regression_postch1_20260716 训练 6.3s、exit 0。
- 100k C-H1A 配方保持 N32+C20+MC+lambda0+hard CDF 与 raw C20 相同，唯一算法变量是 cost_history_mode=cost_lstm。预计训练约 2～4 分钟、70 条评估约 1 分钟。
- 通过门沿用预注册标准：相对 raw C20 的 CDF bias 0.0978 至少下降 25%（即不高于 0.0734），或 predicted mean relative error 从 22.1% 进入 15%，且另一指标不恶化超过 10%、reward/PPO 正常。明显失败就不扩 300k。
- 分歧路线全部保留：C-H1B 为独立 encoder 自有 RMS；C-H1C 为 256 hidden 的容量/速度控制；C-H1N 为 recurrent n-step online/target history；C-H0.6 直接使用 actor (h,c)；C-Q2 adaptive bandwidth；uniform/query-mixture IQN；P-M1 safety setpoint；C-E1 checkpoint/eval-only。双向 LSTM 因使用未来信息违反在线因果性，不列为合法主路线。

### E31：C-H1A 负结果与 C-W1 cost 时间目标对齐

- C-H1A：job DQCAC_DynamicButton_recur_mc_c20_ch1_costlstm_100k_s0，W&B 8vm989u8，commit b7f8a9c，训练 72.6s、exit 0。固定 eval truth 与 raw C20 完全相同：reward 0.5527、outage 0.2286、mean cost 8.957。
- 独立 LSTM 的终评 CDF/mean 只有 0.0643/5.800；raw C20 为 0.1308/6.974。CDF bias 从 0.0978 恶化到 0.1643，mean relative error 从 22.1% 恶化到 35.3%，未过门，不扩 300k。
- profile 显示最后一次训练 s0 prediction 为 7.208，raw 为 8.240；cost QR loss 却从 raw 43.52 降到 14.09。低 loss 与差 s0 calibration 同时出现，说明并非简单欠拟合。
- 目标错配证据：最后一批 episode-level s0 cost mean 约 15，而把全部 return-to-go transition 等权平均后的 cost target mean 只有 5.697。T=1000 时每条 episode 只有 1 个 s0，却有大量后期低 remaining-cost transition；更灵活的 LSTM 能降低全局 loss，但可牺牲真正决定 initial outage/actor risk 的早期状态。
- C-H1 的 cost/joint grad 在 70k 和 100k 分别约 20.7/17.6 并触发 clip=10；提高 clip=30（C-H1A2）与 hidden=256（C-H1C）保留为欠拟合/容量消融，但当前先修证据更直接的 transition objective。
- 完整 profile：_runs/profiles/dqc_cost_history_ch1_100k_2026-07-16/；export：_runs/wandb_export/dqc_cost_history_ch1_100k_2026-07-16/。
- 新增默认兼容的 cost_critic_time_weighting=uniform|risk_discount。risk_discount 使用 max(discount^t,floor)，整批归一化到 mean=1；因此只改 transition 相对质量，不改变 loss 总尺度。默认 uniform 走旧浮点分支，loss 逐元素 exact。
- 加权已覆盖 full/chunk/TBPTT 三路径；chunk 使用全批归一化权重的切片，按 chunk size 聚合，合成测试与 full weighted loss 等价。新增 min/max/ESS、discount/floor 日志和 profile 指标。
- T=1000、discount=.995、floor=0 时归一化权重范围为 0.0337～5.033，ESS fraction=0.3937（每条轨迹约 394 个等效 transition）；比 beta=.95 的极端约 39 个等效 transition 更稳健，也与最终 constrained 配方 beta=.995 对齐。
- 持久化 smoke dqc_cost_time_weight_smoke_20260716 训练 6.4s、exit 0，覆盖加权 QR、PPO、评估和 JSON。
- 下一门 C-W1：raw+C20+MC+lambda0，设置 beta=.995 与 risk_discount/.995；lambda=0 时 beta 不影响 actor，所以真实 reward/cost 应与 raw C20 逐点相同。仍用 CDF bias不高于0.0734或 mean error不高于15%的门；失败则不扫 discount 小网格，转 s0/early stratified replay 或 direct initial-distribution auxiliary loss。

### E32：C-W1 100k 通过，时间目标错配是主瓶颈

- job DQCAC_DynamicButton_recur_mc_c20_timew995_100k_s0，W&B cojepq8r，commit 0d33c8a，训练 58.8s、exit 0。
- 与 raw C20 的 reward、真实 cost/PPO 逐点相同；固定终评 truth 都是 reward 0.5527、outage 0.2286、mean cost 8.957，单变量隔离成立。
- predicted mean cost 从 6.974 提升到 9.003，relative error 从 22.1% 降到 0.51%；hard CDF 从 0.1308 提升到 0.1701，absolute bias 从 0.0978 降到 0.0585，改善约 40.2%。同时通过 mean-error 15% 与 CDF-bias 改善25%两条预注册门。
- 最后训练 rollout truth outage为0.30；raw CDF 0.1844，C-W1为0.2094，误差从0.1156降到0.0906。收益不是终评随机抽样偶然。
- T=1000、discount=.995 的 weight min/max/ESS 与预期一致：0.0337/5.033/0.3937；loss 总尺度维持mean weight=1。训练时间比 raw 59.4s 还少0.6s，差异属噪声，可认为无额外 wall-time成本。
- profile：_runs/profiles/dqc_cost_time_weight_100k_2026-07-16/；export：_runs/wandb_export/dqc_cost_time_weight_100k_2026-07-16/。
- 结论：当前首要瓶颈是均匀 transition QR objective 与 initial-outage/β-risk 的时间分布错配，不是 N=32 分辨率，也不是缺少 LSTM history。C-H1、N64、local τ 的负/弱结果现在得到统一解释：它们没有改变监督质量在时间上的分配。
- 下一门 C-W2：在当前最好 P-S2（T2 sigmoid + window50 PI）上只加入 risk_discount/.995，300k seed0。硬门：outage不高于0.22且reward高于E9的0.658；相对P-S2还要求校准不恶化。预计训练约160s、总计3～4min。
- 保留消融：hard-CDF+C-W1（隔离平滑交互）、discount .99/.997（只做曲线消融，不在当前通过点继续扫）、direct s0 auxiliary、early stratified replay、C-H1+time-weight（检验history在正确目标下是否才有用）。

### E33：C-W2 结果——校准后的 critic 与平滑 CDF 叠加导致过度保守（2026-07-16）

- C-W2 job `DQCAC_DynamicButton_recur_mc_c20_timew995_pi_kp1_w50_smoothT2_300k_s0`（W&B `lkdu70i5`）从 commit `abd86c6` 启动，训练 `164.7s`、exit code 0。相对 P-S2 唯一增加 `cost_critic_time_weighting=risk_discount, discount=.995`；其余 N32、C20、MC、raw cost critic、sigmoid T=2、window50 PI、beta=.995 与 P-S2 相同。
- 130 条终评为 reward `0.4798`、outage `0.1308`、mean cost `9.838`、hard/smooth critic CDF `0.0822/0.0882`、predicted mean cost `4.472`，最终 lambda `0.1273`。相对 P-S2 的 `reward=0.9725, outage=0.2385`，outage 下降 `0.1077`，但 reward 损失 `0.4927`。
- CDF absolute calibration error 为 `|0.0822-0.1308|=0.0486`，略优于 P-S2 的 `|0.1834-0.2385|=0.0551`。因此风险方向并未失效；失败点是 actor 被推到明显过安全的策略区域。按预注册门，虽然 outage 远低于 `0.22`，reward `0.4798<0.658`，故 C-W2 不进入多 seed。
- 后 60k 训练窗口中，P-S2 的 reward/outage/lambda 为 `0.9715/0.2833/0.1730`，C-W2 为 `0.6090/0.1833/0.2235`；C-W2 最后两次 rollout 的 outage 都为 0，reward 为 `0.448/0.520`。这不是终评 130 条造成的偶然偏差，而是训练末策略本身已经保守。
- 两条最终 lambda 很接近（P-S2 `0.1212`，C-W2 `0.1273`），但行为差异很大。直接原因不是 PID 给了更大的 lambda，而是时间加权改善查询区域后，同样单位 lambda 产生更强、更连续的 risk gradient；再与 T=2 平滑 CDF 叠加，风险梯度强度被重复放大。
- 末段 PPO KL 约 `0.00385`、clip fraction `0.187`，无 NaN/Inf；cost grad norm 均值约 `11.34`，joint critic clip fraction 约 `0.667`。梯度裁剪可能限制 critic 跟随末期策略分布，但不能解释策略为何更安全；提高 critic clip 到20（C-WG1）应作为独立优化消融，而不是与控制强度调整同时改变。
- 终评 predicted mean `4.472` 对 truth `9.838` 仍明显低估，说明 on-policy 最后一次训练批的加权拟合不能保证终评状态分布上的全局 mean 校准。由于 hard CDF 误差只有 `0.0486`，当前 actor 查询点仍比全局 mean 更可信；direct s0 auxiliary、early replay 与 checkpoint 后大样本评估继续保留。
- profile：`_runs/profiles/dqc_cost_time_weight_cw2_300k_2026-07-16/`；完整 history：`_runs/wandb_export/dqc_cost_time_weight_cw2_300k_2026-07-16/`。

### E34：C-W3 单变量计划——保留时间加权，只把 smooth CDF 改回 hard CDF

- C-W3 与 C-W2 唯一差异是 `cost_cdf_mode=sigmoid→hard`；温度仍记录为2但 hard 路径不使用。它同时也是相对 hard P-B3 只增加 time weighting 的正交对照。
- 目的：检验 C-W2 的过度保守是否来自“校准增强 + 平滑查询”叠加。理论预期是 hard CDF 降低候选动作风险差的连续强度，使结果落在 P-B3 的 `1.132/0.315` 与 C-W2 的 `0.480/0.131` 之间。
- 仍用 300k seed0 与 130 条终评；预计训练约160秒、含导出分析约5分钟。硬门保持 outage `≤0.22` 且 reward `>0.658`，校准不能明显差于 C-W2/P-S2；失败不扩 seed。
- 若 C-W3 仍过保守，分歧路线依次保留为：actor risk discount `beta .995→.99`；PID 内部 safety setpoint 小于0.2；降低 Kp 或 lambda gain。若 C-W3 不安全，则考虑 time-weighted sigmoid `T=.5/1`。这些都按单变量轻量门测试，不并入同一 run。

### E35：C-W3 结果——critic 已校准，但 hard CDF 的动作风险差仍过稀疏（2026-07-16）

- C-W3 job `DQCAC_DynamicButton_recur_mc_c20_timew995_pi_kp1_w50_hard_300k_s0`（W&B `9wtkz26h`）从 commit `3619c93` 启动，训练 `162.0s`、exit code 0；相对 C-W2 唯一变化为 `sigmoid T2→hard`。
- 130 条终评 reward `0.9481`、outage `0.3385`、mean cost `14.131`、cost quantile `21.0`、critic CDF `0.3168`、predicted mean `13.997`，最终 lambda `0.1000`。reward 通过 `>0.658`，但 outage 明显未通过 `≤0.22`，故不扩 seed。
- 这条 run 的 critic 校准是当前 constrained run 中最好的一组：CDF error `0.0216`，mean relative error约 `0.95%`。因此约束失败不能再主要归因于 critic 低估；即使查询概率准确，hard count 仍不能给候选动作提供足够密集、稳定的局部排序梯度。
- 后60k reward/outage/lambda约为 `0.934/0.286/0.165`；末点 risk-adv std `0.0145`，与平滑路线相当，但 nonzero fraction只有 `0.396`。P-S2/C-W2 的 nonzero fraction分别约 `0.981/1.000`。核心差异是“有多少状态动作收到风险方向”，而不是单纯 risk std 幅值。
- C-W2 与 C-W3 把合理区间夹出：T2 得到 `reward/outage=0.480/0.131`，hard 得到 `0.948/0.338`。两者都数值稳定，说明需要校准 actor 查询核/风险增益，而不是继续堆 critic 容量。C-W3 末段 cost grad clip 较多，但终评校准非常好，因此 C-WG1 提高 critic clip 降为次级优化。
- 五条 hard/T1/T2/time-weight 对齐 profile：`_runs/profiles/dqc_cost_time_weight_cw23_300k_2026-07-16/`；完整 history：`_runs/wandb_export/dqc_cost_time_weight_cw23_300k_2026-07-16/`。

### E36：C-W4 计划——time-weighted sigmoid T=1

- C-W4 与 C-W2 唯一差异是 temperature `2→1`；与 C-W3 相比则只把 hard 查询换成较窄 sigmoid。目标是在 hard 的高 reward/高 outage 与 T2 的低 reward/低 outage 之间寻找可行 Pareto 点。
- 仍用 300k seed0、130 条终评，预计训练约160秒、总计约4分钟。硬门不变：outage `≤0.22` 且 reward `>0.658`；同时记录 CDF calibration、risk-adv nonzero fraction、PPO KL/clip。
- 若 T1 不通过，停止固定 `.5/1/2/hard` 主线网格：若仍不安全，说明需要介于 T1/T2 的自适应/连续增益；若过安全，则说明策略对温度非常敏感，应显式引入 risk gain 或按 local quantile spacing 自适应带宽，而不是用温度碰运气。T=.5/1.5 只保留为论文温度曲线，不作为无止境主线调参。

### E37：C-W4 结果——训练曲线看似通过，但 final PPO 后安全性瞬间失效（2026-07-16）

- C-W4 job `DQCAC_DynamicButton_recur_mc_c20_timew995_pi_kp1_w50_smoothT1_300k_s0`（W&B `5lfdb3hn`）从 commit `03f8ddd` 启动，训练 `157.9s`、exit code 0；相对 C-W2 唯一变化是 temperature `2→1`。
- 训练后60k 的 reward/outage/lambda 为 `0.973/0.143/0.0308`，最后两个 rollout 为 reward `1.220/1.218`、outage `0.10/0.10`，最后 lambda `0`。仅看训练日志，它会同时通过 reward 和 outage 门。
- 但 130 条 final evaluation 为 reward `1.1999`、outage `0.5308`、mean cost `18.315`、cost quantile `29.2`；critic CDF `0.1454`、predicted mean `8.536`。最终策略高回报但严重不安全，CDF bias达到 `-0.385`。
- 代码时序解释了表面矛盾：每条训练日志描述本轮 rollout 的更新前策略；随后 PID 根据 safe window 把 lambda 从 `0.0202→0`，再执行8次 PPO。final evaluation 描述这8次更新后的策略。最后一次 reward-only PPO 的 KL仅 `0.00236`、clip fraction `0.124`，但 Safety-Gym 的风险边界对小策略位移高度敏感，outage可从0.1跃迁到0.53。
- 这不是“多跑130条后发现B=10噪声”可以完全解释：critic predicted mean/CDF 与训练前策略分布一致，却对更新后的 final policy 严重低估；训练数据和最终动作分布已经发生闭环 distribution shift。固定温度主网格停止。
- 六条 CDF/time-weight 对齐 profile：`_runs/profiles/dqc_cost_time_weight_cw234_300k_2026-07-16/`；完整 history：`_runs/wandb_export/dqc_cost_time_weight_cw234_300k_2026-07-16/`。

### E38：C-E1 评估 checkpoint/eval-only 实现与验证

- 新增默认关闭的 `checkpoint_dir` 和 `checkpoint_interval`。DQCAC 在 rollout 完成后、dual/critic/PPO 任何更新前保存 `pre_update_rollout_policy`；统一入口在训练结束后另存 `post_update_final`。phase、iteration、env_steps和rollout reward/outage均写入payload，禁止把两个相位混为一谈。
- checkpoint 保存 agent 直接持有的全部 `nn.Module`（actor、reward/cost critic、targets、obs RMS）、lambda/PID运行时标量和完整结构配置；不保存optimizer/scheduler动量，格式明确命名为 `safety-gym-eval-checkpoint-v1`，用途是严格恢复评估而非声称无损续训。
- 写盘采用同目录 `.tmp` + `os.replace` 原子替换；SSH中断不会把半文件当成有效快照。`--eval_only <path>` 自动从payload恢复algo/env/seed/网络结构，默认关闭W&B，允许用`--set num_envs=...`调整评估并行度。
- 持久化 smoke `dqc_checkpoint_eval_smoke_train_20260716`：2×2×100=400 env steps，训练 `6.1s`、exit 0；生成step200/400两个pre-update和一个post-update final，各326KB。两个独立eval-only job均严格加载、评估、写JSON并回收worker，exit 0。
- step400 pre/post actor SHA256前16位分别为 `5a39d09395733a38` 与 `fa4f4d591309a5b5`，证明phase确实保存不同策略而非重复文件。JSON正确记录`checkpoint_loaded/phase/env_steps`。
- 下一验证 C-E2：复跑 C-W4 且每20k保存一次，先用 rollout 指标筛选 2～3 个快照，再用统一512条评估确认是否存在 reward>0.658 且outage≤0.22 的真实策略。预计训练约160秒；15个快照的空间在 /vepfs 项目盘，不使用20G根目录。若只有pre-update安全而下一次update立即失效，则主修正应是lambda hysteresis/floor或安全setpoint，不是best-checkpoint掩盖训练不稳定。

### E39：C-E2 phase-aligned 快照复评——B=10 rollout 不能替代独立大样本约束判断（2026-07-16）

- C-E2 job `DQCAC_DynamicButton_cw4_timew995_pi_kp1_w50_smoothT1_ckpt20k_300k_s0`（W&B `4dd8zrqf`）从 commit `ad965d7` 启动，训练 `158.3s`、exit 0。它逐点复现 C-W4 的30个 reward/outage/PPO记录，证明保存checkpoint不改变RNG或训练行为。
- 每20k保存一次，共15个pre-update快照和1个post-update final，总计168MB、每个约11MB。便宜门筛出140k/180k/260k/280k/300k五个点，但只对信息量最高的300k和控制最强的180k做独立评估。
- 300k pre-update在训练B=10上是 `reward=1.218,outage=0.10`；140条独立评估却是 `1.219/0.471`，critic CDF/mean为 `0.0790/5.889`，truth outage/mean cost为 `0.471/17.286`。因此即使排除最后PPO相位，10条轨迹仍造成严重选择偏差和critic泛化错觉。
- 180k快照保存时lambda `0.1473`、B=10为 `reward=0.836,outage=0.10`。140条复评为 `0.746/0.264`；扩到520条后为 reward `0.7148`、outage `123/520=0.2365`、Q80 cost `16`、mean cost `10.742`。critic CDF `0.3162`偏保守约0.080，pred mean `12.270`。
- 520条outage的Wilson 95%区间约 `[0.202,0.275]`。它接近真实alpha=0.2且reward超过E9的0.658，但点估计仍高于预注册 `≤0.22` 门，不能宣布可行。其余更弱控制快照不再评估。
- 结论：策略空间中已经出现接近可行、reward约0.715的点，但训练闭环没有稳定停留；checkpoint只揭示问题，不能作为掩盖不稳定的最终算法trick。

### E40：P-M1 PID safety setpoint 实现与正式实验计划

- 新增 `pid_target_prob`，默认None即`target=q_alpha`，旧控制式逐位不变。显式0.15时仅把empirical PID control gap改为`window_prob-0.15`；真实约束gap、critic阈值、最终评估仍严格使用alpha=0.2。
- 日志分离 `dual/prob_gap=window_prob-0.2` 与 `dual/control_prob_gap=window_prob-target`，并记录target/safety margin。这样不会把保守设点误报成论文约束改变。
- 手算测试：同一window outage=0.3、旧pid_i=.1时，默认/显式target=.2都精确输出lambda `.1850000024`；target=.15输出`.2399999946`，真实gap仍0.1、control gap为0.15。语法和diff check通过。
- 持久化 `dqc_pid_target015_smoke_20260716`：400 env steps，训练6.1s、完整评估exit0；启动摘要正确显示`target0.15`。
- P-M1保持C-W4的N32/C20/MC/time-weight.995/sigmoidT1/window50 PI等全部配置，只设`pid_target_prob=.15`，每20k保存快照。预计训练约160秒，总计3～4分钟。
- 通过门：优先看post-update final 130条；若reward>0.658且outage≤0.22，再用520条确认。若接近门则评phase-aligned候选；若明显过保守，下一单变量是target .175或降低Kp；若仍不安全，下一路线是target .10或lambda floor/hysteresis。T1.5、adaptive bandwidth与risk gain保留为actor查询核消融，不与P-M1同时改。

### E41：P-M1 300k 结果——安全设点有效，但控制器在短预算末端仍未稳定（2026-07-16）

- job `DQCAC_DynamicButton_pm1_timew995_pi_target015_smoothT1_ckpt20k_300k_s0`，W&B `tkrudpbq`，从 commit `ce0262f` 启动；训练 `156.6s`、exit code 0，共保存15个pre-update快照和1个post-update final。
- 130条final evaluation为 reward `0.7223±0.4531`、outage `0.2462`、mean cost `9.131`、Q80 cost `17`；critic hard/smooth CDF为 `0.2728/0.2734`，predicted mean cost为 `10.970`，最终lambda为 `0.4038`。
- 与无safety-setpoint的C-W4 final `reward/outage=1.1999/0.5308` 相比，outage下降约`0.2846`，reward仍高于旧E9的`0.6576`。但点估计仍高于预注册`≤0.22`门，因此不能宣称已经可行，也暂不扩多seed。
- 这条run的末段不是稳定平台：lambda从110k开始介入，240k后由`0.231→0.290→0.309→0.349→0.325→0.405→0.404`；最后训练窗口outage为`0.4`，最后一批rollout outage为`0.2`。控制器在300k正处于追赶风险、牺牲reward的阶段，尚未完成闭环收敛。
- PPO末点KL `0.00704`、clip fraction `0.3059`，数值有限但更新幅度偏大；cost grad norm `22.66`且触发clip。没有NaN/Inf或进程错误，所以当前没有“立即停止”的工程证据。
- 结论定级：P-M1 300k是“接近约束且末段仍在改善/调整”，不是明确失败。它应进入1M稳定性验证；若1M仍持续大周期振荡，再单独测试lambda hysteresis/floor、较小actor LR/更少PPO epoch或PID增益，而不是在300k直接换掉算法。

### E42：短跑证据的使用边界与长跑晋级规则（2026-07-16）

- `100k`只用于机制筛选和查错：可以确认梯度尺度错误、目标测度错配、NaN、明显退化，但“没有早期提升”不能写成“长期无效”。N64、local-quantile和cost-LSTM当前结论严格限定为对应seed和100k预算下无早期收益。
- `300k`用于闭环初筛：若末段指标已稳定且明显劣于基线，可以停止；若lambda、outage或reward仍呈系统性改善/补偿趋势，则必须晋级。P-M1属于后者。
- `1M`用于判断controller与actor-critic能否形成稳定平台；至少在300k、600k、1M做phase-aligned checkpoint复评。只有灾难性数值错误，或多个连续窗口稳定落在无价值的支配区域，才提前停止。
- 最终论文结论不得依赖单seed短跑。候选配置需要至少3个seed、每个至少1.5M；与QCPO_refs做完全公平最终比较时应使用相同环境步数，优先补到其`5M`预算，并报告均值、方差和约束置信区间。
- 随机初始化确实可能制造短期false negative，特别是LSTM、更大的quantile critic和PID闭环；但它不能推翻确定性的代码/目标证据，例如N64旧实现梯度随N翻倍、uniform-transition目标不对齐initial outage、rollout与post-update final相位错位。这三类问题无需靠长跑“等它自己好”。
- 下一实验P-M1-L1保持P-M1全部参数不变，只把训练从300k延长到1M并每50k保存checkpoint。按当前吞吐估计纯训练约`8.7min`，加130条终评约`10min`；全程由`launch_background.sh`持久化运行。


### E43：P-M1-L1 延长到1M——600k曾改善，但1M回退，确认闭环极限环（2026-07-16）

- job `DQCAC_DynamicButton_pm1_timew995_pi_target015_smoothT1_ckpt50k_1m_s0`，W&B `p4hbaqtv`，启动commit `ad1c042`（算法代码仍为`ce0262f`）；训练`508.4s`、exit code 0。100个iteration完整记录，无NaN/Inf；每50k保存，共20个pre-update加1个final，约221MB，均位于/vepfs项目盘。
- 1M post-update final 130条为 reward `0.6279±0.7048`、outage `0.2462`、mean cost `11.592`、Q80 cost `16.2`；critic hard/smooth CDF `0.2839/0.2851`，predicted mean `14.284`，lambda `0.4181`。相对300k final的`0.7223/0.2462`，outage没有改善、reward反而下降。
- 分段训练均值揭示非单调动态：0–300k reward/outage/window/lambda为`0.580/0.217/0.191/0.105`；300–600k为`0.702/0.197/0.213/0.213`；600k–1M变成`0.750/0.303/0.298/0.417`。后200k raw/window outage均约`0.305/0.300`，lambda均值已到`0.455`，仍未把策略稳定拉回约束内。
- phase-aligned 520条独立评估为：
  - 300k：reward `0.7764±0.4346`，outage `129/520=0.2481`，Wilson 95% `[0.2129,0.2870]`；critic CDF `0.3214`，bias `+0.0733`。
  - 600k：reward `0.8143±0.6583`，outage `117/520=0.2250`，Wilson 95% `[0.1912,0.2628]`；critic CDF `0.1053`，bias `-0.1197`。
  - 1M：reward `0.6325±0.6478`，outage `127/520=0.2442`，Wilson 95% `[0.2093,0.2829]`；critic CDF `0.2801`，bias `+0.0359`。
- 600k确实优于300k，证明“短跑会漏掉后期改善”不是空想；但1M又回退，且三个点的critic bias在正/负之间大幅翻转。它不是单调慢收敛，而是critic分布漂移、B=10窗口噪声、PID延迟和PPO策略步长耦合形成的极限环。
- 不能直接把raw risk-advantage与reward advantage比较：代码实际会除以`constraint_sigma_ema`。按真实归一化后，risk/reward有效std比值从0–300k的约`0.60`升到600k–1M的`1.47`，末200k约`1.65`；所以后期不安全不是“risk项绝对太小”。更可能是动作风险排序噪声/critic漂移使方向不够可靠，增大lambda只会放大这个非平稳信号。
- 完整history：`_runs/wandb_export/dqc_pm1_1m_2026-07-16/`；profile与图：`_runs/profiles/dqc_pm1_1m_2026-07-16/`；三个520条JSON均保留在`_runs/`。

### E44：P-M2 预注册——actor LR与QCPO_refs对齐到1e-4（2026-07-16）

- P-M2相对P-M1-L1只把`theta_lr0=3e-4→1e-4`。QCPO_refs同样使用8 PPO epochs、clip=.1，但optimizer LR是`1e-4`；当前DQCAC末200k KL/clip均值已到约`0.00519/0.241`，终点为`0.00661/0.323`，对风险边界来说更新偏激进。
- 因更小LR早期必然更慢，本实验不再用300k否决，直接跑1M、每50k保存phase checkpoint；预计训练约8.5–9分钟，final 130条约1分钟。
- 预注册判断：final 130条只做便宜screen；若reward `>0.658`且outage `≤0.27`，或600k–1M末段呈持续安全改善，则扩520条。正式通过仍要求520条点估计outage `≤0.22`且reward `>0.658`，并要求300k/600k/1M不存在P-M1那样的大幅回退。
- 若LR降低能明显压低KL/clip却仍有risk周期，下一正交变量是`num_envs=20`（降低outage/PID和critic batch方差）；若KL稳定但action risk方向仍噪声，测试`num_action_samples=16`；若单次更新偶发越界，新增默认关闭的PPO target-KL early stop。三条路线不混入P-M2。
- 当前仍只做seed0配置选择。随机初始化false negative的可能性保留；任何最终候选必须在3个seed复现，不能用P-M1-L1 seed0否定整个算法族。


### E45：P-M2 1M结果——匹配QCPO_refs的1e-4 LR减小更新，但不改善outage（2026-07-16）

- job `DQCAC_DynamicButton_pm2_timew995_pi_target015_smoothT1_lr1e4_ckpt50k_1m_s0`，W&B `3qqyab3s`，训练`509.2s`、exit code 0；相对P-M1-L1唯一变化是actor LR `3e-4→1e-4`。
- 130条final为 reward `0.6909±0.6167`、outage `0.2462`、critic CDF `0.2200`、pred mean `9.609`、lambda `0.3818`。520条确认评估为 reward `0.6498`、outage `127/520=0.2442`、Q80 cost `16`、critic CDF `0.2166`，bias `-0.0277`。同时未通过reward `>0.658`和outage `≤0.22`两条门，不扩phase快照或多seed。
- 小LR确实压低部分PPO更新：终点KL/clip从P-M1的`0.00661/0.3227`降到`0.00598/0.2221`。但末200k均值只从`0.00519/0.2414`降到`0.00485/0.2219`，幅度有限。
- 代价明显：P-M2末200k训练reward均值`0.3868`，P-M1为`0.7522`；600k–1M raw outage从P-M1的`0.3025`降到`0.2525`，但final大样本outage仍同为`0.2442`。P-M2在830–900k同样出现outage上冲、lambda上升和reward短暂变负，只是周期相位改变。
- 结论：DQCAC的actor LR不能仅因网络/PPO结构相同就机械迁移QCPO_refs参数。较小LR缓和更新并降低训练窗口outage，却没有消除risk signal/dual闭环的非平稳性，而且损害样本效率；它不作为当前推荐配置。
- 对齐导出与图：`_runs/wandb_export/dqc_pm1_pm2_lr_1m_2026-07-16/`、`_runs/profiles/dqc_pm1_pm2_lr_1m_2026-07-16/`。

### E46：P-M3 预注册——固定1M环境步，将num_envs从10增至20（2026-07-16）

- P-M3以P-M1为基线，只改`num_envs=10→20`，并将`num_iterations=100→50`保持总环境步严格等于1M。actor LR恢复`3e-4`，其余PID、critic、CDF、网络和8 PPO epochs不变。
- 每次rollout从10条增到20条，outage=.2时超限轨迹期望数从2增到4；critic每轮看到双倍不同初始状态。PID已有`pid_reference_episodes=10`，leaky-I在常值error下一次B20更新与两次B10严格等价，避免仅因batch变大暗改积分时间尺度。
- 固定env-step比较下，50次policy update各使用20k transitions，P-M1为100次各10k；每个样本仍复用8个PPO epoch。该实验同时检验更低batch方差与更低policy/control更新频率是否能破坏极限环，属于num_envs本身的算法效应。
- 当前128 CPU/A100 80GB资源对B20+N32安全；pairwise QR显存风险主要出现在B与N同时放大。本run每100k保存checkpoint，预计训练约8–10分钟，130条终评约1分钟。
- screen与P-M2相同：final 130若reward `>0.658`且outage `≤0.27`则扩520；正式门为520条outage `≤0.22`且reward `>0.658`，并检查中后期是否比B10少一次完整风险周期。
- 若B20失败，下一单变量为`num_action_samples=4→16`，直接降低冻结risk baseline在8个PPO epoch中被重复放大的Monte-Carlo噪声；随后才实现target-KL early stop。B20、K16和target-KL分别报告，不组合成无法归因的“trick包”。


### E47：P-M3 B20 通过seed0长预算大样本门（2026-07-16）

- job `DQCAC_DynamicButton_pm3_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s0`，W&B `ky7voxle`，训练`390.6s`、exit code 0；相对P-M1仅把B从10增至20、iteration从100减至50，总环境步仍严格为1M。
- 训练吞吐从P-M1的约`1,966 env steps/s`提高到约`2,560 env steps/s`，steps/s提升约30.2%，同预算训练时间下降约23.3%。B20+N32在A100 80GB上无OOM/NaN，硬件可承受。
- 末200k训练均值为 reward `0.7108`、raw/window outage `0.215/0.212`、lambda `0.2804`、KL/clip `0.00286/0.1557`。P-M1对应为`0.7522/0.305/0.300/0.4546/0.00519/0.2414`。B20只损失约0.041 reward，却把训练outage降低0.09并明显压低PPO位移。
- built-in 140条final为 reward `0.7725±0.5300`、outage `32/140=0.2286`、critic CDF `0.1130`。按预注册扩到520条后，final为 reward `0.7180±0.5201`、outage `87/520=0.1673`、Q80 cost `13`、mean cost `7.940`。
- 520条outage Wilson 95%区间为`[0.1377,0.2018]`：点估计明显低于0.2，上界仅比0.2高0.0018。它同时通过reward `>0.658`与outage `≤0.22`，是当前首个1M、520条大样本下通过的DQCAC配置。
- 不能只报告更好看的520条而隐藏140条的0.2286。两者差异说明有限样本和并行评估seed集合仍能造成约0.06波动；因此seed0通过只是晋级依据，不是论文最终结论。
- critic仍低估：520条hard/smooth CDF `0.1058/0.1086`，相对truth `0.1673`偏低`0.0615`，pred mean `6.241`也低于truth `7.940`。B20主要改善batch/control方差和更新频率，并未彻底解决critic跨批泛化。
- 三条1M对齐导出与图：`_runs/wandb_export/dqc_pm123_1m_2026-07-16/`、`_runs/profiles/dqc_pm123_1m_2026-07-16/`。

### E48：P-M3 seed1/2多seed复现计划（2026-07-16）

- seed1/2完全复用P-M3 B20配置，各1M环境步、每100k checkpoint；不改target、LR、K、critic或CDF。两条持久化后台并行运行，CPU worker总数40、A100显存仍有充分余量。
- 单条独占训练实测约6.5分钟；按此前双run共享GPU经验，并行后预计两条约9–12分钟完成，含各自140条built-in评估约11–14分钟。并行会影响wall time，不影响固定env-step算法比较；每条单独记录耗时。
- 140条只作screen。每个新seed若reward `>0.658`且outage `≤0.27`，再串行做520条final确认；无论成败都报告，不以best seed替代三seed统计。
- 最终汇总使用seed0/1/2每seed相同评估协议，报告reward跨seed均值/标准差、总outage计数、每seedWilson区间和三seed聚合区间。若至少2/3 seed通过且聚合outage≤0.2，B20进入主推荐；否则它只作为有益但不足的组件。
- K16、PID uncertainty buffer/window和recent-rollout critic replay全部暂停到多seed结果后，避免在尚未确认B20泛化前继续叠加变量。

### E49：P-M3 多seed结果——B20确定有益，但尚不足以作为稳定主配置（2026-07-16）

- seed1/2 的1M训练均正常结束，耗时分别为 `713.4s/717.9s`，W&B run 为 `lcllgvhx/fwsc50vq`。两条run并行共享GPU，因此单条wall time高于seed0独占时的`390.6s`，但固定的1M环境步数和算法配置不变。
- 内置140条final评估的seed0/1/2 reward-outage为 `0.7725/0.2286`、`0.8404/0.2429`、`0.8451/0.1786`。三条都通过了宽松screen，所以按预注册规则一律扩到同协议520条，没有丢弃不好看的seed。
- 520条结果为：seed0 reward `0.7180±0.5201`、outage `87/520=0.1673`、Wilson 95% `[0.1377,0.2018]`；seed1 reward `0.8622±0.5809`、outage `159/520=0.3058`、区间 `[0.2677,0.3467]`；seed2 reward `0.8670±0.4263`、outage `110/520=0.2115`、区间 `[0.1786,0.2487]`。
- 跨seed reward为 `0.8157±0.0847`（seed间sample std），outage为 `0.2282±0.0707`。合并事件计数为 `356/1560=0.2282`，事件级Wilson 95%区间 `[0.2081,0.2497]`。合并区间只描述这三个已抽seed的episode不确定性，不能消除明显的seed间异质性。
- 按预注册门，seed0和seed2的点估计同时通过`reward>0.658/outage<=0.22`，即2/3 seed通过；但聚合outage `0.2282>0.2`，故整体门失败。B20不进入“已稳定的主推荐”，只记为已证明能降低PPO/control方差、提高吞吐的正向组件。
- 末200k训练rollout outage均值在seed0/1/2上只是`0.215/0.180/0.220`，看起来三者都接近可行；然而seed1的520条独立outage是`0.3058`。这个`+0.126`的差距证明，训练末段的20条on-policy轨迹仍不能替代大样本初始状态泛化评估。
- critic低估在三个seed都存在：hard CDF/truth分别为 `0.1058/0.1673`、`0.1043/0.3058`、`0.1657/0.2115`，其中seed1低估`0.2015`个概率点。因此下一阶段不应把“再多跑几步”当作主修复，而应降低action-baseline噪声并增强cost critic对recent/initial-state分布的holdout校准。
- 完整history与三seed对齐图位于 `_runs/wandb_export/dqc_pm3_b20_multiseed_1m_2026-07-16/` 和 `_runs/profiles/dqc_pm3_b20_multiseed_1m_2026-07-16/`。

### E50：对“现在训练是否太短”的最终判断与早停规则（2026-07-16）

- 结论不是简单的“够”或“不够”。100k足够做机制级screen和发现确定性bug，但不足以否定有慢变量的PID/LSTM/distributional-critic组合；300k足够做闭环趋势screen，但只有当末段已平稳且被基线支配时才可早停；1M是稳定性验证，不是所有组合的默认入场成本。
- P-M1给出了false-negative的直接实例：同一seed的520条复评从300k的`reward/outage=0.776/0.248`改善到600k的`0.814/0.225`，所以“早期不好、之后变好”确实会发生。但到1M又退化为`0.632/0.244`，说明它是闭环相位/极限环，而不是单调的慢收敛。
- 随机初始化可能改变某个工作点出现的时间，但无法解释三seed在同一1M预算下的大幅critic低估和outage差异。当前“失败组合单靠更长训练稳定反转”的优先级已降低；不臆造数值概率，因为三个seed不足以可靠估计这个概率。
- 继续长跑的晋级条件：末20%的reward或constraint仍有持续改善趋势；lambda/critic显示仍在有方向地追赶而非重复周期；至少一个独立大样本checkpoint比早期明显改善；或组件理论上存在明确的慢传播时常。
- 应当早停的条件：末段已平稳且reward/outage同时被基线支配；多个checkpoint重复同样的风险周期；配对试验证明变量只修正梯度尺度却没有改善校准；或失败由确定性目标错配造成。这些情况下继续加步数只会浪费资源。
- 评估协议保持两层：140条只作screen，520条作单seed确认。在outage约0.2时，520条的二项标准误约`0.0175`，95%半宽约`0.034`；140条半宽约`0.066`，不适合判定贴边界的安全性。最终候选至少3 seeds × 1.5M，与QCPO_refs正式比较时再对齐其5M环境步数。
- 当前下一单变量路线是在B20上把`num_action_samples=4→16`，检验冻结后在8个PPO epoch重复使用的action risk baseline方差；同时把recent-rollout/initial-state replay和holdout CDF calibration作为更直接针对seed1 critic低估的独立路线。两者不在同一跑中混合。

### E51：P-M4预注册——B20基础上仅将action baseline K4→K16（2026-07-16）

- P-M4完全复用P-M3 seed0的1M配置，只把`num_action_samples=4→16`。网络、N32、MC cost target、time weighting、T1 sigmoid、PID target=.15、actor LR、8 PPO epochs、B20和总环境步都不变。
- 代码核对确认K只影响两个无梯度的cost-action baseline查询：constraint RMS更新和首个recurrent PPO epoch的冻结risk advantage。实际动作CDF不变，K个baseline动作均从rollout保存的behavior Gaussian参数采样；冻结后的同一risk weight继续供8个PPO epoch使用。因此K4的baseline Monte Carlo标准误在相同动作离散度下约为K16的2倍，噪声还会被8次重复使用。
- K16不会修复cost critic对初始状态的系统偏差，它检验的是“critic给定时，action相对风险排序是否被K4采样噪声放大”。recent/initial-state replay与holdout calibration仍是另一条独立路线，不能把两者同时加入后再归因。
- 100k不足以否定此变量，因为其收益通过数十次policy/dual更新积累；本次直接跑1M，但仍按100k保存phase-aligned checkpoint。参考K4独占GPU训练`390.6s`，K16增加无梯度critic前向，预计纯训练约7～10分钟、含140条终评约9～12分钟。
- 基本门仍为520条`reward>0.658,outage<=0.22`。相对K4 seed0还要求末200k训练outage不高于0.22、reward不低于约0.70，且KL/clip或风险周期至少一项显示稳定性不恶化；否则即使偶然终评好看也不晋级。
- 140条只作是否扩520的screen。若P-M4通过，下一步优先在曾失败的seed1做同配置1M压力复现；若失败则停止K小网格，不试K8/K32碰运气，转recent/initial-state critic replay与holdout校准。

### E52：P-M4 K16完整1M结果——中期PPO更稳，但后期风险周期与回报退化仍存在（2026-07-16）

- job `DQCAC_DynamicButton_pm4_timew995_pi_target015_smoothT1_b20_k16_ckpt100k_1m_s0`，W&B `rx3mofgr`，正常exit 0。纯训练`393.1s`，K4为`390.6s`；K16额外无梯度cost-critic前向没有形成可测的wall-time代价，也无OOM/NaN。
- 140条final为reward `0.6030±0.4853`、outage `25/140=0.1786`、mean cost `7.743`、Q80 cost `13.2`；hard/smooth critic CDF为`0.1346/0.1370`，pred mean `6.838`，lambda `0.2120`。安全点估计较好，但reward低于预注册`>0.658`门。
- 0–300k时K4/K16的reward为`0.344/0.346`、outage `0.113/0.120`，没有早期优势。300–600k时K16确实表现出局部收益：outage从K4的`0.237`降到`0.163`，KL/clip从`0.00404/0.207`降到`0.00222/0.117`，reward仅`0.685→0.664`。
- 这个中期好点没有保持。600k–1M的K16 outage为`0.270`，K4为`0.235`；680k单批outage达到`0.65`，lambda随后升至约`0.51`。末200k K16/K4 reward为`0.621/0.711`，window outage为`0.236/0.212`，KL为`0.00382/0.00286`，clip为`0.1888/0.1557`。
- 因此K16减小action baseline Monte Carlo误差可以短暂降低PPO位移和中期outage，但它不是后期策略—critic—PID周期的主瓶颈。更长的1M预算再次避免了只看300–600k而把K16误判为成功。
- 由于末段趋势门和140条reward门同时失败，不扩520，不跑seed1，也不继续K8/K32小网格。K16保留为负/阶段性正消融；主线转recent/initial-state cost critic replay、holdout CDF calibration或直接s0 auxiliary。
- 完整history和对齐图：`_runs/wandb_export/dqc_pm3_k4_pm4_k16_1m_2026-07-16/`、`_runs/profiles/dqc_pm3_k4_pm4_k16_1m_2026-07-16/`。

### E53：C-S0 recent初始状态辅助目标与prequential holdout实现（2026-07-16）

- 新增默认关闭的`cost_s0_aux_coef`和`cost_s0_replay_batches`。显式开启目前要求`cost_target_mode=mc,cost_history_mode=raw`，防止把n-step bootstrap或旧actor feature语义混入首轮消融。
- 每个新rollout保存实际`(raw s0, behavior a0, 完整trajectory MC cost)`。有限deque只保留最近若干批；raw observation不缓存归一化值，辅助更新时重新通过当前RMS，避免旧输入尺度污染。
- cost目标采用归一化凸组合：`L_cost=(L_transition+c*L_s0)/(1+c)`。因此`c=.25`对应基础/辅助权重`0.8/0.2`，改变监督时间测度而不增加cost objective总尺度，和直接把loss相加或抬高critic LR严格区分。
- 新增训练前/后同批校准。pre使用刚采到、尚未进入本轮任何optimizer step的真实`s0/a0/return`，是prequential holdout；post使用完全相同样本但更新后的critic。指标包含hard CDF、truth、bias/absolute error、逐状态Brier、predicted/true mean cost。它不重采动作、不消耗RNG。
- W&B/profile新增aux loss/scale/replay样本数及pre/post全套校准指标；最终JSON也保存最后一次pre/post字典。profile新增s0 probability与Brier面板。
- 默认精确回归：改前/改后同一两环境短run的obs RMS、actor、reward critic/target、cost critic/target、lambda与runtime逐tensor完全一致。Module hash分别为obs `725eed9a0dd35f01`、actor `b1bb7c15a8c9e891`、reward `7845310dde10f777`、reward target `ff1133ce8d624aba`、cost `aa8d0b0d2b19382a`、cost target `bbbe962d16969daf`。
- 辅助边界回归：两批默认与`coef=.25,replay=2`对照中，obs RMS、actor、reward critic/target hash完全相同，只有cost critic及target不同；证明lambda=0时没有意外改动策略/reward分支。chunk smoke训练`8.8s`、full-batch smoke `8.9s`，均exit0并完成recurrent PPO、QR、checkpoint/JSON和评估。
- profile旧数据回归也通过；缺少新列的历史导出仍可正常生成报告和overview。

### E54：C-S0A 100k固定策略校准门预注册（2026-07-16）

- 为隔离critic效果，C-S0A使用P-M3的B20/N32/MC/time-weight.995/recurrent网络和8 PPO epochs，但设`lambda_max=0`，总预算100k。先跑aux关闭的新基线，再只开`coef=.25,replay_batches=1`；两条policy、reward和真实cost轨迹理论上应逐点相同。
- replay=1首先检验“直接提高当前s0监督质量”本身；只有它改善post/final却没有改善下一批pre holdout时，才晋级replay=4检验跨批稳定。这样不会把direct s0 weighting和old-policy replay一次混入。
- 每条5次B20 rollout、共100条训练初始状态，内置140条独立评估。参考P-M3吞吐，预计纯训练约40秒、含评估约1.5～2分钟；均用持久化后台串行运行。
- 工程门：两条actor/真实reward-cost逐点相同，无NaN/OOM，aux cost总尺度与grad clipping不能显著恶化。
- 算法门：相对新基线，独立评估CDF absolute bias或mean-cost relative error至少改善25%；同时最后三批prequential CDF error或Brier至少改善20%，否则只能说明同批记忆，不能晋级闭环。
- 若C-S0A通过，先做replay=4的同预算正交门，再把胜者放回B20/PID跑300k；若失败停止coef小网格，转独立holdout early stopping或显式ensemble/uncertainty，而不是扫`.1/.25/.5/1`碰运气。


### E55：C-S0A结果与训练预算判定——局部机制100k已足够，策略性能不能据此下结论（2026-07-16）

- 三条持久化后台run均正常exit 0：baseline/replay1/replay4的W&B分别为`j6cd5p1c/5lpevd9t/uyjull0z`，纯训练耗时`45.9/46.5/46.6s`。配置、seed和100k数据预算完全相同，只改变`cost_s0_aux_coef/replay_batches=0/1、.25/1、.25/4`。
- 固定`lambda_max=0`后，三条最终真实policy评估逐位相同：140条reward `0.19234±0.17293`、outage `0.05`、mean cost `3.51429`、Q80 cost `6.2`。baseline与replay1 checkpoint的obs RMS、actor、reward critic/target、lambda及全部runtime状态逐tensor exact，只有cost critic/target的6个参数张量变化。这排除了初始化、策略采样和actor更新差异。
- baseline/replay1/replay4的最终hard CDF分别为`0.01473/0.01629/0.01183`，相对真实`0.05`的absolute error为`0.03527/0.03371/0.03817`。replay1仅改善`4.43%`，replay4反而恶化`8.23%`，均未达到预注册`25%`门。
- 最终predicted mean cost为`2.1851/2.2838/2.0512`，相对真实`3.5143`的relative error约`37.82%/35.01%/41.63%`；replay1只改善约`7.4%`，replay4恶化约`10.1%`。
- 最后三批60k/80k/100k的prequential pre-CDF absolute error三者完全相同，均值`0.017708`；pre-Brier baseline/replay1/replay4为`0.016732/0.016699/0.016699`，改善不足`0.2%`。同批更新后的post-Brier为`0.017611/0.015462/0.017171`，replay1改善约`12.2%`、replay4约`2.5%`，仍低于`20%`门。辅助项主要改变见过数据的拟合，没有改善下一批泛化。
- 结论：停止recent-s0 replay的coef/replay小网格，不把它放入PID闭环跑1M。100k在这里不是用来评价最终策略，而是固定策略下提供5批独立数据刷新和100次cost-critic更新，直接检验该局部作用链；继续长跑缺乏晋级证据。
- 对“短跑会不会误杀慢热组合”的统一规则：100k只否决确定性bug或可直接观测且不过门的局部机制；策略/dual/LSTM组合不能据此作长期结论，至少给300k趋势窗，仍在改善则续到600k/1M。P-M1的520条复评`0.776/0.248→0.814/0.225→0.632/0.244`已经证明中期可改善、后期又回退；因此候选看多个checkpoint而非单个终点，最终至少3 seeds，并与QCPO_refs对齐相同环境步数。
- 完整history与对齐图：`_runs/wandb_export/dqc_cs0_baseline_r1_r4_100k_2026-07-16/`、`_runs/profiles/dqc_cs0_baseline_r1_r4_100k_2026-07-16/`。


### E56：C-S0B统计功效修正与冻结高风险策略校准预注册（2026-07-16）

- 用户质疑100k是否过短后复核事件数：C-S0A五个B20 rollout的outage为`0/0/0/0/0.05`，即前80k没有任何超限轨迹，最后一批仅1条；最终140条truth也只有`0.05`。因此E55能否定“早期低风险分布上已有泛化收益”，但不能外推为接近约束边界时永久无效。
- 为避免重新跑1M等待策略进入风险区，新增默认关闭的`--critic_calibration_from`。它只恢复成熟checkpoint的actor、recurrent actor RMS和raw critic observation RMS；cost/reward critic与optimizer重新初始化，lambda/PID/runtime不恢复，并强制actor、dual及两套RMS冻结。
- 不同critic架构构造会消耗不同随机数，因此所有Module构造和policy-only恢复后再次设置Python/NumPy/Torch/CUDA seed。这样同rollout seed的baseline/候选使用相同环境布局、动作噪声和真实cost轨迹。
- 两个持久化smoke均exit0：旧eval-only回归通过；校准模式完成rollout、MC QR和评估，日志明确`actor_updates/iter=0,policy_frozen=True,obs_stats_frozen=True`。训练后actor（含内部RMS）与raw obs normalizer相对源checkpoint逐tensor exact，源cost critic未加载。
- C-S0B使用P-M3 seed1的1M final actor（原520条outage `0.3058`）作为相关高风险策略，另用rollout seed101。baseline与`coef=.25,replay4`各跑5×B20×T1000=100k、20 critic updates/rollout、140条配对终评；预计每条纯训练约45～60秒、含评估约1.5～2分钟，持久化后台串行。
- 统计功效门：100条训练轨迹至少出现10次超限，否则不作负结论，改为增加冻结数据量；在事件充分时，候选须使最后三批prequential CDF error/Brier至少改善20%，且独立终评CDF absolute error或mean-cost relative error至少改善25%。只改善post同批仍判为记忆，不晋级。
- 若C-S0B仍无pre/final信号，才停止simple replay coef网格并转direct exceedance classifier、bootstrap ensemble/upper-confidence CDF或cross-fit early stopping；若通过，再给冻结critic 300k确认，之后才放回PID闭环。


### E57：C-S0B 100k结果与一次性300k功效扩展（2026-07-16）

- baseline/aux的W&B为`rwqj0kch/x5inqzzi`，训练`46.4/45.9s`、exit0。五批outage逐点同为`0.20/0.10/0.15/0.25/0.30`，共`20/100`次超限；140条终评reward/outage/mean cost也exact为`0.8542/0.2643/10.9786`。actor与两套RMS逐tensor exact，统计功效和配对门均通过。
- 最后三批pre-CDF absolute error从`0.19010→0.15729`，改善`17.3%`，接近但未过20%门；pre-Brier只改善`5.8%`。pre mean-cost absolute bias从`4.0248→3.1092`，改善`22.8%`。post-CDF error改善`21.9%`，更像提高同批/近期拟合。
- 独立终评给出相反结论：hard CDF `0.21964→0.20826`，truth固定`0.26429`，absolute error从`0.04464`增到`0.05603`（恶化`25.5%`）；mean-cost relative error从约`13.63%`增到`16.76%`（恶化约`23%`）。因此100k未通过原晋级门，不能放回PID。
- 额外发现：reward critic checkpoint并非exact。两组参数不共享，但当前`update_critic`对reward+cost参数做联合global grad clip；aux改变joint norm后会连带缩放reward critic梯度。这不影响冻结actor的真实策略比较，却说明“只改变cost监督”仍有优化器级耦合，后续应把separate clipping作为独立工程消融。
- 因只有100个s0标签，且最后三批pre-CDF连续同向改善、仅差2.7个百分点，执行一次明确封顶的300k功效扩展：同source policy/rollout seed重新跑baseline与aux各15批，不改coef或任何其他参数。预计每条训练约2.3分钟、含评估约3分钟，总计约6分钟。
- 300k终止门：至少约30个超限事件；最后三分之一pre-CDF/Brier须改善20%，独立终评CDF或mean relative error须改善25%。不通过即停止simple replay，不跑1M、不扫coef；通过也只进入一次PID 300k门，不直接宣称有效。


### E58：C-S0B 300k大样本确认与live seed1压力测试预注册（2026-07-16）

- baseline/aux训练`121.0/121.2s`、exit0，15批outage逐点exact，总事件`89/300`；actor、actor RMS与raw obs RMS逐tensor exact。140条终评truth exact为reward/outage/mean cost `0.8657/0.2643/10.7929`。
- 140条critic结果从baseline CDF/mean `0.19799/9.4077`变为aux `0.27031/11.2698`，truth `0.26429/10.7929`；CDF absolute error约`0.0663→0.0060`，mean relative error约`12.8%→4.4%`。
- 为消除训练长度消耗不同evaluation RNG的影响，从两个final checkpoint各做同构eval-only 520条。truth逐位exact为reward `0.83494`、outage `146/520=0.28077`、mean cost `11.0173`；baseline CDF/mean为`0.19964/9.4186`，aux为`0.26665/11.2476`。CDF error降低`82.6%`，mean relative error从`14.5%`降到`2.1%`（约降低`85.6%`），确认跨布局泛化而非同批记忆。
- 最后五批pre-CDF/Brier只改善约`10.0%/4.1%`，未过原20%小批门；但B20逐批truth在0.15～0.45间变化，方差远高于520条固定协议。结论定级为“独立大样本强通过、小批pre门未通过”，不声称所有校准指标一致改善。
- 下一步只在P-M3最差的seed1加入`cost_s0_aux_coef=.25,replay_batches=4`跑完整1M；其余网络、B20、C20、MC/time-weight.995、T1、PID target=.15、8 PPO epochs、LR和checkpoint周期完全不变。独占GPU预计训练约6.5～8分钟、含140条评估约8～9分钟，持久化后台运行。
- live通过门：用相同eval-only 520协议比较原seed1 baseline `reward/outage=0.8622/0.3058`。候选需reward保持`>0.658`且outage`≤0.22`，或至少把outage降低0.08且reward保持≥0.75；同时critic CDF absolute error至少降低50%，末200k不能出现更严重极限环。未通过则说明校准改善未转化为policy控制，不扩seed0/2。

### E59：live seed1 1M——校准改善转化为显著安全收益，但牺牲reward（2026-07-16）

- job `DQCAC_DynamicButton_pm3_s0aux025r4_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`，W&B `erayfl8y`，训练`392.3s`、exit code 0。相对原P-M3 seed1只增加`cost_s0_aux_coef=.25,cost_s0_replay_batches=4`，其余网络、B20、C20、PID、PPO、LR与1M预算完全相同。
- 140条内置终评中，baseline到candidate的reward/outage为`0.8404/0.2429 -> 0.7379/0.2429`；小样本没有显示安全收益，但critic CDF从`0.1002`变为`0.2295`，相对同一个truth `0.2429`的absolute error从`0.1426`降到`0.0134`。
- fresh eval-only 520条正式结果为：baseline reward `0.8622±0.5809`、outage `159/520=0.3058`、Wilson 95% `[0.2677,0.3467]`；candidate reward `0.6875±0.5783`、outage `98/520=0.1885`、区间 `[0.1572,0.2243]`。outage绝对下降`0.1173`、相对下降`38.4%`；reward绝对下降`0.1747`、相对下降`20.3%`，但仍高于预注册`>0.658`下限。
- 520条critic CDF从baseline的`0.10427`变为candidate的`0.23696`。相对各自policy truth `0.30577/0.18846`，absolute calibration error由`0.20150`降到`0.04850`，降低`75.9%`，通过至少减半的校准门。
- 时间结构不能被final点掩盖：300–600k candidate相对baseline的reward/outage为`0.827/0.207 vs 0.783/0.243`，看似全面占优；600–800k却回摆为`0.639/0.300 vs 0.823/0.260`，lambda均值`0.454 vs 0.341`；800k–1M为`0.603/0.170 vs 0.825/0.180`。因此600k会产生false positive，完整1M揭示了真实reward代价和中段极限环。
- 结论：recent-s0 auxiliary不再是“只改善冻结critic”的局部trick；它在最差seed1上把大样本outage拉回0.2以内，并通过live晋级门。但它没有消除闭环周期，而且安全收益以明显reward损失为代价，尚不能作为稳定主配置。
- 下一步保持单变量设计，在seed0/2各跑完整1M；不根据seed1结果回调coef或replay次数。两条并行预计训练约12分钟、含140条终评约14分钟。只有多seed聚合仍改善约束且reward可接受，才继续1.5M/同QCPO_refs预算确认。

### E60：C-S0 live三seed结论——救回seed1但把风险转移到seed2，不进入1.5M（2026-07-16）

- seed0/2完整1M训练均exit0，耗时`696.2s/695.7s`，W&B为`nle3jm1q/alf53xvv`。由于seed1已经证明140条outage会漏掉520条改善，本轮透明取消原140 screen，对两个不好看的seed都做了fresh eval-only 520，没有选择性丢弃。
- 520条baseline→aux结果：seed0 `reward 0.7180→0.5152, outage 0.1673→0.1558`；seed1 `0.8622→0.6875, 0.3058→0.1885`；seed2 `0.8670→0.6320, 0.2115→0.3288`。aux在所有seed都降低reward，只在seed1提供大安全收益，并在seed2造成等量反向恶化。
- 三seed reward从`0.8157±0.0847`降到`0.6116±0.0880`。outage从`0.2282±0.0707`变为`0.2244±0.0920`；合并事件仅从`356/1560`降到`350/1560`，Wilson 95%区间`[0.2081,0.2497]`与`[0.2043,0.2457]`高度重叠，不构成有意义的聚合安全改善。
- per-seed CDF absolute error从baseline的`0.0615/0.2015/0.0458`变为aux的`0.0273/0.0485/0.1586`。平均误差虽从0.1029降到0.0781，但seed2误差扩大3.46倍；simple recent replay把一致低估换成更高的seed间偏差，没有形成可靠校准器。
- 末200k训练rollout的三seed均值同样显示reward/outage约为baseline `0.808/0.205`、aux `0.647/0.202`：训练窗口安全几乎不变，reward代价已出现。独立初始状态上的改善只发生在部分seed，不能靠挑checkpoint解释。
- 长度判断：100k不足以否定s0 replay，300k冻结策略发现其校准价值，1M seed1发现live价值；但3 seeds×1M已足以拒绝当前`.25/r4`作为主配置。继续1.5M可能改变周期相位，却没有跨seed定向改善证据，低优先级且不值得直接续跑。
- 下一步停止simple coef/replay网格，转向不会把同批反馈直接灌回共享critic的cross-fit/target-stabilized calibration，以及限制PPO闭环回摆的target-KL early stop；二者分别做默认关闭的单变量消融。

### E61：P-M5 target-KL实现、回归与1M预注册（2026-07-16）

- commit `955b05f`新增DQCAC专用`ppo_target_kl`，默认0完全关闭。正阈值在epoch forward得到behavior→current approximate KL后、backward前判断；越界probe不做optimizer step，剩余actor epochs跳过，但critic的20次更新继续跑满。日志同时记录configured/completed epochs、early-stop与update-applied。
- 固定seed tiny recurrent金样本在改动前后比较：actor、reward/cost critic及target、RMS、lambda与全部runtime逐tensor/逐字段exact，eval JSON exact；唯一新增summary/config字段为`ppo_target_kl=0`。
- `target=1e-8`触发smoke中，每轮configured actor epochs=3、completed=1、early_stop=1、最后probe update_applied=0；critic `learning_steps=9=3×3`，exit0。所有临时checkpoint/job/log/离线W&B已清理。
- 用既有六条1M history选阈值而不扫网格：在P-M3 baseline中第8 epoch KL超过`.004`的rollout占18%，s0-aux中占25.3%；baseline 0–300k仅2.2%，300–600k与600–800k均33.3%。aux的KL与同批outage相关系数约0.394，因此`.004`能命中中段回摆而不会普遍砍掉早期更新。
- P-M5以原P-M3 seed1为压力基线，只增加`ppo_target_kl=.004`，不带s0 aux；B20、C20、N32、T1、PID、LR和8个最大epoch全部不变。直接跑1M，因为100k/300k几乎不会触发该机制，短跑没有检验功效。
- 原seed1 fresh520为reward/outage `0.8622/0.3058`。候选要求520条outage至少下降0.08并到`≤0.22`，reward保持`≥0.75`，且末200k不出现更大的risk周期；无论140条screen好坏都执行520。预计训练6～7分钟、140终评约1～2分钟、520复评约4分钟，总计约11～13分钟，持久化后台运行。

### E62：P-M5 seed1完整1M通过大样本门，已晋级多seed（2026-07-16）

- P-M5相对P-M3 seed1只增加`ppo_target_kl=.004`，不加入已经被三seed否决的recent-s0 replay。job为`DQCAC_DynamicButton_pm5_targetkl004_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`，W&B run为`54y9bpza`；1M纯训练`391.2s`，后台job与内置评估均exit code 0，无NaN/Inf。
- target-KL实际参与了优化，而非未触发的装饰项：50个rollout中19个提前停止actor更新，共完成`308/400`个允许的actor epoch；前/中/后20%阶段的early-stop率为`30%/40%/60%`，后20%平均只完成`4.6/8`个epoch。critic更新不受影响。
- 训练末20%均值相对P-M3 seed1由reward/outage/lambda/KL=`0.8253/0.1800/0.2206/0.00246`变为`0.9185/0.2350/0.1977/0.00387`。target-KL提高了后段reward并限制越界epoch，但训练小批outage略高，说明它可能减慢risk/PID纠偏，不能仅凭训练曲线宣布成功。
- 内置140条final为reward `0.8960`、outage `28/140=0.2000`、Q80 cost `14.2`。随后从同一个post-update final checkpoint做fresh eval-only 520条，得到reward `0.8684±0.5429`、outage `88/520=0.16923`、Q80 cost `13`、mean cost `8.8212`；作业`dqc_pm5_targetkl004_1m_s1_final_eval520_20260716`正常退出。
- 与严格同协议P-M3 seed1的`reward=0.8622±0.5809,outage=159/520=0.30577,Q80=18,mean cost=12.3308`比较，reward差为`+0.0062`，近似95%区间`[-0.0621,+0.0746]`，可视为持平；outage绝对下降`0.13654`，差值近似95%区间`[-0.18760,-0.08548]`。单独Wilson 95%区间从baseline `[0.26771,0.34667]`降为candidate `[0.13946,0.20386]`。
- critic也有改善但尚未校准：初始状态CDF absolute error从`|0.10427-0.30577|=0.20150`降至`|0.29561-0.16923|=0.12638`，约改善37.3%，但由严重低估转成明显高估。P-M5的主要收益不能归因于critic目标变化，因为critic配置与P-M3完全一致，更符合“限制8-epoch PPO过冲、改变策略—PID周期相位”的机制。
- 该结果通过预注册门：outage不高于0.22、相对baseline下降超过0.08、reward不低于0.75，且520条而非140条给出一致方向。因此已原样启动seed0/2各1M的持久化后台复现，不调阈值、不挑checkpoint。两条并行预计约11～13分钟完成训练和内置评估，随后各做同协议520条fresh evaluation；多seed裁决仍使用全部三个seed，不能只保留压力seed1。
- 对齐history与图位于`_runs/wandb_export/dqc_pm3_seed1_vs_pm5_targetkl004_1m_2026-07-16/`和`_runs/profiles/dqc_pm3_seed1_vs_pm5_targetkl004_1m_2026-07-16/`。seed0/2后台job分别为`DQCAC_DynamicButton_pm5_targetkl004_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s0`和同名`s2`。

### E63：P-M5三seed裁决——聚合更安全，但固定KL阈值不稳定支配baseline（2026-07-16）

- seed0/2完全复用seed1的P-M5配置，各训练1M步，训练耗时`700.3s/690.2s`，W&B run为`deugfqif/5y3izber`；两条后台作业、内置140评估和fresh 520评估均exit code 0。并行共享GPU使wall time高于seed1独占的`391.2s`，环境步与算法预算不变。
- target-KL在seed0/1/2分别触发`15/19/20`个rollout，实际完成actor epoch为`321/308/305`，合计`934/1200`，即跳过22.2%的最大更新。它不是仅对seed1生效的偶然开关。
- 520条严格同协议baseline→target-KL为：seed0 reward/outage `0.7180/87/520=0.1673 → 0.7140/127/520=0.2442`；seed1 `0.8622/159/520=0.3058 → 0.8684/88/520=0.1692`；seed2 `0.8670/110/520=0.2115 → 0.6780/96/520=0.1846`。
- candidate逐seed outage Wilson 95%区间为`[0.2093,0.2829]`、`[0.1395,0.2039]`、`[0.1536,0.2202]`。seed0相对baseline恶化`+0.0769`，episode-level差值近似95%区间`[+0.0280,+0.1258]`；seed1改善`-0.1365`，区间`[-0.1876,-0.0855]`。相反方向都超出单纯520回合抽样噪声，说明真正的seed异质性。
- 跨seed reward从`0.8157±0.0847`降为`0.7535±0.1012`，平均下降7.6%，主要来自seed2的`-0.1890`；合并outage从`356/1560=0.2282`降为`311/1560=0.1994`，Wilson区间从`[0.2081,0.2497]`变为`[0.1803,0.2199]`。按episode池化的差值区间刚好为`[-0.05760,-0.00009]`，但三个配对seed差为`+0.0769/-0.1365/-0.0269`、sample std达`0.1067`，不能忽略seed聚类后宣称稳健显著。
- 训练曲线同样表现为安全—性能交换：三seed全程均值reward/outage由`0.6590/0.1973`变为`0.5731/0.1657`；末20%由`0.8083/0.2050`变为`0.6642/0.1933`。target-KL平均减少风险更新幅度，也同时减少有用reward更新。
- critic CDF absolute error的seed均值从`0.1029`降至`0.0633`；seed0甚至由`0.0615`降至`0.0080`，最终outage却从0.1673升到0.2442。故“critic校准变准”不是策略必然更安全的充分条件；固定KL阈值还会截断高lambda下本来必要的risk correction。
- 裁决：`.004`保留为安全优先消融/可选组件，但不作为超过QCPO_refs的主配置，不扫`.003/.005/.006`，也不直接续到1.5M。它通过聚合outage门，却没有满足预注册的reward不退化与跨seed一致性。单看seed1会误判为全面升级，单看seed0又会误判为全面失败；3 seeds×1M+520正是本问题所需的预算。
- 完整history和训练图：`_runs/wandb_export/dqc_pm3_vs_pm5_targetkl004_multiseed_1m_2026-07-16/`、`_runs/profiles/dqc_pm3_vs_pm5_targetkl004_multiseed_1m_2026-07-16/`；后者新增`eval_multiseed.csv`、`eval_multiseed_summary.json`与`eval_multiseed_comparison.png`（2100×960，可解码验证通过）。

### E64：C-X1慢target actor-query实现、回归与1M预注册（2026-07-16）

- 代码审计发现当前每个inner update的顺序是`update_critic(batch) → update_actor(batch) → soft_update_target()`；首个actor epoch因此用已经看过当前rollout标签一次的online cost critic生成并缓存risk advantage。同批标签可经online critic立刻反馈给同批actor，是simple replay正反馈之外更基础的in-batch leakage。
- 新增默认`online`的`cost_actor_query_mode`。显式`target`时，只有actor实际动作/行为策略baseline的cost CDF以及对应`constraint_rms`查询Polyak cost target；online critic仍负责QR训练、pre/post holdout、初始CDF、最终评估，经验PID完全不依赖critic。首个actor查询发生在本轮target同步之前，所以当前rollout标签不能先进入查询网络；随后8个PPO epoch继续复用同一个冻结risk weight。
- 新增`advantage/risk_query_target_online_abs_mean`和summary同名字段，使用完全相同的`s,a,b`测量target与online风险CDF差。online默认返回精确0且不额外forward；target才多做一次无梯度online诊断前向。profiler已纳入该指标。
- 默认兼容回归使用提交`1240549`与当前代码、CPU sync、相同seed、`2 iterations×2 envs×20=80`环境步：actor、reward/cost critic及两个target、obs normalizer共6个module，lambda tensor、runtime、metrics、config、metadata、8条eval和summary逐tensor/逐值exact；仅新增配置字段和工作树输出路径不同。
- 数值测试确认online helper返回online模块、target helper返回target模块、初始两者逐tensor相等；故意修改online末层后target查询保持不变，非法mode被ValueError拒绝。target完整80步smoke训练`8.6s`、exit0，summary正确为`target`，末次target-online gap为`1.05e-9`；由于两批均零cost且只有4个critic step，该极小值只证明链路分离，正式run必须观察成熟阶段gap。
- C-X1正式实验以P-M3失败seed1为压力基线，只改`cost_actor_query_mode=target`；`ppo_target_kl=0`、s0 aux=0、target tau=.05、interval=1、B20/C20/N32/T1/PID和1M预算全部不变。预计独占GPU训练约7～8分钟，140与520评估约5～6分钟，总计12～14分钟，持久化后台运行。
- 520晋级门沿用压力seed标准：outage `≤0.22`且相对0.3058至少下降0.08，reward `≥0.75`；同时要求成熟阶段target-online gap非零、末20%不形成比baseline更大的周期。通过才扩seed0/2；失败则不扫target tau，转真正的两折cross-fit/双critic ensemble。该路线与target-KL保持正交，不在首轮混合。


### E65：C-X1 seed1完整1M结果——早停到100k会误杀，520条门通过（2026-07-16）

- 持久化job `DQCAC_DynamicButton_cx1_targetquery_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`、W&B `3oogqkf5`正常完成；纯训练`393.9s`，每100k保存pre-update rollout checkpoint，最终另存post-update checkpoint。相对P-M3 seed1只把`cost_actor_query_mode=online→target`，target-KL与s0 aux均关闭。
- 这条曲线给出“短跑会误杀”的直接例子：40k/80k训练reward仍为`-0.040/-0.083`，200k才到`0.471`，260k/300k升到`0.634/0.807`，400k达到`0.933`；64–72万为`0.831–1.031`，82–98万多数为`0.781–1.135`。若100k按性能停掉，会错过后续学习。
- 但它不是单调慢收敛。400k附近outage升高后lambda到约`0.29`，44–48万reward回落到`0.504–0.667`；末个1M训练批又是reward/outage `0.747/0.35`。因此延长训练的价值是看清策略—critic—PID周期，不能把“多跑”当作自动稳定化手段，也不能用最后一个小批替代冻结策略评估。
- 对齐1M history后，baseline→C-X1的末20%训练均值为reward `0.8253→0.9002`、outage `0.180→0.210`、lambda `0.2206→0.1535`、KL `0.00246→0.00245`、clip `0.1293→0.1293`。成熟阶段target-online风险查询absolute gap末20%均值`0.0521`、末点`0.1024`，证明开关实际改变了actor监督而非数值上等同online。
- 内置140条final为reward/outage `1.058/31/140=0.2214`，online critic CDF为`0.3616`。它只作screen；同一个post-update final checkpoint的fresh eval-only 520条为reward `0.9661±0.5244`、outage `113/520=0.21731`、Q80 cost `16`、mean cost `10.0135`，Wilson 95%区间`[0.1840,0.2548]`。
- 同协议P-M3 seed1 baseline为reward `0.8622±0.5809`、outage `159/520=0.30577`。C-X1使reward增加`0.1039`（相对`+12.1%`，近似95%区间`[+0.0366,+0.1712]`），outage降低`0.08846`；保守Newcombe差值95%区间为`[-0.16267,-0.01296]`。它通过预注册的`≤0.22`、改善至少0.08、reward≥0.75三项门。
- critic仍未解决：C-X1的online smooth CDF `0.3608`相对truth `0.2173`高估`0.1435`；baseline则为`0.1074`相对`0.3058`低估`0.1984`。absolute error有所下降但偏差符号翻转，说明慢target查询改善了闭环反馈，不等于distributional critic已经校准。
- 按规则不在seed1上续1.5M或扫target tau，而是原样扩展seed0/2各1M+520。机器只有一张A100，改为串行持久化后台运行以避免GPU争用；独占训练实测约6.6分钟，含140 screen约8分钟，单个520复评约5分钟。seed0已启动，只有三seed方向和reward都可接受时才将C-X1列为主候选。
- 正式数据：`_runs/wandb_export/dqc_pm3_seed1_vs_cx1_targetquery_1m_2026-07-16/`、`_runs/profiles/dqc_pm3_seed1_vs_cx1_targetquery_1m_2026-07-16/`；520结果为`_runs/DQCAC_DynamicButton_cx1_targetquery_1m_s1_final_eval520_s1.json`。


### E66：C-X1三seed裁决——长跑避免早期误杀，但慢target把回报提升换成更高风险（2026-07-16）

- seed0/1/2的1M训练均正常exit 0，纯训练耗时`394.4/393.9/389.0s`，W&B run为`5t6sxi2j/3oogqkf5/yod7kvxy`。三条都是独占单张A100串行持久化后台作业，配置只在训练seed和输出路径上不同。
- 内置140条reward/outage为seed0 `1.142/49/140=0.350`、seed1 `1.058/31/140=0.2214`、seed2 `1.163/47/140=0.3357`。没有根据seed1通过而保留、也没有根据seed0/2失败而停止；三个post-update final checkpoint全部按同协议做fresh 520。
- 520条baseline→C-X1结果为：seed0 reward/outage `0.7180/87/520=0.1673 → 1.0455/171/520=0.3288`；seed1 `0.8622/159/520=0.3058 → 0.9661/113/520=0.2173`；seed2 `0.8670/110/520=0.2115 → 1.0767/162/520=0.3115`。
- C-X1在三个seed都提高reward，逐seed增量为`+0.3275/+0.1039/+0.2097`；跨seed均值从`0.8157±0.0847`升到`1.0294±0.0570`，平均约增加26.2%。这是该组件明确且可复现的正面效果。
- 安全性方向不稳定且总体恶化。逐seed outage变化为`+0.1615/-0.0885/+0.1000`，sample std `0.1303`；合并事件从`356/1560=0.2282`升到`446/1560=0.2859`，Wilson 95%区间从`[0.2081,0.2497]`变为`[0.2640,0.3088]`。episode池化差值为`+0.05769`、近似95%区间`[+0.02709,+0.08829]`，但最终解释仍以2/3 seed恶化和强seed异质性为主。
- online critic smooth-CDF absolute error逐seed从baseline `0.0587/0.1984/0.0434`变为C-X1 `0.1841/0.1435/0.1423`，跨seed均值`0.1002→0.1566`。它修复原压力seed1的低估，却让原先较准的seed0/2明显低估；这与outage方向完全一致。
- 三seed训练曲线反而容易给出false positive：末20% reward `0.8083→0.9031`，训练outage `0.2050→0.1983`，lambda `0.2459→0.1378`，看起来全面更好；fresh初始状态评估却在seed0/2失约。训练布局上的经验PID并没有暴露policy的initial-state泛化风险。
- 训练长度结论要同时保留两面：100k会因三个seed尚未形成高reward而误杀C-X1的性能作用；但完整`3 seeds×1M+520`又证明它不是安全主配置。继续1.5M或扫描target tau更可能改变闭环相位，而没有跨seed定向安全证据，因此停止该小网格。
- C-X1定级为“稳定提高reward、但平均放松约束”的机制消融，不与target-KL组合补丁式试错。下一条路线是两个独立cost critic的两折cross-fit：每个critic只在一半环境轨迹上训练，actor对该折状态查询未见该折标签的另一critic，提供比Polyak一步滞后更严格的样本隔离；先做默认兼容回归和冻结机制门，再决定是否进入live 1M。
- 正式history、profile和训练图在`_runs/wandb_export/dqc_pm3_vs_cx1_targetquery_multiseed_1m_2026-07-16/`及`_runs/profiles/dqc_pm3_vs_cx1_targetquery_multiseed_1m_2026-07-16/`。后者含`eval_multiseed.csv`、`eval_multiseed_summary.json`与`eval_multiseed_comparison.png`（2084×755，PNG解码验证通过）。

### E67：C-X2两折cross-fit实现、默认回归与冻结300k预注册（2026-07-16）

- 新增默认关闭的`cost_actor_query_mode=crossfit`。按环境编号把完整轨迹固定拆成两折；主cost critic只接收fold-0标签，独立peer只接收fold-1标签。actor对fold-0样本查询peer、对fold-1查询主critic，所以当前样本的真实cost不能先训练查询它的critic再回灌同批actor。首版刻意限制为偶数`num_envs>=2`、`MC + raw history + full-batch + s0_aux=0`，避免一次实验混入n-step、history encoder、recent replay或chunk更新语义。
- 新评估状态没有fold身份，最终CDF取两critic各自CDF的等权平均，不能先平均quantiles再计数；预测方差按两分布等权mixture计算。同时记录primary CDF、peer CDF、同输入absolute disagreement、两折QR loss和样本比例。checkpoint自动包含peer，eval-only严格恢复；profiler增加crossfit disagreement字段并对旧1M history回归通过。
- 默认兼容回归以提交`5630741`为金样本，相同seed、CPU sync、`2×2×20=80`环境步。obs normalizer、actor、reward/cost critic及两个target共6个module逐tensor exact；lambda、20个runtime字段、41个训练summary字段和非时间/路径eval JSON逐值exact。module hash为obs `86520917b3438940`、actor `645e073e484d5c19`、reward `92f6207845224661`、reward-target `2520ac1ff50d334e`、cost `8e14fb7efcc15e30`、cost-target `5df9d8aa18b2a45c`。
- 机制断言确认fold-0→peer、fold-1→primary路由逐tensor正确；两个critic都被optimizer更新，transition fraction严格`0.5/0.5`，末次fold loss为`0.41993/0.54391`，查询分歧为有限非零`2.62e-8`。完整入口smoke训练`8.7s`并完成checkpoint、双critic eval和JSON；独立eval-only严格加载peer后exit 0。临时脚本、checkpoint和profile只在验证期保留，记录完成后清理。
- 测试还暴露后台launcher复用同名stale job时会暂时保留旧`exit_code`，使轮询器提前误判。本轮在新启动前仅删除该job旧的`worker_started_at/finished_at/exit_code`，不触碰日志与实验产物；`bash -n`和新job完整退出均通过。
- 冻结机制门使用C-S0B完全相同的P-M3 seed1成熟actor、独立rollout seed101和`15×B20×T1000=300k`数据；唯一算法变量是`online→crossfit`。模块构造和policy-only恢复后重置RNG，故真实轨迹应与既有baseline逐批一致。既有520 baseline truth/CDF/mean为outage `146/520=0.28077`、CDF `0.19964`、mean cost truth/pred `11.0173/9.4186`。
- crossfit每个critic只看约150条轨迹，检验的是out-of-fold泛化而非增加标签量。进入live 1M的门为：520条truth与baseline配对一致；ensemble CDF或mean-cost error至少改善25%，另一项不得恶化；primary/peer disagreement不持续扩大，最后五批prequential OOF误差无发散。若不过门，直接停止C-X2，不用短live策略曲线作结论；若通过，再在P-M3 seed1完整跑1M+520，之后仍需seed0/2复现。预计纯训练约2～3分钟、128内置评估约1分钟、520复评约4分钟，总墙钟约7～9分钟，全部持久化后台运行。

### E68：C-X2冻结300k结果——隔离成立，但半数据critic方差使其不过live门（2026-07-16）

- 正式job `DQCAC_DynamicButton_frozen_s1policy_seed101_cx2_crossfit_300k`、W&B `mnwyg6io`正常exit 0；300k训练`125.3s`，与单critic baseline `121.0s`接近，因为两个模型各处理半批、总QR样本量不翻倍。15批reward、cost、outage与baseline逐值exact，总超限事件同为`89/300`；因此差异只来自critic路由与数据拆分。
- 全15批/末5批prequential OOF CDF error为crossfit `0.17104/0.14656`，baseline `0.16917/0.14344`，crossfit分别略差`1.1%/2.2%`；末5批Brier `0.23812 vs 0.24462`只改善`2.7%`。post同批CDF/Brier也从baseline `0.11844/0.20034`恶化到`0.13188/0.23072`。没有“末段仍快速改善、应因短跑继续”的证据。
- 140条内置评估truth exact为outage `0.26429`、mean cost `10.79286`。baseline→crossfit ensemble CDF为`0.19799→0.20324`，absolute error只改善约`7.9%`；pred mean为`9.40770→9.39550`，反而略差。primary/peer CDF为`0.11228/0.29420`，平均绝对分歧`0.19442`，表明ensemble平均掩盖了很大的模型方差。
- 同协议fresh 520 truth逐值exact：reward `0.8349376`、outage `146/520=0.2807692`、mean cost `11.0173077`。baseline→crossfit CDF error `0.0811298→0.0761118`，仅改善`6.19%`；mean-cost error `1.5986857→1.4516167`，仅改善`9.20%`，两项都远低于预注册25%门。primary/peer为`0.120072/0.289243`，误差`0.160697/0.008474`，分歧仍达`0.186358`。
- 裁决：不启动C-X2 live 1M，也不把300k直接续成600k来移动门槛。300k固定策略已有89个tail事件、两项独立520误差和无改善末段趋势，足以回答局部机制；它不证明任何live策略在100k后永远不会变好，而是证明当前“两个模型各看一半数据”的隔离收益不足以抵消数据碎片方差。
- 路线保留为消融：① 600k使每个critic获得约300条轨迹，可单独回答样本量问题，但环境预算翻倍，不是主算法公平提升；② 标准K-fold complement让每个holdout模型看`(K-1)/K`数据，K=5约80%，代价约4倍cost-critic计算；③ 两模型max/UCB在本520上恰为peer `0.28924`、非常接近truth，但这是同一评估后观察，不能据此调参，而且actor若使用包含自身标签的in-fold模型会重新泄漏。
- 下一优先路线C-X3是不分裂数据的pre-update online cache：在本批任何QR step前，用已吸收全部历史批次的online critic计算并冻结risk advantage，再执行critic和8个PPO epoch。它提供当前样本隔离、保留全部历史数据，又避免C-X1的Polyak长期滞后；先做默认exact回归与顺序/缓存断言，再给seed1完整1M而非100k裁决。
- 正式history/profile/图在`_runs/wandb_export/dqc_frozen_baseline_vs_cx2_crossfit_300k_2026-07-16/`和`_runs/profiles/dqc_frozen_baseline_vs_cx2_crossfit_300k_2026-07-16/`；后者含`eval520_comparison.csv/json/png`，PNG为`1961×801`且PIL解码通过。

### E69：C-X3 pre-update online cache实现、回归与1M长跑预注册（2026-07-16）

- 新增默认关闭的`cost_actor_query_mode=preupdate`。每个rollout仍先用真实完整轨迹更新经验PID和constraint RMS；但第一个actor epoch改在任何本批cost-critic QR step之前，用已看过所有历史批次、尚未看当前标签的online critic生成`_risk_cdf/_risk_advantage/_risk_weight`。后续PPO epoch在critic更新后仍复用这份缓存，不再查询。
- 它与C-X1的区别是没有Polyak长滞后，与C-X2的区别是不拆分训练数据。首版显式限制为已验证的recurrent GAE-PPO、MC/raw cost target、full-batch且`s0_aux=0`；这避免把MLP n-step内部bootstrap-action RNG顺序变化混入消融。actor与critic参数完全独立，在这些限制下两个optimizer step可交换，唯一有意变量是risk-query的critic版本。
- 新增`advantage/risk_query_preupdate_postupdate_abs_mean`，在相同`(s,a,budget)`上比较缓存的QR更新前CDF与全部本批QR更新后CDF。该值与原`target_online` gap分开保存，避免把时间漂移误标为两网络差异；W&B、profile与最终JSON均已接入。
- 默认`online`精确回归使用改动前提交`baa786f`的临时worktree与当前代码：80环境步后checkpoint内44个tensor leaves全部逐元素exact（其中43个属于6个module/optimizer状态），评估与summary一致；非数值差异只有checkpoint和W&B路径。这证明重构没有暗中改变历史基线。
- 机制断言通过：两个inner updates的调用顺序精确为`A,C,C,A`；第一次actor调用时cost critic全参数仍与初始值exact，该epoch做`1+K=5`次risk query，第二个epoch新增查询为0且缓存逐元素exact；两次QR step后6个cost参数张量全部改变，pre/post CDF drift为`1.16e-8`。统一入口80步smoke也完成checkpoint、JSON和4轨迹评估、exit 0；初始低风险批次的drift极小只证明链路，不代表成熟阶段机制弱。
- 正式C-X3只在P-M3 seed1上把`online→preupdate`，保持B20/C20/N32、MC/time-weight .995、T1 sigmoid、PID target .15、8个PPO epochs、LR和1M预算全部不变。不用100k早停；C-X1已证明这类时序组件在100k reward仍可为负而后期超过基线。预计A100纯训练`6.5–8分钟`，含140与fresh 520评估后总墙钟`11–14分钟`，全程持久化后台。
- 预注册门与压力seed一致：相对P-M3 seed1的fresh520 `reward/outage=0.8622/0.3058`，候选需outage `≤0.22`且至少下降`.08`，reward `≥0.75`；同时成熟阶段pre/post drift必须非零，末200k不得出现更大的policy–critic–PID周期。通过后原参数扩seed0/2各1M+520；失败则保留为时序消融，不扫更多缓存间隔或叠加target-KL追逐偶然点。

### E70：C-X3 seed1完整1M结果——校准与安全改善，但reward代价使其不进多seed（2026-07-16）

- job `DQCAC_DynamicButton_cx3_preupdate_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`、W&B `xcv8ghw6`正常exit 0；纯训练`390.9s`，与P-M3的约6.5分钟一致，没有增加可测wall-time或显存代价。fresh520 eval-only job也exit 0。
- 完整长跑再次证明不能以中期终点替代结论。0–300k的P-M3/C-X3 reward为`0.2571/0.3015`、outage同为`0.1033`；300–600k为`reward 0.7835/0.7775,outage 0.2433/0.2233`，候选看似略安全且reward持平。但600–800k已变为`0.8233/0.7556,0.260/0.265`，末200k更恶化为`0.8253/0.6130,0.180/0.255`，lambda从`0.2206→0.3203`。若600k停掉会得到false positive。
- pre/post query drift在全程的mean/std/max为`0.09684/0.05287/0.22251`，末200k均值`0.10407`，最终训练JSON为`0.12024`。它与同批outage、lambda、prequential s0 CDF error的Pearson相关分别为`0.626/0.559/0.669`；这是描述性共振，不解读为单向因果，但证明当前批QR更新会大幅移动actor风险信号。
- 内置140条为reward/outage `0.57035/27÷140=0.19286`，hard CDF `0.24487`。严格fresh520中，P-M3→C-X3的reward为`0.86220±0.58095→0.58645±0.61862`，差`-0.27575`（`-32.0%`），近似95%区间`[-0.34869,-0.20281]`；outage为`159/520=0.30577→108/520=0.20769`，差`-0.09808`（`-32.1%`），保守Newcombe 95%区间`[-0.17164,-0.02307]`。两个方向都超出520回合抽样噪声。
- 校准改善是实质性的。hard CDF absolute error从`|0.10427-0.30577|=0.20150`降至`|0.26929-0.20769|=0.06160`，改善`69.4%`；smooth CDF error `0.19837→0.06670`，改善`66.4%`；predicted mean-cost error `5.74198→1.00415`，改善`82.5%`。真实cost mean/Q80也从`12.3308/18`降到`10.1865/15`。
- 预注册裁决：outage `≤0.22`且降低至少`.08`两项通过，但reward `0.58645<0.75`失败，末200k周期也比baseline大。因此C-X3保留为“切断同批反馈能换来更准critic和更安全policy，但会明显压低reward”的机制消融；不扩seed0/2、不扫缓存间隔、不叠加target-KL。
- 正式数据和图位于`_runs/wandb_export/dqc_pm3_seed1_vs_cx3_preupdate_1m_2026-07-16/`和`_runs/profiles/dqc_pm3_seed1_vs_cx3_preupdate_1m_2026-07-16/`；后者含`phase_comparison.csv`、`eval520_comparison.csv`、`eval520_summary.json`、`overview.png`及`eval520_comparison.png`（2700×1350，PIL解码通过）。


### E71：C-IQN1 cost-only uniform IQN 实现、回归与冻结策略校准预注册（2026-07-16）

- 完成度复核：QCPO 的未修正 rollout reuse 与未更新 observation RMS 两个确定性 bug 已修；Q-B 在300k终评 reward=1.621，已回到 QCPO_refs 的正常量级。QCPO 和 DQCAC 的 [512,512]+LSTM512 都完成全尺寸100k/300k验证；DQCAC recurrent 300k终评1.442，高于同结构 QCPO 的1.251。当前主缺口已从网络接线转为 cost distribution 泛化校准和 policy-critic-PID闭环。
- 新增 cost_distribution_model=qr|iqn，默认 qr 保持旧结果。IQN 只替换 action-conditioned cost critic；actor、共享 recurrent reward-V、reward critic、PPO、PID 和 observation normalization 均不变。网络为 (s,a) MLP feature × cosine(πiτ) embedding → scalar quantile head；训练每个 transition 随机采32个 uniform τ，CDF/mean/std 查询使用128个确定性 midpoint τ。
- 概念边界：IQN不是“直接输出CDF”。它估计连续 quantile function Q(s,a,τ)；P(C≥b) 仍由 uniform τ 上的 indicator 或 sigmoid 积分得到。当前先做 uniform-IQN；只有它能提高独立泛化校准，才考虑 query-mixture IQN 或查询附近自适应采样，避免再做没有作用链证据的小网格。
- IQN τ 使用独立 torch.Generator，seed=training_seed+104729。测试证明一次 [5,7] τ 采样前后全局 torch RNG state 完全相同，重置专用 generator 后 τ 逐位复现；所以 critic 采样不会偷偷改变策略动作噪声。
- 数学/工程验证：
  1. IQN default query [6,17] 与显式传入同一 midpoint τ 逐位相同；随机 [6,11] τ 输出具有非零 τ 方向标准差0.11086，所有参数梯度有限。
  2. per-transition 2D τ quantile-Huber loss 与手算误差5.96e-8（float32求和次序），反向梯度有限。
  3. 同一模型/τ/target/time-weight 下，M=11、chunk=4 的 full/chunk loss 误差1.73e-7，最大参数梯度误差2.98e-8。
  4. 对旧提交0c58963做配对短训练：34个 module tensor leaf、lambda tensor和全部runtime逐位相同；summary差异仅为新增IQN元数据。默认QR训练语义未改变。
  5. CPU端到端 IQN smoke（Ntrain=8,Nquery=17,cos=8,chunk=17）训练8.2s、exit0；checkpoint eval-only恢复exit0，随机评估逐项复现。
  6. P-M3 LSTM512 policy-only + CUDA IQN smoke训练6.6s、exit0，覆盖recurrent rollout、CUDA专用τ generator、chunk update、checkpoint和recurrent评估。
- 第一次CUDA smoke不计为算法结果：为了省时把horizon从1000缩至32，却遗漏源配置n_step=100，rollout后处理按空切片报维度错。已增加 n_step<=horizon 的构造期显式校验，并用n_step=8原样重跑通过；该问题与IQN数值无关。
- 评估协议同时修正：任何agent训练/构造完成后、独立评估开始前，都用 seed+777 重置host/CUDA动作RNG；环境worker本来就用seed+777。不同critic参数量不再通过“初始化消耗了多少随机数”污染随机策略评估。通用评估器也改为调用 cost critic 自身的加权CDF/分布矩接口，修正非均匀QR网格过去被简单mean误报的问题。
- 新增 monotonicity 诊断 cost_quantile_crossing_fraction：在升序查询τ上统计 Q(τ[j+1])<Q(τ[j]) 的比例。IQN不强制单调，这个指标用于区分“更密查询”与“高频crossing造成的伪分辨率”。

#### C-IQN1 冻结成熟策略校准门

- 数据源固定为 P-M3 seed1 的1M final actor，原fresh520 reward/outage=0.8622/0.3058；校准rollout seed=101。QR和IQN都只恢复actor与两套observation RMS，critic/target/optimizer从头初始化，policy、dual和RMS完全冻结。
- 两条均使用 B20、T1000、15 iterations=300k、20 critic updates/rollout、MC cost target、risk-discount=.995、critic [256,256]、lr=1e-3、Ntarget=32和chunk=2500。两条都设 advantage_norm=separate，使冻结实验不执行无用的K动作risk-baseline查询；因此每轮behavior数据只由同一个成熟策略和相同action RNG生成。
- 唯一算法差异：baseline为固定 QR-N32；candidate为 uniform-IQN，Ntrain=32、Nquery=128、cosines=64。内置140条只作screen；训练后若方向合格再从两个final checkpoint做同seed、同动作RNG的fresh520配对评估。
- 预注册晋级门：
  1. QR/IQN每批真实reward/cost/outage必须逐点相同，否则先判定配对协议失败；
  2. 最后5批prequential CDF absolute error或Brier至少改善20%，另一项不得恶化超过10%；
  3. 独立评估的hard-CDF absolute error或mean-cost absolute error至少改善25%，另一项不得恶化超过10%；
  4. crossing不能比QR恶化超过0.10，且末段无发散。
- 只有140 screen和fresh520都通过，才在P-M3压力seed1进入一次完整1M live闭环；否则停止standard uniform-IQN，不扫cosine数、Ntrain或Nquery碰运气。query-mixture IQN作为有分歧的独立路线保留，不和uniform-IQN同时修改。
- 既有QR冻结300k纯训练121s。新配对QR预计约2–3min训练、加140评估总约4min；IQN因τ embedding和chunk前向预计训练4–7min、总约6–10min。两条串行总墙钟约10–14min，全程由launch_background.sh持久化，输出只写/vepfs项目盘。

### E72：C-IQN1 300k 初筛与训练长度审计预注册（2026-07-17）

- QR/IQN 正式 300k job 均正常 exit 0，纯训练分别为 `121.88s/125.73s`，W&B run 为 `qrwjysnw/20f7gtvr`。15 批真实 reward、outage、cost、holdout truth 逐点 exact，140 条终评的 reward/outage/mean cost 也 exact 为 `0.86576/0.25714/11.75714`；因此差异没有混入策略或环境随机性。
- 300k 独立终评中，QR→IQN 的 hard-CDF absolute error 为 `0.14442→0.14777`，mean-cost absolute error 为 `3.15970→3.54291`，crossing 为 `0.04862→0.20079`。三项均未通过预注册门，其中 mean error 恶化 `12.1%`，crossing 绝对增加 `0.15217`。
- 最后 5 批 prequential CDF absolute error 为 `0.10844→0.11586`，IQN 恶化 `6.8%`；Brier 为 `0.21388→0.21350`，只改善 `0.17%`，远低于 20% 门。post-CDF error 为 `0.04438→0.06117`，IQN 仍差，但逐批差值从 220k 的约 `+0.0250` 缩至 300k 的 `+0.00391`；这留下了“复杂 IQN 只是收敛更慢”的有限可能。
- 本次不改写 300k 预注册裁决：uniform-IQN 已失败，不能凭延长后偶然好转直接晋级 live 1M。考虑用户提出的短训练 false-negative 风险，以及 post-CDF 相对差距确有收窄，只增加一次独立的**训练长度审计**：同 seed101 从头严格配对重跑 QR/IQN 到 600k，前 300k 应复现实验，后 300k 只回答慢收敛问题；不扫描 Ntrain、Nquery、cosines 或学习率。
- 600k 审计只有在后 5 批 prequential 指标和独立 140 条评估都达到原门、且 crossing gap 回到 `≤0.10` 时，才允许 fresh520；否则停止 standard uniform-IQN。若 600k 只是两者近似持平，它最多说明 IQN 没有明显坏处，不构成增加复杂度或进入 live 闭环的理由。
- 按 300k 实测线性外推，两条各约 `252s` 纯训练；两条并行、加 140 条独立评估预计总墙钟 `5～7min`。仍由 `launch_background.sh` 持久化启动，checkpoint、W&B 与结果文件全部写入 `/vepfs-mlp2/c20250510/251204033/` 下，不占 20G 根盘。

### E73：C-IQN1 600k 长度审计——300k 对 critic 偏短，但 uniform-IQN 仍未过门（2026-07-17）

- QR/IQN 600k job 均正常 exit 0，W&B run 为 `8ry7xn6g/7llo7019`；并行纯训练 wall time 为 `417.21s/433.87s`。两条前 300k 的 reward、outage、cost、pre/post CDF/Brier/mean bias 与 crossing 对原 300k history 全部逐值 exact，证明长度审计没有更换初始化或移动原裁决。
- 600 条校准轨迹中两条真实行为逐批 exact，共有 `167/600` 个超阈值事件。独立 140 条评估也严格配对，truth reward/outage/mean cost 均为 `0.86576/0.25714/11.75714`。
- 训练长度本身影响巨大：QR 的 CDF/mean-cost absolute error 从 300k 的 `0.14442/3.15970` 降至 600k 的 `0.01071/0.77573`，分别改善 `92.6%/75.4%`；IQN 从 `0.14777/3.54291` 降至 `0.00898/0.62543`，分别改善 `93.9%/82.3%`。因此 300k 足以做初筛，却不足以把冻结 distributional critic 的绝对校准误差当作收敛值。
- 同预算比较中，600k IQN 的独立 CDF error 比 QR 小约 `16.1%`，mean-cost error 小约 `19.4%`，出现小到中等优势；但均低于原定 `25%` 门。最后 5 批 prequential CDF/Brier 只改善约 `3.0%/0.9%`，post-CDF 约改善 `5.0%`，post-Brier与mean error略差，没有持续扩大优势。
- IQN 的 monotonicity 仍明显更差：600k 独立 crossing 为 `0.23397`，QR 为 `0.05737`，差 `+0.17660`；最后 5 批差仍为 `+0.15041`。它不是多训练即可消失的早期毛刺。
- 按预注册规则，不做 fresh520，不启动 uniform-IQN live 1M，也不扫 `Ntrain/Nquery/cosines`。定级为：`IQN 在足量固定策略数据下可能带来约 16%～19% 的校准收益，但当前非单调输出和收益幅度不足以证明值得进入非平稳 actor–PID 闭环`。
- 最重要的算法含义不只是 IQN 输赢：当前 live B20 每 20 条轨迹就改变一次 policy，而固定 policy 的 critic 需要数百条轨迹才接近校准；更多同批 gradient steps不能制造新的独立 tail trajectory。下一优先验证应增加每个 policy 版本的独立轨迹数或减慢 actor 更新频率，首选固定 1M 总步的 `B20→B40`，而不是继续增加同一批 critic epoch。
- 正式 history/图/表位于 `_runs/wandb_export/dqc_frozen_qr32_vs_iqn32q128_600k_lenaudit_2026-07-17/` 与 `_runs/profiles/dqc_frozen_qr32_vs_iqn32q128_600k_lenaudit_2026-07-17/`；端点图 `length_audit_endpoint_comparison.png` 为 `2356×748`，PIL 解码验证通过。

### E74：P-M6 B40 长预算预注册——增加每个 policy 版本的独立轨迹（2026-07-17）

- 以 P-M3 seed1 为压力基线，只把 `num_envs=20→40`、`num_iterations=50→25`，总环境步与总完整轨迹仍严格为 `1M/1000`。网络、QR-N32、20次critic update、8次PPO epoch、actor LR、PID、T1 smooth CDF、cost time weighting与所有seed均不变。
- B40每次policy更新有40条独立初始状态轨迹，outage=.2时预期tail事件由4增至8；policy/dual更新次数由50降至25。每次batch翻倍但update次数减半，固定1M下actor与critic看到的总transition-pass不增加，检验的是更低batch方差、更慢policy漂移和更多同版本tail样本。
- `pid_reference_episodes=10`保持不变。代码按episode_scale和几何leak缩放，因此常值error下一次B40积分更新严格等价于两次B20，不会因batch变大暗改每episode的I增益；`pid_window_episodes=50`仍是同样的50条轨迹窗口。
- 不用300k早停。P-M1/C-X1已证明慢变量可在100k～600k改变方向，P-M6除NaN/OOM/确定性错误外完整跑1M，并每5个iteration（200k）保存phase checkpoint。内置160条只作screen；通过才做统一`num_envs=20`的fresh520。
- 压力seed1 fresh520门：相对P-M3的`reward/outage=0.8622/0.3058`，候选需outage `≤0.22`且至少下降`.08`，reward `≥0.75`；同时末200k不出现更大lambda/outage周期。通过才原配置扩seed0/2，否则停止B40，不扫B30/B50/B60。
- A100 80GB对B20只占少量显存，B40+N32预计安全；单条独占训练预计`7～9min`，内置160约1min，若晋级fresh520约5min。全程由`launch_background.sh`持久化，输出写入`/vepfs`。

### E75：P-M6 内置screen通过与fresh520协议校正（2026-07-17，评估启动前）

- P-M6 seed1 1M训练和160条内置评估正常exit 0；内置reward/outage为`0.93045/34÷160=0.2125`，通过screen。critic CDF为`0.08320`，相对truth仍低估`0.12930`，所以必须做fresh520。
- E74写成统一`num_envs=20`是启动前发现的记录错误。既有P-M3 seed1正式fresh520命令实际为`num_eval=512,num_envs=40`，得到严格520条；为复用完全相同的并行布局与动作RNG协议，P-M6也使用`512/40`。这是查阅既有run.sh后的协议校正，不改变checkpoint、门槛或训练结果。
- fresh520仍只做eval-only，门保持outage`≤0.22`且相对P-M3降低至少`.08`、reward`≥0.75`；不根据内置160调阈值。

### E76：P-M6 B40 结果——方向有益但未过安全门，160条产生false positive（2026-07-17）

- P-M6 1M训练`398.24s`、W&B `ksw0hwvv`、训练与内置评估均exit 0；fresh520 eval-only也exit 0。B40+N32实测显存约3.5GB，无OOM/NaN，证明当前128 CPU/A100 80GB可承载。
- B20→B40的分段reward/outage为：0–300k `0.2571/0.1033→0.0237/0.0179`，300–600k `0.7835/0.2433→0.4718/0.1406`，600–800k `0.8233/0.2600→0.6534/0.1800`，800k–1M `0.8253/0.1800→0.8034/0.1450`。B40明显减慢早期reward学习，但后段追上并降低训练outage。
- 末200k lambda/KL/clip从P-M3的`0.22057/0.002459/0.12929`降到`0.03244/0.001222/0.05425`。更大batch与更少policy update确实减小闭环振幅和PPO位移，不是无作用改动。
- 内置160条给出reward/outage=`0.93045/34÷160=0.2125`，看似通过；同一final checkpoint的正式fresh520为`0.92628±0.41427`、`137/520=0.26346`。因此160条安全结论是false positive，不能据此扩seed。
- 同协议P-M3 seed1为`0.86220±0.58095`、`159/520=0.30577`。B40 reward增加`0.06408`（约7.4%，非配对近似95%区间`[0.00276,0.12541]`）；outage下降`0.04231`，保守差值95%区间`[-0.11924,0.03525]`仍跨0。B40的Wilson区间为`[0.22743,0.30296]`，下界已高于0.22门。
- critic仍系统低估：hard-CDF absolute error只从`0.20150`降到`0.17710`（改善12.1%），mean-cost error从`5.74198`降到`4.84288`（改善15.7%）。增加每次policy版本的轨迹数缓解但没有解决initial-state泛化。
- 预注册裁决：reward门通过，但outage既未到`≤0.22`、也未下降`.08`；不扩seed0/2，不扫B30/B50/B60。B40定级为“降低更新方差、略提高reward并略降风险，但不足以成为主配置”。
- 下一路线不再单纯增大B或重复同批QR epoch。优先实现直接查询点exceedance/CDF critic：用MC remaining-cost与remaining-budget的二元标签直接拟合`P(C_remaining≥budget|s,a)`，先在固定policy 600k配对校准；只有跨状态CDF泛化显著优于QR，才进入live闭环。
- 正式history/profile在`_runs/wandb_export/dqc_pm3_b20_vs_pm6_b40_seed1_1m_2026-07-17/`与`_runs/profiles/dqc_pm3_b20_vs_pm6_b40_seed1_1m_2026-07-17/`；后者含phase/eval CSV、置信区间JSON和`1872×1277`的fresh520比较图。


### E77：训练长度重新分级——短跑只筛机制，不能裁决最终性能（2026-07-17）

- 用户质疑“当前训练是否普遍太短”。结论是：**对工程正确性和明显机制失效，现有短跑够用；对网络结构、慢critic、PID闭环和最终算法性能，100k/300k明显不够，1M也只是压力筛查而非论文级终局**。QCPO_refs正式配置为`runner.n_steps=5e6`；因此目前任何1M结果都不能支持“DQCAC最终优于/劣于QCPO_refs”的论文结论。
- 以当前`B20,T1000`计，100k/300k/600k/1M分别只有5/15/30/50个policy版本；B40的1M更只有25次policy/PID更新。LSTM、IQN、更大batch或更慢actor更新在相同env steps下天然启动更慢，不能只看300k reward点估计判死。
- 已经观察到双向反例。冻结critic的IQN在300k比QR更差，到600k反而在独立CDF/mean error上好约16.1%/19.4%；B40在0--300k reward仅0.0237，到末200k追到0.8034且fresh520 reward超过B20。反方向上，C-X3在300--600k reward几乎持平且更安全，到1M末200k reward却降至0.6130；600k停掉会产生false positive。故“前半段不好就停”和“中段好就晋级”都不可靠。
- 既有结论重新定级：确定性代码bug、NaN/OOM、错误梯度/错误标签等仍可短跑裁决；C-X3的1M大幅reward代价和IQN持续crossing足以阻止当前版本晋级，但只表示“当前预算/当前实现不过门”，不等于它们在无限训练下绝无改善。`MLP vs LSTM`的300k、`N32 vs N64/局部tau`的100k--300k、早期PID组合和B40的渐近性能均改标为“早期样本效率证据”，不能作为最终排序。
- 新分层协议：smoke只验证全链路；300k只看机制方向和是否值得投入，不以reward单独判败；冻结policy critic至少600k，并比较最后5批与此前5批，若关键误差仍改善超过10%则预注册延到1.2M；live actor--critic--PID候选至少完整1M、每200k看phase、最终fresh520，除数值/确定性机制错误外不早停；1M仍趋势上升或结果接近门槛者扩到2M。最终入论文的DQCAC/QCPO/QCPO_refs按相同5M预算、至少3 seeds和统一独立评估比较。
- 随机初始化处理：单seed中小差异不再定论。共享actor的critic消融继续使用common-random-number配对；差异小于约20%或置信区间跨0时至少补3个初始化/seed。只有效应巨大、方向在多个phase一致且直接机制指标也恶化，才允许单压力seed停止。
- 复验优先级不是把所有旧组合全部5M重跑。先做直接query-point CDF head的冻结600k--1.2M门；通过后做live 1M。与此同时，最终网络公平性需要把DQCAC的MLP/LSTM最佳配置各跑完整1M；N64/局部tau只有在QR仍是主CDF表示时再做严格配对600k。B40保留为可与有效CDF head组合的减振组件，不因单独1M不过安全门永久删除，也不立即盲扫B30/B50/B60。
- 本条记录时没有活动训练进程；下一条正式实验仍只通过`launch_background.sh`持久化启动，预计时长在启动前记录，输出继续写入`/vepfs-mlp2/c20250510/251204033/`。


### E78：C-DCF1直接超阈概率critic实现、回归验证与600k预注册（2026-07-17）

- C-DCF1已在提交`4b70914`实现，新增`cost_cdf_estimator=quantile|direct`，默认`quantile`。direct不是另一种quantile网络，而是额外的action-conditioned Bernoulli head，输入`(cost_input, action, remaining_budget/cost_limit)`，直接输出`P(C_remaining>=budget|s,a,budget)`；现有QR-N32继续完整训练并报告mean/std/quantile/crossing，作为同run内部对照。
- 监督来自`label_t=1{mc_cost_to_go[t]>=budget[t]}`。由于`budget[t+1]=(budget[t]-cost[t])/gamma_c`，该事件与初始整轨迹超限严格等价；网络仍使用全部transition，因为条件`(s_t,a_t,b_t,t)`随时间改变，正是actor逐步查询的区域。新增`cost_direct_cdf_label_inconsistency_fraction`逐批检查同轨迹标签，必须恒为0。
- head为与cost QR相同宽度的ReLU MLP，最终输出单logit；budget固定除以15，BCEWithLogits保证极端概率数值稳定。不使用正负类重加权，因为class-weighted BCE会改变概率校准目标。它复用QR的risk-discount transition weight和chunk边界，但使用独立Adam、独立grad clip、独立一次optimizer step，不进入reward/QR joint norm或clip。
- direct首次只允许`QR + uniform grid + MC + raw cost history + online query + s0_aux=0`。这是归因约束，不是永久API限制：暂不与IQN、query-mixture、crossfit、target/preupdate或cost-LSTM叠加。MLP与`[512,512]+LSTM512` policy都已接通；direct会替换actor实际动作CDF、K动作baseline、constraint RMS和critic-dual查询，经验PID仍按真实轨迹工作。
- 开启额外网络前后保存/恢复host与CUDA RNG。构造测试表明quantile/direct的actor、reward/cost online/target及normalizer共34个tensor leaves逐位相同，joint critic参数组数量相同，构造后torch RNG逐位相同；direct只有78,593个参数，不会因初始化消费改变首批动作。
- 数学/工程验证：网络支持标量、`[B]`和`[B,1]` budget，BCE梯度全部有限；M=15、risk-weighted的full/chunk=4 loss为`0.6932367086/0.6932367027`，一次Adam后最大参数差`1.75e-10`，标签不一致率0。对旧提交`63b338d`做持久化配对训练，final checkpoint共34个module tensor leaves、lambda/runtime、全部旧eval键和47个共享summary键逐项exact；新增内容仅是direct配置/QR对照元数据与Brier。
- 端到端持久化smoke均exit 0：MLP CPU训练9.1秒；全尺寸`[512,512]+LSTM512` CUDA训练9.9秒，覆盖GAE-PPO两epoch的固定risk cache、observation normalization、chunk critic、checkpoint与循环评估。direct checkpoint保存7个state tensor，eval-only恢复后的旧eval字段逐项exact。短T=32只有0-cost样本，direct仍约0.42只是4次更新后的未收敛先验，不作为算法结果。
- 终评新增逐状态proper score：`cost_cdf_brier_initial`比较selected estimator与每条真实outage标签；direct时同时报告`cost_cdf_qr_initial/qr_brier_initial`。这避免“两个模型总体CDF均值相同，但逐状态排序完全错误”被均值误差掩盖。prequential pre/post也在相同真实a0上同时记录direct主键与`qr_*`内部对照。
- 正式C-DCF1复用P-M3 seed1的1M成熟策略和rollout seed101。既有QR600k run `8ry7xn6g`、checkpoint `dqc_frozen_s1policy_seed101_ciqn_qr32_600k_lenaudit_20260717`作为外部基线；direct 600k仍在同一run同步训练完全相同的QR，所以可以额外验证30批behavior truth及QR权重/指标是否保持exact，不重复浪费一条QR训练。
- 600k门：真实reward/cost/outage的30批history必须与既有QR逐值一致，所有标签不一致率为0且无非有限值；最后5批prequential direct的CDF absolute error或Brier相对同run QR至少改善20%，另一项不得恶化超过10%；独立140条的CDF absolute error或Brier至少改善25%，另一项不得恶化超过10%。140通过后才从同一final checkpoint做fresh520；520也通过才允许P-M3压力seed进入live 1M。
- 若600k未过效果门但direct关键误差从前5批到末5批仍改善超过10%、且相对QR方向一致，则按E77训练长度规则从头预注册1.2M审计；若只近似持平或后段已平台，则停止当前direct，不扫学习率/隐藏层碰运气。通过live后再决定是否与B40减振组合，不能在第一条run同时打开两个变量。
- 既有QR600k纯训练417.2秒。direct多一个仅78.6k参数的scalar head，结合smoke开销预计纯训练8--10分钟、含140终评总墙钟9--12分钟，显存增量远小于QR pairwise loss；正式实验只通过`launch_background.sh`，checkpoint/W&B/log全部写入`/vepfs`。

### E79：C-DCF1 600k结果——不是训练太短，而是head追逐上一批噪声（2026-07-17）

- 正式持久化job正常exit 0，W&B run为`2j7wdmq4`；30批、600条完整轨迹、600k环境步全部完成，纯训练`245.2s`，随后完成140条独立评估。final checkpoint为`_runs/checkpoints/dqc_frozen_s1policy_seed101_cdcf_direct_600k_20260717/final_post_update.pt`。
- 严格配对通过：相对既有QR run `8ry7xn6g`，30批的env step、reward、reward quantile、outage、discounted/undiscounted cost全部逐值exact；direct标签不一致率最大值为0。两个final checkpoint的actor、QR cost online/target、reward online/target和observation normalizer共43个共享state tensor逐位exact，说明新增head没有改变行为或QR基线。
- 末5批prequential CDF absolute error为direct/QR=`0.05169/0.10656`，direct看似改善51.5%；但Brier为`0.23360/0.23253`，direct反而差0.46%。post-update CDF error为`0.03114/0.03750`，只改善17.0%；post Brier为`0.19186/0.20023`，只改善4.2%。proper score没有支持“条件风险排序显著变好”。
- 独立140条truth outage为`0.25714`。direct预测`0.32722`，absolute error=`0.07008`；同run QR预测`0.26786`，error=`0.01071`。direct的总体CDF误差约为QR的6.54倍。direct/QR Brier为`0.19937/0.20282`，direct只改善1.70%，远低于25%门。故不做fresh520、不进入live 1M。
- 末段CDF“改善”不是可靠的慢收敛证据。direct pre预测与当前新批truth的相关仅`0.0196`，却与上一批truth相关`0.8256`；同批更新后的post预测与当前truth相关`0.8844`，一次rollout内20个head update使概率平均移动`0.1070`。QR对应的pre-lag相关为`0.6192`、平均移动`0.0829`，也有追批现象，但明显更弱。
- 最后5批truth均值恰好由此前5批的`0.23`升到`0.33`。direct复制上一批比例时刚好撞上连续高outage区间，才产生末5批CDF error下降；独立评估分布回到`0.257`后，过估立刻暴露。因此该结果不满足E78“末段改善且相对QR方向一致”的1.2M延长条件；不运行无改动的1.2M，也不事后移动门槛。
- 对用户“短跑是否误杀”的回答进一步细化：IQN 300k→600k属于真实慢收敛，应延长；C-DCF1则是跨批遗忘/高方差更新，保持同配方增加步数只会继续追逐最近20条轨迹。后续只允许针对已定位机制的单变量改动：首选每个rollout后更新一次EMA query head，让actor/eval读取跨批低通版本；备选是跨rollout replay或减少direct update次数。三条不能同时打开，均需重新预注册并从600k固定策略门开始。
- 完整history/profile位于`_runs/wandb_export/dqc_frozen_direct_cdf_600k_2026-07-17/`与`_runs/profiles/dqc_frozen_direct_cdf_600k_2026-07-17/`；`overview.png`为`2880×3440`且PIL解码通过。正式checkpoint/history/profile合计约12.6MB，全部位于`/vepfs`。


### E80：C-DCF2 EMA direct-CDF实现与600k预注册（2026-07-17）

- C-DCF2已在提交`80b67b1`实现。新增`cost_direct_cdf_query_mode=online|ema`与`cost_direct_cdf_ema_tau`，默认`online/0.005`；默认不会构造额外网络。online head仍按C-DCF1对每批做20次BCE Adam step，EMA head不进optimizer，只在每次online step后执行`θ_ema←(1-τ)θ_ema+τθ_online`。actor、prequential和评估读取selected EMA，online只作内部对照。
- `τ=0.005`沿用项目target网络的标准Polyak量级，不扫描。20次更新对应每rollout有效新权重`1-(1-.005)^20=0.0953895`，约10个rollout的低通记忆；600次head step后初始参数残余约`(1-.005)^600=0.0494`，故600k已覆盖约3个时间常数，不会因EMA天然慢热而只给极短预算。
- EMA通过`deepcopy(online)`构造，step 0参数逐位相同且不消耗RNG；`requires_grad=False`，checkpoint自动保存独立`cost_exceedance_ema_critic`。日志同时记录selected EMA、online direct、QR、EMA-online参数差、pre/post CDF/Brier和独立评估proper score，能区分“online仍追批但EMA稳定”与“两者都没学到”。
- 回归验证通过：新默认online相对C-DCF1旧smoke的41个checkpoint tensor及所有共享eval/summary数值逐项exact；online/EMA两条在首批更新前的41个共享tensor exact，EMA初始7个state tensor与online exact。解析小网络20次更新得到`0.0953895`，最大误差`7.45e-9`。
- 持久化CPU online、CPU EMA、EMA eval-only与全尺寸`[512,512]+LSTM512` CUDA smoke均exit 0；后者覆盖GAE-PPO、observation normalization与循环评估。EMA checkpoint共8个module，EMA/online最大参数差已非零，eval-only全部评估字段逐项exact。短T=32全为0-cost，只验证链路，不评价算法。
- 正式C-DCF2仍用P-M3 seed1成熟策略、rollout seed101、B20、T1000、30批=600k、20次critic update、MC/raw、risk-discount .995、chunk2500；唯一相对C-DCF1的变量是`query_mode=ema,tau=.005`。同run online head和QR必须与C-DCF1 checkpoint逐位exact，30批behavior truth逐值exact，标签不一致率0且无NaN/Inf。
- 机制门：末5批selected EMA的prequential CDF error或Brier相对same-run online至少改善20%，另一项不得恶化超过10%；selected的平均post-pre概率漂移需不高于online的50%，且对上一批truth的滞后相关不能高于online。主效果门保持不变：末5批selected相对QR至少一项改善20%、另一项不恶化超过10%；独立140条至少一项改善25%、另一项不恶化超过10%。
- 只有140通过才做fresh520，520通过才允许live 1M。若600k未过主门，只有在selected相对QR的末段与独立评估方向一致、且关键prequential误差从此前5批到末5批仍改善超过10%时，才允许一次从头1.2M长度审计；否则停止EMA005，不扫tau，也不同时加入replay/减少update。
- C-DCF1纯训练245.2秒。EMA多一个78.6k参数的无梯度forward/lerp，预计纯训练4.5--5.5分钟、含140终评总墙钟5.5--7分钟；正式任务只由`launch_background.sh`持久化，checkpoint/W&B/history继续写入`/vepfs`。


### E81：C-DCF2 600k结果——EMA修复追批，但没有超过QR（2026-07-17）

- 正式持久化job正常exit 0，W&B run为`z5f8b2zb`；600k纯训练`245.3s`，与C-DCF1的245.2s近似相同，随后完成140条评估。30批behavior reward/cost/outage逐值exact，标签不一致率0；相对C-DCF1，actor、online direct、QR online/target、reward online/target及normalizer共50个共享checkpoint tensor逐位exact，runtime和env_steps也exact。唯一新增权重是7个EMA state tensor。
- EMA确实完成预定的稳定化机制。prequential预测对上一批truth的相关由online的`0.8256`降到`-0.0084`；全程平均post-pre概率漂移由`0.10701`降到`0.00717`，比例仅6.7%，末5批为`0.00150/0.03079`。因此“减少追逐最近20条轨迹”不是失败点。
- 机制收益幅度仍未过门。末5批pre CDF error为EMA/online=`0.04471/0.05169`，只改善13.5%；Brier为`0.22398/0.23360`，只改善4.1%，都低于20%。EMA相对QR的末5批CDF改善58.0%，但Brier只改善3.7%，说明总体概率更平滑，不代表逐状态条件风险排序明显更准。
- 独立140条truth仍为`0.25714`。EMA/online/QR预测分别为`0.33237/0.32722/0.26786`，CDF absolute error为`0.07523/0.07008/0.01071`；EMA在总体校准上甚至略差于online，约为QR误差的7.0倍。Brier为`0.19363/0.19937/0.20282`，EMA相对online/QR只改善2.9%/4.5%，远低于25%独立门。
- 不做1.2M：末5批EMA CDF相对QR更好，但独立CDF更差，方向不一致；EMA Brier从此前5批`0.20194`恶化到末5批`0.22398`，没有“仍以>10%速度改善”的证据。600次EMA step后初始化只剩约4.9%；即使把`0.5`先验残余的量级粗略扣除，仍不足以把0.332拉到比QR 0.268更准。继续同配方主要消耗预算，不满足E80延长条件。
- 裁决：EMA005保留为“成功减少更新方差、但未改善到足以替代QR”的正机制/负性能消融；不做fresh520、不进live 1M、不扫tau。replay或减少direct update仍作为有分歧的可选路线记录，但两版direct均未在独立proper score上接近25%收益，优先级降到QR主路线和actor更新频率之后。
- 这再次回答训练长度问题：300k时EMA仍约0.41，确实太短；完整600k后才能看到它稳定到0.33附近。但“更长才看清”不等于“更长会成功”。当独立评估、proper score和末段趋势不一致时，继续延长会放大选择性报告风险。
- 正式history/profile在`_runs/wandb_export/dqc_frozen_direct_cdf_ema005_600k_2026-07-17/`与`_runs/profiles/dqc_frozen_direct_cdf_ema005_600k_2026-07-17/`；`overview.png`为`2880×3440`、约1.2MB且PIL解码通过，checkpoint约12MB，均在`/vepfs`。


### E82：C-Q3C/C-Q4C N64与局部quantile的600k长度审计预注册（2026-07-17）

- 重新开放原因不是移动旧100k门，而是E77明确把N64/local旧结论降级为“无早期收益”；direct两条600k未能替代QR后，QR仍是actor主CDF表示。现在用同一成熟固定策略补足600k，只回答“更多独立轨迹后表示是否出现稳定泛化收益”，不直接进入policy/PID。
- 共同基线为QR-N32 run `8ry7xn6g`：P-M3 seed1的1M actor、rollout seed101、B20、T1000、30批=600k、MC/raw、risk-discount .995、20次critic update、chunk2500、独立140条评估。既有truth为reward/outage/mean cost=`0.86576/0.25714/11.75714`，QR hard-CDF=`0.26786`，absolute error=`0.01071`，Brier=`0.20282`。
- C-Q3C只把`num_quantiles=32→64`，并启用`quantile_target_reduction=reference_mean,reference_samples=32`，消除旧N64目标数翻倍导致的梯度尺度混杂；uniform τ和其他配置不变。当前实现会同时把reward/cost distributional head改为64，但固定策略下reward critic不影响行为；joint clip效应属于该真实N64算法配置的一部分，后续若有收益再考虑cost-only N64解耦。
- C-Q4C保持N32，只把cost τ网格改为uniform+local mixture：中心`τ*=0.8`、半宽`0.1`、local fraction`0.5`，prediction使用importance weighting保持全局uniform-W1目标，CDF/mean使用对应quadrature权重。使用importance而非旧query-focused，是因为100k已证明后者放大cost gradient并长期clip，无法把收益归因于局部分辨率。
- 两候选在policy-only恢复后统一重置host/CUDA RNG；网络大小不同也必须得到30批完全相同的reward/cost/outage。标签、非有限值、target scale、joint/cost grad和crossing全部记录。N64与local同时后台运行，但各自独立进程/optimizer/W&B，不共享状态；A100 80GB和128 CPU可承载B20×2。
- 600k门：末5批prequential CDF absolute error或Brier相对QR至少改善20%，另一项不得恶化超过10%；独立140条的CDF error、Brier或mean-cost absolute error至少一项改善25%，其余关键项不得恶化超过10%；crossing绝对增加不得超过0.10。只有140过门才做fresh520，520仍过门才允许live 1M。
- 长度续跑规则不变：若600k未过但相对QR的末5批和独立评估方向一致，且关键误差从此前5批到末5批仍改善超过10%，只允许一次从头1.2M；否则停止当前N64/local，不扫N96/N128、local fraction或half-width。两候选不能彼此组合后再解释单变量效果。
- 既有N32 600k训练417.2秒，但当前同硬件direct run约245秒；N64 pairwise计算更大、local与N32近似。并行预计纯训练5--7分钟、含两组140评估总墙钟6--9分钟；均由`launch_background.sh`持久化，输出写`/vepfs`。


### E83：C-Q3C/C-Q4C 600k结果——短跑会误判收敛值，但两种表示均不晋级（2026-07-17）

- 两条正式持久化job均正常exit 0。N64-reference为W&B 'vcy6vawj'、纯训练'439.80s'；local-importance为W&B 'ndbi9jlj'、纯训练'434.27s'。并行共享GPU后，含两组140条独立评估的总墙钟约10分钟，略高于预估6--9分钟。没有NaN、OOM或标签异常。
- 公平性检查通过：两候选与QR32基线的30批reward、reward quantile、outage、mean cost及逐步reward/cost全部逐值exact；final checkpoint中的actor 16个state tensor、observation normalizer 3个tensor和lambda均逐位相同。N64只改变64维reward/cost distributional heads，local只改变cost tau网格/权重；因此真实行为与评估truth完全一致。
- 末5批prequential结果没有支持结构优势。QR32/N64/local的CDF absolute error分别为'0.10656/0.10797/0.10887'，N64和local相对基线恶化'1.32%/2.16%'；Brier分别为'0.23253/0.23191/0.23170'，只改善'0.27%/0.36%'，远低于20%门。N64从此前5批到末5批的CDF/Brier又恶化'9.16%/21.63%'；local恶化'13.79%/21.58%'，没有“仍以超过10%速度收敛”的延长证据。
- 独立140条truth固定为outage '0.25714'、mean cost '11.75714'。QR32的CDF/Brier/mean-error/crossing为'0.26786/0.20282/0.77573/0.05737'。N64为'0.25859/0.20021/0.87055/0.15828'：CDF error从'0.01071'降到'0.00145'，相对改善86.46%，但Brier只改善1.28%，mean error恶化12.22%，crossing绝对增加'0.10090'。后两项分别越过“其他关键量不得恶化10%”和“crossing增加不超过0.10”的预注册边界。
- local的独立CDF/Brier/mean-error/crossing为'0.27644/0.19930/0.80526/0.10023'。CDF error为'0.01929'，相对QR恶化80.07%；Brier只改善1.73%，mean error恶化3.81%，没有任一独立proper/mean指标达到25%改善门。
- N64的86%相对CDF改善不能脱离绝对量和proper score解释：它只把总体预测移动约0.0093，而140条truth的二项标准误约0.0369；基线分母本来只有0.0107，所以相对百分比被放大。若它真实改善逐状态条件概率，Brier、mean和末段prequential应至少给出同方向证据；本次恰好相反，同时monotonicity明显变差。
- 裁决：两条都不做fresh520、不进入live 1M，也不启动同seed原样1.2M。600k已经回答固定策略表示筛选，但不被写成“任何初始化下永久无效”。有分歧的备选消融保留为：未来在主线配置稳定后，用多个critic初始化各600k检验N64点估计是否复现；这比让同一初始化继续训练到1.2M更直接回答初始化偶然性。当前不扫N96/N128、local fraction/window，也不组合N64+local。
- 本轮同时修正旧100k结论的证据等级：短跑确实可能误杀慢收敛表示，N64在600k才出现很准的总体CDF点；但完整门显示“训练更久后出现一个好数”不等于算法胜出。下一主线回到critic数据速度慢于policy移动速度的问题，优先实现默认关闭的actor更新间隔消融，而不是继续增加同一批critic更新次数。
- 正式history/profile位于'_runs/wandb_export/dqc_frozen_qr32_n64_local_600k_lenaudit_2026-07-17/'和'_runs/profiles/dqc_frozen_qr32_n64_local_600k_lenaudit_2026-07-17/'；比较表为'quantile_length_audit_comparison.csv'，六面板图为'quantile_length_audit_comparison.png'（'3000×1500'，PIL解码通过）。

### E84：P-M7 actor cadence=2实现、严格回归、全尺寸验证与1M预注册（2026-07-17）

- 机制动机来自600k冻结critic长度审计：B20下每个policy版本只有20条独立轨迹，把同一批做20次QR update只能增加优化步，不能增加tail事件。P-M7新增默认关闭的`actor_update_interval=2`：cost/reward critic和经验PID仍每个B20 rollout更新，Actor参数与Actor自带observation RMS连续两个rollout冻结；收满两批后沿环境维合并成B40，只执行一次8-epoch PPO。它与P-M6直接把`num_envs=40`不同，后者连critic/PID也每40条才更新，因此两条是可解释的独立消融。
- importance sampling语义已明确实现：每个transition在采样时保存固定`log π_old(a|h)`；每个PPO epoch用当前网络重新前向得到`log π_new`，计算`r=exp(logπ_new-logπ_old)`并做PPO-Clip。更新后不需要另存“当前概率”，因为下一epoch的分子会重新计算，而分母始终是采样策略。两批收集期间Actor和`actor.obs_rms`不变，所以两批都来自同一behavior policy，不是把变过的策略数据混在一起。
- 为避免cadence改变全局动作随机流，每个B20 rollout都按历史首个actor epoch的位置预抽K组behavior baseline action；到期后只把缓存动作送入最新cost critic，不再消耗随机数。合并前把flatten字段恢复为`[T,B,...]`，沿B拼接后再flatten；`h0/c0`、old log-prob、GAE、实际动作和baseline action保持同一episode顺序。构造期暂只开放已验证的recurrent GAE-PPO、MC/raw cost、online query、`s0_aux=0`路径，并要求正式rollout数能被interval整除。
- 默认兼容回归以提交`c81d01b`的临时worktree为金样本，seed123、CUDA、`2×B2×T40=160`步。新代码在`interval=1`时的Actor、observation normalizer、reward/cost critic和两个target共6个module全部逐tensor exact，lambda tensor exact，最终评估dict exact；最大绝对差为0。唯一新增内容是cadence配置、runtime和summary诊断，证明默认值没有暗改P-M3。
- 首次`interval=2`入口smoke在训练开始前暴露`self.warmup_iters`初始化顺序错误并exit 1；已改为按正式初始化同一规则直接解析`args.warmup_iters`。修复后同名持久化作业exit 0。80→160步两个pre-update checkpoint中Actor的16个state项逐位exact，cost/reward critic已更新；160步到final时Actor可学习权重最大变化约`6.0e-4`，更新事件恰为1。Actor内的RMS只在到期后一次合并160条观测；另一个每rollout变化的`obs_normalizer`只服务raw-state critic，不参与LSTM行为策略。
- 小型smoke的首epoch ratio最大偏差为`5.05e-5`，合并4条轨迹、实际2个PPO epoch，末epoch KL=`4.53e-5`、clip fraction=0。合并顺序与baseline RNG的合成断言也逐项exact。该量级与既有interval=1 recurrent CUDA的约`0.8e-5～1.4e-5`同属批处理浮点重放误差，远小于0.1 clip范围，不是off-policy证据。
- 全尺寸40k shape/memory smoke使用P-M3 seed1的B20、T1000、LSTM512、QR32、20 critic update、8 PPO epoch，仅设`interval=2`；持久化训练22.1秒、exit 0，无NaN/OOM，观测显存约2.8GB。一次到期事件正确合并40条轨迹并完成8个epoch；首epoch ratio最大偏差`1.1444e-5`，末epoch KL=`0.005374`、clip fraction=`0.321575`。40k reward/outage只属初始化阶段，不用于算法裁决。
- 正式P-M7预注册为P-M3压力seed1完整1M：`50×B20×T1000`，预计25次Actor事件；MC/time-weight .995、T1 sigmoid、PID target .15、Kp1/Ki.1/window50、QR32、LSTM512、20 critic update、8 PPO epoch、target-KL关闭及全部LR保持不变。纯训练预计约8～10分钟，内置评估后若无明显失效再做fresh520，总墙钟约12～16分钟；全程使用`launch_background.sh`，不能以100k/300k早期reward提前终止。
- screen只用于节省明显失败的520评估：1M内置评估需reward至少0.60、outage不高于0.35，且ratio、KL、loss均有限；否则停止。fresh520正式门相对P-M3 seed1的`reward=0.8622,outage=159/520=0.3058`：候选需outage不高于0.22且至少下降0.08、reward不低于0.75，末200k不能出现更大的Actor–critic–PID周期。通过才扩seed0/2各1M；若接近门且末段仍持续改善，再预注册2M；最终候选才进入与QCPO/QCPO_refs相同5M、多seed协议。
- 有分歧的路线分别保留，不能与本轮混改：① PID也每两个rollout更新，检验控制器cadence对齐；② 简单跳过第一批Actor更新但丢弃其数据，作为“数据利用率”消融；③ 多cost-critic初始化的600k固定策略复验，回答表示结果的初始化偶然性；④ `interval=4`或P-M6 B40与cadence组合。首轮只跑P-M7 interval2，因为它最直接隔离“同一policy独立轨迹数”，其它路线必须在本轮结果后单独预注册。

### E85：P-M7 seed1完整1M结果与2M长度审计预注册（2026-07-17）

- 正式job `DQCAC_DynamicButton_pm7_actorint2_timew995_pi_target015_smoothT1_b20_ckpt100k_1m_s1`绑定提交`9f30bbf`，W&B run `5kro4lx2`、exit 0；1M纯训练`395.6s`，与P-M3约6.5分钟相同。25次Actor事件每次合并40条轨迹并做8个PPO epoch，共200个optimizer step；P-M3为50×8=400 step，但二者trajectory-epoch暴露量同为8000。全程首epoch ratio误差mean/max为`1.2e-5/2.2e-5`，比P-M3的`1.4e-5/2.8e-5`还小，IS/旧log-prob接线没有异常。
- 训练曲线说明1M不能只看一个终点。五个200k阶段的P-M3→P-M7 reward为`0.124→0.003, 0.602→0.340, 0.835→0.663, 0.823→0.790, 0.825→0.812`，候选前600k明显更慢，后400k才接近。outage为`0.065→0.015, 0.175→0.160, 0.280→0.205, 0.260→0.185, 0.180→0.210`；600k和800k附近各出现一次风险周期，末200kstd从P-M3的`0.0678`升到`0.1044`。cadence降低了早中期风险，但没有消除成熟期Actor–PID振荡。
- 末200k P-M3/P-M7的reward=`0.8253/0.8120`、outage=`0.180/0.210`、lambda=`0.2206/0.1928`。PPO只在P-M7到期批记录；末段KL=`0.002459/0.002491`、clip fraction=`0.1293/0.1373`，几乎相同且候选略高。因此收益不是“每个actor事件步子更小”，而更可能来自每次事件使用40条独立轨迹和事件数减半。
- 内置140条P-M7为reward/outage=`0.9575/43÷140=0.3071`，critic hard-CDF=`0.1810`，只通过“非明显失效”的screen，不支持安全达标。随后按预注册运行fresh520。旧P-M3的`0.8622/0.3058`来自加入统一eval RNG reset之前的JSON；为补齐Brier并公平配对，本轮用当前同一evaluator、同seed+777随机流重新评估P-M3，得到`0.83419/158÷520=0.30385`。原/新baseline的outage基本一致，预注册门结论不受协议更新影响；后续数值以当前配对为准。
- 当前统一fresh520的P-M3→P-M7为reward `0.83419→0.96141`，增加`0.12722`（15.25%），保守独立样本95%区间`[+0.06094,+0.19349]`；outage `158/520=0.30385→132/520=0.25385`，下降`0.05000`，Newcombe 95%区间`[-0.10412,+0.00452]`，上界略跨0。候选自身Wilson区间为`[0.21834,0.29296]`，仍整体高于alpha=0.20。
- critic泛化是同方向实质改善：hard-CDF error `0.19952→0.06569`（改善67.08%），smooth error `0.19634→0.06142`（68.72%），mean-cost error `4.97366→1.80671`（63.67%），Brier `0.24852→0.20203`（18.71%）；crossing `0.11241→0.10230`也略降。与N64只改善总体CDF不同，P-M7同时改善proper score、mean和monotonicity，说明更多同策略独立轨迹确实让cost风险估计更可用。
- 原1M强门裁决保持不动：reward≥0.75通过；outage≤0.22和至少下降0.08均失败。因此P-M7不能直接扩seed0/2或称为安全成功。但它在reward、outage、CDF、Brier、mean和crossing上全部同方向优于统一P-M3，且1M只有25个Actor事件，属于E84允许的“接近门、需要长度审计”，不是应在100k/300k丢弃的组合。
- 预注册一次从头2M：只把`num_iterations=50→100`，其余参数、seed和评估协议不变。LR scheduler只依赖实际Actor事件编号，不依赖总迭代数，所以2M run的前1M理论上应与本run逐点复现；100k与1M checkpoint/history将先做exact/数值对拍，失败则不能解释为纯长度效应。预计纯训练约13～14分钟，内置评估后若仍通过screen再做fresh520，总墙钟约18分钟，全程持久化。
- 2M不是移动成功门：fresh520仍需reward≥0.75、outage≤0.22，且Brier不得比1M的`0.20203`恶化超过10%、hard/mean error不得反弹；末400k周期不能继续放大。通过后必须补P-M3相同2M预算，才可谈cadence支配；未通过则停止原样延长，不跑3M，转向已记录的“PID与Actor都每两批更新”独立时序消融。
- 完整history/profile、五阶段CSV和比较图位于`_runs/wandb_export/dqc_pm3_vs_pm7_actorint2_seed1_1m_2026-07-17/`与`_runs/profiles/dqc_pm3_vs_pm7_actorint2_seed1_1m_2026-07-17/`。`fresh520_comparison.png`为`2864×1435`、可由PIL解码；同目录保留`fresh520_comparison.csv/json`和原始`overview.png`。

### E86：P-M7 2M长度审计——不是训练不足，闭环在终点再次失配（2026-07-17）

- 正式持久化job `DQCAC_DynamicButton_pm7_actorint2_timew995_pi_target015_smoothT1_b20_ckpt100k_2m_s1`正常exit 0，W&B run为`mvxemk3z`；100批、2M环境步纯训练`779.4s`，保存50次Actor事件。只相对1M把`num_iterations=50→100`，其余参数和seed不变。
- 长度因果检查通过：2M run在100k与1M checkpoint的六个网络module、lambda、runtime训练状态和非eval指标都与原1M run逐项exact，最大差为0；前1M控制台/history也一致。因此1M后的变化来自新增训练时间，不是初始化、随机流或代码版本漂移。
- 2M后五个200k阶段的`reward/outage/lambda`均值依次为`0.9712/0.280/0.2637`、`0.9255/0.240/0.3821`、`0.9224/0.215/0.2531`、`0.8559/0.255/0.3676`、`0.9177/0.155/0.2074`。lambda在约0.21–0.38间随outage反复升降，1.2M和1.7M附近重复出现高风险→高lambda→低reward→低lambda的周期；没有形成稳定平台。
- 2M post-update内置140条为reward/outage=`1.19819/58÷140=0.41429`，真实mean cost=`15.700`；critic hard/smooth CDF=`0.14219/0.14484`、predicted mean cost=`7.81394`、Brier=`0.31862`。相对同协议1M内置结果`0.95752/0.30714`，reward提高25.1%，但outage恶化34.9%。hard-CDF absolute error从`0.12612`扩大到`0.27210`，mean-cost error从`2.76898`扩大到`7.88606`，Brier从`0.23874`恶化33.5%；crossing虽从`0.10622`降到`0.08733`，不能抵消严重的风险低估。
- 按E85预注册screen，内部outage `0.414>0.35`且Brier明显反弹，所以不做fresh520、不跑3M、不扩seed。目标是0.20，当前失败幅度远大于140条二项抽样误差；为一个已被screen拒绝的终点再花520条不会改变路线裁决。
- 额外用同一eval-only协议评估2M的`rollout_step002000000.pt` pre-update快照，持久化job正常exit 0。更新前reward/outage=`1.123/43÷140=0.30714`，critic/truth=`0.26558/0.30714`、absolute error=`0.04157`、Brier=`0.21670`；最后一次critic+8 epoch PPO联合更新后，reward/outage变为`1.198/0.41429`，critic预测却降为`0.14219`。一次更新让真实风险增加`0.10714`，预测风险反向减少约`0.1234`，CDF error扩大约6.55倍。
- 这个pre/post对拍把极限环定位为真实时序问题，而不只是日志噪声：最后200k rollout policy的平均outage只有0.155，使lambda降到约0.207；终点Actor再依据最新lambda和同批更新后的critic做8次PPO，策略变得更激进，但训练已结束，PID没有下一批轨迹纠正。B20单批在p=0.2附近的二项标准差约0.089，经验率又只能按0.05跳变；当前PID每B20响应两次而Actor每B40才响应一次，控制器与执行器不同频会进一步放大超调。
- 下一单变量路线为P-M8：critic仍每B20做20次更新，Actor仍每两批合并B40做8 epoch；只把经验PID也改为每两批更新一次，并用两批共40条真实cost一次性更新window、I/P项和lambda。这样lambda在Actor冻结期间不变化，B40当前batch outage标准差约降到0.063，并且Actor看到的是与其响应周期一致的控制量。anti-windup、降低Kp/Ki、增大num_envs和target-KL保持关闭，分别留作后续消融，不能同时叠加。
- 完整history在`_runs/wandb_export/dqc_pm7_actorint2_seed1_1m_vs_2m_2026-07-17/`，2M profile和`2880×4128`可解码曲线在`_runs/profiles/dqc_pm7_actorint2_seed1_2m_lenaudit_2026-07-17/`。pre-update诊断JSON为`_runs/DQCAC_DynamicButton_pm7_actorint2_2m_s1_preupdate_eval140_s1.json`；2M checkpoints约221MB，均位于`/vepfs`而非20G根目录。

### E87：P-M8 同频PID实现、精确回归与1M预注册（2026-07-17）

- 新增默认关闭的`pid_update_interval`，默认1逐位保持历史路径。只允许正interval在`empirical_pid`、`outer_interval=1`且与`actor_update_interval`相等时启用，防止把不同policy版本或不同控制边界混在一个无法解释的PID batch中。
- interval2时第一条B20 rollout只把`disc_cost`的20个MC标量加入cache，lambda、window和P/I状态不动；第二条到期后沿episode维拼成B40，一次调用经验PID，随后同一边界才执行B40 actor PPO。critic仍对两条B20各自立即做20次更新，Actor权重/obs RMS仍按P-M7冻结两批；新增cache不保存state/action，显存开销仅40个标量。
- episode-scaled控制语义保持：B40配`pid_reference_episodes=10`得到scale=4，leak为`rho^4`，I增量和`delta_max`按40条轨迹缩放；Kp只在Actor响应边界输出一次。日志/checkpoint/summary新增PID interval、due、累计rollout、batch episode数与事件计数，能直接证明controller与actor是否一一对齐。
- 默认回归使用旧提交`f4315f0`与当前代码跑同seed、同两批tiny recurrent PPO。两个rollout checkpoint和final的六个module、全部tensor、lambda、所有共享runtime字段、eval和summary公共字段逐项exact，最大差0；当前只多出五个PID cadence诊断字段。旧worktree在比较后已删除。
- interval2 tiny时序通过：第一/第二个pre-update checkpoint的Actor参数最大差0，cost critic差`0.00200`；final Actor参数发生变化。runtime从`events=0, accumulated=1, due=0`变为`events=1, accumulated=2, due=1, batch_episodes=4`，首epoch ratio误差`5.58e-5`。
- 强制`cost_limit=0`使4条tiny轨迹outage全为1：理论filtered error=`1-.15-.02=.83`、episode scale=.4、限幅后`delta I=.02`、lambda=`.02+.83=.85`；实际`pid_i=.020000000000000004`、lambda=`.8500000238`，仅float32误差`2.38e-8`。两个pre快照lambda都为0，证明第一批没有偷跑PID。
- 全尺寸smoke覆盖`B20×T1000`、QR32、LSTM512、20次critic step、8次PPO epoch和B40合并，纯训练`21.65s`、exit 0、无NaN/OOM。PID/Actor事件均为1且batch均为40；两批间Actor参数差0，到期后差`0.002295`；首epoch ratio误差`1.43e-5`、末epochKL=`8.48e-4`、clip fraction=`0.0272`。final checkpoint eval-only恢复exit 0，20条评估逐字段exact，新增runtime完整恢复。
- P-M8正式配置只在P-M7 1M上增加`pid_update_interval=2`：seed1、50×B20×T1000=1M、25个Actor/PID事件、MC/raw/online、time weight .995、T1、target .15、Kp1/Ki.1/window50/leak.97/deadband.02、QR32、LSTM512、20 critic update、8 PPO epoch，其余LR和RNG协议不变。预计纯训练6.5–8分钟、140条评估约1分钟；通过screen后的fresh520约4分钟，总墙钟约12–14分钟，全部持久化后台。
- screen仍为reward≥0.60、outage≤0.35、ratio/KL/loss有限；失败则不做520。正式门保持reward≥0.75、fresh520 outage≤0.22，并相对P-M7当前`132/520=.25385`至少下降约.03；Brier不得比`.20203`恶化10%，hard/mean error不得反弹。机制门要求25个PID事件与25个Actor事件严格对齐、每次40条，末200k outage std相对P-M7的`.1044`至少降低20%或末400k不再出现增长周期。
- 1M过门才扩seed0/2。若只接近门（reward≥0.70、outage≤0.28）且最后400k风险/校准仍同向改善，才允许一次2M长度审计；否则停止当前同频参数，不跑3M。下一分歧路线分别保留为anti-windup/更小Kp-Ki、扩大num_envs、target-KL或critic更新后Actor风险验证，不能在P-M8首轮叠加。

### E88：P-M8 seed1 1M结果——同频PID通过强门，但存在可量化reward代价（2026-07-17）

- 正式job正常exit 0，绑定提交`d99b928`，W&B run `yxarpian`；1M纯训练`391.7s`，与P-M7的395.6s等价。PID/Actor各25个事件且每次都为40条轨迹，首epoch ratio误差最终`1.23e-5`，没有NaN/OOM或cadence错位。
- 五个200k阶段P-M7→P-M8的reward为`0.0034→0.0034, 0.3402→0.3428, 0.6631→0.5932, 0.7898→0.7112, 0.8120→0.8005`；outage为`0.015→0.015, 0.160→0.150, 0.205→0.225, 0.185→0.230, 0.210→0.195`。P-M8中段更慢且更不安全，末段才恢复，不能表述为全程支配。
- 预注册减振门通过：末200k outage std从`0.11005`降到`0.07619`，下降30.76%；lambda std从`0.06267`降到`0.04759`，下降24.07%；CDF calibration error均值从`0.07047`降到`0.05094`，改善27.71%。末段reward只下降1.41%。同频把高频追批变成较低幅B40锯齿，但并未令每个训练batch都稳定在0.20。
- 内置140条P-M8为reward/outage=`0.905/33÷140=0.2357`，critic/truth=`0.196/0.236`、Brier=`0.1975`，通过screen；P-M7同协议为`0.9575/0.3071`和Brier`0.2387`。随后按原门执行统一fresh520，而不是用140条宣布成功。
- fresh520的P-M3/P-M7/P-M8分别为reward=`0.83419/0.96141/0.87025`，outage=`158/132/103÷520=0.30385/0.25385/0.19808`。P-M8自身Wilson 95%区间为`[0.16609,0.23449]`；点估计低于alpha=0.20且明确通过0.22工程门，但区间仍跨0.20，不能写成统计上证明真实outage严格小于alpha。
- 相对P-M7，P-M8 reward下降`0.09116`（-9.48%），保守独立95%区间`[-0.14882,-0.03350]`；outage下降`0.05577`（-21.97%），Newcombe 95%区间`[-0.10630,-0.00491]`，两者都不跨0。这是显著的安全–reward交换，不是免费改进。相对P-M3，reward增加`0.03605`但区间`[-0.02988,0.10198]`跨0；outage下降`0.10577`，区间`[-0.15765,-0.05316]`。
- critic总体校准同步改善。P-M7→P-M8 hard/smooth CDF error=`0.06569→0.01436`/`0.06142→0.01124`，改善78.13%/81.69%；mean-cost error=`1.80671→1.55780`，改善13.78%；crossing从`0.10230`升到`0.12004`，绝对恶化0.01774。raw Brier=`0.20203→0.15848`表面改善21.55%，但两个策略的真实outage基率不同，不能把这个百分比单独解释为条件critic提升。按各自520条基率构造常数预测，P-M7/P-M8的climatology Brier约为`0.18941/0.15884`，对应Brier Skill Score约为`-6.66%/+0.23%`：P-M8从劣于常数基线变为接近基线，但尚未证明有强逐状态分辨率。后续同时报告BSS、reliability/resolution与总体CDF误差；原预注册raw-Brier门不事后修改。
- E87的九项预注册门全部通过，所以不在seed1继续扫Kp/Ki或跑2M，直接扩seed0/2各1M并对所有成功训练的final checkpoint做fresh520，避免只评估好seed。两条可并行：每条纯训练约6.5–8分钟，并行墙钟预计8–10分钟；两组520并行预计再4–6分钟，A100 80GB和128 CPU可承载40个训练env或80个评估env。
- 多seed晋级标准预先固定：至少2/3 seed的fresh520 outage≤0.22，三seed平均outage≤0.22且平均reward≥0.75；报告seed作为统计单位的mean±SD以及pooled episode比例，但不以pooled 1560条代替seed方差。若通过，P-M8成为当前DQCAC主候选，随后进入与校准QCPO/QCPO_refs相同5M预算；若失败，先按seed轨迹判断是Kp过强、critic反弹还是初始化敏感，再选择单变量路线。
- 完整history/profile在`_runs/wandb_export/dqc_pm7_vs_pm8_pid_cadence_seed1_1m_2026-07-17/`与`_runs/profiles/dqc_pm7_vs_pm8_pid_cadence_seed1_1m_2026-07-17/`；后者含`phase_comparison.csv`、`fresh520_comparison.csv`、置信区间/门控JSON及比较图。P-M8 fresh520原始JSON为`_runs/DQCAC_DynamicButton_pm8_actorint2_pidint2_1m_s1_final_eval520_s1.json`。

### E89：P-M8三种子裁决——seed1改善不稳定，暂不进入5M（2026-07-17）

- seed0/2与seed1完全同配置完成1M，各有50个history点、25次PID事件和25次Actor事件，最终env step均为1,000,000且exit 0。seed0/2并行纯训练分别为702.73s/724.13s；seed1单独运行是391.7s，差异主要来自两个B20多进程作业同时竞争CPU，墙钟仍约12分钟而不是串行约24分钟。原始离线run为jhm20t8t/bodb903c，完整50行历史已脱敏回放到在线W&B nk6q2qjn/mjs4s7l7；seed1在线run为yxarpian。
- 统一fresh520为：seed0 reward=0.89591,outage=170/520=0.32692；seed1 0.87025,103/520=0.19808；seed2 0.81434,117/520=0.22500。三seed reward为0.86017±0.04171（seed SD），outage为0.25000±0.06796；只1/3 seed满足≤0.22，平均outage也高于0.22，故预注册多seed门失败。1560回合合并比例为390/1560=0.25、Wilson 95%区间[0.22914,0.27208]，但主裁决仍以seed而不是把全部回合伪装成独立算法重复。
- 三个critic都低估fresh风险。seed0/1/2 hard-CDF分别为0.24808/0.18371/0.18438，对truth的误差为0.07885/0.01436/0.04062；predicted-mean cost误差为1.5165/1.5578/1.3984。raw Brier为0.22761/0.15848/0.18384，但相对各自经验基率常数预测的Brier Skill为-3.44%/+0.23%/-5.43%。因此seed1的低raw Brier主要含有低outage基率效应，逐状态风险排序仍没有稳定超过climatology。
- 末200k揭示跨seed分叉：seed0的reward/outage/lambda=0.929/0.335/0.373，seed1为0.801/0.195/0.186，seed2为0.662/0.180/0.195。seed0并非risk penalty数值消失：其risk coefficient、归一化risk advantage和KL都更大，仍学到高reward高风险策略；更像是action-conditioned risk方向/校准不足，而不是简单把lambda乘大即可保证正确方向。
- 对三个rollout_step001000000.pt做同一140回合pre/post诊断：seed0 outage 0.3571→0.2786，seed1 0.2500→0.2357，seed2 0.1857→0.1929。最大绝对变化0.07857，未达到预注册0.08 fresh触发线，而且最差seed0的最后一次更新反而更安全。因此不把post-update guard列为下一优先；seed0问题是持续的末段高风险和critic tail低估，不是单次终点跳变。
- 裁决：P-M8证明“PID与Actor同频、40条trajectory后再响应”是有效稳定性组件，但seed1结论不能复现为稳健主算法；不按当前参数进入5M或QCPO_refs最终比较。正式CSV/JSON/图在_runs/wandb_export/dqc_pm8_pid_cadence_multiseed_1m_2026-07-17/和_runs/profiles/dqc_pm8_pid_cadence_multiseed_1m_2026-07-17/，含三seed曲线、fresh520图和pre/post图，所有PNG已解码验证。

### E90：W&B在线脱敏与P-M9固定安全余量预注册（2026-07-17）

- 此前seed0/2切离线不是W&B容量或账号故障，而是执行层对第三方上传的默认拦截；用户已经明确授权训练指标在线监控。在线验证显示两条clean replay均finished、各50行、max env step 1M，配置无私密键和绝对路径。最初被中止的seed2部分在线run huxmufc8只到约120k，标记为无效工程运行，不进入任何算法比较。
- 提交18b28ca增加wandb_public_config()：过滤wandb_dir/checkpoint_dir/calibration_source_checkpoint、任何绝对路径和名称含token/secret/password/credential/api_key的字段；同时关闭machine metadata、system stats、Git、源码、job和requirements上传。该改动只影响外部日志元数据，不参与前向、RNG或optimizer；离线privacy smoke、路径/token断言、语法和diff检查均通过。可复用的export_wandb_offline.py能从完成的本地.wandb流恢复完整config/summary/history，并拒绝默认读取无ExitRecord的活跃run。
- P-M9只把P-M8的pid_target_prob=0.15→0.10，其它网络、seed0、B20、Actor/PID interval2、QR32、20 critic updates、8 PPO epochs、Kp=1、Ki=.1、window50、1M步和评估协议全部冻结。它不是把论文约束alpha改成0.10，而是把controller安全余量由0.05增到0.10，检验较早维持更大lambda能否抵消seed0约0.079的tail风险低估与闭环滞后。
- 只先跑压力seed0完整1M；除NaN/OOM/确定性错误外不以100k/300k早停，因为P-M7/P-M8已证明闭环方向可在后半段改变。内置140 screen为reward≥0.60,outage≤0.40，只用于发现灾难性退化；通过后fresh520正式门为reward≥0.75,outage≤0.22且相对P-M8 seed0的0.32692至少下降0.08，CDF/mean/Brier不得出现新的明显发散。过门才原参数扩seed1/2，失败则停止固定target下调，不扫0.08/0.12/0.13追端点。
- 单条独占训练预计6.5～8min，内置140约1min，条件fresh520约4～5min，总墙钟12～14min。使用launch_background.sh持久化后台与脱敏在线W&B，checkpoint/log全部留在/vepfs。若路线存在分歧，保留为独立消融：Wilson-UCB/置信上界PID让margin随样本量变化；按独立校准误差自适应target；降低Kp/Ki或anti-windup；以及增强action-risk排序的ensemble/保守查询。它们不与P-M9混合，以保留因果归因。

### E91：P-M9 seed0结果与终点pre/post诊断触发规则（2026-07-17）

- 正式run k4obc0u9、job均exit 0，1M纯训练397.1s，25次PID/Actor事件完整；远端50行history且只含config/output/summary，隐私审计通过。前400k与P-M8几乎相同，target差异生效后末200k P-M8→P-M9的reward/outage/lambda为0.929/0.335/0.373→0.741/0.285/0.394；固定更低target确实压低风险，但降低reward并提高PPO KL。
- fresh520从P-M8的reward/outage=0.89591/170÷520=0.32692变为0.74188/102÷520=0.19615。reward差-0.15403，Welch 95%区间[-0.20742,-0.10064]；outage差-0.13077，Newcombe 95%区间[-0.18305,-0.07754]。安全门和下降0.08门通过，但reward比0.75低0.00812，严格失败。
- 校准门也明确失败：hard/smooth CDF误差0.07885/0.07630→0.13912/0.14231，恶化76%/87%；mean-cost误差1.5165→5.1150，恶化237%；Brier Skill从-3.44%降到-33.65%。P-M9安全并不是critic更准，而是更大lambda与保守风险高估换来的。故不扩seed、不扫其它固定target。
- 终点P-M9的KL/clip=0.00975/0.405，P-M8为0.00457/0.241，且最后lambda升到0.602。为区分“整体target过保守”和“最后一次强PPO过冲”，只对rollout_step001000000.pt做同协议140回合pre/post诊断。只有abs(outage delta)≥0.08，或pre在140条同时reward≥0.75且outage≤0.22，才触发pre-checkpoint fresh520；否则停止。该诊断不能推翻P-M9 final的预注册失败，只决定post-update guard是否值得继续。
- 正式历史、CSV、decision JSON和图位于_runs/wandb_export/dqc_pm8_vs_pm9_pid_target_seed0_1m_2026-07-17/及_runs/profiles/dqc_pm8_vs_pm9_pid_target_seed0_1m_2026-07-17/；overview、phase、fresh三张PNG均可解码并完成目视检查。

### E92：P-M9终点fresh520更新前后——不是简单保留旧策略就能解决（2026-07-17）

- 预注册的140回合诊断中，更新前checkpoint为`reward/outage=0.800/28÷140=0.200`，同时满足`reward≥0.75`和`outage≤0.22`，因此严格触发额外fresh520。该触发只决定是否值得精评，不改写E91对final P-M9的失败裁决。
- 统一fresh520下，最后一次联合更新前后reward为`0.82559→0.74188`，差`-0.08370`，保守独立样本Welch 95%区间为`[-0.14172,-0.02568]`；reward下降不是520条抽样噪声。outage为`122÷520=0.23462→102÷520=0.19615`，差`-0.03846`，Newcombe 95%区间`[-0.08826,+0.01155]`，点估计变安全但差异区间仍跨0。
- 两个端点没有一个同时过原门：pre-update的reward过门，但outage `0.23462>0.22`；post-update的outage过门，但reward `0.74188<0.75`。因此“若更新后风险变差就回滚”的二元guard在本例无解：保留pre仍不安全，接受post仍丢失过多reward。需要控制更新幅度或改变闭环信号，而不是只在两个端点中选一个。
- critic泛化在一次更新后发生数量级退化。hard/smooth CDF error从`0.01058/0.00859`变成`0.13912/0.14231`，相对增加`1215%/1557%`；predicted/true mean cost从`10.183/10.038`变成`13.282/8.167`，absolute error由`0.14485`增至`5.11495`，增加`3431%`。raw Brier `0.18112→0.21074`，恶化16.35%；Brier Skill从`-0.86%`降至`-33.65%`，下降32.79个百分点。crossing虽从`0.14076`降到`0.08772`，说明“不交叉”不能保证概率校准。
- 时序证据与“最近批次过强”一致但尚不能写成单一因果证明：更新前lambda为`0.46557`，24次PID/Actor事件；最后logged rollout outage为`0.45`，第25个B40 PID/Actor事件后lambda升到`0.60228`，PPO末epoch KL/clip为`0.00975/0.40495`。策略真实outage下降，但刚更新的critic在fresh分布上从轻微低估变成严重高估，符合对最近有限trajectory重复优化后跨策略/跨批泛化失配。
- 下一步优先级据此调整：不实现单纯checkpoint accept/reject guard，也不继续扫固定PID target。先用既有三seed history做零训练开销的prequential校准审计，判断`truth-CDF`的符号和持续时间；若符号具有可预测性，预注册默认关闭的“校准偏差EMA→自适应PID安全余量”单变量路线。若符号不可预测，则转向限制单次联合更新幅度或critic跨rollout验证，而不把噪声直接反馈进PID。
- 新增可复用工具`compare_eval_snapshots.py`，统一生成单组reward/Wilson区间、Welch/Newcombe差值区间、CDF/Brier/BSS/mean-cost统计及四面板图。正式结果位于`_runs/profiles/dqc_pm9_pre_post_seed0_1m_2026-07-17/`：`comparison.csv`、`statistics.json`与`comparison.png`（`2775×1895`，PIL解码与目视检查通过）。工具和本地文件不向W&B上传路径。
- 用户再次明确：正式训练允许并应使用W&B在线监控。后续run默认online；run name/group/tags/config只含环境、算法、seed和公开超参数，继续过滤绝对路径、用户名、机器信息、凭据和源码。offline只用于真实断网兜底，不再因一般隐私顾虑默认启用。

### E93：P-M8三种子prequential校准偏差审计预注册（2026-07-17）

- 数据冻结为P-M8 seed0/1/2的既有1M `combined_history.csv`，不新增训练。每两个连续B20 rollout按真实`pid_update_interval=2`聚合成一个B40控制事件；偏差定义为`u=pre-update truth outage - pre-update CDF`，`u>0`表示critic低估风险。post-update CDF不得充当因果预测输入。
- 主审计区间为`400k--1M`，稳健性区间为`600k--1M`。同时报告逐seed与去除seed均值后的pooled lag-1相关、连续事件同号率、单侧二项检验、last-value与因果EMA预测误差。EMA固定`tau=0.2`，预测事件t时只能使用t之前的偏差，不能偷看当前truth。
- 只有四项同时成立，才允许实现“校准偏差EMA→自适应PID target”：pooled lag-1相关`>0.25`；同号率`≥0.65`且单侧`p<0.10`；EMA一步MAE相对永远预测零偏差至少改善10%；EMA改善在至少2/3 seed同方向。门槛失败则直接否决这条反馈路线，避免把白噪声放进PID。
- 本审计预计低于1分钟，不使用GPU、不创建W&B run。输出只写`/vepfs`下本地profile；分析工具若新增必须可复用，使用完的`/tmp`预览与回归文件立即清理。

### E94：prequential偏差不可预测——否决自适应PID margin（2026-07-17）

- 审计按E93原门完成，实际运行约8秒，不启动训练或W&B。三seed共75个B40事件；主区间400k--1M含48个事件/45个lag pair，稳健区间600k--1M含33个事件/30个pair。每个事件恰含两个rollout且`pid_update_due`总数为1，聚合时序验证通过。
- 主区间去除seed均值后的lag-1相关为`-0.2083`，低于`>0.25`门；连续偏差同号`25/45=55.56%`，单侧二项`p=0.2757`，同时失败于65%和p<0.10。last-value MAE=`0.08125`，比永远预测零偏差的`0.05981`恶化35.8%；因果EMA(0.2) MAE=`0.05664`，只改善5.31%，低于10%门。
- 逐seed EMA相对零偏差的改善为seed0/1/2=`+4.33%/+18.08%/-3.01%`；只有“2/3方向为正”通过，幅度与相关门均失败。到600k--1M，pooled lag相关进一步变为`-0.3403`、同号率`13/30=43.33%`、`p=0.8192`，EMA从改善转为恶化3.64%；seed0/2分别恶化16.17%/4.73%。因此结果不是早期冷启动拖累。
- 三seed成熟期mean underestimation仍均为正：`0.03594/0.04014/0.02085`。这说明存在总体低估倾向，但事件级误差快速换符号且常呈反相关；“平均需要安全余量”不能推出“用上一事件偏差动态调下一target”。P-M9已经证明固定更大余量会显著损失reward，自适应反馈又没有可预测输入，故两条都不继续。
- 预注册四项只通过seed方向一致性，overall严格失败。停止“校准偏差EMA→PID target”，不扫描tau/gain/clip，不把负相关事后改造成反向controller。下一路线必须不依赖上一批偏差预测，优先比较受控单次联合更新与跨rollout critic validation；需先核对既有target-KL、preupdate/target/crossfit消融，避免重复。
- 新增可复用`audit_prequential_calibration.py`，支持列映射、event cadence、严格因果last/EMA、去seed均值lag相关、符号二项检验、机械gate与逐seed图。正式文件在`_runs/profiles/dqc_pm8_prequential_calibration_audit_2026-07-17/`：`event_series.csv`、`audit_summary.json`和`calibration_audit.png`（`3105×2269`，PIL与目视检查通过）。

### E95：P-M8 PID窗口离线重放——window100是最小低噪声候选（2026-07-17）

- 发现的尺度问题是：P-M8每个同频事件一次加入B40，而`window=50`只保留1.25个事件、每次替换80%窗口；P-M3的B20+window50保留2.5个事件。当前Kp=1因此几乎直接响应最近B40的二项噪声。该事实不表示窗口越长越好，必须同时量化滞后。
- 新工具用每个B20的整数outage计数精确重建window100/200，并逐式重放`target=.15,Kp=1,Ki=.1,leak=.97,deadband=.02,delta_max=.05,reference=10`。用正式logged window50重放三seed lambda，最大绝对误差仅`1.44e-8`，验证控制公式与事件边界完全一致。
- 400k--1M的seed0/1/2中，window100相对logged window50使控制概率mean absolute jump下降`37.72%/43.33%/36.25%`，lambda jump下降`36.84%/35.64%/38.20%`；用当前窗口预测下一B40 raw outage的MAE变化为`+5.13%/+4.76%/-5.77%`，每个seed都低于探索性10%滞后上限。
- window200的概率/lambda jump下降更大，但压力seed0的next-B40 MAE恶化`15.75%`，超过上限；它在高风险下降后仍保持更大lambda，额外迟滞清晰可见。因此不扫更长窗口，选择所有seed都通过且最小的window100。
- 这是固定已观察轨迹上的信号重放，不是换窗口后的policy反事实，不能据此宣称reward/outage会改善。新增可复用`audit_pid_window_replay.py`，正式CSV/JSON/图在`_runs/profiles/dqc_pm8_pid_window_replay_2026-07-17/`；`pid_window_replay.png`为`3105×2269`，PIL与目视检查通过。探索性选择规则和局限均写入JSON，没有伪装成性能预注册。

### E96：P-M10 window100完整1M预注册（2026-07-17）

- P-M10回到P-M8 seed0，只把`pid_window_episodes=50→100`。保持名义alpha=.20、PID target=.15、Kp1/Ki.1/leak.97/deadband.02、Actor/PID interval2、B20、QR32/C20/MC/time-weight.995、T1 sigmoid、LSTM512、8 PPO epoch、LR、seed与1M总预算完全不变；P-M9的target=.10不参与。
- 不用300k早停；窗口改变controller相位，P-M7/P-M8已证明中后段可反转。只有NaN/OOM/确定性错误提前终止。内置140 screen为reward≥.60、outage≤.40；通过才做统一fresh520。
- fresh520强门：reward≥.75、outage≤.22，且相对P-M8 seed0的`170/520=.32692`至少下降.08；hard/smooth CDF、mean-cost error和Brier Skill不得出现类似P-M9的数量级发散。机制门为400k--1M的window probability或lambda mean absolute jump相对P-M8 seed0至少下降20%，并报告新增滞后。
- seed0通过才原样扩seed1/2；失败则window100保留为“信号减振但无性能收益”消融，不扫描75/125/150/200。若只通过机制门且fresh接近安全门，仍不移动门槛，先检查是否为窗口滞后造成reward或风险代价。
- 独占A100预计纯训练6.5--8分钟、140评估约1分钟、条件fresh520约4--5分钟，总墙钟12--14分钟。正式job只用`launch_background.sh`持久化，W&B使用脱敏online；run name/tags/config只含公开算法语义，checkpoint与绝对路径不上传。


### E97：P-M10结果——控制信号显著减振，但以可重复的reward损失换取安全（2026-07-17）

- 正式持久化job正常`exit 0`，绑定提交`08a806a`；W&B online run为`p5tv7sij`，50行history完整到`1,000,000`步，纯训练`396.77s`。远端name/group/tags只含算法公开语义；115个公开config键中未发现绝对路径、机器/源码/Git字段或token/secret/password/credential类字段。评估使用`wandb_mode=disabled`，没有为纯评估创建远端run。
- 单变量因果检查成立：P-M8和P-M10从初始化到`420k`的全部训练输出一致，首次差异出现在`440k`同一reward下的lambda `0.0691→0.0138`。如果在300k或400k裁决，本实验会被错误地报告为“没有作用”；窗口属于慢闭环变量，完整1M是必要的。
- 五个固定200k阶段中，P-M8→P-M10的`reward/outage/lambda`均值依次为：`0.0256/0.005/0→0.0256/0.005/0`、`0.3825/0.100/0→0.3825/0.100/0`、`0.6763/0.255/0.1555→0.7969/0.275/0.1087`、`0.8263/0.205/0.1253→0.7290/0.290/0.3393`、`0.9294/0.335/0.3731→0.7088/0.205/0.3455`。window100在400--600k先提高reward，随后更早维持安全惩罚并落入低reward、低outage轨迹；它不是简单让所有阶段都变慢。
- 预注册机制门明确通过。400k--1M的实际P-M10相对实际P-M8 seed0，window probability mean absolute jump由`0.07600→0.03267`，下降`57.02%`；lambda jump由`0.09026→0.04286`，下降`52.51%`。预测下一B40 raw outage的MAE还由`0.09100→0.05333`，改善`41.39%`，没有出现离线window200那种明显迟滞。logged window100按正式leaky-PI公式重放lambda的最大误差仅`8.41e-9`。
- 1M内部140回合screen为`reward/outage=0.71810/29÷140=0.20714`，通过宽松的`.60/.40`门；随后持久化fresh520评估正常`exit 0`。正式结果为`reward=0.73771±0.43746`、`outage=104÷520=0.20000`，reward均值95%区间`[0.70011,0.77531]`，outage Wilson区间`[0.16788,0.23652]`。点估计恰好满足名义alpha=.20和工程门.22，但reward比预注册`.75`下限低`.01229`。
- 相对P-M8 seed0 fresh520，reward差为`-0.15820`，Welch 95%区间`[-0.20883,-0.10758]`；outage差为`-0.12692`，Newcombe 95%区间`[-0.17937,-0.07355]`。两项交换都远大于评估抽样误差：窗口平滑确实让策略更安全，也确实损失reward，不能称为免费稳定性提升。
- critic没有发生P-M9那种数量级崩溃，但也没有变成可靠条件风险模型。hard-CDF error `0.07885→0.07506`改善`4.80%`，smooth error `0.07630→0.07713`恶化`1.08%`，mean-cost error `1.5165→1.7297`恶化`14.06%`；raw Brier因outage基率下降而改善`25.57%`，但Brier Skill从`-3.44%→-5.89%`，反而下降`2.45`个百分点。crossing由`0.07463→0.06408`略降，仍不能替代校准证据。
- 严格裁决：机制门、安全门和相对outage改善门通过，校准未发散，但reward门失败；因此不扩seed1/2、不扫描window75/125/150/200，也不因reward区间包含.75而改用区间上界。P-M10被保留为“有效控制减振组件/安全--reward消融”，不是当前主配置。它说明controller噪声是问题的一部分，但平滑scalar lambda不能修复BSS持续为负的action-conditioned risk方向。
- 正式history位于`_runs/wandb_export/dqc_pm10_pid_window100_seed0_1m_2026-07-17/`；机制审计位于`_runs/profiles/dqc_pm10_pid_window100_mechanism_seed0_2026-07-17/`；P-M8/P-M10训练和fresh520比较位于`_runs/profiles/dqc_pm8_pm10_pid_window_seed0_2026-07-17/`。`comparison.png/overview.png/pid_window_replay.png`尺寸分别为`2775×1895/2880×4128/3105×773`，均由PIL解码；fresh比较图已目视检查，标签、区间和alpha参考线正常。

### E98：P-M10终点pre-update轻量诊断预注册（2026-07-17）

- P-M10 fresh520的reward只比门低`.01229`，但末200k训练reward长期均值也只有`.70883`。为区分“最后一次B40联合更新造成终点跳变”与“window100在后400k已形成持续低reward路径”，只对`rollout_step001000000.pt`做同一seed、同一当前evaluator的140回合eval-only诊断；不新增训练，不修改正式E97失败裁决。
- 只比较pre/post的reward、outage、CDF/Brier/mean-cost和crossing。若pre相对post的reward至少高`.08`且outage不恶化超过`.03`，则把“限制单次Actor/critic联合移动”列为下一候选；否则认为主要是跨多个控制事件的路径/风险表示问题，优先保守action-risk不确定性或跨rollout critic validation。无论结果如何，本诊断不触发seed扩展、window扫描或事后checkpoint择优fresh520。
- 预计140回合约1--2分钟，使用`launch_background.sh`持久化、`wandb_mode=disabled`，结果与checkpoint均留在`/vepfs`。


### E99：P-M10 pre/post诊断——最终更新不是低reward主因（2026-07-17）

- eval-only持久化job正常`exit 0`。同一140回合协议下，pre/post reward为`0.68265→0.71810`，差`+0.03545`且Welch 95%区间`[-0.06096,+0.13185]`；outage为`16/140=.11429→29/140=.20714`，差`+.09286`且Newcombe区间`[+.00641,+.17874]`。
- E98的触发条件要求pre reward至少高`.08`且outage最多恶化`.03`；实际pre reward更低，所以严格失败。最后一次更新提高点估计reward并显著增加风险，不能解释P-M10相对P-M8的`.158` reward损失；低reward来自后400k多个控制事件形成的路径。
- pre/post hard/smooth CDF error为`.02991/.02740→.06808/.07245`，mean-cost error `.45062→3.10250`，raw Brier `.10038→.17252`，Brier Skill `+.84%→-5.05%`。最后一次更新仍使critic跨分布校准明显变差，但简单回滚会得到更低reward；不做事后fresh520或checkpoint择优。
- 正式CSV/JSON/图位于`_runs/profiles/dqc_pm10_pre_post_seed0_1m_2026-07-17/`，比较图为`2775×1895`且PIL解码通过。

### E100：C-H1W time-weighted cost-LSTM冻结600k预注册（2026-07-17）

- 旧C-H1只在100k、uniform-transition cost loss下测试。随后C-W1证明每条T1000轨迹中999个后期低cost-to-go样本会主导均匀loss，独立LSTM甚至更容易牺牲s0来降低总体loss；因此旧负结果不能回答“risk-discount正确训练测度下history是否有效”。
- 结构动机仍成立：行为policy是MLP+LSTM，未来动作分布依赖hidden；raw cost critic只估计`Z_c(s,a)`，而正确条件量一般是`Z_c(s,h,a)`。现有`cost_history_mode=cost_lstm`已使用与QCPO_refs同协议的`[observation,previous_cost]→MLP`，再拼`previous_action/previous_reward→LSTM512`，最后仍接当前action输出QR32，不会退化成state-only head。
- 首轮只做冻结policy校准，不进入PID闭环。源策略固定为P-M3 seed1 1M final，rollout seed101；对照是既有raw-QR run`8ry7xn6g`。候选仅设`cost_history_mode=cost_lstm`，保持30×B20×T1000=600k、MC、risk-discount `.995`、QR32、20 critic updates、LR、TBPTT seq100、源actor与observation statistics全部冻结。构造后重置host/CUDA RNG，30批reward/outage/cost必须与raw对照逐值exact。
- 在正式训练前增加只读评估诊断：旧hard-Brier字段保持兼容，另报smooth-Brier、各自Brier Skill、ROC-AUC、outage-vs-safe预测均值差和预测概率标准差。它们不参与loss、RNG、actor或checkpoint选择；目的是区分“总体CDF均值碰巧准确”和“逐轨迹风险排序真正改善”。
- raw基线末5批pre-CDF error/Brier为`.10656/.23253`，post为`.03750/.20023`，pre predicted/true mean为`13.2968/13.1100`。候选600k screen要求：末5批pre的CDF error至少改善20%或Brier至少改善10%，另一项不得恶化10%，mean-cost absolute bias不超过1.0；独立140条的hard或smooth Brier至少改善10%且AUC至少提高.03，或Brier改善20%且AUC不得下降超过.02，mean-cost error≤0.90、crossing≤0.10。
- 只有history screen与独立140同时通过才对raw/cost-LSTM做统一fresh520；fresh门沿用Brier至少改善10%与AUC至少提高.03（或Brier改善20%且AUC不退化）的联合证据。若末5批仍相对前5批改善超过10%且方向一致，可预注册一次1.2M长度审计；否则不扫LSTM hidden、TBPTT或LR，也不进入live P-M8/P-M10。
- 预计单条纯训练7--9分钟、140评估约1--2分钟，总墙钟9--11分钟。正式run使用持久化后台与脱敏W&B online，checkpoint、history和分析产物均写`/vepfs`。

### E101：C-H1W训练前判别指标基线验证（2026-07-17）

- 新增二分类概率指标先通过合成数据边界测试：perfect/constant/inverted预测的ROC-AUC精确为`1/.5/0`，单类别为`NaN`；含重复score的Mann--Whitney平均秩实现与`scipy.stats.rankdata`逐位一致。语法、diff与旧hard-Brier兼容字段检查均通过。
- 随后用持久化eval-only重新评估raw-QR 600k final checkpoint，job正常`exit 0`。reward=`.86576`、outage=`36/140=.25714`、hard/smooth CDF=`.26786/.27331`、mean cost=`11.75714`及crossing=`.05737`均复现旧评估，说明新增只读诊断没有改变策略、环境随机流或既有指标。
- raw critic的hard Brier/AUC/BSS为`.20282/.51976/-6.18%`，smooth为`.20173/.52399/-5.61%`。总体hard CDF只比truth高`.01071`，但逐轨迹排序仅略高于随机且proper score劣于经验基率常数预测；这确认“总体CDF点准确”不能证明actor拿到了可用的action-risk方向。
- 该raw结果固定为C-H1W独立140门的正式对照。下一条训练只加入`cost_history_mode=cost_lstm`，W&B使用脱敏online；若history和独立Brier/AUC联合门失败，则不做fresh520、不扫hidden/TBPTT/LR。

### E102：C-H1W 600k结果——history含排序信号，但当前recurrent critic严重同批过拟合（2026-07-17）

- 正式持久化job绑定提交`62fd422`并正常`exit 0`；W&B online run为`a6w3kp7i`，30行history完整到600k，纯训练`281.35s`。远端117个config键未发现绝对路径或敏感键，name/group/tags只含公开算法语义；完结后文件列表仅`config.yaml/output.log/wandb-summary.json`，无源码、Git、requirements或machine metadata。
- 公平性检查严格通过：30批的env step、iteration、trajectory数、四项reward统计、reward quantile、outage、两种cost mean、逐步reward/cost及budget max/mean/min等16列逐元素exact，最大绝对差均为0。独立140条reward/outage也逐位相同，所以所有critic差异都来自`raw→cost_lstm`，不是policy、环境或RNG漂移。
- 末5批prequential结果明确失败。raw→cost-LSTM的pre-CDF error为`.10656→.14031`，恶化31.67%；pre-Brier为`.23253→.32103`，恶化38.06%；聚合mean-cost bias为`.18675→1.03960`，也越过≤1.0门。相反，同一B20训练后的post-CDF error为`.03750→.03625`，只改善3.33%，post-Brier却从`.20023→.05465`，表面改善72.71%。这组pre/post反差直接证明LSTM在当前批上拟合很好、到下一独立批泛化很差。
- 优化诊断支持同一解释：末5批crossing从raw的`.04581`升到`.25516`，gradient clipping fraction从`.20`升到`1.00`。候选前5批到末5批的pre-CDF error改善10.20%，但pre-Brier反而恶化31.84%，方向不一致；因此不满足1.2M长度审计条件，不能把问题归因于单纯训练不足。
- 独立140条显示history并非完全无信息。hard/smooth AUC从`.51976/.52404`升到`.59014/.58894`，绝对增加`.07038/.06490`；但hard/smooth Brier从`.20282/.20174`恶化到`.28636/.26496`，增加41.19%/31.34%，BSS从`-6.18%/-5.61%`降到`-49.91%/-38.71%`。mean-cost error由`.77573→1.17272`，crossing由`.05737→.26751`。排序略有提升不能抵消概率校准、均值和单调性的全面失败。
- E100的history门、hard/smooth独立联合门、mean error门、crossing门及持续改善门全部失败。严格裁决为：不做fresh520、不跑1.2M、不进入live PID，也不盲扫hidden/TBPTT/LR。旧C-H1的“history无用”结论需修正为：history可能补充policy hidden信息，但现有`B20×20 update`的recurrent优化无法泛化。
- 下一步先做零训练开销工程审计：比较raw/LSTM的pre-clip grad norm、loss、参数量、TBPTT状态边界和输入尺度。若没有接线bug，再从三条分歧路线中各做单变量轻量验证：①降低recurrent critic LR/控制有效步长；②用跨rollout replay或留出批做early-stop，直接约束prequential泛化；③增加独立trajectory数而不增加同批update。不能直接把cost-LSTM装进live网络，也不能用同批post-Brier选择checkpoint。
- 正式表、机械门和图位于`_runs/profiles/dqc_frozen_raw_vs_ch1w_costlstm_600k_2026-07-17/`：`gate_summary.csv`、`gate_decision.json`、`paired_history.csv`、`history_generalization.png`及独立评估比较图；完整W&B导出位于`_runs/wandb_export/dqc_frozen_ch1w_costlstm_qr32_600k_2026-07-17/`。

### E103：20次critic更新诊断修正与零行为影响回归（2026-07-17）

- 代码审计发现旧`critic/grad_clip_fraction`并不是一次rollout内全部critic update的裁剪比例：`critic_info`在循环中被覆盖，所以该字段只代表最后一次update。此前“cost-LSTM末5批每次都裁剪”仍成立，因为末次均为1；但raw的20%只能解释为“5个rollout中有1个末次被裁剪”，不能解释成20次内部update的真实比例。
- 现在保留旧字段语义以兼容历史，并新增每个rollout的`update_count`、全update真实裁剪率、cost/joint gradient norm的first/mean/max/last、reward gradient mean，以及cost QR loss的first/mean/min/last。新增逻辑只复制已经计算出的Python标量，不增加forward/backward、不访问RNG、不改变optimizer或scheduler。
- 补丁前后使用同一P-M3策略、seed303、B2、2批、每批3次critic update做持久化严格回归。两个final checkpoint的全部tensor逐元素相同，tensor差异数为0；除checkpoint目录外所有非tensor状态相同；独立4条评估的全部语义指标相同。补丁后job正常exit 0，证明这是纯诊断改动。
- 该诊断直接回答用户提出的“10/20次更新是否关键”：actor的多次更新确实需要旧策略log-prob和PPO ratio修正；critic反复使用同批数据不产生策略off-policy ratio问题，但会产生统计过拟合。新字段用于区分“首步梯度很大后快速收敛”与“20步始终撞clip”，不再用最后一次冒充全过程。

### E104：C-H0.5W固定actor历史特征600k预注册（2026-07-17）

- C-H1W把独立cost-LSTM从约86k参数的raw cost path扩大到约2.60M参数，约30.1倍，并在每个B20上重复优化20次。其独立AUC从约0.52升到0.59说明history有信号，但post-Brier大幅改善而pre/独立Brier恶化，最符合大容量encoder记忆当前20条轨迹。
- 下一候选不再训练独立cost-history encoder，而是使用`cost_history_mode=actor_feature`：复用冻结成熟policy实际用于产生动作的LSTM hidden作为历史表示，再由action-conditioned cost head估计分布。它比独立cost-LSTM少约2.39M个可训练encoder参数，仍保留DQCAC必须的action输入；也更接近QCPO_refs的共享MLP+LSTM backbone，但此冻结筛选不会让cost梯度回灌policy，因果更干净。
- 正式配置与raw/C-H1W严格相同：P-M3 seed1成熟策略、rollout seed101、B20、T1000、30批=600k、MC、risk-discount=.995、QR32、C20、chunk2500、冻结policy与observation统计；唯一算法变量为`cost_history_mode=actor_feature`。30批reward/cost/outage等truth必须与两条对照逐值exact。
- 预注册主门：末5批prequential CDF absolute error或Brier相对raw至少改善20%，另一项不得恶化超过10%，聚合mean-cost bias绝对值不高于1；独立140条要求hard或smooth Brier至少改善10%且对应AUC至少增加0.03，或Brier改善20%且AUC下降不超过0.02，同时mean-cost error不高于0.90、crossing不高于0.10。只有通过才做fresh520并考虑live。
- 该路线的判别重点不是同批post loss。若AUC保留C-H1W至少一半增益且Brier回到不差于raw，说明“固定共享history+action head”能抑制encoder过拟合；若仍失败，下一项优先把独立cost-LSTM的C20降到C5，直接检验重复更新强度。跨rollout replay/held-out early stop与B40/B80增加独立轨迹保留为独立消融，不能和C5一次混合。
- 预计纯训练5--8分钟、独立140条评估约1分钟；只用`launch_background.sh`持久化，W&B online名称、group和tags仅包含算法语义，不包含绝对路径或隐私字段。600k失败且末段无一致改善时不延长到1.2M。

### E105：C-H0.5W结果——共享actor hidden降低容量，但没有保留cost排序信号（2026-07-17）

- 正式job绑定提交`39bd4a8`并正常`exit 0`，W&B run为`yx3hfl78`；600k纯训练`239.7s`，完整30行history与140条独立评估均已完成。公开117个config键无secret/path值，上传文件仅`config.yaml/output.log/wandb-summary.json`。
- raw、cost-LSTM、actor-feature三条的16个行为/truth字段逐值exact，最大差0；独立reward/outage也同为`.865756/.257143`。因此差异只来自cost表示。actor-feature可训练cost head为206,112参数，移除了独立cost-LSTM的2,393,600参数encoder，但仍大于raw head的86,304参数。
- 末5批pre-CDF error从raw的`.10656`降到`.09469`，只改善11.14%，未过20%门；pre-Brier从`.23253`升到`.24038`，恶化3.38%，聚合mean-cost bias为`1.44556`，越过≤1门。post-CDF/Brier也相对raw恶化18.33%/2.54%，没有“同批和下一批同时改善”的证据。
- 独立hard CDF为`.24643`，与truth`.25714`的absolute error仍是`.010714`，恰好与raw从另一侧得到的`.010714`相同。smooth CDF均值误差改善约70%，但hard/smooth Brier只改善0.56%/0.34%，AUC反而`.51976/.52404→.51709/.51709`。mean-cost error`.77573→1.52765`，crossing`.05737→.12189`。总体均值碰巧更近不能替代逐状态排序与proper score。
- 新诊断揭示旧日志严重低估C20优化压力：actor-feature全程/末5批真实20-update裁剪率为`71.17%/74%`，而旧“最后一次是否clip”为`26.67%/20%`。末5批cost grad first/mean/last=`31.69/16.50/9.56`，QR loss在20步内平均只降约17.9%；多数step仍超过clip=10，只是最后一步常回到阈值下。
- 裁决：所有主门失败，不做fresh520、不跑1.2M、不进live。固定actor hidden由policy/reward目标学习，未保留独立cost-LSTM得到的AUC信号；单纯共享backbone不是自动的公平性收益。下一条只把独立cost-LSTM的C20改为C5，直接验证同批重复更新过强，不给actor-feature继续扫LR或head宽度。

### E106：C-H2U5独立cost-LSTM、5次update冻结600k预注册（2026-07-17）

- C-H1W的独立AUC增益约`.07`证明可训练cost history encoder学到了信号，但C20使post-Brier异常低、pre/独立Brier恶化；C-H0.5W去掉encoder后Brier回到raw附近，却同时失去AUC增益。最小可证伪假设是：需要cost专用history encoder，但每个B20做20次step过强。
- C-H2U5复用C-H1W全部配置，只将`updates_per_episode=20→5`。P-M3 seed1成熟策略、rollout seed101、B20×30=600k、T1000、MC、risk-discount=.995、QR32、cost-LSTM、LR、chunk2500、policy/obs冻结均不变；30批行为truth必须与三条既有对照exact。
- 主门不因观察到C20失败而放宽：末5批pre-CDF error至少改善20%或Brier至少改善10%，另一项不得恶化10%，mean bias≤1；独立hard或smooth Brier至少改善10%且AUC至少比raw提高.03，或Brier改善20%且AUC下降不超过.02，同时mean error≤.90、crossing≤.10。只有通过才做fresh520/live。
- 机制辅助门：相对C-H1W C20，独立Brier至少改善20%，并保留至少一半AUC增益（hard AUC≥`.55495`）；pre与独立方向必须一致。只改善同批post、只降低最后一步grad或只改善总体CDF均值均不算成功。
- 这不是actor的off-policy修正实验：policy完全冻结；critic对固定监督标签重复拟合无需importance ratio，改变C20→C5是在控制sample reuse与泛化。预计纯训练1.5--2.5分钟、140评估约1分钟，总墙钟3--5分钟；仍用持久化后台和脱敏W&B online。失败后不扫C2/C10，转向跨rollout replay/held-out validation或增加独立trajectory数。

### E107：C-H2U5结果——减少update消除了AUC信号，却没有解决高梯度（2026-07-17）

- 正式job正常`exit 0`，W&B run为`ocpqyr9x`；纯训练`243.5s`，只比C20的`281.35s`快13.45%，说明主要耗时在trajectory/recurrent编码与特征刷新，不在额外15个QR step。远端117个config键和3个文件通过隐私审计。
- 16个behavior/truth字段与raw/C20逐值exact，最大差0。末5批pre-CDF/Brier为C5=`.10750/.23932`，raw=`.10656/.23253`，分别恶化0.88%/2.92%；aggregate mean bias为`-2.60855`。post-CDF/Brier=`.09031/.22743`，也比raw恶化140.83%/13.58%。
- 全程/末5批真实5-update裁剪率为98.67%/100%；末5批cost grad first/mean/last=`47.21/43.15/35.84`，5次update后仍远高于clip=10。QR loss last/first=`.9647`，即只下降3.53%。C5不是稳定正则化，而是让大recurrent模型长期处于裁剪后的欠拟合状态。
- 独立hard/smooth Brier为`.20986/.20904`，虽比C20改善26.7%/21.1%，却比raw恶化3.47%/3.62%；hard/smooth AUC从C20的`.590/.589`跌到`.393/.385`，低于随机。CDF absolute error`.02679`是raw的2.5倍，mean error`1.53895`，crossing`.13733`。辅助门的Brier部分通过，但“保留一半AUC增益”彻底失败。
- 裁决：不做fresh520/live，不扫C2/C10。强更新是cost-LSTM学到history排序所必需，但B20+C20又把概率尺度和单调性拟合坏；下一步应提高每个optimizer step的独立轨迹数，而不是继续在同一B20上找epoch甜点。

### E108：C-H3B40独立cost-LSTM、B40+C20冻结600k预注册（2026-07-17）

- 候选将`num_envs=20→40`、`num_iterations=30→15`，总环境步仍严格600k；保留cost-LSTM、C20、MC、time-weight=.995、QR32、LR和冻结policy。每个样本仍被学习20次，但每次梯度看到40条而非20条独立轨迹；总transition-epoch相同，Adam step数从600降到300，位于C5的150与原C20的600之间。
- B40改变env stream分组，训练history不要求逐行exact；固定策略的独立140条reward/outage必须与既有对照exact。B40只有15批，末段用最后3批（120条轨迹）与B20末5批（100条）比较，避免直接拿5×40扩大统计窗口。
- 主门保持：末3批pre-CDF error至少比raw末5批改善20%或Brier改善10%，另一项不恶化10%，mean bias≤1；独立Brier至少比raw改善10%且AUC提高≥.03，或Brier改善20%且AUC不明显下降，同时mean error≤.90、crossing≤.10。
- 机制门相对C20：独立Brier至少改善20%、hard AUC≥`.55495`、pre与独立方向一致；新增head/encoder分支梯度和真实20-update裁剪率用于判断大batch是否降低encoder噪声。只有主门通过才fresh520/live。
- 实测B20任务仅占约1.3GB/80GB A100显存，chunk2500继续限制QR峰值，B40硬件余量充足。预计纯训练3.5--5分钟、独立评估约1分钟，总墙钟5--7分钟，持久化后台与脱敏W&B online。
- 若B40只过机制门、且AUC/Brier方向一致但单一mean/crossing门略失，可把B80作为另行预注册消融；若AUC未达`.55495`或Brier相对C20未改善20%，停止batch scaling，转encoder专用低LR/正则或跨rollout held-out validation，不事后启动B80。

### E109：cost head/history encoder梯度拆分及零影响回归（2026-07-17）

- 新增当前update及rollout内first/mean/max/last两组裁剪前范数：206,112参数cost quantile head与2,393,600参数history encoder。总cost/joint范数、旧末次clip和真实全update clip保持原义；新增计算只读取已有`.grad`，不参与clip或optimizer。
- 使用同一policy、seed304、B2、2批×3次update的cost-LSTM任务做补丁前后持久化回归，两个final checkpoint全部tensor逐元素相同，差异数0；除checkpoint目录外非tensor状态相同，独立4条评估语义字段相同，两个job均`exit 0`。
- 因此B40可以安全回答高梯度主要来自head还是encoder。若encoder占主导，后续优先独立encoder LR/正则；若head占主导，优先修正QR尺度/输出结构。不能继续用合并norm猜测。

### E110：B40/C20标准化裁决——批次内改善没有迁移到独立轨迹（2026-07-17）

- 正式冻结策略run为W&B ahmg534z，配置为cost-LSTM、B40、C20、QR32、600k、seed101；15个rollout批次完整、exit 0，纯训练274.7s。相同B20/C20 cost-LSTM需要281.4s，因此在本环境上把num_envs从20增到40几乎没有增加墙钟；显存约1.95GB/80GB，硬件完全可承受B40。
- 行为控制仍然严格成立：与raw/B20、cost-LSTM/B20及actor-feature/B20的reward、outage、cost与动作统计逐批对应，独立评估行为的reward/outage也完全相同。网络结果不是策略或环境随机路径差异造成的。
- 末3个B40批次共120条轨迹，更新前CDF绝对误差为0.08620，相对raw末5个B20批次共100条的0.10656改善19.11%；更新前Brier为0.22336，只改善3.94%；mean-cost bias为0.9201。这解释了训练过程中出现的局部正信号，但未达到预注册CDF至少20%或Brier至少10%的主门。
- 同批更新后Brier降至0.09650，而更新前下一批仍为0.22336；quantile crossing为0.21559，20/20次内部更新在末3批均触发clip。head/history encoder的平均pre-clip norm约为21.55/29.10，两支都大，不支持“只降低history encoder学习率就能解决”的单一解释。
- B40原训练进程以40个并行环境评估时，num_eval=140被向上取整为160条，不能与既有140条直接比较。随后从同一final checkpoint以B20、严格140条、eval-only重新评估；reward/outage与raw完全相同，分别为0.86576/0.25714。
- 严格140条上，raw→B40 cost-LSTM的hard Brier为0.20282→0.29003，恶化43.00%；smooth Brier恶化40.97%；hard/smooth AUC为0.51976/0.52404→0.38074/0.37901，下降0.1390/0.1450，已经不是“接近随机”，而是风险排序方向明显反转。mean-cost绝对误差0.77573→1.15108，恶化48.39%；crossing 0.05737→0.19240，增加235.34%。
- 相对B20/C20 cost-LSTM，B40的独立hard Brier还恶化1.28%，hard AUC下降0.2094。故增加独立trajectory虽改善了训练末段聚合CDF，却没有修复跨轨迹条件风险泛化；按预注册机制门停止B80，不做fresh520、live或1.2M。
- 当前最重要结论是：history确实含有风险信息，但“独立2.39M recurrent encoder + 同一批QR目标重复强优化”没有稳定把信息转成校准概率。C20/B20得到AUC约0.59但概率失准，C5欠拟合且AUC约0.39，C20/B40同样发生排序反转。下一阶段不再扫batch/epoch，而优先验证优化耦合和跨批泛化机制。

### E111：五候选统一证据包、W&B隐私与下一路线（2026-07-17）

- 统一数据位于_runs/profiles/dqc_frozen_history_modes_600k_2026-07-17/：independent_eval140_comparison.csv、prequential_tail_comparison.csv、history_curves.csv、decision.json和六面板comparison.png。图已解码并目视核验；所有候选都是同一冻结policy、seed101、600k及同一140条独立行为。
- W&B run ahmg534z远端状态为finished，15行history完整；公开config共117个键，没有token/secret/password/credential/api_key键，也没有绝对路径值。审计保存在同目录wandb_privacy_audit.json。
- 下一条最高价值工程/算法假设不是继续扩大网络，而是解除reward critic与cost critic的共同optimizer/共同global clip。当前cost-LSTM的大梯度会通过联合clip缩放reward critic；在冻结policy筛选中这不会改变行为，却会在live训练里直接损害reward value/GAE和actor更新。应先做纯等价回归，再做单变量“reward/cost独立optimizer与独立clip”机制实验。
- 该改动本身不能保证解决冻结cost校准，所以同时保留两条可区分路线：一是跨rollout held-out/early-stop或小型replay，直接抑制同批记忆；二是共享actor backbone但增加cost辅助监督/小adapter，使history既含cost信号又不新增2.39M自由参数。两条不能与optimizer拆分混跑，分别作为消融。
- 评估器还需修复“向量环境按batch向上取整episode数”的口径问题：未来应只聚合前num_eval条，确保B20/B40/B80都严格评估同样数量。该工程修复先做随机流/指标回归，不能让此次160条结果反向参与模型选择。

### E112：精确评估回合数修复与optimizer作用链纠正（2026-07-17）

- 根因已确认：统一MLP评估、QCPO recurrent、DQCAC recurrent、QCPO_refs以及随机calibration都用ceil(num_eval/num_envs)运行完整向量批，却直接聚合全部样本。B40请求140会得到160；DQCAC还把额外s0/a0一起送进cost critic，所以CDF、Brier、AUC和mean也会改变，不只是显示的episode数错误。
- 五条路径现在先校验num_episodes为正，仍让最后一个向量批的每条episode跑满horizon，再统一只保留前E条reward/cost/初始状态/预测概率。这样不提前终止任何环境，不改变被保留轨迹的随机流，同时使num_envs不再静默改变评估样本量。
- 两个持久化eval-only回归均exit 0。B20/E140整除回归的30个eval字段与修复前JSON逐项exact，差异数0；B40/E140现在严格返回140条，其中47条outage，证明非整除截断生效。语法、diff-check和PNG/JSON输出均正常。
- 此修复同时纠正上一条对“拆分reward/cost optimizer”的优先级判断。当前主配置mlp_lstm+gae_ppo的reward-V是actor内部共享value head，GAE由rollout保存的actor_value构造，policy/value由actor_optimizer一次联合更新。critic_optimizer中的reward_critic只服务旧distributional actor路径和诊断；cost gradient的joint clip不会直接缩放当前reward-V或PPO actor。
- 因此不为当前主线实现或运行“独立reward/cost optimizer”1M实验。它对旧distributional模式可能有意义，但没有解释当前recurrent GAE/PPO性能的作用链。保留这一分歧路线作为旧模式消融，不把审计前的假设升级成算法结论。
- 重新审计QCPO_refs后，最直接而尚未移植的稳定组件是：共享policy/reward/cost history backbone、非负c_dist=exp(linear)、cost quantile loss之外的mean-cost MSE锚定（系数0.5），以及Weibull tail loss。下一项先单变量加入默认关闭的mean anchor；它直接针对cost-LSTM“有少量AUC信号但均值/概率尺度失准”，比拆optimizer或事后温度校准更有作用链证据。
- 用现有140条充分统计计算的同样本乐观单调仿射校准上界显示，B20/C20 cost-LSTM即使事后最优缩放，Brier Skill也只有约+0.62%；raw约+0.15%。所以当前问题不只是一个全局temperature/bias，不能靠Platt式校准替代表示与训练目标改进。

### E113：QCPO_refs mean-cost anchor实现与冻结600k预注册（2026-07-17）

- QCPO_refs源码的cost目标已逐式核对：`loss += 0.5 × [0.5(mean(c_dist)-c_return)^2] + quantile_huber_mean + weibull_tail`。当前DQCAC的QR对`N_target`求和而reference对pairwise全部取mean，因此不能直接把0.5加到现有loss；新`cost_mean_anchor_coef=.5`使用有效scale `coef×N_target×quantile_target_scale`，在QR32/legacy_sum下为16，在N64/reference_mean/ref32下仍为16，保持mean/QR相对权重而不让N成为隐含学习率。
- 新参数默认0，接入recurrent TBPTT、整批、transition chunk与crossfit四条cost路径；mean目标使用与QR相同的risk-discount sample weight，query-mixture时使用相同CDF quadrature，IQN uniform tau仍用样本平均。日志新增enabled/configured/effective scale、原始/缩放mean loss及rollout内first/mean/min/last，同时把mean项纳入`cost_objective_loss`。没有混入exp非负输出、Weibull或共享backbone。
- 工程回归用同一P-M3 seed1 policy、rollout seed305、B2×2批×3次critic update。改动前后两个持久化job均exit 0；60个checkpoint tensor leaves逐位完全相同，tensor差异0，全部eval字段与共享summary字段差异0，新增summary只有coef/scale。enabled smoke同配置设0.5后scale严格为16，40个cost/history tensor发生非零变化，所有checkpoint tensor与eval数值有限。解析full/chunk检查loss差`5.12e-9`、gradient最大差0。
- 正式C-H4M复用C-H1W的成熟冻结policy、rollout seed101、B20×30×T1000=600k、C20、MC、risk-discount=.995、QR32、cost-LSTM、chunk2500及140条独立评估；唯一算法变量是`cost_mean_anchor_coef=0.5`。30批behavior/truth必须与raw和C-H1W的16个控制字段逐值exact，独立reward/outage也必须exact。
- 主门不移动：末5批prequential CDF error或Brier相对raw至少改善20%/10%之一，另一项不得恶化10%，聚合mean bias绝对值≤1；独立hard或smooth Brier相对raw至少改善10%且对应AUC提高≥.03，或Brier改善20%且AUC下降≤.02，同时mean-cost error≤.90、crossing≤.10。机制门相对C-H1W要求Brier改善≥20%、保留至少一半AUC增益即hard AUC≥.55495、mean error≤.90且pre/独立方向一致。
- 只有主门通过才做fresh520并考虑live；若只改善mean却丢失AUC，说明anchor把critic收缩为总体均值，不能驱动action选择；若AUC保留但Brier/mean仍不过门，说明QCPO_refs的稳定性还依赖非负输出、Weibull或共享多任务表示。失败后不立即扫描anchor系数，下一候选分别是softplus/exp非负输出、Weibull tail或小型共享多任务backbone，保持单变量。
- 既有同配置无anchor纯训练281.4秒；新项只做quantile mean和标量MSE，预计纯训练4.5--5.5分钟、140评估约1分钟，总墙钟6--7分钟。正式任务只用`launch_background.sh`持久化，W&B online名称/group/tags只含公开算法语义，路径与敏感配置继续过滤。

### E114：cost单位换算审计、错误尺度主动早停与C-H4M修订（2026-07-17）

- E113只对齐了pairwise归约，却漏掉QCPO_refs在`process_returns`开头执行的`cost /= cost_scale`，其正式`cost_scale=10`。reference的QR在大残差区随cost线性缩放，mean MSE则二次缩放；DQCAC保持raw-cost QR时，要匹配reference的mean/QR相对权重，有效系数应为`coef×N_target×target_scale/cost_scale`，不是只乘N。QR32、coef=.5、S=10的正确scale为1.6。
- 第一条正式run `tk2m896n`按旧scale16运行到16批/320k时被中途审计识别为失真：末5批mean-anchor原始loss均值49.59，乘16后约793.37，而QR均值75.93，anchor约强10.45倍；20/20次update裁剪率为100%，crossing约0.275。13个behavior/truth控制字段与C-H1W前16批仍逐值exact，说明不是轨迹漂移。该run随即向训练子进程发送SIGTERM，launcher记录exit143；它被标记为无效尺度工程run，不进入模型比较，也不补剩余280k。
- 新增`cost_mean_anchor_cost_scale`，默认10且必须为正；W&B/summary显式记录configured coefficient、cost unit scale和effective scale。默认关闭的修正前后回归仍有60个tensor leaves逐位exact、eval差异0、共享summary差异0；唯一新summary键是cost scale。corrected enabled smoke得到coef=.5/S=10/effective=1.6，40个cost/history tensor非零变化且全部有限。
- C-H4M的其他预注册条件与门槛不变，但正式单变量现在明确定义为`cost_mean_anchor_coef=.5,cost_mean_anchor_cost_scale=10`。这不是事后调参：10来自QCPO_refs源代码的固定单位换算，1.6由归约和单位解析推导，不由320k效果选择。重跑仍从头使用seed101和同一成熟冻结policy，不能从失真run续训。

### E115：C-H4M2 600k结果——mean校准几乎修复，但条件风险仍不过门（2026-07-17）

- corrected正式run绑定提交`adaac26`，W&B `w7n417ze`、30批/600k完整、exit0，纯训练282.8秒；effective anchor scale始终1.6，anchor/QR比例全程均值0.716、末5批0.674，已回到同量级。远端状态finished、30行history、119个公开config键；无敏感键或绝对路径值，只同步3个标准文件，隐私审计通过。
- 公平性控制严格通过：与raw和无anchor cost-LSTM的env step、iteration、trajectory、四项reward、经验outage、两种cost mean、两种action统计、三个budget统计及reward quantile共16列逐元素exact，所有最大差为0；独立140条reward/outage也同为0.865756/0.257143。
- mean anchor直接目标取得了巨大收益。独立predicted/truth mean cost为11.7227/11.7571，绝对误差0.03444；raw为0.77573、无anchor cost-LSTM为1.17272，因此分别改善95.56%和97.06%。这证明QCPO_refs的mean MSE不是装饰，确实能修复distribution整体位置。
- 但末5批prequential全面失败。raw→anchor的pre-CDF error为0.10656→0.15844，恶化48.68%；pre-Brier 0.23253→0.33247，恶化42.98%；聚合mean bias绝对值0.18675→1.80078。post-Brier仍能到0.08312，明显好于raw的0.20023，却差于无anchor LSTM的0.05465；同批拟合好、下一批失真的核心问题没有解决。
- 独立140条上，hard CDF absolute error为0.01585，略差于raw 0.01071、但比无anchor LSTM 0.06942改善77.17%。hard Brier为0.25822，相对raw恶化27.32%，相对无anchor只改善9.83%，未过20%机制门；hard AUC 0.56143相对raw增加0.04167，保留了超过一半history排序增益。crossing 0.19885虽比无anchor 0.26751下降25.67%，仍远高于≤0.10门和raw 0.05737。
- 因此主门、独立门、机制门全部失败，不做fresh520、不进入live、不扫描anchor系数。结论不是“mean anchor无用”，而是它主要校准无条件一阶矩，无法单独约束逐状态概率、quantile形状和跨rollout泛化。QCPO_refs的稳定性更可能来自mean、非负输出、Weibull tail与共享多任务history的组合。
- 正式证据包位于`_runs/profiles/dqc_frozen_ch4m2_meananchor05_scale10_600k_2026-07-17/`，包含独立/末段CSV、完整history、decision、W&B隐私审计和2682×1507对比图；PNG可由PIL完整解码。下一单变量优先审计并移植QCPO_refs的`c_dist=exp(linear)`非负输出；softplus稳定替代、Weibull tail和共享backbone作为分歧路线分别记录，不能与首轮正输出混改。

### E116：QCPO_refs非负cost输出审计、实现与正式实验选择（2026-07-17）

- 源码单位已逐式核对。QCPO_refs在process_returns开头执行cost /= cost_scale，正式cost_scale=10，cost_limit也在初始化时除以10；模型随后计算c_dist=torch.exp(self.constraint(fc_x))。因此当前始终使用raw cost与raw budget的DQCAC不能直接照抄exp(logit)，物理等价形式是10×exp(logit)。零logit由此预测raw cost 10，接近DynamicButton当前约11--12的真实均值；不乘10则初始化只有1，既不等价也会重新制造严重低估。
- 新增默认关闭的cost_quantile_output=linear|exp|softplus和cost_quantile_output_scale=10。linear直接返回同一个Tensor对象；exp使用S×exp(raw)；softplus使用S×softplus(raw)/log(2)，使零logit也严格映射到S，只改变正值映射的尾部梯度。适配器位于DQCAC层，不包装或改写公共critic，因而不改变网络参数名、初始化、optimizer或旧checkpoint。
- 全链路搜索后，online/target/crossfit、QR/IQN训练、cost-LSTM TBPTT、recent-s0、preupdate漂移、actor概率、critic-Adam dual、初始CDF日志和独立评估全部经过统一_cost_quantiles()；静态搜索已确认没有遗留直接self.cost_critic(...)、self.cost_target_critic(...)或self.cost_crossfit_critic(...)前向。W&B/profile/summary显式记录输出模式与scale。
- 纯张量门通过：linear返回对象identity；exp与softplus的零logit输出都严格为10，所有输出为正，测试区间内前向和反向均有限。默认linear使用同一P-M3成熟策略、seed305、B2×2批×3次update的持久化回归，训练12.56秒、exit0；相对补丁前linear checkpoint，60个tensor leaves逐位相同、最大差0，全部eval字段和共享summary字段差异0，新增summary仅为模式/scale。这证明默认路径没有被适配器调用层次改变。
- mean-anchor=.5/S=10的exp与softplus均按同一4k协议持久化运行，训练12.75/12.62秒、exit0；三条linear/exp/softplus的冻结策略reward、outage、cost及4条终评行为完全exact，三个checkpoint都无NaN/Inf。相对linear-anchor，exp与softplus各有40个cost/history tensor发生非零变化，证明新机制实际进入反向而不是只改日志。
- 4条终评只作工程门，不能作为算法性能证据。透明报告：linear/exp/softplus的predicted mean为1.776/10.794/10.274，mean absolute error为16.474/7.456/7.976；hard CDF error为.500/.375/.484，hard Brier为.500/.349/.485，crossing为.532/.492/.452。exp在这个极小样本上没有数值异常且概率指标优于softplus，并且它是QCPO_refs源实现的精确单位映射，因此正式候选选exp；softplus只保留为exp出现溢出/持续极端梯度时的预注册替代，不把4条偶然AUC当作模型选择证据。
- 正式C-H5E复用C-H4M2的成熟冻结policy、rollout seed101、B20×30×T1000=600k、C20、MC、risk-discount=.995、QR32、cost-LSTM、mean-anchor .5/S=10、chunk2500与独立140条评估；唯一变量是cost_quantile_output: linear→exp，scale10来自QCPO_refs单位而非调参。30批behavior/truth的16个控制字段及独立reward/outage必须与raw、C-H1W、C-H4M2逐值exact。
- 晋级门预先固定：所有checkpoint/history有限；末5批prequential CDF error或Brier相对C-H4M2至少改善20%，另一项不得恶化10%；独立hard Brier须从.25822至少改善20%到≤.20658且AUC不低于.55，mean error保持≤.20，crossing降到≤.10。同时记录20-update裁剪率与anchor/QR比；若长期100% clip且独立门失败，不用“exp来自reference”豁免。通过才做fresh520；失败则不扫output scale，因为10由单位固定。
- 既有同配置C-H4M2纯训练282.8秒；exp仅增加逐元素指数，预计纯训练4.7--5.5分钟、140评估和W&B收尾约1--2分钟，总墙钟6--8分钟。使用launch_background.sh持久化与脱敏W&B online。分歧路线保留为：exp数值失败时跑softplus；exp稳定但Brier/crossing失败时转Weibull tail或共享多任务backbone；二者不与首个exp正式run混改。

### E117：C-H5E 600k结果——exp保留少量排序信号，但放大梯度并破坏mean/shape（2026-07-17）

- 正式job绑定提交a15d0c9，W&B run为wkn3kdn7；30批/600k完整、后台exit0，纯训练282.4秒，与linear mean-anchor的282.8秒相同。checkpoint全部tensor有限，说明10×exp没有发生溢出。W&B远端finished，30行history、121个公开config键，无敏感键和绝对路径值。
- 公平性控制严格通过：相对C-H4M2，env steps、iteration、trajectory、四项reward、outage、两项cost、两项action与三个budget共15列逐元素exact，最大差0；独立140条reward/outage也同为0.865756/0.257143。差异可以归因于linear→exp输出参数化。
- 独立140条hard CDF/Brier/AUC为0.20067/0.23193/0.56944，truth为0.25714；smooth Brier/AUC为0.22455/0.56891。相对linear mean-anchor，hard Brier改善10.18%，AUC增加0.0080，但未达到Brier改善20%的机制门；相对raw，hard Brier仍恶化14.35%，AUC增加0.04968。
- exp破坏了mean-anchor最重要的正收益。predicted/truth mean为10.49095/11.75714，绝对误差1.26619；linear mean-anchor只有0.03444，误差扩大约36.76倍。独立CDF absolute error从0.01585升到0.05647，恶化256.34%；crossing从0.19885升到0.30599，恶化53.88%，远高于0.10门。
- 末5批prequential没有改善：pre-CDF error为0.17344，相对linear anchor的0.15844恶化9.47%；pre-Brier为0.33271，几乎不变但略恶化0.07%；signed mean bias为-2.2923，绝对值相对1.8008恶化27.29%。post-Brier也从0.08312恶化到0.10198。exp没有关闭“本批拟合、下一批失真”的泛化缺口。
- 全程和末5批20/20 update均触发clip。末5批cost/head/history的平均裁剪前范数从linear的88.70/63.19/59.50升到exp的144.44/126.77/65.20；总cost增加62.85%，head约翻倍，符合指数导数放大head梯度的作用链。update-mean口径的scaled-anchor/QR比全程0.678、末5批0.655，与linear的0.716/0.674接近，因此失败不是mean项重新压倒QR。
- 预注册门中只有AUC≥0.55通过；末段pre、独立Brier、mean与crossing门全部失败。因此不做fresh520、不进入live、不扫描output scale；10是QCPO_refs单位，不是自由超参。exp本身数值稳定，所以也不触发“指数溢出才运行softplus”的分支，避免用一个4条smoke更差的近似机制继续消耗600k。
- 结论：非负cost输出不是QCPO_refs稳定性的充分来源。它保留甚至略增强history排序，但无法把排序变成校准概率，还通过指数几何加重head梯度和quantile crossing。下一路线应改变representation/多任务约束，而不是继续改输出激活：优先审计QCPO_refs共享policy/reward/cost MLP+LSTM的真实梯度耦合；Weibull tail作为独立辅助路线保留，但必须明确其quantile detach后主要通过共享backbone间接起作用。
- 统一证据位于_runs/profiles/dqc_frozen_ch5e_meananchor_exp10_600k_2026-07-17/：四候选独立与末段CSV、linear/exp完整history、decision、W&B隐私审计和2755×1557对比图。PNG已由PIL完整解码；环境view_image因系统bwrap故障不可用，不影响数值与文件完整性。

### E118：C-H6 QCPO_refs式共享多任务backbone实现与工程验证（2026-07-17）

- 源码审计确认QCPO_refs不是“policy LSTM旁边再放一个cost-LSTM”：其policy、reward-V、cost quantiles和Weibull头共用同一个`[observation,previous_cost]→MLP[512,512]→concat(previous_action,previous_reward)→LSTM512`表示，并在同一个PPO loss中联合反传。当前DQCAC的`actor_feature`只把rollout feature detach后交给cost head，`cost_lstm`则额外训练约2.39M参数的独立encoder；两条都没有让cost监督训练policy/reward共享表示。
- 新增默认关闭的`cost_shared_backbone_coef`、`cost_shared_backbone_cost_scale=10`和`cost_shared_backbone_huber_kappa=1`。首轮只允许recurrent GAE-PPO、MC、uniform QR、linear output、online query、`cost_history_mode=actor_feature`和`actor_update_interval=1`，显式拒绝IQN、local grid、direct CDF、target/crossfit、独立cost-LSTM、recent-s0及冻结policy组合，保证正式差异只来自cost监督是否进入共享表示。
- DQCAC不能照搬QCPO_refs的state-only cost head，否则`P(C≥b|history,a)`对action失去条件，risk advantage会退化。当前实现保留action-conditioned cost head；actor epoch用同一次recurrent forward取得policy、reward value和可微feature，固定cost head参数，只把QCPO_refs尺度的QR/mean辅助梯度回传到actor body/LSTM。cost head仍只属于critic Adam，actor Adam仍只拥有actor参数。
- 共享辅助QR先把prediction和MC cost都除以10，Huber阈值独立使用QCPO_refs的1，再对transition和quantile取mean；mean项为`cost_mean_anchor_coef×0.5(mean(Q)/10-C/10)^2`。主cost critic继续使用已验证的raw-cost、`huber_kappa=.1`目标，没有因共享消融暗改head loss。可选risk-discount transition权重与主cost critic相同。
- 每次critic head更新前用当前actor和behavior chunk入口h0/c0重算detach feature；actor step后标记dirty。actor不再变化的剩余critic epochs直接复用当前feature，因此C20/A8每rollout约做9次而不是20次额外LSTM刷新。post holdout位于observation RMS合并前，并保证查询最后一次actor step后的当前feature。
- 默认关闭回归通过：复跑`dqc_cost_output_linear_reg_4k_20260717`同一seed305金样本，补丁前后60个checkpoint tensor leaves逐位相同，差异数0、最大绝对误差0；评估语义字段一致，只有新增summary配置、路径和墙钟字段不同。纯手算测试中共享loss与QCPO_refs式公式逐位相同，feature梯度范数`0.0026525`非零，所有cost-head参数梯度均为None。
- 第一次共享smoke在构造期因“全局huber_kappa必须为1”的过严校验主动退出1，没有开始训练。审计发现主DQCAC正式κ为0.1，而共享辅助函数本来已经固定按参考κ=1计算；随后把共享κ改为独立显式参数并移除会暗改主critic的要求。该失败job只属工程校验，不作为算法结果。
- B2、T32、C2/A2 tiny共享smoke正常exit0、训练8.8秒。coef0配对首epoch ratio最大误差为`5.9128e-5`，coef1为`6.0558e-5`，差仅`1.43e-6`；这是逐步LSTM采样与chunk批量重算的既有浮点基线，不是共享机制导致的off-policy概率错误。候选cost-head梯度存在标志严格为0，body/LSTM梯度非零。
- 全尺寸`[512,512]+LSTM512`、B2×T1000×2=4k、C4/A2持久化smoke正常exit0，纯训练12.5秒；43个module tensor leaves全部有限。首epoch ratio误差`9.06e-6`、末epoch KL`3.58e-4`、clip fraction`.003`；最后共享QR/mean/weighted loss为`7.33e-6/3.50e-7/7.51e-6`，body/LSTM联合梯度范数`.1152/.02747`，cost-head梯度标志仍为0。短随机policy几乎无cost，loss数值只作链路证据，不作性能判断。

### E119：C-H6L 共享backbone成对1M live预注册（2026-07-17）

- 首轮直接使用压力seed1跑完整1M，不用100k/300k reward早停。共享representation、cost critic和PID都有慢变量，已有IQN与B40实验证明300k可能误杀，P-M1/C-X3也证明600k可能误判为成功；除NaN/Inf、梯度所有权断言、显存错误或明确发散外，两条都跑满1M并保留每100k checkpoint。
- baseline与candidate共同使用P-M3的B20、T1000、C20/A8、N32、MC、risk-discount=.995、GAE-PPO、observation RMS、LSTM512、sigmoid CDF T1、经验outage PI、PID target=.15、sum normalization和beta=.995。两条都改为`cost_history_mode=actor_feature,cost_mean_anchor_coef=.5,cost_mean_anchor_cost_scale=10`；唯一变量是`cost_shared_backbone_coef=0→1`。因此它回答“在同一个action-conditioned history head上，cost监督进入共享表示是否有益”，既有P-M3 raw三seed作为外部性能语境，不充当唯一因果对照。
- 工程门：初始rollout的reward/cost/outage与动作统计必须配对一致；候选所有checkpoint有限、cost-head actor-gradient标志恒0、共享body/LSTM梯度非零、首epoch ratio误差不高于coef0基线加`1e-3`；C20/A8正常阶段每rollout共享feature实际刷新应约为9次，而不是20次。
- 机制门：候选最后5个rollout的prequential Brier或CDF absolute error相对baseline至少改善10%，另一项不得恶化超过10%；mean-cost error和crossing同时报告，不能用同批post loss代替跨批泛化。共享weighted loss、policy/value loss、body/LSTM/head梯度、KL与clip分phase报告，用于判断coef1是有效监督、过弱还是淹没PPO。
- 性能门不依赖内置140条单点。两条final checkpoint都做相同动作RNG/并行布局的fresh520：候选若outage降到`≤.22`且reward`≥.75`即通过；或相对paired baseline把outage降低至少`.05`且reward下降不超过10%；若baseline本身已安全，则候选需保持`≤.22`并把reward提高至少10%。同时与P-M3 seed1的`reward/outage=.8622/.3058`透明比较。
- 若seed1通过，原参数扩seed0/2各1M+fresh520，不回调coef；至少2/3 seed通过且聚合outage≤.2后才进入2M/5M与QCPO_refs公平终局。若coef1出现明确辅助梯度过强且KL/clip/body norm同步放大，下一条只降coef；若梯度相对PPO近乎为零且表示/CDF无变化，才升coef。不能同时改Weibull、output activation、PID或quantile grid追逐偶然点。
- P-M3单run纯训练约6.5分钟。actor-feature baseline预计7～9分钟；共享candidate因每rollout约9次额外recurrent刷新和8次轻量cost forward，预计10～15分钟。两条并行使用40个CPU worker且显存远低于A100 80GB，预计训练墙钟12～18分钟；内置140评估后，两条fresh520预计再需约5～8分钟并行。全部通过`launch_background.sh`持久化，W&B online名称/tag只含公开算法语义，不上传绝对路径或敏感配置。

### E120：C-H6L 成对1M结果——reward略升，但安全与条件风险校准显著恶化（2026-07-17）

- 共享实现及默认关闭回归提交为`53c9bb3`。两条正式任务均由`launch_background.sh`以`nohup+setsid`持久化运行并正常`exit 0`；coef0/coef1的W&B run分别为`vvfok94n/j69x59um`，各有50行完整history到1,000,000步，远端均为finished。两条从启动到训练、内置128条评估和W&B收尾约12--13分钟；并发时各约2.1GB显存，A100 80GB余量充分。
- 工程门全部通过。coef1全部50批的共享weighted loss非零，全程body/LSTM联合梯度均非零，末5批均值为`.16010/.04852`；actor backward中的cost-head梯度标志恒为0。每批feature刷新计数严格为9，actor epoch看到的表示漂移全程均值`.00619`且50批均非零。首epoch PPO ratio最大误差的全程最大值为coef0/coef1=`2.96e-5/2.31e-5`，没有新增off-policy概率错误；末10批KL/clip约为`.00220/.1138`与`.00252/.1241`，数值稳定且没有NaN/Inf。
- 训练曲线不是单调优势。约260k时共享版reward/outage约`.332/.10`，比基线`.622/.40`更保守；440k时共享版reward反超；550k附近又落后；末10批训练reward均值为coef0/coef1=`.74880/.78160`，共享版只高4.38%，对应训练outage为`.18/.17`。这一反复交叉证明100k/300k早停会误判该组合，但跑满1M也没有产生稳定的大幅训练优势。
- 机制门失败。末5批prequential Brier由`.17623`改善到`.15576`，相对改善11.61%；但CDF absolute error由`.07250`恶化到`.08688`，相对恶化19.83%，越过“另一项不得恶化10%”门。mean-cost bias从`+.8175`变为`-2.6717`，绝对误差扩大约227%；crossing只从`.18355`小幅变为`.18581`。因此不能用Brier单项改善宣称共享表示更准。
- 修复后的评估器严格聚合请求的512条，不再按B40向上取整到520。final checkpoint的fresh512给出：coef0 reward/outage=`.99587/.22656`，coef1=`1.06788/.29492`。共享版reward增加`.07201`即7.23%，但outage增加`.06836`即6.84个百分点/相对30.17%；没有达到reward至少+10%，也没有达到outage≤`.22`，反而离约束更远。两个final checkpoint均独立保存，路径分别为`_runs/checkpoints/dqc_ch6l_actorfeature_meananchor_coef0_b20_1m_s1_20260717/final_post_update.pt`和`_runs/checkpoints/dqc_ch6l_sharedcoef1_meananchor_b20_1m_s1_20260717/final_post_update.pt`。
- 独立条件风险指标同时恶化：hard Brier `.18022→.23635`（+31.15%），hard AUC `.57114→.54220`（-0.02894）；smooth Brier `.18036→.23688`（+31.34%），smooth AUC `.57254→.53936`（-0.03318）。唯一明确的shape收益是crossing `.21585→.16274`（-24.61%），但它没有转化为概率校准或安全收益。predicted mean cost只从`8.9468→9.0573`，而真实mean cost从`9.5195→11.5918`，共享critic对更危险策略仍明显低估。
- 内置128条评估曾给出相反的outage排序（coef0/coef1=`.281/.242`），而fresh512变为`.227/.295`。这不是checkpoint变化，而是小评估样本高方差的直接证据；模型选择必须以预注册的较大fresh评估为准，不能挑选内置128条更好看的结果。训练入口本轮漏传CLI `--tag`，导致两个同seed顶层`DQCAC_DynamicButton_run_s1.json`发生覆盖；训练checkpoint和W&B未丢失，fresh评估已用唯一tag分别落盘。以后正式训练命令必须同时设置CLI `--tag`和W&B name，不能把job name误当结果tag。
- 严格裁决：C-H6L不通过机制门和性能门，不扩seed0/2，不作为默认配置，也不通过调小/调大coef追逐单seed。当前证据最符合“在线action-conditioned QR辅助梯度与reward/PPO对共享history发生负迁移”：它能改变表示、略推高reward并减少crossing，却破坏对未见轨迹的风险排序/概率尺度，使PID拿到更危险的策略。
- 证据位于`_runs/wandb_export/dqc_shared_backbone_pair_1m_s1_2026-07-17/`和`_runs/profiles/dqc_shared_backbone_pair_1m_s1_2026-07-17/`；后者包含`metric_profile.csv/profile.json/report.md/overview.png`。两份fresh512 JSON分别为`_runs/DQCAC_DynamicButton_ch6l_coef0_b20_1m_s1_final_eval520_20260717_s1.json`和`_runs/DQCAC_DynamicButton_ch6l_sharedcoef1_b20_1m_s1_final_eval520_20260717_s1.json`；文件名沿用预注册的eval520标签，但JSON内`num_episodes=512`是修复后的真实精确口径。
- 下一步不继续给同一个actor backbone增加QR权重。优先路线是把“cost信息进入policy history”和“高方差QR直接扰动PPO”分开：①默认保留actor-feature detach head，用跨rollout replay/held-out选择抑制C20同批过拟合；②加入小型cost adapter或对共享cost梯度做PCGrad/正交投影，仅在与PPO/value不冲突时更新共享层；③先验证QCPO_refs的Weibull tail是否提供比32点QR更低方差的共享监督。三条作为独立消融，先做冻结/短程机制门，不同时混入PID、quantile grid或output activation。

### E121：C-H6L coef0三种子——平均安全改善但reward方差扩大（2026-07-17）

- seed1 coef0不是共享梯度候选，却相对旧P-M3 seed1同时把fresh reward/outage从`.86220/.30577`改善到`.99587/.22656`。因此补跑seed0/2，验证`cost_history_mode=actor_feature + mean-anchor=.5/scale10`是否能成为独立的新主线；其它B20/T1000/C20/A8、LSTM512、GAE-PPO、obs RMS、T1、PID target=.15及1M预算保持不变。
- 第一次seed0/2启动把CLI `--tag`错误插入`--set`列表，两条均在argparse阶段立即exit2，没有构造环境、没有checkpoint、没有W&B run也没有训练。修正argv边界后复用job名从头启动；正式W&B run为seed0 `yx08vswe`、seed2 `u75gmeuf`，与seed1 `vvfok94n`共同组成150行完整history，三条均正常exit0。
- 精确fresh512分别为seed0/1/2 reward=`.90480/.99587/.62779`，outage=`.21484/.22656/.16602`；三seed mean±sample-SD为reward `.84282±.19171`、outage `.20247±.03211`。seed0/2通过`≤.22`工程门，seed1只高`.00656`；三seed平均仅高名义alpha `.00247`，但reward的seed2低谷使其尚不稳定。
- 旧P-M3三seed fresh520为reward `.81574±.08468`、outage `.22821±.07072`。新组合平均reward增加`.02708`即3.32%，outage绝对下降`.02573`即相对11.28%，安全seed方差下降约55%；但reward标准差从`.0847`扩大到`.1917`。结论是“有用的平均安全/校准改进”，不是“稳定性能提升”。
- 新组合独立hard Brier/AUC三seed聚合为`.16794±.02029/.57857±.00749`，smooth为`.16772±.02028/.57982±.00776`；AUC跨seed稳定高于随机，优于此前独立cost-LSTM的方向反转。predicted/true mean cost聚合为`8.3019/8.9896`，crossing `.20189±.01735`，说明actor-feature+mean anchor改善了条件排序稳定性，但shape单调性仍差。
- 训练末10批聚合reward/outage为`.76151±.15349/.20667±.07848`。seed0末段为高reward高风险，seed2为低reward安全，训练中多次风险周期；fresh结果不是简单由最后一个B20决定。故不直接跑5M，也不声称超过QCPO_refs；优先把已验证的P-M8 Actor/PID同频组件与更好的cost表示组合，检验是否减少闭环分叉。
- 正式W&B/profile位于`_runs/wandb_export/dqc_actorfeature_meananchor_coef0_1m_multiseed_2026-07-17/`与`_runs/profiles/dqc_actorfeature_meananchor_coef0_1m_multiseed_2026-07-17/`。seed0/2 fresh JSON使用明确`eval512`标签；seed1旧文件名保留`eval520`预注册字符串，但JSON内部精确`num_episodes=512`。

### E122：C-H7C 更好cost表示叠加Actor/PID同频cadence预注册（2026-07-17）

- C-H7C以E121 coef0为基线，只把`actor_update_interval=1→2`和`pid_update_interval=1→2`作为一个已经在P-M8独立验证过的“同频控制组件”加入。critic仍每个B20立即C20；Actor连续收集两批同一behavior共40条轨迹后做8次PPO，经验PID也在相同B40边界用40条真实MC cost更新一次，随后该lambda立即供Actor使用。
- 这不是盲目叠加两个失败trick。E121的actor-feature+mean anchor使三seed outage均值降到`.2025`且AUC稳定约`.579`，但reward seed方差变大；P-M8证明同频cadence能减少成熟期outage/lambda振荡，却因raw critic多seed平均outage`.25`失败。两者分别针对cost表示与controller时序，存在互补作用链。
- 首轮同时跑两个压力端：seed0代表E121的高reward/较高风险端，seed2代表低reward/安全端；每条完整1M，不以300k早停。相对各自E121 final，seed0要求fresh512 outage≤`.22`且reward不低于`.8143`（不损失超过10%）；seed2要求outage≤`.22`且reward≥`.75`。两条同时过门才补seed1，不因其中一条漂亮而扩5M。
- 机制门：25个Actor事件与25个PID事件一一对应，每次Actor batch=40条；首epochratio误差≤`1e-3`，无NaN/Inf；末200k outage或lambda标准差相对E121同seed下降至少20%，或至少不再出现更大的闭环周期。Brier/AUC不能相对E121恶化10%以上，mean/crossing透明报告。
- 若seed0安全但seed2仍低reward，说明cadence不能解决策略盆地/初始化方差，停止该组合并转cost-gradient conflict/adapter或held-out replay；若seed2恢复reward但seed0不安全，说明主要缺口仍是risk guard/PID而非表示。只有两条同时通过才运行seed1并以三seedmean±SD裁决。
- 正式job为`DQCAC_DynamicButton_ch7c_actorfeature_meananchor_cadence2_b20_1m_s0/s2`，W&B online名称、group和tags只含公开算法语义，CLI tag与checkpoint目录均唯一。两条使用`launch_background.sh`持久化；预计并行训练加内置评估约10--14分钟，条件fresh512再约7--9分钟。

### E123：C-H7C构造保护与actor-feature cadence适配（2026-07-17）

- 两条正式任务第一次启动后均在agent构造阶段主动`exit 1`：旧安全检查只允许`actor_update_interval>1`与`MC/raw`组合，seed0/2分别创建了W&B失败run `cndfioye/2k0olnwh`，但没有创建环境训练批、checkpoint或任何性能数据。该事件是工程接线缺失，不是算法负结果，也没有消耗1M训练预算。
- 审计确认多rollout cadence已有states/actions/old-log-prob/GAE和预抽baseline action的time-major合并，但没有携带`actor_feature`。新适配只在`cost_history_mode=actor_feature`时把每条rollout保存的detached `[T*B,H]`恢复为`[T,B,H]`，沿环境维拼接后重新flatten；同时校验shape、feature宽度和`requires_grad=False`。若直接沿flatten维拼接，会把第二条rollout的`t0`放到第一条rollout的`t1`之后，与merged state/action错位，因此不能只放宽构造检查。
- 纯张量测试用`T=2,B=2`验证合并顺序精确为`t0:[rollout1 envs, rollout2 envs], t1:[...]`，`states`与`cost_feature`逐位对齐；默认raw分支不生成feature字段，detach防护也按预期拒绝错误输入。源码语法和`git diff --check`通过，默认raw执行路径没有改变。
- 持久化W&B-disabled完整smoke使用B2、T32、4 rollout、C2/A2、Actor/PID interval2并正常`exit 0`，训练9.5秒。最终checkpoint报告actor/PID事件均为2、每次均合并4条轨迹、缓存余数均为0；首epoch ratio最大误差`9.54e-7`、末KL `3.83e-5`、clip fraction 0，无NaN/Inf。它证明behavior probability、feature对齐与同频事件接线正常，但256环境步不作为性能证据。
- 正式seed0/2将使用唯一`v2` job/name/tag从头重跑，配置与E122完全相同；以既有实测预计每条1M训练加内置评估约7--10分钟，随后条件fresh512约4--5分钟。失败的远端run保留为可审计工程记录，不与正式结果合并。

### E124：C-H7C完整结果——seed0减振但丢reward，seed2校准与性能同时失败（2026-07-17）

- 修复后两条正式job均正常`exit 0`，seed0/2纯训练分别`746.87/740.45s`；W&B run为`ra0sd0yn/5ff083c7`，各50行完整到1M且远端finished。122个config键没有绝对路径或敏感字段，name/group/tags仅含公开算法语义；每条远端只上传`config.yaml/output.log/wandb-summary.json`。
- 机制接线全部通过：seed0/2均为25个Actor事件和25个PID事件，每次40条轨迹；全程首epoch ratio误差最大`2.00e-5/2.28e-5`，无NaN/Inf。相对E121同seed，seed0末200k outage std和mean-absolute-jump下降`32.19%/34.48%`，lambda jump下降`42.35%`；seed2 outage std只降`10.27%`且lambda std增加`94.13%`，没有稳定减振。
- fresh512 seed0为`reward/outage=.68557/.20313`，相对E121 `.90480/.21484`少`.21924` reward、只少`.01172` outage；虽Brier `.17908→.16265`、AUC `.57845→.62603`、crossing `.20735→.13180`，仍未过reward≥`.8143`门。seed2为`.42735/.23828`，相对E121 `.62779/.16602`少`.20045` reward且outage多`.07227`；Brier `.14452→.28361`恶化96.24%，AUC `.58611→.57605`。
- 两个压力seed的性能门没有任何一个通过，因此严格不补seed1、不扩2M/5M。Actor/PID同频cadence可以在部分seed平滑控制，却会改变闭环盆地，不能作为actor-feature+mean-anchor的免费稳定组件。
- 完整history/profile在`_runs/wandb_export/dqc_ch7c_actorfeature_meananchor_cadence2_1m_stressseeds_2026-07-17/`与同名`_runs/profiles/`；E121配对表、裁决JSON和目视检查通过的8面板图在profile的`e121_comparison/`子目录。

### E125：C-H7C pre/post与actor×critic四格诊断——终点爆炸来自最后一批critic过拟合（2026-07-17）

- 同一fresh512协议下，seed0 pre→post为`reward/outage=.63021/.16211→.68557/.20313`；seed2为`.51750/.25781→.42735/.23828`。最后一次联合更新使seed2 reward下降`.09015`、outage改善`.01953`，但pre本身仍远未过`.75/.22`门，所以简单回滚不是解决方案。
- seed2最后训练B20的cost均值约`20.2`。pre→post critic预测均值`10.06→19.77`，fresh真实cost却`11.18→9.73`；Brier `.19738→.28361`。为分解actor hidden变化和critic变化，额外构造只交换module state的两个临时hybrid，并保持相同512条评估随机协议。
- 四格结果非常明确：pre actor+pre critic预测mean/Brier=`10.06/.19738`；final actor+pre critic=`10.03/.18254`；pre actor+final critic=`19.93/.28496`；final actor+final critic=`19.77/.28361`。固定critic换actor几乎不动预测，固定actor换final critic立即翻倍；主因是B20×C20把最后一批高cost标签写进head，而不是actor feature坐标漂移。
- 因此下一主实验不启用feature refresh与cadence2叠加，而回到E121 interval1并单独测试`num_envs=40`：同一1M预算下每次critic看到40条独立轨迹、总iteration从50降到25。若仍出现同样的pre/post泛化断裂，再实现跨rollout replay或held-out early stop。

### E126：默认关闭的current actor-feature刷新开关已实现并通过机制测试（2026-07-17）

- 新参数`cost_actor_feature_refresh=False`默认逐式保持旧actor-feature结果；True仅在MC/actor-feature/recurrent路径开放。每个PPO optimizer step后把当前critic batch标脏，下一critic step用当前actor和保存的chunk h0/c0重算detach feature；interval2合并dict与当前rollout dict分别处理，cost梯度仍不能进入actor。
- 纯张量测试证明关闭时严格no-op，开启时正确更新feature、drift、dirty与count。B2/T32、C3/A2、interval2持久化smoke正常`exit 0`：2个Actor事件、2个PID事件，每个due batch恰好3次刷新；首epoch ratio误差`9.54e-7`、末KL `7.87e-5`、clip=0，无非有限值。
- hybrid证据说明它是正确性/独立消融，而不是本次C-H7C失败的主要修复；因此暂不和B40混合。若B40解决最后一批过拟合但仍有跨epoch feature失配，再单独打开该开关。

### E127：C-H8B40更多独立trajectory预注册（2026-07-17）

- C-H8B40回到E121 seed2，只改`num_envs=20→40`并把`num_iterations=50→25`，总预算仍为1M。Actor/PID interval均回到1；每次critic C20、actor A8、QR32、MC、risk-discount、actor-feature、mean-anchor、LSTM512、obs RMS、sigmoid T1、PID target=.15和所有LR不变，feature refresh保持False。
- 该变量同时让每个critic optimizer step看到40而非20条独立轨迹，并把每1M的critic/actor事件减半；每条trajectory仍被同一事件的20/8个epoch使用。这是扩大`num_envs`的真实工程语义，不通过C40偷偷增加总sample reuse。
- 首轮只跑压力seed2完整1M，不用300k早停。性能门为fresh512 reward≥`.75`且outage≤`.22`；相对E121 seed2，末5批pre-Brier不得恶化10%，fresh Brier不得恶化10%、AUC不得下降超过`.03`、mean error≤1.0。机制门为25个Actor/PID事件、batch40、ratio≤`1e-3`、无NaN/Inf，并检查final pre/post critic均值是否仍被单批推移超过5。
- seed2通过才扩seed0/1；失败则不扫B60/B80，直接实现跨rollout replay/held-out选择。单条B40预计训练加内置评估约8--13分钟，fresh512约4--5分钟；只用持久化后台和脱敏W&B online。

### E128：C-H8B40 seed2 1M结果——吞吐和reward明显改善，但严格校准门失败（2026-07-17）

- 正式任务绑定提交`82cddce`，由`launch_background.sh`以`nohup+setsid`持久化运行，job正常`exit 0`；W&B run为`brtk50w8`，25行history完整到1,000,000环境步，远端状态finished。纯训练403.94秒；W&B运行口径475.47秒，相比E121/B20 seed2的830.64秒缩短42.76%。B40在当前80GB A100上只提高并行采样量，没有形成显存或墙钟瓶颈。
- 工程门全部通过：25个Actor事件与25个PID事件，每次batch严格40条trajectory；首epoch behavior/current ratio最大误差`1.43e-5`，末次KL/clip为`.00103/.0440`，无NaN/Inf。默认关闭的current actor-feature刷新计数为0，正式变量确实只有B20→B40及相应事件数减半。
- fresh512的E121/B20→C-H8/B40 reward为`.62779→.74496`，绝对增加`.11717`、相对增加18.66%；outage为`.16602→.21484`，增加4.88个百分点但仍低于预注册上限`.22`。reward距离`.75`门只差`.00504`，但门槛不能在看见结果后放宽，所以性能联合门严格失败。
- 条件风险结果有正有负。hard AUC `.58611→.59387`、Brier Skill `-.04384→-.00979`、crossing `.18246→.15732`，说明排序、相对常数基线的概率质量和quantile形状略有改善；但absolute hard Brier `.14452→.17034`，恶化17.86%，超过允许的10%。两条策略的真实outage基率不同，因此Brier与Brier Skill方向不冲突：B40不是完全失去排序，而是绝对概率误差仍较大。
- predicted/true mean cost从`8.739/8.031`变为`7.823/9.254`，绝对误差`.708→1.431`，未过≤1门。末5批pre-Brier只比B20恶化6.58%，pre-CDF error由约`.120→.0913`改善，最后一批post predicted/target mean为`7.668/7.675`；没有再出现C-H7C seed2把fresh predicted mean从约10推到约20的终点爆炸。扩大独立trajectory确实缓解了最后一批记忆，但没有解决下一rollout/fresh策略下的条件校准。
- 严格裁决为`strict_fail_no_seed_expansion`：不补seed0/1、不扫B60/B80，也不因reward置信区间覆盖`.75`而事后改门。B40仍保留为下一机制实验的工程底座，因为它把训练吞吐提高约1.75倍、消除了最严重的末批过拟合，并把压力seed2推到接近目标的reward/outage区域。
- 下一主路线按E127转向默认关闭的跨rollout cost replay或held-out critic selection。首选只缓存上一批detached actor feature、action、time-step和MC cost，与当前B40目标做等权凸组合；它直接把critic监督从“同一40条重复C20次”扩展到两个rollout，同时保持reward critic、PPO、PID和总环境步不变。若一批stale feature的坐标漂移成为问题，再与已经实现的current feature refresh作独立消融，不能同时混入Weibull、IQN或PID调参。
- 证据包位于`_runs/profiles/dqc_ch8b40_actorfeature_meananchor_b40_1m_s2_2026-07-17/e121_comparison/`，包含fresh512对比CSV、训练history、末段汇总、门槛JSON、W&B隐私审计和六面板图。隐私审计覆盖123个公开config键，没有敏感键或绝对路径值；远端仅同步标准W&B配置、输出与summary文件。

### E129：C-H9R跨rollout完整cost replay实现、验证与1M预注册（2026-07-17）

- 新增默认关闭的`cost_transition_replay_batches=0,cost_transition_replay_coef=1`。首轮正式路线只允许MC、actor-feature、QR、uniform grid、linear output、online quantile query、actor interval1、shared coef0、s0 aux0和feature refresh False；它只改变cost head监督，不重放reward、PPO、PID或observation RMS。
- 启用`batches=1,coef=1`后，第一个rollout仍用100% current cost objective；该轮全部critic/actor/log完成后才把detached feature、action、step和MC cost入队。从第二轮起，current/replay各占0.5，两个权重和为1；K批replay先内部求均值再乘统一权重，不随K放大critic learning-rate。
- default-off用既有seed305、4k cost-LSTM金样本回归，补丁前后60个checkpoint tensor leaves逐元素完全相同，差异数0。B2/T32、3 rollout、C3/A2、chunk16启用smoke正常exit0；首次replay在critic learning step3激活，后两轮共6个replay update，每次1批/64 transitions，PPO首epochratio误差`7.43e-5`，44个checkpoint tensor全部有限。
- 纯张量full/chunk对拍使用64条、QR8、risk-discount和mean-anchor：QR/mean/combined/scaled loss最大差`9.54e-7`，cost-head梯度最大差`1.19e-7`。这证明chunk只改变浮点归约顺序，没有按块重复放大replay梯度。
- 第一版故意缓存stale behavior actor feature，而不同时缓存recurrent输入并重算current feature。C-H7C actor×critic四格显示一次actor更新只让固定critic预测约变化0.03，而换最后critic会变化约9.88；因此先隔离“增加跨rollout监督”更有信息量。若本轮校准改善但出现feature drift证据，再把旧批current-feature重算作为独立消融。
- 正式C-H9R以C-H8B40 seed2为逐项基线：B40×25、总1M、C20/A8、QR32、MC、risk-discount=.995、actor-feature、mean-anchor=.5/scale10、LSTM512、obs RMS、sigmoid T1、经验PI/PID target=.15及全部LR不变；唯一变量是transition replay `0→1 batch,coef1`。预计首轮20个critic step无replay，后24轮×20=480个replay update，每次40,000条旧transition；缓存feature约81.9MB，A100 80GB有充分余量。
- 工程门：first active step=20、update events=480、active时samples=40,000/batches=1/current scale=.5/replay scale=.5；ratio≤`1e-3`、无NaN/Inf/OOM。机制门：末5批pre-Brier相对C-H8的约`.15842`改善至少10%，pre-CDF error不得恶化10%；最后一批current/replay target mean及两个loss透明报告，防止只把旧批平均值机械写入head。
- 性能门保持fresh512 reward≥`.75`且outage≤`.22`；hard Brier必须≤E121 seed2的1.10倍即约`.15898`、AUC不得比C-H8 `.59387`下降超过`.03`、mean-cost error≤1。全部通过才扩seed0/1；失败不扫replay coef或B60/B80，优先实现上一rollout held-out选择critic epoch，区分“混合旧标签有用”与“需要显式泛化选择”。
- B40无replay纯训练403.94秒。完整replay额外做一次等规模cost-head前向/反向，预计纯训练9--13分钟、内置128评估约1--2分钟、fresh512约4--5分钟；正式训练只用持久化后台和脱敏W&B online。

### E130：C-H9R完整结果——排序形状局部改善，但概率校准与安全闭环显著恶化（2026-07-17）

- 正式后台job正常`exit 0`，训练run为`7ob6dayd`，25行history完整到1M且远端finished。实际纯训练416.7秒，只比无replay B40的403.94秒慢3.2%，远低于保守估计；W&B运行口径为487.44秒。工程门全部通过：first-active-step=20、后24轮共480次replay update、每次40,000条旧transition、current/replay scale均为0.5；PPO首epochratio最大误差`2.47e-5`，无NaN/Inf/OOM。
- 内置128评估曾给出reward/outage=`.80256/.27344`、hard AUC `.64071`、crossing `.09551`；但fresh512为`.78413/.31641`、AUC `.55918`、crossing `.09684`。小评估集上的排序提升没有在更可靠的512条口径上保持到预注册AUC门，不能据此扩seed。
- 相对C-H8无replay fresh512，reward `.74496→.78413`，差`.03917`，Welch 95%区间`[-.02502,.10336]`覆盖0；outage `.21484→.31641`，增加`.10156`，Newcombe 95%区间`[.04752,.15484]`完全大于0。reward增益没有统计把握，风险恶化却清晰且幅度很大。
- critic不是完全退化：crossing `.15732→.09684`，说明分位数形状更有序；但hard Brier `.17034→.27366`恶化60.66%，Brier Skill `-.00979→-.26525`，hard CDF error `.05212→.11957`增加129.39%。predicted/true mean从`7.823/9.254`变成`17.803/14.025`，绝对误差`1.431→3.778`。replay把旧批高cost监督长期保留后形成保守偏置，不等于校准改善。
- 训练末5批同样提前暴露失败：pre-Brier `.15842→.29304`恶化84.98%，而预注册要求至少改善10%；pre-CDF abs error `.09125→.16547`恶化81.34%，允许上限只有`.10038`。末5批outage `.18→.355`、lambda `.0870→.5616`；最后一批pre predicted/true mean=`18.577/10.550`，post-Brier仍为`.22954`。这不是只在final fresh评估偶然翻车。
- 严格裁决为`strict_fail_no_seed_expansion`：不跑seed0/1，不扫replay coef、buffer深度、B60/B80。简单等权训练replay被否决为主路线；它证明跨批标签能改善quantile crossing，却也证明“降低混合训练loss”不能保证下一policy/新trajectory上的概率校准。
- 下一路线改为上一rollout只做held-out validation，用其选择或早停当前rollout的cost-critic step，而不把旧标签直接写入目标。首版需要保持reward critic继续C20、只冻结cost branch后续梯度，并明确保存/恢复cost参数与Adam状态；否则所谓early stop会暗中改变reward critic或留下optimizer动量污染。小adapter/PCGrad和Weibull仍保留为后续独立消融，不与held-out首轮混合。
- 完整导出在`_runs/wandb_export/dqc_ch9r_transitionreplay1_actorfeature_meananchor_b40_1m_s2_2026-07-17/`，C-H8/C-H9R共同history与曲线在`_runs/wandb_export/dqc_ch8b40_vs_ch9r_replay1_seed2_1m_2026-07-17/`和同名`_runs/profiles/`。fresh表、门槛JSON、末段表、隐私审计及`2775×1895`比较图位于C-H9R profile的`e121_ch8_comparison/`；全部PNG可由PIL解码。125个公开config键无敏感键或绝对路径，远端只同步标准config/output/summary文件。

### E131：C-H10G上一rollout retention holdout guard实现、验证与1M预注册（2026-07-17）

- 新机制默认关闭，参数为`cost_holdout_guard=False, relative_tolerance=.05, absolute_tolerance=.002`。上一rollout只缓存真实s0处的detached actor feature、behavior a0和完整MC cost；它不进入QR/mean backward，不重放actor/PID/reward，也没有第二套importance ratio。准确说它是“对当前批更新的跨rollout保留性验证”，不是从未被历史critic见过的全局holdout。
- 每轮step0先在上一批计算T1 smooth-Brier并保存cost head和该组参数的Adam `step/exp_avg/exp_avg_sq`。每个当前批cost step后重算；若超过`best*(1+.05)+.002`，立即恢复best cost参数/Adam状态，同轮余下C-step仍计算完整cost梯度并参与joint clip，但在Adam前把cost grad设为None。这样reward critic继续完成C20且保持相同joint-clip尺度，被拒绝的cost动量不会暗中推进。
- 默认关闭复跑seed305/4k cost-LSTM金样本，补丁前后60个checkpoint tensor leaves逐元素完全相同，missing/extra/different均为0。启用B2/T32、3 rollout、C3/A2 smoke正常exit0：first active step=3、guard events=2、每次2条holdout、ratio误差`8.63e-5`、44个checkpoint tensor全有限；该零cost短测三步Brier单调改善，因此没有伪造rollback。
- 另用完整T1000、冻结训练好policy、critic lr=.1和零容忍带做强制回滚压力测试，2个guard event均在step2停止并恢复step1。末轮initial/best/selected smooth-Brier=`.021970/2.94e-9/2.94e-9`，`selected/attempted/stop=1/2/2`。纯张量测试进一步证明cost权重与Adam槽精确恢复、reward更新保留；grad=None后cost权重/Adam step不动而reward Adam继续前进。
- 正式C-H10G以C-H8 B40 seed2为唯一基线变量：B40×25、1M、C20/A8、QR32、MC、actor-feature、mean-anchor=.5/scale10、LSTM512、obs RMS、T1、经验PI/PID target=.15及全部LR不变；transition replay保持0，只打开guard与`.05/.002`容忍带。预期first-active-step=20、guard events=24、每次samples=40，首轮仍完整C20。
- 工程门要求events=24、first step=20、至少一次真实restore、ratio≤`1e-3`且无NaN/Inf；透明报告每轮selected/attempted/stop update和cost-update retained fraction，若几乎全选step0则按“critic冻结退化”解释，不包装为early stopping成功。机制门仍要求末5批pre-Brier相对C-H8 `.15842`改善至少10%，pre-CDF error不得恶化10%。
- fresh512性能门不变：reward≥`.75`、outage≤`.22`、hard Brier≤`.15898`、AUC≥`.56387`、mean-cost error≤1。全部通过才补seed0/1；失败不调容忍带、不与replay叠加，下一独立路线为小cost adapter/PCGrad或QCPO_refs式Weibull低方差辅助。预计纯训练7--9分钟、内置128评估1--2分钟、fresh512约4--5分钟；全部使用持久化后台和脱敏W&B online。

### E132：C-H10G完整结果——总体概率几乎校准，但以显著reward损失换取过度安全（2026-07-17）

- 正式持久化job正常`exit 0`，绑定提交`38b2012`；W&B online run为`yd2a4r4x`，25行history完整到1,000,000步且远端finished。纯训练`408.38s`，只比C-H8无guard的`403.94s`慢1.10%；内置128评估为reward/outage=`.62198/.17969`。随后同样通过`launch_background.sh`运行fresh512，耗时225秒并正常`exit 0`，评估阶段关闭W&B，未产生额外远端run。
- 工程门全部通过：first active step=20、guard events=24、每次40条保留轨迹；9/24事件提前停止，21/24事件发生best-state恢复。最终被选择的cost step均值为`6.875/20`，只有3/24选择step0；实际optimizer在停止前平均执行`17.08/20`个cost step。因此机制不是把critic简单冻结。全程PPO首epoch ratio最大误差`1.53e-5`，无NaN/Inf/OOM。
- guard对已见上一批的保持能力有效，但没有通过真正的下一rollout机制门。末5批pre-Brier从C-H8的`.15842`变为`.16919`，恶化6.80%，而预注册要求至少改善10%；pre-CDF abs error从`.09125`降到`.06047`，改善33.73%；pre mean-cost bias均值从`-1.212`变为`-2.774`，低估反而加深。它减少总体概率误差的同时，没有稳定改善完整条件分布。
- fresh512的C-H8→C-H10G为reward `.74496→.57864`、outage `.21484→.14063`。reward差`-.16632`的Welch 95%区间为`[-.22836,-.10428]`；outage差`-.07422`的Newcombe 95%区间为`[-.12079,-.02742]`。两者区间都不跨0：guard带来明确的安全改善，也带来明确且更大的reward代价，不能称为免费性能提升。
- critic总体校准显著改善：hard CDF error `.05212→.00110`（下降97.89%），smooth CDF error `.04935→.00651`（下降86.80%），hard Brier `.17034→.12244`（下降28.12%），predicted/true mean cost由`7.823/9.254`变为`7.979/7.023`，绝对误差`1.431→.956`。crossing `.15732→.12443`也改善。但hard AUC `.59387→.53609`，Brier Skill `-.00979→-.01316`；低Brier主要来自最终策略outage基率更低和总体概率对齐，不代表逐状态排序变强。
- 预注册性能门中，outage、Brier和mean error通过，reward `.57864<.75`与AUC `.53609<.56387`失败；训练末5批Brier机制门也失败。因此严格裁决为`strict_fail_no_seed_expansion`：不补seed0/1，不扫描`.05/.002`容忍带，不与training replay叠加。该机制保留为“灾难性遗忘/安全优先”消融，而不是DQCAC主配置。
- 证据位于`_runs/wandb_export/dqc_ch10g_holdoutguard_actorfeature_meananchor_b40_1m_s2_2026-07-17/`、配对导出和训练曲线`_runs/wandb_export|profiles/dqc_ch8b40_vs_ch10g_holdoutguard_seed2_1m_2026-07-17/`，以及fresh统计与图`_runs/profiles/dqc_ch10g_holdoutguard_actorfeature_meananchor_b40_1m_s2_2026-07-17/ch8_fresh512_comparison/`。两张PNG均通过PIL解码。远端128个公开config键无绝对路径或真实敏感字段，name/group/tags均只含公开算法语义；同步文件仍只有标准`config.yaml/output.log/wandb-summary.json`。
- 下一独立路线不再用旧rollout直接决定整个cost head的停止位置。优先测试参数隔离的小cost adapter，并在共享表示梯度上记录cosine/冲突率；若冲突显著，再做PCGrad门控。与此并列保留QCPO_refs式Weibull/parametric tail辅助头，用低维尾部监督提供稳定信号。两条都必须保持B40/seed2/1M及现有PPO、PID、QR32不变，先过reward/outage/AUC联合门；不得同时混入IQN、quantile加密或PID调参。


### E133：目标口径纠正——outage必须接近alpha，而不是越低越好（2026-07-17）

- 用户明确最终任务是约束优化：真实违约概率应落在名义alpha=0.20附近，在满足该条件后尽可能提高mean reward。outage低于目标不是无条件收益；它通常代表lambda、critic概率或闭环控制把策略推入过度保守盆地，牺牲了本可获得的reward。
- 后续候选统一使用独立fresh512点估计的双侧工作带[0.18,0.22]。高于0.22记为风险过大，低于0.18记为过度保守或闭环失配；只有进入工作带的候选才按mean reward排序。边界附近仍报告二项比例置信区间，不能把一次点估计伪装成精确约束保证。
- 既有实验中使用的pid_target_prob=0.15是为有限样本、critic偏差与控制滞后设置的内部补偿setpoint，不等于最终希望达到15% outage。它是否仍应为0.15，只能在cost representation稳定后用独立大样本重新校准；当前不因C-H10G过度保守而立即和adapter同时调PID。
- 按新口径重评C-H10G：reward/outage=.57864/.14063，虽outage更低且Brier更好，但落到工作带下方，属于明显过度保守，不是优于C-H8的安全解。retention guard只作为cross-rollout遗忘诊断保留，不进入性能主线，也不扩seed或扫描容忍带。
- 以前预注册的单侧outage<=.22只可视为早期安全上界，不能继续充当最终模型选择规则。后续所有正式decision、表格和最终summary必须同时报告与0.20的有符号偏差、是否落入双侧带和mean reward。

### E134：C-H11小型cost adapter实现、零影响回归与梯度诊断（2026-07-17）

- 为隔离C-H6中QR辅助梯度直接扰动整个actor MLP+LSTM的问题，在recurrent base feature之后新增默认关闭的残差adapter：H→W→H、Tanh、末层零初始化，正式W=64、scale=1。policy/value始终使用adapter后的feature；共享cost辅助分支把base feature detach后再经过同一个adapter，因此cost梯度只能训练约束小子空间，不能写入原MLP/LSTM backbone。
- adapter既接受PPO/value主任务梯度，也接受cost辅助梯度，所以新增无副作用的autograd.grad诊断，记录两者在adapter参数上的范数、dot、cosine和冲突率。正式首轮不直接加入PCGrad；先测量冲突，只有长期负内积与性能退化共同出现时，正交投影才有证据基础。
- 默认cost_adapter_width=0严格兼容。相同seed305、B2、两批、C3的持久化金样本回归均exit0；补丁前后checkpoint各60个tensor leaves，键集合、shape、dtype和逐元素值全部相同，different_count=0、最大浮点差=0。额外纯张量测试还确认adapter构造保存并恢复CPU RNG，零初始化时初始policy/value输出完全一致。
- 正式尺寸H=512、W=64的持久化smoke训练11.7秒并exit0，adapter参数为66112；3个rollout产生3次actor事件、6次梯度诊断。首epoch PPO ratio最大误差2.861e-6，cost head在actor backward中的梯度存在标志为0，主任务与cost辅助在adapter上的梯度都非零且有限。
- smoke中6次诊断有3次负内积，冲突率0.5；running cosine均值0.00897，最后一次cosine 0.04032。它说明两个目标在小adapter上大体近乎正交、但局部冲突真实存在；样本只有192 env steps，不能推断最终性能，也不能据此提前启用PCGrad。
- 正式比较必须成对运行：C-H11A为adapter64、shared coef0的容量控制，C-H11B为同一adapter64、shared coef1的cost监督候选。两条初始输出/RNG、B40/T1000/C20/A8、1M预算、seed2、actor-feature、mean-anchor及PID全部相同，唯一有效差异是cost辅助梯度。两条都必须做fresh512；outage在[.18,.22]内后再比较reward。若只有control提升，归因为额外PPO容量；若candidate提升且control不提升，才归因为隔离后的cost supervision。


### E135：C-H11成对1M结果——adapter cost梯度进入目标带，但reward显著下降（2026-07-17）

- 实现与预注册绑定提交7d70cb6。C-H11A/C-H11B都使用B40、T1000、C20/A8、1M、seed2、actor-feature、mean-anchor、GAE-PPO、LSTM512、adapter64；唯一有效变量为shared cost coef 0/1。两条由launch_background持久化并发运行，均exit0，纯训练832.1/835.3秒；W&B dsze6uux/yys4r1ip均finished、各25行到1M、只同步3个标准文件。每条132个config键，无敏感键或绝对路径值，公开算法tag隐私审计通过。
- 工程门全部通过。两条都有66112个adapter参数和25次Actor事件；首epoch ratio最大误差为1.335e-5/1.615e-5。candidate共记录200次独立梯度诊断，112次负点积，冲突率.56、running cosine=-.002009；最后primary/aux norm=.00494/.01407，cost-head actor-gradient标志恒0。cost监督确实只进入adapter，既不是空操作，也没有破坏behavior probability。
- 训练曲线继续发生多次闭环反转。matched-budget末20%训练reward为control/candidate=.6845/.5871，outage=.140/.155，lambda=.0695/.1553；最后单批又变为reward=.833/.505、outage=.250/.150。400k、600k、800k和终点的排序不同，再次证明中途单窗口不能裁决。
- fresh512给出control reward/outage=.78887/.24023，candidate=.62898/.18359。按修订后的双侧规则，control高于[.18,.22]、风险过大；candidate点估计进入工作带，但reward下降.15989即20.27%，Welch95=[-.21495,-.10483]，差异明确。outage下降.05664，Newcombe95=[-.10643,-.00655]。candidate不能因进入目标带就判成功，因为原C-H8已在带内且reward=.74496。
- 相对C-H8，candidate reward下降.11598即15.57%，Welch95=[-.17907,-.05289]；outage下降.03125但Newcombe95=[-.08009,.01774]跨0。candidate的Wilson95 outage区间为[.15246,.21944]，点估计刚进带并不表示显著优于C-H8。
- critic仍未修复。control→candidate的true/pred outage为.240/.177→.184/.096，candidate低估.08765；hard AUC .5492→.5408，Brier .19162→.16051虽随更低基率下降，但Brier Skill -.04983→-.07088反而更差。hard CDF error恶化37.55%，crossing .14163→.19090，mean-cost error只从2.003小幅降到1.840。它把策略推向低reward端，却没有获得更有区分力的条件风险估计。
- adapter-only control相对C-H8 reward增加.04391但95%区间[-.01685,.10467]跨0，outage增加.02539且区间[-.02599,.07662]也跨0；Brier恶化12.49%、mean error恶化40%。因此额外PPO容量只有不确定的reward倾向，不能单独晋级。
- 标准PCGrad不值得直接跑全量。25个W&B记录的rollout末事件cosine均值-.00817、绝对值中位数.0283；14个负事件平均cosine-.03865。按正交投影公式，负事件平均仍保留99.883%的aux norm，只移除0.117%，不可能解释20% reward损失。cosine>0才更新的强门控会删除约56%更新，属于另一个独立消融，不应与PCGrad混称。
- 严格裁决：C-H11不扩seed、不启用PCGrad全量、adapter和cost共享监督均不进入默认配置。证据包位于_runs/wandb_export/dqc_ch11_adapter_pair_1m_s2_2026-07-17与_runs/profiles/dqc_ch11_adapter_pair_1m_s2_2026-07-17，含完整history、profile、三组fresh512统计CSV/JSON/PNG和privacy_audit。下一主路线转向QCPO_refs式低维Weibull tail辅助，检验降低cost分布监督方差是否比32点QR共享梯度更有效；强正余弦门控只作为分歧路线记录。


### E136：QCPO_refs Weibull尾部头源码审计——它是共享表示正则，不是直接CDF替代（2026-07-17）

- reference对cost distribution使用均匀tau，并定义`c_tau=-log(1-tau)`；cost quantile输出经过exp保证为正，另由共享history feature预测`alpha_W=4*sigmoid(linear(feature))`和`beta_W=exp(linear(feature))`。尾部比例为0.3，原设置N=25，并同时保留quantile与mean cost监督。
- `weibull_tail_loss`先sort当前quantiles再detach，只在上30%尾部计算`0.5*(log(beta_W)+(1/alpha_W)*log(c_tau)-log(q_tail))^2`。因此它不直接拟合真实MC cost的Weibull极大似然，也不通过该项更新quantile输出层；它主要让低维alpha/beta头及其共享feature去解释当前QR尾部形状。
- 这一区别决定首轮不能把Weibull CDF直接替换DQCAC actor使用的quantile CDF，也不能与PID、IQN、local grid、adapter或replay同时改。首轮只检验“低维尾部辅助是否改善cost critic跨rollout表示”，actor仍查询原32点QR。
- DQCAC适配采用action-conditioned cost-critic hidden trunk，而不是reference的state-only head；这是因为当前actor风险优势必须比较同一state下不同action。QR/mean继续由真实MC cost监督，Weibull target只来自detach后的当前QR上尾，cost除以10后进入log域，并沿用risk-discount transition权重。
- reference的正值quantile通过exp天然满足log定义域；当前主线linear quantile为保持历史可比性不改输出族，因此新增epsilon=1e-3和tail clamp fraction诊断。若成熟训练长期大量clamp，说明这项reference正则与linear输出不兼容，应停止而不是隐藏截断。

### E137：C-H12 Weibull辅助实现、零影响回归与smoke验证（2026-07-17）

- 新增默认关闭参数`cost_weibull_tail_coef=0, tail_prob=.3, cost_scale=10, epsilon=1e-3`。启用时新增`WeibullTailHead`，从cost critic最后隐藏层预测alpha/log-beta；辅助梯度进入Weibull头和cost hidden trunk，不进入actor/MLP+LSTM，也不通过该项进入quantile输出层。头只加入critic Adam和joint gradient clip。
- 正式首轮配置被限制为C-H8兼容路径：recurrent actor-feature、MC、QR、uniform、linear、online quantile CDF、feature refresh关闭、full-batch critic、shared/adapter/replay/holdout/s0 auxiliary全关闭。这样Weibull是唯一新增机制，PID仍保持内部target=.15；最终裁决继续用fresh真实outage接近alpha=.20的双侧规则。
- 纯张量验证中，`forward_features→final Linear`与原forward逐元素相同；N32/tail=.3得到index21和11个尾点；head为514参数。log-domain实现与reference等价式最大差`3.73e-9`；detach target无梯度、cost trunk和Weibull头梯度非零、quantile final layer不接收该辅助梯度。
- 默认关闭持久化4k回归耗时12.7秒并exit0。与上一版金检查点比较，旧/新各60个tensor leaves，missing/extra/different均为0，最大浮点绝对差0；新增代码在coef0时没有改变既有模型、优化器或随机序列。
- 启用B2×T32×3 rollout、C3/A2 smoke耗时7.6秒并exit0。最终tail loss=5.9474、head grad norm=11.782、alpha范围[2.240,2.472]、beta均值=.2887、tail clamp fraction=0；514个参数和全部48个checkpoint tensors均有限。PPO首epochratio最大误差`5.66e-6`，说明新增critic正则没有破坏behavior probability。
- 启用检查点随后由独立持久化eval-only任务成功重建并加载`cost_weibull_head`，4回合验证exit0。短测的零cost/outage只反映T32任务过短，不能用于性能判断，也不作为“outage越低越好”的证据。
- 下一步先做固定成熟policy、同seed、同600k监督预算的coef0/1配对机制验证；预计每条约4--6分钟，先于任何live 1M。门槛为全程有限、late clamp不高于10%、head梯度非零，并要求末5批prequential Brier至少改善10%或AUC增加至少.02，另一项不得恶化超过10%。未通过则停止Weibull live闭环；通过后才在C-H8 seed2上跑1M。
- live候选的fresh512主规则已修正为：outage点估计在[.18,.22]才进入reward比较；低于.18判为过度保守，高于.22判为风险过大。进入工作带后要求mean reward至少不低于C-H8的.74496并争取明显提高，同时报告Wilson区间、BSS/AUC、CDF/mean误差和crossing。不会因Weibull把outage压得更低就宣布提升。


### E138：C-H12固定策略600k配对结果——Weibull拟合健康但未改善QR泛化（2026-07-17）

- 正式实现/记录绑定提交`867bb65/daa68be`。control与candidate都冻结同一个成熟MLP+LSTM策略及observation RMS，使用rollout seed312、B40×15=600k、T1000、C20、actor-feature、MC、risk-discount、mean-anchor=.5、QR32和full-batch critic；唯一变量是`cost_weibull_tail_coef=0/1`。两条均由`launch_background.sh`持久化、W&B online运行并exit0。
- 配对因果性成立。15个记录点的reward、reward quantile、真实outage、mean cost、prequential truth及truth cost mean逐值完全相同，所有最大绝对差均为0。control/candidate纯训练耗时243.9/247.5秒，Weibull只增加1.48%墙钟。
- Weibull机制确实工作且数值稳定。candidate全程/末5批tail loss均值为.74473/.08363，末点.08620；末5批head grad norm均值2.186，非零。tail clamp全程均值仅2.17e-5、末5批严格0；alpha末5批均值3.9908，接近4的上界，beta均值1.415。没有NaN/Inf，说明失败不是非法log、clamp主导、梯度消失或代码空操作。
- 但下一rollout proper score没有改善。末5批pre-Brier control→candidate为.205674→.206123，恶化.22%；pre-CDF absolute error .069688→.073750，恶化5.83%；pre mean-cost bias绝对值1.0245→1.2715，恶化24.11%。末5批post-Brier也从.184263升至.186299，恶化1.11%。全程pre-Brier同样恶化约.96%。
- 独立128条使用完全相同的policy/eval随机流，truth均为reward=.869919、outage=.25、Q20 cost=17.6、mean cost=11.211。hard CDF只从.165039变为.165283，absolute error仅改善.000244；hard Brier .196060→.195183，改善.45%，远低于10%门；AUC .577311→.569987，下降.007324。BSS从-4.57%变为-4.10%，仍低于常数基率预测。smooth Brier改善.44%，smooth AUC下降.00293，结论一致。
- shape方面只有crossing从.18473降到.16759，改善9.28%；predicted mean从8.0303降到8.0011，相对真实11.211反而略差。它与transition replay等结果一致：quantile形状更规整不等于查询点概率或action排序更准。
- 严格裁决为`mechanism_fail_no_live_1m`：不进入C-H8 live seed2、不扫描Weibull coef/tail proportion/alpha上界，也不因低训练开销而追加1.2M。source loss是对detach QR尾部的自蒸馏；当前证据表明它能拟合自己的tail，却没有增加真实标签信息或未见状态的条件风险分辨率。
- 更贴近reference的“把Weibull梯度写入共享actor history”保留为次级分歧路线，但不优先：完整共享QR和adapter共享监督已经一致损害reward，而本轮又证明tail自蒸馏没有先改善QR概率。若未来测试，必须单独记录PPO/value与Weibull在共享参数上的梯度冲突，不能把它和non-crossing结构同时加入。
- 两条W&B run为control `k3pbid7r`、candidate `bdqi1v4s`，各15行、138个公开config键、293个summary键；隐私审计无敏感键和绝对路径值，只同步3个标准文件。完整导出/profile位于`_runs/wandb_export|profiles/dqc_ch12_weibull_frozen_pair_600k_s312_2026-07-17`，含history、CSV/JSON/Markdown和1.4MB overview图。
- 下一优先路线转向尚未验证且直接对应现有失败模式的non-crossing quantile结构。此前uniform-IQN在600k固定策略下使CDF/mean error改善约16.1%/19.4%，但crossing恶化到.234；NQ-Net式单调头可能保留连续tau/局部分辨率收益并消除伪crossing。先审计同仓库NQ-Net源码与损失，再做默认关闭、固定策略600k机制门；不与Weibull、PID或shared actor梯度混合。

### E139：C-NQ1 NQ-Net论文审计、非交叉cost critic实现与固定策略预注册（2026-07-17）

- 本地Deep-Distributional-Learning-with-Non-crossing-Quantile-Network是空gitlink，外层只记录commit 4cacd436ddd391711c23248bb08058bcb08ae312；没有.gitmodules、内层.git或memory所述stash，整个vepfs也没有另一份NQ源码。因此没有向空gitlink写入伪恢复内容，公式直接审计官方arXiv 2504.08215。论文一般NQ-Net输出mean与pre-activated gaps，用ELU+1保证严格正gap；Atari的NQ-Net*明确改为ReLU，因为离散游戏回报的相邻quantile差经常接近0。我们的episode cost同样是离散/原子型，首个候选预注册为ReLU，elu1保留为机制失败后的独立激活消融。
- 实现使用非冗余等价参数化：最后Linear仍输出N个raw值，第一个为全部quantile的mean，余下N-1个为相邻gap；[0,cumsum(gaps)]按行中心化后加mean。于是mean(q)=v且q[i+1]-q[i]=activation(gap[i])。论文K+1公式中的首gap会被中心化完全抵消；去掉它既不改变可表示quantile集合，也让NQ32与QR32保持相同206112个参数、相同state_dict形状和相同初始化RNG消耗。
- 新配置为cost_distribution_model=nq,cost_nq_gap_activation=relu|elu1，默认仍是qr。NQ首轮只允许uniform32、MC、detach actor-feature、linear output、online quantile CDF、full-batch critic，并与IQN、local grid、direct CDF、feature refresh、replay、guard、Weibull、shared backbone及adapter互斥；reward critic、GAE/PPO、PID和观测归一化完全不变。
- 纯张量门通过：相同seed的QR/NQ共6个state张量逐位相等；ReLU crossing精确0，输出均值误差最大2.76e-7，相邻gap公式误差最大4.77e-7；ELU+1最小gap 0.3618>0，N=1边界和全部梯度有限。默认QR持久化4k金样本训练12.9秒并exit 0，相对补丁前checkpoint的60个tensor leaves missing/extra/different=0/0/0、最大差0。
- 启用NQ的B2×T1000×3 rollout、C3、actor-feature smoke训练16.6秒并exit 0；43个模块tensor全有限，训练/8条评估crossing均为0。随后eval-only持久化任务正常重建nq/relu并加载6000步checkpoint，证明config、保存和恢复链路完整。短测CDF仍为0只说明6k监督不足，不用于性能裁决。
- 正式机制门复用C-H12的冻结成熟policy与seed312：B40×15=600k、C20、QR/NQ32、MC、risk-discount=.995、actor-feature、mean-anchor=.5/scale10、LSTM512、obs RMS和T1完全相同，只把QR head换成ReLU-NQ。QR控制已由run k3pbid7r给出；默认回归证明可直接复用，不再重复消耗同一控制预算。NQ训练预计4--6分钟、内置128评估约1--2分钟，fresh512仅在机制门过后再花4--5分钟。
- 晋级要求首先是crossing从QR约.185降到精确0；更重要的是末5批prequential Brier或CDF absolute error至少改善10%，另一项不得恶化10%，独立128条hard Brier不能恶化、AUC至少不低于QR .5773且mean-cost误差下降。若只消除crossing而proper score/AUC不改善，则严格失败，不跑live 1M；若ReLU出现大量dead gap并在早期100--200k明显欠拟合，记录后才运行ELU+1固定策略轻量消融，不能事后扫描未预注册gap bias。

### E140：C-NQ1固定策略正式结果——零crossing没有改善查询概率（2026-07-17）

- 正式ReLU-NQ任务由`launch_background.sh`持久化运行，绑定提交`ed3859f`；job为`DQCAC_DynamicButton_cnq1_nqrelu_frozen_b40_600k_s312`，W&B run为`8de2ybaw`。B40×15共600k监督步、C20、QR/NQ32、MC、risk-discount=.995、actor-feature、mean-anchor=.5/scale10、LSTM512与成熟冻结policy均和QR控制`k3pbid7r`相同。15批reward、真实cost、outage和env_steps逐点最大差均为0，比较只改变cost分布head。纯训练`246.67s`，相对QR控制`243.91s`只慢约1.13%；任务正常exit0、远端finished且只有15行history。
- 结构门通过：训练全部15点和独立128条评估的crossing均为精确0；QR末5批crossing均值`.14597`，独立评估为`.18473`。但proper-score门失败。末5批pre-Brier为QR/NQ `.205674/.205894`，NQ恶化0.11%，远未达到10%改善；pre-CDF abs error `.069688→.066406`改善4.71%，pre mean-cost绝对bias `1.1830→1.0497`改善11.27%，但critic更新后的post-Brier、post-CDF error和post mean-bias分别恶化1.45%、4.93%和5.93%。
- 独立128条终评的真实reward/outage/mean cost完全相同，均为`.869919/.25/11.21094`。QR→NQ的hard Brier为`.196060→.203445`，恶化3.77%；hard AUC `.577311→.564779`，下降`.01253`；Brier Skill `-.04565→-.08504`。smooth Brier同样`.195100→.200472`恶化；只有smooth AUC偶然增加`.00358`，不足以推翻hard查询和prequential proper score的共同失败。预测mean cost还从`8.0303`降到`7.8725`，对真实值低估更重。
- ReLU gap没有发生“全部dead”的退化：独立评估预测分布std仍为`7.0328`、CDF为`.1616`，训练中的Ghat和mean均持续变化。因此不触发预注册的ELU+1补救路线；ELU+1会强迫离散cost原子之间严格正gap，反而可能制造不存在的插值。严格裁决为`strict_fail_no_live_policy`：不跑ELU+1、不跑live 1M、不扩seed，也不扫描gap bias。
- 更关键的算法结论是，当前hard-CDF定义为`N^{-1}Σ_i 1[q_i≥d]`，对quantile排列是置换不变的；把同一组QR输出排序或仅消除crossing不会直接改变CDF或mean。NQ只有通过改变联合参数化和训练轨迹才可能间接改善概率，本实验表明这种耦合反而损害Brier/AUC。因此crossing不是当前DQCAC查询误差的主因，下一主线应转向action-conditioned风险credit/条件泛化与闭环setpoint，而不是继续优化分位数顺序。
- 完整导出在`_runs/wandb_export/dqc_cnq1_nqrelu_frozen_b40_600k_s312_2026-07-17/`；QR/NQ配对history、`comparison.json`和profile位于`_runs/wandb_export|profiles/dqc_ch12_qr_vs_cnq1_nqrelu_frozen_seed312_600k_2026-07-17/`。`overview.png`为`2880×4128`且通过PIL解码。NQ公开config共139个键，无敏感键名或绝对路径值；W&B name/group/tags只含公开算法语义。

### E141：C-H8 B40补齐seed0/1——把当前Pareto底座的稳定性测清（2026-07-17）

- NQ、Weibull、adapter、retention guard和training replay均未通过各自机制/性能门，当前唯一同时接近最终双侧目标的配置仍是C-H8 seed2：fresh512 reward/outage=`.74496/.21484`。E128因reward比事前`.75`门少`.00504`而严格未晋级，这个裁决不撤销；但在后续候选全部被同一底座支配后，补seed0/1的目的变成“测量当前最好底座的训练seed方差”，不是事后把E128包装成成功。
- 两条完全复用C-H8：B40×25=1M、T1000、C20/A8、actor/PID interval1、QR32/MC、risk-discount=.995、detach actor-feature、mean-anchor=.5/scale10、LSTM512、obs RMS、T1 sigmoid、经验PI/PID内部target=.15及所有LR不变；只改训练seed、唯一tag/checkpoint和脱敏W&B name。当前提交上的NQ/Weibull/adapter/guard等新参数全部保持默认关闭，已有逐位回归覆盖默认路径。
- 由于B40 seed2前300k几乎不涨而后段明显恢复，两条不以100k/300k reward早停；只有NaN/Inf/OOM、ratio断言或确定性工程错误才停止。每条纯训练按seed2实测约7--9分钟，内置128约1--2分钟；单A100串行。随后无论内置128好坏都做独立fresh512，每条约4分钟，以免小评估集选择性扩算。
- 最终报告每seed相对`.20`的有符号outage偏差、Wilson区间和reward，并同时给3-seed mean±SD及合并事件率。工作带仍是fresh512点估计[.18,.22]；低于.18记为过度保守，高于.22记为风险过大。只有进入工作带后才按reward排序；更低outage不自动加分。若B40跨seed仍分叉，下一步优先做固定QR表示后的PID setpoint/risk-gain校准或延长当前底座，而不是继续增加quantile结构。

### E142：C-H8 B40三seed审计——平均outage接近0.20是两侧失配相互抵消（2026-07-17）

- seed0/1训练均由`launch_background.sh`持久化完成，纯训练`407.1/406.8s`、exit0；W&B run为`jmg0phkm/02gjpbeh`，连同既有seed2 `brtk50w8`均finished且每条25个记录点到1M。三条公开config共检查397个值，没有敏感键或绝对路径；每条只同步3个标准W&B文件。
- fresh512逐seed reward/outage为seed0 `.73447/86÷512=.16797`、seed1 `.81951/128÷512=.25000`、seed2 `.74496/110÷512=.21484`。只有seed2落入[.18,.22]工作带；seed0过度保守，seed1违反风险预算。三seed均值为reward `.76631±.04637`、outage `.21094±.04115`，但合并`324/1536=.21094`不能掩盖两侧分叉。
- 与完全同协议的B20三seed比较，B20 reward/outage均值为`.84282±.19171/.20247±.03211`，合并`311/1536=.20247`。B40使reward均值下降`.07651`，outage平均绝对目标偏差从`.02513`恶化到`.03229`，目标带命中仍为`1/3`；它只把reward的seed标准差压低，却没有把约束控制稳定在0.20。
- B40相对B20的逐seedreward变化为`-.17033/-.17636/+.11717`，outage变化为`-.04688/+.02344/+.04883`，方向不一致；三个seed的配对t区间都很宽，不能宣称B40显著支配或退化。critic同样没有一致提升：平均Brier `.16794→.17055`，AUC `.57857→.58569`，逐seed正负混合。
- 训练曲线也不是稳态。B40三个seed末5批outage均值/标准差分别为`.165/.115`、`.215/.084`、`.180/.069`；末5批落入工作带的比例为`0/0/.4`。lambda与下一批outage变化的相关为`-.50/-.48/-.61`，说明惩罚方向总体有效，但响应滞后且每批仍跨越目标两侧，符合负反馈极限环而非方向完全接反。
- 公平解释必须包含更新事件数：相同1M环境步下，B20有50次Actor/PID事件和1000次critic Adam step，B40只有25次事件和500次critic step；批量翻倍使trajectory-epoch暴露量近似不变，但Adam/controller时间步减半。因此“num_envs更大”同时改变了梯度方差与学习时间尺度，不能只按batch大小解释。
- 证据位于`_runs/wandb_export/dqc_ch8b40_multiseed_1m_2026-07-17/`和`_runs/profiles/dqc_ch8b40_multiseed_1m_2026-07-17/`，包括三seed完整history、fresh512 CSV/JSON、训练稳定性CSV、逐seed统计与`3300×1056`对比图；全部5张PNG通过PIL解码。

### E143：B40–2M长度审计预注册——只回答25次更新是否欠训练（2026-07-17）

- 下一条只在C-H8压力seed1把`num_iterations=25→50`，得到`B40×50×T1000=2M`；网络、QR32/MC、C20/A8、LSTM512、obs RMS、actor-feature、mean-anchor、PPO/GAE/IS、PID target=.15及所有增益完全不变。评估checkpoint不保存optimizer/scheduler动量，所以不能伪装成无损续训，必须从头运行。
- 2M的前1M理论上应与当前seed1逐点复现。先比较25条history以及1M pre-update checkpoint的六个module、lambda和公共runtime；若不一致，后1M不能被解释为纯长度效应。纯训练预计13--16分钟，内部128约1--2分钟；除NaN/OOM/确定性错误外跑满，任务继续使用持久化后台和脱敏W&B online。
- 内部评估仅作节省明显失败评估的宽screen：reward至少`.60`、outage在`.05--.40`且所有关键量有限，才花约3--4分钟做fresh512。正式晋级要求fresh512 outage落入[.18,.22]、reward至少不低于seed1 1M的`.81951`，且末段outage振幅和critic Brier/AUC不能明显恶化。失败则停止原样3M，不用选择某个漂亮训练窗口改写结论。
- 若2M通过，再扩seed0/2并进入同预算QCPO/QCPO_refs比较；若失败，下一算法路线不是继续压低PID target，而是给DQCAC actor加入默认关闭的真实轨迹outage policy-gradient校正。该校正用on-policy二元违约标签提供无偏风险方向，distributional critic保留为低方差局部项/基线；先做梯度尺度、leave-one-out baseline与PPO-Clip机制验证，再决定live 1M。PID window100/降低Kp-Ki保留为控制消融，但已有P-M10表明它主要沿reward--risk前沿移动工作点，不能单独修复AUC接近随机的问题。

### E144：B40–2M长度审计——终点回到目标带，但更多更新没有形成稳态（2026-07-17）

- 正式持久化job绑定提交`c307fff`，W&B run `yfxjglt0`、exit0；B40×50共2M、50次Actor/PID事件，纯训练`803.2s`。前1M与原1M run的25行、313个共同数值metric逐项exact；800k和1M pre-update checkpoint各44个tensor leaves、32个runtime与5个rollout metric也exact，最大差0。
- 第二个1M的训练均值相对第一个1M为reward `.43962→.90098`、outage `.137→.246`、lambda `.0652→.2786`。末400k reward/outage/lambda为`.94349/.270/.36411`，outage标准差`.05986`、相邻批平均绝对跳变`.09167`；末200k仍为`.83649/.265`。长度提高reward，但没有把真实风险稳定在0.20。
- 内置128为reward/outage `.8816/.2578`，critic CDF `.2979`；按宽screen执行fresh512。统一fresh512的1M→2M为reward `.81951→.79542`，差`-.02408`、Welch95 `[-.09223,+.04406]`；outage `128/512=.25→109/512=.21289`，差`-.03711`、Newcombe95 `[-.08857,+.01458]`。2M点估计进入[.18,.22]，Wilson95为`[.17964,.25042]`，但reward低于预注册`.81951`门，因此不扩seed、不跑3M。
- critic变化是混合的：hard CDF error改善32.6%，Brier改善9.96%、AUC `.5510→.5928`、crossing下降`.0376`；但Brier Skill `-.0587→-.0667`仍为负且更差，mean-cost error `3.457→5.002`恶化44.7%，预测从低估翻为高估。终点更安全不能解释为distributional critic全面变准。
- 结论不是“更久完全没用”，而是“1M欠更新不是主要瓶颈”。2M得到一个更接近目标的终点，却发生多轮`.075↔.325`跨带摆动；最后400k训练均值仍明显不安全。按预注册规则停止原样延长，转向risk-credit校正。
- 证据位于`_runs/wandb_export/dqc_ch8b40_seed1_1m_vs_2m_2026-07-17/`和`_runs/profiles/dqc_ch8b40_seed1_1m_vs_2m_2026-07-17/`；两条公开config共274个值，无敏感键或绝对路径。common1m/full2m/fresh512三张PNG均通过PIL解码。

### E145：真实轨迹outage residual校正预注册（2026-07-17）

- 当前risk advantage是`Acritic=p_hat(s,a)-V_hat(s)`，低方差但其方向受fresh AUC约.55--.59限制。新增默认关闭系数`cost_actor_mc_correction_coef=eta∈[0,1]`，定义`Aeta=Acritic+eta*(I_outage-p_hat)= (1-eta)Acritic+eta*(I_outage-V_hat)`。
- `I_outage`由每条完整on-policy trajectory的`disc_cost>=limit`产生并沿时间广播。budget递推保证同一标签等价于每时刻“remaining cost是否超过remaining budget”。`V_hat`仍由behavior policy下K个独立动作的CDF均值给出，是action-independent control variate；即使不准也不改变score-function期望。eta=1时实际动作的critic预测完全抵消，只保留真实标签方向和critic状态基线。
- DQCACBeta的`beta^t`时间权重、PPO old log-prob、每epoch重算ratio、reward min/risk max clip、sum normalization与经验PID全部不变。constraint RMS必须改为跟踪真正使用的`Aeta`，否则真实0/1 residual会被旧critic约`.004`的尺度放大；eta=0必须逐tensor exact保持当前路径。
- 实现先通过解析代数/符号测试、eta0金样本回归、mixed-outage PPO梯度方向、recurrent B40×T1000 smoke和checkpoint恢复。首条live只测试eta=1、seed1、C-H8 B40 1M，预计训练7--9分钟、fresh512约3--4分钟；正式门为outage落入[.18,.22]且reward至少`.82`，并要求末段振荡不放大。eta=.25/.5作为偏差--方差消融保留，只有eta1方向正确但明显过强时才依次轻量验证，不能事后无界扫描。


### E146：trajectory outage actor correction实现、严格回归与正式1M启动门（2026-07-17）

- 实现绑定提交`6634e12`。新增默认关闭参数`cost_actor_mc_correction_coef=eta`并强制范围`[0,1]`；启用时要求完整episodic trajectory。核心函数同时构造`Acritic=p_hat-V_hat`、`AMC=I_outage-V_hat`和residual `I_outage-p_hat`。eta=0直接返回原critic tensor，不执行无效加零；eta=1直接返回MC tensor，不通过两次相消形成额外舍入误差；中间值才执行`Acritic+eta*residual`。
- `disc_cost>=cost_limit`先得到每episode一个二元标签，再按rollout的time-major索引`[t0:B,t1:B,...]`广播到`T*B`。`actor_update_interval>1`时，多个rollout的`disc_cost`沿环境维合并后再广播，避免把rollout-major标签错配给time-major state/action。LSTM与MLP actor都走同一helper；首个PPO epoch缓存blended/critic/MC/residual/label，后续epoch只重算当前log-probability和ratio，不允许critic更新移动同一behavior batch的风险监督。
- constraint RMS现在跟踪actor真正使用的`Aeta`，不是固定跟踪`Acritic`。W&B新增critic/MC advantage标准差、residual标准差、实际correction绝对均值、critic--MC相关、trajectory标签均值和eta；旧的`risk_adv_*`主键明确代表最终blended信号。beta时间权重、固定old log-prob、每epoch importance ratio、reward min clip、risk max clip、sum normalization和经验PID均未改变。
- 解析测试覆盖两条轨迹、三时刻的time-major标签`[0,1,0,1,0,1]`，eta=0/1/.25均与手算逐元素相同；合并4条轨迹后的标签顺序同样通过。纯autograd测试在正/负MC advantage下得到log-ratio梯度`[+.2000,-.04975,+.1575,-.0600]`，证明梯度下降会降低高风险动作概率、提高低风险动作概率，beta只缩放不翻转方向。
- 默认兼容性使用补丁前提交`e2ea6b3`、seed401、B2×T32、3 rollout、C3/A2的持久化金样本，与补丁后显式eta=0复跑逐checkpoint比较。final及64/128/192步四个checkpoint各44个tensor/array leaves完全相同，共176次比较最大绝对差0；730个共同数值状态也全部exact。训练reward/outage、最终随机评估和checkpoint模块均一致，新增诊断没有改变RNG、优化器或浮点更新路径。
- eta=1的recurrent cadence2强制标签短测用`cost_limit=-1`只为保证非零correction，不用于性能判断。B2×T32、4 rollout、C3/A2训练8.5秒并exit0；44个checkpoint tensor leaves全部有限，2次actor事件都正确使用4条合并trajectory，首epoch ratio最大误差`1.23024e-4<1e-3`。独立持久化eval-only成功从final checkpoint重建eta=1配置并加载256步状态，exit0。
- MLP eta=1短测的actor更新本身正常，但暴露两个既有终态兼容bug：通用summary直接访问只有recurrent actor才有的`cost_adapter`，统一打印又假定MLP评估一定返回AUC/BSS/smooth字段。前者改为安全可选属性，后者对未提供的纯诊断显示nan；第三次相同短测训练7.2秒、保存final checkpoint、评估和JSON均完整exit0。这两处只影响训练后的汇总/显示，不改变模型或评估数值。
- 正式首轮严格复用C-H8 B40 seed1：`25×40×1000=1M`、C20/A8、actor/PID interval1、QR32、MC cost、risk-discount=.995、detach actor-feature、mean-anchor=.5/scale10、MLP+LSTM512、observation RMS、T1 sigmoid、GAE-PPO、固定old log-prob和经验PI/PID内部target=.15；唯一算法变量为`eta=1`。不同时调整PID、quantile、CDF、网络、replay或guard。预计纯训练7--9分钟、内置128约1--2分钟、fresh512约3--4分钟，全部使用`launch_background.sh`、脱敏W&B online和eval-only禁用W&B。
- B40已有慢启动和长周期，除NaN/Inf/OOM、ratio断言或确定性工程错误外跑满1M。正式裁决首先要求fresh512真实outage落入双侧工作带`[.18,.22]`，然后要求mean reward至少达到seed1底座`.81951`（预注册简写`.82`），并报告Wilson/Newcombe/Welch区间、末段outage振幅、constraint RMS、critic--MC相关、Brier/AUC/BSS、mean-cost error和crossing。低于.18不因更安全自动晋级，高于.22判风险过大。
- 分歧路线已预先固定：若eta=1把outage从高风险侧移向或越过目标但reward明显损失，说明真实方向有效而方差/强度过大，才按单变量依次考虑eta=.5、.25的轻量或1M验证；若eta=1仍高于.22且MC信号/尺度正常，减小eta只会增加critic偏置，不做盲扫，转controller cadence或更直接的trajectory baseline；若eta=1同时进带且提高reward，先扩seed0/2而不是继续调参。

### E147：eta=1真实trajectory风险方向没有把闭环校准到0.20（2026-07-17）

- 正式job `DQCAC_DynamicButton_ch13_mcfix_eta1_b40_1m_s1` 由持久化后台正常完成，exit0；W&B online run为`7yu7ublr`并已finished。配置严格复用C-H8 seed1的B40×25×T1000、C20/A8、LSTM512、QR32/MC、GAE-PPO、经验PID target=.15，仅把`cost_actor_mc_correction_coef`从0改为1。纯训练耗时`411.1s`，25条history完整到1M，无NaN/Inf；首epoch behavior ratio最大误差`1.72e-5`，说明失败不是importance-ratio或旧概率覆盖错误。
- 训练没有进入目标附近稳态。相对eta0基线，末5批reward/outage/lambda由`.7563/.215/.1697`变为`1.288/.410/.7225`；最终训练批为`1.362/.450/.8397`。lambda及risk coefficient持续升到`.8397/.4564`，但outage仍处于`.35--.45`高风险区，不能把结果解释为控制器没有收到违约信号。
- 内置独立128回合为reward/outage=`1.318/60÷128=.46875`，Wilson95%区间`[.3845,.5548]`，远离名义`.20`和工程带`[.18,.22]`。critic预测`.379`，低估真实outage约`.089`；hard Brier/AUC/BSS为`.2559/.5973/-2.76%`。该点超过预注册宽screen上界`.40`，即使按二项区间下界也明显超标，因此停止约3--4分钟的fresh512，不用更多评估样本确认一个已经明确的失败。
- eta1不是数值爆炸。末5批MC/blended优势标准差为`.3998`，constraint EMA sigma为`.3542`，二者尺度匹配；PPO末段KL约`.00125`、clip fraction约`.0577`，没有大步越界。真正的问题是信号质量：同批critic优势标准差仅`.00628`，MC优势约为其64倍；critic--MC相关仅`.00294`，几乎正交。eta1把低方差的动作条件方向完全替换成每条trajectory共享的二元标签，得到理论上方向正确、实际方差很大且缺少时刻/动作区分的credit。
- 不直接运行eta=.5/.25。当前`Aeta=Acritic+eta*residual`随后除以同一`Aeta`的EMA标准差；当residual比critic大两个数量级时，把eta从1缩到.5或.25会让分子和归一化分母近似同比缩小，稳态有效梯度及方向几乎不变。即使eta=.25，MC项标准差仍约是critic的16倍。因此原预注册的“eta扫描”在当前归一化下不是有效强度消融，盲跑1M没有信息价值。
- 下一候选应让混合参数具有可辨识语义：分别跟踪critic与MC residual的RMS，按`A=Acritic+rho*(sigma_critic/sigma_residual)*residual`配平后再做一次总归一化；rho才控制真实相对贡献。另一条独立路线是把B40提高到B80并保持每次Actor/PID更新的trajectory数增加，同时按总环境步重新标定更新事件数。两条不能同时改；先做默认关闭实现、符号/eta0逐位回归和B80显存smoke，再选择一条正式1M。最终裁决继续是fresh outage进入`[.18,.22]`后最大化mean reward，不以outage更低为优。
- 完整证据位于`_runs/wandb_export/dqc_ch8b40_vs_ch13_mcfix_eta1_b40_1m_s1_2026-07-17/`和同名`_runs/profiles/`。profile新增trajectory-MC分解曲线，明确显示critic/MC尺度、residual、相关性和label基率；这些绘图改动只扩展离线分析，不影响训练。

### E148：RMS配平的trajectory residual收缩实现与正式预注册（2026-07-17）

- 新模式默认关闭，参数为`cost_actor_mc_correction_mode=raw|rms_balanced`。raw逐式保留`Acritic+eta*(I-p_hat)`；balanced定义`A=Acritic+rho*s*(I-p_hat)`，其中`s=clip(sigma_critic/sigma_residual, max=1)`、两个sigma的数值下限为`1e-4`。rho=1使实际修正项与critic advantage具有约相同RMS，而不是让二元MC项以约64倍尺度完全替换critic。该估计有意以偏差换方差，不能声称rho=1仍是完整无偏trajectory REINFORCE。
- 两套component EMA只在behavior batch首次真正进入actor时更新一次。online模式下这与actor实际查询的critic版本一致；随后8个PPO epoch冻结scale、风险优势和old log-prob，只重算当前策略分子。constraint RMS用同一个最终blended advantage更新一次，负责整体尺度；它不再把rho的分量比例抵消。actor cadence合并时统计覆盖全部独立trajectory，MLP和MLP+LSTM共用同一helper。
- 新增W&B诊断包括balance scale、critic/residual reference sigma、实际`correction_std/critic_std`、两套component EMA及模式/数值保护配置。离线profile同步绘制这些字段；因此正式实验能直接验证rho是否真的产生预期相对贡献，而不是只看最终outage猜测机制。
- 解析测试使用两条轨迹、三时刻的time-major标签，raw eta0/1端点逐元素正确；balanced rho=1和rho=.25的实际修正/critic RMS比分别为`1.0/.25`，第二次PPO式查询不再推进EMA且输出完全相同。默认兼容用补丁前提交`00697e1`、seed404、B2×T32、3 rollout、C3/A2生成金样本；补丁后同配置复跑正常exit0。四个checkpoint共176个tensor/array leaves逐元素完全一致，最大差0；排除纯墙钟后178个共享JSON数值完全一致。
- 首次enabled smoke在构造期发现校验早于`self.advantage_norm`初始化，尚未进入训练；校验改为读取同一`args.advantage_norm`后复用原job名重跑。recurrent rho1 smoke训练7.3秒、3个actor事件、44个checkpoint tensor全有限，首epoch ratio误差`4.05e-5`；MLP rho1 smoke训练7.4秒、41个tensor全有限、ratio误差0。两个短测都用`cost_limit=-1`强制非零标签，只证明链路，不提供性能证据。
- 正式C-H14只改C-H8 seed1的risk estimator：B40×25×T1000=1M、C20/A8、LSTM512、QR32/MC、actor-feature、mean-anchor=.5/scale10、obs RMS、GAE-PPO、beta=.995、经验PID target=.15与所有LR保持不变；设置`correction_coef=1, mode=rms_balanced, floor=1e-4, ratio_max=1`。W&B使用脱敏name `DQCAC_DynamicButton_ch14_mcbalance_rho1_b40_1m_s1`和group `dqcac_trajectory_mc_balance_dynamicbutton`。
- 预计纯训练7--9分钟、内置128评估1--2分钟。机制门要求全部有限、首epoch ratio小于`1e-3`、末段实际修正/critic RMS比在`[.5,2]`且balance scale不长期触及非有限值。内置宽screen为reward至少`.60`、outage位于`.05--.40`；通过才做约3--4分钟fresh512。
- 最终首先要求fresh outage点估计进入双侧带`[.18,.22]`；reward最低参考是同seed 2M带内点`.79542`，目标是不低于1M基线`.81951`。低于.18且reward下降记为过度保守，高于.22记为风险不足；更低outage只有在reward同时更高时才构成支配。若rho1过度保守，下一单变量为rho=.5；若仍明显不安全但方向改善，可讨论允许rho>1或B80独立trajectory消融，不能同时改batch、PID和rho。


### E149：C-H14 rho=1配平残差——风险估计显著变准，但策略越过目标进入保守侧（2026-07-17）

- 正式job `DQCAC_DynamicButton_ch14_mcbalance_rho1_b40_1m_s1`绑定提交`f342528`，由`launch_background.sh`持久化完成，exit0；W&B run为`eklrfou4`且已finished。配置相对C-H8 seed1只增加`mode=rms_balanced,rho=1`，B40×25×T1000=1M、C20/A8、LSTM512、QR32/MC、actor-feature、mean-anchor=.5/scale10、obs RMS、GAE-PPO、beta=.995、经验PID内部target=.15和所有LR均不变。纯训练`407.2s`，全部关键数值有限。
- 机制门通过。后20%的实际MC修正/critic风险优势标准差比均值为`1.4016`，范围`[.9024,1.8331]`，位于预注册`[.5,2]`；balance scale均值`.01669`，critic/residual EMA尺度约`.00504/.30221`，证明二元残差已按约60倍尺度差配平而不是直接压进actor。首epoch behavior ratio链路继续正常。相对C-H8，后20% PPO clip fraction由`.12843`降到`.06341`，KL由`.002319`降到`.001341`，说明rho1确实改变并收紧了策略更新。
- 内置128条为reward/outage `.7434/.1328`，通过宽screen后执行完全独立fresh512。最终C-H8 rho0→C-H14 rho1为reward `.81951→.70292`、outage `128/512=.25000→69/512=.13477`。reward差为`-.11659`，Welch95区间`[-.17289,-.06029]`；outage差为`-.11523`，Newcombe95区间`[-.16282,-.06719]`。C-H14 outage的Wilson95为`[.10789,.16708]`，整个区间都低于双侧工作带`[.18,.22]`，所以不是128条小样本的偶然偏低。
- critic质量却有一致改善：hard-CDF absolute error `.09546→.01678`（降低82.4%），smooth-CDF error降低79.8%，Brier `.19851→.11452`（降低42.3%），mean-cost absolute error `3.4572→.6883`（降低80.1%），crossing略降。这说明trajectory residual的真实风险方向有价值；失败点是rho1把工作点推得过远并损失reward，不是机制未生效。
- 严格裁决为`reject_overconservative_but_direction_useful`。本项目的控制目标是让真实outage约等于0.20后最大化mean reward，不是最小化outage：点估计低于.18且reward下降记为过度保守；只有outage更低同时reward更高才构成Pareto支配。训练末段两条run的批均值都约.22，而fresh终点相差.115，也再次证明B40单批训练outage不能替代独立评估。
- retention guard与该目标无关。它只在上一rollout上用smooth-Brier选择/回滚cost critic参数及Adam状态，不更新lambda、不设outage setpoint，也不给旧批反传；既有C-H10G的fresh reward/outage约`.579/.141`同样属于低收益的过度保守。因此guard继续默认关闭，仅保留为critic遗忘消融，不能拿来追求更低outage。
- 完整history与profile在`_runs/wandb_export/dqc_ch8b40_vs_ch14_mcbalance_rho1_b40_1m_s1_2026-07-17/`和`_runs/profiles/dqc_ch8b40_vs_ch14_mcbalance_rho1_b40_1m_s1_2026-07-17/`；fresh512目录含CSV、JSON和比较图。公开W&B config未发现绝对路径或敏感键；本地export metadata含工作目录是分析溯源信息，未上传到W&B。

### E150：C-H15 rho=.5单变量预注册——在rho0不安全与rho1过保守之间校准工作点（2026-07-17）

- 下一条严格复用C-H14，只把`cost_actor_mc_correction_coef:1→.5`。公式仍为`A=Acritic+rho*(sigma_critic/sigma_residual)*(I-p_hat)`，因此预期MC修正RMS约为rho1的一半；不改PID target/增益、B40、网络、quantile、critic更新数、PPO epoch或学习率。它回答的是一个可辨识的风险credit强度问题，不是同时搜索多个超参数。
- 使用同一压力seed1、B40×25×T1000=1M、C20/A8，持久化后台和脱敏W&B online。预计纯训练约7分钟、内置128约1--2分钟；内部宽screen为reward至少`.60`、outage位于`.05--.40`且全部数值有限，通过才花约3--4分钟做fresh512。
- 机制门要求后段实际修正/critic RMS比大致位于`[.25,1.25]`且不出现非有限值；性能门仍首先要求fresh512 outage点估计进入`[.18,.22]`，然后reward至少不低于同seed 2M带内参考`.79542`，目标是不低于1M的`.81951`。低于.18且reward下降仍是过度保守，高于.22仍是风险不足，不因接近某条训练曲线放宽。
- 若rho=.5进带且reward达到参考，先扩seed0/2，不再继续插值找漂亮seed1；若仍低于.18，停止增加trajectory residual，不把rho=.25默认包装成改进，转PID内部setpoint/控制器同步的独立校准；若高于.22但相对rho0明显改善且reward保留，再把rho=.75作为有边界的插值消融记录，不能同时改B80或PID。若训练继续出现相位循环，则controller--actor同步作为下一条正交实验。


### E151：C-H15 rho=.5——reward基本保留但outage仍在目标带上方（2026-07-17）

- 正式job `DQCAC_DynamicButton_ch15_mcbalance_rho05_b40_1m_s1`绑定预注册提交`9e5a4fb`，由持久化后台正常完成，exit0；W&B run为`z40u5ajt`且已finished。相对C-H14只把rho从1减到.5，其余B40×25×T1000=1M、C20/A8、LSTM512、QR32/MC、GAE-PPO、mean-anchor、obs RMS、PID target=.15和LR完全相同。纯训练`408.3s`，所有数值有限。
- 因果时序正确：lambda为0的前400k与rho0/rho1轨迹逐点相同；44万步lambda首次激活后才分叉。训练仍有明显周期，52万步outage为0，64/84/100万步又分别为`.30/.325/.375`。末20% reward/outage/lambda均值为`.72060/.215/.16282`，outage标准差`.11247`，所以rho减半没有消除闭环振荡。
- 内置128为reward/outage `.8150/.2500`，通过宽screen后完成fresh512。最终rho0→rho.5为reward `.81951→.79999`、outage `128/512=.25000→119/512=.23242`。reward差`-.01952`的Welch95为`[-.07285,+.03382]`；outage差`-.01758`的Newcombe95为`[-.06986,+.03482]`，两项都不能排除随机差异。rho.5自身outage Wilson95为`[.19791,.27092]`；点估计高于`[.18,.22]`，严格未过双侧工作带，但reward超过预注册最低参考`.79542`。
- rho1→rho.5的变化更明确：reward增加`.09707`，95%区间`[.04008,.15406]`；outage增加`.09766`，区间`[.05038,.14460]`。三个端点rho0/.5/1的reward/outage依次为`.8195/.2500`、`.8000/.2324`、`.7029/.1348`，说明配平残差强度确实沿reward--risk前沿移动工作点，但rho=.5仍未把点校准到0.20。
- critic证据相对rho0是混合的：Brier改善5.17%、mean-cost error改善5.37%，hard/smooth CDF error却恶化8.0%/15.8%，crossing增加.0088；相对rho1则所有校准量明显变差。不能把rho=.5写成distributional critic全面改进。
- 关键机制缺口是名义rho与实际贡献仍不相等。rho=.5后20%的`correction_std/critic_std`均值为`.8842`、范围`[.5725,1.5407]`；rho1后20%均值也为`1.4016`。原因是component EMA decay=.1用历史critic尺度配平，而当前critic advantage会快速收缩或扩张。EMA降低单批尺度噪声，却让风险修正相对当前critic发生滞后增益，可能加重相位循环。因此不直接运行EMA-rho=.75；它的实际强度仍不可控。

### E152：current-batch配平实现、回归与C-H16预注册（2026-07-17）

- 新增默认`cost_actor_mc_balance_reference=ema`，逐位保留C-H14/C-H15；显式`batch`时用当前冻结behavior batch的critic/residual标准差构造scale。只要两者高于`1e-4` floor且scale ratio不触及max1，就有`std(correction)/std(critic)=rho`。component EMA仍更新并记录，只是不再驱动batch模式的scale；同一batch后续PPO epoch继续复用首epoch冻结信号。
- 改动前先以seed407生成EMA rho1金样本，改动后不显式设置reference复跑。final及64/128/192四个checkpoint共有176个公共tensor/array leaves逐元素完全一致，最大差0；唯一新增summary路径是`cost_actor_mc_balance_reference=ema`。说明默认路径、随机流、优化器、模型和旧EMA公式没有改变。
- 解析测试使用两轨迹×三时刻，batch rho1实际比为`.99999994`，time-major标签为`[0,1,0,1,0,1]`；第二次PPO式查询的输出完全相同且两个EMA状态不推进。recurrent batch-rho1 smoke训练`7.1s`、exit0，四个checkpoint共172个模块tensor leaves全部有限，首epochratio最大误差`4.57e-5`；独立eval-only成功重建`reference=batch`并加载final checkpoint。
- C-H16严格复用C-H14的rho1，只把reference从ema改为batch；这是单一尺度时序消融，不改rho、PID、B40、网络、critic、PPO或LR。选择rho1而不是事后拟合rho=.67，是因为它把C-H14后段实际比约1.40降低并固定到1.00，同时略高于C-H15的后段实际均值.884；两个既有端点的真实outage位于.135和.232，故该强度有合理机会落在0.20附近。
- 正式仍用seed1、B40×25×T1000=1M、C20/A8、持久化后台和脱敏W&B online；预计纯训练7分钟、内部128为1--2分钟，宽screen为reward≥.60、outage在.05--.40且全部有限，通过才做fresh512约3--4分钟。机制硬门是每个Actor事件实际比与1的误差小于`1e-3`；若floor/max触发则必须显式报告，不能假装精确。
- 性能仍先要求fresh outage点估计进入`[.18,.22]`，再要求reward至少`.79542`，目标不低于`.81951`。通过后扩seed0/2；若精确比仍出现跨带周期或终点失败，则优先controller--actor同步，而不是继续无界扫描rho。保留的分歧路线包括EMA-rho=.75有界插值和B80独立trajectory增量，但前者受本次已观察的增益漂移污染，后者同时改变更新时钟，均不与C-H16混合。


### E153：C-H16 batch-reference rho1——训练周期显著收缩，但fresh策略仍过度保守（2026-07-17）

- 正式job `DQCAC_DynamicButton_ch16_mcbalance_batchref_rho1_b40_1m_s1`绑定提交`ba44093`，持久化后台正常exit0；W&B run为`5x6gm1nz`且finished。相对C-H14只把`cost_actor_mc_balance_reference=ema→batch`，rho仍为1；B40×25×T1000=1M、C20/A8、LSTM512、QR32/MC、mean-anchor、obs RMS、GAE-PPO、PID target=.15及所有LR不变。纯训练`404.1s`。
- 机制硬门远超通过：25个Actor事件的`correction_std/critic_std`范围`[.99999994,1.00000012]`，最大绝对误差`1.19e-7`；首epoch behavior IS ratio误差约`1e-5`，没有floor/max异常、NaN或OOM。名义rho与当前批实际相对贡献终于一致。
- 稳定性得到真实改善。48--76万步的8批outage都在`[.175,.225]`；最后5批均值/标准差为`.235/.03391`，相对EMA-rho1的`.220/.09925`和EMA-rho.5的`.215/.11247`，波动标准差降低约66%--70%。最后一次outage为.175而非单向发散。batch reference因此是有用的减振组件，不因最终性能失败而撤销。
- 内置128为reward/outage `.7947/.1172`；按宽screen完成fresh512后为`.75667/83÷512=.16211`。outage Wilson95为`[.13272,.19653]`，点估计低于`[.18,.22]`；reward95%为`[.72520,.78813]`，上界仍低于目标`.79542`。所以严格裁决是`reject_overconservative_but_stabilizing`。
- 相对rho0基线，reward差`-.06284`的Welch95为`[-.11158,-.01410]`，outage差`-.08789`的Newcombe95为`[-.13694,-.03841]`；两项交换都明确。相对EMA-rho1，batch-reference reward提高`.05375`，区间`[.00102,.10647]`，outage增加`.02734`但区间跨0；它在相同rho下减少过度保守并提高reward，却仍未到目标前沿。
- fresh critic hard-CDF为`.11005`、truth为`.16211`，absolute error`.05206`；Brier/AUC/BSS为`.13933/.5464/-2.57%`，predicted/true mean cost为`6.059/7.525`。它相对rho0的CDF、Brier和mean error分别改善45.5%、29.8%、57.6%，但BSS仍负且AUC偏低，不能宣称条件风险排序已解决。
- C-H16失败后不能把`controller--Actor同步`写成未测试路线：P-M8已完整实现cadence2并跑三seed，只有seed1 fresh520达到`.870/.198`，seed0/2为`.896/.327`和`.814/.225`；P-M9 target=.10与P-M10 window100也分别完成并暴露reward代价。下一条应针对当前batch-reference的过保守setpoint做单变量校准，不重复旧实验。

### E154：C-H17 PID target=.175预注册——把工程安全余量从.05减到.025（2026-07-17）

- C-H17严格复用C-H16，仅将`pid_target_prob=.15→.175`；真实chance constraint、评估alpha和critic查询阈值始终保持.20，`pid_safety_margin`相应从.05变为.025。batch-reference、rho1、B40、网络、QR、PPO、critic更新和所有PID增益不变。它检验的是控制工作点，不是再次改变trajectory risk credit。
- 选择.175而非直接.20，是因为C-H16 fresh outage为.162，距双侧带下界.18约.018；一次减半安全余量是有界校准，保留约.025补偿训练--fresh分布差。历史P-M1也预先记录过.175作为过保守时的路线；这不是从无界网格中挑值。
- 仍用seed1、B40×25×T1000=1M和持久化脱敏W&B online；预计纯训练约7分钟、内部128为1--2分钟，reward≥.60且outage在.05--.40才做fresh512约3--4分钟。机制门要求实际修正/critic比继续在`1±1e-3`、日志PID target准确为.175、IS ratio和数值健康。
- 性能门不变：fresh outage点估计进入`[.18,.22]`后，reward至少`.79542`，目标不低于`.81951`；outage更低不加分。若通过，立即扩seed0/2而不是继续在seed1调target。若仍低于.18，才把target=.20作为一条预先记录的最后边界校准；若高于.22，则.175方向过强，停止setpoint插值。若进带但reward仍低于.795，则说明该trajectory residual主要沿既有前沿换安全，下一步转P-M8与batch residual的正交组合或重新设计状态动作credit，不用更多PID小数点掩盖前沿未提升。

### E155：C-H17结果——严格工作带near-miss，但reward显著提高（2026-07-17）

- 正式job `DQCAC_DynamicButton_ch17_batchref_rho1_pidtarget0175_b40_1m_s1`绑定预注册提交`a009936`，持久化后台正常exit0；W&B run为`pmt9uwtz`且finished。纯训练`454.5s`，内部128回合reward/outage为`.9342/.2734`；通过宽screen后，fresh512回合为reward `.91604`、outage `115/512=.22461`。
- 严格裁决保持预注册口径：outage点估计比工程带上界`.22`高`.00461`，因此不能事后改写为通过，记为`reject_strict_near_miss_high_reward`。但其Wilson95为`[.19059,.26273]`，包含名义目标`.20`；reward95为`[.88075,.95134]`，所以从“约束附近最大化reward”的科学目标看，它是需要跨seed复核的强候选，而不是明显不安全的失败点。
- 相对C-H8 rho0基线，reward提高`.09654`，Welch95为`[.04524,.14784]`，提升具有统计证据；outage降低`.02539`，Newcombe95为`[-.07735,.02674]`，方向有利但尚不能排除零差异。点估计上C-H17同时提高reward并降低outage，形成Pareto改进；这不等价于已经证明跨seed超过基线。
- 相对C-H16 target=.15，reward提高`.15938`且95%区间`[.11204,.20672]`，outage提高`.06250`且区间`[.01410,.11063]`。因此`.15→.175`确实把策略从过度保守端显著推向高reward/高risk端，不是评估噪声造成的假移动。
- 机制门继续通过：25个Actor事件的实际`correction_std/critic_std`都在`[.99999988,1.00000012]`，不存在EMA增益漂移；首epoch IS、PPO数值和全部W&B记录有限。后20% reward/outage/lambda均值为`.8810/.2550/.1815`，PPO clip/KL为`.0995/.00201`。
- 但setpoint放松重新放大了闭环周期。后20% outage标准差从C-H16的`.03391`升到`.10416`，范围`.125--.400`；batch reference只固定风险残差相对critic的当前批尺度，不能消除PID积分、40条二项观测噪声和Actor滞后共同造成的跨批振荡。
- fresh critic hard/smooth CDF绝对误差为`.06201/.06481`，相对C-H8改善约35.0%/30.0%；mean-cost error改善15.7%，Brier仅改善2.2%，BSS反而从`-.0587`降到`-.1152`。因此总体阈值校准更近，但条件风险排序仍弱，不能把reward提升归因于critic已经全面准确。

### E156：C-H17 seed0/2稳健性审计预注册（2026-07-17）

- 因seed1严格超过`.22`，按E154停止setpoint小数插值，不再试`.165/.17/.20`来追单seed。扩seed0/2不是把near-miss事后判成通过，而是验证显著reward提升和接近目标的outage是否可复现；seed1仍保留原始严格失败标签。
- 两个run逐项复用C-H17，仅改随机seed与checkpoint/name/tag；每个仍为B40×25×T1000=1M、C20/A8、batch-reference rho1、PID target=.175、QR32/MC、LSTM512、obs RMS、GAE-PPO和脱敏W&B online。128核、227GiB内存、A100 80GiB当前空闲，两个40-env任务并行预计单run纯训练约8--10分钟、共同墙钟约10--13分钟；若资源竞争使吞吐或数值异常则停止并改串行，不能把并发差异当算法差异。
- 每个seed训练后先做内部128宽screen，有限且reward≥.60、outage在`.05--.40`才做独立512。报告所有seed原值、Wilson/reward区间、seed均值/标准差和相对C-H8同seed结果；不只汇报最好seed。
- 严格稳健性报告同时保留两层：逐seed仍按`[.18,.22]`判带内；组层面检查三seed mean outage是否在该带、变异是否小于C-H8/P-M8，并比较mean reward。只有risk接近目标且reward优势不由单一seed驱动，才把C-H17升级为最终候选；否则转向闭环减振或状态动作risk credit，不继续调seed1 setpoint。

### E157：C-H17三seed结果——risk工作点更一致，但reward优势尚不稳健（2026-07-18）

- seed0/2首次并发启动均在训练前因W&B默认90秒`init_timeout`退出，exit1且env step为0；不是容量、权限、GPU或算法错误。串行完成在线握手并设置`WANDB_INIT_TIMEOUT=180`后，retry run `l9fqp53h/91e81k7l`均finished；seed1为`pmt9uwtz`。上传配置继续经`wandb_public_config`脱敏，评估禁用W&B。失败空run不进入任何性能统计。
- retry只改变外部W&B握手超时和run名，不改变算法配置。seed0/2并发纯训练为`803.0s/779.8s`，显著慢于seed1独占的`454.5s`；共同墙钟仍较串行短，但以后正式训练按仓库纪律独占执行，避免CPU worker竞争。三个run均为1M步、25个Actor/PID事件、无NaN/OOM，后段风险修正/critic标准差比约等于1。
- 独立fresh512结果为：seed0 reward/outage `.94081/110÷512=.21484`，seed1 `.91604/115÷512=.22461`，seed2 `.71253/119÷512=.23242`。严格`[.18,.22]`只有seed0通过；seed1/2分别高上界`.00461/.01242`，不能按“约等于.20”取消逐seed失败标签。
- 三seed C-H17 reward为`.85646±.12526`、outage为`.22396±.00881`（seed sample SD）；合并事件`344/1536=.22396`，Wilson95为`[.20381,.24548]`。相比C-H8的`.76631±.04637/.21094±.04115`，reward均值提高`.09015`，outage均值提高`.01302`；但n=3配对t95分别为`[-.20674,.38703]`与`[-.07727,.10331]`，均跨0，不能宣布跨seed统计显著。
- 分seed相对C-H8的reward差为`+.20634/+.09654/-.03243`。seed0/1的episode-level Welch区间分别为`[+.13580,+.27687]`与`[+.04524,+.14784]`，seed2为`[-.09319,+.02832]`；因此两seed显著提高、弱seed无显著变化，而不是三seed一致支配。outage差为`+.04688/-.02539/+.01758`，每个Newcombe区间都跨0。
- C-H17把跨seed outage SD从`.04115`降到`.00881`，下降约78.6%，说明batch residual+target .175把不同初始化的风险工作点聚到约.224附近；这是比P-M8的`.250±.06796`更稳定的进步。但它仍有约+.024的系统偏差，且reward SD扩大到.125，不能只看聚合均值升级为最终配置。
- fresh critic只有seed0显示可靠的条件信息：seed0 hard Brier/AUC/BSS为`.1649/.6146/+2.25%`，seed1为`.1942/.5583/-11.52%`，seed2为`.1883/.5375/-5.57%`。所以retention guard不启用；它只能依据旧rollout Brier回滚cost critic，不能修复seed2的reward慢学习，也不是outage setpoint控制器。
- 正式多seedCSV/JSON/图位于`_runs/profiles/dqc_ch17_batchrho1_target0175_b40_1m_multiseed_2026-07-18/`，完整W&B history位于对应`_runs/wandb_export/`目录；另有三组逐seed C-H8/C-H17比较图。当前结果仍是1M筛选，远不能和QCPO_refs的5M reward `1.6696`作最终排序。

### E158：C-H18弱seed 2M长度审计预注册（2026-07-18）

- 选择seed2而不是seed0/1，是因为它fresh reward最低`.71253`，但训练末20% reward均值`.61386`、全程斜率`+.70097/M`，仍有慢热可能。用最弱seed检验“1M太短”比继续延长最好seed更保守，也能直接回答用户关于短训练误杀的质疑。
- C-H18从头训练，逐项复用C-H17 seed2，只把`num_iterations=25→50`，得到B40×50×T1000=2M；不从轻量eval checkpoint伪续训，因为该checkpoint不保存完整optimizer/RNG/env状态。PID target=.175、batch-rho1、C20/A8、QR32/MC、LSTM512、obs RMS、GAE-PPO、LR及W&B脱敏均不变。
- 独占资源预计纯训练约14--16分钟，内部128约1--2分钟；通过宽screen才做fresh512约3--4分钟，总计约19--22分钟。使用`launch_background.sh`和W&B online，`WANDB_INIT_TIMEOUT=180`只影响外部握手。
- 长度因果门：2M run的1M checkpoint/history必须与C-H17 seed2在公共网络、lambda、已保存PID/runtime状态和非eval指标上exact或解释所有预期差异；轻量评估checkpoint不保存optimizer，不能声称对未保存状态做过比较。若总预算被scheduler读取而导致前1M不同，则不能把结果称为纯长度延长，必须停止并报告。
- 2M宽screen要求数值有限、reward≥.65且outage在`[.05,.40]`。长度晋级门要求fresh reward至少比1M提高.05到`≥.76253`，且outage点估计进入原严格带`[.18,.22]`；若reward提高但outage> .24，说明只是沿reward--risk前沿变激进，不算训练时长修复。若reward不足或后1M趋势转平/反复，则停止长预算解释，不跑seed0/1的2M。
- 只有弱seed同时通过reward和risk门，才把相同2M预算扩seed0/1；三seed 2M通过后才讨论5M与QCPO_refs统一预算。当前不继续setpoint小数插值，也不与retention guard、IQN、quantile加密或新PID组件混合。

### E159：C-H18弱seed 2M长度审计——训练太短假设成立，但critic仍弱（2026-07-18）

- 正式job `DQCAC_DynamicButton_ch18_batchref_rho1_pidtarget0175_b40_2m_s2`绑定提交`f3f6329`，持久化后台exit0；W&B run `muht6zqr` finished。相对C-H17 seed2只把25×B40=1M改为50×B40=2M，独占纯训练`793.9s`，比14--16分钟预估略快；50个Actor/PID事件、1000 critic updates全部完成，无非有限值。
- 长度因果门完全通过。1M边界checkpoint与原C-H17 seed2的43个module leaves、1个lambda tensor、81个runtime/PID leaves和5个rollout metrics逐位相同，差异数0、最大绝对差0；配置只差`num_iterations`、checkpoint/W&B名称和固定预算tag。故后1M变化可归因于新增长度，不是初始化、并发、代码或调度器。
- 内部128由1M的reward/outage `.7356/.2344`变为2M的`.9457/.2031`，通过宽screen。随后fresh512由`.71253/119÷512=.23242`改善为`.93826/99÷512=.19336`；2M reward95为`[.88216,.99437]`，outage Wilson95为`[.16149,.22980]`。点估计进入原`[.18,.22]`带并接近alpha=.20。
- reward差`+.22574`的Welch95为`[+.15845,+.29302]`，显著通过预注册+.05门；outage差`-.03906`的Newcombe95为`[-.08904,+.01113]`，方向有利但区间跨0。准确结论是“reward提升明确、outage点估计同时改善”，不能说outage下降已统计显著。
- 后20%训练reward均值`.95023`、斜率`+.4382/M`，相对1M末20%的`.61386`明显提高。同期训练batch outage为`.2775±.09966`、lambda `.3127±.1424`，范围仍大；最终fresh安全不代表PID--actor周期已经消失。2M最后几批处在高lambda后的安全相位，继续更长预算可能再次移动工作点，因此必须扩seed而不能只选这个终点。
- 性能改善不是distributional critic全面变准。fresh hard/smooth CDF error从`.02625/.02436`恶化到`.04968/.05338`，mean-cost error从`1.085`增到`3.275`；Brier下降10.5%但因基率变化，BSS从`-5.57%`进一步到`-8.12%`，AUC只有`.5211`。trajectory residual在critic较弱时仍给出风险方向，reward actor靠更长训练恢复；critic仍是超过QCPO_refs的主要算法缺口。
- 正式证据位于`_runs/profiles/dqc_ch17_1m_vs_ch18_2m_batchrho1_target0175_s2_fresh512_2026-07-18/`和`_runs/profiles/dqc_ch18_batchrho1_target0175_s2_full2m_2026-07-18/`，包含fresh CSV/JSON/PNG与完整2M曲线；W&B导出在对应`_runs/wandb_export/`目录。

### E160：C-H18 seed0/1 2M扩展预注册（2026-07-18）

- 按E158成功分支，seed0/1各自从头复用C-H18全部参数，仅改随机seed和输出名；不调PID target、rho、LR、quantile或checkpoint。两条严格串行独占，避免1M多seed时并发把单run从454s拖到约800s；每条预计纯训练13--15分钟、内部128约1--2分钟、fresh512约3--4分钟，两条总墙钟约36--44分钟。
- 两条都要在1M边界与各自C-H17 checkpoint做公共module/tensor/runtime/metrics exact检查。内部数值有限、reward≥.65且outage在`[.05,.40]`后，无论结果漂亮与否都做fresh512；不根据训练末批挑checkpoint。
- 2M三seed主报告以seed为重复单位：逐seed原值、sample mean±SD、paired 1M→2M差和合并episode比例并列。严格晋级要求至少2/3 seed outage在`[.18,.22]`、三seed mean outage也在该带、mean reward≥.90且不允许任何seed reward<.80；同时1M→2M reward平均至少+.05，不能靠提高mean outage超过+.01换取。
- 若严格晋级，C-H18成为2M主候选，再预注册与QCPO_refs统一5M预算；若只在seed2成功或终点随周期分叉，则不直接跑5M，先做末段多checkpoint稳定性/闭环减振。critic AUC/BSS仍作为解释指标，不因策略过门就取消；retention guard、IQN和quantile加密继续作为独立消融，不与长度扩展混合。

### E161：C-H18三seed 2M结果——reward慢热得到确认，但风险命中门失败（2026-07-18）

- seed0/1正式任务均由`launch_background.sh`持久化、独占运行并正常exit0；W&B run分别为`8niycfuh/fhhs0gja`，seed2为`muht6zqr`。三条均为B40×50×T1000=2M、C20/A8、batch-reference rho1、PID target=.175、QR32/MC、LSTM512、obs RMS和GAE-PPO。纯训练耗时分别约793.3/790.5/793.9秒，均低于预估的13--15分钟上沿，全部张量有限。
- 长度因果门在三个seed上全部通过。每条2M run的1M边界与对应C-H17 run比较，43个公共module leaves、1个lambda tensor、81个runtime/PID leaves和5个rollout metrics差异数均为0、最大绝对差0；配置差异仅为总iteration、checkpoint/W&B名称与预算tag。因此1M→2M变化来自新增训练长度，不是初始化、代码版本、并发或学习率调度差异。
- fresh512逐seed结果为：seed0 reward/outage `1.17622/124÷512=.24219`，seed1 `1.00524/136÷512=.26563`，seed2 `.93826/99÷512=.19336`。严格`[.18,.22]`仍只有seed2通过；seed0/1分别高于上界.02219/.04563。不能把outage下降或终点相位当作普遍规律。
- 三seed 2M reward为`1.03991±.12271`，outage为`.23372±.03687`；合并事件`359/1536=.23372`，Wilson95为`[.21324,.25554]`。相对1M的`.85646±.12526/.22396±.00881`，paired reward变化为`+.23541/+.08920/+.22574`，均值`+.18345`；seed-level t95为`[-.01967,.38657]`，n=3仍很宽。paired outage变化为`+.02734/+.04102/-.03906`，均值`+.00977`，t95为`[-.09664,.11617]`。
- reward慢热证据很强：三个seed的episode-level reward差95%区间均为正，且2M所有seed reward均大于.80；因此后续候选不能只凭100--300k或单个1M弱seed淘汰。但预注册主门失败：只有1/3 seed进风险带，mean outage高于.22，故C-H18不直接晋级5M，也不能与QCPO_refs的正式5M终点作最终排序。
- critic结果是混合且不支持“多训练就会学准”。seed0 hard-CDF误差从.00806恶化到.07172、mean-cost误差从.071增到2.264、BSS由+2.25%变为-2.90%；seed2同样恶化。seed1却把hard-CDF误差从.06201降到.00366、mean-cost误差从2.915降到.337、BSS升到+0.82%，但真实outage仍高达.26563。即使总体阈值校准较准，也不等于action-conditioned风险排序和PID--actor闭环能命中.20。
- 三seed末20%训练outage分别约`.265±.084/.308±.059/.278±.100`，都高于内部target .175且持续周期；末段lambda分别约`.289±.088/.461±.073/.313±.142`。batch风险修正/critic标准差比仍逐批约等于1，说明失败不是配平增益漂移，而是有限B40二项反馈、PID记忆、Actor响应滞后及风险credit误差的组合。
- retention guard不改变这个裁决。它只用上一rollout smooth-Brier选择/回滚cost critic和Adam，不更新PID setpoint、lambda或reward actor；既有guard结果`.579/.141`是低reward过度保守。它保持默认关闭，只作遗忘诊断，不能以更低outage作为成功指标。
- 完整证据位于`_runs/profiles/dqc_ch18_batchrho1_target0175_b40_2m_multiseed_2026-07-18/`，包括逐seed/聚合CSV、统计JSON和1M--2M比较图；W&B完整history在对应`_runs/wandb_export/`目录。正式裁决为`reject_5m_risk_hit_rate_failed_but_keep_2m_budget`：保留2M作为后续live候选的最低正式性能预算，但先修风险信用或闭环，不盲目延长。

### E162：C-H19 late-checkpoint相位审计预注册——先判断终点分叉能否用验证集选模消除（2026-07-18）

- 本实验不重新训练、不改变任何权重，只复用C-H18三seed已经保存的pre-update rollout checkpoint。候选固定为1.4M、1.6M、1.8M、2.0M和2.0M final-post-update五个时点；1.0M及更早已经由C-H17/C-H18证明reward欠训练，不纳入“成熟策略”选择，避免用低reward安全点伪装成功。
- 每个候选先在共同但未参与训练的256条validation episodes上评估，三个训练seed都使用相同环境/动作随机流`eval seed=10000`，W&B disabled。候选集在查看validation前固定，不能看到结果后加入1.2M或删除不漂亮时点。三seed共15次eval-only任务，使用`launch_background.sh`持久化；按单512评估约3.7分钟估算，单256约1.9分钟，三条并发一波预计2.5--4分钟，五波总墙钟约13--20分钟。
- 每个训练seed独立采用相同词典序规则：先筛validation outage点估计位于`[.17,.23]`的候选，再在可行候选中选择mean reward最高者；若无可行候选，只选择`|outage-.20|`最小者并标记validation失败，reward仅用于等距决胜。validation用稍宽带吸收256条二项噪声，最终成功门仍是独立test的严格`[.18,.22]`。
- 选模后使用完全不同的`eval seed=20000`做512条fresh test。若选中项不是final-post，则同一test流同时复评final-post作paired工程对照；test不再改选择。最终必须报告每seed选中step/phase、validation reward/outage、fresh test reward/outage、相对final的变化、三seedmean±SD及命中数。
- 这项路线只在“选中checkpoint的fresh test至少2/3进`[.18,.22]`、mean outage进带且mean reward不低于C-H18 final的1.0399”时晋级为可复用checkpoint-selection trick。outage低于.18不算通过；如果validation选模不能跨seed复现，则证明周期不可由低成本终点选择可靠修复，下一步才投入trajectory estimator/闭环算法改动。
- checkpoint选择不能使用test、原final fresh512或训练seed特有的漂亮rollout；validation与test输出分别落到带有公开语义的JSON/log目录，不创建一次性脚本。评估只写vepfs现有`_runs`，预计新增日志/JSON远小于100MB；原checkpoint目录约6.7GB，不复制权重。

### E163：C-H19 validation完成与独立test冻结（2026-07-18）

- 五个候选×三seed共15个validation任务全部持久化exit0；每个严格256 episodes、共同eval seed10000、W&B disabled。三seed按1.4M/1.6M/1.8M/2.0M-pre/final-post排列的reward/outage分别为：seed0 `.966/.270, 1.097/.270, 1.173/.242, 1.197/.215, 1.173/.234`；seed1 `1.149/.324, 1.176/.348, 1.221/.359, 1.030/.285, .973/.309`；seed2 `.961/.293, .885/.262, 1.095/.312, .918/.254, .966/.242`。
- 选择规则在看test前机械执行。seed0只有2.0M-pre落入validation宽带`[.17,.23]`，故选`rollout_step002000000.pt`；seed1无可行候选，按与.20绝对距离最小选择2.0M-pre；seed2无可行候选，选择距离最近的final-post。对应validation reward/outage为`1.19692/.21484, 1.02980/.28516, .96586/.24219`，只有1/3 validation可行。
- validation已经说明checkpoint相位不是完整解法：seed1即使选最接近者仍高目标.085，seed2高.042。2.0M-pre相对final-post在seed0把outage `.234→.215`且reward `1.173→1.197`，在seed1把outage `.309→.285`且reward `.973→1.030`；方向有利，但必须由新随机流复现，不能用validation自身宣布Pareto改进。
- 独立test现冻结为eval seed20000、每项512 episodes、严格工作带`[.18,.22]`。先并发评估三个预先选中checkpoint；由于seed0/1选中pre而非final，再用同一test流各评估final-post作为paired工程对照。seed2的selected即final，不重复计算。三条selected并发预计约9--13分钟，两条control并发约7--10分钟，合计约16--23分钟。
- 晋级规则保持E162：selected test至少2/3 seed进带、seed mean outage进带、mean reward不低于C-H18原final的1.03991，且不能靠任一seed低于.18抵消另一seed高于.22。若失败，停止checkpoint-selection主线；图表仍作为“终点相位真实但不足以稳定控制”的消融证据，下一正式训练转B80独立trajectory batch或正交闭环改动。
- validation候选CSV、冻结selection CSV/JSON和PNG位于`_runs/profiles/dqc_ch19_phaseaudit_ch18_2m_multiseed_2026-07-18/`。selection JSON在test启动前生成，记录validation/test seed、episode数、宽带/严格带、checkpoint绝对来源及选择原因；没有创建一次性脚本。

### E164：C-H19独立test——validation checkpoint选择把策略推向更高reward、更高risk（2026-07-18）

- 三条selected与两条final control均由持久化eval-only正常exit0，test固定eval seed20000、每项严格512 episodes、W&B disabled。selected逐seed reward/outage为`1.26171/133÷512=.25977`、`1.08327/145÷512=.28320`、`.88551/117÷512=.22852`；严格`[.18,.22]`为0/3通过。
- selected三seed reward为`1.07683±.18818`，outage为`.25716±.02744`；合并事件`395/1536=.25716`，Wilson95为`[.23593,.27961]`，整个区间都高于.22。虽然reward高于C-H18原final均值1.0399，但风险明显超支，正式裁决是`strict_fail_stop_checkpoint_selection`。
- 同一test流的final-post control为seed0/1/2 `1.11710/.22070, .98955/.23242, .88551/.22852`；三seedmean reward/outage为`.99738±.11599/.22721±.00597`，合并`349/1536`、Wilson95 `[.20695,.24884]`，同样0/3严格进带。seed0只高上界.00070也仍按预注册失败，不能移动门。
- validation选择的pre相对final在seed0/1的test中分别把reward提高`+.14461/+.09372`，同时把outage提高`+.03906/+.05078`；seed2不变。三seed平均selected-final为reward `+.07944`、outage `+.02995`，seed-level t95因n=3均跨0。它沿高reward--高risk方向移动，没有提高约束前沿。
- 更关键的是风险排序不泛化：validation上seed0/1 pre比final的outage低`.0195/.0234`，独立test却分别高`.0391/.0508`，两seed都翻转。256条validation不足以从高度相关的late checkpoint中稳定识别约.02--.05的真实风险差，继续加checkpoint密度或换validation seed属于过拟合选模，不再尝试。
- 本轮证明终点相位真实但不是可复用解法。final-post共同test的outage SD只有.006却系统集中在.227附近；与其对每seed挑不同终点，不如在训练中降低每次风险观测和trajectory residual的方差，并重新校准闭环工作点。
- 完整validation/test候选、冻结selection、逐seed/聚合CSV、统计JSON和两张PNG均位于`_runs/profiles/dqc_ch19_phaseaudit_ch18_2m_multiseed_2026-07-18/`；`test_checkpoint_selection_audit.png`已解码验证。没有保留临时脚本。


### E165：C-H20 B80固定暴露量预注册——每次风险更新的独立trajectory翻倍（2026-07-18）

- C-H20严格复用C-H18 batch-rho1/target=.175主线，只做“trajectory batch scaling”：`B40×50→B80×25`，总环境步仍2M。critic exposure为`50×40×C20=25×80×C20=40000` trajectory-epochs，actor exposure为`50×40×A8=25×80×A8=16000`，所以不是给B80额外训练样本或epoch；差异是每个梯度/PID事件独立trajectory翻倍、optimizer/controller事件减半。
- PID窗口必须`50→100`作为batch scaling的定义性配套，而非独立window调参。B40/W50成熟时每次40条新样本替换80%窗口并保留10条旧样本；B80/W100同样替换80%并保留20条旧样本，均为1.25个batch。若B80仍用W50，deque会丢掉当前80条中的前30条且不保留旧批，既不公平也不是有效使用B80。Ki/Kp、leak、deadband、delta max、reference episodes和target全部不变；episode scaling保证常值误差下每条trajectory累计控制增益一致。
- 在outage=.20附近，单事件二项标准误从`sqrt(.16/40)=.0632`降至`sqrt(.16/80)=.0447`，理论下降29.3%。current-batch RMS residual也从80条完整on-policy标签估计，仍要求实际correction/critic std ratio为`1±1e-3`。QR32、MC、mean anchor、LSTM512、GAE-PPO、obs RMS、所有LR与PID target .175均冻结。
- 先跑一次B80×T1000完整event工程smoke，覆盖C20/A8、PID、PPO ratio、checkpoint重建与有限性；预计训练30--60秒、16回合评估约1分钟，峰值显存远低于80GB。smoke只裁决OOM/shape/ratio/NaN，不作性能结论。
- 正式首轮选择C-H18风险压力seed1：B80×25×T1000=2M，从头训练、W&B online脱敏、`launch_background.sh`持久化，预计纯训练11--16分钟、内部128约1--2分钟、共同eval seed20000 fresh512约4分钟，总计约17--22分钟。C-H18 seed1同test参考为reward/outage `.98955/.23242`。
- 由于C-H18已证明1M会误杀慢热策略，B80正式run不按前300k/1M reward早停；只有NaN/Inf/OOM、ratio断言或确定工程错误才停止。性能晋级要求fresh512 outage进入`[.18,.22]`，reward至少.94且目标不低于参考.98955；低于.18且reward下降判过度保守，高于.22判风险不足。通过才原样扩seed0/2，不扫描B60/B100/B120或同时改target/rho。
- 若B80只降低训练raw outage抖动却fresh仍失败，说明剩余主因是policy响应/条件risk credit而非独立标签数；下一步转明确的控制器/actor时钟或风险估计器改动。若reward/outage同时改善，则B80作为工程组件晋级，最终仍需三seed2M和统一5M QCPO_refs比较。

### E166：C-H20 B80完整event工程门通过（2026-07-18）

- 持久化smoke使用B80×T1000、C20/A8、batch-rho1、PID target .175/W100和正式LSTM512配置，单个完整event纯训练66.42秒、exit0；80条trajectory、1次Actor事件、1次PID事件全部完成，PID事件确认接收80条完整cost。
- 首个PPO epoch的`max|ratio-1|=1.7166e-5`，远低于1e-3门；没有OOM、NaN、Inf、shape广播或worker泄漏。短评估16回合reward/outage `.054/.062`只验证链路，不进入性能判断。
- checkpoint写入`rollout_step000080000.pt`与`final_post_update.pt`。独立持久化eval-only使用不同eval seed和B4 worker成功重建B80训练配置、加载`phase=post_update_final, step=80000`并完成4回合，exit0；证明num_envs只影响采样批量，不把checkpoint结构锁死在B80。
- 工程门全部通过，按E165启动正式seed1 2M。基于smoke每80k约66秒的保守外推为27.5分钟，但单event包含固定初始化/保存开销且C-H18整段实测约13分钟；正式训练墙钟预估修订为13--24分钟，内部128与fresh512另约5--7分钟。不会用smoke的短策略表现早停正式run。



### E167：C-H20 B80正式结果——反馈噪声下降，但optimizer时钟减半导致reward欠训练（2026-07-18）

- 正式job `DQCAC_DynamicButton_ch20_b80_w100_batchrho1_target0175_2m_s1`由`launch_background.sh`持久化完成，exit0；W&B run `ff9ymksp` finished并同步3个文件。配置为B80×25×T1000=2M、W100、C20/A8、batch-rho1、PID target .175、QR32/MC、LSTM512、obs RMS与GAE-PPO；相对C-H18 B40基线只做E165预注册的batch scaling。纯训练`924.2s`，含内置128评估的后台总墙钟约`1047s`，无NaN/OOM/ratio错误。
- 内置128条reward/outage为`.82593/.26563`。完全独立的共同eval seed20000 fresh512为reward `.833011`、outage `116/512=.226563`，Wilson95为`[.19242,.26478]`。outage仍高于双侧工程带`[.18,.22]`，因此不能因它低于B40就宣布风险通过。
- 同一test流的C-H18 B40 seed1为`.989551/119÷512=.232422`。B80-B40 reward差`-.156540`，Welch95 `[-.218470,-.094610]`，是明确的性能损失；outage差`-.005859`，Newcombe95 `[-.057317,+.045634]`，不能排除零差异。B80既未命中约束，也未保住reward，正式裁决为`reject_no_seed_expansion_but_keep_noise_reduction_evidence`。
- 降噪机制确实存在。全部训练事件的相邻batch outage绝对跳变均值由B40的`.06939`降到B80的`.04115`，约下降40.7%；后20% outage标准差由`.05921`降到`.04430`，约下降25.2%。但后段相邻跳变`.07222→.06875`改善很小，fresh风险也几乎不变；更多并行环境只减少单批Bernoulli噪声，不会自动消除Actor/PID相位周期或条件risk credit误差。
- reward欠训练与optimizer时钟一致。固定2M时B40有50个Actor事件，B80只有25个；每个事件仍是8次full-batch PPO更新，所以Adam step数减半，尽管总trajectory exposure相同。后20% PPO KL由`.002436`降到`.001295`，clip fraction由`.12936`降到`.06133`，ratio std由`.06940`降到`.05062`；reward训练均值由`1.15470`降到`.74490`，fresh reward显著下降。reward value explained variance相近`.7808/.7921`，不支持把主要损失归因于value网络完全失效。
- cost critic没有随B80稳定变准。fresh hard/smooth CDF error相对B40分别恶化32.5%/7.3%，mean-cost error由`.0665`增到`1.3244`，Brier仅改善0.4%而BSS恶化1.42个百分点，crossing增加`.0261`。因此不把critic LR同时放大；先隔离actor时钟，避免一次实验同时改变两条优化链。
- 图表和统计位于`_runs/profiles/dqc_ch18_b40_vs_ch20_b80_batchrho1_target0175_2m_s1_test512_e20000_2026-07-18/`与同名训练profile目录，两张PNG已解码验证；W&B export位于`_runs/wandb_export/dqc_ch20_b80_w100_batchrho1_target0175_2m_s1_2026-07-18/`及B40/B80合并目录。未创建一次性脚本。
- retention guard不参与本结论。guard只以旧rollout smooth-Brier回滚cost critic与Adam，不改变PID、lambda、setpoint或reward actor；既有guard约`.579/.141`是低收益过度保守。全项目统一优化顺序是先让真实outage贴近alpha=.20，再最大化mean reward：`<.18`且reward下降记为过度保守，`>.22`记为预算超支，outage不是越小越好。

### E168：C-H21 B80 actor学习率2倍补偿预注册（2026-07-18）

- 下一条严格复用C-H20 seed1，只把`theta_lr0:3e-4→6e-4`；critic LR仍`1e-3`，B80×25、W100、C20/A8、PPO clip .1、target-KL关闭、PID target/增益、batch-rho1、网络、quantile与2M预算全部冻结。它回答“B80失败是否主要来自固定env-step下Actor Adam step减半”，不是新一轮联合调参。
- 名义累计actor步长由`25×8×3e-4`恢复为`25×8×6e-4`，等于B40的`50×8×3e-4`；这只是Adam下的工程近似，不声称严格等价于把B80分成两个B40 optimizer minibatch。QCPO_refs公开配置默认`minibatches=1, epochs=8`，所以不能把多minibatch误写成参考算法已有trick。
- 先不放大critic LR：B80 fresh critic的CDF/mean-cost/BSS并未改善，且旧实验中更高critic LR有过不利证据。actor-only补偿保持风险估计器可归因；若成功，再讨论把sampling batch与optimizer minibatch显式解耦这一更规范但需代码改造的路线。
- 使用同一压力seed1、持久化后台、脱敏W&B online和B80 checkpoint；依据C-H20实测，纯训练预计15--18分钟、内置128约2分钟，fresh512约4分钟，总计约21--25分钟。C-H18已证明1M可能误杀慢热策略，所以除NaN/Inf/OOM、PPO ratio断言或明显发散外，正式run跑满2M，不因前半段reward低而提前停止。
- 机制门要求首epoch`max|ratio-1|<1e-3`、全部有限、后段KL/clip相对C-H20明显恢复但不出现持续饱和；由于`ppo_target_kl=0`，若KL爆炸只能按预注册工程故障停止，不能事后用early-stop改变配置。正式性能先要求fresh512 outage进入`[.18,.22]`，再要求reward至少`.94`且目标不低于B40参考`.98955`；低于.18且reward下降仍判过保守。
- 若seed1同时通过风险和reward门，原样扩seed0/2；若reward恢复但outage高于.22，说明actor补偿只把策略沿高reward--高risk方向推进，停止LR扫描并转optimizer minibatch/闭环设计；若reward仍低，则B80的不足不只是actor step数，停止B80主线，不尝试9e-4或1.2e-3。无论结果如何，都保留为num_envs与optimizer时钟消融。


### E169：C-H21 B80 actor学习率补偿——恢复PPO步幅但放大闭环振荡，严格失败（2026-07-18）

- 正式job `DQCAC_DynamicButton_ch21_b80_actorlr6e4_target0175_2m_s1`由`launch_background.sh`持久化完成，exit0；W&B run `irwt6awc` finished并同步3个文件。机械比较C-H20/C-H21各74个`--set`参数，算法差异只有`theta_lr0:.0003→.0006`，其余变化仅为name/group/tag/checkpoint路径。纯训练`934.1s`，25个B80 Actor/PID事件、500个critic step全部完成，无NaN/OOM/ratio错误。
- 训练闭环出现比C-H20更大的相位循环：1.04M时reward/outage/lambda为`.658/.300/.160`，1.44M反转为`.550/.113/0`，1.92M又冲到`.981/.412/.392`，2M末批为`.991/.338/.379`。这不是单调趋近0.20，而是高LR让Actor越过工作点后由PID滞后纠偏，再反向越过。
- 后20%训练reward/outage/lambda为`.87687±.10198/.28750±.08216/.17873±.17339`；C-H20为`.74490±.06465/.21500±.04430/.13764±.06005`。reward有所恢复，但outage、lambda方差和周期振幅同时明显增加。后段PPO KL由`.001295→.002437`、clip fraction由`.06133→.12424`、ratio std由`.05062→.06666`，数值已恢复到B40的`.002436/.12936/.06940`量级；机制假设“原B80 actor步幅不足”部分成立，但用LR翻倍补偿不稳定。
- 内置128为reward/outage `.93770/.27344`。统一eval seed20000的fresh512为reward `.909769`、outage `161/512=.314453`；reward95为`[.87334,.94620]`，outage Wilson95为`[.27574,.35593]`，整个风险区间远高于双侧工作带`[.18,.22]`。严格性能门同时失败：outage不进带，reward也低于最低`.94`。
- 相对原B80 C-H20，reward差`+.076757`的Welch95为`[+.020983,+.132532]`，outage差`+.087891`的Newcombe95为`[+.033494,+.141611]`；两者都显著增加，说明它沿更高reward--更高risk方向移动，不是前沿支配。相对B40 C-H18，reward差`-.079782`区间`[-.137966,-.021599]`，outage差`+.082031`区间`[+.027439,+.135985]`；候选被B40在两个主指标上同时支配。
- fresh cost critic仍弱：hard/smooth CDF为`.26514/.26706`而truth为`.31445`，绝对误差`.04932/.04739`；hard Brier/AUC/BSS为`.21803/.5671/-1.14%`，predicted/true mean cost为`10.349/12.254`。相对C-H20，CDF、smooth-CDF、mean-cost和Brier误差分别恶化29.5%/38.4%/43.8%/20.8%；不能把风险超支归因于critic已经给出更准确而Actor单纯更激进。
- 正式裁决为`reject_stop_lr_scaling_no_seed_expansion`。不试9e-4/1.2e-3，也不因训练末reward接近1而忽略outage。结果支持下一条若继续B80，应使用多个较小optimizer step而非一次更大Adam step，并保持整批PID统计；真正trajectory minibatch必须冻结整批old log-prob、GAE与risk weights，不能把现有只累积梯度的`critic_minibatch_size`误当成optimizer minibatch。
- 完整W&B history与三run对齐曲线位于`_runs/wandb_export/dqc_ch18_b40_ch20_b80_ch21_b80_actorlr6e4_2m_s1_2026-07-18/`和同名`_runs/profiles/`。fresh比较CSV/JSON/PNG分别位于`_runs/profiles/dqc_ch20_b80_vs_ch21_b80_actorlr6e4_2m_s1_test512_e20000_2026-07-18/`及`dqc_ch18_b40_vs_ch21_b80_actorlr6e4_2m_s1_test512_e20000_2026-07-18/`；三张PNG均已解码验证，未创建临时脚本。

### E170：C-H22 B80采样、2×B40 trajectory optimizer-minibatch实现与工程门（2026-07-18）

- 新增`optimizer_minibatch_trajectories`，默认`0`严格保留历史整批Adam；正数按完整环境trajectory拆分，每个子批执行独立optimizer step。它与`critic_minibatch_size`不同：后者只把同一个整批loss分块backward并执行一次Adam，不能恢复B80丢失的优化时钟。首轮实现有意只开放已经验证的recurrent/GAE-PPO/actor-feature/MC-QR/online主线，拒绝IQN/NQ、crossfit/preupdate、replay、retention guard、Weibull、shared/adapter、feature refresh、direct CDF、s0辅助、target-KL与critic chunk组合，避免首个性能结果混入多变量。
- recurrent切片以环境trajectory索引为唯一单位：`[T,B,*]`沿B切，`[T*B,*]`先还原time-major再切B并展平，`[B,*]`沿episode切，`[K,T*B,A]` baseline action显式还原为`[K,T,B,A]`。纯张量编号测试选择env `[3,1]`后，state/action/old-logprob/GAE/cost feature/baseline action/h0均得到`[3,1,7,5,11,9]`的正确time-major序列，证明没有把LSTM轨迹或风险标签交叉配对。
- 完整B80上的GAE/value target、rollout `old_log_probs`和cost trajectory标签在采样后冻结；首个actor epoch前，使用完成两个B40 critic step后的同一online critic在完整B80上只查询一次risk CDF、K个behavior baseline action和batch-rho1尺度，只推进一次component/constraint EMA。随后两个B40 actor子批只切片缓存。importance ratio的分母始终是采样时保存的`π_behavior(a|h)`；每个子批step重新计算当前策略分子，不保存上一个更新后的probability作为新分母，否则会把PPO目标错误改成逐step近端链。
- 每个optimizer epoch使用独立CPU generator打乱80条trajectory；该RNG不推进policy Gaussian action噪声。两个B40 critic各自`zero_grad/backward/clip/Adam.step`，并在每个真实step后推进Polyak target与`learning_steps`；两个B40 actor同样各自Adam。PID、outage、trajectory residual尺度和risk EMA仍看完整B80。actor scheduler仍按25个rollout事件推进，不伪装成50个事件；相对B40的50次调度，`b=10000,c=.9`下终点LR差仅约0.22%，作为已记录的残余时钟差，不与首轮同时再改scheduler。
- 默认关闭精确回归使用改动前gold与改动后post同seed503、B4/T32、3 rollouts、C3/A2。四个checkpoint在每个时点的44个公共模块/运行tensor leaves全部逐位相同，final的209个公共旧标量也相同，eval JSON逐项相同；新代码只增加配置/诊断字段，没有改变默认权重、采样RNG、PID或旧指标。
- 启用小smoke使用seed504、B4→2×B2、T32、C3/A2，持久化后台`21.7s`、exit0。每事件实际critic/actor step为`6/4`，三轮actor/PID事件都完成；完整behavior batch首ratio最大误差`4.8161e-5`，末KL/clip为`6.59e-5/0`，四个checkpoint全部有限。独立eval-only重载exit0，评估逐项相同，checkpoint自动恢复`optimizer_minibatch_trajectories=2`。
- 正式规模工程门使用seed505、B80/T1000、C20/A8、`optimizer_minibatch_trajectories=40`，纯训练`66.59s`、exit0；实际完成40个critic Adam和16个actor Adam，PID/actor风险批各80条完整trajectory。首ratio最大误差`1.7166e-5`，末KL/clip为`.001961/.10875`，pre/final checkpoint全部有限，无OOM、NaN、shape或worker错误。它与旧B80整批工程门`66.42s`耗时近似，说明两个B40的optimizer开销被更小的critic pairwise张量抵消；该短run只证明工程可行，不作性能判断。
- 代码改动集中在`agents/dqc_ac_beta_gpu.py`和`run_experiment.py`；新增W&B/JSON键显式报告optimizer子批大小/数量、每事件配置step数、实际actor step数，以及全部子批KL/clip/ratio-std的mean/max。所有测试均用`launch_background.sh`持久化，W&B disabled；没有创建一次性脚本，gold/post/smoke产物保留在对应`_runs/jobs`、`_runs/logs`和`_runs/checkpoints`以便复核。

### E171：C-H22 seed1 2M正式性能实验预注册（2026-07-18）

- C-H22严格复用C-H20 seed1全部算法参数和2M数据预算，仅新增`optimizer_minibatch_trajectories=40`：B80×25×T1000、W100、C20/A8、theta LR `3e-4`、critic LR `1e-3`、QR32/MC、mean anchor、batch-rho1、PID target `.175`、LSTM512、obs RMS和GAE-PPO均冻结。相对C-H21恢复LR到`3e-4`，不把多小步与大单步混合。
- 每个B80 event变为`20×2=40`个critic step和`8×2=16`个actor step；25 events总计1000/400个step，分别与C-H18 B40×50的1000/400完全相同。critic/actor trajectory exposure仍为40000/16000，PID事件仍为25且每次看B80，故实验只测试“低方差完整风险统计 + 恢复B40 optimizer时钟”，不是增加样本、epoch或学习率。Polyak target也按1000个真实critic step推进。
- 使用压力seed1从头训练，W&B online名称/tag仅含公开算法语义，不含路径或隐私；通过`launch_background.sh`持久化。单事件门与C-H20正式耗时共同估算纯训练约15--25分钟，内置128约2分钟、共同eval seed20000 fresh512约4分钟，总计约21--31分钟。根分区不写大文件，全部checkpoint/W&B/log继续位于vepfs工作区。
- C-H18已经证明1M可能误杀慢热策略，因此除NaN/Inf/OOM、ratio断言或确定工程错误外跑满2M，不以早期reward低作性能早停。机制门为首ratio`<1e-3`、40/16 step计数正确、全部有限，后段KL/clip不持续饱和。
- 正式晋级仍按双侧目标：fresh512 outage必须进入`[.18,.22]`，然后reward至少`.94`，目标是不低于同test B40 seed1 `.98955`。`outage<.18`且reward下降记过度保守，`outage>.22`记风险预算超支，不能把更低outage自动算提升。通过才原样扩seed0/2；失败则停止该分支，不扫描B60/B100或第三个minibatch，并转向PID--actor时钟解耦/leave-one-out trajectory风险基线等下一项消融。


### E172：C-H22正式结果——约束命中但reward未恢复，optimizer时钟不是主要瓶颈（2026-07-18）

- 正式job `DQCAC_DynamicButton_ch22_b80_optmb40_batchrho1_target0175_2m_s1`绑定提交`d8ac0dd`，由`launch_background.sh`持久化完成、exit0；W&B run `76q0yofg` finished并同步。相对C-H20只增加`optimizer_minibatch_trajectories=40`，即每个B80 rollout在每个epoch拆成两个完整B40 trajectory子批，各自执行独立Adam step；PID、risk cache、GAE、rollout behavior log-prob和batch-rho1仍在完整B80上冻结。
- 25个event全部完成，每event实际40个critic step、16个actor step，总计1000/400，与C-H18 B40完全相同；首behavior-ratio最大误差`1.7166e-5`，末次聚合KL/clip约`.001961/.10875`，无NaN、Inf、OOM或shape错误。纯训练`1015.4s`，约比C-H20的`924.2s`慢9.9%，其中包含本次较慢的W&B初始化；单event工程门耗时`66.59s`与C-H20的`66.42s`近似。
- 训练闭环没有消除循环。outage从1.04M的`.200`升至1.52M的`.338`，随后降至2M pre-update的`.163`；lambda同步从0升至约`.319`再回到`.164`。末20% reward/outage/lambda为`.8221/.2325/.2533`，只是相位最终落在较安全一侧，不能解释为单调收敛。
- 内置128回合reward/outage为`.8236/.2031`。独立eval seed20000的fresh512由持久化后台正常exit0，得到reward `.812337`、outage `100/512=.195313`，Wilson95 `[.16330,.23187]`；点估计严格进入预注册双侧工作带`[.18,.22]`，但reward低于最低门`.94`，更低于目标B40基线`.98955`，因此按规则不扩seed0/2。
- 相对同测试流C-H20 B80整批，reward变化`-.02067`、Welch95 `[-.07708,+.03573]`，outage变化`-.03125`、Newcombe95 `[-.08111,+.01877]`，两项区间都跨0。恢复Adam step数没有带来可确认的reward提升；它可能把终点风险推低，但不能排除评估噪声。
- 相对C-H18 B40，reward差`-.17721`的Welch95为`[-.23600,-.11843]`，是明确损失；outage差`-.03711`的Newcombe95为`[-.08718,+.01316]`，不能确认真实下降。故“B80 reward差主要因为optimizer step减半”被否定；剩余差异更符合每80条才更新一次Actor/PID、整批risk cache在16个actor step中陈旧，以及PID响应滞后的组合。
- fresh critic仍不能提供可靠条件risk credit：hard/smooth CDF为`.16309/.16715`而truth为`.19531`，hard Brier/AUC/BSS为`.16478/.5241/-4.85%`，mean-cost误差`.31715`。相对B40虽Brier下降9.1%，BSS反而下降3.28个百分点且AUC接近随机；较低总体Brier主要受较低outage基率影响，不能当作critic变好。
- eval-only最初因checkpoint恢复训练时mini-B40、而评估`num_envs=40`触发“mini必须小于full batch”校验。修复`be8b3ab`在eval-only加载后把纯训练参数`optimizer_minibatch_trajectories`重置为0，不改变网络或checkpoint权重；B2烟雾测试和正式B40并行fresh512均exit0。训练checkpoint中保存的16次actor更新诊断仍被保留。
- 四run完整训练profile与对齐图位于`_runs/profiles/dqc_ch18_b40_ch20_b80_ch21_lr2_ch22_optmb40_2m_s1_2026-07-18/`；C-H18/C-H22和C-H20/C-H22的CSV、统计JSON、PNG位于对应`_runs/profiles/dqc_ch18_b40_vs_ch22_.../`及`dqc_ch20_b80_vs_ch22_.../`目录，PNG均已做文件格式验证。正式裁决为`reject_no_seed_expansion_constraint_hit_but_reward_failed`，节省约40--50分钟无效多seed计算。


### E173：C-H23 leave-one-out trajectory empirical baseline实现与工程门（2026-07-18）

- C-H18/C-H22的batch-rho1修正使用`residual=I_outage-p_hat(s,a)`；它没有实现此前文档预留的leave-one-out经验基线。新参数`cost_actor_mc_baseline_mode`默认`critic`逐式保留旧定义，显式`leave_one_out`时，对第j条trajectory构造`b_-j=sum_{k!=j} I_k/(B-1)`和`A_LOO=I_j-b_-j`，再令修正分量为`A_LOO-A_critic`。因此raw eta=1严格满足`A_critic+correction=A_LOO`，不会把完整trajectory估计器直接叠到critic估计器上重复计算风险梯度。
- 自排除不是形式细节。若用含自身标签的全批均值`bar I`作baseline，baseline与本轨迹score相关，有限B期望梯度会缩小为`(1-1/B)`。三条Bernoulli轨迹、policy probability .3的全部`2^3`情况精确枚举中，真实/LOO梯度期望均为`.21`，含自身全批基线只得到`.14`。其它轨迹在固定behavior policy下与第j条环境/动作噪声独立，所以`b_-j`是合法的trajectory-level control variate。
- LOO episode基线先在[B]上计算，再以`[T,B]` time-major广播到`[T*B]`；actor cadence合并batch时使用实际`_actor_num_envs`，不能按构造Agent时的单rollout B切错。样本数校验也放在实际actor batch：这允许`num_envs=1, actor_interval=2`形成两条有效轨迹，并允许任意并行度eval-only；真正B=1且触发LOO训练时才给出明确错误。
- LOO同时兼容raw和rms-balanced。正式候选继续使用batch-reference rho1；scale由当前behavior batch的`std(A_critic)/std(A_LOO-A_critic)`冻结，使实际correction/critic标准差比仍精确为1。新增W&B键报告baseline mean/std及LOO开关；checkpoint config和training summary保存模式，PPO old log-prob、GAE、outage标签与risk cache仍在behavior rollout首次actor查询时冻结。
- 解析张量门使用B3/T2、labels`[1,0,0]`，LOO基线严格为`[0,.5,.5,0,.5,.5]`；改变第0条自身标签不改变其baseline，time-major对齐、raw端点恒等式和batch-rho比1全部通过。另做上述精确期望枚举，LOO无有限B自包含偏差。
- 默认关闭回归复用seed503、B4/T32、3 rollout、C3/A2的已有gold。新run纯训练`21.4s`、exit0；四个checkpoint逐个比较，43个module leaves、1个lambda tensor、31--36个runtime leaves、5/177个公共metrics全部mismatch0，独立eval30个公共字段也mismatch0。唯一新增final metric是新模式字段，不改变任何旧权重、RNG、PID或评估。
- 启用smoke使用seed506、B4/T32、C3/A2、强制阈值-1产生全1事件标签，持久化训练`20.9s`、exit0；43个module leaves全部有限，LOO模式从checkpoint summary恢复，首ratio误差`7.82e-5`，末KL/clip`2.09e-5/0`，每事件2个actor step完成。B2和B1两个eval-only checkpoint重载均exit0，证明训练基线模式不会错误锁死评估并行度。
- 改动只涉及`agents/dqc_ac_beta_gpu.py`和`run_experiment.py`，没有创建一次性脚本；解析检查使用命令行内联代码，回归/smoke/eval全部由`launch_background.sh`持久化。LOO减少的是经验baseline自相关，不保证一定降低`I-p_hat`方差或提升前沿，正式性能必须由2M独立评估裁决。

### E174：C-H23 seed1 2M LOO正式性能预注册（2026-07-18）

- C-H23逐项复用C-H18 seed1：B40×50×T1000=2M、C20/A8、theta LR`3e-4`、critic LR`1e-3`、QR32/MC、actor-feature、mean anchor、sigmoid T1、batch-reference rho1、PID target`.175`/window50、LSTM512、obs RMS与GAE-PPO全部冻结。唯一算法差异是`cost_actor_mc_baseline_mode: critic→leave_one_out`；optimizer minibatch保持0，不混入失败的B80路线。
- 选择seed1是因为同测试流C-H18为reward/outage`.98955/.23242`，具有高reward但风险略超支，能直接检验LOO去偏是否把风险推入目标带而保住收益。C-H18三seed已证明1M可能误杀慢热策略，所以除NaN/Inf/OOM、ratio错误或确定工程故障外跑满2M，不以100--1000k表现提前停止。
- 依据同配置C-H18纯训练`790.5s`，LOO只增加O(B)标量运算；预计纯训练13--16分钟，W&B初始化与内置128约2--5分钟，fresh512约4分钟，总计约19--25分钟。训练使用脱敏W&B online和`launch_background.sh`，checkpoint/log全部写vepfs，不占root大盘。
- 机制门要求50个Actor/PID事件、1000/400 critic/actor step完成，首ratio`<1e-3`，correction/critic标准差比保持`1±1e-3`，LOO baseline mean等于该behavior batch outage mean且全部有限。baseline在episode维的std理论上约为label std/(B-1)，只作索引/自排除诊断，不作为性能门。
- 正式fresh512先要求outage点估计进入双侧`[.18,.22]`，再要求reward至少`.94`，目标不低于C-H18同test的`.98955`。outage低于.18且reward下降仍是过度保守；高于.22是风险预算超支；进带但reward低于.94说明只移动旧前沿。通过才原样扩seed0/2，不针对seed1扫描rho或PID小数。
- 若LOO失败，不把它与含自身全批baseline混淆：后者已有解析有限B偏差，不能作为补救。下一分歧路线保留为①B40中对risk cache做有限次数重新查询并监控目标漂移；②直接让Actor/PID使用更频繁的新on-policy rollout而减少单批epoch；③对trajectory与critic梯度做显式余弦/方差最优组合。三条分别验证，不与本次LOO混跑。


### E175：C-H23正式结果——LOO数学去偏成立，但显著恶化真实outage（2026-07-18）

- 正式job与fresh512均由持久化后台正常exit0；W&B run `wq2076gf` finished。纯训练`811.0s`，50个Actor/PID事件、1000/400 critic/actor optimizer step全部完成，无NaN/Inf/OOM。首epoch ratio误差约`2e-5`，末KL/clip为`.00237/.11805`；LOO baseline逐批均值严格等于batch outage、correction/critic标准差比约1，机制门全部通过。
- C-H23与C-H18在lambda首次响应前逐位一致；200k checkpoint的43个module leaves及rollout指标差异数0。故性能差异来自LOO风险基线，而不是RNG、reward actor、初始化或环境轨迹被意外改变。
- 后20%训练中，C-H18→C-H23 reward由`1.15470→1.00019`，outage由`.3075→.2600`，lambda由`.4614→.2247`。LOO在训练batch上表现得更安全，但reward下降且闭环仍在`.15--.45`间振荡；它没有产生稳定贴近.20的单调收敛。
- 内置128为reward/outage `1.1158/.3281`。固定eval seed20000的fresh512为`1.006335/164÷512=.320313`，reward95 `[.96361,1.04906]`、outage Wilson95 `[.28136,.36194]`，明显高于双侧工作带`[.18,.22]`。
- 同一test流C-H18为`.989551/119÷512=.232422`。LOO-control reward差`+.01678`的Welch95为`[-.04554,+.07911]`，不能确认收益；outage差`+.08789`的Newcombe95为`[+.03313,+.14196]`，是统计显著恶化。因此正式裁决为`reject_no_seed_expansion_outage_significantly_worse`，节省seed0/2约40分钟。
- fresh hard/smooth CDF约`.35193/.35418`而truth `.32031`，Brier由C-H18的`.18120`恶化到`.22300`，mean-cost error由`.0665`增到`.6893`。LOO不训练critic，差异来自被不同风险梯度引导到的新状态分布；不能把critic误差变化误写为LOO监督本身。
- 更关键的机制量：C-H18后段`std(Acritic)≈.00388`、`std(trajectory label)≈.3783`，两者相关约`.0031`；LOO后对应相关约`.00024`。全局LOO基线去掉有限B自包含偏差，却没有保留state/action条件credit；batch-rho只把高方差残差缩到微小critic尺度，不能创造条件排序。
- 完整W&B对齐曲线在`_runs/profiles/dqc_ch18_vs_ch23_loo_2m_s1_2026-07-18/`；fresh512 CSV/JSON/PNG在`_runs/profiles/dqc_ch18_b40_vs_ch23_loo_2m_s1_test512_e20000_2026-07-18/`。分析脚本已补充LOO与optimizer-minibatch指标，提交`5c70940`；未创建一次性脚本。
- 下一主线不继续扫描LOO rho或PID小数点。结构性候选优先级为：①迁移QCPO_refs的recurrent state-value cost distribution与quantile-GAE cost advantage，同时保留DQCAC action-conditioned Q作为可消融分支；②若先做较小改动，则加入state-dependent cost-value baseline替代K动作Monte-Carlo baseline；③最后才做score-weighted梯度余弦/方差最优混合。risk-cache重查询只改变同一批内的critic预测，不足以修复当前近零条件相关，降为机制消融。

### E176：QCPO_refs式state-cost distribution与quantile-GAE风险信用实现、工程门（2026-07-18）

- C-H23把当前根因定位为风险credit缺失：后段action-conditioned critic advantage标准差只有约`.00388`，trajectory标签标准差约`.3783`，相关约`.0031`。本轮因此不再扫描PID、quantile数量或IQN，而是迁移QCPO_refs真正影响Actor的链路：recurrent state-value cost distribution、sorted quantile TD residual和backward quantile-GAE。原action-conditioned distributional Q继续训练和记录，作为诊断与后续消融，不在首轮删除。
- `RecurrentActorValue`新增可选`cost_value_quantiles`与正值cost distribution head；默认0时不创建参数，也不改变任何历史state dict。DQCAC新增`cost_actor_advantage_mode=qcpo_state_quantile_gae`，在behavior rollout冻结的state quantiles上构造one-step QR target、quantile-GAE、mean lambda-return和目标tail quantile advantage。PPO分母仍始终是采样时behavior log-prob，每个epoch只重算当前策略分子；risk advantage也在任何本rollout optimizer update前冻结。
- 首轮严格使用`cost_state_gradient_mode=head_only`：cost head挂在与QCPO_refs同类的recurrent feature上，但cost loss只更新head，不反向改变共享MLP/LSTM；Actor仍可通过冻结的risk advantage更新主干。这隔离“quantile-GAE credit公式是否有效”，避免同时改变representation。若head-only通过，shared-backbone才作为下一条单独消融；若失败，不能事后把共享梯度混入同一run挽救。
- state-cost监督按`cost/10`缩放，默认32个uniform quantiles、QR-Huber κ=1、quantile loss系数1、mean anchor系数.5、cost GAE λ=.97、tail probability .30。目标outage仍由PID围绕`.20`控制；tail probability只是QCPO_refs式Actor风险信用的查询位置，不把“越低outage越好”写进目标。首轮不迁移Weibull density ratio，因为那会同时改变优势幅度和分布假设，保留为核心链路通过后的消融。
- 默认关闭逐位回归已由持久化job `dqc_state_default_reg_s503_20260718`验证：4个checkpoint共172个module tensor leaves、4个running tensor leaves、128个公共运行字段、152个公共checkpoint metrics和215个final JSON leaves，mismatch全部0、最大差0。说明新代码没有改变旧DQCAC的RNG、权重、PID、优化器或独立评估。
- 解析公式门使用T3/B2/N4、非排序预测、γ=.9、λ=.5和独立手算，one-step quantile target与backward advantage最大误差均0，terminal只保留即时cost。梯度隔离门中state-head参数梯度有限且非零（norm约`9.575`），输入feature、MLP和LSTM梯度全部为None。
- 启用smoke `dqc_state_qgae_headonly_smoke_s507_20260718`由持久化后台完成，纯训练`21.2s`、exit0；6个网络全部有限，head末梯度norm`.19956`，总/quantile/mean loss为`.14682/.020879/.251875`，预测/目标均值`1.0539/1.0238`，首epoch`max|ratio-1|=9.20e-5`，2个Actor update均完成。独立eval-only重载同样exit0，checkpoint明确包含cost head权重。短smoke只证明公式、梯度与序列接线正常，不作为性能证据。

### E177：C-H24 600k state-quantile-GAE head-only机制筛选预注册（2026-07-18）

- 正式候选回到C-H18 seed1/B40主线：T1000、C20/A8、theta LR`3e-4`、critic LR`1e-3`、LSTM512、observation RMS、GAE-PPO、QR32/MC action-Q、mean anchor、PID target`.175`/window50/Kp1/Ki.1/leak.97/deadband.02。唯一核心替换是Actor风险credit从action-CDF+batch-rho改为E176的state quantile-GAE；旧Q critic仍训练但不驱动该分支Actor。
- 采用1个40k warmup rollout：它只训练reward/cost critics和新state-cost head，不更新Actor或PID，使随机初始化的state head先获得一批监督。之后14个rollout进入Actor/PID，总环境步600k。这个初始化差异是新head可用所必需，明确记录而不伪装成与C-H18完全单变量；若晋级2M，保持同一warmup配置。
- 600k只作机制门，不能因早期reward低就判最终性能失败。继续到2M的条件是：全部有限、首epochratio误差`<1e-3`、state-head loss/误差没有持续爆炸、state risk advantage保持非退化方差、Actor/PID对风险信号有可解释响应且PPO不过度持续clip。即使600k reward尚未超过C-H18，也只要机制健康就继续；只有NaN/OOM、序列/ratio断言、head发散、risk advantage塌缩或完全错误方向才提前停止。
- 600k采用持久化`launch_background.sh`、脱敏W&B online、每200k checkpoint；根据旧C-H18 2M约13.2分钟及新head额外recurrent forward，预计纯训练6--10分钟，内置128评估后总计约8--13分钟。若晋级，2M单seed预计20--30分钟，fresh512约4分钟；通过双侧outage带`[.18,.22]`且reward不降后才扩seed0/2。
- 若head-only机制健康但2M性能不通过，只允许再做一条QCPO_refs式shared-backbone消融；其余IQN、局部quantile加密、N=64、Weibull和PID细扫全部暂缓。这样本阶段只解决关键问题，不继续铺开低优先级组合。
