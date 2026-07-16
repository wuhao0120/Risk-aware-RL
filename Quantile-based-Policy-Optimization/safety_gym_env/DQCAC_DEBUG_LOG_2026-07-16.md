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
