# Safety-Gym 中 QCPO / DQCACBeta 排查与改进总结

**日期：2026-07-18（UTC）**

## 一句话结论

DQCACBeta最初“均值和quantile基本不涨、剧烈波动”既有工程问题，也有算法问题：观测未正确归一化、多次Actor更新缺少固定behavior策略的PPO/importance-ratio约束、reward credit过弱等工程问题已经修复；修复后reward可以稳定上升。但最终限制性能的不是quantile数量或PID，而是**action-conditioned风险信用几乎没有有效排序信息**。本轮最后迁移的QCPO_refs式共享MLP+LSTM、state cost distribution和quantile-GAE能够训练，却把策略推到高reward、高outage前沿，未超过QCPO_refs，也没有满足outage约0.20的目标。

## 最重要的结果

统一使用DynamicButton、训练seed1、2M environment steps及独立eval seed20000的512回合测试：

| 配置 | 风险信用 | fresh reward | fresh outage | 与目标0.20的关系 | 裁决 |
|---|---|---:|---:|---|---|
| C-H18 当前稳定控制 | action-Q + trajectory batch-rho | 0.989551 | 0.232422 | 略高0.0324 | 保留为当前DQCAC基线 |
| C-H25 | state quantile-GAE，head-only | 1.608736 | 0.470703 | 高0.2707 | 拒绝 |
| C-H27 | state quantile-GAE，共享MLP+LSTM | 1.613182 | 0.494141 | 高0.2941 | 拒绝 |

C-H27相对C-H18的reward增加`0.623631`，Welch 95%区间`[0.562545, 0.684716]`；但outage也增加`0.261719`，Newcombe 95%区间`[0.203929, 0.317002]`。这不是约束性能提升，只是策略变得更激进。C-H27相对C-H25的reward差`0.004446`、outage差`0.023438`，两个95%区间都跨0；共享表示没有显著改善head-only前沿。

![三种2M训练曲线](_runs/profiles/dqc_ch18_ch25_headonly_ch27_shared_2m_s1_2026-07-18/overview.png)

![C-H18与共享路线的fresh512比较](_runs/profiles/dqc_ch18_vs_ch27_stateqgae_shared_2m_s1_test512_e20000_2026-07-18/comparison.png)

## 做过的代码改动

### 1. 修复训练不涨的工程链路

- 修复并统一observation running mean/std，使高维Safety-Gym输入真正归一化。旧DQCAC的主要早期故障之一就是未归一化观测导致网络输出和梯度尺度不稳定。
- reward Actor改为GAE + PPO clip；多次Actor update始终使用rollout时保存的`old_log_prob`作分母，计算`pi_current/pi_behavior`。分母在本批8个epoch内不更新。
- 加入learnable action std、LSTM序列训练、reward value baseline、gradient clipping和更完整的W&B诊断。
- QCPO也做了相同方向的行为校准：observation RMS、GAE/value、固定behavior probability的PPO clip，避免同一轨迹重复更新造成无约束off-policy漂移。

关于importance ratio，正确答案是：**每次rollout存一次behavior log-prob；同一批数据的所有epoch都对它计算ratio。** 每次参数更新后保存新的当前概率并替换分母并不正确，那会把标准PPO变成逐epoch近端链，隐藏当前策略相对真实采样策略的累计漂移。本轮所有正式run的首epoch`max|ratio-1|`都在约`5e-5`以内，说明这条链路正常。

### 2. 对齐QCPO_refs网络与风险组件

- 策略从纯MLP对齐为`[512,512] tanh + LSTM512`，输入包含观测及cost history；reward policy/value共享recurrent feature。
- 增加recurrent state-cost quantile head，支持`head_only`和`shared_backbone`两种梯度模式。
- 实现QCPO_refs式sorted quantile TD residual和backward cost GAE；rollout结束后、任何optimizer更新前冻结cost advantage和监督target。
- `shared_backbone`中，policy loss、reward-value loss、state-cost QR loss和mean-anchor loss在同一PPO epoch联合反传，cost梯度能到达MLP、LSTM和cost head，不创建第二个重复优化共享参数的optimizer。
- 新增独立`cost_state_discount`。共享正式路线用`.99`与QCPO_refs Bellman/GAE折扣一致；旧head-only默认保持历史cost discount，不污染回归。
- 增加state advantage与真实trajectory outage标签相关性、MLP/LSTM/head梯度范数、PPO KL/clip、Brier/AUC/BSS、crossing等诊断。

核心实现提交：

- `86f6230`：共享recurrent state-cost quantile训练与工程门。
- `bc03067`：600k机制筛选和2M预注册记录。

## 为什么说代码已经工作，但算法仍不够

C-H27的384步smoke中，纯state-cost梯度到达MLP/LSTM/head的范数分别为`0.26405/0.06606/0.30993`，checkpoint重载正常；head-only旧路径逐tensor回归完全一致。600k时，state advantage与同trajectory outage标签的后段相关性达到`0.2646`，scaled mean MAE降到`0.0478`，所以它通过机制门并按约定跑到2M，而不是被短跑误杀。

2M后，所有数值仍健康：

| 机制指标 | C-H27 2M结果 |
|---|---:|
| 纯训练时间 | 813.3 s |
| 后20% state advantage/outage相关 | 0.1890 |
| 最后相关 | 0.2239 |
| 后20% state risk std | 0.1363 |
| 后20% body/LSTM/head纯cost梯度范数 | 0.01530 / 0.00698 / 0.00276 |
| 后20% PPO KL / clip fraction | 0.000677 / 0.02698 |
| 最大首epoch ratio误差 | < 4.953e-5 |
| 后20% reward / outage / lambda | 1.585 / 0.530 / 1.359 |

问题在于“状态危险”不等于“动作责任”。state-value distribution估计从当前历史状态沿现策略继续走的风险；它不能单独判断同一状态下动作A与动作B谁更容易导致未来违约。旧action-conditioned critic在C-H18后段的风险优势标准差只有约`0.003883`，而trajectory二元标签约`0.3783`，两者相关约`0.00312`。也就是说，真正用于区分动作的信号近似为零且近似随机。

state quantile-GAE把风险信号放大了，共享cost训练也让状态表征更有风险信息，但动作归因仍不充分。因此lambda即使升到1以上，Actor仍沿高reward、高outage方向移动。C-H27的fresh critic AUC只有`0.5685`，Brier为`0.2813`、Brier skill为`-12.53%`，也支持其条件概率排序仍弱。

## 各候选组件的结论

### PPO / IS、GAE、观测归一化

有用，应该保留。它们修复了训练不涨和多epoch漂移，是当前所有可运行配置的基础，但不能独立解决风险动作信用。

### MLP + LSTM

为了与QCPO_refs公平对比应该使用，本项目已经对齐。LSTM对部分记忆状态和reward学习有帮助，但不是“换上就稳定”的trick；最终shared LSTM路线仍出现高outage。

### PID lambda

有用但不是根因修复。当前经验outage PID使用内部target `.175`，目标是抵消闭环偏差后让真实outage靠近`.20`，不是让outage越低越好。配置为window50、`Kp=1, Ki=.1, leak=.97, deadband=.02, max_delta=.05`。它能响应outage，却受风险credit、有限B40二项噪声和Actor滞后影响而周期振荡。网络不同会改变被控对象增益，所以QCPO_refs的PID数值不能直接无条件迁移；但当前失败不是再细调PID即可解决，因为lambda升高时策略风险方向仍错误。

### num_envs

`num_envs=B`表示每个rollout并行采集的独立完整轨迹数。硬件可以承受B80，但在固定2M步下，B40有50次Actor/PID新数据事件，B80只有25次。B80将相邻outage噪声约降低40.7%，却减少控制器和on-policy刷新时钟；实际fresh结果B80约`.8330/.2266`，低于B40的`.9896/.2324`。将B80拆成B40 optimizer minibatch可把outage推到`.1953`，reward仍只有`.8123`。因此当前推荐B40，不盲目开大。

### quantile 32→64与查询点局部加密

已经做过固定策略600k公平验证。N64使用`reference_mean`归一化，避免quantile数翻倍时QR梯度也翻倍。独立140回合中，N64把总体CDF绝对误差从`0.01071`降到`0.00145`，但mean error由`0.77573`恶化到`0.87055`，crossing由`0.05737`升到`0.15828`，末段prequential CDF也未改善。局部加密的CDF/Brier改善只有约0.36%量级且未过门。结论是：它们可能改善一个查询点，却没有一致改善条件分布或策略信用。

### IQN和平滑CDF

uniform-IQN在冻结策略600k时相对QR把独立CDF/mean误差改善约`16.1%/19.4%`，但低于预注册25%门，且crossing从`0.0574`恶化到`0.2340`；没有进入live闭环。平滑CDF主要降低离散quantile计数跳变，属于诊断/数值平滑，不会自动产生动作排序。它们可以作为未来action-credit修复成功后的表示消融，不是当前第一优先级。

### Brier、retention guard与Weibull

- Brier是预测概率与0/1结果之间的均方误差，越低越好。本轮只用于诊断，没有加入Actor或critic loss，不是新控制机制。
- retention guard只在holdout Brier恶化时回滚cost critic，防止遗忘；它不更新PID setpoint。已有结果reward/outage约`.579/.141`，属于过度保守且低reward，不符合“outage约0.20后最大化reward”的目标。
- Weibull在QCPO_refs中用于尾部分布拟合和正权重缩放。它能改变风险优势幅度，但本轮失败是方向而非幅度；在核心共享路线失败后没有继续消耗正式预算。

## QCPO与QCPO_refs

旧QCPO的主要问题与DQCAC早期相似：同一轨迹重复Actor更新却没有固定behavior ratio，且observation normalizer没有真实更新。补上obs RMS、GAE/value和PPO clip后，MLP QCPO 300k后段reward由约`-0.1139`提升到`0.2297`，行为恢复正常；但轨迹级indicator仍天然高方差、样本效率低。

QCPO_refs稳定性的来源不是某一个神奇组件，而是组合：共享MLP+LSTM状态表示、state cost distribution、quantile TD/GAE、PPO clip、reward value、observation normalization和PID闭环。我们迁移的共享核心证明这些组件能改善状态级风险学习，但DQCAC若要超过它，仍必须保留并真正学好action-conditioned风险差异。

现有QCPO_refs正式5M结果约为reward/outage `1.6696/.1538`。它reward更高，但outage低于目标0.20，偏保守；而DQCAC目前不是更优前沿。因为预算和seed并未完全统一，不能把该单点写成最终统计排名，但足以否定“当前DQCAC已经超过refs”。

## 当前推荐配置

如果现在需要一个可复现的DQCACBeta主配置，推荐C-H18，而不是高reward但严重违约的C-H25/C-H27：

```text
env=DynamicButton, cost_limit=15, target_outage=0.20
num_envs=40, horizon=1000, train_steps>=2M
policy_arch=mlp_lstm, hidden=[512,512], lstm_hidden=512
observation_normalization=true
reward_actor_mode=gae_ppo, ppo_clip=0.1
actor_lr=3e-4, actor_epochs=8, critic_epochs=20
num_quantiles=32, cost_distribution_model=qr
cost_history_mode=actor_feature, num_action_samples=4
cost_mean_anchor_coef=0.5, cost_scale=10
cost_critic_time_weighting=risk_discount, beta=0.995
cost_actor_mc_correction_coef=1.0
cost_actor_mc_correction_mode=rms_balanced
cost_actor_mc_balance_reference=batch
dual_update_mode=empirical_pid, pid_target_prob=0.175
pid_window=50, Kp=1.0, Ki=0.1, leak=0.97
pid_deadband=0.02, pid_max_delta=0.05
```

C-H18的2M三seed fresh512结果是reward `1.03991±0.12271`、outage `0.23372±0.03687`。它仍没有稳定命中严格`[0.18,0.22]`，所以应称“当前最佳可复用DQCAC基线”，不能称约束已经解决。

## 下一步只建议做什么

如果继续研究，最高优先级只有一个：**构造可验证的action-conditioned causal risk advantage**。一个合理最小路线是共享recurrent表示上的distributional `Q_c(s,a)`与state `V_c(s)`联合训练，用`Q_c(s,a)-V_c(s)`提供动作差异，同时保留trajectory score-function项作无偏校正。正式训练前先做固定策略/动作扰动验证：同一历史状态采多个动作，检查风险优势是否能预测未来违约差异，并要求跨seed AUC、Brier和score-weighted gradient相关稳定高于随机。

只有这个门通过后，才值得依次做：non-crossing quantile/IQN或查询点加密、Weibull尾部、PID再整定和B80降噪。否则这些组件只会更精确地拟合一个没有动作因果信息的风险量。

本阶段按最初约定路线已经停止扩展：没有继续跑C-H27 seed0/2或5M，也没有把IQN、Weibull和PID扫描混到失败配置中。所有正式训练均通过`launch_background.sh`持久化运行，W&B配置经过脱敏，数据和checkpoint写入`/vepfs-mlp2/c20250510/251204033/`下；没有保留一次性临时脚本。

## 证据位置

- 2M训练曲线：`_runs/profiles/dqc_ch18_ch25_headonly_ch27_shared_2m_s1_2026-07-18/`
- C-H18 vs C-H27 fresh512：`_runs/profiles/dqc_ch18_vs_ch27_stateqgae_shared_2m_s1_test512_e20000_2026-07-18/`
- C-H25 vs C-H27 fresh512：`_runs/profiles/dqc_ch25_headonly_vs_ch27_shared_2m_s1_test512_e20000_2026-07-18/`
- 完整W&B导出：`_runs/wandb_export/dqc_ch18_ch25_headonly_ch27_shared_2m_s1_2026-07-18/`
- 详细逐实验记录：`DQCAC_DEBUG_LOG_2026-07-16.md`
- 机制诊断与决策：`safety-gym-dqcac-diagnosis-2026-07-14.md`
