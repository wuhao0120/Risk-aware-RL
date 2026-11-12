# safety_gym_env DESIGN —— Reward-Side Risk-Aware RL（当前实现）

> 当前版本：2026-08-01。默认 DQC 仍为 MLP-only `online_1step_v1`；另新增显式非默认
> `finite_nstep_v1`。下方历史记录保留已完成的消融与旧实验，仅供追溯。

## 当前研究对象与工程近似

默认 continuing 分支的理论对象是 `G_infinity=sum gamma^t r_t`，`gamma=0.99`，训练和正式评估使用
原始 reward scale 的 `G_1000=sum_{t=0}^{999} gamma^t r_t` 作为截断近似。finite 分支则把
`G_1000` 本身定义为优化对象。两者统一直接约束为
`P(G_1000 <= q) <= alpha`，首轮 `alpha=0.2`，等号计入 failure。`q` 是用户为每个环境明确
选择的固定 raw-return threshold，不是当前策略分位数。经验分位数统一使用 NumPy 线性插值
`np.percentile(G, 100*alpha, method="linear")`，只作 QPO/QCPO_REF/诊断，不能替代直接
failure 计数判断可行性。

默认 continuing one-step 分支中 `terminated` 不 bootstrap，`truncated` 从旧 episode 的
`final_observation` bootstrap；finite 分支在 `terminated` 和第 1000 步 horizon `truncated` 都将
尾项严格置零，只在 n-step endpoint 仍属于同一 finite episode 时 bootstrap。经验 `G_1000`
本身不添加 critic bootstrap。vector adapter 会分开返回 reset observation 与 final observation，
collector 在某 slot 边界后把该 slot 标为无效，禁止拼接 reset 后的新 episode。

Safety Gym cost 仅记录为 `diagnostic/environment_cost_*`。当前 active actor、critic target、dual、
可行性和 checkpoint 选择都没有 cost 输入。

## 当前算法身份

- `QPO`：纯 `max Q_0.2(G_1000)` two-timescale baseline；无固定 q、mean objective 或 dual。
- `QCPO`：direct trajectory Monte Carlo，actor 权重为 `A_mean-lambda_p*A_failure`，经验
  `P(G_1000<=q)` 驱动无动量 projected dual；每个 fresh rollout 只做一次 actor update。
- `QCPO_REF`：reward lower-quantile baseline，PPO 使用 `A_reward+lambda_Q*A_quantile`；critic 是
  state-only `[512,512]` Tanh MLP，不含 LSTM，输出 mean value、signed monotone quantiles 和
  三参数 lower-tail Weibull；dual residual 为 `q-Q_alpha`，经验值使用线性插值。
- `DQCAC`/`DQCACBeta`：`online_1step_v1` 单 reward distribution critic；reward budget
  `b_next=(b-r)/gamma`；actor 使用 `gamma^t A_mean-lambda_p beta^t A_CDF`，默认
  `beta=0.995`。每批只遍历当前 rollout 一次并做一次 score-function actor update；dual 只用
  完成轨迹直接计数。live 实现没有 replay、n-step、PPO、GAE 或 advantage normalization。
- `DQCACBeta` 的非默认 `finite_nstep_v1`：严格优化 `G_1000`，critic 使用 remaining-time 与
  detached actor `h,c` 上下文，100-step distributional target、epoch 末 hard copy、10-batch
  fixed-policy fitted warmup，并固定 `beta=1`。其完整定义见下方专节；默认分支和历史 checkpoint
  行为不变。
- 历史 `DQCACBetaLSTM` 消融只把独立 MLP critic 换成 recurrent distribution critic，其余
  rollout、actor、one-step target、dual、beta、预算和评估保持一致。实现与 checkpoint 留在
  `ca306c1` 供追溯；active CLI/export/test runner 自 `8dbcc48` 起不再包含该算法。

所有 reward-side actor variant 统一使用 observation RMS → FC512×2 → LSTM512 → tanh mean，learnable log_std 初始
`sigma=1`。collector 保存 `A_raw`（未截断 Gaussian 样本）和
`A_exec=clip(A_raw, action_space.low, action_space.high)`；`A` 只是其他算法兼容用的 raw alias。
DQC 仅在 actor log-prob 使用 `A_raw`，环境实际动作及全部 critic query 使用 `A_exec`。硬 clip 是
many-to-one Box 映射，不能把普通 Normal log-prob 错算在 `A_exec` 上。

## DQC 当前 estimator 审计（2026-07-30）

当前 DQC advantage **不是 trajectory LOO，也不是 GAE**。对 critic atoms `z_j(s_t,a_t)`：

```text
Q_mean(s,a) = mean_j z_j(s,a)
Psi(s,a,b)  = mean_j 1{z_j(s,a) <= b}
V_mean(s)   ~= mean_m Q_mean(s, a_m),        a_m ~ pi(.|history_t)
V_CDF(s,b)  ~= mean_m Psi(s, a_m, b),        M=4
A_mean      = Q_mean(s,a_exec) - V_mean(s)
A_CDF       = Psi(s,a_exec,b) - V_CDF(s,b)
A_DQC_t     = gamma^t A_mean - lambda_p beta^t A_CDF
```

因此它是 **同一 history 下独立 policy-action Monte Carlo 积分得到的 action-marginal baseline**。
QCPO 的 LOO 则在同 batch 的其他完整 trajectories 之间构造 baseline；QCPO_REF 的 GAE 则从
`delta_t=r_t+gamma V(s_{t+1})-V(s_t)` 反向递推。三者不能混称。当前 DQC 的 mean/CDF
advantage 及组合权重全部 detach，既不分别 normalization，也不做 combined normalization。

actor 与 dual 使用两个职责不同、刻意不混用的 failure estimator：

- actor risk credit assignment 使用 critic CDF `Psi_phi(s,a,b)` 及其 action baseline；
- dual、feasible、checkpoint selection 使用完成轨迹的直接计数
  `mean_i 1{G_1000_i<=q}`，不信任 critic 自报概率。

这种 hybrid 是当前推荐口径：critic 提供逐 transition、较低方差的方向；经验概率守住真实约束。
若 actor 也只用完整轨迹 indicator，算法会趋近 direct-MC QCPO；若 dual 改用 critic CDF，critic
失准时可能出现真实 violation 很高但 lambda 不响应。

actor 在完整 `[T,B]` recurrent rollout 上重新计算 `log pi(A_raw|history)`，损失固定除以
`B*1000`，每个 rollout 只执行一次 Adam step。critic target 在 epoch 前一次性 detach；全部 valid
transition 经一次无放回 `randperm` 覆盖后才做一次 Polyak update。truncated target action 使用旧
episode 的 `final_observation` 与清零前 actor context，普通 transition 直接复用下一步 `A_exec`。

### 显式 fitted one-step warmup 诊断

`critic_warmup_mode` 是 checkpointed debug identity，不改变 active 默认：

- `legacy`（默认）要求 `warmup_critic_epochs=1`，逐位保留原 `online_1step_v1` 行为：warmup
  只跳过 actor Adam/dual，RMS 与绝对 batch-index scheduler 仍推进，critic 每批一遍后 `tau=.005`；
- `fitted_hard` 只允许 `critic_target_source=one_step`。在 `iteration<warmup_iters` 时完整冻结
  actor 参数/log_std、observation RMS、actor Adam/param-group LR、scheduler、dual 与全部相关 counters；
  Bellman labels 只构造一次，在 `warmup_critic_epochs=E` 个无放回 regression epochs 内 target
  逐位不变，所有 Adam minibatches 结束后才执行一次 `target <- online` hard copy；
- warmup 结束后的第一个 actor batch 先把 LR 同步到绝对 batch index，再执行真实 actor Adam step；
  不伪造 optimizer/scheduler step。随后恢复原来的 rollout 末 RMS/LR 更新顺序。

该模式仍然是 one-step distributional TD，不是 finite-suffix MC，也不会让 diagnostic MC 指标进入
dual、feasibility 或 checkpoint selection。`critic_warmup_mode` 与 `warmup_critic_epochs` 同时进入 W&B
config、checkpoint signature/state/summary；旧 state 缺字段时只按 `legacy/1` 解释，跨 mode/E resume
即使允许普通 signature transfer 也会在任何 live mutation 前失败。实验已表明 fitted-hard 会加速
conditional atom/CDF collapse，因此它保留为可复现的非默认否证分支，不是推荐训练设置。

## DQC online-first MLP baseline 与 LSTM 单组件消融

`pure online` 在这里定义为：每 batch 仍采 25 条完整 trajectory（约 25k transitions），critic、
target 和 optimizer 跨 batch 持久化，但 critic 只从当前 rollout 抽样更新，batch 后丢弃 transitions；
actor 始终只使用 fresh rollout。它不等于单 transition 更新，也不重新初始化 critic。

推荐基线 `DQCACBeta-Online-Core` 固定公共变量：`B=25,T=1000,iterations=200,gamma=.99,
alpha=.2,q(SimpleButton)=-.03,actor_lr=2e-4,beta=.995`。算法侧固定 `atoms=32`、critic
`[512,512]`、`critic_lr=3e-4`、one-step TD、critic minibatch 1024、critic epochs 1、
`tau=.005`、warmup 5、action samples 4、combined normalization off、actor/critic grad clip 10、
`lambda_init/max=0/50`、经验 dual 初始步长 `.01`。

critic 每 batch 只使用当前 fresh rollout，所有 valid transitions 无放回打乱并恰好遍历一次，随后
丢弃；target critic 在整个 critic epoch 内冻结，epoch 结束后只做一次 Polyak update。actor 对完整
recurrent rollout 只做一次 score-function optimizer step。clean baseline 不使用 replay、n-step、
PPO ratio/clipping、多 actor epoch 或 GAE；这些均作为独立后续组件，不能混入核心有效性验证。

本轮 clean baseline 的 distributional critic 固定为 feed-forward MLP：
`concat(observation, action_exec) -> Linear(512)+ReLU -> Linear(512)+ReLU -> 32 atoms`。
它不读取 actor 的 `h,c`，也不维护自己的 recurrent state；因此当前 rollout 的 valid transitions
可以无放回打乱后做普通 critic minibatch。该版本已作为 commit `ea2194a` 固定，当前 active
实现也只包含这一 MLP 路径。历史 critic-LSTM 曾以独立 identity 加入 `ca306c1`，完成消融后由
`8dbcc48` 从 active code 删除，没有改写 MLP baseline 的历史。

除本次 LSTM 单组件消融外，后续组件仍必须逐项加入：

1. correctness gate：lambda=0 reward-only、固定 policy critic calibration、boundary/action/history；
2. online-core，一环境一 seed 验证 reward、CDF 和 dual；
3. fresh PPO，多 epoch 是唯一变化；
4. 公平 replay 消融先比较 `online n=1` 与 `replay n=1`，actor 仍 fresh；
5. 再研究 sequence replay、burn-in、behavior log-prob 与 Retrace/V-trace 类截断校正；
6. 最后逐项消融 action samples、atoms、beta=1、prioritized replay、mean-GAE；recurrent critic
   按下方独立组件实施，不与这些组件合并上线。

### 历史已完成消融：独立 recurrent distributional critic

这一组件不属于 MLP clean baseline，而是独立 `DQCACBetaLSTM`。它要回答的问题是：当 observation 不是充分 Markov
表示，或 recurrent actor 使未来行为依赖历史时，显式条件化到 history 的
`Z(history_t, action_t)` 是否改善 mean/CDF advantage、quantile calibration 和最终约束表现。

推荐结构使用 critic 自己的参数和 `h,c`，不复用或拼接 actor 的 `h,c`：

```text
critic_input_t = concat(observation_t, previous_action_exec_t)
critic_context_t = CriticLSTM(encoder(critic_input_t), critic_context_{t-1})
atoms_t = QuantileHead(concat(critic_context_t, current_action_exec_t))
```

当前 action 放在 recurrent core 之后，保证实际 action 和 action-marginal baseline 的候选 actions
共享完全相同的 history context。actor 与 critic 的 recurrent state 相互独立，critic loss 不得
反向更新 actor。

状态与 boundary 语义按 episode 而不是 optimizer update 定义：

- 每条 trajectory 开始时 actor、online critic 和 target critic 分别以零 `h,c` 开始；
- trajectory 内连续 carry，某个 vector slot 结束时只清该 slot；
- `terminated` 不 bootstrap；
- `truncated` 先用旧 episode 的 `final_observation` 和旧 history context 构造 TD target，再清状态；
- reset observation 不得进入旧 episode target；
- 若 checkpoint 只保存在完整 trajectory 边界，临时 `h,c` 不入 checkpoint；未来若允许跨 batch
  延续 episode，才必须把逐 slot recurrent state 纳入保存/恢复语义。

训练数据必须保持时间顺序，不能继续随机抽独立 transition。首个 recurrent 版本采用完整
trajectory minibatch：每组 1 条、每条最多 1000 步完整 BPTT，`B=25` 时一个 epoch 共 25 组。
这让一次 recurrent Adam step 的 transition 数接近 MLP 的 1024，并匹配每 rollout 约 25 个 critic
optimizer steps。所有 valid transitions 使用一次，target critic 在 epoch 内冻结、epoch 末只做一次
Polyak update。首版不加入 TBPTT、burn-in 或 replay。只有在后续 sequence replay 中，才需要处理 stored hidden
state 的参数陈旧、burn-in prefix、sequence 不跨 episode，以及 behavior-policy correction。

实验时 recurrent critic 必须作为 MLP baseline 之后的单组件消融：保持 actor、数据源、TD horizon、
dual、beta、训练步数和 evaluation seeds 不变，并同时报告参数量、吞吐、atom crossing、coverage、
CDF calibration 与最终 empirical failure probability，避免把网络容量差异误判成纯 history 收益。

不能直接复制 QCPO_REF 的 quantile GAE 到 DQC risk branch：前者来自 implicit quantile/density-ratio
推导；DQC 的 CDF value 若要 GAE，需先在 augmented state `(observation, history, remaining_budget)`
上推导 TD residual。MLP baseline 固定为 `online_1step_v1`；recurrent critic 使用独立
`online_1step_lstm_v1` 与 `critic_arch=lstm_independent_v1`。任何 replay、fresh PPO 或 GAE 仍须
作为后续单组件消融，不能静默进入任一 checkpoint identity。

## DQC validation calibration（仅诊断）

每次 exact-N validation 会把每条轨迹的初始 `(s0, executed a0)` reward-critic atoms 与同一条
`G_1000` outcome 配对，记录 initial CDF、直接 empirical failure、signed/absolute gap、Brier、
10-bin ECE、CDF 0/1 saturation、action-CDF dispersion、atom crossing、quantile coverage、
predicted-return bias/RMSE。全部位于 `validation/diagnostic/reward_critic/*`，且 terminal 单独打印
一行关键量；这些字段不进入 dual、feasibility 或 checkpoint selection。

## 评估、checkpoint 与阶段门

`evaluate_policy_vec` 在所有统计之前裁到精确 N（正式默认 N=1000），报告 mean/std、直接
failure probability/residual、Wilson 95% CI、线性插值经验 quantile、worst 5%、ties 与
near-threshold mass。direct-chance checkpoint 在 validation feasible 集中最大化 mean；若无 feasible
则最小化 residual 再最大化 mean。QPO checkpoint 单独最大化线性插值经验 quantile。checkpoint 原子写入，
包含 actor/RMS、critic/target、optimizers、scheduler、dual、optimizer counters 与 RNG，不保存 replay。
active MLP checkpoint signature 固定记录 `dqc_update_mode=online_1step_v1`、target source、minibatch、
critic epoch 及 fitted-warmup mode/E；resume 在修改 live state 前严格核对算法、环境、q、alpha、
gamma、horizon、网络和关键配置。旧 replay-era
DQC 明确拒绝。历史 LSTM checkpoint 保留其独立 identity，但 active code 不提供加载入口；其他算法的
ordinary legacy/transfer
仍按原显式开关处理。validation 与 rolling-latest checkpoint 各自按独立 interval 调度，
`validation_every=0` 即使 final checkpoint force 也不会暗中运行 evaluation。

`finite_nstep_v1` 另外把 horizon mode、target source、n、time feature、actor context、target-update
mode 和 beta 写入 W&B/config/checkpoint signature。任何跨 horizon/context/time-feature/n/
target-update-mode 的 resume 都在修改 actor、critic、optimizer 或 RNG 前失败；新架构必须 fresh 训练。

MLP correctness gate 与历史 critic-LSTM 单组件消融都已完成 SimpleButton seed 0（固定
`q=-0.03`）的 5M 与 exact-1000 final。两者都不满足 alpha=0.2 的约束；LSTM 没有改善 primary
feasibility 或 mean return，因此 active 实现继续使用 MLP。MLP baseline=`ea2194a`，历史 recurrent
实现/单测/smoke=`ca306c1`，MLP-only active 回退=`8dbcc48`。不会自动启动其他 seed 或环境；
更广的四环境主阶段仍要求用户根据 QPO anchor 明确选择其余 q，代码不会自动扫描或推导
threshold。

## 2026-07-30 至 2026-07-31 验证状态

active MLP 依赖无关测试 runner `tests/run_tests.py`：56/56 通过，覆盖 budget/QR 数学、
terminated/truncated/reset、raw/executed action、transition 全覆盖、target epoch 冻结、fixed-BT 单 actor
step、detach/dual 方向、legacy/fitted warmup、完整 policy-side 冻结、hard-copy 边界、crossing/clip、
cost 隔离、checkpoint identity/round-trip 和公共日志。MLP CUDA+sync
6-batch smoke 与 latest resume 第 7 批通过。历史 `ca306c1` 曾达到 52/52，并额外验证 recurrent history、
post-LSTM current-action head、ordinary/truncated target critic context、完整 BPTT 与 MLP checkpoint
互斥；历史 LSTM CUDA+sync smoke/resume counters 为 actor/critic/target/dual=1/12/6/1。这些代码已从
active runner 删除，但结果仍供追溯。

正式 MLP gate 完成 5M steps（train 2806.97 秒）。所有 validation 均不可行；选中 batch 180：
mean=0.00339、failure=0.36、critic CDF=0.22656、CDF gap=-0.13344、return RMSE=0.15891、
atom crossing row/adjacent=0.62/0.03452、lambda=0.33624。batch 200 latest lambda=0.3667，
actor/critic/target/dual counters=195/5000/200/195。

选中 policy 的 exact-1000 final：mean=0.00083、std=0.15373、linear q20=-0.09093、failure=0.400、
residual=+0.200、Wilson95=[0.37007,0.43069]、critic CDF=0.22591；约束未满足。W&B run
`kqbn3fre` 已同步，结果 JSON 为 `_runs/DQCACBeta_SimpleButton_dqc_online_1step_mlp_v1_s0.json`。

正式 LSTM gate 使用完全相同的 actor、rollout、one-step target、dual、beta、5M env-step、
exact-100 validation 和 exact-1000 final 协议。训练 3088.59 秒，选中 batch 100：
validation mean=-0.00415、failure=0.36、critic CDF=0.32031、CDF gap=-0.03969、
return RMSE=0.12788、crossing row/adjacent=1.00/0.30161、lambda=0.18081。batch 200 latest
lambda=0.37563，actor/critic/target/dual counters=195/5000/200/195。

选中 LSTM policy 的 exact-1000 final：mean=-0.01639、std=0.11905、linear q20=-0.09892、
failure=0.407、residual=+0.207、Wilson95=[0.37696,0.43775]、critic CDF=0.32134。
相对 MLP，failure 增加 0.007、mean 降低 0.01723；CDF absolute gap 从 0.17409 降至
0.08566、return RMSE 从 0.16209 降至 0.11946，但 crossing row 从 0.692 增至 1.000，
adjacent crossing 从 0.03790 增至 0.30648。LSTM critic 有 2,413,600 参数，MLP critic
有 311,328 参数（7.75×）；训练吞吐从 1781.28 降至 1618.86 env steps/s（-9.12%）。

因此本次 gate 支持“history 改善部分预测校准，但没有改善策略约束”的结论，不支持把 LSTM
升级为默认 critic。该判断只覆盖 SimpleButton seed 0，且 7.75× 参数量是 capacity confound；
若以后重启研究，需做容量匹配与多 seed，而不能把本轮结果外推到所有 POMDP。W&B run
`zm8ho1zc` 已同步，结果 JSON 为
`_runs/DQCACBetaLSTM_SimpleButton_dqc_online_1step_lstm_v1_s0.json`。

## 2026-07-31 MLP convergence diagnosis

用项目 exporter/W&B API 完整拉取 SimpleButton seed 0 的 DQC `kqbn3fre`、QCPO `t3wbzj8x`、
QPO `4u8fhkmp`，三者各 200 个 training rows。QCPO early/late 20-batch mean 为 0.001/1.004、
failure 为 0.382/0.016；QPO q20 为 -0.085/-0.017。DQC 对应 mean 为 -0.007/0.014，
failure 为 0.376/0.366，q20 为 -0.083/-0.100。环境、公共 actor 与 empirical dual 可学习，
停滞集中在 DQC 的 critic-to-actor 链路。

QR loss 从 early 0.056 降到 late 0.024，但独立 validation 同时显示：

- predicted return mean 始终约 +0.05，没有跟随 policy outcome；
- average atom spread 从 0.220 收缩到 0.109；
- critic CDF 从 0.319 降到 0.223，真实 failure 仍约 0.4，absolute gap 反而增大；
- return RMSE 从约 0.135 恶化到 0.175。

因此 loss 下降主要是 online critic 更贴合自己的 slow bootstrapped target 与分布收缩，不能证明
真实 return distribution 校准改善。每 rollout 约 25 个 online Adam step，却只在 epoch 末执行一次
`tau=.005` Polyak；5M 训练共 5000/200 次 online/target update，初始 target 的名义 EMA 保留系数
仍为 `(1-.005)^200≈.367`。latest checkpoint 的 online-target parameter L2 gap 仍约为 online norm 的 7%。

actor 侧 mean-advantage std 约 0.0113→0.0047，CDF-advantage std 0.0205→0.0112，实际 Adam
parameter step 也比 matched QCPO 小约 11 倍。lambda 对 empirical violation 正确上升到 0.367，但
它放大的是未校准、action discrimination 持续减弱的 CDF signal。

后续 debug 优先级：target/Bellman timescale → counterfactual action ranking/sensitivity → actor 实际
update/KL 与 Adam-eps 区域 → time-bucket finite/continuing calibration。collector 固定采 1000 步不是
TD 数据错误；严格 finite `G_1000` 若要逐 transition 一致，需 time-to-go-conditioned critic。

## 2026-08-01 finite-horizon DQC 分支定义

### 目标、推导边界与算法身份

当前 QPO/QCPO 的训练 outcome、constraint、dual、feasibility、checkpoint selection 和 evaluation
都直接使用完整 `G_1000`，因此可以严格解释为 finite-horizon 算法：

```text
maximize_theta E[G_1000]  subject to  P(G_1000 <= q) <= alpha.
```

此前的 finite-suffix MC 是这个有限时域下的真实 suffix sample supervisor，但不是 continuing
`Z(s,a)` critic oracle。若不把 time-to-go 与 actor history 纳入状态，论文式 Eq. (28)--(29) 的
time-homogeneous continuing Bellman/CDF 推导不能直接覆盖固定 1000 步边界；同一 observation/action
会混合不同剩余时长及 LSTM policy context。另一个推导差异是 `beta<1` 会把 constraint score term
额外按时间衰减，通常不等于 QCPO 的 finite-horizon population constraint gradient。因此 finite
分支固定 `beta=1`，risk term 不再乘 `beta^t`。

这一语义使用独立且非默认的 checkpoint identity `finite_nstep_v1`：

```text
horizon_mode=finite
critic_target_source=n_step
critic_n_step=100
critic_time_feature=remaining_fraction
critic_actor_context=true
target_update_mode=hard
beta=1.0
```

默认 `online_1step_v1` 仍是 continuing/one-step/no-time/no-context/Polyak，已有运行和 checkpoint
不受影响。finite 分支的 W&B group 固定为 `dqc_finite_horizon_v1`，seed-0 variant 为
`n100_time_context_beta1_s0`，使用独立输出目录并从 fresh 参数开始。

### Critic 输入与 100-step target

critic 仍为 feed-forward MLP；只扩展其输入，不改变 actor 或环境 observation 接口：

```text
critic_features_t = [s_t, (H-t)/H, detach(h_t), detach(c_t)]
critic_query_t    = [critic_features_t, a_exec_t]
```

这里 `H=1000`，所以 t=0 的 remaining fraction 为 1，t=999 为 .001。普通当前查询使用 `h0_t/c0_t`；
bootstrap endpoint 使用 rollout 中 `t+m` 对应的 `h0/c0` 和 `A_exec`，不会把 reset observation 或
清零后的 context 混入旧 episode target。相同 observation/action、不同 actor context 因而形成不同
critic 输入，但 critic loss 对 actor context 完全 detach。

对每个有效 transition 和 atom j，target 为：

```text
y_tj = sum_{k=0}^{m-1} gamma^k r_{t+k}
       + gamma^m I{endpoint remains in the same finite episode}
         target_psi_j(s_{t+m}, h_{t+m}, c_{t+m}, a_exec_{t+m}),
m = min(100, H-t, distance to the first true boundary).
```

`terminated` 或第 1000 步 `truncated` 的 boundary reward 会计入前缀，而尾项为零；只有 endpoint
仍在同一 episode 内才 bootstrap。target 在每个 critic epoch 开始前一次性 materialize/detach，
epoch 内冻结，全部 minibatch 完成后恰好执行一次 `target <- online` hard copy。

### Actor、CDF、warmup 与诊断

finite actor 权重为 `gamma^t A_mean - lambda A_CDF`。CDF indicator 不再在 late timestep 用
float32 transformed budget 做除法，而以 float64 计算全局等价式
`C_t + gamma^t z_j <= q`；原 `b_t` 只保留为兼容日志。dual、feasibility、checkpoint selection
和 evaluation 始终用真实完成轨迹的 inclusive `G_1000 <= q`。

前 10 batches 是 fixed-policy warmup：每批对同一代 labels 做 4 个完整 critic regression epochs，
之后只 hard copy 一次；actor 参数/log_std、observation RMS、actor Adam/param-group LR、scheduler、
dual 及对应 counters 均逐位冻结。从 batch 11 起，每批一个 critic epoch/一次 hard copy，再执行一次
真实 actor update 和一次 empirical dual update。

训练日志除既有 outcome 与 calibration 外，还记录 actor gradient norm/精确 pre/post-step Gaussian
KL、raw/executed action spread，以及 early/middle/late risk-gradient contribution。validation 仍报告
finite-suffix calibration、coverage、CDF gap 与 atom spread；这些诊断不替代真实 failure gate。

---

# 历史记录：2026-07-14 Cost-CMDP 版本（已停用）


> 对齐目标: NeurIPS'22 "Quantile Constrained RL: Constraining Outage Probability"
> (Jung et al., KAIST; 代码 wyjung0625/QCPO)。本文件记录环境对齐差异、三算法逐式
> 对照、约束口径决策与最小 pipeline 验证结果 (体例照 portfolio_env_inf/DESIGN.md)。
> 断点主锚点见 PROGRESS.md。

## 1. 环境对齐 (paper_envs.py)

论文用 OpenAI safety-gym (mujoco-py) Engine + config dict; 本目录用 safety-gymnasium
1.0.0 (新 mujoco) 任务子类逐项复刻【任务语义】。对齐口径: 机器人(point)/任务(goal/button)/
障碍数量与坐标/lidar 参数/cost 来源与约束结构逐项一致; obs 向量【编码】不可能与
mujoco-py 版逐字节一致 (传感器/lidar 实现不同) —— 已接受残差。

| 论文 config | 论文 env id | 本目录短名 | 注册 id | 任务 | 障碍 | obs 维度 |
|---|---|---|---|---|---|---|
| config0 | SimpleButtonEnv-v0 | SimpleButton | SafetyPointSimpleButton0-v0 | button | 2 buttons(固定@(-1,-1),(1,1)) + 3 hazards(固定@(0,0),(-1,1),(0.5,-0.5)) | 60 |
| config1 | DynamicEnv-v0 | Dynamic | SafetyPointDynamic0-v0 | goal | 3 hazards(随机) | 44 |
| config2 | GremlinEnv-v0 | Gremlin | SafetyPointGremlin0-v0 | goal | 5 hazards + 3 gremlins(移动), 场地±2, lidar_max_dist=5 | 60 |
| config3 | DynamicButtonEnv-v0 | DynamicButton | SafetyPointDynamicButton0-v0 | button | 6 buttons(随机), 无 hazard | 44 |

验证 (2026-07-14, `_probe_paper_envs.py` / `_probe_cost.py`):
- 4 env 组件挂载正确 (geoms/mocaps 数量与论文 config 一致); gremlin 实测移动 (max Δpos≈2.4)。
- 定向驾驶探针 (比例控制器开向最近障碍) cost 全部触发: SimpleButton cost_hazards=494/1000 步,
  Dynamic 470, Gremlin 466, DynamicButton cost_buttons=939 → cost 布线正确。
- 与论文一致: episode horizon=1000 (timeout 截断, 无终止态), 动作 [-1,1]^2, cost 每步∈{0,1,...}
  (多源求和 cost_sum)。

差异记录 (接受残差):
1. obs 编码/维度 ≠ 原 safety-gym (例: 原版含 prev_cost 拼接后维度不同) —— 任务语义对齐, 逐字节不可能。
2. safety-gymnasium 的 goal/button 达成后【重置目标位置继续】(与原版一致), 不终止 episode。
3. QCPO_ref 移植版按论文 wrapper 行为在 obs 末尾拼 prev_cost (维度+1); 用户两算法不拼 (其算法不需要)。

## 2. 约束口径 (用户决策 2026-07-14: 双口径, 默认未折扣对齐论文)

优化问题 (CMDP, 目标与约束两条回报流):  max_θ E[R]  s.t.  P_θ(C ≥ d) ≤ ω
- R = Σ γ^t r_t (γ=0.99, 论文一致)
- C = Σ γc^t c_t; **默认 γc=1.0 (未折扣) → C=Σc, d=15, ω=0.2** = 论文 launch_qcpo.py 口径
  (cost_limit=15, target_prob=0.2)。三算法同口径互比 + 与论文可比。
- 折扣口径开关: `--set cost_gamma=0.99 cost_limit=<CALIB 值>` (DQCAC 自动切回 continuing 配方)。

CALIB (随机策略, 64 episodes, 2026-07-14): d=15 处随机策略 outage —
SimpleButton≈0.21, Gremlin≈0.27 (初始违反, λ 应响应), Dynamic≈0.05, DynamicButton≈0.0
(初始可行, λ 应≈0; 训练后期靠近障碍才可能违反)。约束在 4 个 env 上均非平凡也非不可行。

## 3. 三算法逐式对照

### 3.1 用户 QCPO (agents/qcpo_gpu.py)
与 portfolio_env_inf 已验证版逐式一致, 仅约束换到 cost 流上尾:
- 权重 w(τ) = (R−μ_R)/σ_R − λ·𝟙{C≥d} (轨迹级, 广播全 timestep); EMA 归一化 (decay=0.01)。
- Dual: λ ← [λ + ε_k·(P̂(C≥d) − ω)]_+ (本批经验 outage)。
- 未折扣口径下 C=Σc 天然由 cost_gamma=1 实现 (代码无分支)。

### 3.2 用户 DQC-AC-β (agents/dqc_ac_beta_gpu.py)
portfolio 版扩展为【双分布式 critic】(reward/cost 两条流), 核心三机制不变:
- Actor 权重 w = γ^t·Â_m − λ·β^t·Â_c; Â_m = mean(ψ^r)−V̂_m; Â_c = Ψ̂(s,a,b)−V̂_c (上尾 CDF)。
- budget 递推 b_0=d, b_{t+1}=(b_t−c_t)/γc (γc=1 时退化为 b−c, 对应论文 remain_cost)。
- Dual on cost-critic: Ĝ = P(C≥d|s0,a0) (同源信号, 已验证配方)。
- **口径开关 episodic** (默认由 cost_gamma 推断):
  - γc=1 (论文口径) → episodic=True: episode 末【不 bootstrap】(未折扣无穷和发散) +
    critic_step_feature=True (剩余 cost 分布依赖剩余步数; risk_sensitive_env 已验证的有限期界配方)。
  - γc<1 → episodic=False: 截断恒 bootstrap (continuing; γc^1000≈4e-5 残差可忽略),
    step-blind critic (portfolio/risk_inf 配方)。

### 3.3 QCPO_ref 移植版 (agents/qcpo_ref.py + qcpo_ref_model.py + dist_rl_utils_ref.py)
用户决策: 核心算法文件【逐行移植】到本栈, 跑在同一套 safety-gymnasium 环境上
(若跑原 rlpyt + 旧 safety-gym, 仿真动力学不同 → 与用户算法不可比)。
- 数学逐式保留: PPO ratio-clip (epochs=8, clip=0.1) + PID-Lagrangian
  (pid_i ← [pid_i+Ki·(Q̂_{1−ω}(ep_cost)−d)]_+, sum_norm L=(J_r+λJ_c)/(1+λ));
  cost 分位 critic c_dist=exp(linear) (N=25) + quantile huber; Weibull 尾 (α=4σ(·), β=exp(·))
  + weibull_tail_loss + LDP 概率比 compute_prob_ratio 加权分位 GAE (取 target_outage 列);
  LSTM(512, skip) 策略, 输入含 prev_action/prev_reward, obs running-m/v 归一化 + clip(±10);
  obs 拼 prev_cost; cost_scale=10; 首迭代牺牲 (只刷 obs_rms)。
- 只换管线: rlpyt CpuSampler → SafetyVecEnv(MP) 时间主序 [T,B] rollout;
  transform_sample 的 new_T=100 切块 + 块首 LSTM 状态逐式复刻 (rollout 时逐步记录 h/c)。
- rlpyt 工具等价物 (dist_rl_utils_ref.py): valid_mean/valid_from_done/discount_return/GAE
  按 rlpyt.algos.utils 逐式重写; dist_rl_utils.py 数学函数原样搬。
- 约束口径 = 论文自身口径: 未折扣 ep_cost 对 d=15 (PID), critic 折扣 0.99 (混合方案, 原文一致)。

## 4. 采样架构 (CPU 多核并行仿真 + GPU update)

- `envs/safety_env_vec.py` SafetyVecEnv: 单进程串行 (调试/对拍基准)。
- `envs/safety_env_vec_mp.py` SafetyVecEnvMP: B 个 worker 进程各持 1 个 SafetyEnv
  (spawn 上下文, 干净子进程), 父进程广播动作/回收 (obs,r,c) → GPU 只跑网络。
  【每个 env 内部仍是串行 mujoco step】= 论文 rlpyt CpuSampler 并行语义, 不改动力学。
- `make_vec_env(..., backend='mp'|'sync')` 统一工厂; 训练/评估/CALIB 全走它 (args.vec_backend)。
- **等价性对拍通过** (`_probe_vec_equiv.py`, 2026-07-14): 同 seed + 同动作序列,
  sync vs mp 的 obs/reward/cost 最大偏差 = 0 (逐元素一致)。
- 吞吐 (Dynamic, 随机动作): sync≈40 steps/s (B=4); **mp B=16 ≈ 13.3k steps/s,
  B=32 ≈ 17.2k steps/s** (128 核, 还有余量; 全量实验可开 B=32-64)。

## 5. 统一对比标准 (硬约定)

1. 同一环境 (paper_envs) + 同 γ=0.99 / 同口径 (d=15, ω=0.2, 未折扣) / 同 horizon=1000。
2. 同评估协议: evaluate_policy_vec (QCPO_REF 用同口径自管 LSTM 版), 整段=一 episode,
   报 mean(R) / P̂(C≥d) / Q_{1−ω}(C) (+critic 校准诊断)。
3. 同 env-step 预算 (wandb x 轴 progress/env_steps), project=safety_gym_qcrl。

## 6. 最小 pipeline 验证 (2026-07-14, 60 iters × B16 × T1000 ≈ 0.96M env-steps/run)

冒烟链路: 4 env 自测 → 两算法 4 env 各 3 迭代 (无 NaN) → 未折扣口径 episodic 配方冒烟
→ QCPO_REF 4 迭代冒烟 (PID/critic/LSTM 路径通) → 3 算法 × 4 env 短训 (tag=minimal)。

结果 (12/12 run 完成, 无 NaN; eval=64 episodes, 未折扣口径 d=15/ω=0.2):

| algo | env | R(mean) | C_undisc | outage P̂(C≥15) | Q_0.8(C) | λ_final | critic Ĝ vs 真值 |
|---|---|---|---|---|---|---|---|
| QCPO_REF | SimpleButton | 0.964 | 7.7 | 0.234 | 18.0 | 4.33 (PID) | — |
| QCPO_REF | Dynamic | 1.159 | 9.9 | **0.188 ✓** | 11.8 | 5.04 | — |
| QCPO_REF | Gremlin | 0.376 | 11.5 | 0.266 | 24.4 | 13.26 | — |
| QCPO_REF | DynamicButton | 1.432 | 15.1 | 0.453 | 22.8 | 6.62 | — |
| DQCAC | SimpleButton | 0.029 | 89.6 | 0.547 | 192.0 | 4.09 | 0.809 vs 0.807 ✓ |
| DQCAC | Dynamic | -0.077 | 47.0 | 0.281 | 119.0 | 4.20 | 0.826 vs 0.856 ✓ |
| DQCAC | Gremlin | -0.071 | 80.6 | 0.703 | 152.2 | 4.16 | 0.891 vs 0.896 ✓ |
| DQCAC | DynamicButton | 0.033 | 8.9 | **0.172 ✓** | 14.0 | 0.00 ✓(可行λ=0) | 0.049 vs 0.172 |
| QCPO | SimpleButton | -0.003 | 97.6 | 0.641 | 189.8 | 23.86 | — |
| QCPO | Dynamic | 0.077 | 56.5 | 0.406 | 149.0 | 22.30 | — |
| QCPO | Gremlin | -0.051 | 81.3 | 0.609 | 170.4 | 26.21 | — |
| QCPO | DynamicButton | -0.045 | 5.9 | **0.125 ✓** | 9.8 | 0.00 ✓(可行λ=0) | — |

行为判读 (最小 pipeline 门 = 行为正确, 非收敛性能):
- **QCPO_REF**: 行为最接近论文预期 —— R 显著上升 (0.4~1.4), outage 已压到 ω 附近
  (Dynamic 0.19≤0.2 达标), PID λ 正常响应。0.96M steps ≈ 论文预算 (5M) 的 1/5, 未收敛属预期。
- **DQCAC**: cost-critic 校准优秀 (Ĝ vs 经验 outage 偏差 <0.03, n_step=100 修复生效),
  λ 在慢时间尺度上爬升中; 60 iters 含 30 warmup → 实际策略更新仅 30 iters, outage 未及压下。
  可行环境 (DynamicButton) λ 正确保持 0。
- **QCPO**: λ 对违反正确响应 (22~26), 可行环境 λ=0; 但 MC 轨迹级样本效率低 (960 条轨迹
  不足以驱动 60 维 obs 的 MLP), R/outage 改善需全量预算 —— 与 portfolio 环境结论一致
  (QCPO 讲样本效率短板)。
- 已修复两个 pipeline 问题: ① evaluation.py 对 step-feature critic 的 s0 增广;
  ② DQCAC 未折扣口径下 1-step TD 传播过慢 (cost-critic 恒 0, λ 不动) → 默认 n_step=100。
- QCPO_REF 的 eval cost_cdf_initial≈0 属预期: 其 critic 学【折扣】cost (论文混合方案),
  对照未折扣 d 天然低估, 仅诊断量, PID 用经验分位数不受影响。

## 6b. 新旧环境/算法交叉验证 (2026-07-14, 轻量 ~10min 级)

**目的**: 验证 zprl 复刻环境 + QCPO_REF 移植版与【原版栈】(py3.8 + torch1.5.1cpu +
mujoco-py2.0/mujoco200 + 旧 safety-gym + rlpyt + 原 qcpo 代码, conda env `qcpo_ref`,
驱动脚本 `_run_old_qcpo.py`) 行为一致。三层验证:

**(1) 算法数学逐元素对拍 (`_verify_ref_port.py`, 真 rlpyt 依赖) — 全部 Δ=0**:
dist_rl_utils 6 函数 (quantile huber / Weibull tail / 分位 TD / 分位 GAE / LDP 概率比 /
normalize) + rlpyt GAE/discount_return/valid_from_done + QcpoModel 前向全部 6 头 +
LSTM 状态 + obs_rms update + 高斯 PPO ratio (发现并修正: rlpyt logπ 的 z 分母带
EPS=1e-8, 已逐位对齐)。⇒ 移植版数学与原版逐元素一致。

**(2) 环境随机策略分布对照 (`_probe_old_env.py` 48ep vs `_probe_new_env_uniform.py` 64ep,
同均匀随机动作 U(-1,1), 整 1000 步 episode)**:

| env | obs (旧/新) | Σc mean (旧/新) | P(Σc≥15) (旧/新) | 备注 |
|---|---|---|---|---|
| SimpleButton | 60 / 60 ✓ | 38.5 / 29.0 | 0.208 / 0.188 | 同重尾形状 |
| Dynamic | 44 / 44 ✓ | 21.2 / 14.2 | 0.125 / 0.109 | |
| Gremlin | 60 / 60 ✓ | 45.2 / 40.2 | 0.292 / 0.344 | |
| DynamicButton | 44 / 44 ✓ | 0.25 / 0.44 | 0.000 / 0.000 | 几乎无 cost, 一致 |

obs 维度逐 env 完全一致; horizon=1000、单步 cost∈{0,1,..}、随机 R≈0 两边一致;
Σc 分布同量级同重尾, outage@d=15 差异在 MC 误差内 (48/64 ep, 二项 std≈0.05-0.06)。

**(3) 训练早期动力学对照 (SimpleButton, ~170k steps ≈ 论文预算 3.4%, 单 seed)**:
- 原版栈 (168k, batch 24k/iter): PID λ 单调升 1.8→17.1 (响应违反 ✓), 窗口 CostAvg
  117→79 (下压 ✓), ProbOutage 0.25→0.25 (窗口滞后, 未及压到 ω), Return 波动 ±0.5。
- 移植版 (176k, batch 16k/iter, GPU): 批 outage 0.06~0.38 波动, eval(32ep)
  outage=0.094≤ω ✓, C_mean=16.1, R(disc)=0.101; train_time 仅 78s。
- 判读: 【机制层一致】(PID 响应违反、cost 被下压、约束朝 ω 收敛); 定量轨迹在
  3% 预算 + 单 seed 下不可比 (两边批统计口径不同: 100-traj 滞后窗口 vs 16-traj 批;
  且旧栈早期 cost 水平更高)。残差来源 = obs 编码/mujoco 引擎版本 + seed 噪声,
  为【已接受残差】—— 这正是把基线移植到同一环境跑的原因 (公平对比不依赖新旧env等价)。

## 7. 已知残差 / 全量实验前待办

- 论文全量预算 5M steps/run (我们最小 pipeline 0.96M); 全量时 num_iterations≈300 × B=16
  或等价组合, 并按论文报 (0.1/0.2/0.3) 多组 ω。
- QCPO_ref 的 LSTM rollout 逐步过 GPU (T=1000 次小 forward), rollout 段占比高;
  全量时可考虑 B 加大摊薄。
- DQCAC warmup_iters=30 (先校准 cost-critic); 短训 60 iters 实际策略更新只有 30 iters。
- 探针脚本 `_probe_*.py` / `_smoke_*.py` / `_run_minimal.sh` 为临时件, 验证后可删。
