# safety_gym_env —— Reward-Side 实施进展

> 当前断点：2026-08-05。分支 `exp/dynamic-seed0-v1`；冻结 executable SHA=
> `ddfe281df5ca1efa074332d836571464d589641c`。Dynamic seed 0 的 QPO anchor、`q0=.28`
> 初始约束和 `q1=1.85` strict 三算法 campaign 均已完成 400 个正式 batch、5M/10M
> checkpoint/hash/signature、W&B online sync 与 CRN exact-N=1000；strict verifier exit 0。
> 下一步只在用户决定后扩 seed，不按单 seed 自动声明普遍算法排名。

## 当前完成项

- [x] vector adapter 保留逐 slot `terminated`、`truncated`、`final_observation` 和 reset observation。
- [x] collector 不跨 episode，truncation critic bootstrap 与 raw `G_1000` 统计分离。
- [x] 公共 FC512×2+LSTM512 Gaussian actor、observation RMS、`A_raw` Gaussian log-prob。
- [x] exact-N reward-side evaluation、NumPy linear percentile、Wilson CI、ties/near-q。
- [x] format-versioned atomic checkpoint、resume/eval-only、validation selection 与 RNG state。
- [x] Safety QPO（纯 quantile identity）新增并接入 CLI/export。
- [x] QCPO 改为 direct reward chance MC，explicit probability dual，单次 fresh update。
- [x] QCPO_REF 改为 reward lower quantile、signed monotone atoms、location Weibull、PPO 加号。
- [x] DQCACBeta 默认固定为 `online_1step_v1`：单 reward MLP critic、one-step TD、一次 critic
  epoch、一次 score-function actor update、经验 probability dual；无 replay/n-step/PPO/GAE/normalization。
- [x] 新增显式非默认 `finite_nstep_v1`：严格 finite `G_1000`、n=100、remaining fraction、
  detached actor `h/c`、beta=1、float64 global CDF、10-batch fixed-policy E4 warmup 与 hard target。
- [x] 历史 LSTM 单组件消融已完成并留在 `ca306c1`；active CLI/export/test runner 已在
  `8dbcc48` 精确恢复为 MLP-only，不再暴露 `DQCACBetaLSTM`。
- [x] `A_raw/A_exec` 双字段贯通 collector、sync/MP adapter 与全部 critic query；truncation 保存
  `final_observation` 和清零前 actor context，普通 target action 直接复用下一步 `A_exec`。
- [x] DQC exact-N validation calibration：CDF gap/Brier/ECE/saturation、atom crossing/coverage、
  predicted-return bias/RMSE；只记 diagnostic，不参与模型选择。
- [x] `run_algo.sh`/`run_full.sh` 实施 QPO-first gate；main phase 缺任一用户 q 即退出。
- [x] checkpoint 保存并严格核对 experiment signature；legacy/transfer 需显式开关。validation 与
  rolling checkpoint interval 已解耦，`validation_every=0` 不会被 checkpoint force 重新启用。
- [x] active runner 当前 63/63 定向测试通过；除原 56 项外，新增 finite n-step 边界、
  remaining-time/context、beta=1 population gradient、float64 CDF、完整 warmup freeze 与跨 identity
  resume fail-fast。
- [x] finite SimpleButton CUDA+sync 12-batch smoke 通过；batch 11--12 已发生真实 actor/dual update，
  latest counters actor/critic/target/dual=`2/84/12/2`。
- [x] finite SimpleButton seed 0 decision gate：fresh 2.5M 已直接满足 failure/q20 扩展条件，随后从
  同一 latest checkpoint 连续恢复至 5M；两段 W&B history、checkpoint counters 与 signature 闭环。
- [x] finite SimpleButton seed 0 selected batch 200：exact-N=1000、历史相同 seeds 的 paired
  evaluation 与 20,000 次 bootstrap 完成；best/latest 均为 batch 200 且 actor state 逐位一致。
- [x] MLP SimpleButton CUDA+sync 6-batch smoke + latest resume 第 7 批通过。
- [x] 历史 LSTM SimpleButton CUDA+sync 6-batch smoke + latest resume 第 7 批通过；checkpoint counters
  actor/critic/target/dual=`1/12/6/1`，并完成 validation/checkpoint/final 小评估。
- [x] SimpleButton seed 0 MLP 正式 gate：5M steps、20 次 exact-100 validation、best/latest 原子
  checkpoint、W&B `kqbn3fre` 与 final exact-1000 evaluation 全部完成。
- [x] SimpleButton seed 0 LSTM 正式消融：相同 5M/validation/final 协议完整执行，W&B
  `zm8ho1zc` 已同步；未改善 primary feasibility，停止扩展。

## 2026-07-30 DQC MLP baseline 与 LSTM 消融停止点

- advantage 是相同 actor history 下的 critic action-marginal `Q-V`/CDF baseline，不是 trajectory
  LOO 或 GAE；mean/CDF baseline、两类 advantage 与最终权重均 detach，且不做 normalization。
- actor risk 梯度使用 critic CDF；dual、feasible、checkpoint selection 只使用完整 fresh
  trajectories 的 inclusive empirical failure probability。
- critic 仅训练当前 rollout：one-step detached target、无放回覆盖一次、epoch 内 target 冻结，
  全部 minibatch 后仅做一次 Polyak update。live state 不含 replay、n-step 或 replay checkpoint。
- 普通 target action 复用 rollout 下一步 `A_exec`；terminated 不 bootstrap；truncated 使用旧 episode
  `final_observation` 和清零前 actor recurrent context，禁止 reset observation/零 hidden state。
- actor 每 rollout 只有一次 fixed-`BT` score-function update，不含 ratio、PPO clip、多 actor epoch、
  GAE、entropy bonus或 advantage normalization。
- 前 5 批只更新 critic；actor RMS 和 batch-index LR schedule 继续推进，但 actor/dual optimizer
  state 不变。第 6 批起按 critic → actor → empirical dual 顺序更新并记录 post-update lambda。
- checkpoint identity 固定为 `online_1step_v1`，不保存 replay；旧 replay-era DQC state 在修改
  live state 前明确拒绝。
- LSTM 只替换 critic，并使用独立 `online_1step_lstm_v1`/`lstm_independent_v1` identity；每条
  trajectory 一次完整 BPTT Adam step，current action 不进入 recurrent history。

SimpleButton seed 0、`q=-0.03` 的正式 MLP gate 已完成 5M steps，训练耗时 2806.97 秒。所有
validation 均不可行；模型选择到 batch 180：mean=0.00339、failure=0.36、critic CDF=0.22656、
CDF gap=-0.13344、return RMSE=0.15891、atom crossing row/adjacent=0.62/0.03452，post-update
lambda=0.33624。batch 200 latest 的 lambda=0.3667，counters 为 actor/critic/target/dual=
195/5000/200/195，总环境步数 5,000,000。

LSTM 正式 gate 也完成 5M steps，耗时 3088.59 秒。模型选择到 batch 100：
mean=-0.00415、failure=0.36、critic CDF=0.32031、CDF gap=-0.03969、return RMSE=0.12788、
atom crossing row/adjacent=1.00/0.30161，post-update lambda=0.18081。batch 200 latest 的
lambda=0.37563，counters 同为 actor/critic/target/dual=195/5000/200/195。

| selected policy 的 exact-1000 final | MLP | critic LSTM | LSTM−MLP |
|---|---:|---:|---:|
| mean return | 0.00083 | -0.01639 | -0.01723 |
| std / linear q20 | 0.15373 / -0.09093 | 0.11905 / -0.09892 | -0.03468 / -0.00799 |
| inclusive failure `P(G<=q)` | 0.400 | 0.407 | +0.007 |
| Wilson 95% | [0.37007, 0.43069] | [0.37696, 0.43775] | overlap |
| critic CDF / absolute gap | 0.22591 / 0.17409 | 0.32134 / 0.08566 | +0.09544 / -0.08844 |
| return RMSE | 0.16209 | 0.11946 | -0.04263 |
| crossing row / adjacent | 0.69200 / 0.03790 | 1.00000 / 0.30648 | +0.30800 / +0.26858 |
| train seconds / env steps·s⁻¹ | 2806.97 / 1781.28 | 3088.59 / 1618.86 | +10.03% / -9.12% |
| critic parameters | 311,328 | 2,413,600 | 7.75× |

primary feasibility 没有改善：两者 failure 都约为 0.40，明显高于 alpha=0.2，LSTM 的 final mean
还更低。LSTM 的 CDF calibration gap 和 return RMSE 改善，说明 history 对预测有信息；但它没有转化成
更好的约束策略，并伴随严重 atom crossing、7.75× 参数量和较低吞吐。基于这个 seed 0 gate，默认实现
继续使用 MLP；LSTM 实现只保留在 Git 历史，不再是 active 研究入口，也不自动扩展三种子或其他环境。
单 seed 且参数量不匹配是历史消融的结论边界，不能据此声称 recurrent critic 在所有任务上都无用。

版本边界：MLP baseline=`ea2194a`；历史 LSTM 实现/测试=`ca306c1`；MLP-only active 回退=
`8dbcc48`。W&B 分别为 `kqbn3fre`/`zm8ho1zc`，结果 JSON 分别为
`_runs/DQCACBeta_SimpleButton_dqc_online_1step_mlp_v1_s0.json` 和
`_runs/DQCACBetaLSTM_SimpleButton_dqc_online_1step_lstm_v1_s0.json`。

## 2026-07-31 MLP W&B 收敛审计

完整拉取 SimpleButton seed 0 的 DQC `kqbn3fre`、QCPO `t3wbzj8x`、QPO `4u8fhkmp`，各 200 行。
QCPO 的 early/late 20-batch failure 从 0.382 降到 0.016、mean 从 0.001 升到 1.004；QPO q20
从 -0.085 升到 -0.017。DQC failure 仅 0.376→0.366，q20 反而 -0.083→-0.100，停滞是
DQC-specific，不是环境、公共 actor 或 empirical dual 全部失效。

- DQC QR loss 0.056→0.024，但 validation predicted mean 始终约 +0.05，atom spread
  0.220→0.109，CDF 0.319→0.223，而真实 failure 约 0.4；CDF gap 与 return RMSE 均恶化。
- 5000 个 online critic Adam step 只对应 200 次 `tau=.005` target update；初始 target 参数的
  名义 EMA 保留系数仍为 `.995^200≈0.367`。latest online/target 参数 L2 gap 约为 online norm 的 7%。
- mean-advantage std 约 0.0113→0.0047，CDF-advantage std 0.0205→0.0112，actor signal 与实际
  Adam parameter step 同时收缩；lambda 正确上升，但放大的是未校准且 action discrimination 变弱的 CDF。

当前优先定位为 target/Bellman 时间尺度与 critic action ranking，其次才是 finite/continuing 时域差和
MLP history 边缘化。QR loss 下降只表示更贴合 bootstrapped target，不能作为真实 return calibration 证据。

其余环境仍等待各自 QPO 结果与用户明确选择，不自动扫描或推导 threshold。旧 `_probe_*`、
`_smoke*`、`_run_minimal.sh` 等历史脚本和实验数据本轮不删除。

## 2026-07-31 direct target-tau 2.5M 提前停止

按用户要求回退 anchor/debug 框架后，Safety-Gym tracked tree 已恢复到 `0a0ecf1`，回退 commit
为 `31316c1`。不重复已有 5M MLP baseline；fresh SimpleButton seed 0 仅把 epoch 末
`target_tau=.005` 改为 `1-.995^25=.1177797571`，原计划最多 2.5M。W&B run=`37nx8qat`。

该分支在 batch 20 得到第二个外部 validation 后提前停止；中断前共上传 23 行、575k env steps。
batch 10/20 的 fixed-q failure 与原 baseline 基本相同，但 critic CDF gap 分别从
`-.101/-.084` 恶化到 `-.208/-.255`。batch 20 的 Brier `0.249→0.308`、coverage MAE
`0.137→0.211`、atom spread `0.206→0.082`，return RMSE `0.135→0.136` 没有改善。
同期 batch 11–20 QR loss `0.0530→0.0195`，但 mean-advantage std、CDF-advantage std 和 actor
gradient norm 分别缩小约 81%、60%、83%。

结论：简单加快 epoch target 追踪只让 online critic 更容易拟合快速移动的 bootstrap target，同时
加剧 atom/action signal 收缩和真实 tail-CDF 低估；“主要故障只是 target lag”被直接否证。该 tau
被拒绝，不扩展到 2.5M，也不用于正式默认。下一步应直接检验 one-step Bellman target 本身，例如
fresh finite-suffix-MC supervision，而不是继续增大 tau。

## 2026-07-31 finite-suffix MC 50-batch 诊断

实现 commit=`0a3e924`。新增显式 `critic_target_source={one_step,finite_suffix_mc}`，默认仍为
`one_step`，因此没有改变 active baseline。MC 分支对每条 fresh trajectory 反向计算
`G_t=R_t+gamma*G_{t+1}`，以一个 `[M,1]` scalar return sample 监督全部固定 QR heads；termination、
horizon truncation 和 invalid adapter rows 均不 bootstrap。该路径不查询、不更新 target critic，且
source 进入 W&B config、checkpoint signature/state/summary；即使允许普通 signature transfer，跨 source
resume 仍在任何参数 mutation 前失败。51/51 回归和 6-batch CUDA smoke 均通过；smoke 第 6 批
actor/dual 各更新一次，target update 仍为 0。

正式分支为 fresh SimpleButton seed 0、`B=25,T=1000,gamma=.99,q=-.03,alpha=.2`，前 5 批
critic-only，随后正常更新到 batch 50（1.25M env steps）。训练耗时 696.57 秒。W&B
run=`fmwgb9ab`，state=`finished`，完整 history 恰为 50 行；iteration 0..49、env steps
25k..1.25M 严格单调，validation 恰在 batch 10/20/30/40/50 且各 N=100，final N=1000。
除 operational show-progress/路径/命名外，与原 `kqbn3fre` 的唯一算法 config 差异为
`critic_target_source: one_step -> finite_suffix_mc`。latest checkpoint 的 actor/critic/target/dual
计数为 `45/1250/0/45`，全部 online/target 参数有限；best 为 batch 30，计数为
`25/750/0/25`。

| batch 30/40/50 validation 均值 | one-step baseline | finite-suffix MC | MC−baseline |
|---|---:|---:|---:|
| fixed-q failure | 0.4267 | 0.4167 | -0.0100 |
| linear q20 | -0.10549 | -0.10501 | +0.00048 |
| mean return | -0.01583 | -0.00985 | +0.00598 |
| absolute CDF gap | 0.13312 | 0.04302 | -0.09010 (-67.7%) |
| return RMSE | 0.13866 | 0.11692 | -0.02174 (-15.7%) |
| Brier | 0.26411 | 0.25086 | -0.01325 (-5.0%) |
| quantile coverage MAE | 0.14169 | 0.05363 | -0.08806 (-62.1%) |
| predicted atom std | 0.18356 | 0.09388 | -0.08968 |
| initial state-action CDF std | 0.05381 | 0.12087 | +0.06706 |
| adjacent-head crossing | 0.14441 | 0.29667 | +0.15226（恶化） |
| mean / CDF advantage std | 0.01035 / 0.02305 | 0.00672 / 0.01245 | -35.1% / -46.0% |
| actor grad norm | 0.001098 | 0.000449 | -59.1% |
| QR loss | 0.04313 | 0.02155 | -50.0% |

固定 batch 30/40/50 的 milestone-paired bootstrap 只有 3 个聚合点，不能替代逐 episode paired CI；
在这个明确限制下，MC−baseline 的 absolute CDF gap 95% 区间为 `[-.11663,-.06449]`，RMSE
为 `[-.02397,-.01986]`，三个点均改善。failure 差值区间 `[-.03567,+.01917]`、q20 差值
`[-.00108,+.00181]` 均包含 0。原 run 未保存逐 episode validation artifact，因此不伪造更强的
policy 显著性结论。

selected batch 30 的 exact-N=1000 final 为 mean `-0.00131`、q20 `-0.08813`、failure
`0.375`、Wilson95 `[.34553,.40543]`，明显未达到 failure<=.20 和 q20>=-.03。原 baseline 的
exact-N=1000 来自完整 200-batch run 所选 batch 180（failure=.400、q20=-.09093、
mean=.00083），不是 compute-aligned 的 50-batch 对照，且 Wilson 区间重叠，故不能把 MC 的
0.375 宣称为 policy 改善。

证据支持的结论分两层：

1. **支持**：“one-step Bellman/finite-horizon target 是 MLP critic 外部校准故障的重要来源”。
   MC 在全部后期 validation 上同时降低 CDF bias、coverage error 和 return RMSE；这不是只看
   self-TD/QR loss 得出的结论。
2. **拒绝**：“只把 TD target 换成 sampled true suffix return 就能修好 DQC 收敛”。failure/q20
   没有稳定改善，actor advantage 与 gradient 反而更小；MC 也把 adjacent crossing 提高约 2 倍。

本实验仍不能把 bootstrap propagation 与 finite/continuing horizon 语义完全分开：finite MC 同时
移除了 target bootstrap 和 horizon 末 bootstrap，而且 active MLP 没有 time-to-go 输入，会把不同
remaining horizon 的 suffix 分布边缘化。`action_cdf_std` 只是不同 initial state-action pair 的条件
CDF 变化，不能证明 counterfactual action 排序正确。finite-suffix MC 因此保留为显式非默认诊断，
不直接升级正式算法，也不扩成 5M。下一步应先对 batch-30 best critic 做 forced-action continuation
oracle；若 ranking 正确，再做 fixed-batch actor optimizer/variance 排查，若 ranking 错误则回到
critic action conditioning、time-to-go 或 distributional n-step。

证据落点：结果 JSON=`_runs/DQCACBeta_SimpleButton_finite_suffix_mc_50b_v1_s0.json`；完整 W&B
导出=`_runs/wandb_export/finite_suffix_mc_50b_v1/`；best/latest checkpoint=
`_runs/checkpoints/DQCACBeta_SimpleButton_finite_suffix_mc_50b_v1_s0/{best,latest}.pt`。

## 2026-07-31 finite-suffix MC batch-50→200 续训：确认迟发收敛

用户指出 QPO 在 batch 50 同样未达到 q20=-.03，因此不能用 1.25M steps 否证 finite-suffix MC。
本轮从原 run `fmwgb9ab` 的 **latest batch-50** checkpoint（不是 selected batch-30 best）恢复，
继续训练到总 batch 200/5M steps；新增 150 batches 的 W&B run=`o652xnzh`，训练耗时
2086.40 秒。两段合计训练耗时约 2782.97 秒。恢复了 actor/critic/optimizer/scheduler/dual/RMS、
计数器及主进程 Python/NumPy/Torch RNG；新建 vector workers 会从 seed 0–24 重新开始，故这是统计上
正确的续训而不是与不中断 5M 逐位相同的轨迹。

完整 history 闭环通过：`fmwgb9ab` 50 行 + `o652xnzh` 150 行恰为 200 行，iteration 0..199、
env steps 25k..5M 严格每行 +25k，无重复/缺口；20 次 validation 恰在 batch 10,20,...,200，
每次 exact N=100。seam counters 连续：batch 50→51 的 actor steps 45→46、critic steps
1250→1275；final actor/critic/target/dual=`195/5000/0/195`。config 的算法字段保持一致，唯一诊断
source 为 `critic_target_source=finite_suffix_mc`；default 仍是 `one_step`。

| fixed-N=100 validation | batch 50 | batch 90 | batch 100 | batch 150 | batch 200 |
|---|---:|---:|---:|---:|---:|
| mean return | -.00688 | .07276 | .09655 | .29101 | .59973 |
| linear q20 | -.10507 | -.03507 | -.00908 | .03370 | .17503 |
| fixed-q failure | .44 | .22 | .12 | .06 | .00 |
| critic CDF / empirical failure | .40375/.44 | .25594/.22 | .07094/.12 | .01750/.06 | .00094/.00 |
| CDF Brier | .26182 | .19466 | .10601 | .05647 | .00003 |

batch 90 仍未过线；batch 100 首次同时满足 failure<=.20 与 q20>=-.03（Wilson upper=.19812），
此后直到 batch 200 连续 11 个 validation 全部满足。best 和 latest 最终都为 batch 200：iteration=199、
env_steps=5M、signature/validation metrics 一致，14 个 actor/RMS state tensors 逐位相等；因此 final test
确实评估 batch-200 latest，而不是回退到历史 batch-30 best。

| seed 0 selected policy，exact-N=1000 | DQC one-step | DQC finite MC | QPO | QCPO |
|---|---:|---:|---:|---:|
| selected batch | 180 | 200 | 200 | 200 |
| mean return | .00083 | **.63348** | .09795 | 1.07164 |
| linear q20 | -.09093 | **.15489** | -.00950 | .54625 |
| fixed-q failure | .400 | **.007** | n/a | .025 |
| Wilson 95% | [.37007,.43069] | **[.00339,.01438]** | n/a | — |

finite MC 达到既定 success gate，并显著优于原 DQC MLP baseline；它也超过本 seed 的 QPO，但仍未追平
QCPO 的 mean/q20（计划本来就不要求追平 QCPO）。为避免只用两个独立 summary 做显著性声明，另以
明确的 reset seeds 777..1776 和每个 25-episode round 相同的 Gaussian action RNG，重新评估
one-step batch 180 与 finite-MC batch 200。paired exact-N=1000 得到：

- one-step：mean=-.001565、q20=-.097078、failure=.401；
- finite MC：mean=.616870、q20=.160752、failure=.006；
- paired mean delta=`+.618435`，20,000 次 episode-paired bootstrap 95% CI=
  `[+.583607,+.654083]`，完全不跨 0。

窗口分析解释了为何 batch 50 的判断过早。early validation（batch 10–50）MC/one-step 的
failure=`.418/.420`、q20=`-.10537/-.10578` 几乎相同，但 MC 已先把 |CDF gap|
从 `.11694` 降到 `.03294`、coverage MAE 从 `.14052` 降到 `.05243`、return RMSE 从
`.13715` 降到 `.11762`。mid（batch 60–120）MC 的 mean/failure 变为 `.07781/.21571`，
one-step 仍为 `-.01426/.38286`；late（batch 130–200）则为 `.39742/.04125` 对
`-.00188/.37500`。MC 的 mean-advantage std 在 mid/late 恢复到 `.01389/.01528`，而 one-step
降到 `.00807/.00519`；总 actor grad 也保持 `.00084/.00096`，说明 actor optimizer 并未永久失效。
late CDF advantage 降低是 failure、critic CDF 和 lambda 都趋近 0 后风险项自然消失，actor 继续优化 mean。

critic 仍不是全面无缺陷：MC late raw return RMSE=.28729，高于 one-step .16227，但同期 policy return
std 已扩大到 .40139；逐 milestone 的 RMSE/return-std 均值反而为 `.7193/1.0569`，且 absolute mean
bias 更低。MC 的 adjacent-head crossing 全程更高，但从 early .3066 降到 late .1650，batch 200 为
.08968；`row crossing=1` 只是 32 heads 中“至少一对倒序”的 OR 指标，不能等同于 CDF 失效。final
exact-1000 的 CDF=.003875 对 empirical=.007、Brier=.00707、coverage MAE=.03385，阈值概率校准正常。

**修订结论**：50-batch 台账中“单独替换 target 没有修复 policy 收敛”被本轮更长训练直接推翻。
faster target-tau 已失败、finite MC 成功，故主因不是简单 Polyak lag，而是 one-step bootstrap/传播目标。
不过 finite MC 同时移除了 bootstrap、改为严格 finite-horizon suffix、并引入单样本 MC target，当前证据
还不能区分这三者。它证明 MLP、actor optimizer 和公共训练骨架可以收敛，因此不再把 action-ranking/
Adam epsilon 作为首要故障假设。若升级正式默认，严格 G_1000 方向应评估 remaining-horizon 输入与多 seed
复现；若坚持 continuing 目标，则应比较同轨迹 distributional n-step。两者都改变算法语义，按协议在此
暂停而不静默覆盖 `online_1step_v1`。

证据落点：续训结果 JSON=
`_runs/DQCACBeta_SimpleButton_finite_suffix_mc_resume_200b_v1_s0.json`；完整导出=
`_runs/wandb_export/finite_suffix_mc_resume_200b_v1/`；batch-200 checkpoint=
`_runs/checkpoints/DQCACBeta_SimpleButton_finite_suffix_mc_resume_200b_v1_s0/{best,latest}.pt`；paired
逐 episode artifact/脚本/summary=
`_runs/paired_eval/finite_mc_b200_vs_one_step_b180_s0/`。实现 commit=`0a3e924`；未修改默认 target、
`DESIGN.md` 或 `AGENTS.md`，未启动其他 seed/环境。


## 2026-07-31 one-step warmup 与 fitted-hard 传播诊断

### 机械延长现有 warmup：拒绝

fresh SimpleButton seed 0 保持 `critic_target_source=one_step`、`target_tau=.005` 和全部网络/优化器
设置不变，只把 `warmup_iters=5` 改为 50；actor optimizer 与 dual 因而全程不更新。该 legacy
warmup 仍按历史定义更新 actor observation RMS 和绝对 batch-index scheduler，所以它检验的是
“直接把原 warmup 拉长”，不是严格 fixed policy。W&B run=`yepv76zn`，state=`finished`；完整导出
恰为 50 行、env steps 25k..1.25M 严格单调，validation 恰在 batch 10/20/30/40/50。

| validation batch | mean | q20 | fixed-q failure | critic CDF | signed CDF gap | Brier | return RMSE | row crossing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | -.01390 | -.10692 | .42 | .32281 | -.09719 | .25415 | .13479 | 1.00 |
| 20 | -.01535 | -.10691 | .43 | .31688 | -.11313 | .25785 | .13554 | 1.00 |
| 30 | -.01541 | -.10608 | .43 | .30281 | -.12719 | .26163 | .13551 | 1.00 |
| 40 | -.01520 | -.10668 | .43 | .29219 | -.13781 | .26573 | .13513 | 1.00 |
| 50 | -.01421 | -.10693 | .42 | .28687 | -.13313 | .26438 | .13432 | .99 |

真实 policy outcome 基本不动，外部 CDF gap/Brier 反而随更多 warmup 恶化；best selection 仍回到
batch 10。因而“原实现只是 5 批不够，机械改成 50 批即可”被否证。artifact：
`_runs/wandb_export/one_step_warmup50_v1/`；checkpoint：
`_runs/checkpoints/DQCACBeta_SimpleButton_one_step_warmup50_v1_s0/`。

### fixed-policy fitted-hard：极端 target 传播也被拒绝

实现 commit=`81381ac` 新增显式非默认 `critic_warmup_mode={legacy,fitted_hard}` 与
`warmup_critic_epochs`。默认仍为 `legacy/1`，所以 `online_1step_v1` 训练行为不变；
`fitted_hard` 只接受 `one_step`：每 rollout 先一次性 materialize Bellman labels，target 在 E 个
完整无放回 regression epochs 内逐位冻结，全部 Adam step 后只 hard-copy 一次。warmup 内 actor
参数/log_std、observation RMS、actor Adam/param-group LR、scheduler、dual 和对应 counters 全部
逐位不变。mode/E 进入 W&B config、checkpoint signature/state/summary，旧字段缺失只解释为
`legacy/1`，普通 transfer 不能跨 warmup identity。

56/56 CPU 回归通过。单 GPU 6-batch `fitted_hard/E4` smoke 通过；latest checkpoint 为
critic/target/actor/dual=`600 steps/6 copies/0/0`，actor Adam state 为空、scheduler last_epoch=0，
signature 明确为 `one_step,fitted_hard,E4,warmup=6`。

正式诊断分支为 fresh seed 0、`warmup_iters=50,E=4`，W&B run=`6iiem299`。按两次连续外部
validation 恶化规则在 batch 23 中断，state=`killed`，已完整导出 23 行；有效里程碑如下：

| validation batch | empirical failure | critic CDF | signed gap | Brier | conditional atom std | return mean bias | return RMSE | action-CDF std | row crossing |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | .45 | .06531 | -.38469 | .39448 | .05392 | +.06398 | .13609 | .01472 | 1.00 |
| 20 | .45 | .00000 | -.45000 | .45000 | .01072 | +.06397 | .13627 | .00000 | 1.00 |

同期训练 QR loss 从 batch 1 的 `.06363` 降到 batch 20 的 `.000309`（约 -99.5%），但 adjacent
crossing 从 `.21315` 升到 `.42325`，外部 conditional atom spread 与 action-CDF variation 几乎
归零。predicted return mean 仍约 `.0464`，没有跟随真实 mean `-.0176`。这不是数值爆炸；是
**自举分布收缩并保留正 location bias**。hard copy 只是把 slow-Polyak 分支已出现的错误加速到
极端：self-TD loss 越好，真实 tail-CDF 越坏。

因此当前证据边界为：

1. finite-suffix MC 的成功仍证明 actor/MLP/QR训练骨架并非完全不可用，但它只是一项 target 消融；
2. 失败不等于“critic 还没训完”。50 批 legacy 不改善，而 hard/E4 在 20 批内更快坍缩；
3. 已否证单纯 target parameter lag、更多同代 regression 和更快 target copy；尚未区分随机输出
   初始化残留、fresh continuous transitions 下 one-step distributional target 缺少足够随机性注入、
   以及 critic 未含 actor history/time-to-go 导致的条件状态别名；
4. 下一最小诊断应先在仍为 one-step TD 的前提下把 critic 最后一层零初始化，移除随机 location
   污染；若 mean bias 消失但 atom/CDF 仍塌缩，则证据将直接支持 distributional n-step target。
   n-step 仍是 TD，不是 finite-suffix MC；正式选择 n 或改 finite-horizon/time-to-go 前暂停确认。

失败分支保留为显式非默认开关，不升级 active default。完整导出：
`_runs/wandb_export/fitted_hard_E4_warmup50_v1_earlystop/`；rolling checkpoints：
`_runs/checkpoints/DQCACBeta_SimpleButton_fitted_hard_E4_warmup50_v1_s0/`。本轮未修改
`AGENTS.md`、未删除实验数据、未启动其他 seed/环境。

## 2026-08-01 finite_nstep_v1 实现与验证

本轮不覆盖 continuing baseline，而是新增独立算法身份 `finite_nstep_v1`。固定语义为
`horizon_mode=finite`、`critic_target_source=n_step`、`critic_n_step=100`、
`critic_time_feature=remaining_fraction`、`critic_actor_context=true`、
`target_update_mode=hard`、`beta=1.0`。CLI preset 将 W&B group/name 设为
`dqc_finite_horizon_v1/n100_time_context_beta1_s0`，同时使用独立 tag、输出与 checkpoint 路径；
用户显式给出冲突字段时不静默改写，由 agent 初始化 fail-fast。

critic 继续使用前馈 MLP，但 state-side 输入扩为
`[s_t,(1000-t)/1000,detach(h_t),detach(c_t)]`，action-side 仍为 `A_exec`。100-step target
会累积到 n、真实 terminal 或 horizon truncation 中最先到达者；boundary reward 被保留，tail
严格为零，且只有 endpoint 仍属于同一 episode 才用对应 `S/h0/c0/A_exec` bootstrap。reset
observation 和清零 context 不会泄漏进 target。finite CDF 用 float64 全局判断
`C_t+gamma^t z_j<=q`；兼容 budget 只保留日志。

前 10 batches 每批只生成一代 Bellman labels，冻结 target 后做 4 个完整 regression epochs，并在
批末 hard copy 一次。warmup 内 actor/log_std、RMS、actor Adam 与 LR、scheduler、dual 和 counters
逐位不变。batch 11 起恢复每批一个 critic epoch/一次 hard copy/一次 actor update/一次 empirical
dual update。finite actor 组合项为 `gamma^t A_mean-lambda A_CDF`，不再对 risk term 乘
`beta^t`；真实 `G_1000` 仍是 dual、feasibility、selection 和 evaluation 的唯一裁决量。

新增 7 项定向测试并注册到 dependency-free runner，总计 63/63 通过：

- 手算完整 100-step、末尾不足 100 步、真实 terminal 与 horizon truncation target；
- t=0/t=999 remaining fraction、bootstrap endpoint 的 time/context/action；
- 相同 observation/action 不同 detached context 的 critic 输入差异；
- float64 global CDF 与高精度 direct-budget 等价；
- beta=1 无风险时间折扣，并在小型 finite MDP 上等于 QCPO constraint population gradient；
- 前 10 批完整 policy-side freeze、第 11 批 actor/dual 真更新、每批一次 hard copy；
- W&B/checkpoint 新字段齐全且 continuing checkpoint 不能装入 finite 分支。

单 GPU sync 12-batch smoke 也通过。latest checkpoint 的 iteration=11、env_steps=24000，
actor/critic/target/dual counters=`2/84/12/2`，actor Adam 已有 11 个 parameter states，
scheduler `last_epoch=12`；由此覆盖了 warmup 结束后的真实 actor/dual 路径。smoke artifact 位于
`_runs/checkpoints/DQCACBeta_SimpleButton_finite_nstep_v1_smoke12_s0/` 和对应结果 JSON，
均属于运行产物，不纳入 Git。

训练 gate 保持既定协议：fresh seed 0 先跑 100 batches/2.5M steps，每 10 批 exact-N=100
validation；除 NaN、输出爆炸或连续严重退化外不中途停止。batch 100 后比较 one-step DQC、
finite-suffix MC、QPO、QCPO 的 outcome/critic/actor 指标。若已满足
`failure<=.20 && q20>=-.03`，或 batch 70--100 同时出现 mean/q20 正斜率、failure 负斜率、
CDF calibration 未恶化且 action spread 非零，则从同一 checkpoint 续到 200 batches/5M；
否则在 2.5M 停止且不追加超参分支。若续训，最终沿用历史 1000 evaluation seeds 做 paired
bootstrap mean comparison。

## 2026-08-01 finite_nstep_v1 2.5M decision gate

fresh SimpleButton seed 0 使用唯一预注册分支 `n100_time_context_beta1_s0`，W&B
group=`dqc_finite_horizon_v1`、run=`qarw01dv`。首段完整训练 100 batches/2.5M steps，导出恰为
100 行、iteration 0..99、env steps 25k..2.5M；10 次 validation 均为 exact N=100。batch 100
checkpoint 的 actor/critic/target/dual counters=`90/3250/100/90`，signature 明确包含 finite、n=100、
remaining fraction、actor context、hard target 与 beta=1，未加载 continuing/MC state。

| matching-batch validation | DQC one-step | finite-suffix MC | finite n-step | QPO | QCPO |
|---|---:|---:|---:|---:|---:|
| mean return | -.0112 | .09655 | **.14416** | .0480 | .3528 |
| linear q20 | -.1116 | -.00908 | **.02160** | -.0345 | .04376 |
| fixed-q failure | .39 | .12 | **.09** | n/a | .07 |

finite n-step 因此直接满足第一条扩展条件 `failure<=.20 && q20>=-.03`，不需要依赖趋势豁免。
作为一致性检查，batch 70--100 的 per-batch OLS slope 仍为 mean `+.001741`、q20 `+.001388`、
failure `-.003200`；Brier slope=`-.002394`，尽管 absolute CDF-gap slope=`+.001006` 有末段波动。
batch-100 critic 的 CDF/empirical=`.05094/.09`、Brier=.08565、coverage MAE=.20551、
return RMSE=.20315、conditional atom std=.05055。它比同期 finite MC 的 Brier=.10601 与
absolute CDF gap=.04906 更好，但 coverage、RMSE 与 atom spread 更差，不能称为全面校准胜出。

actor 路径没有退化：batch 100 pre-clip grad norm=.001482、post-step KL=6.38e-6，raw/executed
action std=.99582/.71763；对应 finite MC 与 one-step 的 grad norm 分别约 .000553/.001019。
分时 risk-term RMS 为 early/middle/late=.000945/.000185/0，符合 `gamma^t`、finite CDF 与
late threshold 已脱离 atoms 的共同作用。lambda=.05398，仍由真实 trajectory failure 更新。
selected batch 100 的独立 exact-N=1000 为 mean=.18046、q20=.00975、failure=.083，Wilson 95%
`[.06746,.10174]`，critic CDF=.05131。按协议从这个 run 的 **latest batch 100** 原状态续训，
没有新增超参分支。

## 2026-08-01 finite_nstep_v1 batch-100→200：5M 成功门槛

续训 W&B run=`3fmysa32` 新增 100 行；与首段拼接后恰为 200 行、iteration 0..199、env steps
25k..5M 严格每行 +25k，20 次 validation 恰在 batch 10,20,...,200。resume seam 的 actor/critic
counters 从 batch 100 的 `90/3250` 连续到 batch 101 的 `91/3275`；final
actor/critic/target/dual=`190/5750/200/190`。两段 W&B wall runtime 合计约 3634.75 秒（包含终测与
同步，不等同于纯训练时间）。final best/latest 都是 iteration=199、5M steps，metrics/signature
相同，14 个 actor state tensors 逐位相等；终测没有回退到早期 checkpoint。

| fixed-N=100 validation | batch 100 | batch 130 | batch 160 | batch 180 | batch 190 | batch 200 |
|---|---:|---:|---:|---:|---:|---:|
| mean return | .14416 | .35510 | .67968 | .86058 | 1.15563 | **1.45666** |
| linear q20 | .02160 | .06536 | .13921 | .25308 | .54312 | **.79453** |
| fixed-q failure | .09 | .01 | .03 | .00 | .00 | **.00** |
| critic CDF / empirical | .05094/.09 | .000/.01 | .000/.03 | .000/.00 | .000/.00 | .000/.00 |
| CDF Brier | .08565 | .01000 | .03000 | .00000 | .00000 | .00000 |
| return RMSE | .20315 | .37328 | .62821 | .61342 | .68457 | .73390 |

| seed 0 selected policy，独立 exact-N=1000 | DQC one-step | DQC finite MC | DQC finite n-step | QPO | QCPO |
|---|---:|---:|---:|---:|---:|
| selected batch | 180 | 200 | 200 | 200 | 200 |
| mean return | .00083 | .63348 | **1.35410** | .09795 | 1.07164 |
| linear q20 | -.09093 | .15489 | **.74609** | -.00950 | .54625 |
| fixed-q failure | .400 | .007 | **.001** | n/a | .025 |
| Wilson 95% | [.37007,.43069] | [.00339,.01438] | **[.00018,.00564]** | n/a | — |

finite n-step 达到既定 success gate，在本 seed/协议下同时超过 one-step、finite MC、QPO，并在
mean/q20 上超过 QCPO；这仍是单 seed 结果，不外推成跨 seed 排名。为避免独立 summary 的伪配对，
沿用历史 reset seeds 777..1776 和每个 25-episode round 相同的 Gaussian action RNG：

- one-step batch 180：mean=-.001565、q20=-.097078、failure=.401；
- finite n-step batch 200：mean=1.373868、q20=.783436、failure=.001；
- paired mean delta=`+1.375433`，20,000 次 episode-paired bootstrap 95% CI=
  `[+1.332788,+1.418140]`，完全不跨 0；
- 同一 common-random-number artifacts 下，finite n-step 相对 finite MC 的 paired mean
  delta=`+.756998`，95% CI=`[+.717474,+.796193]`。

窗口指标显示提升不是 action collapse：early/mid/late validation 的 mean=.00336/.14334/.76486、
failure=.382/.14143/.015；actor grad norm=.000567/.001169/.001609，KL 从 3.88e-7 增至
2.79e-5，raw/executed action std 从 .9987/.7177 到 1.0346/.7274。late lambda 均值降至 .00132，
final 为 0；risk contribution 同步归零，是实际 failure 已远低于 alpha 后的结果，mean actor signal
仍继续增长。

critic 结论需保持边界。final exact-1000 threshold CDF=.000 对 empirical=.001，gap/Brier=.001，
adjacent crossing=.00458，说明目标阈值附近的决策量可用；但 coverage MAE=.06802、mean bias=-.11323、
return RMSE=.63343，弱于 finite MC 的 .03385/-.06353/.37925。CDF 和 action-CDF std 饱和为 0
也意味着当前安全区内不再提供 counterfactual risk ranking 证据。因此成功结论是 finite n-step
训练出的 policy/outcome 与 threshold feasibility 成功，不是 critic 全分布误差已经解决。

证据落点：两段完整导出分别为 `_runs/wandb_export/finite_nstep_v1_100b_s0/` 与
`_runs/wandb_export/finite_nstep_v1_resume_200b_s0/`；selected checkpoint 与结果 JSON 位于
`_runs/checkpoints/DQCACBeta_SimpleButton_n100_time_context_beta1_s0_s0/{best,latest}.pt` 和
`_runs/DQCACBeta_SimpleButton_n100_time_context_beta1_s0_s0.json`；paired 逐 episode returns、脚本与
summary 位于 `_runs/paired_eval/finite_nstep_b200_vs_one_step_b180_s0/`。实现 commit=`81162a4`。
默认 `online_1step_v1`、历史 checkpoint、`AGENTS.md` 与 PDF 均未修改；未启动其他 seed、环境或
超参分支。

## 2026-08-04 QCPO_REF v2、统一主步数、DQC GAE-Mean 与 q=2.3 campaign

### 已落地的实现与版本边界

当前 active 分支为 `exp/qcpo-ref-v2-dqc-gae`，正式 q=2.3 campaign 固定使用
SHA=`2138e196bffd1ca35cf762818ede748b813b8f92`。核心提交依次为：

- `41d614b`：QCPO_REF actor 在完整 `A_reward-lambda*A_quantile` 上恢复 outer `gamma**t`，
  normalization 先于 occupancy weighting，PPO objective 使用固定 `B*H` 分母；新 identity 为
  `paper_negated_return_eq275_discounted_bh_v2`，旧 uniform-time checkpoint 在参数加载前拒绝。
- `bd207db`：拆分 `main_env_steps`、`initialization_env_steps` 与 `actual_env_steps`。QPO 的 2 批、
  DQC 的 10 批初始化不再占正式 400 batch；初始化不写正式 W&B history、不 validation/select，
  每批可原子 resume。QCPO/QCPO_REF 无独立初始化，最初 actor 更新仍计入正式训练。
- `4c25aa3`：DQC 新增 `state_value_gae_v1`。独立 scalar value critic 以
  `[observation,remaining-time,detach(actor h/c)]` 估计 mean GAE；原 action-conditioned 32-atom
  n=100 QR critic 只负责 CDF/outage。组合权重为
  `W_t=gamma**t*A_GAE_t-lambda*A_CDF_t`，完整 valid population 只除标准差、不减均值，actor 使用
  PPO clip=.1、最多 E8、target KL=.01。
- `2138e19`：新增 `q=2.3` 三算法 binding campaign、fresh-artifact guard、阶段门控、可读 W&B
  display name、5M freeze、10M endpoint 校验与 CRN exact-N=1000 post-hoc evaluation。

实现完成时 dependency-free/joint/checkpoint 回归共 112/112 通过，QCPO_REF v2 与 DQC
GAE-Mean CUDA smoke 通过。正式 campaign 启动后代码保持冻结。无关 `talk/`、PDF 删除和未跟踪
`Risk_aware_RE.pdf`、`prompt.md` 均为用户工作区内容，本轮未触碰。

### q=-0.03 正式对比已经闭环：确认 threshold 非 binding

上一 campaign `qcpo_ref_discounted_dqc_gae_seed0_v1` 已按 qref→dqc 顺序完成。共同设置为
SimpleButton、seed 0、actor LR=2e-4、alpha=.2、gamma=.99、B=25、H=1000、400 个正式 batch；
validation exact-N=100/seed555，终测 exact-N=1000/seed777。10M 结果如下：

| exact-N=1000，q=-.03 | QCPO_REF endpoint | DQC endpoint | QCPO_REF selected | DQC selected |
|---|---:|---:|---:|---:|
| mean | 2.83827 | 2.83329 | 2.87333 | 2.81875 |
| q20 | 2.38658 | 2.38617 | 2.37034 | 2.34111 |
| failure `P(G<=q)` | .000 | .000 | .000 | .000 |
| return variance | .50453 | .63718 | .55049 | .73635 |

两者 endpoint 几乎相同，且整个目标区间 failure=0，说明 `q=-.03` 无法检验风险梯度差异。
DQC 的 0--2M normalized validation mean AUC=`2.12716`，QCPO_REF=`1.92763`；由此只能说
DQC 前期增长更快，不能在非 binding threshold 下证明约束机制更好。完整结果已记录于
`campaigns/qcpo_ref_discounted_dqc_gae_seed0_v1/manifest.json`，W&B IDs 为
QCPO_REF=`qrefd2a`、DQC=`dqcg2m1a`。

### q=2.3 fresh seed-0 campaign：记录时快照

用户选定 `q=2.3`，要求 DQC 与 QCPO_REF 先跑，二者完整结束和 artifact 校验后再跑 QCPO；全部
使用 online W&B。共同设置保持 actor LR=2e-4、alpha=.2、gamma=.99、B=25、H=1000、
400 个正式 batch/10M main steps。DQC 另有 10 批 critic-only initialization，因此物理交互为
10.25M；QCPO_REF/QCPO 均为 10M。三个 run 使用同一 group
`binding_q2p3_simplebutton_seed0_v1`：

| 算法 | W&B ID / display name | 当前状态 |
|---|---|---|
| DQC GAE-Mean | `dqcg23s0` / `DQC-GAE-Mean-q2.30-lr2e4-s0` | `finished`，400/400，exact-N=1000 已完成 |
| QCPO_REF discounted v2 | `qref23s0` / `QCPO_REF-DiscOcc-v2-q2.30-lr2e4-s0` | `finished`，400/400，exact-N=1000 已完成 |
| QCPO trajectory LOO | `qcpo23s0` / `QCPO-q2.30-lr2e4-s0` | `finished`，400/400，exact-N=1000 已完成 |

已完成两算法的共同 CRN seed777 exact-N=1000 primary 结果：

| checkpoint | DQC mean / q20 / failure / variance | QCPO_REF mean / q20 / failure / variance | QCPO mean / q20 / failure / variance |
|---|---:|---:|---:|
| 5M endpoint | 2.63564 / 2.16989 / .312 / .48006 | .83179 / .40993 / .978 / .33516 | .70411 / .17005 / .970 / .42403 |
| 10M endpoint | 2.76875 / 2.29047 / .206 / .82434 | 1.75805 / 1.11944 / .808 / .58787 | 1.35746 / .80828 / .913 / .47724 |
| 10M selected | 2.78966 / 2.33471 / .172 / .72169 | 1.68021 / 1.09600 / .839 / .50719 | 1.52226 / 1.01687 / .903 / .42990 |

DQC 10M endpoint failure=.206，距离 alpha=.2 仅 .006，但按点估计仍不可行；selected batch 370
为可行 `.172`。QCPO_REF 和 QCPO 的 endpoint/selected 均明显不可行。0--2M normalized validation
mean AUC 分别为 DQC=`2.17414`、QCPO_REF=`1.30748`、QCPO=`.04943`。所以 q=2.3 在本 seed
确实把 DQC 与两种 trajectory 方法分开，但这仍只是 seed-0 工程证据，不能外推为跨 seed 排名。

原子进度、checkpoint 与 raw evaluation 分别位于
`_runs/binding_q2p3_simplebutton_seed0_v1/{progress,checkpoints,evaluations}/`。tmux 为
`binding_q23_dqc_qref_then_qcpo_s0`；协调器在 DQC/QCPO_REF 两个 evaluation JSON 就绪后才启动
QCPO，阶段顺序正确执行，最终 tmux 与训练进程均已退出。tracked manifest 为
`campaigns/binding_q2p3_simplebutton_seed0_v1/manifest.json`；三个 W&B run 均为
`finished/step399`，并已补执行显式 `wandb sync`。

### 当前结构澄清与 recurrent QCPO_REF 待办

- 所有 active 算法的 actor 均为 `obs RMS -> Tanh FC512x2 -> LSTM512 -> Gaussian mean`；LSTM 的
  `h/c` 是历史压缩状态，不是手工拼入 actor observation，actor 也未显式输入 previous
  action/reward/cost。
- 当前 DQC distributional critic 和 scalar value critic 都是独立 MLP；finite state 输入显式拼接
  remaining fraction 与 detached actor `h/c`。critic backward 不进入 actor。
- **本次已经完成的 q=2.3 QCPO_REF 不是 recurrent critic 版本**：actor 为上述 LSTM，但
  `RewardQuantileModel` 是仅看当前 observation 的独立 Tanh MLP512x2，没有 time feature 或 actor
  `h/c`。因此该结果必须标注为 `recurrent actor + observation-only MLP critic`。
- 原论文理论目标写作 infinite discounted horizon；原作者 Safety-Gym timeout 会置 `done=1`，代码在
  1000-step timeout 不 bootstrap。原作者网络则是 observation MLP + previous action/reward/cost
  进入共享 LSTM，再由共同 recurrent feature 输出 actor、reward value、cost quantiles 与 Weibull。
- 用户已提出把本地 QCPO_REF critic 改为 LSTM。为保持三算法 actor 容量可比，当前建议是
  **独立 recurrent critic**，由 observation feature + previous raw action/reward 驱动自己的
  `h_V/c_V`，而不是照搬会让 critic loss 更新 actor trunk 的共享网络。该改动必须新建 checkpoint/
  W&B identity，并在当前 q=2.3 coordinator 完全退出后实施；否则后启动的 QCPO 会混入不同代码 SHA。

### DQC 梯度口径与 seed 结论边界

DQC 每个正式 batch 先收集 B=25 条完整 fresh trajectory，再统一更新；trajectory 是独立 Monte
Carlo 样本，但 actor/critic 不是每条轨迹只有一个标量信号。actor 使用每个 valid timestep 的
`gamma**t*A_GAE-lambda*A_CDF`，一阶 on-policy 形式为
`sum_{b,t} W_bt * grad log pi(A_raw_bt|history_bt)/(B*H)`；PPO E8 对同一 detached weight 做
per-decision ratio/clip。distributional critic 使用每个 timestep 的 n=100 QR-TD target，scalar
critic 使用每个 timestep 的 fixed lambda-return。只有 dual 用每条完整轨迹一个
`G_b=sum_t gamma**t*r_bt`，每正式 batch 以 25 个 `1{G_b<=q}` 更新一次 lambda。

原 QCPO 论文所有实验使用 10 个不同 random seeds，并报告 mean ± standard deviation，但正文和当前
参考代码未列出具体 seed 数值。当前全部正式结果仍只有 seed 0，只能作为工程 gate；QCPO 完成后先做
同 seed 三算法比较，是否扩 seed 必须另行决定，不能按单 seed 声称普遍优于原方法。

### 下一断点

1. 先提交旧 q=2.3 tracked manifest 与本进展断点，保持其 code SHA=`2138e19` 可审计。
2. QCPO_REF 保留 state-value mean、19 个 monotone cost atoms、Weibull、finite one-step TD/GAE，
   只把 critic 输入改为 DQC-style remaining-time + detached actor `h0/c0`，并新增强制 checkpoint identity。
3. 补齐 current/next context、t=0/t=999、termination/truncation、旧 checkpoint fail-fast、save/load
   下一步一致性和 actor/critic 参数隔离测试，再运行全部回归与 GPU smoke。
4. 通过后以新 W&B ID、checkpoint root 和 manifest fresh 训练 q=2.3/seed0/400 main batches；
   旧 `qref23s0` 保留为 observation-only MLP critic 对照，不覆盖、不 resume。

## 2026-08-05 Dynamic seed 0 单种子两阶段正式实验

正式 harness 与算法配方冻结在 SHA
`ddfe281df5ca1efa074332d836571464d589641c`。本轮只扩展实验编排，不修改 agent、loss、网络、
环境或通用训练数学。DQC 固定为 `finite_nstep_v1 + state_value_gae_v1 + sampled-action M=4
CDF baseline`，不含 learned state-CDF 网络/optimizer；QCPO_REF 使用 finite remaining-time 与
detached actor-context critic；QCPO 使用 trajectory LOO。全部 run 为 Dynamic、seed 0、B=25、
H=1000、gamma=.99、alpha=.2、actor LR=2e-4、400 main batches=10M main steps。

QPO anchor 的 10M endpoint 为 mean=.68632、linear q20=.28568、std=.43990；selected batch 390
为 mean=.66633、q20=.27384、std=.42784。tracker 仅作诊断：endpoint/selected tracker 分别为
.27210/.26227，相对独立 empirical q20 偏差为 -.01358/-.01158。用户据此确认初始阈值
`q0=.28`。

`q0=.28` 初始阶段 exact-N=1000 如下。primary 是 compute-matched 10M endpoint；括号中为
validation-selected secondary：

| 算法 | mean endpoint (selected) | q20 endpoint (selected) | std endpoint (selected) | failure endpoint (selected) | early 0--2M normalized AUC |
|---|---:|---:|---:|---:|---:|
| DQCACBeta | 2.13694 (2.11924) | 1.95532 (1.94713) | .24120 (.24270) | .001 (.000) | 1.63149 |
| QCPO_REF | 2.07838 (2.02793) | 1.88994 (1.84999) | .28066 (.28524) | .000 (.000) | 1.52807 |
| QCPO | 1.35083 (1.44776) | 1.08848 (1.23273) | .29962 (.30045) | .001 (.003) | .06581 |

strict 阈值选择为 **`q1=1.85`**，而不是 2.0。理由是 1.85 几乎等于 QCPO_REF selected
q20=1.84999，同时距 DQC endpoint q20 只有约 .105、距 QCPO_REF endpoint 只有约 .040，属于
“可达到但有压力”的判别门槛；2.0 高于初始阶段三个算法的全部 endpoint q20（包括 DQC 的
1.95532），更可能让三者全部失败，信息量较低。改变阈值后三个算法均从 fresh 参数重新训练，
没有 resume `q0` checkpoint。

`q1=1.85` strict 阶段的 primary 10M endpoint exact-N=1000：

| 算法 | mean | linear q20 | std / variance | failure | Wilson 95% | early 0--2M normalized AUC |
|---|---:|---:|---:|---:|---:|---:|
| DQCACBeta | 2.14544 | 1.97645 | .23406 / .05478 | .088 | [.07198,.10717] | 1.66190 |
| QCPO_REF | 1.79895 | 1.58973 | .34422 / .11849 | .519 | [.48802,.54984] | 1.38792 |
| QCPO | 1.30104 | 1.00088 | .36629 / .13417 | .962 | [.94827,.97219] | .09998 |

secondary validation-selected exact-N=1000：DQC 仍选 iteration 399，指标与 endpoint 相同；
QCPO_REF 选 iteration 389，mean=1.81844、q20=1.59529、std=.32918、variance=.10836、
failure=.501、Wilson95=[.47007,.53193]；QCPO 选 iteration 349，mean=1.35336、q20=1.08174、
std=.35294、variance=.12457、failure=.955、Wilson95=[.94032,.96620]。

因此在这个明确限定的 Dynamic seed-0 工程证据里，只有 DQC 的 strict endpoint/selected 满足
`P(G_1000<=1.85)<=.2`；QCPO_REF 和 QCPO 均明显不可行。这不能外推成跨 seed 显著性或普遍
算法排名。也不能把旧 state-CDF 结果说成“完全没有提升”：它只有 seed 0 的 10M endpoint 正信号，
early/selected 不稳定，同时模型复杂度约 +20.6%、checkpoint 约 +25.7%、critic 更新量约 +50%，
证据不足以升级为默认；同样不声称实测 wall-clock 必然更慢。

strict W&B IDs 为 DQC=`dysdqqc3e9ef85_q81987245s0`、QCPO_REF=
`dysqrqc3e9ef85_q81987245s0`、QCPO=`dysqcqc3e9ef85_q81987245s0`；三者显式 sync 均正常
返回 `done`，服务端均为 finished、主历史恰 400 行、iteration 0..399。step clocks 为 DQC
10M main + 250k initialization = 10.25M actual，QCPO_REF/QCPO 均为 10M main/actual。
初始阶段 QCPO 正常 sync；DQC/QCPO_REF 的 CLI sync 遇到 W&B 服务 HTTP 500 后使用用户授权的
bypass，但 API 独立核验均为 finished/400/[0,399]，原始结果与校验状态未缺失。

tracked 小型总 manifest 为 `campaigns/dynamic_seed0_v1/manifest.json`；strict 原子 stage manifest
和 raw exact 结果位于 `_runs/dynamic_seed0_v1/strict_qc3e9ef85_q81987245/`。最终
`formal_dynamic_seed0.py verify --stage strict --return-threshold 1.85 --q0 .28` exit 0。

---



# 历史记录：2026-07-10 至 2026-07-14 Cost-CMDP 迁移（已停用）


> 目标：把用户的 **QCPO**(MC 轨迹) 与 **DQC-AC-β**(per-transition) 迁移到 NeurIPS 2022
> "Quantile Constrained RL: Constraining Outage Probability" (Jung, Cho, Park, Sung, KAIST)
> 的 Safety-Gym 点机器人环境 (SimpleButton/Dynamic/Gremlin/DynamicButton)，先跑通最小
> pipeline，验证行为正确、与论文口径对得上；随后(下一步)全量实验。
> 本文件是**断点恢复主锚点** —— 记录计划/决策/进展/发现，随时可从此续跑。

## 0. 关键路径与事实 (环境侦察结论)
- **真实工作 conda 环境 = `zprl`**：`/vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python`
  (py3.10, torch 2.3.1+cu121, mujoco 2.3.5, gym0.21+gymnasium1.2.1, wandb, SB3)。
  ⚠ 用户记忆里的 `distRL` 在本机是**失效软链接**(→ `.../Risk-aware-RL/.conda/distRL`, 已不存在)，勿用。
- **GPU** = 1× NVIDIA A100-SXM4-80GB。**联网**经代理 `http_proxy=http://127.0.0.1:7890` (git/pip/curl 均走它)。
- **NIPS 参考代码** 已 clone 到 `/vepfs-mlp2/c20250510/251204033/QCPO_nips_ref`
  (repo `wyjung0625/QCPO`, rlpyt-based; 只借用其**环境定义**，不跑其 rlpyt 训练栈)。
- **本工作目录** = `Quantile-based-Policy-Optimization/safety_gym_env/` (与 risk_sensitive_env_inf /
  portfolio_env_inf 同级同构)。**参考模板 = portfolio_env_inf/** (四算法 GPU 栈 + vec_base + DESIGN.md)。

## 1. 用户决策 (2026-07-10, AskUserQuestion)
- **环境实现 = safety-gymnasium 忠实复现**：用真实 mujoco 环境 + CPU VecEnv (对应用户早年"50-worker"
  计划)。忠实于论文、能对得上；throughput 受 CPU 限 (论文本身 batch_B=1 / 5M 步)。非 GPU 并行。
- **conda 环境 = 直接装进 zprl**：接受 safety-gymnasium 带来的降级 (gymnasium 1.2.1→0.28.1,
  mujoco 2.3.5→2.3.3, pygame 2.1.2→2.1.0)。**当前正在后台安装** (task bdmdni1ce)。

## 2. NIPS 环境是什么 (侦察结论)
- `qcpo/safety_gym_envs/config_safety_gym_env.py` 里 config0-3 = safety-gym `Engine` 配置 dict：
  - config0: task=button, 2 buttons, 3 hazards (点机器人) → 约 `SimpleButtonEnv-v0`
  - config1: task=goal, 3 hazards → 约 `DynamicEnv-v0`(或 SimpleEnv)
  - config2: task=goal, 5 hazards + 3 gremlins → 约 `GremlinEnv-v0`
  - config3: task=button, 6 buttons → 约 `DynamicButtonEnv-v0`
- 机器人 = **point** (`xmls/point.xml`)，**连续 2D 动作** → 用户现有高斯 Actor 直接适用，**无需 Categorical**。
- 观测 ≈ lidar(goal/hazards/buttons/gremlins) + 机器人传感器，~40-60 维连续向量 (+可选 prev_cost)。
- **CMDP 结构**：max E[reward 回报 R]  s.t.  **P(cost 回报 C ≥ d) ≤ ω** (outage probability)。
  `cost_limit d`=15/25，`target_prob ω`=0.2/0.3，γ=0.99。环境 wrapper 里
  `remain_discounted_cost=(remain-cost)/γ` == **用户 DQCAC 的 budget 递推**(天然契合)。
- ⚠ 与用户已验证环境的**结构差异**：那些环境 max E[Z] 与约束 P(Z≤q) 在**同一个 Z**；这里
  **目标(reward)与约束(cost)是两条不同的回报流** → 算法需扩展为"双回报流"。

## 3. NIPS 自家 QCPO 方法 (仅供对照，不是我们要跑的)
PID-Lagrangian PPO + **cost 分位数 critic**(n_quantile=25, `c_dist=exp(linear)`) + **Weibull 尾部模型**
(c_w_alpha/c_w_beta 外推极端 outage) + LSTM 策略。dual: `pid_i += (Q_{1-ω}(ep_cost) - cost_limit)·Ki`。
→ 与用户的 QCPO(经验分位数 dual) / DQCAC-β(budget+Abel-β+critic-CDF dual) 是**同问题不同方法**，正好可比。

## 4. 算法迁移设计 (双回报流扩展；数学与用户模板一致，仅把约束换到 cost 上、方向翻为上尾)
- **User-QCPO on safety-gym**：目标项用 reward 回报 R (EMA 归一化)，约束指示用 cost 回报
  `𝟙{C ≥ d}`，dual `λ←[λ+ε(P̂(C≥d)-ω)]_+`。每条轨迹算两个回报 R、C。**清爽扩展**。
- **User-DQCAC-β on safety-gym**：需要**两个 critic**：
  - reward critic (标量 V_r 或分布式取均值) → 均值优势 Â_m；
  - **cost 分布式 QR critic** ψ_c(s,a) → 约束 CDF `Ψ̂(s,a,b)=(1/N)Σ𝟙{ψ_c,i ≥ b}` (上尾)；
  - budget 在 **cost** 上：b_0=d, b_{t+1}=(b_t-c_t)/γ；Abel-β 只加在 cost 约束项；
  - dual on critic `P(C≥d|s0,a0)`。其余(两时间尺度、warmup、target 软更新、EMA 归一化)沿用已验证配方。
- 采样骨架需从 `*VecTorch`(GPU) 改为 **CPU VecEnv**(gymnasium 向量环境/多进程) → GPU 只跑网络与更新。
  vec_base 的 `_rollout_core` 要出一个 CPU-VecEnv 变体 (step 后搬 tensor 上 GPU)。

## 5. 分阶段计划 (Opus4.8 统筹；子agent/Workflow 执行 fan-out)
- [x] **P0 环境 (完成 2026-07-10)**：safety-gymnasium 1.0.0 装入 zprl (EXIT=0)。冒烟测试通过
  (headless `MUJOCO_GL=egl`，见 `_smoke.py`)：SafetyPointGoal1 obs(60)/act(2)∈[-1,1]/info
  {cost_hazards,cost_sum}；SafetyPointButton1 obs(76)/act(2)/info {cost_buttons,cost_hazards,
  **cost_gremlins**,cost_sum}；step 返回 **6 元组 (obs,reward,cost,term,trunc,info)**；
  episode horizon=1000；动作需 tanh 压缩。**可行性门通过**。
  候选最小 pipeline env：SafetyPointGoal1(简单,obs60) + SafetyPointButton1(含gremlins,obs76)。
- [ ] **P1 环境细读**：把论文 config0-3 映射/移植到 safety-gymnasium 任务；确认 reward/cost/终止/horizon；
  确定 (d, ω) 与评估协议，供"对得上论文"。
- [ ] **P2 脚手架**：新建 envs/ + agents/(vec_base+qcpo+dqcac+common) + utils/ + run_experiment.py + DESIGN.md
  (照搬 portfolio_env_inf 可复用件)。
- [ ] **P3 环境适配**：Safety-gym env → Vec 接口 (reset→[B,obs], step→(obs,r,cost,done))；CPU VecEnv；
  q/(d,ω) 用**带探索噪声**校准 (risk env q-gotcha 教训)。
- [ ] **P4 算法迁移**：按 §4 双回报流扩展 QCPO + DQCAC-β；保留全部已验证 knob；对照 Math fidelity。
- [ ] **P5 最小 pipeline**：每 env×算法小跑 (小B/少iter/wandb disabled)：无 NaN、λ 动、P(C≥d)→ω、
  DQCAC cost-critic 校准 (cdf_init≈empirical)；行为符合预期 + 与论文 ballpark 对齐 → 写 DESIGN.md 报告。
- [ ] **P6 全量实验** (下一步，暂不做)。

## 6. 当前状态 / 下一步
- ✅ **P0 环境** (装 safety-gymnasium + 冒烟)。
- ✅ **P2 脚手架 + P3 环境适配(通用) + P4 算法迁移 + run_experiment.py (代码完成 2026-07-10)**:
  `envs/`(safety_env 薄封装 + safety_env_vec CPU VecEnv 含 cost)、`utils/`(tanh-μ Actor + eval)、
  `agents/`(vec_base CMDP rollout + qcpo_gpu + dqc_ac_beta_gpu 双critic + common +indicator_ge)、
  `run_experiment.py`(QCPO/DQCAC/CALIB, 论文 env 表)。冒烟通过 (`_smoke_train.py` 两算法各 3 迭代无 NaN)。
- ✅ **P3-align 环境对齐论文 (完成 2026-07-14)**: `envs/paper_envs.py` 已接线进 envs/__init__ 与
  SafetyEnv (短名解析); 4 env 自测通过 (组件/坐标/cost 触发/gremlin 移动均验证, 见 DESIGN.md §1;
  定向驾驶探针: hazards/buttons cost 全部触发)。obs 编码 ≠ 原 safety-gym = 已接受残差。
- ✅ **P3b CALIB (2026-07-14)**: 4 env 随机策略 cost 分布 + d=15 处 outage (SimpleButton 0.21 /
  Gremlin 0.27 / Dynamic 0.05 / DynamicButton 0.0) → 约束非平凡且可行, d=15 直接可用。
- ✅ **约束口径 (用户决策 2026-07-14, 推翻 07-10 折扣默认)**: 双口径, **默认=未折扣论文口径
  (cost_gamma=1, d=15, ω=0.2)**; QCPO 由 cost_gamma=1 天然覆盖; DQCAC 新增 episodic 开关
  (γc=1 → episode 末不 bootstrap + critic_step_feature=True 有限期界配方; γc<1 → 原 continuing
  配方)。折扣口径留作 `--set cost_gamma=0.99 cost_limit=<CALIB 值>`。
- ✅ **CPU 多进程并行仿真 (2026-07-14)**: `envs/safety_env_vec_mp.py` SafetyVecEnvMP (spawn,
  1 env/worker) + `make_vec_env` 工厂 (backend='mp' 默认/'sync' 对拍)。**sync↔mp 等价性对拍
  Δ=0**; 吞吐 sync 40 → mp B=16 13.3k / B=32 17.2k env-steps/s (`_probe_vec_equiv.py`)。
- ✅ **QCPO_ref 核心移植 (用户决策: 移植到本栈同环境跑, 保证动力学可比; 2026-07-14)**:
  `agents/qcpo_ref.py` + `qcpo_ref_model.py` + `dist_rl_utils_ref.py` — PPO+PID-Lagrangian +
  分位 cost critic + Weibull 尾 + LSTM(512) + obs 归一化 + prev_cost 拼接, 数学逐行保留,
  只换 rlpyt 管线为本目录 [T,B] rollout (new_T=100 切块 + 块首 LSTM 状态)。冒烟通过。
  `run_experiment.py --algo QCPO_REF` 可用 (评估走 agent.evaluate_vec 同口径 LSTM 版)。
- ✅ **P5 最小 pipeline 完成 (2026-07-14)**: 3 算法 × 4 env × 60 iters × B16 × T1000
  (≈0.96M env-steps/run), 12/12 跑通无 NaN, 结果表 + 行为判读见 **DESIGN.md §6**。
  要点: QCPO_REF 行为最接近论文 (R↑ + outage→ω + PID 响应); DQCAC cost-critic 校准
  |Ĝ−真值|<0.03; 两可行环境 λ 正确=0; QCPO λ 响应正确但 MC 样本效率低 (符合旧结论)。
  途中修复: ① evaluation.py 对 step-feature critic 的 s0 增广 (旧版 eval 崩);
  ② DQCAC 未折扣口径 1-step TD 传播过慢 (critic 恒 0) → run_experiment 默认 n_step=100。
  ⚠ wandb online 需 mihomo 代理在跑 (`/vepfs-mlp2/c20250510/251204033/start_mihomo.sh`,
  曾因代理挂掉导致 wandb.init 超时批量失败, 已补跑; 台账 tag=minimal 完整)。
- 🔜 **P6 全量实验** (等用户确认): 论文预算 5M steps (≈300 iters×B16), 多 ω (0.1/0.2/0.3),
  多 seed; DQCAC/QCPO 超参 (lr/λ-lr/warmup/n_step) 需按 safety-gym 尺度全量调参;
  mp 后端可开 B=32-64 (吞吐 17k+ steps/s)。

**续跑第一步**: 用户审阅 DESIGN.md §6 + wandb(tag=minimal) → 拍板 P6 全量实验矩阵
(env × ω × seed × 预算) 与调参优先级。

## 7. 记录规范
- 本 PROGRESS.md 每阶段更新 (计划/进展/发现)。设计细节 (环境差异表、算法逐式对照、稳定性修复、
  验证结果) 写入同目录 **DESIGN.md** (照 portfolio_env_inf/DESIGN.md 体例)。遵循用户注释规范
  (中文、50%+ 密度、五层注释) 与"最小修改/适配器优先"。
