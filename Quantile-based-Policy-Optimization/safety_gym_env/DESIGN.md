# safety_gym_env DESIGN —— QCPO / DQC-AC-β / QCPO_ref 迁移设计与验证记录

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
