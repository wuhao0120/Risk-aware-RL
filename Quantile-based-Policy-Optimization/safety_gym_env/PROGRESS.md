# safety_gym_env —— QCPO / DQC-AC-β 迁移到 NIPS'22 安全约束环境 · 进展与断点记录

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
