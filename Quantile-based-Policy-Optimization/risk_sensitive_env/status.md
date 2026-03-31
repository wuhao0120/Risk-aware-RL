# 项目状况 — QPO vs QCPO vs DQC-AC 对比实验

## 当前状态: 全面迁移到 WIDEGAP 环境

所有实验统一使用 `preset='widegap'` + MLP Actor `[8,8]`。旧 default 环境的配置已清除。

## 环境设计

### Q_0.25 递减条件

Q_0.25 随风险 r 递减的理论条件：`k_mean / k_std < 4.77 / G ≈ 0.055`（G ≈ 86.7）

| 体制 | default (旧) | 比值 | Q趋势 | widegap (新) | 比值 | Q趋势 |
|------|-------------|------|-------|-------------|------|-------|
| calm | k_mean=0.3, k_std=4 | 0.075 | **递增** ← QPO冒险 | k_mean=0.3, k_std=8 | 0.038 | **递减** ✓ |
| volatile | k_mean=0.5, k_std=8 | 0.063 | 微增 | k_mean=0.5, k_std=12 | 0.042 | **递减** ✓ |
| crisis | k_mean=-0.2, k_std=15 | <0 | 递减 ✓ | k_mean=-0.3, k_std=20 | <0 | 递减 ✓ |

### WIDEGAP 理论前沿 (MC 验证)

| r | Mean E[U] | Q_0.25 | 适合算法 |
|---|-----------|--------|---------|
| 0.0 | 86.7 | 83.4 | ← QPO 最优点 |
| 0.1 | 88.5 | 81.1 | |
| 0.2 | 90.5 | 78.5 | |
| 0.3 | 93.6 | 76.5 | ← QCPO 目标区间 |
| 0.5 | 98.2 | 70.1 | |
| 1.0 | 105.7 | 50.4 | |

平稳分布: calm ~57%, volatile ~25%, crisis ~19%（crisis 更频繁）

## 已实施的全部修改

### envs/risk_env.py
- 添加 `WIDEGAP_REGIME_PARAMS` + `WIDEGAP_TRANSITION_MATRIX`
- 添加 `preset` 参数 (`'default'` / `'widegap'`)

### utils/model.py — Actor MLP 化
- Actor 新增 `hidden_dims` 参数: `None`=线性(向后兼容), `[8,8]`=MLP
- 结构: `3→8(Tanh)→8(Tanh)→1`，105 个参数

### agents/qpo.py
- Actor 常驻 GPU（删除 CPU↔GPU 来回搬运）
- `choose_action` / `select_action` 加 `torch.no_grad()`
- 支持 `actor_hidden_dims` 参数
- 新增 `_compute_regime_stats()`: 每 episode 记录 per-regime 的体制占比和平均 action

### agents/qcpo.py
- 同 QPO: Actor 常驻 GPU、`torch.no_grad()`、MLP 支持、per-regime 日志
- `constraint_clamp_ratio` 参数（方案B，保留但默认不启用）
- `normalize_gradient='scale'` 参数（方案C，当前实验启用）
- `debug/lambda_over_density` 诊断日志

### agents/dqc_ac.py
- 支持 `actor_hidden_dims` 参数

### train_evaluate.ipynb — 全面重写
12 个 cell，全部使用 widegap 环境：
1. 标题
2. 导入
3. "理论前沿"
4. MC 前沿可视化
5. "QPO"
6. QPO 训练
7. "QCPO"
8. QCPO 训练
9. "DQC-AC"
10. DQC-AC 训练
11. "评估"
12. 三算法统一评估

## 超参数配置

| 参数 | QPO | QCPO | DQC-AC |
|------|-----|------|--------|
| actor_hidden_dims | [8, 8] | [8, 8] | [8, 8] |
| max_episode | 100000 | 100000 | 100000 |
| gamma | 0.99 | 0.99 | 0.99 |
| q_alpha | 0.25 | 0.25 | 0.25 |
| quantile_threshold | — | 80 | 80 |
| lambda_a | — | 0.01 | 0.01 |
| outer_interval | — | 20 | 20 |
| h_n / density_bandwidth | — | 0.05 | 0.05 |
| normalize_gradient | — | 'scale' | — |
| updates_per_episode | — | — | 10 |
| num_quantiles | — | — | 32 |
| batch_size | — | — | 64 |

## 已知问题与观察

### QPO 双时间尺度振荡

训练中观察到 action 先升后降的振荡现象（0-30k 步），原因：

1. **0-10k**: q_est 初始化偏低 (~77 vs 真实 ~83) → indicator 几乎不触发 → 无有效梯度 → 策略随噪声漫游
2. **10-25k**: q_est 追上来，indicator 开始触发，但触发的"坏轨迹"把策略推离 r=0 → action 升到 ~0.3
3. **25k+**: q_est 到 ~83，r>0 的真实 Q_0.25 远低于 q_est → 大量 indicator 触发 → action 被推回

**本质**: 策略 θ 和分位数估计 Q 形成追逐振荡。indicator 梯度天然稀疏（只有 25% 轨迹贡献），q_est 跟踪有延迟。

### 训练波动与 regime 关联

raw_reward 波动时大时小的原因：
- crisis 体制占比高的 episode → 方差大（k_std=20）
- action 偏高时进一步放大方差

已添加 per-regime 日志 (`regime/crisis_frac`, `action/risk_calm` 等) 可在 wandb 验证。

### MLP 初始化效应

MLP `[8,8]` + 正交初始化 + Tanh → 初始输出 ≈ 0（正负分量互相抵消），碰巧接近最优 r≈0。
线性策略初始输出 ≈ 0.5-0.7，需要更多步才能收敛。不是算法差异，是初始化运气。

## 三算法对比预期

| 维度 | QPO | QCPO | DQC-AC |
|------|-----|------|--------|
| 目标 | max Q_α | max E[U] s.t. Q_α≥C | 同 QCPO |
| 预期 action | r≈0 (所有regime) | r>0 (calm/volatile), r≈0 (crisis) | 同 QCPO |
| 预期 mean | ~87 | ~93-95 (高 7-10%) | ~93-95 |
| 预期 Q_0.25 | ~83 (最高) | ~80 (≥C) | ~80 (≥C) |
| 收敛速度 | 慢 (MC, 25% indicator) | 慢 (MC + λ耦合) | **快** (TD, replay buffer) |
| DQC-AC 优势 | — | — | 每步10次更新, crisis经验复用 |

## 文件结构

```
risk_sensitive_env/
├── agents/
│   ├── qpo.py          ← Actor常驻GPU + MLP支持 + per-regime日志
│   ├── qcpo.py         ← 同上 + normalize_gradient + density诊断
│   ├── dqc_ac.py       ← MLP支持
│   └── qcpo_raw.py     ← MLP支持 (nu版本, 备用)
├── envs/
│   └── risk_env.py     ← WIDEGAP preset 已添加
├── utils/
│   ├── model.py        ← Actor支持hidden_dims参数
│   └── evaluation.py
├── train_evaluate.ipynb ← 12 cells, 全 widegap
├── fix/                 ← 旧 Fix A/B/C 训练曲线 (历史参考)
└── status.md            ← 本文件
```
