# -*- coding: utf-8 -*-
"""
PortfolioVecTorch —— 全 GPU、B 路并行向量化的平稳 continuing 投资组合环境。

与同目录 portfolio_env.py (numpy 单 env) 的关系:
    公式逐式一致 (GBM 采样 / softmax 权重 / 买入交易成本 / 百分比奖励 / 权重漂移 /
    滚动窗口 est_mu/est_cov / 观测拼装), 仅把标量循环换成 [B, ...] 批量张量运算,
    参数默认从一个 PortfolioEnv 实例 (ref_env) 读取 → 保证与 numpy 版完全同分布。

接口对齐 risk_sensitive_env_inf/envs/risk_env_torch.py 的 RiskSensitiveVecTorch:
    reset() → [B, obs_dim];  step(actions [B,K]) → (next_obs [B,obs_dim], reward [B], done bool);
    属性 n / B / device / observation_space / action_space;  render() 返回日志序列。
    → 四个 GPU 智能体 (QPO/QCPO/QPPO/DQCAC) 的 rollout 代码无需关心环境差异。
"""
import numpy as np
import torch


class _Space:
    """轻量 shape 容器, 使智能体的 np.prod(env.observation_space.shape) 可用。"""

    def __init__(self, shape):
        self.shape = tuple(shape)


class PortfolioVecTorch:
    """B 路并行、全 GPU 的平稳 continuing 投资组合环境 (与 PortfolioEnv 同分布)。"""

    def __init__(self, num_envs=256, device=None, ref_env=None):
        """
        Args:
            num_envs: 并行 env 数 B
            device:   torch 设备 (默认 cpu)
            ref_env:  PortfolioEnv 实例 —— 市场/成本/窗口/奖励参数全部从它读取
                      (必须提供, 保证与 numpy 评估环境完全同分布)
        """
        assert ref_env is not None, "PortfolioVecTorch 需要 ref_env (PortfolioEnv) 提供参数"
        self.B = int(num_envs)                                 # 并行 env 数
        self.device = device if device is not None else torch.device('cpu')

        # -------------------- 从 ref_env 复制参数 (同分布保证) --------------------
        self.n = int(ref_env.n)                                # 截断长度 T
        self.K = int(ref_env.K)                                # 股票数
        self.M = int(ref_env.M)                                # 收益窗口样本数 (=24)
        self.tc = float(ref_env.tc)                            # 买入费率 c
        self.reward_mode = str(ref_env.reward_mode)            # 'pct' / 'log'
        self.cov_scale = float(ref_env.cov_scale)              # est_cov 观测缩放

        # GBM 预计算量 (与 numpy 版同一公式): log g = drift + z @ (√dt·L)^T
        self._drift = torch.as_tensor(ref_env._drift, dtype=torch.float32,
                                      device=self.device)      # [K] (μ-Σii/2)·dt
        self._cholT = torch.as_tensor(ref_env._chol.T, dtype=torch.float32,
                                      device=self.device)      # [K,K] (√dt·L)^T, 右乘用

        # est_cov 下三角索引 (含对角, 15 项, 与 numpy 版 np.tril_indices 一致)
        idx = torch.tril_indices(self.K, self.K)
        self._tril_r, self._tril_c = idx[0].to(self.device), idx[1].to(self.device)

        # 单 env space shape (智能体只用 .shape)
        obs_dim = self.K + self.K * (self.K + 1) // 2 + self.K # 25
        self.observation_space = _Space((obs_dim,))
        self.action_space = _Space((self.K,))

        # -------------------- 运行时状态 --------------------
        self.step_count = 0                                    # 当前步 (B 路锁步)
        self.w = None                                          # 持仓权重 [B, K]
        self.hist = None                                       # 收益窗口 [B, K, M] (%/步)
        self._maxw_buf = []                                    # 每步 batch 平均 max 权重 (render)
        self._turn_buf = []                                    # 每步 batch 平均买入换手 (stats)

    # ============================================================ 内部工具 ============================================================
    def _sample_gross_returns(self):
        """采一步 GBM 毛收益 [B, K]: g = exp(drift + z @ (√dt·L)^T), z~N(0,I), 全 GPU。"""
        z = torch.randn(self.B, self.K, device=self.device)    # [B,K] 标准正态
        return torch.exp(self._drift + z @ self._cholT)        # [B,K] 毛收益

    def _make_obs(self):
        """
        组装观测 [B, 25] (与 numpy 版 _make_obs 逐式一致):
            est_mu [B,K]; est_cov [B,K,K] (ddof=1); obs=[est_mu, tril·scale, w]。
        """
        est_mu = self.hist.mean(dim=2)                         # [B,K] 窗口均值
        centered = self.hist - est_mu.unsqueeze(2)             # [B,K,M] 去均值
        # 批量协方差: cov[b] = centered[b] @ centered[b]^T / (M-1)  (ddof=1, 同 np.cov)
        est_cov = torch.einsum('bkm,bjm->bkj', centered, centered) / (self.M - 1)
        tril = est_cov[:, self._tril_r, self._tril_c] * self.cov_scale  # [B,15] 下三角缩放
        return torch.cat([est_mu, tril, self.w], dim=1)        # [B, 25]

    # ============================================================ Vec API ============================================================
    def reset(self):
        """重置全部 B 个 env: 预热 M 步窗口 + 随机 softmax 初始权重。返回 [B, obs_dim] (GPU)。"""
        self.step_count = 0
        self._maxw_buf, self._turn_buf = [], []

        # 预热收益窗口: M 步纯模拟器 (与 numpy 版 reset 一致)
        self.hist = torch.stack(
            [100.0 * (self._sample_gross_returns() - 1.0) for _ in range(self.M)],
            dim=2)                                             # [B, K, M]

        # 随机初始权重: softmax(N(0,1) logits) (与 numpy 版一致)
        self.w = torch.softmax(torch.randn(self.B, self.K, device=self.device), dim=1)

        return self._make_obs()

    def step(self, actions):
        """
        actions: [B, K] torch (GPU)。返回 (next_obs [B,obs_dim], reward [B], done bool)。
        公式与 PortfolioEnv.step 逐式一致 (见该文件 step 的 7 步注释), 全程批量。
        """
        # ---- 1. softmax → 目标权重 u [B,K] ----
        u = torch.softmax(actions, dim=1)

        # ---- 2. 买入交易成本因子 κ [B] ----
        turnover_buy = torch.clamp(u - self.w, min=0.0).sum(dim=1)  # [B] 买入权重总量
        kappa = 1.0 - self.tc * turnover_buy                   # [B] 成本因子

        # ---- 3. GBM 一步 + 组合毛收益 ----
        g = self._sample_gross_returns()                       # [B,K]
        Gp = (u * g).sum(dim=1)                                # [B] 组合毛收益

        # ---- 4. 奖励 (百分比, 平稳) ----
        if self.reward_mode == 'log':
            reward = 100.0 * torch.log(kappa * Gp)             # [B] 对数收益
        else:
            reward = 100.0 * (kappa * Gp - 1.0)                # [B] 简单收益 (默认)

        # ---- 5. 权重漂移 ----
        self.w = (u * g) / Gp.unsqueeze(1)                     # [B,K] 新权重

        # ---- 6. 滚动窗口更新 (roll 左移 + 末位写入本步收益) ----
        x = 100.0 * (g - 1.0)                                  # [B,K] 本步各股收益
        self.hist = torch.roll(self.hist, shifts=-1, dims=2)
        self.hist[:, :, -1] = x

        # ---- 7. continuing 截断 + 日志缓存 ----
        self.step_count += 1
        self._maxw_buf.append(self.w.max(dim=1).values.mean().detach())   # batch 平均集中度
        self._turn_buf.append(turnover_buy.mean().detach())               # batch 平均买入换手
        done = (self.step_count == self.n)                     # 截断点 (truncated, 非终止)

        return self._make_obs(), reward, done

    # ============================================================ 日志辅助 ============================================================
    def render(self, mode=None):
        """返回本段 rollout 每步的 batch 平均 max 权重序列 [T] (numpy) —— 集中度作风险代理。"""
        if not self._maxw_buf:
            return None
        return torch.stack(self._maxw_buf).cpu().numpy()

    def stats(self):
        """返回本段 rollout 的附加统计 (per-stock 平均权重 [K], 平均买入换手率): 供智能体日志。"""
        out = {}
        if self.w is not None:
            out['mean_weights'] = self.w.mean(dim=0).detach().cpu().numpy()   # [K] batch 平均权重
        if self._turn_buf:
            out['avg_turnover'] = float(torch.stack(self._turn_buf).mean().item())
        return out

    def close(self):
        return None
