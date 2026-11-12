# -*- coding: utf-8 -*-
"""
移植保真验证 —— 【原版 QCPO 代码原文件】(真 rlpyt 依赖) vs 本目录移植版, 相同输入逐元素对拍。

验证对象 (算法数学层, 与仿真无关):
  A. dist_rl_utils.py 6 个数学函数 (quantile huber / weibull tail / 分位 TD 目标 /
     分位 GAE / LDP 概率比 / normalize) + rlpyt.algos.utils 的 GAE/discount_return/valid_from_done。
  B. QcpoModel 网络前向 (obs 归一化 + MLP + LSTM(skip) + μ/V/c_dist/Weibull 头):
     参数互拷后同输入对拍 (mu/log_std/r_value/c_dist/c_w_alpha/c_w_beta/h/c)。
  C. 高斯 logπ / PPO likelihood-ratio: 我们的 _logp 公式 vs rlpyt Gaussian 分布。
  D. obs RunningMeanStd update 语义对拍。

原版代码路径: QCPO_nips_ref/qcpo (不修改任何原文件); rlpyt: dependencies/rlpyt (git 原库)。
判据: max|Δ| < 1e-5 (float32 数值噪声级) → 移植保真。
"""
import os, sys, importlib.util
sys.path.insert(0, '.')
sys.path.insert(0, '/vepfs-mlp2/c20250510/251204033/dependencies/rlpyt')   # 真 rlpyt

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
torch.set_default_dtype(torch.float32)

REF_DIR = '/vepfs-mlp2/c20250510/251204033/QCPO_nips_ref/qcpo'


def load_module(name, path):
    """按路径加载原版模块 (不改原文件)。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def report(name, *pairs):
    """逐输出对拍并打印 max|Δ|。pairs = (label, ours, theirs)。返回是否全部通过。"""
    ok = True
    for label, a, b in pairs:
        d = float((a - b).abs().max().item())
        flag = 'OK ' if d < 1e-5 else 'FAIL'
        if d >= 1e-5:
            ok = False
        print(f"  [{flag}] {name}.{label:24s} max|Δ|={d:.3e}")
    return ok


all_ok = True

# ============================================================ A. 数学函数对拍 ============================================================
print("\n===== A. dist_rl_utils (原版原文件 + 真 rlpyt) vs dist_rl_utils_ref (移植) =====")
# 原版 dist_rl_utils.py 里 import 了 rlpyt.projects.qcpo.* 吗? 检查: 它只 import rlpyt.utils.* → 直接加载
ref_utils = load_module('ref_dist_rl_utils', os.path.join(REF_DIR, 'dist_rl_utils.py'))
import agents.dist_rl_utils_ref as port_utils
from rlpyt.algos.utils import (discount_return as rl_discount_return,
                               generalized_advantage_estimation as rl_gae,
                               valid_from_done as rl_valid_from_done)

T, B, N = 50, 8, 25
tail_prob, omega = 0.3, 0.2
tail_ind = int(np.floor(N * (1 - tail_prob) - 0.5))          # =17 (原式)
tau = torch.arange(N).float() / N + 1 / 2 / N
c_tau = -torch.log(1. - tau)

reward = torch.randn(T, B)
cost = torch.rand(T, B) * 0.3
done = torch.zeros(T, B); done[-1] = 1.
valid = torch.ones(T, B)
value = torch.randn(T, B)
bv = torch.randn(B)
q_dist = torch.exp(torch.randn(T, B, N) * 0.5)               # 正分位数 (c_dist=exp 头同域)
q_bdist = torch.exp(torch.randn(B, N) * 0.5)
q_target = torch.exp(torch.randn(T, B, N) * 0.5)
w_alpha = 4 * torch.sigmoid(torch.randn(T, B))               # α∈(0,4) (原头同域)
w_beta = torch.exp(torch.randn(T, B) * 0.3)
bw_alpha = 4 * torch.sigmoid(torch.randn(1, B))
bw_beta = torch.exp(torch.randn(1, B) * 0.3)

all_ok &= report(
    'math',
    ('normalize', port_utils.normalize(value.clone(), valid), ref_utils.normalize(value.clone(), valid)),
    ('quantile_huber_loss',
     port_utils.quantile_huber_loss(q_dist, q_target, tau, valid),
     ref_utils.quantile_huber_loss(q_dist, q_target, tau, valid)),
    ('weibull_tail_loss',
     port_utils.weibull_tail_loss(q_dist, w_alpha, w_beta, c_tau, tail_ind, valid),
     ref_utils.weibull_tail_loss(q_dist, w_alpha, w_beta, c_tau, tail_ind, valid)),
    ('quantile_target_est',
     port_utils.quantile_target_estimation(cost, q_dist, done, q_bdist.unsqueeze(0), 0.99),
     ref_utils.quantile_target_estimation(cost, q_dist, done, q_bdist.unsqueeze(0), 0.99)),
)
a_ours = port_utils.gae_quantile_simple(cost, q_dist, done, q_bdist, 0.99, 0.97)
a_ref = ref_utils.gae_quantile_simple(cost, q_dist, done, q_bdist, 0.99, 0.97)
all_ok &= report('math',
                 ('gae_quantile.adv', a_ours[0], a_ref[0]),
                 ('gae_quantile.ret', a_ours[1], a_ref[1]))

c_tail = torch.sort(q_dist)[0][:, :, tail_ind:]
torch.manual_seed(1)
pr_ours = port_utils.compute_prob_ratio(cost, c_tail, w_alpha, w_beta, done,
                                        bw_alpha, bw_beta, 0.99, log_clip_range=0.5)
torch.manual_seed(1)
pr_ref = ref_utils.compute_prob_ratio(cost, c_tail, w_alpha, w_beta, done,
                                      bw_alpha, bw_beta, 0.99, log_clip_range=0.5)
all_ok &= report('math', ('compute_prob_ratio', pr_ours, pr_ref))

# rlpyt.algos.utils (原版 process_returns 用) vs 我们的等价物
adv_o, ret_o = port_utils.generalized_advantage_estimation(reward, value, done, bv, 0.99, 0.97)
adv_r, ret_r = rl_gae(reward, value, done, bv, 0.99, 0.97)
all_ok &= report(
    'rlpyt.algos.utils',
    ('discount_return',
     port_utils.discount_return(reward, done, bv, 0.99), rl_discount_return(reward, done, bv, 0.99)),
    ('gae.advantage', adv_o, adv_r),
    ('gae.return', ret_o, ret_r),
    ('valid_from_done', port_utils.valid_from_done(done), rl_valid_from_done(done)),
)

# ============================================================ B. QcpoModel 前向对拍 ============================================================
print("\n===== B. QcpoModel (原版原文件 + 真 rlpyt models) vs QcpoRefModel (移植) =====")
ref_model_mod = load_module('ref_qcpo_model', os.path.join(REF_DIR, 'qcpo_model.py'))
from agents.qcpo_ref_model import QcpoRefModel

# 注: lstm_skip 的残差连接要求 最后隐层尺寸 == lstm_size (原版同此约束; 论文配置 512/512)
kw = dict(observation_shape=(45,), action_size=2, n_quantile=25,
          hidden_sizes=[64, 64], lstm_size=64, lstm_skip=True, constraint=True,
          init_log_std=0., normalize_observation=True, var_clip=1e-6)
ours = QcpoRefModel(**kw)
theirs = ref_model_mod.QcpoModel(**kw)

# 参数互拷: 我们的 state_dict → 原版键名 (body.* → body.model.*; 丢弃 tau/c_tau buffer,
# 原版把它们存成普通属性不进 state_dict)
sd = {}
for k, v in ours.state_dict().items():
    if k in ('tau', 'c_tau'):
        continue
    sd[('body.model.' + k[len('body.'):]) if k.startswith('body.') else k] = v.clone()
missing, unexpected = theirs.load_state_dict(sd, strict=True), None
# 同输入前向 (含非零 obs_rms 统计: 先给两边同样刷一批)
obs_stats = torch.randn(500, 45) * 2 + 1
ours.update_obs_rms(obs_stats)
theirs.update_obs_rms(obs_stats)
d_mean = float((ours.obs_rms.mean - theirs.obs_rms.mean).abs().max())
d_var = float((ours.obs_rms.var - theirs.obs_rms.var).abs().max())
print(f"  [{'OK ' if max(d_mean, d_var) < 1e-5 else 'FAIL'}] model.obs_rms_update          "
      f"max|Δmean|={d_mean:.3e} max|Δvar|={d_var:.3e}")
all_ok &= (max(d_mean, d_var) < 1e-5)

Tm, Bm, H = 7, 3, 64
obs = torch.randn(Tm, Bm, 45)
pa = torch.randn(Tm, Bm, 2)
pr = torch.randn(Tm, Bm)
h0 = torch.randn(1, Bm, H); c0 = torch.randn(1, Bm, H)
with torch.no_grad():
    mu_o, ls_o, val_o, rnn_o = ours(obs, pa, pr, (h0, c0))
    mu_t, ls_t, val_t, rnn_t = theirs(obs, pa, pr, (h0, c0))
all_ok &= report(
    'model',
    ('mu', mu_o, mu_t), ('log_std', ls_o, ls_t),
    ('r_value', val_o.r_value, val_t.r_value),
    ('c_dist', val_o.c_dist, val_t.c_dist),
    ('c_w_alpha', val_o.c_w_alpha, val_t.c_w_alpha),
    ('c_w_beta', val_o.c_w_beta, val_t.c_w_beta),
    ('rnn.h', rnn_o.h, rnn_t.h), ('rnn.c', rnn_o.c, rnn_t.c),
)
# tau / c_tau 常量本身
all_ok &= report('model', ('tau', ours.tau, theirs.tau), ('c_tau', ours.c_tau, theirs.c_tau))

# ============================================================ C. 高斯 logπ / PPO ratio 对拍 ============================================================
print("\n===== C. 高斯 logπ / likelihood-ratio vs rlpyt.distributions.gaussian =====")
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from agents.qcpo_ref import QCPORefGPU

dist = Gaussian(dim=2)
mean_new = torch.randn(64, 2); ls_new = torch.randn(64, 2) * 0.2
mean_old = torch.randn(64, 2); ls_old = torch.randn(64, 2) * 0.2
act = torch.randn(64, 2)
ratio_rl = dist.likelihood_ratio(act, old_dist_info=DistInfoStd(mean=mean_old, log_std=ls_old),
                                 new_dist_info=DistInfoStd(mean=mean_new, log_std=ls_new))


class _Shim:                                                  # 只借 _logp (静态公式, 不建 env)
    _log_2pi = float(np.log(2 * np.pi))
    _logp = QCPORefGPU._logp


shim = _Shim()
ratio_ours = torch.exp(shim._logp(act, mean_new, ls_new) - shim._logp(act, mean_old, ls_old))
all_ok &= report('gaussian', ('ppo_ratio', ratio_ours, ratio_rl))

print(f"\n{'='*60}\nVERIFY {'PASS: 移植版与原版逐元素一致 (<1e-5)' if all_ok else 'FAIL: 存在偏差, 见上'}")
sys.exit(0 if all_ok else 1)
