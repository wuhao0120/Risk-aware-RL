# -*- coding: utf-8 -*-
"""
dist_rl_utils_ref.py —— NIPS'22 QCPO 参考实现的分布式 RL 数学工具【逐行移植】。

来源: QCPO_nips_ref/qcpo/dist_rl_utils.py (wyjung0625/QCPO)。
移植原则: 数学逐式保留, 仅去掉 rlpyt 依赖 (valid_mean/zeros → torch 原生等价物)。
所有张量均为时间主序 [T, B, ...] (与 rlpyt 一致)。

额外移植 (原在 rlpyt.algos.utils):
    discount_return / generalized_advantage_estimation / valid_from_done
"""
import torch


# ============================================================ rlpyt 原生等价物 ============================================================
def valid_mean(tensor, valid=None):
    """valid 掩码均值 (rlpyt.utils.tensor.valid_mean 等价)。valid=None → 全量均值。"""
    if valid is None:
        return tensor.mean()
    valid = valid.type(tensor.dtype)
    return (tensor * valid).sum() / max(valid.sum().item(), 1e-8)


def valid_from_done(done):
    """
    rlpyt.algos.utils.valid_from_done 等价: done [T,B] → valid [T,B]。
    首个 done【之前及当步】有效, 之后无效 (recurrent 训练不跨 episode)。
    """
    done = done.type(torch.float)
    valid = torch.ones_like(done)
    valid[1:] = 1 - torch.clamp(torch.cumsum(done[:-1], dim=0), max=1)
    return valid


@torch.no_grad()
def discount_return(reward, done, bootstrap_value, discount):
    """rlpyt.algos.utils.discount_return 等价: 折扣回报 (done 截断, 末端 bootstrap)。"""
    ret = torch.zeros_like(reward)
    nd = (1 - done.type(reward.dtype))
    last = bootstrap_value
    for t in reversed(range(len(reward))):
        last = reward[t] + discount * last * nd[t]
        ret[t] = last
    return ret


@torch.no_grad()
def generalized_advantage_estimation(reward, value, done, bootstrap_value,
                                     discount, gae_lambda):
    """rlpyt.algos.utils.generalized_advantage_estimation 等价: 返回 (advantage, return)。"""
    advantage = torch.zeros_like(reward)
    nd = (1 - done.type(reward.dtype))
    last_adv = torch.zeros_like(bootstrap_value)
    last_value = bootstrap_value
    for t in reversed(range(len(reward))):
        delta = reward[t] + discount * last_value * nd[t] - value[t]
        last_adv = delta + discount * gae_lambda * nd[t] * last_adv
        advantage[t] = last_adv
        last_value = value[t]
    return_ = advantage + value
    return advantage, return_


# ============================================================ QCPO 专属 (dist_rl_utils.py 逐行) ============================================================
def normalize(value, valid=None):
    """(原样) valid 掩码标准化。"""
    if valid is not None:
        valid_mask = valid > 0
        mean = value[valid_mask].mean()
        std = value[valid_mask].std()
    else:
        mean = value.mean()
        std = value.std()
    return (value - mean) / max(std, 1e-6)


def quantile_huber_loss(quantiles, target_quantiles, tau, valid):
    """(原样) 分位数 Huber 损失 (κ=1); [T,B,N] × [T,B,N'] 全配对。"""
    pairwise_delta = target_quantiles[:, :, None, :] - quantiles[:, :, :, None]  # T x B x N x N'
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss


def weibull_tail_loss(quantiles, c_w_alpha, c_w_beta, c_tau, tail_ind, valid):
    """(原样) Weibull 尾部拟合损失: 对 sorted 分位数尾部做 log-Weibull 回归。"""
    quantiles_t = torch.sort(quantiles)[0].detach()
    loss = (0.5 * (torch.log(c_w_beta)[:, :, None]
                   + c_w_alpha.reciprocal()[:, :, None] * torch.log(c_tau[None, None, tail_ind:])
                   - torch.log(quantiles_t[:, :, tail_ind:])) ** 2).mean()
    return loss


@torch.no_grad()
def quantile_target_estimation(reward, r_dist, done, bootstrap_r_dist, discount):
    """(原样) 1-step 分布式 TD 目标: target[t] = r[t] + γ·dist[t+1]·nd[t]。"""
    quantile_target = torch.zeros_like(r_dist)
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd
    quantile_target[-1] = reward[-1].unsqueeze(1) + discount * bootstrap_r_dist.squeeze(0) * nd[-1].unsqueeze(1)
    for t in reversed(range(len(reward) - 1)):
        quantile_target[t] = reward[t].unsqueeze(1) + discount * r_dist[t + 1] * nd[t].unsqueeze(1)
    return quantile_target


@torch.no_grad()
def gae_quantile_simple(reward, r_dist, done, r_bdist, discount, gae_lambda):
    """(原样) 逐分位数 GAE (sorted 分位数): 返回 (advantage_quantile, return_quantile)。"""
    r_dist_t = torch.sort(r_dist)[0]
    r_bdist_t = torch.sort(r_bdist)[0]

    advantage_quantile = torch.zeros_like(r_dist)
    return_quantile = torch.zeros_like(r_dist)

    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd

    advantage_quantile[-1] = reward[-1][:, None] + nd[-1][:, None] * discount * r_bdist_t - r_dist_t[-1]
    for t in reversed(range(len(reward) - 1)):
        delta = reward[t][:, None] + nd[t][:, None] * discount * r_dist_t[t + 1] - r_dist_t[t]
        advantage_quantile[t] = delta + discount * gae_lambda * nd[t][:, None] * advantage_quantile[t + 1]
    return_quantile[:] = advantage_quantile + r_dist_t
    return advantage_quantile, return_quantile


@torch.no_grad()
def compute_prob_ratio(cost, c_weibull_tail, c_w_alpha, c_w_beta, done, c_bw_alpha, c_bw_beta,
                       discount, log_clip_range=0.2, normalize=True):
    """(原样) LDP/Weibull 概率比: 用相邻时刻 Weibull 尾部模型外推 outage 概率变化率。"""
    EPS = 1e-3
    nd = 1 - done
    nd = nd.type(cost.dtype) if isinstance(nd, torch.Tensor) else nd

    c_w_alpha_next = torch.cat((c_w_alpha[1:], c_bw_alpha), 0)
    c_w_beta_next = torch.cat((c_w_beta[1:], c_bw_beta), 0)

    c_weibull_tail_target = (c_weibull_tail - cost[:, :, None]) / discount
    possible_ind = (c_weibull_tail_target > EPS)
    c_weibull_tail_target = torch.clamp(c_weibull_tail_target, min=EPS)

    log_prob_ratio = torch.zeros_like(c_weibull_tail)
    log_prob_c_weibull_tail = torch.log(c_w_alpha)[:, :, None] - c_w_alpha[:, :, None] * torch.log(c_w_beta)[:, :, None] \
        + (c_w_alpha - 1)[:, :, None] * torch.log(c_weibull_tail) \
        - torch.pow(c_weibull_tail / c_w_beta[:, :, None], c_w_alpha[:, :, None])
    log_prob_c_weibull_tail_target = torch.log(c_w_alpha_next)[:, :, None] - c_w_alpha_next[:, :, None] * torch.log(c_w_beta_next)[:, :, None] \
        + (c_w_alpha_next - 1)[:, :, None] * torch.log(c_weibull_tail_target) \
        - torch.pow(c_weibull_tail_target / c_w_beta_next[:, :, None], c_w_alpha_next[:, :, None])

    log_prob_ratio[possible_ind] = (log_prob_c_weibull_tail_target
                                    - torch.log(torch.tensor(discount))
                                    - log_prob_c_weibull_tail)[possible_ind]

    # Normalize log_prob_ratio (原样)
    if normalize:
        avg_log = torch.mean(log_prob_ratio, dim=[0, 1])
        std_log = torch.std(log_prob_ratio, dim=[0, 1])
        log_prob_ratio = (log_prob_ratio - avg_log[None, None, :]) / (5.0 * std_log[None, None, :])

    log_prob_ratio = nd[:, :, None] * torch.clamp(log_prob_ratio, min=-log_clip_range, max=log_clip_range)
    prob_ratio = torch.exp(log_prob_ratio)

    return prob_ratio
