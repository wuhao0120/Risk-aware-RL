# -*- coding: utf-8 -*-
"""
wandb_monitor.py —— DQC-AC-β (GPU) 训练曲线【按需】体检脚本

用途: 用 wandb 公共 API 拉取一个 run 的历史曲线 (含【正在跑】的 run), 自动判 4 条成功判据 +
      发散/NaN/尺度/λ 卷绕等健康项, 打印简报并存一张关键曲线图。供人工随时运行, 或把输出贴给 Claude 分析。

前提: 在线 wandb 且已 `wandb login` (本机已同步到 wandb.ai)。

用法:
    python wandb_monitor.py                       # 自动选 project=RiskSensitiveEnv 里最新的 DQCACBeta* run
    python wandb_monitor.py --name DQCACBetaGPU   # 按显示名(正则子串)过滤, 取最新
    python wandb_monitor.py --run <run_id>        # 指定 run id (wandb run 页面 URL 末段那串)
    python wandb_monitor.py --project RiskSensitiveEnv --entity <你的entity>
    python wandb_monitor.py --no-plot             # 只出文字, 不画图

输出:
    控制台: 进度 / 4 判据 verdict / 健康检查 / 8 段趋势 / 针对性建议
    图片:   _wandb_monitor.png (mean / q_est / P(Z<=q) / lambda 四面板)
"""

import argparse
import os
import numpy as np

try:
    import wandb
except ImportError:
    raise SystemExit("未安装 wandb。先 `pip install wandb` 并 `wandb login`。")


# ============================================================ 取数 ============================================================
def fetch_run(args):
    """
    定位并返回目标 run 对象 (wandb.apis.public.Run)。
    优先级: --run(精确 id) > --name(显示名正则, 取最新) > 默认取 project 里最新的匹配 run。
    """
    api = wandb.Api(timeout=30)                                # 公共 API 客户端

    # 解析 entity (默认用登录账号的默认 entity)
    entity = args.entity or api.default_entity
    if entity is None:
        raise SystemExit("拿不到 wandb entity。请先 `wandb login`, 或用 --entity 指定。")
    project_path = f"{entity}/{args.project}"

    # 情况 1: 指定了 run id → 直接取
    if args.run:
        try:
            return api.run(f"{project_path}/{args.run}")
        except Exception as e:
            print(f"[warn] 按 id 取 run 失败 ({e!r}), 回退到按名字过滤最新。")

    # 情况 2/3: 按显示名正则过滤, 按创建时间倒序取最新一个
    filters = {"display_name": {"$regex": args.name}} if args.name else {}
    runs = api.runs(project_path, filters=filters, order="-created_at")
    runs = list(runs)
    if not runs:
        raise SystemExit(f"在 {project_path} 里没找到匹配 name~='{args.name}' 的 run。"
                         f" 检查 --project/--name, 或用 --run 指定 id。")
    return runs[0]                                             # 最新的那个


def load_history(run):
    """
    拉取 run 的逐步历史为 {metric: np.array}, 以及 x 轴 (累计 env-step, 缺失则用 _step)。
    samples 设大以避免 wandb 默认 500 点下采样 (训练迭代通常几百到上千)。
    """
    df = run.history(samples=200000, pandas=True)             # DataFrame, 行=每次 wandb.log
    if df is None or len(df) == 0:
        return {}, np.array([]), 0
    S = {}
    for col in df.columns:
        try:
            S[col] = df[col].to_numpy(dtype=float)            # 数值列转 np.array (非数值列跳过)
        except (TypeError, ValueError):
            pass
    x = S.get('progress/env_steps')
    if x is None or np.all(np.isnan(x)):
        x = np.asarray(df.get('_step', np.arange(len(df))), dtype=float)  # 退化用 _step
    return S, x, len(df)


# ============================================================ 体检工具 ============================================================
def last_seg(a, frac=8):
    """末 1/frac 段均值 (代表收敛后水平); 全 NaN 返回 nan。"""
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return float('nan')
    return float(np.mean(a[-max(1, len(a) // frac):]))


def seg_means(a, n=8):
    """分 n 段均值字符串 (看趋势方向)。"""
    a = np.asarray(a, float)
    L = max(1, len(a) // n)
    return '  '.join(f'{np.nanmean(a[i*L:(i+1)*L]):7.3f}' for i in range(n))


def has_nan_tail(a, tail=20):
    """最近 tail 步内是否出现 NaN/Inf (发散信号)。"""
    a = a[-tail:] if len(a) > tail else a
    return bool(np.any(~np.isfinite(a))) if len(a) else False


def verdict_line(level, name, detail):
    """统一格式: [OK]/[WARN]/[BAD] 名称 — 细节。"""
    mark = {'OK': 'OK  ', 'WARN': 'WARN', 'BAD': 'BAD '}[level]
    return f"  [{mark}] {name:22s} {detail}"


# ============================================================ 主流程 ============================================================
def main():
    p = argparse.ArgumentParser(description="DQC-AC-β(GPU) wandb 训练曲线按需体检")
    p.add_argument('--project', default='RiskSensitiveEnv', help='wandb project 名 (=env_name)')
    p.add_argument('--entity', default=None, help='wandb entity (默认用登录账号默认 entity)')
    p.add_argument('--run', default=None, help='精确 run id (run 页面 URL 末段)')
    p.add_argument('--name', default='DQCACBeta', help='按显示名正则子串过滤, 取最新 (默认 DQCACBeta*)')
    p.add_argument('--no-plot', action='store_true', help='只出文字不画图')
    args = p.parse_args()

    run = fetch_run(args)
    cfg = dict(run.config)
    q = float(cfg.get('quantile_threshold', 5.0))             # 约束阈值 q
    alpha = float(cfg.get('q_alpha', 0.25))                   # 违反概率上限 α
    lam_max = float(cfg.get('lambda_max', 50.0))              # λ 上界
    adv_norm = cfg.get('advantage_norm', '?')
    beta = cfg.get('beta', '?')

    S, x, n_rows = load_history(run)
    print("=" * 78)
    print(f"RUN: {run.name}  (id={run.id}, state={run.state})")
    print(f"  config: algo={cfg.get('algo_name','?')} advantage_norm={adv_norm} beta={beta} "
          f"q={q} alpha={alpha} lambda_max={lam_max} "
          f"num_envs={cfg.get('num_envs','?')} num_iterations={cfg.get('num_iterations','?')}")
    if n_rows == 0:
        print("  [!] 该 run 还没有历史数据 (刚启动?), 稍后再试。")
        return
    env_step_now = x[~np.isnan(x)][-1] if np.any(~np.isnan(x)) else float('nan')
    if 'progress/iteration' in S:                             # 取【最新】迭代号 (非均值)
        _it = S['progress/iteration'][~np.isnan(S['progress/iteration'])]
        iters_now = float(_it[-1]) if len(_it) else n_rows
    else:
        iters_now = n_rows
    print(f"  进度: 已记录 {n_rows} 步 | 当前 env_step≈{env_step_now:,.0f} | iteration≈{iters_now:.0f}")

    def g(k):
        return S.get(k, np.array([np.nan]))

    # ---------------- 4 条成功判据 ----------------
    print("\n----- 4 条成功判据 (末 1/8 段) -----")
    q_est = last_seg(g('disc_reward/quantile_reward'))
    P = last_seg(g('constraint/empirical_prob'))
    lam = last_seg(g('lambda/value'))
    mean_r = last_seg(g('disc_reward/aver_reward'))
    issues = []                                                # 收集问题→末尾给建议

    # 判据1: q_est ≥ q
    if np.isnan(q_est):
        print(verdict_line('WARN', 'q_est (α-分位数)', '无数据'))
    elif q_est >= q * 0.98:
        print(verdict_line('OK', 'q_est (α-分位数)', f'{q_est:.3f} ≥ q={q} ✓'))
    else:
        print(verdict_line('BAD', 'q_est (α-分位数)', f'{q_est:.3f} < q={q} (尾部仍偏低)'))
        issues.append('q_est_low')

    # 判据2: P(Z≤q) ≤ α
    if np.isnan(P):
        print(verdict_line('WARN', 'P(Z≤q)', '无数据'))
    elif P <= alpha + 0.01:
        print(verdict_line('OK', 'P(Z≤q)', f'{P:.3f} ≤ α={alpha} ✓'))
    else:
        print(verdict_line('BAD', 'P(Z≤q)', f'{P:.3f} > α={alpha} (约束未满足)'))
        issues.append('P_high')

    # 判据3: λ settle (不顶上界, 不一直爬)
    lam_series = g('lambda/value')
    lam_series = lam_series[~np.isnan(lam_series)]
    if len(lam_series) >= 16:
        prev = float(np.mean(lam_series[-len(lam_series)//4: -len(lam_series)//8])) if len(lam_series) >= 8 else lam
        climbing = (lam - prev) > 0.08 * max(abs(prev), 1.0)   # 末段较前段仍明显上升
    else:
        climbing = False
    if np.isnan(lam):
        print(verdict_line('WARN', 'λ', '无数据'))
    elif lam >= 0.98 * lam_max:
        print(verdict_line('BAD', 'λ', f'{lam:.2f} 顶到 lambda_max={lam_max} (约束太难/归一化尺度异常)'))
        issues.append('lam_pegged')
    elif climbing:
        print(verdict_line('WARN', 'λ', f'{lam:.2f} 仍在爬 (未 settle)'))
        issues.append('lam_climbing')
    else:
        print(verdict_line('OK', 'λ', f'{lam:.2f} 已 settle (< {lam_max})'))

    # 判据4: mean 稳在高位 (不崩)
    mean_series = g('disc_reward/aver_reward')
    mean_peak = np.nanmax(mean_series) if np.any(~np.isnan(mean_series)) else float('nan')
    if np.isnan(mean_r):
        print(verdict_line('WARN', 'mean return', '无数据'))
    elif not np.isnan(mean_peak) and mean_r < 0.7 * mean_peak:
        print(verdict_line('BAD', 'mean return', f'{mean_r:.2f} 较峰值 {mean_peak:.2f} 显著回落 (过约束/崩溃?)'))
        issues.append('mean_collapse')
    else:
        print(verdict_line('OK', 'mean return', f'{mean_r:.2f} (峰值 {mean_peak:.2f})'))

    # ---------------- 健康检查 ----------------
    print("\n----- 健康检查 -----")
    # NaN/发散
    nan_keys = [k for k in ['disc_reward/aver_reward', 'lambda/value', 'critic/quantile_huber_loss',
                            'actor/loss'] if has_nan_tail(g(k))]
    if nan_keys:
        print(verdict_line('BAD', 'NaN/发散', f'最近若干步出现非有限值: {nan_keys}'))
        issues.append('nan')
    else:
        print(verdict_line('OK', 'NaN/发散', '关键指标近段无 NaN/Inf'))

    # critic loss: 尺度 vs 失败 (对比 critic 预测 std 是否跟得上真实 return std)
    closs = last_seg(g('critic/quantile_huber_loss'))
    pred_std = last_seg(g('critic/pred_return_std'))
    real_std = last_seg(g('disc_reward/return_std'))
    if not np.isnan(pred_std) and not np.isnan(real_std) and real_std > 1e-6:
        ratio = pred_std / real_std
        if 0.6 <= ratio <= 1.5:
            print(verdict_line('OK', 'critic 校准', f'pred_std/real_std={ratio:.2f} (跟得上; loss={closs:.3f} 多为尺度而非失败)'))
        else:
            print(verdict_line('WARN', 'critic 校准', f'pred_std/real_std={ratio:.2f} (critic 没跟上回报分布尺度)'))
            issues.append('critic_miscal')

    # qcpo 归一化尺度
    sig_ret = last_seg(g('norm/return_sigma_ema'))
    sig_c = last_seg(g('norm/constraint_sigma_ema'))
    if not np.isnan(sig_ret):
        sc_txt = f'σ_ret={sig_ret:.2f}'
        sc_txt += f'  σ_c={sig_c:.4f}' if not np.isnan(sig_c) else ''
        bad_sigc = (not np.isnan(sig_c)) and (sig_c < 1e-3 or sig_c > 5.0)
        print(verdict_line('WARN' if bad_sigc else 'OK', 'qcpo 归一化尺度',
                           sc_txt + (' (σ_c 异常)' if bad_sigc else '')))
        if bad_sigc:
            issues.append('sigc_bad')

    # 约束校准误差
    cal = last_seg(g('constraint/cdf_calibration_error'))
    if not np.isnan(cal):
        print(verdict_line('OK' if cal < 0.15 else 'WARN', 'CDF 校准误差', f'|Ĝ-经验|={cal:.3f}'))

    # ---------------- 8 段趋势 ----------------
    print("\n----- 8 段趋势 (start → end) -----")
    for k, label in [('disc_reward/aver_reward', 'mean'),
                     ('disc_reward/quantile_reward', 'q_est'),
                     ('constraint/empirical_prob', 'P(Z≤q)'),
                     ('lambda/value', 'lambda'),
                     ('action/avg_risk_episode', 'risk r'),
                     ('disc_reward/return_std', 'return_std')]:
        if k in S:
            print(f'  {label:12s}: {seg_means(S[k])}')

    # ---------------- 针对性建议 ----------------
    print("\n----- 建议 -----")
    if not issues:
        print("  训练健康, 4 判据均达标, 无需改动。继续观察 / 可加 seed 验证可复现。")
    else:
        tips = {
            'P_high':       "P 压不下 α: 确认 advantage_norm='qcpo'(不是 separate); 已是 qcpo 则查 σ_c 是否异常、适当增大 warmup_iters。",
            'lam_climbing': "λ 仍卷绕: 多半是 advantage_norm 非 qcpo 或迭代数不够(λ 约需 250+ 迭代才 settle); 也可略升 actor_grad_clip。",
            'lam_pegged':   "λ 顶上界: 约束太难或归一化尺度异常; 检查 norm/constraint_sigma_ema, 必要时升 lambda_max 或确认 q 设得是否过严。",
            'q_est_low':    "q_est 低于 q: 若 P 已≤α 属边界正常(概率约束本就软); 若 P 也超标见上。想更贴约束可试 β=0.98。",
            'mean_collapse':"mean 回落=过约束: β 调小(0.90 更近视稳)、或确认 λ 是否过大、check 是否 advantage_norm 配错。",
            'nan':          "发散/NaN: 降 critic_lr 或 actor LR(theta_a)、收紧 critic_grad_clip; 检查是否 σ 归一化除零。",
            'critic_miscal':"critic 未跟上回报尺度: 增 updates_per_episode 或 critic_lr; 多为可观察项, 非致命。",
            'sigc_bad':     "σ_c 异常(过小/过大): 检查约束优势是否退化; σ_c 极小时 a_c 会被放大, 留意 actor 梯度。",
        }
        for it in dict.fromkeys(issues):                       # 去重保序
            print(f"  - {tips.get(it, it)}")

    # ---------------- 画图 ----------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 2, figsize=(14, 9))
            panels = [('disc_reward/aver_reward', 'mean return', None),
                      ('disc_reward/quantile_reward', 'alpha-quantile q_est', q),
                      ('constraint/empirical_prob', 'P(Z<=q)', alpha),
                      ('lambda/value', 'lambda', lam_max if lam_max <= 60 else None)]
            for ax_i, (k, title, hl) in zip(ax.flat, panels):
                xa = x
                ya = g(k)
                m = min(len(xa), len(ya))
                ax_i.plot(xa[:m], ya[:m], color='tab:red', lw=1.4)
                if hl is not None:
                    ax_i.axhline(hl, color='k', ls='--', alpha=.6,
                                 label=('lambda_max' if k == 'lambda/value' else f'target={hl}'))
                    ax_i.legend()
                ax_i.set_title(title)
                ax_i.set_xlabel('env_steps')
                ax_i.grid(alpha=.3)
            fig.suptitle(f"{run.name}  (state={run.state})")
            plt.tight_layout()
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_wandb_monitor.png')
            plt.savefig(out, dpi=120)
            print(f"\n曲线图已存: {out}")
        except Exception as e:
            print(f"\n[plot 跳过] {e!r}")
    print("=" * 78)


if __name__ == '__main__':
    main()
