#!/bin/bash
# 单算法正式实验: 4 个论文环境并行, 日志打到【终端】(可 tee 一份到文件)。
# 设计给 tmux 用: 3 个 window 各跑一个算法, 实时看进度。
#
# 用法 (在 tmux 里):
#   ./run_algo.sh QCPO_REF 0
#   ./run_algo.sh DQCAC 0
#   ./run_algo.sh QCPO 0
#
# 环境变量:
#   NUM_ENVS=10          # B, 默认 10 (三算法同开时 12×10=120 worker)
#   MAX_PARALLEL=4       # 该算法下同时跑几个 env (默认 4=全开)
#   ITERS=               # 不设则按 5M/(B×1000) 自动算 (=500 when B=10)
set -u
cd "$(dirname "$0")"
PY=/vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python
ALGO=${1:?用法: ./run_algo.sh <QCPO_REF|DQCAC|QCPO> [seed]}
SEED=${2:-0}
ENVS=(SimpleButton Dynamic Gremlin DynamicButton)

NUM_ENVS=${NUM_ENVS:-10}              # 三算法同时开时: 12×10=120 worker ≈ 吃满 128 核
MAX_PARALLEL=${MAX_PARALLEL:-4}
HORIZON=${HORIZON:-1000}
BUDGET=${BUDGET:-5000000}
ITERS=${ITERS:-$(( BUDGET / (NUM_ENVS * HORIZON) ))}   # B=10 → 500 iter ≈ 5.0M steps

mkdir -p _runs/logs
export MUJOCO_GL=egl
ss -ltn 2>/dev/null | grep -q ':7890' || /vepfs-mlp2/c20250510/251204033/start_mihomo.sh

echo "===== $ALGO seed=$SEED B=$NUM_ENVS iters=$ITERS parallel=$MAX_PARALLEL $(date '+%m-%d %H:%M') ====="

run_one() {
    local ENV=$1
    local done_json="_runs/${ALGO}_${ENV}_full_s${SEED}.json"
    local logf="_runs/logs/${ALGO}_${ENV}_full_s${SEED}.log"
    if [ -f "$done_json" ]; then
        echo "[skip] $ALGO $ENV (已有 $done_json)"
        return 0
    fi
    echo "[start] $ALGO $ENV $(date '+%H:%M')"
    # tee: 终端实时看 + 文件留底 (失败可回看)
    $PY -u run_experiment.py --algo "$ALGO" --env "$ENV" --seed "$SEED" --num_eval 128 \
        --tag full --wandb_mode online \
        --set num_iterations=$ITERS num_envs=$NUM_ENVS log_interval=10 \
              wandb_name="${ALGO}_${ENV}_full_s${SEED}" \
        2>&1 | tee "$logf"
    local ec=${PIPESTATUS[0]}
    echo "[done] $ALGO $ENV exit=$ec $(date '+%H:%M')"
    return $ec
}

# 作业池: 最多 MAX_PARALLEL 个 env 同时跑
for ENV in "${ENVS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 3
    done
    run_one "$ENV" &
done
wait
echo "===== $ALGO seed=$SEED ALL DONE $(date '+%m-%d %H:%M') ====="
