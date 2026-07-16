#!/bin/bash
# 正式实验批处理 (CPU 拉满版)
#
# 调度: 把 (algo × env) 展平成队列, 用 MAX_PARALLEL 路并发跑满 (默认 8)。
# 预算固定 5M env-steps/run = NUM_ENVS × HORIZON × ITERS。
#
# 用法:
#   # 推荐: 8 路并发 × B=16 = 128 worker, 吃满 128 核
#   nohup ./run_full.sh 0 > _runs/logs/full_seed0.log 2>&1 &
#
#   # 可选: 4 路 × B=32 (同样 128 worker; 每批轨迹更多, 迭代更少)
#   NUM_ENVS=32 MAX_PARALLEL=4 nohup ./run_full.sh 0 > _runs/logs/full_seed0_b32.log 2>&1 &
#
#   # 只跑子集
#   nohup ./run_full.sh 1 QCPO_REF,DQCAC > _runs/logs/full_seed1.log 2>&1 &
#
# 断点续跑: 已有 _runs/<ALGO>_<ENV>_full_s<SEED>.json 的自动跳过。
set -u
cd "$(dirname "$0")"
PY=/vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python
SEED=${1:-0}
ALGOS=${2:-QCPO_REF,DQCAC,QCPO}
ENVS=(SimpleButton Dynamic Gremlin DynamicButton)

NUM_ENVS=${NUM_ENVS:-10}              # B: 三算法×4env 同时开 → 12×10=120 worker
MAX_PARALLEL=${MAX_PARALLEL:-12}      # 默认可同时跑满 12 个 (algo,env)
HORIZON=${HORIZON:-1000}
BUDGET=${BUDGET:-5000000}             # 论文口径 5M env-steps
ITERS=${ITERS:-$(( BUDGET / (NUM_ENVS * HORIZON) ))}   # B=10 → 500 iter ≈ 5.0M

mkdir -p _runs/logs
export MUJOCO_GL=egl

ss -ltn 2>/dev/null | grep -q ':7890' || /vepfs-mlp2/c20250510/251204033/start_mihomo.sh

echo "===== config seed=$SEED B=$NUM_ENVS iters=$ITERS budget≈$((NUM_ENVS*HORIZON*ITERS)) "
echo "      max_parallel=$MAX_PARALLEL algos=$ALGOS $(date '+%m-%d %H:%M') ====="

run_one() {  # $1=algo $2=env
    local done_json="_runs/$1_$2_full_s${SEED}.json"
    if [ -f "$done_json" ]; then
        echo "[skip] $1 $2 seed=$SEED"
        return 0
    fi
    echo "[start] $1 $2 seed=$SEED $(date '+%H:%M')"
    $PY -u run_experiment.py --algo "$1" --env "$2" --seed "$SEED" --num_eval 128 \
        --tag full --wandb_mode online \
        --set num_iterations=$ITERS num_envs=$NUM_ENVS log_interval=10 \
              wandb_name="$1_$2_full_s${SEED}" \
        > "_runs/logs/$1_$2_full_s${SEED}.log" 2>&1
    echo "[done] $1 $2 seed=$SEED exit=$? $(date '+%H:%M')"
}

# 展平 (algo, env) 队列, 用作业池保持最多 MAX_PARALLEL 个在跑
pids=()
for ALGO in ${ALGOS//,/ }; do
    for ENV in "${ENVS[@]}"; do
        # 等池子有空位
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
            sleep 5
        done
        run_one "$ALGO" "$ENV" &
        pids+=($!)
    done
done
wait
echo "FULL SEED=$SEED ALL DONE $(date '+%m-%d %H:%M')"
