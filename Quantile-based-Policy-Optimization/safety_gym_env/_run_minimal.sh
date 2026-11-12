#!/bin/bash
# 最小 pipeline: 3 算法 × 4 论文环境短训 (60 iters × B16 × T1000 ≈ 0.96M env-steps/run)。
# 分 3 批 (同算法 4 env 并行, 每批 4×16=64 CPU workers), wandb online 台账 tag=minimal。
set -u
cd "$(dirname "$0")"
PY=/vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python
mkdir -p _runs/logs
export MUJOCO_GL=egl

run_one() {  # $1=algo $2=env
    local extra=""
    $PY -u run_experiment.py --algo "$1" --env "$2" --seed 0 --num_eval 64 \
        --tag minimal --wandb_mode online \
        --set num_iterations=60 log_interval=10 wandb_name="$1_$2_minimal" \
        > "_runs/logs/$1_$2.log" 2>&1
    echo "[done] $1 $2 exit=$?"
}

for ALGO in QCPO DQCAC QCPO_REF; do
    echo "===== batch $ALGO ====="
    for ENV in SimpleButton Dynamic Gremlin DynamicButton; do
        run_one "$ALGO" "$ENV" &
    done
    wait
done
echo "MINIMAL PIPELINE ALL DONE"
