#!/usr/bin/env bash
# Run the 2.4M pilot with 150 iterations, or the 10M protocol with 625.
# Usage: bash run_simplebutton_dca.sh OUTPUT_DIR SEED [ITERATIONS]
set -euo pipefail
out=${1:?Specify a fresh run directory}
seed=${2:?Specify a training seed}
iterations=${3:-625}
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export MUJOCO_GL=${MUJOCO_GL:-disable}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-python}" "$here/dca_simplebutton.py" --out "$out" --seed "$seed" \
    --iterations "$iterations" --batch 16 --workers "${WORKERS:-12}" --device "${DEVICE:-cuda}" \
    --gamma "${GAMMA:-1.0}" --width 96 --atoms 64 --actions 4 --warmup 8 --warmup-steps 50 --critic-steps 100 \
    --minibatch 512 --ppo-epochs 8 --actor-lr 0.0006 --critic-lr 0.0003 --replay-episodes 64 \
    --ki 0.4 --kp 5 --reward-scale 100 --eval-seed "${EVAL_SEED:-91000000}" --eval-episodes "${EVAL_EPISODES:-2048}" "${@:4}"
