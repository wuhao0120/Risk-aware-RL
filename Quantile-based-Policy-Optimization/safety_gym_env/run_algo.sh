#!/bin/bash
# Run one reward-side algorithm over the four paper environments.
# QPO needs no threshold. Other algorithms require explicit per-environment raw-return q values:
#   Q_SIMPLEBUTTON=... Q_DYNAMIC=... Q_GREMLIN=... Q_DYNAMICBUTTON=... ./run_algo.sh QCPO 0
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python}
ALGO=${1:?usage: ./run_algo.sh <QPO|QCPO|QCPO_REF|DQCAC> [seed]}
SEED=${2:-0}
case "$ALGO" in QPO|QCPO|QCPO_REF|DQCAC) ;; *) echo "unknown algo: $ALGO" >&2; exit 2;; esac
ENVS=(SimpleButton Dynamic Gremlin DynamicButton)
NUM_ENVS=${NUM_ENVS:-25}
MAX_PARALLEL=${MAX_PARALLEL:-4}
HORIZON=${HORIZON:-1000}
BUDGET=${BUDGET:-5000000}
ITERS=${ITERS:-$(( BUDGET / (NUM_ENVS * HORIZON) ))}
NUM_EVAL=${NUM_EVAL:-1000}
VALIDATION_EPISODES=${VALIDATION_EPISODES:-100}
VALIDATION_EVERY=${VALIDATION_EVERY:-10}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50}
WANDB_MODE=${WANDB_MODE:-online}
ACTOR_LR0=${ACTOR_LR0:-0.0002}
RUN_LABEL=${RUN_LABEL:-}
if [ -n "$RUN_LABEL" ] && [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_LABEL must match [A-Za-z0-9][A-Za-z0-9._-]*" >&2
    exit 2
fi
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LABEL_SUFFIX=${RUN_LABEL:+_$RUN_LABEL}
LABEL_DISPLAY=""
if [ -n "$RUN_LABEL" ]; then LABEL_DISPLAY=" | label=$RUN_LABEL"; fi
RUN_ID="${ALGO}_s${SEED}${LABEL_SUFFIX}_${RUN_STAMP}_$$"
RUN_TAG="${RUN_LABEL:-full}_${RUN_STAMP}_$$"
LOG_DIR=${LOG_DIR:-"_runs/logs/$RUN_ID"}
PROGRESS_DIR=${PROGRESS_DIR:-"_runs/progress/$RUN_ID"}
STOP_FILE="$PROGRESS_DIR/.workers_stopped"
mkdir -p "$LOG_DIR" "$PROGRESS_DIR"
export MUJOCO_GL=${MUJOCO_GL:-egl}

threshold_for() {
    case "$1" in
        SimpleButton) printf '%s' "${Q_SIMPLEBUTTON:-}" ;;
        Dynamic) printf '%s' "${Q_DYNAMIC:-}" ;;
        Gremlin) printf '%s' "${Q_GREMLIN:-}" ;;
        DynamicButton) printf '%s' "${Q_DYNAMICBUTTON:-}" ;;
    esac
}

if [ "$ALGO" != QPO ]; then
    for ENV in "${ENVS[@]}"; do
        if [ -z "$(threshold_for "$ENV")" ]; then
            echo "$ALGO requires explicit q values; set" \
                "Q_SIMPLEBUTTON/Q_DYNAMIC/Q_GREMLIN/Q_DYNAMICBUTTON" >&2
            exit 2
        fi
    done
fi

run_one() {
    local ENV=$1
    local logf="$LOG_DIR/${ALGO}_${ENV}_s${SEED}.log"
    local statusf="$PROGRESS_DIR/${ALGO}_${ENV}_s${SEED}.json"
    local q_args=()
    local experiment_args=(actor_lr0="$ACTOR_LR0")
    if [ -n "$RUN_LABEL" ]; then experiment_args+=(experiment_label="$RUN_LABEL"); fi
    if [ "$ALGO" != QPO ]; then
        q_args=(--return_threshold "$(threshold_for "$ENV")")
    fi
    # Append instead of truncating, but keep child tqdm/warnings out of the shared terminal.
    # The agent publishes atomic JSON to statusf; progress_monitor.py is the sole terminal writer.
    printf '\n[%s] start algo=%s env=%s seed=%s\n' "$(date -u +%FT%TZ)" \
        "$ALGO" "$ENV" "$SEED" >> "$logf"
    "$PY" -u run_experiment.py \
        --algo "$ALGO" --env "$ENV" --seed "$SEED" \
        --num_eval "$NUM_EVAL" --validation_episodes "$VALIDATION_EPISODES" \
        --validation_every "$VALIDATION_EVERY" \
        --checkpoint_every "$CHECKPOINT_EVERY" --tag "$RUN_TAG" \
        --wandb_mode "$WANDB_MODE" "${q_args[@]}" \
        --set num_iterations="$ITERS" num_envs="$NUM_ENVS" horizon="$HORIZON" \
        show_progress=false show_eval_progress=false progress_file="$statusf" \
        wandb_name="${ALGO}_${ENV}_${RUN_ID}" "${experiment_args[@]}" >> "$logf" 2>&1
}

# One parent-owned bar covers all four child runs. MAX_PARALLEL applies only to trainers;
# the lightweight monitor is deliberately excluded by allowing one additional background job.
"$PY" -u progress_monitor.py \
    --status-dir "$PROGRESS_DIR" \
    --expected-runs "${#ENVS[@]}" \
    --total-per-run "$ITERS" \
    --label "$ALGO seed=$SEED | actor_lr=$ACTOR_LR0${LABEL_DISPLAY}" \
    --stop-file "$STOP_FILE" &
MONITOR_PID=$!

TRAIN_PIDS=()
for ENV in "${ENVS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$((MAX_PARALLEL + 1))" ]; do sleep 3; done
    run_one "$ENV" &
    TRAIN_PIDS+=("$!")
done

# ``if wait`` avoids set -e aborting before the stop marker is written when one child fails.
FAILED=0
for PID in "${TRAIN_PIDS[@]}"; do
    if ! wait "$PID"; then FAILED=1; fi
done
touch "$STOP_FILE"
wait "$MONITOR_PID" || true

if [ "$FAILED" -ne 0 ]; then
    echo "$ALGO seed=$SEED failed; inspect $LOG_DIR and $PROGRESS_DIR" >&2
    exit 1
fi
echo "$ALGO seed=$SEED complete; child logs: $LOG_DIR; status: $PROGRESS_DIR"
