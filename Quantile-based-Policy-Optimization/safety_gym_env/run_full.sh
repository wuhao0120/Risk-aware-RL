#!/bin/bash
# Two-stage first-round launcher.
#   ./run_full.sh qpo              # QPO, four envs, seeds 0/1/2; then inspect results and stop.
#   ENVS_CSV=SimpleButton MAX_PARALLEL=3 ./run_full.sh qpo  # one env, three seeds.
#   ACTOR_LR0=0.0002 RUN_LABEL=qpo_actor_lr2e4_only ENVS_CSV=SimpleButton \
#       MAX_PARALLEL=3 ./run_full.sh qpo 0,1,2          # one-variable LR ablation.
#   Q_SIMPLEBUTTON=... Q_DYNAMIC=... Q_GREMLIN=... Q_DYNAMICBUTTON=... ./run_full.sh main
#   ALGOS_CSV=QCPO ENVS_CSV=SimpleButton Q_SIMPLEBUTTON=-0.03 ./run_full.sh main 0,1,2
# The script never derives or scans q automatically.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/vepfs-mlp2/c20250510/251204033/.conda/envs/zprl/bin/python}
# Spawned MuJoCo workers import NumPy/PyTorch but do scalar environment work. Restricting each
# numerical runtime to one thread prevents B workers from multiplying into thousands of idle
# BLAS/OpenMP threads; CUDA kernels in the parent training process are unaffected.
THREADS_PER_PROCESS=${THREADS_PER_PROCESS:-1}
export OMP_NUM_THREADS="$THREADS_PER_PROCESS"
export MKL_NUM_THREADS="$THREADS_PER_PROCESS"
export OPENBLAS_NUM_THREADS="$THREADS_PER_PROCESS"
export NUMEXPR_NUM_THREADS="$THREADS_PER_PROCESS"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_PROCESS"
export BLIS_NUM_THREADS="$THREADS_PER_PROCESS"
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
PHASE=${1:-qpo}
SEEDS_CSV=${2:-0,1,2}
case "$PHASE" in
    qpo) ALGOS=(QPO) ;;
    main) ALGOS=(QCPO QCPO_REF DQCAC) ;;
    *) echo "usage: ./run_full.sh <qpo|main> [comma_separated_seeds]" >&2; exit 2 ;;
esac
# A scoped algorithm filter lets a research stage run one implemented method without creating a
# separate launcher. The default qpo/main algorithm sets above remain unchanged when it is absent.
if [ -n "${ALGOS_CSV:-}" ]; then
    IFS=',' read -r -a ALGOS <<< "$ALGOS_CSV"
    for ALGO in "${ALGOS[@]}"; do
        case "$ALGO" in
            QPO|QCPO|QCPO_REF|DQCAC) ;;
            *) echo "unknown ALGOS_CSV entry: $ALGO" >&2; exit 2 ;;
        esac
    done
fi
ENVS_CSV=${ENVS_CSV:-SimpleButton,Dynamic,Gremlin,DynamicButton}
IFS=',' read -r -a ENVS <<< "$ENVS_CSV"
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
# Actor LR is an explicit controlled variable shared by all algorithms in this experiment round.
# RUN_LABEL is written into filenames, W&B names/tags, and config so an ablation is self-describing.
ACTOR_LR0=${ACTOR_LR0:-0.0002}
RUN_LABEL=${RUN_LABEL:-}
if [ -n "$RUN_LABEL" ] && [[ ! "$RUN_LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "RUN_LABEL must match [A-Za-z0-9][A-Za-z0-9._-]*" >&2
    exit 2
fi
# Optional prior tag, e.g. RESUME_TAG=full_20260729T140009Z_220827. Each seed resolves its
# own latest.pt while the resumed run writes into a new timestamped checkpoint directory.
RESUME_TAG=${RESUME_TAG:-}
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LABEL_SUFFIX=${RUN_LABEL:+_$RUN_LABEL}
LABEL_DISPLAY=""
if [ -n "$RUN_LABEL" ]; then LABEL_DISPLAY=" | label=$RUN_LABEL"; fi
RUN_ID="${PHASE}${LABEL_SUFFIX}_${RUN_STAMP}_$$"
RUN_TAG="${RUN_LABEL:-full}_${RUN_STAMP}_$$"
LOG_DIR=${LOG_DIR:-"_runs/logs/$RUN_ID"}
PROGRESS_DIR=${PROGRESS_DIR:-"_runs/progress/$RUN_ID"}
STOP_FILE="$PROGRESS_DIR/.workers_stopped"
mkdir -p "$LOG_DIR" "$PROGRESS_DIR"
export MUJOCO_GL=${MUJOCO_GL:-egl}
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

for ENV in "${ENVS[@]}"; do
    case "$ENV" in
        SimpleButton|Dynamic|Gremlin|DynamicButton) ;;
        *) echo "unknown ENVS_CSV entry: $ENV" >&2; exit 2 ;;
    esac
done

threshold_for() {
    case "$1" in
        SimpleButton) printf '%s' "${Q_SIMPLEBUTTON:-}" ;;
        Dynamic) printf '%s' "${Q_DYNAMIC:-}" ;;
        Gremlin) printf '%s' "${Q_GREMLIN:-}" ;;
        DynamicButton) printf '%s' "${Q_DYNAMICBUTTON:-}" ;;
    esac
}

if [ "$PHASE" = main ]; then
    for ENV in "${ENVS[@]}"; do
        if [ -z "$(threshold_for "$ENV")" ]; then
            echo "main phase paused: missing user-selected q for $ENV" >&2
            exit 2
        fi
    done
fi

run_one() {
    local ALGO=$1 ENV=$2 SEED=$3
    local logf="$LOG_DIR/${ALGO}_${ENV}_s${SEED}.log"
    local statusf="$PROGRESS_DIR/${ALGO}_${ENV}_s${SEED}.json"
    local q_args=()
    local resume_args=()
    local experiment_args=(actor_lr0="$ACTOR_LR0")
    if [ -n "$RUN_LABEL" ]; then experiment_args+=(experiment_label="$RUN_LABEL"); fi
    if [ "$ALGO" != QPO ]; then q_args=(--return_threshold "$(threshold_for "$ENV")"); fi
    if [ -n "$RESUME_TAG" ]; then
        local resume_path="_runs/checkpoints/${ALGO}_${ENV}_${RESUME_TAG}_s${SEED}/latest.pt"
        if [ ! -f "$resume_path" ]; then
            echo "missing resume checkpoint: $resume_path" >> "$logf"
            return 2
        fi
        resume_args=(--resume "$resume_path")
    fi
    # Child output is append-only and reconnectable with tail -f. Atomic status snapshots drive
    # exactly one parent progress bar, so concurrent environments never print four tqdm lines.
    printf '\n[%s] start algo=%s env=%s seed=%s\n' "$(date -u +%FT%TZ)" \
        "$ALGO" "$ENV" "$SEED" >> "$logf"
    # exec+setsid makes this background job the process-group leader. The EXIT trap can then
    # stop exactly this trainer and all vector workers without touching unrelated jobs.
    exec setsid "$PY" -u run_experiment.py \
        --algo "$ALGO" --env "$ENV" --seed "$SEED" \
        --num_eval "$NUM_EVAL" --validation_episodes "$VALIDATION_EPISODES" \
        --validation_every "$VALIDATION_EVERY" \
        --checkpoint_every "$CHECKPOINT_EVERY" --tag "$RUN_TAG" \
        --wandb_mode "$WANDB_MODE" "${q_args[@]}" "${resume_args[@]}" \
        --set num_iterations="$ITERS" num_envs="$NUM_ENVS" horizon="$HORIZON" \
        show_progress=false show_eval_progress=false progress_file="$statusf" \
        wandb_name="${ALGO}_${ENV}_s${SEED}_${RUN_ID}" "${experiment_args[@]}" \
        >> "$logf" 2>&1
}

# The denominator includes queued jobs as zero progress. For the default QPO stage it is
# 3 seeds * 4 environments * ITERS, producing one ETA for the complete invocation.
EXPECTED_RUNS=$(( ${#SEEDS[@]} * ${#ALGOS[@]} * ${#ENVS[@]} ))
"$PY" -u progress_monitor.py \
    --status-dir "$PROGRESS_DIR" \
    --expected-runs "$EXPECTED_RUNS" \
    --total-per-run "$ITERS" \
    --label "${PHASE^^} $ENVS_CSV | ${#SEEDS[@]} seeds | B=$NUM_ENVS | actor_lr=$ACTOR_LR0${LABEL_DISPLAY}" \
    --stop-file "$STOP_FILE" &
MONITOR_PID=$!

TRAIN_PIDS=()

# Clean up this invocation's exact trainer process groups on Ctrl-C, TERM, or a shell error.
# Normal completion also enters this trap; kill -0 filters out already-finished workers.
cleanup_children() {
    local STATUS=$?
    trap - EXIT INT TERM
    touch "$STOP_FILE"
    for PID in "${TRAIN_PIDS[@]}"; do
        if kill -0 "$PID" 2>/dev/null; then kill -TERM -- "-$PID" 2>/dev/null || true; fi
    done
    if kill -0 "$MONITOR_PID" 2>/dev/null; then kill -TERM "$MONITOR_PID" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    exit "$STATUS"
}
trap cleanup_children EXIT INT TERM

for SEED in "${SEEDS[@]}"; do
    for ALGO in "${ALGOS[@]}"; do
        for ENV in "${ENVS[@]}"; do
            while [ "$(jobs -rp | wc -l)" -ge "$((MAX_PARALLEL + 1))" ]; do sleep 5; done
            run_one "$ALGO" "$ENV" "$SEED" &
            TRAIN_PIDS+=("$!")
        done
    done
done

FAILED=0
for PID in "${TRAIN_PIDS[@]}"; do
    if ! wait "$PID"; then FAILED=1; fi
done
touch "$STOP_FILE"
wait "$MONITOR_PID" || true

if [ "$FAILED" -ne 0 ]; then
    echo "$PHASE stage failed; inspect $LOG_DIR and $PROGRESS_DIR" >&2
    exit 1
fi
if [ "$PHASE" = qpo ]; then
    echo "QPO stage complete. Inspect per-seed linear empirical Q_0.2 results and choose fixed q for the selected environments before phase main. Logs: $LOG_DIR Status: $PROGRESS_DIR"
else
    echo "reward-side main stage complete. Logs: $LOG_DIR Status: $PROGRESS_DIR"
fi
