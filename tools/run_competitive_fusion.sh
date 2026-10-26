#!/usr/bin/env bash
# Run from any directory; all paths below are relative to the repository.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-cub200}"
CACHE="${CACHE:-outputs/safd_metric_v2_20261022/cache/${DATASET}_safd.npz}"
OUT="${OUT:-outputs/competitive_fusion_${DATASET}_20261026}"
FIT_SEED="${FIT_SEED:-2}"
DATA_ROOT="${DATA_ROOT:-../datasets/}"
GPU="${GPU:-0;}"
read -r -a FITTING_SEEDS <<< "${FITTING_SEEDS:-2 3 4}"
read -r -a EVAL_SEEDS <<< "${EVAL_SEEDS:-1 2 3 4 5}"
mkdir -p "$OUT/logs" "$OUT/models"

case "${1:-screen}" in
  screen)
    for S in "${FITTING_SEEDS[@]}"; do
      "$PYTHON" -u tools/train_competitive_fusion.py \
        --cache "$CACHE" --method group --no-support-quality \
        --seed "$S" --sequences 20 --teacher-shot 15 --steps 0 \
        --device cpu --threads 1 \
        --output "$OUT/models/${DATASET}_fixed_fit${S}.npz" \
        2>&1 | tee "$OUT/logs/fixed_fit${S}.log"
    done
    ;;
  train)
    ANCHOR="$OUT/models/${DATASET}_fixed_fit${FIT_SEED}.npz"
    for NAME in group joint_evidence joint joint_safd; do
      EXTRA=()
      case "$NAME" in
        group) EXTRA=(--method group --no-support-quality) ;;
        joint_evidence) EXTRA=(--method joint --no-support-quality) ;;
        joint) EXTRA=(--method joint) ;;
        joint_safd) EXTRA=(--method joint --auxiliary safd --auxiliary-probability 0.25) ;;
      esac
      "$PYTHON" -u tools/train_competitive_fusion.py \
        --cache "$CACHE" --anchor-model "$ANCHOR" "${EXTRA[@]}" \
        --seed "$FIT_SEED" --sequences 20 --teacher-shot 15 \
        --steps 300 --episode-pool 100 --validate-every 50 \
        --device cpu --threads 1 \
        --output "$OUT/models/${DATASET}_${NAME}_fit${FIT_SEED}.npz" \
        2>&1 | tee "$OUT/logs/${NAME}_fit${FIT_SEED}.log"
    done
    ;;
  plan|benchmark|summarize)
    MODE=(--dry-run)
    [[ "$1" != benchmark ]] || MODE=(--execute)
    [[ "$1" != summarize ]] || MODE=(--summarize-only)
    "$PYTHON" -u tools/run_competitive_fusion_experiments.py \
      --dataset "$DATASET" --seeds "${EVAL_SEEDS[@]}" \
      --model "group=$OUT/models/${DATASET}_group_fit${FIT_SEED}.npz" \
      --model "joint_evidence=$OUT/models/${DATASET}_joint_evidence_fit${FIT_SEED}.npz" \
      --model "joint=$OUT/models/${DATASET}_joint_fit${FIT_SEED}.npz" \
      --model "joint_safd=$OUT/models/${DATASET}_joint_safd_fit${FIT_SEED}.npz" \
      --reference-model joint --support-root "$OUT/support" \
      --output-root "$OUT/benchmark_fit${FIT_SEED}" "${MODE[@]}" \
      --opts DATASET.ROOT "$DATA_ROOT" DEVICE.GPU_ID "$GPU"
    ;;
  *)
    echo 'Usage: bash tools/run_competitive_fusion.sh screen|train|plan|benchmark|summarize' >&2
    exit 2
    ;;
esac
