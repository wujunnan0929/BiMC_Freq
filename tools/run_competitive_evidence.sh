#!/usr/bin/env bash
# A bounded eight-trial base-cache diagnosis; no benchmark task is launched.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
PYTHON="${PYTHON:-python}"
DATASET="${DATASET:-cub200}"
FIT_SEED="${FIT_SEED:-2}"
CACHE="${CACHE:-outputs/safd_metric_v2_20261022/cache/${DATASET}_safd.npz}"
ANCHOR="${ANCHOR:-outputs/competitive_fusion_${DATASET}_20261026/models/${DATASET}_fixed_fit${FIT_SEED}.npz}"
OUT="${OUT:-outputs/competitive_evidence_${DATASET}_fit${FIT_SEED}_20261026}"
if [[ -e "$OUT" || -e "${OUT}.log" || -e "${OUT}.tar.gz" ]]; then
  echo 'Choose a fresh OUT directory/log/archive; previous evidence is retained.' >&2
  exit 2
fi
mkdir -p "$(dirname -- "$OUT")"
"$PYTHON" -u tools/diagnose_competitive_evidence.py \
  --cache "$CACHE" --anchor-model "$ANCHOR" --output-dir "$OUT" \
  --seed "$FIT_SEED" --sequences 20 --teacher-shot 15 \
  --methods group joint \
  --representations raw standardized prior prior_standardized \
  --steps 300 --episode-pool 100 --validate-every 50 \
  --probe-steps 100 --device cpu --threads 1 \
  2>&1 | tee "${OUT}.log"
tar -czf "${OUT}.tar.gz" -C "$(dirname -- "$OUT")" "$(basename -- "$OUT")"
echo "Diagnostic summary: $OUT/summary.json"
echo "Return archive: ${OUT}.tar.gz"
