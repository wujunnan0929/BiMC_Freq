#!/usr/bin/env bash
# Error types of the stored test predictions for all three benchmarks (no new test evaluation).
#   bash fdrfreq/scripts/errtypes.sh
# Env: TAG (vitb16), DATASETS ("cub200 cifar100 miniimagenet")
set -euo pipefail
TAG=${TAG:-vitb16}
for DS in ${DATASETS:-cub200 cifar100 miniimagenet}; do
  python -m fdrfreq.errtypes --dataset "$DS" --tag "$TAG" > /dev/null
  echo "wrote outputs/fdrfreq/${DS}_${TAG}/errtypes.md and errtypes.json"
done
echo "Send outputs/fdrfreq/*_${TAG}/errtypes.json (and errtypes.md) back."
