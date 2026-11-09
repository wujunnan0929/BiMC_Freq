#!/usr/bin/env bash
# PAPER_PLAN sec 13 E5: shot sweep K in {2, 3, 10} (1 and 5 exist), development episodes only.
#   bash fdrfreq/scripts/dev_shots.sh cub200
set -euo pipefail
DS=$1
TAG=${TAG:-vitb16}; KS=${KS:-"2 3 10"}; SEQ=${SEQ:-10}; MODE=${SPLIT_MODE:-contiguous}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
for K in $KS; do
  D=$OUT/shot${K}_hm
  python -m fdrfreq.run select --bank "$BANK" --shot "$K" --split-mode "$MODE" --objective hm \
    --tolerance 100 --sequences "$SEQ" --matrix fdrfreq/configs/paper.json --out "$D" \
    --entries BiMC BiMC_ens A1 A2 DCR_ens M4a
  python -m fdrfreq.run devse --bank "$BANK" --dir "$D" --shot "$K" --sequences "$SEQ" --split-mode "$MODE" \
    --ref BiMC_ens --entries BiMC_ens A2 DCR_ens M4a | tee "$D/devse_shot${K}.txt"
done
python -m fdrfreq.run summary --dirs $(for K in 1 $KS 5; do echo "$OUT/shot${K}_hm"; done | sort -u) \
  --out "$OUT/summary_shots.txt" --entries BiMC BiMC_ens A2 DCR_ens M4a
echo "Send $OUT/summary_shots.txt and $OUT/shot*_hm/devse_shot*.txt back."
