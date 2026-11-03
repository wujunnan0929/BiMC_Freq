#!/usr/bin/env bash
# Final-method stage (PAPER_PLAN sec 12.9), development only: select the M4 family on a dataset
# whose bank and baseline selections (A2, Proto, BiMC, ...) already exist (run dev_core.sh first),
# then gates and a summary. No test image is read.
#   bash fdrfreq/scripts/dev_m4.sh cub200
# Env: TAG (vitb16), SHOTS ("1 5"), SEQ (10), SPLIT_MODE (contiguous)
set -euo pipefail
DS=$1
TAG=${TAG:-vitb16}; SHOTS=${SHOTS:-"1 5"}; SEQ=${SEQ:-10}; MODE=${SPLIT_MODE:-contiguous}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
M4="M4 M4_lowvar M4_mag M4_fisher M4_freq M4_ebratio M4_random M4_segment M4_visualsim M4_holistic
M4_soft M4_ebmask P4_Proto P4_BiMC Ours_ULA_M4"
DIRS=""
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  for need in A2 Proto BiMC; do
    if [ ! -f "$D/locked_$need.json" ]; then echo "missing $D/locked_$need.json: run dev_core.sh first"; exit 1; fi
  done
  python -m fdrfreq.run select --bank "$BANK" --shot "$SHOT" --split-mode "$MODE" --objective hm \
    --tolerance 100 --sequences "$SEQ" --matrix fdrfreq/configs/paper.json --out "$D" --entries $M4
  python -m fdrfreq.run gates --dir "$D"
  DIRS="$DIRS $D"
done
python -m fdrfreq.run summary --dirs $DIRS --out "$OUT/summary_m4.txt" --entries \
  Proto TEEN FDR_tf_published BiMC BiMC_ens ULA A2 K0 M1 M3 Q7e_eb $M4
echo "Send $OUT/summary_m4.txt and $OUT/shot*_hm/gates.txt back."
