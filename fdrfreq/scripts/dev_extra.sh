#!/usr/bin/env bash
# PAPER_PLAN sec 13, stage A on the existing banks (development episodes only, no test image):
# E1 S1_soft, E2 M4a_ape / M4a_apelow, E3 M4a_k01 / M4a_k02 (1-shot), E4 DCR_ens, E10a ULA_fb,
# E10b BiMC_rt (+ BiMC_rt_D when BiMC_rt gains >= 0.3 dev Last over BiMC_ens), E7 dev standard
# errors, E8 error profile, E9 budget prediction, E11 timing.
#   bash fdrfreq/scripts/dev_extra.sh cub200
# Env: TAG (vitb16), SHOTS ("1 5"), SEQ (10), SPLIT_MODE (contiguous)
set -euo pipefail
DS=$1
TAG=${TAG:-vitb16}; SHOTS=${SHOTS:-"1 5"}; SEQ=${SEQ:-10}; MODE=${SPLIT_MODE:-contiguous}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
HERE=$(dirname "$0")
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  for need in BiMC_ens A2 M4a; do
    if [ ! -f "$D/locked_$need.json" ]; then echo "missing $D/locked_$need.json"; exit 1; fi
  done
  ENTRIES="DCR_ens S1_soft M4a_ape M4a_apelow ULA_fb BiMC_rt"
  if [ "$SHOT" = "1" ]; then ENTRIES="$ENTRIES M4a_k01 M4a_k02"; fi
  python -m fdrfreq.run select --bank "$BANK" --shot "$SHOT" --split-mode "$MODE" --objective hm \
    --tolerance 100 --sequences "$SEQ" --matrix fdrfreq/configs/paper.json --out "$D" --entries $ENTRIES
  if [ "$SHOT" = "1" ]; then
    python "$HERE/_same_sel.py" "$D/select_DCR_ens.json" "$D/select_M4a.json" | tee "$D/r4_check.txt"
  fi
  GAIN=$(python -c "import sys; print(float(sys.argv[1]) - float(sys.argv[2]))" \
    "$(python "$HERE/_sel_last.py" "$D/select_BiMC_rt.json")" "$(python "$HERE/_sel_last.py" "$D/select_BiMC_ens.json")")
  echo "[$DS shot $SHOT] BiMC_rt - BiMC_ens dev Last: $GAIN"
  RT=""
  if python -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.3 else 1)" "$GAIN"; then
    python -m fdrfreq.run select --bank "$BANK" --shot "$SHOT" --split-mode "$MODE" --objective hm \
      --tolerance 100 --sequences "$SEQ" --matrix fdrfreq/configs/paper.json --out "$D" --entries BiMC_rt_D
    RT="BiMC_rt_D"
  fi
  python -m fdrfreq.run devse --bank "$BANK" --dir "$D" --shot "$SHOT" --sequences "$SEQ" --split-mode "$MODE" \
    --ref BiMC_ens --entries BiMC_ens A2 M4a DCR_ens S1_soft M4a_soft M4a_random M4a_ape M4a_apelow ULA_fb BiMC_rt $RT \
    | tee "$D/devse_shot${SHOT}.txt"
done
python -m fdrfreq.run profile --bank "$BANK" --out "$OUT" --shots $SHOTS | tee "$OUT/error_profile.txt"
python -m fdrfreq.run kpredict --bank "$BANK" --out "$OUT" --locked "$OUT/shot1_hm/locked_M4a.json" | tee "$OUT/budget_prediction.txt"
python -m fdrfreq.run timing --bank "$BANK" --out "$OUT" --locked "$OUT/shot1_hm/locked_DCR_ens.json" --shot 1 | tee "$OUT/timing.txt"
python -m fdrfreq.run summary --dirs $(for S in $SHOTS; do echo "$OUT/shot${S}_hm"; done) --out "$OUT/summary_extra.txt" --entries \
  BiMC_ens LA ULA ULA_fb BiMC_rt BiMC_rt_D A2 M4a DCR_ens S1_soft M4a_soft M4a_random M4a_lowvar M4a_fisher M4a_freq \
  M4a_ape M4a_apelow M4a_k01 M4a_k02
echo "Send $OUT/summary_extra.txt, $OUT/shot*_hm/devse_shot*.txt, $OUT/shot1_hm/r4_check.txt, $OUT/error_profile.json, $OUT/budget_prediction.json, $OUT/timing.txt back."
