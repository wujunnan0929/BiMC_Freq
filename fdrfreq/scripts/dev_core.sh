#!/usr/bin/env bash
# Stage 1 for a dataset, final protocol (PAPER_PLAN sec 12.8): bank -> BiMC parity -> base-only
# selection of the paper's core entries (contiguous pseudo split, HM objective) -> gates -> errconc.
# No benchmark test image is used except by `parity` (published baselines only).
#   bash fdrfreq/scripts/dev_core.sh cifar100 ../datasets 0
# Env: BACKBONE (ViT-B/16), TAG (vitb16), SHOTS ("1 5"), SEQ (10), SPLIT_MODE (contiguous),
#      EXTRA=1 also runs the frequency-analysis entries (Q1/Q2/Q3, F1/F3/F4/F5).
set -euo pipefail
DS=$1; ROOT=$2; GPU=${3:-0}
BB=${BACKBONE:-ViT-B/16}; TAG=${TAG:-vitb16}; SHOTS=${SHOTS:-"1 5"}; SEQ=${SEQ:-10}
MODE=${SPLIT_MODE:-contiguous}
if [ "$DS" = "cub200" ]; then DCT=${DCT_LAYERS:-"6 9 12"}; else DCT=${DCT_LAYERS:-"9"}; fi
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
if [ ! -f "$BANK/meta.json" ]; then
  python -m fdrfreq.extract --dataset "$DS" --data-root "$ROOT" --out "$BANK" --backbone "$BB" \
    --gpu "$GPU" --dct-layers $DCT --sigmas 1 2 4 --aug flip gray --seeds 1 2 3 4 5 --keywords
fi
if [ ! -f "$OUT/parity_shot5.json" ]; then
  python -m fdrfreq.run parity --bank "$BANK" --seeds 1 2 3 4 5 --out "$OUT"
fi
CORE="BiMC BiMC_ens ZS ZS_desc Proto TEEN FDR_tf_published LA ULA_published ULA A1 A2 K0 K1 K1_fdrcoord
K2_random M1 M3 M3_lowvar M3_mag M3_fisher M3_freq M3_ebratio M3_random M3_global M3_visualsim
M3_holistic Q7_eb Q7e_eb P_Proto P_BiMC Ours_ULA_M3"
if [ "${EXTRA:-0}" = "1" ]; then CORE="$CORE Q1 Q2_classperm Q3_random F1 F3_aug F4_noise F5_randband"; fi
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  python -m fdrfreq.run select --bank "$BANK" --shot "$SHOT" --split-mode "$MODE" --objective hm \
    --tolerance 100 --sequences "$SEQ" --matrix fdrfreq/configs/paper.json --out "$D" --entries $CORE
  python -m fdrfreq.run gates --dir "$D"
done
mkdir -p "$OUT/errconc"
python -m fdrfreq.run errconc --bank "$BANK" --out "$OUT/errconc" --shots 1 5 --split-mode "$MODE" \
  | tee "$OUT/errconc/errconc.txt"
python -m fdrfreq.run summary --dirs $(for S in $SHOTS; do echo "$OUT/shot${S}_hm"; done) --out "$OUT/summary.txt"
echo "Send $OUT/summary.txt, $OUT/shot*_hm/gates.txt and $OUT/errconc/errconc.txt back before any test evaluation."
