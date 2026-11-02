#!/usr/bin/env bash
# Stage 1 (development only): bank -> BiMC parity -> diagnostics -> base-only selection -> gates.
# No benchmark test image is used except by `parity`, which only reproduces the published baselines.
#   bash fdrfreq/scripts/dev.sh cub200 ../datasets 0
# Env: BACKBONE (default ViT-B/16), TAG (default vitb16), DCT_LAYERS, SHOTS (default "5 1"), SEQ (default 10)
set -euo pipefail
DS=$1; ROOT=$2; GPU=${3:-0}
BB=${BACKBONE:-ViT-B/16}; TAG=${TAG:-vitb16}; SHOTS=${SHOTS:-"5 1"}; SEQ=${SEQ:-10}
if [ "$DS" = "cub200" ]; then DCT=${DCT_LAYERS:-"6 9 12"}; else DCT=${DCT_LAYERS:-"9"}; fi
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
if [ ! -f "$BANK/meta.json" ]; then
  python -m fdrfreq.extract --dataset "$DS" --data-root "$ROOT" --out "$BANK" --backbone "$BB" \
    --gpu "$GPU" --dct-layers $DCT --sigmas 1 2 4 --aug flip gray --seeds 1 2 3 4 5 --keywords
fi
python -m fdrfreq.run parity --bank "$BANK" --seeds 1 2 3 4 5 --out "$OUT"
for SHOT in $SHOTS; do
  D=$OUT/shot$SHOT
  python -m fdrfreq.run diagnose --bank "$BANK" --shot "$SHOT" --out "$D"
  python -m fdrfreq.run select --bank "$BANK" --shot "$SHOT" --sequences "$SEQ" \
    --matrix fdrfreq/configs/paper.json --out "$D"
  python -m fdrfreq.run gates --dir "$D"
done
echo "Read $OUT/shot*/gates.txt, fix the final method composition, then run final.sh."
