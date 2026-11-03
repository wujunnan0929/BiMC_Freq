#!/usr/bin/env bash
# Frozen final method M4a (PAPER_PLAN sec 12.11), development only: select the M4a family on a
# dataset whose baseline selections (A2, Proto, BiMC) already exist, then gates, the k-sensitivity /
# matched-budget frontier at the selected alpha, and a summary. No test image is read.
#   bash fdrfreq/scripts/dev_m4a.sh cub200
# Env: TAG (vitb16), SHOTS ("1 5"), SEQ (10), SPLIT_MODE (contiguous), FRONTIER (1)
set -euo pipefail
DS=$1
TAG=${TAG:-vitb16}; SHOTS=${SHOTS:-"1 5"}; SEQ=${SEQ:-10}; MODE=${SPLIT_MODE:-contiguous}
FRONTIER=${FRONTIER:-1}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
M4A="M4a M4a_lowvar M4a_mag M4a_fisher M4a_freq M4a_ebratio M4a_random M4a_text M4a_segment
M4a_holistic M4a_soft M4a_ebmask P4a_Proto P4a_BiMC Ours_ULA_M4a"
DIRS=""
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  for need in A2 Proto BiMC; do
    if [ ! -f "$D/locked_$need.json" ]; then echo "missing $D/locked_$need.json: run dev_core.sh first"; exit 1; fi
  done
  python -m fdrfreq.run select --bank "$BANK" --shot "$SHOT" --split-mode "$MODE" --objective hm \
    --tolerance 100 --sequences "$SEQ" --matrix fdrfreq/configs/paper.json --out "$D" --entries $M4A
  python -m fdrfreq.run gates --dir "$D"
  if [ "$FRONTIER" = "1" ]; then
    # k sweep of four channel rankings at M4a's selected r, tau, alpha (frontier_shot*.json is overwritten;
    # keep the M4 copy)
    if [ -f "$D/frontier_shot${SHOT}.json" ] && [ ! -f "$D/frontier_m4_shot${SHOT}.json" ]; then
      cp "$D/frontier_shot${SHOT}.json" "$D/frontier_m4_shot${SHOT}.json"
    fi
    FB=$(python -c "import json,sys; print(1 if json.load(open(sys.argv[1]))['config'].get('segments') is None else 0)" "$D/locked_M4a.json")
    OVR=""
    if [ "$FB" = "1" ]; then OVR="--override fdrfreq/configs/frontier_m4a_fallback.json"; fi
    python -m fdrfreq.run frontier --bank "$BANK" --shot "$SHOT" --split-mode "$MODE" --sequences "$SEQ" \
      --base-locked "$D/locked_M4a.json" $OVR --alpha-from-locked \
      --axes prior:var prior:freq prior:random prior:lowvar --k 0.05 0.1 0.2 0.3 0.4 0.6 0.8 1.0 --out "$D"
    cp "$D/frontier_shot${SHOT}.json" "$D/frontier_m4a_shot${SHOT}.json"
  fi
  DIRS="$DIRS $D"
done
python -m fdrfreq.run summary --dirs $DIRS --out "$OUT/summary_m4a.txt" --entries \
  Proto TEEN FDR_tf_published BiMC BiMC_ens ULA A2 M3 M4 M4_visualsim M4_soft $M4A
echo "Send $OUT/summary_m4a.txt, $OUT/shot*_hm/gates.txt and $OUT/shot*_hm/frontier_m4a_shot*.json back."
