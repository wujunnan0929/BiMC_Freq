#!/usr/bin/env bash
# PAPER_PLAN sec 13 E5: single test evaluation of the shot sweep (K in {2, 3, 10}).
set -euo pipefail
DS=$1
TAG=${TAG:-vitb16}; KS=${KS:-"2 3 10"}; SEEDS=${SEEDS:-"1 2 3 4 5"}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
REPORT=$OUT/test_shots.md
: > "$REPORT"
for K in $KS; do
  D=$OUT/shot${K}_hm
  L=""; for E in BiMC BiMC_ens A2 DCR_ens M4a; do L="$L $D/locked_$E.json"; done
  python -m fdrfreq.run evaluate-once --bank "$BANK" --shot "$K" --seeds $SEEDS --out "$D" --locked $L
  python -m fdrfreq.report --dir "$D" --anchor BiMC_ens --shot "$K" \
    --pairs BiMC_ens:DCR_ens BiMC_ens:M4a DCR_ens:M4a A2:M4a --hier-json "$D/pairs_shot${K}.json" > "$D/tables_shots.md"
  { echo "# $DS $K-shot (test, seeds $SEEDS)"; cat "$D/tables_shots.md"; echo; } >> "$REPORT"
done
echo "Send $REPORT back."
