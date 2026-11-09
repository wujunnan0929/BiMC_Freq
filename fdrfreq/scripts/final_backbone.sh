#!/usr/bin/env bash
# PAPER_PLAN sec 13 E6: single test evaluation on the second backbone.
set -euo pipefail
DS=$1
TAG=${TAG:-vitl14}; SHOTS=${SHOTS:-"1 5"}; SEEDS=${SEEDS:-"1 2 3 4 5"}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
ENTRIES="BiMC BiMC_ens ZS_desc Proto TEEN LA ULA ULA_fb A1 A2 DCR_ens M4a M4a_random M4a_lowvar M4a_fisher M4a_soft S1_soft"
PAIRS="BiMC_ens:DCR_ens BiMC_ens:M4a DCR_ens:M4a A2:M4a M4a_random:M4a M4a_lowvar:M4a M4a_fisher:M4a M4a_soft:M4a S1_soft:M4a
ULA_fb:DCR_ens LA:DCR_ens"
REPORT=$OUT/test_backbone.md
: > "$REPORT"
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  L=""; for E in $ENTRIES; do L="$L $D/locked_$E.json"; done
  python -m fdrfreq.run evaluate-once --bank "$BANK" --shot "$SHOT" --seeds $SEEDS --out "$D" --locked $L
  python -m fdrfreq.report --dir "$D" --anchor BiMC_ens --shot "$SHOT" --pairs $PAIRS \
    --hier-json "$D/pairs_shot${SHOT}.json" > "$D/tables_backbone.md"
  { echo "# $DS $SHOT-shot $TAG (test, seeds $SEEDS)"; cat "$D/tables_backbone.md"; echo; } >> "$REPORT"
done
echo "Send $REPORT back."
