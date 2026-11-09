#!/usr/bin/env bash
# PAPER_PLAN sec 13: the single test evaluation of the configurations locked by dev_extra.sh.
# evaluate-once refuses any entry that already has a test evaluation and logs every run in
# TEST_LOG.jsonl. Also recomputes every pre-specified pair with the two-level bootstrap.
#   bash fdrfreq/scripts/final_extra.sh cub200
# Env: TAG (vitb16), SHOTS ("1 5"), SEEDS ("1 2 3 4 5")
set -euo pipefail
DS=$1
TAG=${TAG:-vitb16}; SHOTS=${SHOTS:-"1 5"}; SEEDS=${SEEDS:-"1 2 3 4 5"}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
PAIRS="BiMC_ens:DCR_ens BiMC_ens:M4a DCR_ens:M4a A2:M4a S1_soft:M4a M4a_soft:M4a M4a_random:M4a M4a_lowvar:M4a M4a_fisher:M4a
M4a_freq:M4a M4a_ape:M4a M4a_apelow:M4a BiMC_ens:ULA_fb ULA_fb:DCR_ens LA:DCR_ens BiMC_ens:BiMC_rt BiMC_rt:BiMC_rt_D
BiMC_ens:M4a_k01 BiMC_ens:M4a_k02 M4a_k01:M4a M4a_k02:M4a"
REPORT=$OUT/test_extra.md
: > "$REPORT"
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  LOCKED=""
  for E in DCR_ens S1_soft M4a_ape M4a_apelow ULA_fb BiMC_rt BiMC_rt_D M4a_k01 M4a_k02; do
    if [ -f "$D/locked_$E.json" ]; then LOCKED="$LOCKED $D/locked_$E.json"; fi
  done
  python -m fdrfreq.run evaluate-once --bank "$BANK" --shot "$SHOT" --seeds $SEEDS --out "$D" --locked $LOCKED
  python -m fdrfreq.report --dir "$D" --anchor BiMC_ens --shot "$SHOT" --pairs $PAIRS \
    --hier-json "$D/pairs_shot${SHOT}.json" > "$D/tables_extra.md"
  { echo "# $DS $SHOT-shot (test, seeds $SEEDS)"; cat "$D/tables_extra.md"; echo; } >> "$REPORT"
done
echo "Send $REPORT and $OUT/shot*_hm/pairs_shot*.json back (after all three datasets, run holm.sh)."
