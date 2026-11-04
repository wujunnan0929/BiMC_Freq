#!/usr/bin/env bash
# The single benchmark evaluation (PAPER_PLAN sec 12.12). Evaluates the configurations locked on the
# development episodes (shot*_hm) on the benchmark test set, 5 support seeds, and writes the tables
# with the pre-registered paired comparisons. Refuses to run twice (TEST_DONE marker).
#   bash fdrfreq/scripts/final_m4a.sh cub200
# Env: TAG (vitb16), SHOTS ("1 5"), SEEDS ("1 2 3 4 5"), SEQ (10), SPLIT_MODE (contiguous)
set -euo pipefail
DS=$1
TAG=${TAG:-vitb16}; SHOTS=${SHOTS:-"1 5"}; SEEDS=${SEEDS:-"1 2 3 4 5"}; SEQ=${SEQ:-10}
MODE=${SPLIT_MODE:-contiguous}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
BASELINES="ZS ZS_desc Proto TEEN FDR_tf_published BiMC BiMC_ens LA ULA_published ULA A1 A2"
OURS="M4a Ours_ULA_M4a"
CONTROLS="M4a_random M4a_lowvar M4a_fisher M4a_mag M4a_freq M4a_ebratio"
ABLATIONS="M4a_soft M4a_text M4a_segment M4a_holistic M4a_ebmask"
PLUGINS="P4a_Proto P4a_BiMC"
ENTRIES="$BASELINES $OURS $CONTROLS $ABLATIONS $PLUGINS"
PAIRS="BiMC_ens:M4a A2:M4a BiMC_ens:A2 M4a_random:M4a M4a_lowvar:M4a M4a_fisher:M4a M4a_freq:M4a
M4a_soft:M4a M4a:M4a_text M4a:Ours_ULA_M4a Proto:P4a_Proto TEEN:P4a_Proto BiMC:P4a_BiMC ULA:M4a"

# 1) everything that is still missing is selected on the development episodes first (no test image)
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  if [ -f "$D/TEST_DONE" ]; then
    echo "$D was already evaluated on the test set ($(cat "$D/TEST_DONE")). Not running again."; exit 1
  fi
  MISSING=""
  for E in $ENTRIES; do [ -f "$D/locked_$E.json" ] || MISSING="$MISSING $E"; done
  if [ -n "$MISSING" ]; then
    # keep the matrix order so that every chain target is locked before the entries that chain on it
    MISSING=$(python -c "import json,sys; o=[e['name'] for e in json.load(open('fdrfreq/configs/paper.json'))['entries']]; m=sys.argv[1:]; print(' '.join(sorted(m, key=o.index)))" $MISSING)
    echo "[$DS shot $SHOT] selecting on dev first: $MISSING"
    python -m fdrfreq.run select --bank "$BANK" --shot "$SHOT" --split-mode "$MODE" --objective hm \
      --tolerance 100 --sequences "$SEQ" --matrix fdrfreq/configs/paper.json --out "$D" --entries $MISSING
  fi
done

# 2) the one test evaluation
REPORT=$OUT/test_m4a.md
: > "$REPORT"
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  LOCKED=""
  for E in $ENTRIES; do LOCKED="$LOCKED $D/locked_$E.json"; done
  python -m fdrfreq.run evaluate --bank "$BANK" --shot "$SHOT" --seeds $SEEDS --out "$D" --locked $LOCKED
  date '+%Y-%m-%d %H:%M:%S' > "$D/TEST_DONE"
  python -m fdrfreq.report --dir "$D" --anchor BiMC_ens --shot "$SHOT" --pairs $PAIRS > "$D/tables.md"
  python -m fdrfreq.report --dir "$D" --anchor BiMC_ens --shot "$SHOT" --pairs $PAIRS --format latex > "$D/tables.tex"
  { echo "# $DS $SHOT-shot (test, seeds $SEEDS)"; cat "$D/tables.md"; echo; } >> "$REPORT"
done
echo "Send $REPORT back."
