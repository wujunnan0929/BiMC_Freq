#!/usr/bin/env bash
# Stage 2 (one shot at the test set): evaluate locked configurations and write the tables.
#   bash fdrfreq/scripts/final.sh cub200 "ZS Proto TEEN FDR_tf_published BiMC BiMC_ens LA A1 A2 K0 K1 ..."
# Env: TAG (default vitb16), SHOTS (default "5 1"), SEEDS (default "1 2 3 4 5"), ANCHOR (default A2)
set -euo pipefail
DS=$1; ENTRIES=$2
TAG=${TAG:-vitb16}; SHOTS=${SHOTS:-"5 1"}; SEEDS=${SEEDS:-"1 2 3 4 5"}; ANCHOR=${ANCHOR:-A2}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
for SHOT in $SHOTS; do
  D=$OUT/shot$SHOT
  LOCKED=""
  for E in $ENTRIES; do LOCKED="$LOCKED $D/locked_$E.json"; done
  python -m fdrfreq.run evaluate --bank "$BANK" --shot "$SHOT" --seeds $SEEDS --out "$D" --locked $LOCKED
  python -m fdrfreq.report --dir "$D" --anchor "$ANCHOR" --shot "$SHOT" > "$D/tables.md"
  python -m fdrfreq.report --dir "$D" --anchor "$ANCHOR" --shot "$SHOT" --format latex > "$D/tables.tex"
  echo "tables: $D/tables.md"
done
