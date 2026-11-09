#!/usr/bin/env bash
# PAPER_PLAN sec 13 E6: second backbone. Extracts a bank without blur / DCT / augmentation views
# (GPU), checks BiMC parity, selects the reduced entry list on development episodes, and runs the
# error profile and the budget prediction. No test image is classified except by `parity`
# (published baselines only, as for ViT-B/16).
#   bash fdrfreq/scripts/dev_backbone.sh cub200 ../datasets 0
# Env: BACKBONE (ViT-L/14), TAG (vitl14), SHOTS ("1 5"), SEQ (10), SPLIT_MODE (contiguous)
set -euo pipefail
DS=$1; ROOT=$2; GPU=${3:-0}
BB=${BACKBONE:-ViT-L/14}; TAG=${TAG:-vitl14}; SHOTS=${SHOTS:-"1 5"}; SEQ=${SEQ:-10}
MODE=${SPLIT_MODE:-contiguous}
BANK=banks/${DS}_${TAG}
OUT=outputs/fdrfreq/${DS}_${TAG}
if [ ! -f "$BANK/meta.json" ]; then
  python -m fdrfreq.extract --dataset "$DS" --data-root "$ROOT" --out "$BANK" --backbone "$BB" \
    --gpu "$GPU" --sigmas --aug --dct-layers --seeds 1 2 3 4 5
fi
if [ ! -f "$OUT/parity_shot5.json" ]; then
  python -m fdrfreq.run parity --bank "$BANK" --seeds 1 2 3 4 5 --out "$OUT"
fi
# Parity gate (PAPER_PLAN 13.9 R6): on miniImageNet the 5-seed mean Last of BiMC / BiMC-dagger must be
# within 0.5 of the published ViT-L/14 values (BiMC supplementary, Table 4: 94.46 / 94.46). Stop otherwise;
# PARITY_OK=1 continues only after the difference has been diagnosed and written down.
python - "$OUT/parity_shot5.json" "$DS" "$BB" "${PARITY_OK:-0}" <<'EOF'
import json, sys
path, ds, bb, ok = sys.argv[1:5]
runs = json.load(open(path, encoding='utf-8'))
pub = {'BiMC': 94.46, 'BiMC_ens': 94.46} if (ds == 'miniimagenet' and bb == 'ViT-L/14') else {}
bad = []
for name in ('BiMC', 'BiMC_ens'):
    vals = [r[name]['last'] for r in runs.values()]
    mean = sum(vals) / len(vals)
    line = f'parity {name}: 5-shot test Last {mean:.2f} over {len(vals)} seeds'
    if name in pub:
        line += f' (published {pub[name]:.2f}, diff {mean - pub[name]:+.2f})'
        if abs(mean - pub[name]) > 0.5:
            bad.append(name)
    print(line)
if bad and ok != '1':
    sys.exit(f'parity gate failed for {bad}: send {path} back before any selection (PARITY_OK=1 to override)')
EOF
ENTRIES="BiMC BiMC_ens ZS_desc Proto TEEN LA ULA ULA_fb A1 A2 DCR_ens M4a M4a_random M4a_lowvar M4a_fisher M4a_soft S1_soft"
for SHOT in $SHOTS; do
  D=$OUT/shot${SHOT}_hm
  python -m fdrfreq.run select --bank "$BANK" --shot "$SHOT" --split-mode "$MODE" --objective hm \
    --tolerance 100 --sequences "$SEQ" --matrix fdrfreq/configs/paper.json --out "$D" --entries $ENTRIES
  python -m fdrfreq.run devse --bank "$BANK" --dir "$D" --shot "$SHOT" --sequences "$SEQ" --split-mode "$MODE" \
    --ref BiMC_ens --entries BiMC_ens A2 DCR_ens M4a M4a_random M4a_soft S1_soft | tee "$D/devse_shot${SHOT}.txt"
done
python -m fdrfreq.run profile --bank "$BANK" --out "$OUT" --shots $SHOTS | tee "$OUT/error_profile.txt"
python -m fdrfreq.run kpredict --bank "$BANK" --out "$OUT" --locked "$OUT/shot1_hm/locked_M4a.json" | tee "$OUT/budget_prediction.txt"
python -m fdrfreq.run summary --dirs $(for S in $SHOTS; do echo "$OUT/shot${S}_hm"; done) --out "$OUT/summary_backbone.txt" \
  --entries $ENTRIES
echo "Send $OUT/summary_backbone.txt, $OUT/parity_shot*.json, $OUT/error_profile.json, $OUT/budget_prediction.json back."
