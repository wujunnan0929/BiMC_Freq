#!/usr/bin/env bash
# Holm correction across the six main settings (three datasets x 1/5-shot) for the pre-specified
# family (PAPER_PLAN sec 13.9 R7). The shot sweep (K = 2, 3, 10) is not part of the family.
#   bash fdrfreq/scripts/holm.sh            (after final_extra.sh on all three datasets)
set -euo pipefail
TAG=${TAG:-vitb16}
python -m fdrfreq.report --holm outputs/fdrfreq/*_${TAG}/shot[15]_hm/pairs_shot[15].json --family \
  BiMC_ens:DCR_ens BiMC_ens:M4a A2:M4a M4a_random:M4a M4a_lowvar:M4a M4a_fisher:M4a M4a_freq:M4a M4a_soft:M4a \
  | tee "outputs/fdrfreq/holm_${TAG}.md"
