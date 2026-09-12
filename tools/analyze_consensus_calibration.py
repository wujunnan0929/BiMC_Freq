"""Inspect saved base calibration reports without loading images or a model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_incremental_residual_experiments import write_json


def inspect_report(report):
    if report.get('data_source') != 'base_training_only':
        raise ValueError('Expected a base-only consensus calibration report.')
    sizes = report['candidate_class_counts']
    old_way = sizes[0]
    old_count = new_count = 0
    for sequence in report['validation_sequences']:
        counts = [len(indices) for indices in sequence['query_indices']]
        for seen in sizes:
            old_count += sum(counts[:old_way])
            new_count += sum(counts[old_way:seen])
    total = old_count + new_count
    if total != report['validation_query_occurrences'] or not old_count or not new_count:
        raise ValueError('Saved sequence counts disagree with validation_query_occurrences.')
    candidates = []
    for candidate in report['candidate_results']:
        row = {key: candidate[key] for key in ('lambda', 'accuracy', 'corrected', 'damaged')}
        row.update(old_delta_pp=100 * (candidate['old_corrected'] - candidate['old_damaged']) / old_count,
                   new_delta_pp=100 * (candidate['new_corrected'] - candidate['new_damaged']) / new_count,
                   selection=candidate.get('selection'), stages=candidate.get('stages'))
        candidates.append(row)
    return {
        'schema_version': report.get('schema_version'), 'seed': report['seed'],
        'selected_lambda': report['selected_lambda'],
        'selection_objective': report.get('selection_objective', 'micro_all'),
        'old_query_fraction_all_stages': old_count / total,
        'old_query_occurrences': old_count, 'new_query_occurrences': new_count,
        'selection_audit': report.get('selection_audit'),
        'balanced_safe_lambda': report.get('balanced_safe_lambda'),
        'candidate_results': candidates,
        'limitation': ('Repeated queries are occurrences, not independent samples. '
                       'Schema 1 lacks stage counts: new stage-based choices cannot be recovered from it.'),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(args.input_root.rglob('consensus_calibration.json'))
    if not paths:
        parser.error('No consensus_calibration.json found; summary.json alone is insufficient.')
    if args.output.resolve() in {path.resolve() for path in paths}:
        parser.error('Output must not overwrite an input report.')
    rows = []
    for path in paths:
        report = inspect_report(json.loads(path.read_text(encoding='utf-8')))
        rows.append({'path': str(path), **report})
        print(f"{path.parent}: lambda={report['selected_lambda']:g}, "
              f"objective={report['selection_objective']}, "
              f"old occurrence fraction={report['old_query_fraction_all_stages']:.3%}")
        for candidate in report['candidate_results']:
            print(f"  lambda={candidate['lambda']:g} acc={candidate['accuracy']:.4f} "
                  f"old_delta={candidate['old_delta_pp']:+.4f} "
                  f"new_delta={candidate['new_delta_pp']:+.4f}")
    write_json(args.output, {'data_source': 'saved_base_calibration_only', 'reports': rows})
    print('Diagnostic report: ' + str(args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
