"""Offline guidance diagnostics and a small allowlisted artifact archive; no model loading."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np


DATASETS = {'cub200': 100, 'cifar100': 60, 'miniimagenet': 60}
RAW_FILES = {'metrics.json', 'config.yaml', 'run_spec.json', 'run_state.json',
             'support.json', 'run.log', 'guidance_calibration.json'}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset_name(value):
    name = re.sub('[^a-z0-9]', '', str(value).lower())
    if name not in DATASETS:
        raise ValueError(f'Unknown dataset {value!r}; expected {list(DATASETS)}')
    return name


def load_predictions(path):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in ('sample_id', 'target', 'prediction')}
    for key, value in result.items():
        if value.ndim != 1:
            raise ValueError(f'{path}: {key} must be one-dimensional')
    if not len(result['target']) or len({len(v) for v in result.values()}) != 1:
        raise ValueError(f'{path}: arrays must have equal, nonzero lengths')
    ids = result['sample_id'].tolist()
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path}: duplicate sample IDs')
    if result['sample_id'].dtype.kind not in 'iuUS':
        raise ValueError(f'{path}: sample IDs must be integer or string values')
    for key in ('target', 'prediction'):
        if result[key].dtype.kind not in 'iu' or np.any(result[key] < 0):
            raise ValueError(f'{path}: {key} must contain nonnegative integer labels')
    return result


def compare_predictions(reference, candidate, num_base):
    """Align exact sample identities, then count corrected/damaged decisions."""
    a, b = load_predictions(reference), load_predictions(candidate)
    ids_a, ids_b = a['sample_id'].tolist(), b['sample_id'].tolist()
    if set(ids_a) != set(ids_b):
        raise ValueError('Prediction files have different sample ID sets')
    positions = {value: i for i, value in enumerate(ids_b)}
    order = [positions[value] for value in ids_a]
    target = a['target']
    if not np.array_equal(target, b['target'][order]):
        raise ValueError('Targets differ for aligned sample IDs')
    old, new = a['prediction'], b['prediction'][order]
    before, after = old == target, new == target
    result = {'candidate_reordered': ids_a != ids_b}
    for group, mask in (('all', np.ones(len(target), dtype=bool)),
                        ('base', target < num_base), ('novel', target >= num_base)):
        count = int(mask.sum())
        corrected = int((~before & after & mask).sum())
        damaged = int((before & ~after & mask).sum())
        result[group] = {
            'n': count, 'changed': int(((old != new) & mask).sum()),
            'corrected': corrected, 'damaged': damaged,
            'reference_accuracy': float(100 * before[mask].mean()) if count else None,
            'candidate_accuracy': float(100 * after[mask].mean()) if count else None,
            'delta_accuracy_pp': 100 * (corrected - damaged) / count if count else None,
        }
    return result


def numeric_summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return None
    if not np.isfinite(values).all():
        raise ValueError('Non-finite diagnostic values')
    return {'n': int(values.size), 'mean': float(values.mean()),
            'min': float(values.min()), 'max': float(values.max())}


def calibration_summary(report):
    rows = report.get('candidate_results', [])
    reference = next((row for row in rows if float(row['rho']) == 0), None)
    keys = ('mean_base_accuracy', 'mean_novel_accuracy', 'mean_harmonic_accuracy')
    result = {key: report.get(key) for key in (
        'selected_rho', 'selection_status', 'data_source', 'protocol_sha256',
        'fallback_to_G1', 'maximum_mean_group_drop_pp', 'calibration_seconds')}
    result['candidates'] = []
    for row in rows:
        entry = {key: row.get(key) for key in ('rho', 'feasible', *keys)}
        entry['delta_vs_rho0_pp'] = {
            key: float(row[key]) - float(reference[key]) for key in keys
            if reference is not None and key in row and key in reference}
        if reference is not None and row.get('stages') and reference.get('stages'):
            stages = {(v['sequence'], v['stage']): v for v in reference['stages']}
            current = {(v['sequence'], v['stage']): v for v in row['stages']}
            if len(stages) != len(reference['stages']) or len(current) != len(row['stages']):
                raise ValueError('Duplicate calibration sequence/stage identities')
            if stages.keys() != current.keys():
                raise ValueError('Calibration candidates use different sequence/stage identities')
            deltas = [current[k]['harmonic_accuracy'] - stages[k]['harmonic_accuracy'] for k in stages]
            entry['stage_harmonic_delta_vs_rho0_pp'] = numeric_summary(deltas)
        result['candidates'].append(entry)
    return result


def relation_summary(files):
    fields = ('movement_degrees', 'interscale_top5_overlap', 'entropy_visual',
              'entropy_semantic', 'entropy_paired', 'visual_semantic_relation_cosine')
    values = {key: [] for key in fields}
    for path in files:
        report = read_json(path).get('diagnostics') or {}
        for key in fields:
            if report.get(key) is not None:
                values[key].extend(np.asarray(report[key], dtype=float).reshape(-1).tolist())
    return {key: numeric_summary(v) for key, v in values.items()}


def prototype_comparison(reference, candidate):
    a, b = read_json(reference), read_json(candidate)
    ids_a, ids_b = a.get('class_ids', []), b.get('class_ids', [])
    da, db = a.get('diagnostics') or {}, b.get('diagnostics') or {}
    if not da.get('guided_prototypes') or not db.get('guided_prototypes'):
        return None
    if len(set(ids_a)) != len(ids_a) or len(set(ids_b)) != len(ids_b) or set(ids_a) != set(ids_b):
        raise ValueError('Guidance files have duplicate or different class IDs')
    pa, pb = np.asarray(da['guided_prototypes']), np.asarray(db['guided_prototypes'])
    order = [ids_b.index(value) for value in ids_a]
    if pa.ndim != 2 or pb.ndim != 2 or pa.shape != pb.shape or len(pa) != len(ids_a):
        raise ValueError('Guided prototype shapes disagree with class IDs')
    pb = pb[order]
    lengths = np.linalg.norm(pa, axis=1) * np.linalg.norm(pb, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Invalid guided prototype norm')
    angle = np.rad2deg(np.arccos(np.clip(np.sum(pa * pb, axis=1) / lengths, -1, 1)))
    return {'guided_G3_to_G4_angle_degrees': numeric_summary(angle)}


def allowlisted(path):
    return path.name in RAW_FILES or re.fullmatch(
        r'(?:guidance_session_\d+\.json|predictions_session_\d+\.npz)', path.name) is not None


def discover(roots, cases, variants, label=None):
    runs, issues, found, duplicates = {}, [], set(), set()
    for root in roots:
        if not root.is_dir():
            issues.append(f'Missing root directory: {root}')
            continue
        for spec_path in root.rglob('run_spec.json'):
            directory = spec_path.parent.resolve()
            if directory in found:
                continue
            found.add(directory)
            try:
                spec = read_json(spec_path)
                if spec.get('variant') not in variants:
                    continue
                metrics = read_json(directory / 'metrics.json') if (directory / 'metrics.json').is_file() else {}
                command = spec.get('command', [])
                command_dataset = None
                if '--data_cfg' in command:
                    command_dataset = dataset_name(Path(command[command.index('--data_cfg') + 1].replace('\\', '/')).stem)
                raw_dataset = metrics.get('dataset') or label
                if raw_dataset is None:
                    raw_dataset = command_dataset
                dataset = dataset_name(raw_dataset)
                if command_dataset and dataset != command_dataset:
                    raise ValueError(f'metrics dataset {dataset} conflicts with run_spec command dataset {command_dataset}')
                if label and dataset != label:
                    raise ValueError(f'--dataset {label} conflicts with metrics dataset {dataset}')
                seed = int(spec['seed'])
                if cases and (dataset, seed) not in cases:
                    continue
                key = (dataset, seed, spec['variant'])
                if key in duplicates:
                    raise ValueError(f'Duplicate run identity {key}; supply narrower roots')
                if key in runs:
                    previous = runs.pop(key)
                    duplicates.add(key)
                    raise ValueError(f'Duplicate run identity {key}: {previous["path"]} and {directory}; both excluded; supply narrower roots')
                runs[key] = {'path': directory, 'spec': spec, 'metrics': metrics}
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.append(f'{spec_path}: {error}')
    return runs, issues


def inspect_run(key, run, issues):
    dataset, seed, variant = key
    directory, spec, metrics = run['path'], run['spec'], run['metrics']
    row = {'dataset': dataset, 'seed': seed, 'variant': variant, 'path': str(directory),
           'fingerprint': spec.get('fingerprint'), 'source_hash': spec.get('source_hash'),
           'clip_weight_sha256': metrics.get('clip_weight_sha256'),
           'metrics_status': metrics.get('status'), 'summary': metrics.get('summary'), 'missing': []}
    for filename in ('metrics.json', 'config.yaml', 'support.json', 'run_state.json'):
        path = directory / filename
        if not path.is_file():
            row['missing'].append(filename)
    if metrics.get('status') != 'completed':
        issues.append(f'{dataset}/{variant}/seed_{seed}: metrics status is not completed')
    if metrics and metrics.get('seed') != seed:
        issues.append(f'{dataset}/{variant}/seed_{seed}: metrics seed differs from run_spec')
    state_path = directory / 'run_state.json'
    if state_path.is_file():
        state = read_json(state_path)
        if state.get('status') != 'completed' or not spec.get('fingerprint') or state.get('fingerprint') != spec['fingerprint']:
            issues.append(f'{dataset}/{variant}/seed_{seed}: run_state is not completed or its fingerprint differs from run_spec')
    row['support_sha256'] = digest(directory / 'support.json') if (directory / 'support.json').is_file() else None
    row['num_base'] = DATASETS[dataset]
    row['num_base_source'] = 'standard_dataset_default'
    if (directory / 'config.yaml').is_file():
        match = re.search(r'^\s*NUM_INIT_CLS:\s*(\d+)\s*$', (directory / 'config.yaml').read_text(encoding='utf-8'), re.M)
        if match:
            row.update(num_base=int(match.group(1)), num_base_source='config.yaml')
    session_ids = [int(s['session']) for s in metrics.get('sessions', [])]
    if not session_ids:
        issues.append(f'{dataset}/{variant}/seed_{seed}: no completed session rows in metrics.json')
    run['sessions'] = session_ids
    for session in session_ids:
        filename = f'predictions_session_{session:02d}.npz'
        if not (directory / filename).is_file():
            row['missing'].append(filename)
        else:
            predictions = load_predictions(directory / filename)
            accuracy = float(100 * (predictions['target'] == predictions['prediction']).mean())
            saved = next(s for s in metrics['sessions'] if int(s['session']) == session).get('accuracy')
            if saved is None or not np.isfinite(float(saved)) or abs(accuracy - float(saved)) > 1e-6:
                issues.append(f'{dataset}/{variant}/seed_{seed}: {filename} accuracy {accuracy} disagrees with metrics session accuracy {saved}')
    if variant != 'G1' and variant != 'G0':
        calibration = directory / 'guidance_calibration.json'
        if calibration.is_file():
            row['calibration'] = calibration_summary(read_json(calibration))
        else:
            row['missing'].append(calibration.name)
        for session in session_ids:
            filename = f'guidance_session_{session:02d}.json'
            if not (directory / filename).is_file():
                row['missing'].append(filename)
        row['relations'] = relation_summary(sorted(directory.glob('guidance_session_*.json')))
    if row['missing']:
        issues.append(f'{dataset}/{variant}/seed_{seed}: missing ' + ', '.join(row['missing']))
    return row


def analyze(roots, cases, variants, label=None):
    runs, issues = discover(roots, cases, variants, label)
    if not runs:
        issues.append('No selected run_spec.json found. Summary JSON alone is insufficient; use original per-run result directories.')
    for dataset, seed in sorted(cases or {(k[0], k[1]) for k in runs}):
        for variant in variants:
            if (dataset, seed, variant) not in runs:
                issues.append(f'Missing selected run: {dataset}/{variant}/seed_{seed}')
    details = {}
    for key, run in sorted(runs.items()):
        try:
            details[key] = inspect_run(key, run, issues)
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f'{run["path"]}: {error}')
    comparisons = []
    pairs = [('G1', variant) for variant in variants if variant != 'G1']
    if 'G3' in variants and 'G4' in variants:
        pairs.append(('G3', 'G4'))
    for dataset, seed in sorted({(k[0], k[1]) for k in runs}):
        for reference, candidate in pairs:
            ka, kb = (dataset, seed, reference), (dataset, seed, candidate)
            if ka not in details or kb not in details:
                continue
            a, b = details[ka], details[kb]
            support_match = a['support_sha256'] == b['support_sha256'] if a['support_sha256'] and b['support_sha256'] else None
            clip_match = a['clip_weight_sha256'] == b['clip_weight_sha256'] if a['clip_weight_sha256'] and b['clip_weight_sha256'] else None
            protocol_a, protocol_b = (v.get('calibration', {}).get('protocol_sha256') for v in (a, b))
            protocol_match = protocol_a == protocol_b if protocol_a and protocol_b else None
            if support_match is False or clip_match is False or protocol_match is False or a['num_base'] != b['num_base']:
                issues.append(f'{dataset}/seed_{seed} {candidate} vs {reference}: support/CLIP/base-count/calibration-protocol mismatch; refusing paired analysis')
                continue
            sessions_a, sessions_b = set(runs[ka]['sessions']), set(runs[kb]['sessions'])
            if sessions_a != sessions_b:
                issues.append(f'{dataset}/seed_{seed} {candidate} vs {reference}: different completed session sets')
            for session in sorted(sessions_a & sessions_b):
                name = f'predictions_session_{session:02d}.npz'
                pa, pb = runs[ka]['path'] / name, runs[kb]['path'] / name
                if not pa.is_file() or not pb.is_file():
                    continue
                try:
                    row = {'dataset': dataset, 'seed': seed, 'reference': reference,
                           'candidate': candidate, 'session': session, 'support_match': support_match,
                           'clip_match': clip_match, 'calibration_protocol_match': protocol_match,
                           **compare_predictions(pa, pb, a['num_base'])}
                    if reference == 'G3' and candidate == 'G4':
                        name = f'guidance_session_{session:02d}.json'
                        if (runs[ka]['path'] / name).is_file() and (runs[kb]['path'] / name).is_file():
                            row['prototype_comparison'] = prototype_comparison(runs[ka]['path'] / name, runs[kb]['path'] / name)
                    comparisons.append(row)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    issues.append(f'{dataset}/seed_{seed} {candidate} vs {reference} session {session}: {error}')
    report = {'schema_version': 1, 'status': 'complete' if not issues else 'incomplete',
              'roots': [str(p) for p in roots], 'selected_cases': [list(k) for k in sorted(cases)],
              'selected_variants': variants, 'runs': list(details.values()),
              'comparisons': comparisons, 'issues': issues,
              'limitations': [
                  'movement_degrees is RAW to guided, not G1 to G4.',
                  'Guided G3 to G4 angles use matching class IDs; G1 has no guidance JSON, so its exact prototype shift is unavailable here.',
                  'No checkpoint/pickle is loaded. Equal accuracy does not imply equal predictions.',
                  'These diagnostics do not establish causality or select hyperparameters from test results.']}
    return report, runs


def write_reports(output, report):
    output.mkdir(parents=True, exist_ok=True)
    (output / 'diagnostic.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    rows = []
    for pair in report['comparisons']:
        for group in ('all', 'base', 'novel'):
            rows.append({key: pair[key] for key in ('dataset', 'seed', 'reference', 'candidate', 'session')} |
                        {'group': group, **pair[group]})
    with (output / 'paired_predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        columns = ['dataset', 'seed', 'reference', 'candidate', 'session', 'group', 'n', 'changed',
                   'corrected', 'damaged', 'reference_accuracy', 'candidate_accuracy', 'delta_accuracy_pp']
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Frequency guidance offline diagnosis', '', f'Status: **{report["status"]}**. Runs: {len(report["runs"])}.', '',
             'All accuracy differences are percentage points. No model is executed.', '',
             '| Dataset | Seed | Comparison | Last available session | Changed | Corrected | Damaged | Delta accuracy |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    latest = {}
    for pair in report['comparisons']:
        key = pair['dataset'], pair['seed'], pair['reference'], pair['candidate']
        if key not in latest or pair['session'] > latest[key]['session']:
            latest[key] = pair
    for pair in latest.values():
        total = pair['all']
        lines.append(f'| {pair["dataset"]} | {pair["seed"]} | {pair["candidate"]}−{pair["reference"]} | {pair["session"]} | '
                     f'{total["changed"]} | {total["corrected"]} | {total["damaged"]} | {total["delta_accuracy_pp"]:+.6f} |')
    lines += ['', '## Issues', ''] + (['- ' + value for value in report['issues']] or ['None.'])
    lines += ['', '## Interpretation limits', ''] + ['- ' + value for value in report['limitations']]
    lines += ['', 'Calibration candidate deltas and relation summaries are in diagnostic.json; all sessions and Base/Novel groups are in paired_predictions.csv.',
              'If incomplete, provide the original selected run folders. Do not rerun or change parameters merely to fill a report.', '']
    (output / 'diagnosis.md').write_text('\n'.join(lines), encoding='utf-8')


def make_archive(path, runs, roots, output):
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for (dataset, seed, variant), run in sorted(runs.items()):
        for source in sorted(run['path'].iterdir()):
            if source.is_file() and not source.is_symlink() and allowlisted(source):
                entries.append((source, f'raw/{dataset}/{variant}/seed_{seed}/{source.name}'))
    for index, root in enumerate(roots):
        if root.is_dir():
            for source in sorted(root.glob('summary*.json')):
                if source.is_file() and not source.is_symlink() and re.fullmatch(r'summary(?:_[a-zA-Z0-9_-]+)?\.json', source.name):
                    entries.append((source, f'summaries/root_{index}/{source.name}'))
    for name in ('diagnostic.json', 'diagnosis.md', 'paired_predictions.csv'):
        entries.append((output / name, 'reports/' + name))
    with tarfile.open(path, 'w:gz') as archive:
        for source, name in entries:
            archive.add(source, arcname=name, recursive=False)
    return len(entries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True, help='Original result root; repeat for multiple datasets.')
    parser.add_argument('--case', action='append', default=[], help='dataset:seed, e.g. cub200:1; repeat to select cases.')
    parser.add_argument('--dataset', choices=list(DATASETS), help='Only needed when dataset identity is absent; must match saved metadata.')
    parser.add_argument('--variants', nargs='+', default=['G1', 'G3', 'G4'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--archive', help='Optional .tar.gz of selected raw JSON/NPZ/config/log files and reports; excludes checkpoints/images.')
    args = parser.parse_args(argv)
    try:
        if len(set(args.variants)) != len(args.variants) or any(not re.fullmatch(r'G[0-4](?:_[A-Za-z0-9_]+)?', v) for v in args.variants):
            raise ValueError('Variants must be distinct guidance variant names such as G1, G3, G4')
        if args.archive and not args.archive.endswith('.tar.gz'):
            raise ValueError('--archive must end with .tar.gz')
        cases = set()
        for value in args.case:
            name, seed = value.rsplit(':', 1)
            if int(seed) < 1:
                raise ValueError('Case seeds must be positive')
            cases.add((dataset_name(name), int(seed)))
        roots, output = [Path(p).resolve() for p in args.root], Path(args.output).resolve()
        report, runs = analyze(roots, cases, args.variants, args.dataset)
        write_reports(output, report)
        print(f'{report["status"]}: {len(report["runs"])} runs, {len(report["comparisons"])} paired session comparisons; {output / "diagnosis.md"}')
        if args.archive:
            archive = Path(args.archive).resolve()
            count = make_archive(archive, runs, roots, output)
            print(f'Archive: {archive} ({count} files; no checkpoints or images)')
        for issue in report['issues'][:12]:
            print('Issue:', issue)
        return 0 if report['status'] == 'complete' else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
