"""Create an authenticated, compact analysis zip from a base-only study.

Only named result artifacts and optional source snapshots are included. Image
feature caches, weights, checkpoints, arbitrary files and symlinks are excluded.
Packaging never imports numpy/torch, loads arrays, or runs experiments.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import zipfile

SCHEMA_VERSION = 1
SUCCESS = {'completed', 'skipped'}
ROOT_REQUIRED = ('summary.json', 'input_provenance.json', 'plan.json')
RUN_REQUIRED = ('config.json', 'run_spec.json', 'run_state.json', 'calibration.json',
                'protocol.json', 'metrics.json', 'stage_metrics.json',
                'predictions.npz', 'diagnostics.json')
JSON_NAMES = {
    *ROOT_REQUIRED, *RUN_REQUIRED, 'group_diagnostics.json', 'layer_selection.json',
    'cache_metadata.json', 'attribute_manifest.json', 'attribute_coverage.json',
    'source_hashes.json', 'diagnostics.json', 'selection.json', 'plan.json',
}
TEXT_NAMES = {'analysis.md', 'analysis_zh.md', 'results.csv', 'summary.csv',
              'stage_metrics.csv', 'run.log', 'config.yaml'}
BLOCKED_PARTS = {'.git', '.aws', '.ssh', 'secrets', 'secret', 'private',
                 '__pycache__', 'weights', 'checkpoints', 'cache', 'caches'}


def _read_json(path):
    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        return json.load(handle, parse_constant=invalid)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _relative_directory(root, supplied):
    if not isinstance(supplied, str) or not supplied:
        raise ValueError('A run is missing its relative output_dir.')
    name = supplied.replace('\\', '/')
    relative = PurePosixPath(name)
    if relative.is_absolute() or PureWindowsPath(supplied).drive or '..' in relative.parts:
        raise ValueError('Run output_dir must stay within the study root: ' + supplied)
    target = root.joinpath(*relative.parts)
    cursor = root
    for part in relative.parts:
        cursor = cursor/part
        if cursor.is_symlink():
            raise ValueError('Run output_dir traverses a symlink: ' + supplied)
    resolved = target.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError('Run output_dir escapes the study root: ' + supplied)
    return target


def _allowed(relative):
    parts = relative.parts
    lowered = [part.lower() for part in parts]
    if any(part in BLOCKED_PARTS or part.startswith('.') for part in lowered):
        return False
    name = lowered[-1]
    if ('.local.' in name or 'api_key' in name or 'credential' in name
            or 'password' in name or name.endswith(('.pt', '.pth', '.ckpt'))):
        return False
    if name in JSON_NAMES or name in TEXT_NAMES:
        return True
    if name.endswith('.npz'):
        return name.startswith(('predictions', 'stage_predictions', 'diagnostics'))
    if parts[0] == 'source_snapshot':
        return relative.suffix.lower() in ('.py', '.yaml', '.yml', '.json', '.md', '.txt')
    return False


def _inventory(root):
    included, excluded = [], []
    for directory, folders, names in os.walk(root, followlinks=False):
        base = Path(directory)
        retained = []
        for name in sorted(folders):
            path = base/name
            rel = path.relative_to(root)
            if path.is_symlink():
                excluded.append({'path': rel.as_posix(), 'reason': 'symlink_directory'})
            elif name.lower() in BLOCKED_PARTS or name.startswith('.'):
                excluded.append({'path': rel.as_posix(), 'reason': 'excluded_directory'})
            else:
                retained.append(name)
        folders[:] = retained
        for name in sorted(names):
            path = base/name
            relative = path.relative_to(root)
            if path.is_symlink():
                excluded.append({'path': relative.as_posix(), 'reason': 'symlink_file'})
            elif _allowed(relative) and path.is_file():
                included.append(path)
            else:
                excluded.append({'path': relative.as_posix(), 'reason': 'not_analysis_allowlist'})
    return sorted(included, key=lambda p: p.relative_to(root).as_posix()), excluded


def _audit(root, summary, files):
    errors, missing = [], []
    paths = {path.relative_to(root).as_posix() for path in files}
    for name in ROOT_REQUIRED:
        if name not in paths:
            missing.append(name)
        else:
            try:
                if not isinstance(_read_json(root/name), dict):
                    errors.append('Root metadata must be a JSON object: ' + name)
            except (OSError, ValueError, TypeError) as error:
                errors.append('Unreadable root metadata ' + name + ': ' + str(error))
    if summary.get('data_source') != 'base_training_only':
        errors.append('Summary does not certify base_training_only data.')
    if summary.get('benchmark_evaluation_performed') is not False:
        errors.append('Summary does not certify that benchmark evaluation was absent.')
    rows = summary.get('runs')
    if not isinstance(rows, list) or not rows:
        errors.append('Summary has no run records.')
        rows = []
    expected = summary.get('expected_runs')
    expected_by_identity = {}
    if not isinstance(expected, list) or not expected:
        errors.append('Summary is missing its explicit expected_runs plan.')
        expected = []
    for row in expected:
        if not isinstance(row, dict):
            errors.append('An expected run is not an object.')
            continue
        identity = (row.get('fit_seed'), row.get('stage'), row.get('variant'))
        if identity in expected_by_identity:
            errors.append('Duplicate expected run identity: ' + repr(identity))
        expected_by_identity[identity] = row
        try:
            _relative_directory(root, row.get('output_dir'))
        except ValueError as error:
            errors.append(str(error))
    if summary.get('planned_runs') != len(expected):
        errors.append('Summary planned_runs differs from its explicit plan.')
    if (root/'plan.json').is_file() and not (root/'plan.json').is_symlink():
        try:
            plan = _read_json(root/'plan.json')
            if plan.get('runs') != expected:
                errors.append('plan.json differs from summary.expected_runs.')
            if (summary.get('family_fingerprint') is not None
                    and plan.get('family_fingerprint') != summary['family_fingerprint']):
                errors.append('Study plan family fingerprint differs from summary.')
        except (OSError, ValueError, TypeError, AttributeError) as error:
            errors.append('Unreadable study plan: ' + str(error))
    selection_path = root/'layer_selection.json'
    if selection_path.is_file() and not selection_path.is_symlink() and 'layer_selection' in summary:
        try:
            if _read_json(selection_path) != summary['layer_selection']:
                errors.append('layer_selection.json differs from the study summary.')
        except (OSError, ValueError, TypeError) as error:
            errors.append('Unreadable layer selection: ' + str(error))
    identities = set()
    completed, failed = 0, 0
    for row in rows:
        if not isinstance(row, dict):
            errors.append('A run record is not an object.')
            continue
        identity = (row.get('fit_seed'), row.get('stage'), row.get('variant'))
        if identity in identities:
            errors.append('Duplicate run identity: ' + repr(identity))
        identities.add(identity)
        if identity not in expected_by_identity:
            errors.append('Unplanned run record: ' + repr(identity))
        elif row.get('output_dir') != expected_by_identity[identity].get('output_dir'):
            errors.append('Run output_dir differs from its plan: ' + repr(identity))
        status = row.get('status')
        if status in SUCCESS:
            completed += 1
        else:
            failed += 1
            errors.append('Incomplete run ' + repr(identity) + ': ' + str(status))
        try:
            folder = _relative_directory(root, row.get('output_dir'))
        except ValueError as error:
            errors.append(str(error))
            continue
        prefix = folder.relative_to(root).as_posix()
        # Even a failed run may have useful partial files; report every missing
        # expected analysis artifact rather than silently certifying completion.
        required = list(RUN_REQUIRED)
        spec_file = folder/'run_spec.json'
        if spec_file.is_file() and not spec_file.is_symlink():
            try:
                definition = _read_json(spec_file).get('definition', {})
                if definition.get('kind') in ('typed', 'legacy'):
                    required.append('diagnostics.npz')
                if definition.get('text') == 'attribute_groups':
                    required.append('group_diagnostics.json')
            except (OSError, ValueError, TypeError, AttributeError):
                pass  # The required metadata validation below reports the error.
        for name in required:
            relative = prefix + '/' + name
            if relative not in paths:
                missing.append(relative)
            elif name.endswith('.json'):
                try:
                    if not isinstance(_read_json(folder/name), dict):
                        errors.append('Run metadata must be a JSON object: ' + relative)
                except (OSError, ValueError, TypeError) as error:
                    errors.append('Unreadable run metadata ' + relative + ': ' + str(error))
        for name in ('run_spec.json', 'run_state.json'):
            path = folder/name
            if path.is_file() and not path.is_symlink():
                try:
                    content = _read_json(path)
                    if ('fingerprint' in content and row.get('fingerprint') is not None
                            and content['fingerprint'] != row['fingerprint']):
                        errors.append('Run fingerprint mismatch: ' + prefix + '/' + name)
                    if name == 'run_state.json' and status in SUCCESS and content.get('status') not in SUCCESS:
                        errors.append('Run state is not complete: ' + prefix)
                except (OSError, ValueError, TypeError) as error:
                    errors.append('Unreadable run metadata ' + prefix + '/' + name + ': ' + str(error))
        protocol_file = folder/'protocol.json'
        if row.get('protocol_sha256') and protocol_file.is_file() and not protocol_file.is_symlink():
            try:
                protocol = _read_json(protocol_file)
                payload = {key: value for key, value in protocol.items() if key != 'sha256'}
                actual = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                                   allow_nan=False).encode('utf-8')).hexdigest()
                if actual != protocol.get('sha256') or actual != row['protocol_sha256']:
                    errors.append('Selection protocol checksum mismatch: ' + prefix)
            except (OSError, ValueError, TypeError, AttributeError) as error:
                errors.append('Unreadable selection protocol ' + prefix + ': ' + str(error))
    if summary.get('completed_runs') != completed:
        errors.append('Summary completed_runs differs from its run records.')
    if summary.get('failed_runs', 0) != failed:
        errors.append('Summary failed_runs differs from its run records.')
    pending = set(expected_by_identity)-identities
    if pending:
        errors.append('Unstarted planned configurations: ' + str(len(pending)))
    if summary.get('pending_runs') != len(pending):
        errors.append('Summary pending_runs differs from its explicit plan.')
    study_complete = completed == len(expected_by_identity) and failed == 0 and not pending
    if summary.get('study_complete') is not study_complete:
        errors.append('Summary study_complete does not match the full expected plan.')
    return {'complete': not errors and not missing,
            'errors': errors, 'missing_artifacts': sorted(set(missing)),
            'expected_run_records': len(expected_by_identity),
            'pending_run_records': len(pending),
            'unstarted_runs': [expected_by_identity[key] for key in sorted(pending)],
            'run_records': len(rows), 'completed_run_records': completed,
            'incomplete_run_records': failed}


def build_manifest(root, *, allow_incomplete=False):
    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError('Study root cannot be a symlink.')
    root = supplied.resolve()
    if not root.is_dir():
        raise ValueError('Study root does not exist: ' + str(root))
    summary_path = root/'summary.json'
    if not summary_path.is_file() or summary_path.is_symlink():
        raise ValueError('A regular summary.json is required to identify study results.')
    summary = _read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError('Study summary must be a JSON object.')
    files, excluded = _inventory(root)
    audit = _audit(root, summary, files)
    if not audit['complete'] and not allow_incomplete:
        detail = '; '.join(audit['errors'] + audit['missing_artifacts'][:10])
        raise ValueError('Incomplete analysis results; use --allow-incomplete to mark a partial bundle. ' + detail)
    manifest = {'schema_version': SCHEMA_VERSION,
                'format': 'multilayer_completion_analysis_bundle_v1',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'method': summary.get('method'), 'data_source': summary.get('data_source'),
                'requested_stages': summary.get('requested_stages'),
                'fit_seeds': summary.get('fit_seeds'),
                'planned_runs': summary.get('planned_runs'),
                'complete': audit['complete'], 'allow_incomplete': bool(allow_incomplete),
                'audit': audit, 'files': [
                    {'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                     'sha256': _sha256(path)} for path in files],
                'excluded': excluded,
                'contents_note': 'Configuration, selection protocol, metrics, compact predictions, '
                                 'diagnostics and provenance only. Full feature caches and model weights are excluded.'}
    return manifest


def package_results(root, output, *, allow_incomplete=False):
    """Write a new zip, refusing stale, incomplete or escaping artifact paths."""
    supplied_output = Path(output)
    if supplied_output.is_symlink():
        raise ValueError('Analysis bundle output cannot be a symlink.')
    output = supplied_output.resolve()
    if output.suffix.lower() != '.zip':
        raise ValueError('Analysis bundle output must end in .zip.')
    if output.exists():
        raise FileExistsError('Refusing to overwrite an existing analysis bundle: ' + str(output))
    manifest = build_manifest(root, allow_incomplete=allow_incomplete)
    root = Path(root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open('xb') as handle:
            created = True
            with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for item in manifest['files']:
                    source = root.joinpath(*PurePosixPath(item['path']).parts)
                    if (source.is_symlink() or root not in source.resolve().parents
                            or _sha256(source) != item['sha256']):
                        raise ValueError('Analysis artifact changed while packaging: ' + item['path'])
                    archive.write(source, arcname=item['path'])
                archive.writestr('bundle_manifest.json', json.dumps(
                    manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        return manifest
    except BaseException:
        if created and output.is_file():
            output.unlink()
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True, help='New .zip path; never overwritten.')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Include partial results with explicit missing/error inventory.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        manifest = package_results(args.root, args.output, allow_incomplete=args.allow_incomplete)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('Cannot package analysis results: ' + str(error), file=sys.stderr)
        return 2
    print('Analysis bundle:', str(Path(args.output).resolve()))
    print('Complete:', manifest['complete'], 'files:', len(manifest['files']),
          'uncompressed bytes:', sum(item['bytes'] for item in manifest['files']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
