"""Package base-only attribute transfer results and verify both cache identities.

This standard-library-only utility reads result files, never executes source
snapshots, never loads arrays, and excludes full feature caches and weights.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import package_multilayer_completion_results as common

ROOT_REQUIRED = (*common.ROOT_REQUIRED, 'cache_metadata.json', 'grounding_metadata.json',
                 'grounding_manifest.json', 'source_hashes.json')
RUN_REQUIRED = (*common.RUN_REQUIRED, 'teacher_diagnostics.json', 'offset_fit.json')
ADDITIONAL_JSON = {'grounding_metadata.json', 'grounding_manifest.json',
                   'teacher_diagnostics.json', 'offset_fit.json', 'inner_protocol.json'}


def _allowed(relative):
    if relative.name in ADDITIONAL_JSON:
        # Reuse common sensitive-name and parent-directory checks by replacing
        # only the allowed basename with another known analysis artifact.
        return common._allowed(relative.with_name('diagnostics.json'))
    return common._allowed(relative)


def _inventory(root):
    included, excluded = [], []
    for directory, folders, names in os.walk(root, followlinks=False):
        base = Path(directory)
        retained = []
        for name in sorted(folders):
            path = base/name
            relative = path.relative_to(root)
            if path.is_symlink():
                excluded.append({'path':relative.as_posix(), 'reason':'symlink_directory'})
            elif name.startswith('.') or name.lower() in common.BLOCKED_PARTS:
                excluded.append({'path':relative.as_posix(), 'reason':'excluded_directory'})
            else:
                retained.append(name)
        folders[:] = retained
        for name in sorted(names):
            path = base/name
            relative = path.relative_to(root)
            if path.is_symlink():
                excluded.append({'path':relative.as_posix(), 'reason':'symlink_file'})
            elif _allowed(relative) and path.is_file():
                included.append(path)
            else:
                excluded.append({'path':relative.as_posix(), 'reason':'not_analysis_allowlist'})
    return sorted(included, key=lambda p:p.relative_to(root).as_posix()), excluded


def _audit(root, summary, files):
    audit = common._audit(root, summary, files)
    errors, missing = audit['errors'], audit['missing_artifacts']
    paths = {path.relative_to(root).as_posix() for path in files}
    documents = {}
    for name in ROOT_REQUIRED:
        if name not in paths:
            missing.append(name)
            continue
        try:
            document = common._read_json(root/name)
            if not isinstance(document, dict) and name != 'grounding_manifest.json':
                errors.append('Root metadata must be an object: '+name)
            documents[name] = document
        except (OSError, ValueError, TypeError) as error:
            errors.append('Unreadable root metadata '+name+': '+str(error))
    provenance = documents.get('input_provenance.json', {})
    cache = documents.get('cache_metadata.json', {})
    grounding = documents.get('grounding_metadata.json', {})
    if not isinstance(provenance, dict): provenance = {}
    if not isinstance(cache, dict): cache = {}
    if not isinstance(grounding, dict): grounding = {}
    family = summary.get('family_fingerprint')
    if not isinstance(family, str) or not family:
        errors.append('Summary requires a family_fingerprint.')
    if provenance.get('family_fingerprint') != family:
        errors.append('Input provenance family fingerprint differs from summary.')
    visual_sha = provenance.get('cache_sha256')
    grounding_sha = provenance.get('grounding_sha256')
    if not isinstance(visual_sha, str) or not visual_sha or cache.get('cache_sha256') != visual_sha:
        errors.append('Visual cache identity differs between metadata and provenance.')
    if (not isinstance(grounding_sha, str) or not grounding_sha
            or grounding.get('grounding_sha256') != grounding_sha):
        errors.append('Grounding cache identity differs between metadata and provenance.')
    if grounding.get('source_cache_sha256') != visual_sha:
        errors.append('Grounding sidecar was not derived from the recorded visual cache.')
    metadata_sha = cache.get('metadata_payload_sha256')
    if (not isinstance(metadata_sha,str) or not metadata_sha
            or grounding.get('source_metadata_sha256') != metadata_sha):
        errors.append('Grounding source metadata identity differs from the visual cache.')
    classes = cache.get('base_class_ids')
    if (not isinstance(classes,list) or not classes or any(not isinstance(c,int) or isinstance(c,bool) for c in classes)
            or len(set(classes))!=len(classes) or grounding.get('base_class_ids') != classes):
        errors.append('Grounding class identities differ from the visual cache.')
    if (grounding.get('clip_weight_sha256') != cache.get('clip_weight_sha256')
            or not cache.get('clip_weight_sha256')):
        errors.append('Grounding and visual caches do not record the same CLIP weights.')
    if ('grounding_manifest' in grounding and 'grounding_manifest.json' in documents
            and grounding['grounding_manifest'] != documents['grounding_manifest.json']):
        errors.append('Grounding manifest differs from the recorded sidecar metadata.')
    sources = documents.get('source_hashes.json', {})
    if not isinstance(sources,dict) or not sources:
        errors.append('Source snapshot provenance must contain source hashes.')
    if isinstance(sources, dict):
        for name, digest in sources.items():
            # The declared snapshot is resolved as a bounded study-relative
            # path, never as an absolute path supplied by metadata.
            try:
                source = common._relative_directory(root, 'source_snapshot/'+name)
                relative = source.relative_to(root).as_posix()
                if relative not in paths:
                    missing.append(relative)
                elif not isinstance(digest, str) or common._sha256(source) != digest:
                    errors.append('Source snapshot checksum mismatch: '+relative)
            except (ValueError, OSError, TypeError) as error:
                errors.append('Invalid source snapshot entry: '+str(error))
    for row in summary.get('runs', []):
        if not isinstance(row, dict):
            continue
        try:
            folder = common._relative_directory(root, row.get('output_dir'))
        except ValueError:
            continue  # The shared protocol audit has already reported this.
        prefix = folder.relative_to(root).as_posix()
        for name in RUN_REQUIRED:
            relative = prefix+'/'+name
            if relative not in paths:
                missing.append(relative)
            elif name.endswith('.json'):
                try:
                    if not isinstance(common._read_json(folder/name), dict):
                        errors.append('Run metadata must be an object: '+relative)
                except (OSError, ValueError, TypeError) as error:
                    errors.append('Unreadable run metadata '+relative+': '+str(error))
        for name in ('run_spec.json', 'run_state.json'):
            relative = prefix+'/'+name
            if relative in paths:
                try:
                    content = common._read_json(folder/name)
                    if isinstance(content, dict) and (not row.get('fingerprint')
                            or content.get('fingerprint') != row['fingerprint']):
                        errors.append('Run fingerprint mismatch or absent: '+relative)
                    if name == 'run_spec.json' and isinstance(content, dict) and content.get('family_fingerprint') != family:
                        errors.append('Run family fingerprint mismatch: '+relative)
                except (OSError, ValueError, TypeError):
                    pass
    audit['missing_artifacts'] = sorted(set(missing))
    audit['errors'] = list(dict.fromkeys(errors))
    audit['complete'] = not audit['missing_artifacts'] and not audit['errors']
    return audit


def build_manifest(root, *, allow_incomplete=False):
    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError('Study root cannot be a symlink.')
    root = supplied.resolve()
    summary_path = root/'summary.json'
    if not root.is_dir() or not summary_path.is_file() or summary_path.is_symlink():
        raise ValueError('A regular summary.json in the study root is required.')
    summary = common._read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError('Study summary must be an object.')
    files, excluded = _inventory(root)
    audit = _audit(root, summary, files)
    if not audit['complete'] and not allow_incomplete:
        details = '; '.join(audit['errors']+audit['missing_artifacts'][:10])
        raise ValueError('Incomplete attribute transfer results; use --allow-incomplete for a partial bundle. '+details)
    return {'schema_version':1, 'format':'attribute_transfer_analysis_bundle_v1',
            'created_at':datetime.now(timezone.utc).isoformat(), 'method':summary.get('method'),
            'data_source':summary.get('data_source'), 'benchmark_evaluation_performed':False,
            'fit_seeds':summary.get('fit_seeds'), 'planned_runs':summary.get('planned_runs'),
            'complete':audit['complete'], 'allow_incomplete':bool(allow_incomplete), 'audit':audit,
            'files':[{'path':p.relative_to(root).as_posix(), 'bytes':p.stat().st_size,
                      'sha256':common._sha256(p)} for p in files], 'excluded':excluded,
            'contents_note':'Base-development candidate metrics, all-blend geometric diagnostics, '
                            'old/novel scores, offset fitting and provenance. Full feature caches '
                            'and CLIP weights are excluded. No arrays are loaded while packaging.'}


def package_results(root, output, *, allow_incomplete=False):
    supplied = Path(output)
    if supplied.is_symlink():
        raise ValueError('Analysis bundle output cannot be a symlink.')
    output = supplied.resolve()
    if output.suffix.lower() != '.zip':
        raise ValueError('Analysis bundle output must end in .zip.')
    if output.exists():
        raise FileExistsError('Refusing existing analysis bundle: '+str(output))
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
                            or common._sha256(source) != item['sha256']):
                        raise ValueError('Analysis artifact changed while packaging: '+item['path'])
                    archive.write(source, arcname=item['path'])
                archive.writestr('bundle_manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
        return manifest
    except BaseException:
        if created and output.is_file():
            output.unlink()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args(argv)
    try:
        manifest = package_results(args.root,args.output,allow_incomplete=args.allow_incomplete)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print('Cannot package attribute transfer results: '+str(error),file=sys.stderr)
        return 2
    print('Analysis bundle:',str(Path(args.output).resolve()))
    print('Complete:',manifest['complete'],'files:',len(manifest['files']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
