import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fgdr_synth import write_cache
from tools import fgdr_run


def _write_keywords(stem, num_classes=30, dim=32):
    rng = np.random.RandomState(0)
    pair_class, pair_kw, masks, class_kw = [], [], [], {}
    for c in range(num_classes):
        kws = sorted(rng.choice(10, 3, replace=False).tolist())
        class_kw[str(c)] = kws
        for k in kws:
            m = np.zeros(dim, bool); m[rng.choice(dim, 6, replace=False)] = True
            pair_class.append(c); pair_kw.append(k); masks.append(m)
    masks = np.stack(masks)
    np.savez(str(stem) + '.npz', pair_class=np.array(pair_class), pair_keyword=np.array(pair_kw),
             mask_final=masks, mask_fdr=masks[::-1].copy())
    Path(str(stem) + '.json').write_text(json.dumps({'vocab': [f'k{i}' for i in range(10)],
                                                     'class_keywords': class_kw}))


def test_end_to_end_cli(tmp_path, capsys):
    stem = tmp_path / 'cache' / 'syn'
    write_cache(stem)
    kw = tmp_path / 'cache' / 'kw'
    _write_keywords(kw)
    pseudo = ['--n-fit', '5', '--way', '2', '--n-sessions', '2', '--query-per-class', '3']
    assert fgdr_run.main(['parity', '--cache', str(stem), '--out', str(tmp_path / 'parity'),
                          '--seeds', '1', '2']) == 0
    parity = json.loads((tmp_path / 'parity' / 'results.json').read_text())
    assert set(parity['table']) == {'BiMC', 'BiMC_dagger'}
    config = {'reference': 'BiMC_dagger', 'methods': [
        {'name': 'K0', 'calibration': {'type': 'holistic', 'weight_source': 'text'},
         'grid': {'calibration.lambda': [0.0, 0.2]}},
        {'name': 'K1', 'calibration': {'type': 'segments', 'weight_source': 'text', 'lambda': 0.1,
                                       'segments': {'source': 'keyword', 'topr': 3}},
         'grid': {'calibration.segments.alpha': [0.0, 0.3]}},
        {'name': 'Q1', 'calibration': {'type': 'segments', 'weight_source': 'text',
                                       'segments': {'source': 'frequency', 'q': 6, 'topr': 3}},
         'grid': {'calibration.segments.alpha': [0.0, 0.3]}},
        {'name': 'A1', 'anchor': {'enabled': True}, 'grid': {'anchor.alpha': [0.0, 0.2]}},
        {'name': 'F1', 'anchor': {'enabled': True, 'alpha': 0.2,
                                  'blocks': [{'kind': 'dct', 'layer': 0, 'band': 2, 'dim': 4}]}}]}
    cfg_path = tmp_path / 'cfg.json'
    cfg_path.write_text(json.dumps(config))
    dev = tmp_path / 'dev'
    assert fgdr_run.main(['dev', '--cache', str(stem), '--config', str(cfg_path), '--out', str(dev),
                          '--keywords', str(kw), '--split-seeds', '2', '--sequences', '2',
                          '--max-base-drop', '100', '--max-novel-drop', '100', *pseudo]) == 0
    draft = json.loads((dev / 'selection_draft.json').read_text())
    assert set(draft['families']) == {'K0', 'K1', 'Q1', 'A1', 'F1'}
    lock = tmp_path / 'selection.json'
    base_cfg = ROOT / 'fgdr' / 'configs' / 'baselines.json'
    assert fgdr_run.main(['lock', '--draft', str(dev / 'selection_draft.json'), '--families', 'K0', 'K1', 'Q1',
                          'A1', 'F1', '--out', str(lock), '--baselines-config', str(base_cfg)]) == 0
    final = tmp_path / 'final'
    assert fgdr_run.main(['final', '--cache', str(stem), '--lock', str(lock), '--out', str(final),
                          '--keywords', str(kw), '--seeds', '1', '2']) == 0
    result = json.loads((final / 'results.json').read_text())
    assert result['reference'] == 'BiMC_dagger'
    assert {'K0', 'K1', 'Q1', 'A1', 'F1', 'FDR_tf', 'GDA_only'} <= set(result['table'])
    assert (final / 'table.md').exists()
    assert fgdr_run.main(['compare', '--results', str(final / 'results.json'), '--a', 'BiMC_dagger',
                          '--b', 'A1']) == 0
    assert fgdr_run.main(['diagnose-masks', '--cache', str(stem), '--out', str(tmp_path / 'dm'), '--q', '6',
                          '--shots', '1', '5', '--draws', '2', '--keywords', str(kw)]) == 0
    report = json.loads((tmp_path / 'dm' / 'mask_diagnostics.json').read_text())
    assert 'blur' in report and 'gate_G1prime_pass' in report
    assert fgdr_run.main(['diagnose-views', '--cache', str(stem), '--out', str(tmp_path / 'dv'),
                          '--split-seeds', '2', '--sequences', '1', *pseudo]) == 0
    views = json.loads((tmp_path / 'dv' / 'view_diagnostics.json').read_text())
    assert 'blur0' in views and 'aug0' in views


def test_final_refuses_unlocked(tmp_path):
    stem = tmp_path / 'syn'
    write_cache(stem)
    fake = tmp_path / 'fake.json'
    fake.write_text(json.dumps({'status': 'locked', 'selected': {}}))
    try:
        fgdr_run.main(['final', '--cache', str(stem), '--lock', str(fake), '--out', str(tmp_path / 'f')])
    except ValueError:
        return
    raise AssertionError('final must refuse a lock without a valid checksum')


def test_prototype_diagnostics(tmp_path):
    stem = tmp_path / 'syn'
    write_cache(stem)
    meta = json.loads((tmp_path / 'syn.json').read_text())
    from fgdr import cache as fc
    lock = {'status': 'locked', 'cache_meta_sha256': meta['meta_sha256'],
            'selected': {'methods': [{'name': 'K0', 'calibration': {'type': 'holistic', 'weight_source': 'text',
                                                                     'lambda': 0.3}}]}}
    lock['lock_sha256'] = fc.canonical_sha(lock)
    path = tmp_path / 'lock.json'
    path.write_text(json.dumps(lock))
    assert fgdr_run.main(['diagnose-prototypes', '--cache', str(stem), '--lock', str(path),
                          '--out', str(tmp_path / 'dp'), '--seeds', '1', '2']) == 0
    report = json.loads((tmp_path / 'dp' / 'prototype_diagnostics.json').read_text())
    assert {'BiMC', 'BiMC_dagger', 'K0'} <= set(report)
