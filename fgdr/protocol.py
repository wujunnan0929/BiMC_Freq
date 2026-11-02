"""Benchmark episodes and base-only pseudo-incremental episodes.

An :class:`Episode` always uses contiguous labels ``0..C-1`` in session order,
with the first ``num_base`` labels being the (pseudo-)base classes. Methods never
see dataset class ids except through ``Episode.class_ids`` (used to look up
text-side resources such as keyword lists).
"""

from dataclasses import dataclass, field

import numpy as np

VIEW_KEYS = ('blur', 'aug', 'dct', 'dct_rand')

# Pseudo-incremental defaults per dataset (fit classes, way, sessions, query cap).
PSEUDO_DEFAULTS = {
    'cub200': dict(n_fit=50, way=10, n_sessions=5, query_per_class=None),
    'cifar100': dict(n_fit=30, way=5, n_sessions=6, query_per_class=100),
    'miniimagenet': dict(n_fit=30, way=5, n_sessions=6, query_per_class=100),
}


@dataclass
class Bank:
    feats: np.ndarray                 # [n, D] unit
    labels: np.ndarray                # [n] contiguous episode labels
    rows: np.ndarray                  # rows into the source split arrays
    split: str                        # 'train' or 'test'
    views: dict = field(default_factory=dict)

    def subset(self, index):
        return Bank(self.feats[index], self.labels[index], self.rows[index], self.split,
                    {k: v[index] for k, v in self.views.items()})


@dataclass
class Episode:
    kind: str
    num_base: int
    sessions: list                    # list[np.ndarray] contiguous ids per session
    support: list                     # list[Bank]
    query: Bank
    text: np.ndarray                  # [C, D]
    desc_proto: np.ndarray            # [C, D]
    desc: np.ndarray                  # [M, D]
    desc_targets: np.ndarray          # [M] contiguous
    class_ids: np.ndarray             # [C] dataset class id for each contiguous label
    info: dict = field(default_factory=dict)

    @property
    def num_classes(self):
        return int(sum(len(s) for s in self.sessions))


def make_sessions(num_classes, num_init, num_inc):
    sessions = [np.arange(0, num_init)]
    for start in range(num_init, num_classes, num_inc):
        sessions.append(np.arange(start, min(start + num_inc, num_classes)))
    return sessions


def replica_support_indices(train_targets, sessions, base_shot, inc_shot, seed):
    """Exact replica of DatasetManager support sampling (RandomState, per-class choice)."""
    targets = np.asarray(train_targets)
    rng = np.random.RandomState(seed)
    out = []
    for task_id, class_ids in enumerate(sessions):
        shot = base_shot if task_id == 0 else inc_shot
        selected = []
        for class_id in class_ids:
            candidates = np.flatnonzero(targets == class_id)
            if shot is not None and shot != -1 and shot <= len(candidates):
                candidates = rng.choice(candidates, size=shot, replace=False)
            selected.append(candidates)
        out.append(np.concatenate(selected).astype(np.int64))
    return out


def _views(arrays, rows, views):
    return {k: arrays[k][rows] for k in views if k in arrays}


def _row_lookup(sample_ids):
    order = np.argsort(sample_ids)
    return order, sample_ids[order]


def _rows_for(sample_ids_sorted, order, wanted):
    wanted = np.asarray(wanted, dtype=np.int64)
    pos = np.searchsorted(sample_ids_sorted, wanted)
    if np.any(pos >= len(sample_ids_sorted)) or np.any(sample_ids_sorted[pos] != wanted):
        raise ValueError('Support sample ids are missing from the training cache')
    return order[pos]


def _text_subset(train, class_ids):
    """Text arrays re-indexed to the given dataset class order (contiguous labels)."""
    class_ids = np.asarray(class_ids)
    text = train['text'][class_ids]
    desc_proto = train['desc_proto'][class_ids]
    position = {int(c): i for i, c in enumerate(class_ids)}
    keep = np.array([int(t) in position for t in train['desc_targets']])
    desc = train['desc'][keep]
    targets = np.array([position[int(t)] for t in train['desc_targets'][keep]], dtype=np.int64)
    return text, desc_proto, desc, targets


def support_key(shot, seed):
    return f'shot{int(shot)}_seed{int(seed)}'


def benchmark_episode(train, test, meta, shot, seed, views=VIEW_KEYS):
    """The real FSCIL sequence for one support draw. ``test`` may be None (no queries)."""
    sessions_ds = [np.asarray(s, dtype=np.int64) for s in meta['sessions']]
    order_ds = np.concatenate(sessions_ds)
    position = np.full(int(order_ds.max()) + 1, -1, dtype=np.int64)
    position[order_ds] = np.arange(len(order_ds))
    key = support_key(shot, seed)
    if key not in meta['supports']:
        raise KeyError(f'No cached support draw {key}; available: {sorted(meta["supports"])}')
    order, sorted_ids = _row_lookup(train['sample_id'])
    support = []
    start = 0
    sessions = []
    for ids, sample_ids in zip(sessions_ds, meta['supports'][key]):
        rows = _rows_for(sorted_ids, order, sample_ids)
        labels = position[train['labels'][rows]]
        if np.any(labels < 0):
            raise ValueError('Support rows contain classes outside the session layout')
        support.append(Bank(train['final'][rows], labels, rows, 'train', _views(train, rows, views)))
        sessions.append(np.arange(start, start + len(ids)))
        start += len(ids)
    if test is not None:
        qrows = np.arange(len(test['labels']))
        qlabels = position[test['labels']]
        keep = qlabels >= 0
        qrows = qrows[keep]
        query = Bank(test['final'][qrows], qlabels[keep], qrows, 'test', _views(test, qrows, views))
    else:
        d = train['final'].shape[1]
        query = Bank(np.zeros((0, d), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64), 'test')
    text, desc_proto, desc, targets = _text_subset(train, order_ds)
    return Episode('benchmark', len(sessions_ds[0]), sessions, support, query, text, desc_proto,
                   desc, targets, order_ds, {'shot': shot, 'seed': seed, 'dataset': meta.get('dataset')})


def pseudo_episode(train, meta, *, n_fit, way, n_sessions, shot, split_seed, sequence_seed,
                   fit_fraction=2. / 3., query_per_class=None, views=VIEW_KEYS):
    """Base-only pseudo FSCIL: A (pseudo-base) / V (pseudo-novel) split of base classes.

    Only training images of the original base classes are touched. The A/V
    split and A's fit/query partition depend only on ``split_seed``; the V
    support draws and session order depend on ``sequence_seed``.
    """
    base = np.asarray(meta['sessions'][0], dtype=np.int64)
    if n_fit + way * n_sessions > len(base):
        raise ValueError('Pseudo protocol needs more base classes than available')
    rng_split = np.random.RandomState(split_seed)
    perm = rng_split.permutation(base)
    fit_classes, novel_classes = perm[:n_fit], perm[n_fit:n_fit + way * n_sessions]
    labels_all = train['labels']
    fit_rows, query_rows = [], []
    for c in fit_classes:
        rows = rng_split.permutation(np.flatnonzero(labels_all == c))
        cut = max(2, int(np.ceil(fit_fraction * len(rows))))
        fit_rows.append(rows[:cut])
        q = rows[cut:]
        query_rows.append(q if query_per_class is None else q[:query_per_class])
    rng_seq = np.random.RandomState(int(sequence_seed) + 7919 * int(split_seed))
    novel_order = rng_seq.permutation(novel_classes)
    support_rows = [np.concatenate(fit_rows)]
    for s in range(n_sessions):
        chunk = novel_order[s * way:(s + 1) * way]
        rows_s = []
        for c in chunk:
            rows = rng_seq.permutation(np.flatnonzero(labels_all == c))
            if len(rows) <= shot:
                raise ValueError('Pseudo-novel class has too few images for support + query')
            rows_s.append(rows[:shot])
            q = rows[shot:]
            query_rows.append(q if query_per_class is None else q[:query_per_class])
        support_rows.append(np.concatenate(rows_s))
    class_order = np.concatenate([fit_classes, novel_order])
    position = {int(c): i for i, c in enumerate(class_order)}
    def bank(rows):
        labels = np.array([position[int(c)] for c in labels_all[rows]], dtype=np.int64)
        return Bank(train['final'][rows], labels, rows, 'train', _views(train, rows, views))
    support = [bank(r) for r in support_rows]
    query = bank(np.concatenate(query_rows))
    sessions = [np.arange(0, n_fit)] + [np.arange(n_fit + s * way, n_fit + (s + 1) * way)
                                        for s in range(n_sessions)]
    text, desc_proto, desc, targets = _text_subset(train, class_order)
    return Episode('pseudo', n_fit, sessions, support, query, text, desc_proto, desc, targets,
                   class_order, {'split_seed': split_seed, 'sequence_seed': sequence_seed,
                                 'shot': shot, 'dataset': meta.get('dataset')})


def pseudo_episodes(train, meta, *, shot=5, split_seeds=(2, 3, 4), sequences=20, views=VIEW_KEYS,
                    **overrides):
    params = dict(PSEUDO_DEFAULTS.get(meta.get('dataset'), PSEUDO_DEFAULTS['cub200']))
    params.update({k: v for k, v in overrides.items() if v is not None})
    for split_seed in split_seeds:
        for sequence in range(sequences):
            yield pseudo_episode(train, meta, shot=shot, split_seed=split_seed,
                                 sequence_seed=sequence, views=views, **params)
