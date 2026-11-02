"""Session protocols: the real benchmark and base-only pseudo-incremental episodes.

An Episode is a list of sessions. Session t has
    classes[t]   global class ids that arrive at t
    support[t]   train-bank rows available at t (session 0 = all base rows)
    queries[t]   rows (in ``query_split``) whose labels are in the classes
                 seen up to t
``num_base`` is the number of session-0 classes. Methods may only read the
support rows of sessions <= t and the text of classes seen so far.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Episode:
    name: str
    classes: list
    support: list
    queries: list
    query_split: str
    meta: dict = field(default_factory=dict)

    @property
    def num_base(self):
        return len(self.classes[0])

    @property
    def num_sessions(self):
        return len(self.classes)

    def seen(self, t):
        return np.concatenate(self.classes[:t + 1])


def task_classes(num_classes, num_init, num_inc):
    tasks = [np.arange(0, num_init)]
    for start in range(num_init, num_classes, num_inc):
        tasks.append(np.arange(start, min(start + num_inc, num_classes)))
    return tasks


def replica_support_indices(train_labels, tasks, base_shot, inc_shot, seed):
    """Exact replica of DatasetManager._initialize_support_protocol.

    One np.random.RandomState(seed) is consumed in session order; a class is
    sub-sampled with rng.choice(candidates, shot, replace=False) whenever
    shot != -1 and shot <= #candidates (also when equal, which permutes).
    """
    rng = np.random.RandomState(int(seed))
    labels = np.asarray(train_labels)
    out = []
    for t, classes in enumerate(tasks):
        shot = base_shot if t == 0 else inc_shot
        chosen = []
        for c in classes:
            cand = np.flatnonzero(labels == c)
            if shot is not None and shot != -1:
                if shot < 1:
                    raise ValueError('shot must be positive or -1')
                if shot <= len(cand):
                    cand = rng.choice(cand, size=shot, replace=False)
            chosen.append(cand)
        out.append(np.concatenate(chosen).astype(np.int64))
    return out


def benchmark_episode(bank, seed, shot=None, use_stored=True):
    """Real FSCIL sequence: base = full base support, queries = test images."""
    tasks = task_classes(bank.num_classes, bank.num_init, bank.num_inc)
    shot = bank.inc_shot if shot is None else int(shot)
    support = bank.stored_support(seed) if (use_stored and shot == bank.inc_shot) else None
    source = 'stored_engine_manifest'
    if support is None:
        support = replica_support_indices(bank.labels('train'), tasks, bank.base_shot, shot, seed)
        source = 'replica_sampler'
    test_labels = bank.labels('test')
    queries = []
    for t in range(len(tasks)):
        seen = np.concatenate(tasks[:t + 1])
        queries.append(np.flatnonzero(np.isin(test_labels, seen)))
    return Episode(name=f'bench_seed{seed}_shot{shot}', classes=tasks, support=support,
                   queries=queries, query_split='test',
                   meta={'seed': int(seed), 'shot': shot, 'support_source': source,
                         'kind': 'benchmark'})


def pseudo_episodes(bank, split_seed, n_sequences=10, shot=None, way=None,
                    base_fraction=0.5, query_fraction=0.2):
    """Base-only pseudo-incremental episodes (never touch test images).

    The base classes are split once per ``split_seed`` into A (pseudo-base) and
    V (pseudo-novel). A's training rows are split into fit rows (pseudo-base
    support) and held-out query rows. Each sequence re-orders V into sessions
    of ``way`` classes and draws ``shot`` supports per V class; V queries are
    other training rows of V, capped at the A per-class query count so base
    and novel queries are balanced as on the real test sets.
    """
    shot = bank.inc_shot if shot is None else int(shot)
    way = bank.num_inc if way is None else int(way)
    labels = bank.labels('train')
    base = np.arange(bank.num_init)
    rng = np.random.RandomState(10007 * int(split_seed) + 17)
    perm = rng.permutation(base)
    n_a = int(round(base_fraction * len(base)))
    A, V = np.sort(perm[:n_a]), np.sort(perm[n_a:])
    fit_rows, a_query = [], []
    per_class_q = []
    for c in A:
        rows = rng.permutation(np.flatnonzero(labels == c))
        nq = max(1, int(round(query_fraction * len(rows))))
        a_query.append(rows[:nq])
        fit_rows.append(rows[nq:])
        per_class_q.append(nq)
    fit_rows = np.concatenate(fit_rows)
    a_query = np.concatenate(a_query)
    q_cap = int(np.median(per_class_q))
    episodes = []
    for s in range(int(n_sequences)):
        r = np.random.RandomState(1000003 * int(split_seed) + 7919 * s + 1)
        order = r.permutation(V)
        chunks = [np.sort(order[i:i + way]) for i in range(0, len(order), way)]
        classes, support = [A], [fit_rows]
        v_query = {}
        for chunk in chunks:
            sup = []
            for c in chunk:
                rows = r.permutation(np.flatnonzero(labels == c))
                if len(rows) <= shot:
                    raise ValueError(f'class {c} has too few training images for shot={shot}')
                sup.append(rows[:shot])
                v_query[int(c)] = rows[shot:shot + q_cap]
            classes.append(chunk)
            support.append(np.concatenate(sup))
        queries = []
        for t in range(len(classes)):
            parts = [a_query] + [v_query[int(c)] for ch in classes[1:t + 1] for c in ch]
            queries.append(np.concatenate(parts))
        episodes.append(Episode(name=f'pseudo_split{split_seed}_seq{s}_shot{shot}',
                                classes=classes, support=support, queries=queries,
                                query_split='train',
                                meta={'split_seed': int(split_seed), 'sequence': s,
                                      'shot': shot, 'way': way, 'kind': 'pseudo',
                                      'A': A.tolist(), 'V': V.tolist()}))
    return episodes


def check_episode(episode, bank):
    """Defensive checks: no support/query overlap, labels consistent."""
    labels = bank.labels('train')
    seen_sup = set()
    for t, (cls, sup) in enumerate(zip(episode.classes, episode.support)):
        if not np.isin(labels[sup], cls).all():
            raise ValueError(f'session {t} support has foreign labels')
        seen_sup.update(sup.tolist())
    if episode.query_split == 'train':
        overlap = seen_sup.intersection(np.concatenate(episode.queries).tolist())
        if overlap:
            raise ValueError(f'{len(overlap)} pseudo query rows are also support rows')
    qlabels = bank.labels(episode.query_split)
    for t, q in enumerate(episode.queries):
        if not np.isin(qlabels[q], episode.seen(t)).all():
            raise ValueError(f'session {t} queries include unseen classes')
    return True
