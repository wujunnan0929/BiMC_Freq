"""NumPy-only metrics for class-incremental evaluation and JSON summaries."""

from collections.abc import Mapping, Sequence

import numpy as np


def _validate_inputs(scores, targets, task_id, class_groups):
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [N, C], got {scores.shape}")
    if targets.ndim != 1:
        raise ValueError(f"targets must have shape [N], got {targets.shape}")
    if scores.shape[0] != targets.shape[0]:
        raise ValueError("scores and targets must contain the same number of samples")
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        raise ValueError("scores and targets must not be empty")
    if not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
        raise TypeError("scores must be a real numeric array")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    if not np.issubdtype(targets.dtype, np.integer) or np.issubdtype(targets.dtype, np.bool_):
        raise TypeError("targets must contain integer class indices")
    if isinstance(task_id, (bool, np.bool_)) or not isinstance(task_id, (int, np.integer)):
        raise TypeError("task_id must be an integer")
    task_id = int(task_id)
    if not isinstance(class_groups, Sequence) or isinstance(class_groups, (str, bytes)):
        raise TypeError("class_groups must be a sequence of class-index sequences")
    if not 0 <= task_id < len(class_groups):
        raise ValueError(f"task_id {task_id} is outside class_groups")

    normalized_groups = []
    all_class_ids = []
    for group_index, group in enumerate(class_groups):
        values = np.asarray(group)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError(f"class group {group_index} must be a non-empty 1-D sequence")
        if (not np.issubdtype(values.dtype, np.integer)
                or np.issubdtype(values.dtype, np.bool_)):
            raise TypeError(f"class group {group_index} must contain integer indices")
        values = values.astype(np.int64, copy=False)
        if np.any(values < 0) or len(np.unique(values)) != len(values):
            raise ValueError(f"class group {group_index} contains invalid or duplicate indices")
        normalized_groups.append(values)
        all_class_ids.extend(values.tolist())
    if len(set(all_class_ids)) != len(all_class_ids):
        raise ValueError("class indices must be unique across class groups")

    seen_groups = normalized_groups[:task_id + 1]
    seen_classes = np.concatenate(seen_groups)
    expected_classes = np.arange(len(seen_classes), dtype=np.int64)
    if not np.array_equal(np.sort(seen_classes), expected_classes):
        raise ValueError(
            "seen class indices must be contiguous from zero so score columns map to labels"
        )
    if scores.shape[1] != len(seen_classes):
        raise ValueError(
            f"scores has {scores.shape[1]} columns but task {task_id} has "
            f"{len(seen_classes)} seen classes"
        )
    if np.any(targets < 0) or not np.isin(targets, seen_classes).all():
        raise ValueError("targets contain a class that is not visible in this session")
    missing = seen_classes[~np.isin(seen_classes, np.unique(targets))]
    if len(missing):
        raise ValueError(f"evaluation data has no samples for seen classes {missing.tolist()}")
    return scores, targets.astype(np.int64, copy=False), task_id, seen_groups


def _accuracy(correct, mask):
    if not np.any(mask):
        raise ValueError("cannot compute accuracy for an empty class subset")
    return float(np.mean(correct[mask]) * 100.0)


def compute_session_metrics(scores, targets, task_id, class_groups):
    """Compute one all-seen evaluation row using dataset-global class indices.

    Returned rates are percentages. ``task_acc`` preserves full floating-point
    precision and contains one micro accuracy for every task group seen so far.
    """
    scores, targets, task_id, groups = _validate_inputs(
        scores, targets, task_id, class_groups,
    )
    predictions = np.argmax(scores, axis=1)
    correct = predictions == targets
    masks = [np.isin(targets, group) for group in groups]
    task_acc = [_accuracy(correct, mask) for mask in masks]
    accuracy = float(np.mean(correct) * 100.0)
    base_accuracy = task_acc[0]
    per_class = [
        _accuracy(correct, targets == class_id)
        for group in groups for class_id in group
    ]

    result = {
        'session': task_id,
        'accuracy': accuracy,
        'base_accuracy': base_accuracy,
        'novel_accuracy': None,
        'harmonic_accuracy': None,
        'current_novel_accuracy': None,
        'old_to_new_rate': None,
        'task_acc': task_acc,
        'class_macro_accuracy': float(np.mean(per_class)),
        'inc_session_mean_accuracy': None,
    }
    if task_id == 0:
        return result

    novel_mask = np.logical_or.reduce(masks[1:])
    novel_accuracy = _accuracy(correct, novel_mask)
    denominator = base_accuracy + novel_accuracy
    harmonic = (2.0 * base_accuracy * novel_accuracy / denominator
                if denominator > 0.0 else 0.0)
    current_classes = groups[-1]
    historical_mask = np.logical_or.reduce(masks[:-1])
    result.update({
        'novel_accuracy': novel_accuracy,
        'harmonic_accuracy': float(harmonic),
        'current_novel_accuracy': task_acc[-1],
        'old_to_new_rate': float(
            np.mean(np.isin(predictions[historical_mask], current_classes)) * 100.0
        ),
        'inc_session_mean_accuracy': float(np.mean(task_acc[1:])),
    })
    return result


def _task_acc(row, expected_length=None):
    values = row.get('task_acc') if isinstance(row, Mapping) else row
    if (not isinstance(values, Sequence) or isinstance(values, (str, bytes))
            or len(values) == 0):
        raise ValueError("each session must contain a non-empty task_acc sequence")
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(
            f"task_acc must have {expected_length} entries, got {len(values)}"
        )
    parsed = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
            raise TypeError("task_acc entries must be numeric")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("task_acc entries must be finite")
        parsed.append(value)
    return parsed


def compute_forgetting(task_acc, history):
    """Compute average task forgetting; improvements intentionally remain negative."""
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise TypeError("history must be a sequence of prior sessions")
    current = _task_acc(task_acc)
    task_id = len(current) - 1
    if task_id == 0:
        if len(history) != 0:
            raise ValueError("base-session forgetting requires empty history")
        return None
    if len(history) != task_id:
        raise ValueError(
            f"history must contain sessions 0..{task_id - 1}, got {len(history)} rows"
        )
    rows = [_task_acc(row, expected_length=session + 1)
            for session, row in enumerate(history)]
    drops = []
    for old_task in range(task_id):
        best_prior = max(rows[session][old_task]
                         for session in range(old_task, task_id))
        drops.append(best_prior - current[old_task])
    return float(np.mean(drops))


def add_forgetting(session, history):
    """Return a JSON-ready copy of a session row with its forgetting value."""
    if not isinstance(session, Mapping):
        raise TypeError("session must be a mapping")
    result = dict(session)
    result['forgetting'] = compute_forgetting(result, history)
    return result


def _finite_metric(row, key, allow_none=False):
    value = row.get(key)
    if value is None and allow_none:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"session metric {key} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"session metric {key} must be finite")
    return value


def summarize_sessions(sessions):
    """Build the compact final/average JSON summary from ordered session rows."""
    if (not isinstance(sessions, Sequence) or isinstance(sessions, (str, bytes))
            or len(sessions) == 0):
        raise ValueError("sessions must be a non-empty ordered sequence")
    normalized = []
    for expected_session, row in enumerate(sessions):
        if not isinstance(row, Mapping):
            raise TypeError("each session must be a mapping")
        if row.get('session') != expected_session:
            raise ValueError("sessions must be ordered consecutively from session zero")
        _task_acc(row, expected_length=expected_session + 1)
        normalized.append({
            key: _finite_metric(row, key, allow_none=(expected_session == 0))
            for key in ('accuracy', 'base_accuracy', 'novel_accuracy',
                        'harmonic_accuracy', 'old_to_new_rate')
        })
    final = normalized[-1]
    forgetting = compute_forgetting(sessions[-1], sessions[:-1])
    return {
        'final_accuracy': final['accuracy'],
        'average_accuracy': float(np.mean([row['accuracy'] for row in normalized])),
        'base_accuracy': final['base_accuracy'],
        'novel_accuracy': final['novel_accuracy'],
        'harmonic_accuracy': final['harmonic_accuracy'],
        'old_to_new_rate': final['old_to_new_rate'],
        'forgetting': forgetting,
    }


__all__ = [
    'compute_session_metrics',
    'compute_forgetting',
    'add_forgetting',
    'summarize_sessions',
]
