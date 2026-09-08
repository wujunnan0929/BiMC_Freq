"""Paired prediction diagnostics; labels are used only after inference."""

import numpy as np


def prediction_diagnostics(reference, output, targets, task_id, class_groups,
                           eligible=None, encoded=None):
    reference, output = np.asarray(reference), np.asarray(output)
    targets = np.asarray(targets, dtype=np.int64)
    if reference.shape != output.shape or reference.ndim != 2:
        raise ValueError('Reference and output must have matching [N,C] shapes.')
    if len(targets) != len(reference) or reference.shape[1] < 2:
        raise ValueError('Diagnostics require matching targets and at least two classes.')
    first = reference.argmax(1)
    remaining = reference.copy()
    remaining[np.arange(len(first)), first] = -np.inf
    second = remaining.argmax(1)
    prediction = output.argmax(1)
    before, after = first == targets, prediction == targets
    changed = first != prediction
    top2_hit = before | (second == targets)
    base = np.isin(targets, class_groups[0])
    current = np.isin(targets, class_groups[task_id]) if task_id else np.zeros_like(base)
    history_ids = np.concatenate(class_groups[1:task_id]) if task_id > 1 else []
    masks = {'all': np.ones_like(base), 'base': base,
             'historical_incremental': np.isin(targets, history_ids),
             'current_new': current}

    def row(mask):
        count = int(mask.sum())
        corrected = int((~before & after & mask).sum())
        damaged = int((before & ~after & mask).sum())
        result = {
            'count': count, 'corrected': corrected, 'damaged': damaged,
            'wrong_to_wrong': int((~before & ~after & changed & mask).sum()),
            'changed': int((changed & mask).sum()),
            'reference_accuracy': float(100 * before[mask].mean()) if count else None,
            'accuracy': float(100 * after[mask].mean()) if count else None,
            'delta_accuracy_pp': 100 * (corrected - damaged) / count if count else None,
            'top2_recall': float(100 * top2_hit[mask].mean()) if count else None,
            'top2_headroom_pp': float(100 * (top2_hit[mask].sum() - before[mask].sum()) / count)
                                if count else None,
        }
        for name, values in (('eligible', eligible), ('encoded', encoded)):
            if values is not None:
                values = np.asarray(values, dtype=bool)
                if values.shape != targets.shape:
                    raise ValueError(name + ' must match targets.')
                result[name + '_count'] = int(values[mask].sum())
                result[name + '_rate'] = float(100 * values[mask].mean()) if count else None
        return result

    result = {name: row(mask) for name, mask in masks.items()}
    seen_incremental = np.concatenate(class_groups[1:task_id+1]) if task_id else []
    for name, values in (('reference', first), ('output', prediction)):
        result['base_to_all_incremental_' + name + '_rate'] = (
            float(100 * np.isin(values[base], seen_incremental).mean()) if base.any() else None
        )
    result['prediction_outside_reference_top2_count'] = int(
        ((prediction != first) & (prediction != second)).sum()
    )
    return result
