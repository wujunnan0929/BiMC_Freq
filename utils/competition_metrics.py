"""Separate original-base/incremental group confusion from within-group errors."""
import numpy as np


def competition_errors(scores, targets, num_old):
    scores, y = np.asarray(scores), np.asarray(targets)
    if scores.ndim != 2 or y.shape != (len(scores),) or not 0 < num_old <= scores.shape[1]:
        raise ValueError('Competition metrics require N x C scores and a valid original-base count.')
    prediction = scores.argmax(-1)
    old, pred_old = y < num_old, prediction < num_old
    wrong = prediction != y
    masks = {'old_to_novel': old & ~pred_old, 'novel_to_old': ~old & pred_old,
             'old_within_group_errors': old & pred_old & wrong,
             'novel_within_group_errors': ~old & ~pred_old & wrong}
    return {'old_queries': int(old.sum()), 'novel_queries': int((~old).sum()),
        'counts': {key: int(mask.sum()) for key, mask in masks.items()},
        'rates': {key: (float(100.*mask.sum()/((old if key.startswith('old_') else ~old).sum()))
                       if bool((old if key.startswith('old_') else ~old).any()) else None)
                  for key, mask in masks.items()}, 'rate_unit': 'percent_within_true_group'}
