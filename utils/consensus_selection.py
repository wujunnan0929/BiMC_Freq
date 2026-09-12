"""Base-validation selection rules; no model, benchmark labels, or file access."""

import math


OBJECTIVES = ('micro_all', 'micro_incremental', 'balanced_incremental')


def summarize_counts(counts):
    result = dict(counts)
    n = counts['count']
    result.update(
        accuracy=100 * counts['correct'] / n if n else None,
        reference_accuracy=100 * counts['reference_correct'] / n if n else None,
        delta_accuracy_pp=100 * (counts['corrected'] - counts['damaged']) / n if n else None,
        eligible_rate=100 * counts['eligible'] / n if n else None,
        top2_recall=100 * counts['top2_hits'] / n if n else None,
    )
    return result


def score_candidate(stages, objective, max_group_drop_pp=-1.0):
    if objective not in OBJECTIVES:
        raise ValueError('Unknown calibration objective: ' + str(objective))
    if not math.isfinite(max_group_drop_pp) or (max_group_drop_pp < 0 and max_group_drop_pp != -1):
        raise ValueError('MAX_GROUP_DROP_PP must be -1 (disabled) or finite nonnegative.')
    incremental = [stage for stage in stages if stage['stage'] > 0]
    selected = stages if objective == 'micro_all' else incremental
    if not selected or not incremental:
        raise ValueError('Selection requires nonempty incremental validation stages.')
    n = sum(stage['groups']['all']['count'] for stage in selected)
    if n <= 0:
        raise ValueError('Selection requires validation queries.')
    if objective == 'balanced_incremental':
        scores = []
        for stage in incremental:
            groups = stage['groups']
            if any(groups[name]['count'] <= 0 for name in ('old', 'new')):
                raise ValueError('Balanced selection requires both old and new queries at every stage.')
            scores.append(sum(groups[name]['accuracy'] for name in ('old', 'new')) / 2)
        score = sum(scores) / len(scores)
    else:
        score = 100 * sum(stage['groups']['all']['correct'] for stage in selected) / n
    # The guard is on each stage, pooled over episodes, for old and ALL incremental
    # classes. It is a validation constraint, not a test-time no-harm guarantee.
    violations = []
    if max_group_drop_pp >= 0:
        for stage in incremental:
            for name in ('old', 'new'):
                group = stage['groups'][name]
                if not group['count']:
                    raise ValueError('Group constraint requires old and new validation queries.')
                if group['delta_accuracy_pp'] < -max_group_drop_pp - 1e-12:
                    violations.append({'stage': stage['stage'], 'group': name,
                                       'delta_accuracy_pp': group['delta_accuracy_pp']})
    return {'objective': objective, 'score': score, 'feasible': not violations,
            'constraint_violations': violations,
            'query_occurrences': n,
            'old_query_fraction': sum(stage['groups']['old']['count'] for stage in selected) / n,
            'stages': [stage['stage'] for stage in selected]}


def select_lambda(candidates, objective, max_group_drop_pp=-1.0):
    eligible = []
    for candidate in candidates:
        result = score_candidate(candidate['stages'], objective, max_group_drop_pp)
        if result['feasible']:
            eligible.append((candidate['lambda'], result['score']))
    if not eligible:
        raise ValueError('No feasible lambda; zero must be evaluated as a fallback.')
    best = max(score for _, score in eligible)
    # Equal rational accuracies can differ by floating-point summation noise.
    return min(value for value, score in eligible if score >= best - 1e-12)
