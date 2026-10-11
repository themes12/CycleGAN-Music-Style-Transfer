"""Note-based diagnostics and a train-only validation split for guarded variants."""
import numpy as np


def split_training(array, count, seed):
    if not isinstance(count, int) or count < 0 or count >= len(array):
        raise ValueError('validation_per_domain must leave nonempty training data')
    if not count:
        return array, array[:0], {'training_count': len(array), 'validation_indices': []}
    validation = np.sort(np.random.RandomState(seed).permutation(len(array))[:count])
    mask = np.ones(len(array), dtype=bool)
    mask[validation] = False
    return array[mask], array[validation], {'training_count': int(mask.sum()),
        'validation_indices': validation.tolist(), 'order': 'original order with validation indices removed'}


def note_metrics(origin, output):
    origin, output = np.asarray(origin) > .5, np.asarray(output) > .5
    if origin.shape != output.shape or not len(origin):
        raise ValueError('Nonempty, matching phrase arrays are required')
    n = len(origin)
    tp, actual, predicted = int((origin & output).sum()), int(origin.sum()), int(output.sum())
    silent = int((~output.reshape(n, -1).any(axis=1)).sum())
    return {'n': n, 'silent_count': silent, 'silent_fraction': silent / n,
            'note_density': float(output.mean()),
            'note_precision': tp / predicted if predicted else 0.,
            'note_recall': tp / actual if actual else 0.,
            'note_f1': 2 * tp / (actual + predicted) if actual + predicted else 0.}


def assess_outputs(origin, transfer, cycle, limits):
    if not all(np.isfinite(a).all() for a in [origin, transfer, cycle]):
        return {'passed': False, 'reasons': ['non-finite output']}
    result = {name: note_metrics(origin, value) for name, value in [('transfer', transfer), ('cycle', cycle)]}
    reasons = []
    for name, metrics in result.items():
        if metrics['silent_fraction'] > limits['max_silent_fraction']:
            reasons.append(name + ': silent fraction above limit')
        if metrics['note_density'] > limits['max_note_density']:
            reasons.append(name + ': note density above limit')
    if result['cycle']['note_f1'] < limits['min_cycle_f1']:
        reasons.append('cycle: note F1 below limit')
    if result['cycle']['note_precision'] < limits['min_cycle_precision']:
        reasons.append('cycle: note precision below limit')
    return {**result, 'passed': not reasons, 'reasons': reasons}
