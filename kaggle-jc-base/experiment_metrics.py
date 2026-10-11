"""Explicit probability, success-conditioned, and legacy accuracy aggregations."""
import numpy as np
from experiment_health import note_metrics


def transfer_metrics(probabilities, origin, transfer, cycle, source_index):
    # probabilities: phrase, [origin, transfer, cycle], [A, B]
    p = probabilities[:, :, source_index]
    strength = (p[:, 0] + p[:, 2]) / 2 - p[:, 1]
    success = (p[:, 0] > .5) & (p[:, 1] < .5)
    accuracy = (probabilities.argmax(axis=2) == source_index).mean(axis=0)
    return {
        'n': len(p), 'mean_source_probability': p.mean(axis=0).tolist(),
        'legacy_source_classification_rate': accuracy.tolist(),
        'legacy_accuracy_derived_strength': float((accuracy[0] + accuracy[2]) / 2 - accuracy[1]),
        'signed_probability_strength_all': float(strength.mean()),
        'signed_probability_strength_successful': float(strength[success].mean()) if success.any() else None,
        'successful_count': int(success.sum()), 'successful_fraction': float(success.mean()),
        'legacy_absolute_origin_transfer_difference': float(np.abs(p[:, 0] - p[:, 1]).mean()),
        'binary_content_mse': float(np.mean((origin.astype(float) - transfer.astype(float)) ** 2)),
        'binary_cycle_mse': float(np.mean((origin.astype(float) - cycle.astype(float)) ** 2)),
        'note_density': [float(a.mean()) for a in [origin, transfer, cycle]],
        'transfer_notes': note_metrics(origin, transfer),
        'cycle_notes': note_metrics(origin, cycle),
        'silent_transfers_counted_successful': int((success & ~transfer.reshape(len(transfer), -1).any(axis=1)).sum()),
    }
