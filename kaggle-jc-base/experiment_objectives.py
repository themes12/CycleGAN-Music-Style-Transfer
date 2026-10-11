"""Explicit experimental objectives; the default returns the unchanged legacy loss."""
from tf_compat import tf


def balanced_l1(real, reconstructed, note_prevalence):
    if not 0 < note_prevalence < 1:
        raise ValueError('Balanced L1 requires both note and empty bins in training data')
    weights = real * (.5 / note_prevalence) + (1-real) * (.5 / (1-note_prevalence))
    return tf.reduce_mean(weights * tf.abs(real-reconstructed))


def generator_objective(model, config, densities):
    mode = config.get('cycle_loss', 'legacy_l1')
    metadata = {'cycle_loss': mode, 'training_note_prevalence': densities,
                'lambda_cycle': config['lambda_cycle']}
    if mode == 'legacy_l1':
        return model.g_loss, model.cycle_loss, metadata
    if mode != 'balanced_l1':
        raise ValueError('Unknown cycle_loss: ' + str(mode))
    cycle = config['lambda_cycle'] * (
        balanced_l1(model.real_A, model.fake_A_, densities['A']) +
        balanced_l1(model.real_B, model.fake_B_, densities['B']))
    metadata['note_weight'] = {key: .5 / value for key, value in densities.items()}
    metadata['empty_weight'] = {key: .5 / (1-value) for key, value in densities.items()}
    return model.g_loss - model.cycle_loss + cycle, cycle, metadata
