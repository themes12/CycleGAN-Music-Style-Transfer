"""Registered Table VII-IX configurations; compatible with Python 3.5."""
PROTOCOL = 'native-tf14-paper-selected-nine-v1'
PAIRS = {
    'JC': {'a': 'JC_J', 'b': 'JC_C', 'genres': ['Jazz', 'Classic'], 'table': 'VII', 'train': 11216, 'test': 1125, 'classifier_reference': .8889},
    'CP': {'a': 'CP_C', 'b': 'CP_P', 'genres': ['Classic', 'Pop'], 'table': 'VIII', 'train': 14598, 'test': 1864, 'classifier_reference': .8466},
    'JP': {'a': 'JP_J', 'b': 'JP_P', 'genres': ['Jazz', 'Pop'], 'table': 'IX', 'train': 11216, 'test': 1125, 'classifier_reference': .6707},
}
SELECTED = [
    ('JC', 'base', 1., .697), ('JC', 'partial', 0., .716), ('JC', 'full', .01, .690),
    ('CP', 'base', .1, .336), ('CP', 'partial', 1., .526), ('CP', 'full', 1., .581),
    ('JP', 'base', .01, .354), ('JP', 'partial', .01, .373), ('JP', 'full', 0., .305),
]
RUNS = [{'id': pair + '-' + model, 'pair': pair, 'model': model, 'sigma_d': sigma,
         'paper_strength': strength, 'table': PAIRS[pair]['table']} for pair, model, sigma, strength in SELECTED]
REMAINING = [r['id'] for r in RUNS if r['id'] != 'JC-base']
# Table rows in order A, A->B, A->B->A, B, B->A, B->A->B (percent).
PAPER_ROWS = {
    'JC-base': [88.09,20.62,88.18,92.53,20.71,92.53],
    'JC-partial': [88.09,9.87,87.73,92.53,25.87,89.51],
    'JC-full': [88.09,20.,85.16,92.53,20.09,90.49],
    'CP-base': [86.91,45.12,83.26,80.04,49.14,72.48],
    'CP-partial': [86.91,38.57,87.39,80.04,23.13,79.45],
    'CP-full': [86.91,26.61,87.18,80.04,24.30,80.15],
    'JP-base': [60.53,21.60,57.51,73.60,40.62,74.40],
    'JP-partial': [60.53,14.84,58.84,73.60,43.11,72.09],
    'JP-full': [60.53,23.64,60.53,73.60,49.24,73.06],
}
CLASSIFIER_PAPER = {
    'JC': [88.89,88.53,87.87,84.71,83.07,74.93],
    'CP': [84.66,83.42,81.97,81.12,78.14,70.28],
    'JP': [67.07,66.18,63.78,62.40,61.72,59.96],
}


def configuration(run_id=None, classifier_pair=None):
    if classifier_pair:
        if classifier_pair not in PAIRS:
            raise ValueError('Unknown classifier pair')
        entry = dict(pair=classifier_pair, model='base', sigma_d=0.)
    else:
        entry = next((r for r in RUNS if r['id'] == run_id), None)
        if entry is None:
            raise ValueError('Unknown experiment: ' + str(run_id))
    pair = PAIRS[entry['pair']]
    return dict(protocol=PROTOCOL, tensorflow_required='1.4.0', python_required='3.5.4',
        pair=entry['pair'], dataset_a=pair['a'], dataset_b=pair['b'], genres=pair['genres'],
        model=entry['model'], sigma_d=entry['sigma_d'], gamma=1., cycle_loss='legacy_l1',
        seed=19, batch_size=16, epochs=30, classifier_epochs=30, decay_start_epoch=10,
        learning_rate=.0002, beta1=.5, beta2=.999, lambda_cycle=10., ngf=64, ndf=64,
        deterministic=False, checkpoint_every_steps=100, report_every_steps=10,
        evaluate_epochs=[20, 30], classifier_robustness_sigmas=[0., .01, .1, .2, .3, .5, 1.],
        replay_pool_size=50 if entry['model'] != 'base' else 0,
        mixed_source=('JCP_mixed' if entry['model'] == 'full' else 'pair_training_union' if entry['model'] == 'partial' else None),
        paper_reference={'table': pair['table'], 'strength': entry.get('paper_strength'),
                         'classifier_accuracy_sigma0': pair['classifier_reference']})


def validate_config(config):
    pair = PAIRS[config['pair']]
    if (config['dataset_a'], config['dataset_b']) != (pair['a'], pair['b']):
        raise ValueError('Pair and label orientation mismatch')
    if config['model'] not in ('base', 'partial', 'full') or config['cycle_loss'] != 'legacy_l1':
        raise ValueError('Only registered legacy model objectives are supported')
    expected = 'JCP_mixed' if config['model'] == 'full' else 'pair_training_union' if config['model'] == 'partial' else None
    if config['mixed_source'] != expected:
        raise ValueError('Wrong mixed training source')
    if config['model'] != 'base' and config['replay_pool_size'] <= 0:
        raise ValueError('Non-base legacy runs require a replay pool')
