"""Verified published splits; no resplitting or new labels. Python 3.5."""
import hashlib
import json
from pathlib import Path
import numpy as np
from suite_plan import PAIRS, validate_config


def file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class PreparedData:
    def __init__(self, root, config=None):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.arrays = {}
        if config:
            validate_config(config)
            keys = [config['dataset_' + side] + '_' + mode for side in ['a', 'b'] for mode in ['train', 'test']]
            if config['model'] == 'full':
                keys.append('JCP_mixed')
        else:
            keys = sorted(self.manifest['splits'])
        for key in keys:
            info = self.manifest['splits'][key]
            path = self.root / (key + '.npy')
            if file_sha(path) != info['sha256']:
                raise ValueError('Staged dataset changed: ' + key)
            a = np.load(str(path), mmap_mode='r', allow_pickle=False)
            if a.shape != (info['count'], 64, 84, 1) or a.dtype != np.bool_:
                raise ValueError('Invalid binary piano-roll array: ' + key)
            self.arrays[key] = a
        for pair in PAIRS.values():
            for domain in [pair['a'], pair['b']]:
                for mode in ['train', 'test']:
                    key = domain + '_' + mode
                    if key in self.arrays and len(self.arrays[key]) != pair[mode]:
                        raise ValueError('Published split count mismatch: ' + key)
        if 'JCP_mixed' in self.arrays and len(self.arrays['JCP_mixed']) != 33648:
            raise ValueError('Published mixed dataset count mismatch')


def mixed_training(data, config, a, b):
    if config['model'] == 'base':
        return None
    if config['model'] == 'partial':
        return np.concatenate([a, b], axis=0)
    return data.arrays['JCP_mixed']
