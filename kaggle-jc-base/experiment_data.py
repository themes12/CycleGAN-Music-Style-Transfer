"""Materialize the unchanged JC split as compact boolean arrays on Linux storage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from repro_data import Dataset, EXPECTED_JC


def prepare(source, destination):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    data = Dataset(source)
    data.require_jc()
    manifest = {'source': str(data.path), 'splits': {}}
    identity = hashlib.sha256()
    for split, count in EXPECTED_JC.items():
        domain, mode = split.split('/')
        names = data.split(domain, mode)
        key = domain + '_' + mode
        array = np.lib.format.open_memmap(str(out / (key + '.npy')), mode='w+', dtype=np.bool_,
                                         shape=(count, 64, 84, 1))
        entries = []
        for i, name in enumerate(names):
            raw_sha = hashlib.sha256(data.raw(name)).hexdigest()
            array[i] = data.array(name)
            entries.append({'name': name, 'sha256': raw_sha})
            identity.update((name + ':' + raw_sha + '\n').encode())
        array.flush()
        del array
        file_sha = hashlib.sha256((out / (key + '.npy')).read_bytes()).hexdigest()
        manifest['splits'][key] = {'count': count, 'sha256': file_sha, 'samples': entries}
        print('Prepared', key, count, flush=True)
    manifest['identity'] = identity.hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


class PreparedData:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.arrays = {}
        for key, info in self.manifest['splits'].items():
            path = self.root / (key + '.npy')
            if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
                raise ValueError('Staged dataset changed: ' + key)
            a = np.load(str(path), mmap_mode='r', allow_pickle=False)
            assert a.shape == (info['count'], 64, 84, 1) and a.dtype == np.bool_
            self.arrays[key] = a
        for split, count in EXPECTED_JC.items():
            assert len(self.arrays[split.replace('/', '_')]) == count


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    prepare(a.source, a.output)
