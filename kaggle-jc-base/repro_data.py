"""Published per-phrase split reader; ZIPs need no extraction or resplitting."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

EXPECTED_JC = {'JC_J/train': 11216, 'JC_C/train': 11216,
               'JC_J/test': 1125, 'JC_C/test': 1125}


class Dataset:
    def __init__(self, path, allow_empty=False):
        self.path = Path(path).resolve()
        self.archive = zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.names = sorted(n for n in (self.archive.namelist() if self.archive else
                            [p.relative_to(self.path).as_posix() for p in self.path.rglob('*.npy')])
                            if n.endswith('.npy'))
        if (not self.names and not allow_empty) or len(set(self.names)) != len(self.names):
            raise ValueError('Empty dataset or duplicate ZIP member names')

    def raw(self, name):
        return self.archive.read(name) if self.archive else (self.path / name).read_bytes()

    def array(self, name):
        a = np.load(io.BytesIO(self.raw(name)), allow_pickle=False)
        if a.shape not in ((64, 84), (64, 84, 1), (1, 64, 84, 1)):
            raise ValueError('{}: unexpected shape {}'.format(name, a.shape))
        if not np.isin(a, [0, 1]).all():
            raise ValueError('{}: expected binary published piano roll'.format(name))
        return a.reshape(64, 84, 1).astype(np.float32)

    def split(self, domain, split):
        prefix = '{}/{}/'.format(domain, split)
        return [n for n in self.names if n.startswith(prefix)]

    def require_jc(self):
        counts = Counter(str(Path(n).parent).replace('\\', '/') for n in self.names)
        for split, expected in EXPECTED_JC.items():
            if counts[split] != expected:
                raise ValueError('Incomplete published split {}: {} != {}'.format(split, counts[split], expected))

    def batch(self, names):
        return np.stack([self.array(n) for n in names])


def audit(args):
    data = Dataset(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    counts, shapes, dtypes = Counter(), Counter(), Counter()
    digest = hashlib.sha256()
    invalid = []
    with gzip.open(out / 'manifest.jsonl.gz', 'wt', encoding='utf-8') as f:
        for i, name in enumerate(data.names):
            raw = data.raw(name)  # ZIP CRC is verified by ZipFile.read.
            a = np.load(io.BytesIO(raw), allow_pickle=False)
            sha = hashlib.sha256(raw).hexdigest()
            counts[name.rsplit('/', 1)[0]] += 1
            shapes[str(a.shape)] += 1
            dtypes[str(a.dtype)] += 1
            good = a.shape in ((64, 84), (64, 84, 1), (1, 64, 84, 1)) and np.isin(a, [0, 1]).all()
            if not good:
                invalid.append(name)
            row = {'name': name, 'sha256': sha, 'shape': list(a.shape), 'dtype': str(a.dtype),
                   'min': float(a.min()), 'max': float(a.max()), 'nonzero': int(np.count_nonzero(a))}
            line = json.dumps(row, sort_keys=True)
            digest.update((line + '\n').encode())
            f.write(line + '\n')
            if i % 10000 == 0:
                print('validated {}/{}'.format(i, len(data.names)), flush=True)
    result = {'data': str(data.path), 'total_npy': len(data.names), 'counts': dict(counts),
              'shapes': dict(shapes), 'dtypes': dict(dtypes), 'invalid': invalid,
              'manifest_sha256': digest.hexdigest(), 'range': [0, 1] if not invalid else None}
    if args.compare_root:
        other = Dataset(args.compare_root, allow_empty=True)
        missing = sorted(set(data.names) - set(other.names))
        extra = sorted(set(other.names) - set(data.names))
        archive_names = set(data.names)
        mismatched, unreadable = [], []
        for n in other.names:
            if n not in archive_names:
                continue
            try:
                if hashlib.sha256(other.raw(n)).digest() != hashlib.sha256(data.raw(n)).digest():
                    mismatched.append(n)
            except OSError as error:
                unreadable.append({'name': n, 'error': str(error)})
        result['extracted_comparison'] = {'path': str(other.path), 'path_exists': other.path.exists(),
            'count': len(other.names),
            'missing': len(missing), 'extra': len(extra), 'content_mismatch': len(mismatched),
            'unreadable': len(unreadable),
            'counts': dict(Counter(n.rsplit('/', 1)[0] for n in other.names))}
        (out / 'extracted-differences.json').write_text(json.dumps(
            {'missing': missing, 'extra': extra, 'content_mismatch': mismatched,
             'unreadable': unreadable}), encoding='utf-8')
    (out / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if invalid:
        raise ValueError('Invalid published phrases; see audit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--compare-root')
    parser.add_argument('--output', required=True)
    audit(parser.parse_args())
