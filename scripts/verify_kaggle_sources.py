"""Verify published TF1.4 snapshots against the historical experiment hashes."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, help='Optional directory with all three release ZIP files')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / 'docs/kaggle-tf14-provenance.json').read_text(encoding='utf-8'))
    failures = []
    for name, expected in record['source_sha256'].items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(name)
    if args.assets:
        for name, info in record['release_assets'].items():
            path = args.assets / name
            if not path.is_file() or sha256(path) != info['sha256']:
                failures.append('release asset: ' + name)
    if failures:
        raise SystemExit('Missing or changed historical files:\n' + '\n'.join(failures))
    print('Verified {} historical source files.'.format(len(record['source_sha256'])))
    if args.assets:
        print('Verified {} release archives.'.format(len(record['release_assets'])))


if __name__ == '__main__':
    main()
