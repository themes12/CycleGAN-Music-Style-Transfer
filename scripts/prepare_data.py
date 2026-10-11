"""Extract, validate and fingerprint the ready-to-use piano-roll archive."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

DOMAINS = ('JC_J', 'JC_C', 'JP_J', 'JP_P', 'CP_C', 'CP_P', 'JCP_mixed')


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def phrase_path(filename):
    path = PurePosixPath(filename.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or any(':' in part for part in path.parts):
        raise ValueError('Unsafe path in archive: {}'.format(filename))
    # Accept both root-level genre folders and a wrapper folder around them.
    for index, part in enumerate(path.parts):
        if part in DOMAINS:
            relative = PurePosixPath(*path.parts[index:])
            if relative.suffix != '.npy':
                return None
            valid = (part == 'JCP_mixed' and len(relative.parts) == 2) or (
                part != 'JCP_mixed' and len(relative.parts) == 3 and
                relative.parts[1] in ('train', 'test'))
            if not valid:
                raise ValueError('Unexpected dataset layout: {}'.format(filename))
            return relative
    return None


def prepare(archive, output, limit=None, expected_sha256=None):
    archive, output = Path(archive), Path(output)
    if limit is not None and limit < 1:
        raise ValueError('--limit_per_split must be positive')
    if output.exists() and any(output.iterdir()):
        raise ValueError('{} is not empty. Choose a new --output directory.'.format(output))
    archive_hash = checksum(archive)
    if expected_sha256 and archive_hash.lower() != expected_sha256.lower():
        raise ValueError('Dataset checksum differs from --expected_sha256')
    groups = defaultdict(list)
    with zipfile.ZipFile(archive) as source:
        seen = set()
        for entry in source.infolist():
            relative = phrase_path(entry.filename)
            if relative is None or entry.is_dir():
                continue
            if relative in seen:
                raise ValueError('Duplicate phrase path in archive: {}'.format(relative))
            seen.add(relative)
            groups[str(relative.parent)].append((relative, entry))
        required = ['{}/{}'.format(domain, split) for domain in DOMAINS[:-1]
                    for split in ('train', 'test')] + ['JCP_mixed']
        missing = [group for group in required if not groups[group]]
        if missing:
            raise ValueError('Missing dataset folders: {}'.format(', '.join(missing)))
        output.mkdir(parents=True, exist_ok=True)
        counts = {}
        content = hashlib.sha256()
        for group in sorted(groups):
            selected = sorted(groups[group], key=lambda pair: str(pair[0]))[:limit]
            for relative, entry in selected:
                array = np.load(io.BytesIO(source.read(entry)), allow_pickle=False)
                if array.shape == (1, 64, 84, 1):
                    array = array[0]
                if array.shape != (64, 84, 1) or not np.all((array == 0) | (array == 1)):
                    raise ValueError('{} must contain a binary piano roll of shape (64,84,1)'.format(relative))
                target = output.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(target), array.astype(np.bool_))
                content.update(str(relative).encode('utf-8') + b'\0')
                content.update(hashlib.sha256(target.read_bytes()).digest())
            counts[group] = len(selected)
            print('{}: {} phrases'.format(group, len(selected)), flush=True)
    manifest = {'archive': archive.name, 'archive_sha256': archive_hash,
                'content_sha256': content.hexdigest(), 'limit_per_split': limit,
                'shape': [64, 84, 1], 'counts': counts}
    (output / 'manifest.json').write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    print('Dataset ready: {} (SHA256 {})'.format(output, archive_hash), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default='Dataset (use directly).zip')
    parser.add_argument('--output', default='datasets')
    parser.add_argument('--limit_per_split', type=int, help='extract a deterministic subset for a smoke test')
    parser.add_argument('--expected_sha256', help='optional expected checksum of the source ZIP')
    args = parser.parse_args()
    try:
        prepare(args.archive, args.output, args.limit_per_split, args.expected_sha256)
    except (ValueError, OSError, zipfile.BadZipFile) as error:
        parser.exit(1, 'Data preparation failed: {}\n'.format(error))


if __name__ == '__main__':
    main()
