"""Build a shareable source ZIP with an optional ready-to-use dataset ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment import git_state


def package(output, dataset=None):
    files = list(ROOT.glob('*.py'))
    names = ['README.md', 'LICENSE', 'requirements.txt', 'requirements-lock.txt',
             'environment.yml', 'Dockerfile', '.gitignore', '.dockerignore']
    files.extend(ROOT / name for name in names if (ROOT / name).is_file())
    for folder in ('scripts', 'tests', 'docs', 'imgs'):
        files.extend(path for path in (ROOT / folder).rglob('*')
                     if path.is_file() and '__pycache__' not in path.parts
                     and path.suffix in ('.py', '.md', '.png', '.jpg'))
    output = Path(output).resolve()
    if output.suffix.lower() != '.zip':
        raise ValueError('--output must end in .zip')
    output.parent.mkdir(parents=True, exist_ok=True)
    if dataset is not None:
        dataset = Path(dataset).resolve()
        if not dataset.is_file() or dataset.suffix.lower() != '.zip':
            raise ValueError('--include_dataset must point to an existing dataset ZIP')
        if dataset == output:
            raise ValueError('The share package must not overwrite the input dataset ZIP')
    manifest = {'git': git_state(), 'files': {}}
    with zipfile.ZipFile(str(output), 'w', compression=zipfile.ZIP_DEFLATED) as target:
        for path in sorted(set(files)):
            relative = path.relative_to(ROOT).as_posix()
            raw = path.read_bytes()
            target.writestr(relative, raw)
            manifest['files'][relative] = hashlib.sha256(raw).hexdigest()
        if dataset is not None:
            raw = dataset.read_bytes()
            target.writestr('Dataset (use directly).zip', raw, compress_type=zipfile.ZIP_STORED)
            manifest['files']['Dataset (use directly).zip'] = hashlib.sha256(raw).hexdigest()
        target.writestr('SHARE_MANIFEST.json', json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    print('Ready to share: {} ({:.1f} MiB)'.format(output, output.stat().st_size / 1024**2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(ROOT / 'dist' / 'CycleGAN-Music-Experiments.zip'))
    parser.add_argument('--include_dataset', help='include the original dataset ZIP in the share package')
    args = parser.parse_args()
    try:
        package(args.output, args.include_dataset)
    except (OSError, ValueError) as error:
        parser.exit(1, 'Packaging failed: {}\n'.format(error))


if __name__ == '__main__':
    main()
