"""Build a portable Kaggle upload from an explicit code list and the audited JC data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_data import PreparedData

CODE_FILES = [
    'tf_compat.py', 'model.py', 'module.py', 'ops.py', 'utils.py', 'write_midi.py',
    'style_classifier.py', 'repro_data.py', 'experiment.py', 'experiment_data.py',
    'experiment_metrics.py', 'experiment_health.py', 'experiment_objectives.py',
    'KAGGLE.md', 'kaggle/bootstrap.py', 'kaggle/session.py', 'kaggle/preflight.py',
    'kaggle/config.json', 'kaggle/conda-linux-64.lock', 'kaggle/pip-bootstrap.lock',
    'kaggle/requirements-tf14.lock', 'kaggle/requirements-tf14.in', 'kaggle/cyclegan_tf14.ipynb',
    'evidence/kaggle-tf14-verification.json',
]


def build(prepared, output):
    # Verifies every prepared data SHA and the original domain/split counts.
    data = PreparedData(prepared)
    output.mkdir(parents=True, exist_ok=False)
    bundle = output / 'cyclegan-kaggle-tf14'
    bundle.mkdir()
    for name in CODE_FILES:
        destination = bundle / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    (bundle / 'data').mkdir()
    for name in ['manifest.json'] + [key + '.npy' for key in data.manifest['splits']]:
        shutil.copy2(prepared / name, bundle / 'data' / name)
    files = {p.relative_to(bundle).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(bundle.rglob('*')) if p.is_file()}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    manifest = {'protocol': 'native-tf14-jazz-classic-base-v1',
        'git_base': git.stdout.decode().strip() if git.returncode == 0 else None,
        'source_of_truth': 'Per-file SHA256 includes uncommitted working-tree changes.',
        'data_identity': data.manifest['identity'], 'files': files}
    (bundle / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2))
    archive = output / 'cyclegan-kaggle-tf14.zip'
    with zipfile.ZipFile(str(archive), 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as output_zip:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                output_zip.write(str(path), path.relative_to(output).as_posix())
    shutil.copy2(ROOT / 'kaggle/cyclegan_tf14.ipynb', output / 'cyclegan_tf14.ipynb')
    print(json.dumps({'zip': str(archive), 'bytes': archive.stat().st_size,
        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'data_identity': data.manifest['identity'], 'files': len(files)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory, never overwrite an existing bundle')
    a = p.parse_args()
    build(a.prepared_data.resolve(), a.output.resolve())
