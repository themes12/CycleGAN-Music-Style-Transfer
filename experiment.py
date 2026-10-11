"""Portable paths and provenance for the original TensorFlow 1 experiments."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def safe_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
        raise ValueError("Use 1-64 letters, digits, underscores, dots or hyphens: {!r}".format(value))
    return value


def model_directory(args):
    if args.type == 'classifier':
        return 'classifier_{}2{}_{}_{}'.format(
            args.dataset_A_dir, args.dataset_B_dir, args.run_id, args.sigma_c)
    return '{}2{}_{}_{}_{}'.format(
        args.dataset_A_dir, args.dataset_B_dir, args.run_id, args.model, args.sigma_d)


def piano_roll_files(data_root, domain, split=None):
    directory = Path(data_root) / safe_name(domain)
    if split is not None:
        directory = directory / split
    files = sorted(str(path) for path in directory.glob('*.npy'))
    if not files:
        raise ValueError("No .npy phrases in {}. Run scripts/prepare_data.py first.".format(directory))
    return files


def configure_device(device):
    # Configure visibility before importing TensorFlow or any model modules.
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif device != 'auto':
        if not re.fullmatch(r'\d+(,\d+)*', device):
            raise ValueError("--device must be auto, cpu, or GPU indices such as 0 or 0,1")
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    elif 'CUDA_VISIBLE_DEVICES' not in os.environ and 'SGE_GPU' in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']


def git_state():
    root = str(Path(__file__).resolve().parent)
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, stderr=subprocess.DEVNULL,
            timeout=5).decode().strip()
        dirty = bool(subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=root, stderr=subprocess.DEVNULL,
            timeout=5).decode().strip())
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def source_fingerprint():
    root = Path(__file__).resolve().parent
    files = list(root.glob('*.py')) + list(root.glob('requirements*.txt'))
    files.extend((root / 'scripts').glob('*.py'))
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode('utf-8') + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def write_metadata(args, tensorflow_version):
    """Save one record per train/test invocation; keep earlier records intact."""
    directory = Path(args.log_dir) / model_directory(args)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    path = directory / 'metadata_{}_{}.json'.format(args.phase, stamp)
    data_manifest = Path(args.data_root) / 'manifest.json'
    dataset = None
    if data_manifest.is_file():
        raw = data_manifest.read_bytes()
        dataset = {'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                   'manifest': json.loads(raw.decode('utf-8'))}
    try:
        packages = subprocess.check_output(
            [sys.executable, '-m', 'pip', 'freeze'], stderr=subprocess.DEVNULL,
            timeout=30).decode().splitlines()
    except (OSError, subprocess.SubprocessError):
        packages = None
    record = {'arguments': vars(args), 'started_at': utc_now(), 'status': 'running',
              'git': git_state(), 'source_sha256': source_fingerprint(), 'python': platform.python_version(),
              'platform': platform.platform(), 'tensorflow': tensorflow_version,
              'packages': packages, 'dataset': dataset}
    update_metadata(path, record)
    return path, record


def update_metadata(path, record):
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
