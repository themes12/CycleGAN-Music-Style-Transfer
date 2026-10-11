"""Install the isolated Python 3.5 / TF 1.4 runtime from locked artifacts.

Run this file with the modern Kaggle notebook Python, not Python 3.5.
The notebook kernel and its packages are never replaced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
MAMBA_URL = 'https://micro.mamba.pm/api/micromamba/linux-64/2.3.3'
MAMBA_SHA256 = 'e7274528ceb9c20d048a428d6c22d7e02e268f8ffb762c4c365422347c8b8ba2'


def download(url, destination, sha256):
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == sha256:
        return
    temporary = destination.with_suffix('.download')
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            output.write(block)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
        raise ValueError('Download checksum mismatch: ' + url)
    temporary.replace(destination)


def runtime_environment(prefix):
    env = os.environ.copy()
    # Avoid importing libraries from the host notebook's modern Python/CUDA.
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    env.pop('LD_PRELOAD', None)
    driver = '/usr/lib/wsl/lib' if Path('/usr/lib/wsl/lib').exists() else '/usr/local/nvidia/lib64'
    env.update(PATH=str(prefix / 'bin') + ':' + env.get('PATH', ''),
               LD_LIBRARY_PATH=str(prefix / 'lib') + ':' + driver,
               PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               MPLBACKEND='Agg', PIP_DISABLE_PIP_VERSION_CHECK='1')
    certificates = '/etc/ssl/certs/ca-certificates.crt'
    if Path(certificates).exists():
        env.update(SSL_CERT_FILE=certificates, PIP_CERT=certificates,
                   REQUESTS_CA_BUNDLE=certificates)
    return env


def install(prefix):
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('This lock targets Linux x86_64 (Kaggle P100).')
    prefix = prefix.resolve()
    cache = prefix.parent / 'tf14-download-cache'
    cache.mkdir(parents=True, exist_ok=True)
    locks = [HERE / 'conda-linux-64.lock', HERE / 'pip-bootstrap.lock', HERE / 'requirements-tf14.lock']
    fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in locks)).hexdigest()
    marker = prefix / 'kaggle-runtime.json'
    if marker.exists():
        if json.loads(marker.read_text())['lock_sha256'] != fingerprint:
            raise RuntimeError('Runtime lock changed. Choose a new environment prefix.')
        print('Reusing locked TF 1.4 runtime:', prefix, flush=True)
        return prefix / 'bin/python'
    if prefix.exists():
        raise RuntimeError('Incomplete runtime exists. Choose a new prefix; it was not deleted: ' + str(prefix))
    archive = cache / 'micromamba-2.3.3.tar.bz2'
    download(MAMBA_URL, archive, MAMBA_SHA256)
    executable = cache / 'micromamba'
    with tarfile.open(str(archive)) as package:
        # Extract just this executable, never arbitrary archive paths.
        with executable.open('wb') as output:
            output.write(package.extractfile('bin/micromamba').read())
    executable.chmod(0o755)
    subprocess.run([str(executable), 'create', '-y', '-r', str(cache / 'mamba'),
                    '-p', str(prefix), '-f', str(locks[0])], check=True)
    python = prefix / 'bin/python'
    env = runtime_environment(prefix)
    for lock in locks[1:]:
        subprocess.run([str(python), '-m', 'pip', 'install', '--require-hashes', '-r', str(lock)],
                       env=env, check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], env=env, check=True)
    marker.write_text(json.dumps({'lock_sha256': fingerprint, 'python': '3.5.4',
        'tensorflow': '1.4.0', 'cuda': '8.0', 'cudnn': '6.0.21'}, indent=2))
    return python


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', type=Path, default=Path('/kaggle/temp/cyclegan-tf14'))
    args = p.parse_args()
    print(install(args.prefix))
