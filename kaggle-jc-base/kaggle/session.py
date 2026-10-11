"""Modern-Python supervisor for native TF1.4 subprocesses and Kaggle resumes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

from bootstrap import runtime_environment


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def verify_bundle(bundle):
    manifest = json.loads((bundle / 'bundle-manifest.json').read_text())
    for relative, digest in manifest['files'].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()):
            raise ValueError('Unsafe bundle path: ' + relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Bundle file changed: ' + relative)
    return manifest


def prepare_run(bundle, root, resume=None):
    if resume:
        resume = resume.resolve()
        if root.exists() and root.resolve() != resume:
            raise ValueError('Output already exists; do not overwrite it with a resume input.')
        if not (resume / 'run-manifest.json').exists():
            raise ValueError('Resume must point to a previous cyclegan-tf14 output directory.')
        if root.resolve() != resume:
            shutil.copytree(resume, root)
    if root.exists():
        manifest = json.loads((root / 'run-manifest.json').read_text())
        for name, digest in manifest['code_sha256'].items():
            path = (root / 'code' / name).resolve()
            if not path.is_relative_to((root / 'code').resolve()):
                raise ValueError('Unsafe saved code path')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Saved run source/config changed: ' + name)
        return
    manifest = verify_bundle(bundle)
    root.mkdir(parents=True)
    (root / 'code').mkdir()
    hashes = {}
    for name, digest in manifest['files'].items():
        if name.startswith('data/'):
            destination = root / name
        else:
            destination = root / 'code' / name
            hashes[name] = digest
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / name, destination)
    save(root / 'run-manifest.json', {'code_sha256': hashes, 'source': manifest,
        'protocol': 'native-tf14-jazz-classic-base-v1'})


def export_results(root):
    archive = root.parent / 'tf14-results.zip'
    temporary = archive.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                output.write(path, str(path.relative_to(root.parent)))
    temporary.replace(archive)
    return archive


def execute(command, log_path, environment, deadline, progress=None):
    with log_path.open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        stopping = False
        terminated_at = None
        previous = None
        def stop(_signum, _frame):
            nonlocal stopping, terminated_at
            stopping = True
            if terminated_at is None and process.poll() is None:
                process.terminate()
                terminated_at = time.monotonic()
        old_handlers = {s: signal.signal(s, stop) for s in [signal.SIGINT, signal.SIGTERM]}
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    stop(None, None)
                if terminated_at is not None and time.monotonic() - terminated_at > 180:
                    process.kill()
                    process.wait()
                    raise RuntimeError('Child did not stop in time; inspect the previous checkpoint.')
                if progress and progress.exists():
                    value = progress.read_text()
                    if value != previous:
                        info = json.loads(value)
                        print(json.dumps({k: info[k] for k in ['state', 'kind', 'step', 'total_steps',
                            'epoch', 'recent_step_seconds'] if k in info}), flush=True)
                        previous = value
                time.sleep(5)
        finally:
            if process.poll() is None:
                stop(None, None)
                try:
                    process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    if process.returncode and not stopping:
        tail = log_path.read_text(errors='replace')[-6000:]
        raise RuntimeError('Command failed; see {}\n{}'.format(log_path, tail))
    return not stopping


def run(args):
    root, bundle = args.output.resolve(), args.bundle.resolve()
    deadline = time.monotonic() + args.seconds
    prepare_run(bundle, root, args.resume)
    code = root / 'code'
    config_path = code / 'kaggle/config.json'
    config = json.loads(config_path.read_text())
    if config.get('tensorflow_required') != '1.4.0' or config.get('cycle_loss') != 'legacy_l1':
        raise ValueError('This notebook is the native TF1.4 legacy-loss experiment.')
    env = runtime_environment(args.python.parent.parent)
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    status = {'state': 'starting', 'protocol': config['protocol'], 'device': args.device}
    def publish(**values):
        status.update(values, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        save(root / 'session-status.json', status)
    common = [str(args.python), str(code / 'experiment.py'), '--config', str(config_path),
              '--data', str(root / 'data'), '--device', args.device]
    try:
        publish(phase='preflight')
        ready = execute([str(args.python), str(code / 'kaggle/preflight.py'), '--device', args.device,
                         '--report', str(root / 'preflight.json')], root / 'preflight.log', env, deadline)
        if not ready:
            publish(state='paused', reason='session_budget')
            return
        # Smoke has a different config and directories; its weights never seed training.
        if args.mode == 'smoke':
            smoke_config = dict(config, protocol='native-tf14-smoke-only', batch_size=1)
            save(root / 'smoke-config.json', smoke_config)
            common[common.index('--config') + 1] = str(root / 'smoke-config.json')
        for kind in ['gan', 'classifier']:
            stage = root / (('smoke-' if args.mode == 'smoke' else '') + kind)
            if (stage / 'complete.json').exists():
                continue
            remaining = deadline - time.monotonic()
            if remaining < 300:
                publish(state='paused', phase=kind, reason='session_budget')
                return
            command = common + ['train', '--kind', kind, '--run-dir', str(stage),
                                '--max-seconds', str(remaining - 180)]
            if args.mode == 'smoke':
                command += ['--max-steps', '1']
            if stage.exists():
                if not (stage / 'checkpoints/checkpoint').exists():
                    raise RuntimeError('Previous stage has no restorable checkpoint: ' + str(stage))
                command.append('--resume')
            publish(state='running', phase=kind)
            ready = execute(command, root / (stage.name + '.log'), env, deadline,
                            progress=stage / 'status.json')
            if not ready or (args.mode != 'smoke' and not (stage / 'complete.json').exists()):
                publish(state='paused', phase=kind, reason='session_budget')
                return
        if args.mode == 'smoke':
            publish(state='smoke_complete', note='Only one G/D and one classifier update; no 30-epoch result.')
            return
        for epoch in config['evaluate_epochs']:
            stage = root / 'evaluation-epoch-{:02d}'.format(epoch)
            if (stage / 'report.json').exists():
                continue
            if deadline - time.monotonic() < 600:
                publish(state='paused', phase=stage.name, reason='session_budget')
                return
            if stage.exists():
                stage.rename(stage.with_name(stage.name + '-incomplete-' + str(time.time_ns())))
            publish(state='running', phase=stage.name)
            ready = execute(common + ['evaluate', '--run-dir', str(stage), '--gan-checkpoint',
                str(root / 'gan' / 'epoch-{:02d}'.format(epoch)), '--classifier-checkpoint',
                str(root / 'classifier' / 'epoch-{:02d}'.format(config['classifier_epochs']))],
                root / (stage.name + '.log'), env, deadline)
            if not ready:
                publish(state='paused', reason='session_budget')
                return
        reports = {str(epoch): json.loads((root / 'evaluation-epoch-{:02d}'.format(epoch) /
                   'report.json').read_text()) for epoch in config['evaluate_epochs']}
        save(root / 'results.json', {'protocol': config['protocol'], 'reports': reports,
            'interpretation': 'Native TF1.4 legacy loss. Inspect silence and note metrics; completion is not reproduction success.'})
        publish(state='complete', phase='evaluated')
    except BaseException as error:
        publish(state='failed', error=repr(error))
        raise
    finally:
        archive = export_results(root)
        print('Saved results and resumable state:', archive, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--python', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('/kaggle/working/cyclegan-tf14'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--mode', choices=['smoke', 'train'], default='train')
    p.add_argument('--seconds', type=float, default=36000)
    p.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    args = p.parse_args()
    if args.seconds <= 0:
        p.error('--seconds must be positive')
    if args.device == 'cpu' and args.mode != 'smoke':
        p.error('CPU is allowed only for local smoke verification.')
    run(args)
