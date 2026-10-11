"""Budgeted, resumable Kaggle queue. Run with the modern notebook Python."""
import argparse
import csv
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite_plan import PROTOCOL, PAIRS, RUNS, REMAINING, configuration, PAPER_ROWS, CLASSIFIER_PAPER
from bootstrap import runtime_environment


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or '\\' in relative:
        raise ValueError('Unsafe manifest path: ' + relative)
    return path


def verify_bundle(bundle):
    manifest = json.loads((bundle / 'bundle-manifest.json').read_text())
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('Not a paper-suite bundle')
    for relative, expected in manifest['files'].items():
        if sha(safe_path(bundle, relative)) != expected:
            raise ValueError('Bundle changed: ' + relative)
    return manifest


def verify_saved(root):
    manifest = json.loads((root / 'suite-manifest.json').read_text())
    if manifest['protocol'] != PROTOCOL:
        raise ValueError('Use a previous paper-suite output, not the old single JC output')
    for relative, expected in manifest['immutable_files'].items():
        if sha(safe_path(root, relative)) != expected:
            raise ValueError('Saved suite code/data/baseline changed: ' + relative)
    return manifest


def prepare_run(bundle, root, resume=None):
    if resume:
        resume = resume.resolve()
        verify_saved(resume)
        if root.exists() and root.resolve() != resume:
            raise ValueError('Output exists; refusing to overwrite it with another session')
        if root.resolve() != resume:
            shutil.copytree(resume, root)
    if root.exists():
        return verify_saved(root)
    manifest = verify_bundle(bundle)
    root.mkdir(parents=True)
    immutable = {}
    for relative, digest in manifest['files'].items():
        destination = relative if relative.startswith(('data/', 'baseline/')) else 'code/' + relative
        path = safe_path(root, destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle / relative, path)
        immutable[destination] = digest
    result = dict(protocol=PROTOCOL, immutable_files=immutable, source=manifest)
    save(root / 'suite-manifest.json', result)
    return result


def execute(command, log_path, environment, deadline, progress=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if time.monotonic() >= deadline:
        return False
    with log_path.open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        stopped_at = None
        def stop(_signum=None, _frame=None):
            nonlocal stopped_at
            if stopped_at is None:
                stopped_at = time.monotonic()
                if process.poll() is None:
                    process.terminate()
        handlers = {s: signal.signal(s, stop) for s in [signal.SIGINT, signal.SIGTERM]}
        previous = None
        try:
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    stop()
                if stopped_at is not None and time.monotonic() - stopped_at > 180:
                    process.kill()
                    process.wait()
                    raise RuntimeError('Child did not stop; use its last durable checkpoint')
                if progress and progress.exists():
                    raw = progress.read_text()
                    if raw != previous:
                        v = json.loads(raw)
                        print(json.dumps({k: v[k] for k in ['state', 'kind', 'epoch', 'step', 'total_steps'] if k in v}), flush=True)
                        previous = raw
                time.sleep(2)
        finally:
            if process.poll() is None:
                stop()
                try:
                    process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
    if process.returncode and stopped_at is None:
        raise RuntimeError('Command failed. {}\n{}'.format(log_path, log_path.read_text(errors='replace')[-5000:]))
    return stopped_at is None


def stage_complete(stage, pair, kind):
    marker = stage / 'complete.json'
    if not marker.exists():
        return False
    result = json.loads(marker.read_text())
    n = PAIRS[pair]['train'] * (2 if kind == 'classifier' else 1)
    expected = (n // 16) * 30
    prefix = stage / 'epoch-30' / ('model-' + str(expected))
    valid = result.get('state') == 'complete' and result.get('completed_steps') == expected
    valid = valid and Path(str(prefix) + '.index').is_file() and Path(str(prefix) + '.data-00000-of-00001').is_file()
    if valid:
        valid = all(Path(str(prefix) + suffix).stat().st_size > 0 for suffix in ['.index', '.data-00000-of-00001'])
    if not valid:
        raise ValueError('Invalid completion or missing final checkpoint: ' + str(stage))
    return True


def collect(root):
    rows = []
    classifiers = {}
    for entry in RUNS:
        for epoch in [20, 30]:
            report_path = (root / 'baseline' / ('evaluation-epoch-' + str(epoch)) / 'report.json'
                if entry['id'] == 'JC-base' else root / 'runs' / entry['id'] / ('evaluation-epoch-' + str(epoch)) / 'report.json')
            row = dict(run=entry['id'], pair=entry['pair'], model=entry['model'], sigma_d=entry['sigma_d'],
                       table=entry['table'], epoch=epoch, paper_strength=entry['paper_strength'], state='pending')
            if report_path.exists():
                report = json.loads(report_path.read_text())
                if report.get('limited_smoke'):
                    raise ValueError('Smoke report cannot enter paper tables')
                if entry['id'] != 'JC-base' and any(report.get(k) != entry[k] for k in ['pair', 'model', 'sigma_d']):
                    raise ValueError('Report belongs to a different experiment: ' + str(report_path))
                scores = report['binary_legacy_pipeline']
                expected_n = PAIRS[entry['pair']]['test']
                if any(scores[d]['n'] != expected_n for d in ['AtoB', 'BtoA']):
                    raise ValueError('Incomplete test evaluation: ' + str(report_path))
                row.update(state='complete', provenance='imported_prior_run' if entry['id'] == 'JC-base' else 'suite_run',
                    classifier_accuracy=report['classifier_robustness']['0.0'],
                    strength_argmax=scores['bidirectional_legacy_accuracy_derived_strength'],
                    strength_probability=scores['bidirectional_signed_probability_strength_all'])
                classifiers[entry['pair']] = report['classifier_robustness']
                row['gap_percentage_points'] = 100 * (row['strength_argmax'] - row['paper_strength'])
                for direction in ['AtoB', 'BtoA']:
                    metrics = scores[direction]
                    row[direction + '_success'] = metrics['successful_fraction']
                    row[direction + '_success_count'] = metrics['successful_count']
                    row[direction + '_n'] = metrics['n']
                    for i, stage in enumerate(['origin', 'transfer', 'cycle']):
                        row[direction + '_' + stage + '_source_rate'] = metrics['legacy_source_classification_rate'][i]
                        row[direction + '_' + stage + '_source_probability'] = metrics['mean_source_probability'][i]
                    for mode in ['transfer', 'cycle']:
                        for metric in ['silent_count', 'note_density', 'note_precision', 'note_recall', 'note_f1']:
                            row[direction + '_' + mode + '_' + metric] = metrics[mode + '_notes'][metric]
            rows.append(row)
    save(root / 'results.json', {'protocol': PROTOCOL, 'rows': rows,
        'all_nine_complete': all(r['state'] == 'complete' for r in rows),
        'scope': 'Published selected settings, not the 54-model sigma search. Inspect note quality separately.'})
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (root / 'paper-tables.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (root / 'classifier-table.csv').open('w', newline='', encoding='utf-8-sig') as out:
        writer = csv.DictWriter(out, fieldnames=['pair', 'sigma_c', 'paper_accuracy', 'our_accuracy', 'state'])
        writer.writeheader()
        for pair in PAIRS:
            for i, sigma in enumerate([0., .01, .1, .2, .3, .5, 1.]):
                value = classifiers.get(pair, {}).get(str(sigma))
                writer.writerow(dict(pair=pair, sigma_c=sigma, paper_accuracy=CLASSIFIER_PAPER[pair][i]/100 if i<6 else '',
                    our_accuracy=value if value is not None else '', state='complete' if value is not None else 'pending'))
    lines = ['# Tables VII-IX — selected configurations', '',
        'Both registered epochs are retained. Missing rows are pending, not zero scores.', '',
        '| Run | Epoch | Paper strength | Our argmax strength | Our probability strength | State |',
        '|---|---:|---:|---:|---:|---|']
    for r in rows:
        fmt = lambda key: '{:.2f}%'.format(100 * r[key]) if key in r else '—'
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(r['run'], r['epoch'], fmt('paper_strength'), fmt('strength_argmax'), fmt('strength_probability'), r['state']))
    by_run = {r['run']: r for r in rows if r['epoch'] == 30}
    labels = ['A', 'A→B', 'A→B→A', 'B', 'B→A', 'B→A→B']
    for pair, info in PAIRS.items():
        lines += ['', '## Table {} — {} ↔ {} (epoch 30)'.format(info['table'], *info['genres']), '',
            'Source-class values: each cell is **paper / ours (argmax rate)**. Probability means are retained separately in CSV.', '',
            '| Stage | Base | Partial | Full |', '|---|---:|---:|---:|']
        for i, label in enumerate(labels):
            direction = 'AtoB' if i<3 else 'BtoA'
            stage = ['origin','transfer','cycle'][i%3]
            cells=[]
            for model in ['base','partial','full']:
                name=pair+'-'+model;r=by_run[name];key=direction+'_'+stage+'_source_rate'
                ours='{:.2f}%'.format(r[key]*100) if key in r else 'pending'
                cells.append('{:.2f}% / {}'.format(PAPER_ROWS[name][i],ours))
            lines.append('| {} | {} |'.format(label,' | '.join(cells)))
        cells=[]
        for model in ['base','partial','full']:
            r=by_run[pair+'-'+model];ours='{:.2f}%'.format(r['strength_argmax']*100) if 'strength_argmax' in r else 'pending'
            cells.append('{:.2f}% / {}'.format(r['paper_strength']*100,ours))
        lines.append('| Strength | {} |'.format(' | '.join(cells)))
    lines += ['', 'Reference: https://arxiv.org/pdf/1809.07575 — Tables IV and VII–IX.',
        'Repository architecture, half-normal noise and aggregation differ from paper prose; these are descriptive comparisons.']
    (root / 'PAPER-TABLES.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return all(r['state'] == 'complete' for r in rows)


def export_results(root):
    """Omit redundant rolling checkpoints only for fully completed training stages."""
    destination = root.parent / 'paper-suite-results.zip'
    temporary = destination.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.name == '.session.lock':
                continue
            if 'checkpoints' in path.parts:
                stage = path.parent.parent
                if (stage / 'complete.json').exists() and (stage / 'epoch-30').exists():
                    continue
            archive.write(path, path.relative_to(root.parent).as_posix())
    temporary.replace(destination)
    return destination


def run(args):
    root = args.output.resolve()
    # Reserve 15 minutes inside the supplied session allowance for ZIP export.
    deadline = time.monotonic() + max(0, args.seconds - 900)
    prepare_run(args.bundle.resolve(), root, args.resume)
    code = root / 'code'
    if sha(Path(__file__).resolve()) != sha(code / 'kaggle/suite_session.py'):
        raise ValueError('Resume must use the frozen supervisor from the saved code directory')
    env = runtime_environment(args.python.parent.parent)
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    lock = root / '.session.lock'
    # flock locks are released by the OS even when a prior session crashes.
    import fcntl
    with lock.open('w') as lockfile:
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another suite supervisor is already using this output')
        status = dict(protocol=PROTOCOL, mode=args.mode, state='starting', selected=args.runs)
        def publish(**items):
            status.update(items, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            save(root / 'session-status.json', status)
        def train_stage(stage, config_path, pair, kind):
            if args.mode == 'train' and stage_complete(stage, pair, kind):
                return True
            if args.mode == 'smoke' and (stage / 'smoke-complete.json').exists():
                return True
            remaining = deadline - time.monotonic()
            if remaining < 300:
                return False
            if shutil.disk_usage(root).free < 3 * 1024**3:
                publish(state='paused', reason='disk_space_less_than_3GiB')
                return False
            command = [str(args.python), str(code / 'experiment.py'), 'train', '--config', str(config_path),
                '--data', str(root / 'data'), '--device', args.device, '--kind', kind,
                '--run-dir', str(stage), '--max-seconds', str(remaining - 180)]
            if args.mode == 'smoke':
                command += ['--max-steps', '1']
            if stage.exists():
                if not (stage / 'checkpoints/checkpoint').exists():
                    # No durable weights exist to restore; keep the failed attempt
                    # and restart this stage from the same registered initial seed.
                    preserved = stage.with_name(stage.name + '-no-checkpoint-' + str(time.time_ns()))
                    stage.rename(preserved)
                    print('Restarting stage with no durable checkpoint; prior attempt kept at', preserved, flush=True)
                else:
                    command.append('--resume')
            publish(state='running', phase=kind, stage=str(stage.relative_to(root)))
            ready = execute(command, stage.parent / (stage.name + '.log'), env, deadline, stage / 'status.json')
            if ready and args.mode == 'smoke':
                last = json.loads((stage / 'last-invocation.json').read_text())
                if last['completed_steps'] < 1:
                    raise ValueError('Smoke did not execute an update')
                save(stage / 'smoke-complete.json', dict(updates=last['completed_steps'], quality_claim=False))
                return True
            return ready and stage_complete(stage, pair, kind)
        try:
            publish(phase='preflight')
            if not execute([str(args.python), str(code / 'kaggle/preflight.py'), '--device', args.device,
                            '--report', str(root / 'preflight.json')], root / 'preflight.log', env, deadline):
                publish(state='paused', reason='session_budget')
                return
            for run_id in args.runs:
                entry = next(r for r in RUNS if r['id'] == run_id)
                pair = entry['pair']
                prefix = root / 'smoke' if args.mode == 'smoke' else root
                config_path = code / 'kaggle/configs' / (run_id + '.json')
                classifier_config = code / 'kaggle/configs' / ('classifier-' + pair + '.json')
                if args.mode == 'smoke':
                    for original in [config_path, classifier_config]:
                        config = json.loads(original.read_text())
                        config.update(protocol=PROTOCOL + '-smoke', batch_size=1)
                        save(prefix / 'configs' / original.name, config)
                    config_path = prefix / 'configs' / config_path.name
                    classifier_config = prefix / 'configs' / classifier_config.name
                classifier = (root / 'baseline/classifier' if pair == 'JC' else prefix / 'classifiers' / pair)
                if pair != 'JC' and not train_stage(classifier, classifier_config, pair, 'classifier'):
                    publish(state='paused', reason=status.get('reason', 'session_budget'))
                    return
                gan = prefix / 'runs' / run_id / 'gan'
                gan.parent.mkdir(parents=True, exist_ok=True)
                if not train_stage(gan, config_path, pair, 'gan'):
                    publish(state='paused', reason=status.get('reason', 'session_budget'))
                    return
                if args.mode == 'smoke':
                    continue
                for epoch in [20, 30]:
                    evaluation = gan.parent / ('evaluation-epoch-' + str(epoch))
                    if (evaluation / 'report.json').exists():
                        continue
                    if deadline - time.monotonic() < 600:
                        publish(state='paused', reason='session_budget')
                        return
                    if evaluation.exists():
                        evaluation.rename(evaluation.with_name(evaluation.name + '-incomplete-' + str(time.time_ns())))
                    publish(state='running', phase='evaluate', stage=run_id, epoch=epoch)
                    ready = execute([str(args.python), str(code / 'experiment.py'), 'evaluate', '--config', str(config_path),
                        '--data', str(root / 'data'), '--device', args.device, '--run-dir', str(evaluation),
                        '--gan-checkpoint', str(gan / ('epoch-' + str(epoch))),
                        '--classifier-checkpoint', str(classifier / 'epoch-30')], gan.parent / (evaluation.name + '.log'), env, deadline)
                    if not ready:
                        publish(state='paused', reason='session_budget')
                        return
                    if not (evaluation / 'report.json').exists():
                        raise RuntimeError('Evaluation exited without report')
                    collect(root)
            all_complete = collect(root)
            publish(state='smoke_complete' if args.mode == 'smoke' else 'complete' if all_complete else 'selection_complete', phase='finished')
        except BaseException as error:
            publish(state='failed', error=repr(error))
            raise
        finally:
            try:
                collect(root)
            except Exception as error:
                save(root / 'summary-error.json', {'error': repr(error)})
                print('Could not rebuild tables; retaining raw artifacts:', error, flush=True)
            print('Exporting resumable output...', flush=True)
            print('Saved:', export_results(root), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/cyclegan-paper-suite'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--mode', choices=['train', 'smoke'], default='train')
    parser.add_argument('--seconds', type=float, default=36000)
    parser.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--runs', nargs='+', choices=REMAINING, default=REMAINING)
    args = parser.parse_args()
    if args.seconds <= 900 or len(set(args.runs)) != len(args.runs):
        parser.error('Need more than 900 seconds and distinct run IDs')
    if args.device == 'cpu' and args.mode != 'smoke':
        parser.error('CPU is permitted only for bounded smoke checks')
    run(args)
