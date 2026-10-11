"""Custom schedules around the frozen TF1.4 engines; modern Python only."""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import time


FIELDS = {
    'epochs', 'classifier_epochs', 'batch_size', 'learning_rate', 'sigma_d',
    'seed', 'lambda_cycle', 'ngf', 'ndf', 'decay_start_epoch', 'evaluate_epochs',
    'beta1', 'beta2',
}


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('SETTINGS ต้องเป็น dictionary เช่น {"epochs": 50}')
    unknown = set(settings) - FIELDS
    if unknown:
        raise ValueError('ไม่รองรับ SETTINGS: ' + ', '.join(sorted(map(str, unknown))))


def configuration(bundle, experiment, settings):
    validate_settings(settings)
    path = bundle / 'kaggle' / ('config.json' if experiment == 'JC-base'
                               else 'configs/' + experiment + '.json')
    config = json.loads(path.read_text())
    config.update(settings)
    for key in ('epochs', 'classifier_epochs', 'batch_size', 'ngf', 'ndf'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + ' ต้องเป็นจำนวนเต็มมากกว่า 0')
    if type(config['seed']) is not int or not 0 <= config['seed'] < 2**31:
        raise ValueError('seed ต้องเป็นจำนวนเต็มตั้งแต่ 0 ถึง 2147483647')
    for key in ('learning_rate', 'lambda_cycle', 'sigma_d', 'beta1', 'beta2'):
        value = config[key]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(key + ' ต้องเป็นตัวเลขที่มีค่าจำกัด')
        minimum_ok = value >= 0 if key in ('sigma_d', 'beta1', 'beta2') else value > 0
        if not minimum_ok or (key in ('beta1', 'beta2') and value >= 1):
            raise ValueError('ค่า ' + key + ' อยู่นอกช่วงที่ใช้ได้')
    shortest = min(config['epochs'], config['classifier_epochs'])
    if 'decay_start_epoch' not in settings:
        config['decay_start_epoch'] = min(config['decay_start_epoch'], shortest // 3)
    decay = config['decay_start_epoch']
    if type(decay) is not int or not 0 <= decay < shortest:
        raise ValueError('decay_start_epoch ต้องตั้งแต่ 0 และน้อยกว่า epochs กับ classifier_epochs')
    epochs = settings.get('evaluate_epochs', [e for e in config['evaluate_epochs'] if e <= config['epochs']])
    if not isinstance(epochs, list) or any(type(e) is not int or not 1 <= e <= config['epochs'] for e in epochs):
        raise ValueError('evaluate_epochs ต้องเป็นรายการ epoch ตั้งแต่ 1 ถึง epochs')
    config['evaluate_epochs'] = sorted(set(epochs + [config['epochs']]))
    if len(config['evaluate_epochs']) > 30:
        raise ValueError('เลือก evaluate_epochs ได้ไม่เกิน 30 ค่า รวม epoch สุดท้าย')
    config['protocol'] = 'native-tf14-custom-v1'
    config['paper_reference'] = None
    return config


def request_id(request):
    raw = json.dumps(request, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def expected_steps(config, manifest, kind):
    a, b = config.get('dataset_a', 'JC_J'), config.get('dataset_b', 'JC_C')
    counts = [manifest['splits'][key + '_train']['count'] for key in (a, b)]
    examples = min(counts) if kind == 'gan' else sum(counts)
    batches = examples // config['batch_size']
    if not batches:
        raise ValueError('batch_size ใหญ่กว่าจำนวนข้อมูลสำหรับ ' + kind)
    return batches * config['epochs' if kind == 'gan' else 'classifier_epochs']


def stage_complete(stage, config, manifest, kind):
    marker = stage / 'complete.json'
    if not marker.exists():
        return False
    result = json.loads(marker.read_text())
    expected = expected_steps(config, manifest, kind)
    epoch = config['epochs' if kind == 'gan' else 'classifier_epochs']
    checkpoint = stage / ('epoch-{:02d}'.format(epoch)) / ('model-' + str(expected))
    if result.get('state') != 'complete' or result.get('completed_steps') != expected:
        raise ValueError('จำนวน steps ไม่ตรงกับ custom config: ' + kind)
    for suffix in ('.index', '.data-00000-of-00001'):
        path = Path(str(checkpoint) + suffix)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError('ไม่พบ checkpoint สุดท้ายของ ' + kind)
    return True


def run(args):
    bundle, root = args.bundle.resolve(), args.output.resolve()
    request = json.loads(args.request.read_text())
    config = configuration(bundle, request['experiment'], request['settings'])
    if config != request['config']:
        raise ValueError('Custom config changed after launch')
    runner_bytes = Path(__file__).read_bytes()
    if hashlib.sha256(runner_bytes).hexdigest() != request['runner_sha256']:
        raise ValueError('Custom runner changed after launch')
    previous = root / 'custom-run.json'
    if previous.exists() and json.loads(previous.read_text()) != request:
        raise ValueError('Resume requires the same custom settings and source')
    sys.path.insert(0, str(bundle / 'kaggle'))
    helper = importlib.import_module('session' if request['experiment'] == 'JC-base' else 'suite_session')
    helper.prepare_run(bundle, root)
    helper.save(previous, request)
    helper.save(root / 'custom-config.json', config)
    (root / 'custom-runner.py').write_bytes(runner_bytes)
    manifest = json.loads((root / 'data/manifest.json').read_text())
    env = helper.runtime_environment(args.python.resolve().parent.parent)
    deadline = time.monotonic() + max(0, args.seconds - 900)
    status = dict(state='starting', protocol=config['protocol'], experiment=request['experiment'],
                  epochs=config['epochs'], classifier_epochs=config['classifier_epochs'])
    archive = root.parent / (root.parent.name + '-results.zip')

    def publish(**values):
        status.update(values)
        helper.save(root / 'session-status.json', status)

    common = [str(args.python), str(root / 'code/experiment.py'), '--config',
              str(root / 'custom-config.json'), '--data', str(root / 'data'), '--device', 'gpu']
    try:
        for kind in ('gan', 'classifier'):
            expected_steps(config, manifest, kind)
        publish(phase='preflight')
        if not helper.execute([str(args.python), str(root / 'code/kaggle/preflight.py'), '--device', 'gpu',
                               '--report', str(root / 'preflight.json')], root / 'preflight.log', env, deadline):
            publish(state='paused', reason='session_budget')
            return
        for kind in ('gan', 'classifier'):
            stage = root / kind
            if stage_complete(stage, config, manifest, kind):
                continue
            remaining = deadline - time.monotonic()
            if remaining < 300:
                publish(state='paused', reason='session_budget')
                return
            if shutil.disk_usage(root).free < 3 * 1024**3:
                publish(state='paused', reason='disk_space_less_than_3GiB')
                return
            command = common + ['train', '--kind', kind, '--run-dir', str(stage),
                                '--max-seconds', str(remaining - 180)]
            if stage.exists():
                if (stage / 'checkpoints/checkpoint').exists():
                    command.append('--resume')
                else:
                    stage.rename(stage.with_name(stage.name + '-interrupted-' + str(time.time_ns())))
            publish(state='running', phase=kind)
            ready = helper.execute(command, root / (kind + '.log'), env, deadline, stage / 'status.json')
            if not ready or not stage_complete(stage, config, manifest, kind):
                publish(state='paused', reason='session_budget')
                return
        reports = {}
        for epoch in config['evaluate_epochs']:
            stage = root / ('evaluation-epoch-{:02d}'.format(epoch))
            report_path = stage / 'report.json'
            if not report_path.exists():
                if deadline - time.monotonic() < 600:
                    publish(state='paused', reason='session_budget')
                    return
                if stage.exists():
                    stage.rename(stage.with_name(stage.name + '-interrupted-' + str(time.time_ns())))
                publish(state='running', phase='evaluate', epoch=epoch)
                ready = helper.execute(common + ['evaluate', '--run-dir', str(stage),
                    '--gan-checkpoint', str(root / 'gan' / ('epoch-{:02d}'.format(epoch))),
                    '--classifier-checkpoint', str(root / 'classifier' /
                                                  ('epoch-{:02d}'.format(config['classifier_epochs'])))],
                    root / (stage.name + '.log'), env, deadline)
                if not ready:
                    publish(state='paused', reason='session_budget')
                    return
            report = json.loads(report_path.read_text())
            report.update(protocol=config['protocol'], paper_reference=None,
                          paper_caveat='Custom settings; classifier trained for this run. Not a registered paper result.')
            helper.save(report_path, report)
            reports[str(epoch)] = report
        helper.save(root / 'results.json', dict(protocol=config['protocol'], config=config, reports=reports,
                    classifier='Trained from scratch for this custom run; no historical classifier reused.'))
        publish(state='complete', phase='evaluated')
    except BaseException as error:
        publish(state='failed', error=repr(error))
        raise
    finally:
        helper.export_results(root).replace(archive)
        print('Saved custom results:', archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'python', 'output', 'request'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=36000)
    run(parser.parse_args())
