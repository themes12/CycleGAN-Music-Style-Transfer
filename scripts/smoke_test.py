"""Exercise real training, checkpoint restore, MIDI export and classifier evaluation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_data import prepare


def invoke(command, log, expect_failure=False):
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=str(ROOT), stdout=stream,
                                stderr=subprocess.STDOUT)
    if expect_failure:
        text = log.read_text(encoding='utf-8')
        if result.returncode == 0 or 'No checkpoint for run_id' not in text:
            raise RuntimeError('Missing-checkpoint check failed; see {}'.format(log))
    elif result.returncode:
        tail = '\n'.join(log.read_text(encoding='utf-8').splitlines()[-25:])
        raise RuntimeError('Command failed; see {}\n{}'.format(log, tail))


def verify(output, run_id, model, classifier=False):
    name = 'JC_J2JC_C_{}_{}_0.0'.format(run_id, model)
    checkpoint = output / 'checkpoint' / name
    if not (checkpoint / 'checkpoint').is_file() or not list(checkpoint.glob('*.index')):
        raise RuntimeError('No saved checkpoint in {}'.format(checkpoint))
    arrays = []
    for direction in ('AtoB', 'BtoA'):
        directory = output / 'test' / name / direction
        for variant in ('origin', 'transfer', 'cycle'):
            files = list((directory / 'npy' / variant).glob('*.npy'))
            if len(files) != 1:
                raise RuntimeError('Expected one {} phrase in {}'.format(variant, directory))
            array = np.load(str(files[0]), allow_pickle=False)
            if array.shape != (1, 64, 84, 1) or array.dtype != np.bool_:
                raise RuntimeError('Invalid generated piano roll: {}'.format(files[0]))
            arrays.append(array)
        midis = list((directory / 'mid').glob('*.mid'))
        if len(midis) != 3:
            raise RuntimeError('Missing generated MIDI files in {}'.format(directory))
        for midi in midis:
            pretty_midi.PrettyMIDI(str(midi))
        if classifier:
            ranking = directory / 'mid_attach_prob' / 'Rankings_{}.txt'.format(direction)
            if not ranking.is_file() or len(ranking.read_text().splitlines()) != 2:
                raise RuntimeError('Missing classifier ranking: {}'.format(ranking))
    metadata = list((output / 'logs' / name).glob('metadata_*.json'))
    if len(metadata) != 3 or any(json.loads(path.read_text())['status'] != 'completed' for path in metadata):
        raise RuntimeError('Training/test metadata does not report completion')
    return arrays


def verify_checkpoint_repeatability(output, model):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf
    checkpoints = output / 'checkpoint'
    original = checkpoints / 'JC_J2JC_C_smoke-{}_{}_0.0'.format(model, model)
    repeated = checkpoints / 'JC_J2JC_C_smoke-repeat_{}_0.0'.format(model)
    first = tf.train.load_checkpoint(str(next(original.glob('*.index')).with_suffix('')))
    second = tf.train.load_checkpoint(str(next(repeated.glob('*.index')).with_suffix('')))
    variables = first.get_variable_to_shape_map()
    if variables != second.get_variable_to_shape_map() or any(
            not np.array_equal(first.get_tensor(name), second.get_tensor(name)) for name in variables):
        raise RuntimeError('Seeded CPU runs produced different checkpoint weights or optimizer state')
    return len(variables)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', default=str(ROOT / 'Dataset (use directly).zip'))
    parser.add_argument('--output', help='new directory for data, checkpoints and test reports')
    parser.add_argument('--models', nargs='+', choices=('base', 'partial', 'full'),
                        default=['base', 'partial', 'full'])
    args = parser.parse_args()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output = Path(args.output or ROOT / 'dist' / ('smoke-' + stamp)).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be a new or empty directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest = prepare(args.archive, output / 'datasets', limit=4)
    common = ['--data_root', str(output / 'datasets'), '--checkpoint_dir', str(output / 'checkpoint'),
              '--sample_dir', str(output / 'samples'), '--test_dir', str(output / 'test'),
              '--log_dir', str(output / 'logs'), '--device', 'cpu', '--epoch', '1',
              '--batch_size', '2', '--train_size', '2', '--test_size', '1',
              '--ngf', '8', '--ndf', '8', '--sigma_d', '0', '--sigma_c', '0',
              '--seed', '19', '--print_freq', '2']
    first_arrays = None
    for index, model in enumerate(args.models):
        run_id = 'smoke-' + model
        print('Testing {}: one training batch, both directions, checkpoint restore'.format(model), flush=True)
        command = [sys.executable, str(ROOT / 'scripts' / 'run_experiment.py')] + common + [
            '--run_id', run_id, '--model', model]
        if index == 0:
            command.append('--evaluate_classifier')
        invoke(command, output / (model + '.log'))
        arrays = verify(output, run_id, model, classifier=index == 0)
        if index == 0:
            first_arrays = arrays
    model = args.models[0]
    print('Checking repeatability with seed 19 in a separate run', flush=True)
    invoke([sys.executable, str(ROOT / 'scripts' / 'run_experiment.py')] + common + [
        '--run_id', 'smoke-repeat', '--model', model], output / 'repeat.log')
    repeated = verify(output, 'smoke-repeat', model)
    if any(not np.array_equal(left, right) for left, right in zip(first_arrays, repeated)):
        raise RuntimeError('Seeded CPU runs produced different generated phrases')
    tensor_count = verify_checkpoint_repeatability(output, model)
    print('Checking that a missing checkpoint fails instead of exporting random weights', flush=True)
    invoke([sys.executable, str(ROOT / 'main.py')] + common + [
        '--run_id', 'missing', '--model', model, '--phase', 'test'],
        output / 'missing-checkpoint.log', expect_failure=True)
    report = {'status': 'passed', 'models': args.models, 'classifier': 'both directions',
              'repeatability': 'identical generated arrays on CPU with seed 19',
              'identical_checkpoint_tensors': tensor_count,
              'missing_checkpoint': 'failed as expected', 'dataset': manifest,
              'note': 'Small networks and a data subset; this checks execution, not model quality.'}
    (output / 'smoke-report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('PASS: {}'.format(output / 'smoke-report.json'), flush=True)


if __name__ == '__main__':
    main()
