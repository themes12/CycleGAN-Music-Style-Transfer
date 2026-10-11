"""Config and orchestration tests; no TensorFlow/GPU workload is launched."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

from scripts import vast_custom as custom
from scripts import vast_launcher as launcher

ROOT = Path(__file__).resolve().parents[1]

# A subprocess stand-in exercises the real supervisor's pause/resume and export
# contracts without replacing its immutable-file handling or using a paid GPU.
ENGINE = '''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('command')
for key in ('kind', 'config', 'data', 'device', 'run-dir', 'max-seconds', 'gan-checkpoint', 'classifier-checkpoint'):
    p.add_argument('--' + key)
p.add_argument('--resume', action='store_true')
a = p.parse_args()
c = json.loads(Path(a.config).read_text())
r = Path(a.run_dir)
r.mkdir(parents=True, exist_ok=True)
if a.command == 'train':
    attempts_file = r / 'attempts.json'
    attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
    if attempts:
        assert a.resume, 'Restarted a paused stage instead of resuming'
        assert json.loads((r / 'config.json').read_text()) == c
    attempts.append(a.resume)
    attempts_file.write_text(json.dumps(attempts))
    (r / 'config.json').write_text(json.dumps(c))
    (r / 'checkpoints').mkdir(exist_ok=True)
    (r / 'checkpoints/checkpoint').write_text('saved state')
    if a.kind == 'gan' and len(attempts) == 1:
        (r / 'status.json').write_text(json.dumps({'state': 'paused', 'epoch': 1}))
    else:
        count = 8 if a.kind == 'gan' else 16
        epochs = c['epochs' if a.kind == 'gan' else 'classifier_epochs']
        steps = count // c['batch_size'] * epochs
        for epoch in c['evaluate_epochs'] if a.kind == 'gan' else [epochs]:
            checkpoint = r / ('epoch-%02d' % epoch)
            checkpoint.mkdir(exist_ok=True)
            for suffix in ('.index', '.data-00000-of-00001'):
                (checkpoint / ('model-' + str(steps) + suffix)).write_text('weights')
        (r / 'complete.json').write_text(json.dumps({'state': 'complete', 'completed_steps': steps}))
else:
    assert Path(a.gan_checkpoint).name == 'epoch-%02d' % c['epochs']
    assert Path(a.classifier_checkpoint).name == 'epoch-%02d' % c['classifier_epochs']
    (r / 'report.json').write_text(json.dumps({'paper_reference': {'old': True}, 'protocol': c['protocol']}))
'''


class CustomSettingsTests(unittest.TestCase):
    def test_custom_schedule_and_validation(self):
        bundle = ROOT / 'kaggle-jc-base'
        short = custom.configuration(bundle, 'JC-base', {'epochs': 2})
        self.assertEqual(short['evaluate_epochs'], [2])
        self.assertEqual(short['decay_start_epoch'], 0)
        self.assertEqual(short['classifier_epochs'], 30)
        long = custom.configuration(bundle, 'JC-base', {'epochs': 50, 'seed': 42})
        self.assertEqual(long['evaluate_epochs'], [20, 30, 50])
        for settings in ({'epochs': True}, {'epochs': 0}, {'batch_size': 1.5},
                         {'learning_rate': float('nan')}, {'sigma_d': -1}, {'seed': -1},
                         {'beta1': 1}, {'epochs': 3, 'decay_start_epoch': 3},
                         {'evaluate_epochs': [99]}, {'typo': 3}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                custom.configuration(bundle, 'JC-base', settings)

    def test_output_isolation_and_original_presets(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prefix = workspace / 'env'
            for experiment in launcher.EXPERIMENTS:
                bundle = ROOT / ('kaggle-jc-base' if experiment == 'JC-base' else 'kaggle-paper-suite')
                original, original_root, _ = launcher.training_command(experiment, workspace, bundle, prefix)
                self.assertIn('--mode', original)
                options = {'epochs': 50, 'batch_size': 8, 'sigma_d': .2}
                first, first_root, _ = launcher.training_command(experiment, workspace, bundle, prefix, options)
                second, second_root, _ = launcher.training_command(experiment, workspace, bundle, prefix, dict(reversed(list(options.items()))))
                _, changed_root, _ = launcher.training_command(experiment, workspace, bundle, prefix, dict(options, seed=42))
                self.assertEqual(first_root, second_root)
                self.assertNotEqual(first_root, original_root)
                self.assertNotEqual(first_root, changed_root)
                self.assertNotIn('--runs', first)  # One custom experiment, never the fixed eight-run queue.
                request = json.loads(Path(first[first.index('--request') + 1]).read_text())
                self.assertEqual(request['config']['epochs'], 50)
                self.assertFalse(first_root.exists())

    def test_nonstandard_completion_requires_final_weights(self):
        config = dict(epochs=2, classifier_epochs=3, batch_size=3)
        manifest = {'splits': {'JC_J_train': {'count': 8}, 'JC_C_train': {'count': 8}}}
        self.assertEqual(custom.expected_steps(config, manifest, 'gan'), 4)
        self.assertEqual(custom.expected_steps(config, manifest, 'classifier'), 15)
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            (stage / 'complete.json').write_text(json.dumps({'state': 'complete', 'completed_steps': 4}))
            with self.assertRaises(ValueError):
                custom.stage_complete(stage, config, manifest, 'gan')

    def test_pause_resume_evaluation_and_export(self):
        for experiment in ('JC-base', 'CP-base'):
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                bundle = directory / 'input'
                (bundle / 'kaggle/configs').mkdir(parents=True)
                source = ROOT / ('kaggle-jc-base' if experiment == 'JC-base' else 'kaggle-paper-suite')
                helper = 'session.py' if experiment == 'JC-base' else 'suite_session.py'
                for relative in ('kaggle/' + helper, 'kaggle/bootstrap.py'):
                    (bundle / relative).write_bytes((source / relative).read_bytes())
                if experiment != 'JC-base':
                    (bundle / 'suite_plan.py').write_bytes((source / 'suite_plan.py').read_bytes())
                config_name = 'kaggle/config.json' if experiment == 'JC-base' else 'kaggle/configs/CP-base.json'
                (bundle / config_name).write_bytes((source / config_name).read_bytes())
                (bundle / 'experiment.py').write_text(ENGINE)
                (bundle / 'kaggle/preflight.py').write_text('pass\n')
                (bundle / 'data').mkdir()
                keys = ('JC_J', 'JC_C') if experiment == 'JC-base' else ('CP_C', 'CP_P')
                (bundle / 'data/manifest.json').write_text(json.dumps({'splits': {key + '_train': {'count': 8} for key in keys}}))
                files = {path.relative_to(bundle).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in bundle.rglob('*') if path.is_file()}
                (bundle / 'bundle-manifest.json').write_text(json.dumps({'protocol': 'native-tf14-paper-selected-nine-v1', 'files': files}))
                command, output, archive = launcher.training_command(experiment, directory, bundle, directory / 'env',
                    {'epochs': 2, 'classifier_epochs': 3, 'batch_size': 3})
                command[command.index('--python') + 1] = sys.executable
                # Speed up polling only; deadlines, subprocesses and outputs are real.
                wrapper = 'import time,runpy,sys; sleep=time.sleep; time.sleep=lambda n:sleep(min(n,.01)); path=sys.argv.pop(1); runpy.run_path(path,run_name="__main__")'
                command = [sys.executable, '-c', wrapper] + command[2:]
                for expected in ('paused', 'complete'):
                    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
                    self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                    state = json.loads((output / 'session-status.json').read_text())
                    self.assertEqual(state['state'], expected)
                self.assertEqual(json.loads((output / 'gan/attempts.json').read_text()), [False, True])
                self.assertEqual(json.loads((output / 'classifier/attempts.json').read_text()), [False])
                report = json.loads((output / 'evaluation-epoch-02/report.json').read_text())
                self.assertIsNone(report['paper_reference'])
                with zipfile.ZipFile(archive) as package:
                    self.assertIn('run/custom-run.json', package.namelist())
                    self.assertIn('run/classifier/epoch-03/model-15.index', package.namelist())
                    self.assertIsNone(package.testzip())


if __name__ == '__main__':
    unittest.main()
