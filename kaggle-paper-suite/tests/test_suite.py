import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import shutil
from unittest.mock import patch
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'kaggle'))
from suite_plan import configuration, REMAINING, RUNS, PAIRS, validate_config
from replay import ReplayPool
from suite_data import mixed_training
import suite_session as session
from experiment import trim_checkpoint_log


class SuiteTests(unittest.TestCase):
    def test_paper_selected_matrix(self):
        expected = {'JC-partial': 0., 'JC-full': .01, 'CP-base': .1, 'CP-partial': 1.,
                    'CP-full': 1., 'JP-base': .01, 'JP-partial': .01, 'JP-full': 0.}
        self.assertEqual(set(REMAINING), set(expected))
        for run, sigma in expected.items():
            c = configuration(run)
            self.assertEqual(c['sigma_d'], sigma)
            validate_config(c)
        self.assertEqual(len(RUNS), 9)

    def test_labels_and_mixed_membership(self):
        a, b, third = np.ones((2, 1)), np.ones((2, 1))*2, np.ones((6, 1))*3
        data = type('Data', (), {'arrays': {'JCP_mixed': third}})()
        self.assertIsNone(mixed_training(data, configuration('CP-base'), a, b))
        np.testing.assert_array_equal(mixed_training(data, configuration('CP-partial'), a, b), np.concatenate([a,b]))
        self.assertIs(mixed_training(data, configuration('CP-full'), a, b), third)
        c = configuration('CP-full')
        self.assertEqual(c['genres'], ['Classic', 'Pop'])
        c['dataset_a'] = 'JC_J'
        with self.assertRaises(ValueError): validate_config(c)

    def test_replay_restore_after_history_is_used(self):
        uninterrupted, resumed = ReplayPool(2, (1,)), ReplayPool(2, (1,))
        used_old = False
        for step in range(20):
            a, b = np.array([step], np.float32), np.array([-step], np.float32)
            expected = uninterrupted.query(a,b,19+step)
            if step == 7:
                resumed = ReplayPool(2, (1,))
                resumed.values = stored.copy();resumed.count = count
            actual = resumed.query(a,b,19+step)
            np.testing.assert_array_equal(actual, expected)
            used_old |= not np.array_equal(expected[0], a)
            stored, count = resumed.values.copy(), resumed.count
        self.assertTrue(used_old)

    def test_saved_code_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);(root/'code').mkdir();(root/'code/x').write_text('original')
            session.save(root/'suite-manifest.json', {'protocol':session.PROTOCOL,'immutable_files':{'code/x':session.sha(root/'code/x')}})
            session.verify_saved(root)
            (root/'code/x').write_text('changed')
            with self.assertRaises(ValueError):session.verify_saved(root)
            with self.assertRaises(ValueError):session.safe_path(root, '../outside')

    def test_resume_copy_then_verify(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);old=root/'old';old.mkdir();(old/'code').mkdir();(old/'code/x').write_text('frozen')
            session.save(old/'suite-manifest.json',{'protocol':session.PROTOCOL,'immutable_files':{'code/x':session.sha(old/'code/x')}})
            session.prepare_run(root/'unused',root/'new',old)
            self.assertEqual((root/'new/code/x').read_text(),'frozen')
            with self.assertRaises(ValueError):session.prepare_run(root/'unused',root/'new',old)

    def test_completion_needs_step_count_and_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp);session.save(stage/'complete.json',{'state':'complete','completed_steps':21030})
            with self.assertRaises(ValueError):session.stage_complete(stage,'JC','gan')
            (stage/'epoch-30').mkdir()
            for ext in ['.index','.data-00000-of-00001']:(stage/'epoch-30'/('model-21030'+ext)).write_bytes(b'fixture')
            self.assertTrue(session.stage_complete(stage,'JC','gan'))
            with self.assertRaises(ValueError):session.stage_complete(stage,'CP','gan')

    def test_export_preserves_incomplete_replay_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'suite';root.mkdir()
            for name in ['done','pending']:
                stage=root/'runs'/name/'gan';(stage/'checkpoints').mkdir(parents=True)
                (stage/'checkpoints/model.data').write_bytes(b'fixture replay and weights')
            stage=root/'runs/done/gan';(stage/'epoch-30').mkdir();(stage/'epoch-30/model.data').write_bytes(b'final')
            session.save(stage/'complete.json',{'state':'complete'})
            with zipfile.ZipFile(session.export_results(root)) as archive:
                names=archive.namelist()
                self.assertIn('suite/runs/pending/gan/checkpoints/model.data',names)
                self.assertIn('suite/runs/done/gan/epoch-30/model.data',names)
                self.assertNotIn('suite/runs/done/gan/checkpoints/model.data',names)

    def test_pending_is_not_zero_or_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            self.assertFalse(session.collect(root))
            result=json.loads((root/'results.json').read_text())
            self.assertEqual(len(result['rows']),18)
            self.assertTrue(all(r['state']=='pending' and 'strength_argmax' not in r for r in result['rows']))

    def test_smoke_never_enters_tables(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            session.save(root/'runs/JC-full/evaluation-epoch-20/report.json',{'limited_smoke':True})
            with self.assertRaises(ValueError):session.collect(root)

    def test_crash_log_tail_is_preserved_and_trimmed(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'steps.jsonl'
            path.write_text('{"step":1}\n{"step":2}\n{"step":3}\n{"step":')
            trim_checkpoint_log(path,2)
            self.assertEqual([json.loads(s)['step'] for s in path.read_text().splitlines()],[1,2])
            self.assertEqual(len(list(path.parent.glob('*.superseded-*'))),1)
            path.write_text('broken\n{"step":2}\n')
            with self.assertRaises(ValueError):trim_checkpoint_log(path,2)

    def test_expired_budget_does_not_spawn(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(session.subprocess,'Popen') as spawn:
            self.assertFalse(session.execute(['unused'],Path(temp)/'log',{},session.time.monotonic()-1))
            spawn.assert_not_called()

    def test_generated_notebook_and_configs(self):
        notebook=json.loads((ROOT/'kaggle/cyclegan_paper_suite.ipynb').read_text(encoding='utf-8'))
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':compile(''.join(cell['source']), 'cell'+str(i), 'exec')
        for run in REMAINING:
            self.assertEqual(json.loads((ROOT/'kaggle/configs'/(run+'.json')).read_text()),configuration(run))

    def test_queue_runs_eight_gans_two_classifiers_and_resumes_without_retraining(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'suite';(root/'code/kaggle').mkdir(parents=True)
            shutil.copy2(ROOT/'kaggle/suite_session.py',root/'code/kaggle/suite_session.py')
            def report(entry):
                notes={k:0 for k in ['silent_count','note_density','note_precision','note_recall','note_f1']}
                scores={d:{'n':PAIRS[entry['pair']]['test'],'successful_fraction':.5,'successful_count':100,
                    'legacy_source_classification_rate':[.8,.3,.8],'mean_source_probability':[.79,.31,.78],
                    'transfer_notes':notes,'cycle_notes':notes} for d in ['AtoB','BtoA']}
                scores.update(bidirectional_legacy_accuracy_derived_strength=.5,bidirectional_signed_probability_strength_all=.4)
                return dict(pair=entry['pair'],model=entry['model'],sigma_d=entry['sigma_d'],limited_smoke=False,
                            binary_legacy_pipeline=scores,classifier_robustness={'0.0':.8})
            for e in [20,30]:session.save(root/f'baseline/evaluation-epoch-{e}/report.json',report(RUNS[0]))
            dispatched=[]
            def execute(command,*args,**kwargs):
                if 'train' in command:
                    kind=command[command.index('--kind')+1];stage=Path(command[command.index('--run-dir')+1])
                    pair=stage.name if kind=='classifier' else stage.parent.name.split('-')[0]
                    steps=(PAIRS[pair]['train']*(2 if kind=='classifier' else 1)//16)*30
                    session.save(stage/'complete.json',{'state':'complete','completed_steps':steps})
                    (stage/'epoch-30').mkdir()
                    for ext in ['.index','.data-00000-of-00001']:(stage/'epoch-30'/('model-'+str(steps)+ext)).write_bytes(b'fixture')
                    dispatched.append((kind,pair,stage.parent.name))
                elif 'evaluate' in command:
                    stage=Path(command[command.index('--run-dir')+1]);entry=next(r for r in RUNS if r['id']==stage.parent.name)
                    session.save(stage/'report.json',report(entry))
                return True
            flock=SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *args:None)
            args=SimpleNamespace(output=root,bundle=root/'unused',resume=None,python=Path('/fake/env/bin/python'),
                                 seconds=36000,device='gpu',mode='train',runs=REMAINING)
            with patch.dict(sys.modules,{'fcntl':flock}),patch.object(session,'prepare_run'),patch.object(session,'execute',execute),patch.object(session,'export_results',return_value=Path('fixture.zip')):
                session.run(args)
                self.assertEqual(sum(k=='gan' for k,*_ in dispatched),8)
                self.assertEqual([p for k,p,_ in dispatched if k=='classifier'],['CP','JP'])
                self.assertEqual(json.loads((root/'session-status.json').read_text())['state'],'complete')
                dispatched.clear();session.run(args)
                self.assertFalse(dispatched)


if __name__ == '__main__':
    unittest.main(verbosity=2)
