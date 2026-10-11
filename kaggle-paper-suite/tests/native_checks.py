"""Bounded native TF1.4 CPU checks on synthetic binary fixtures, no real training."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment as runner
from tf_compat import tf, build_info, gpu_details, gpu_devices
from model import cyclegan
from style_classifier import Classifer
from suite_plan import configuration
runner.np, runner.tf, runner.cyclegan, runner.Classifer = np, tf, cyclegan, Classifer
runner.build_info, runner.gpu_details, runner.gpu_devices = build_info, gpu_details, gpu_devices
OUT = Path(os.path.realpath(sys.argv[1]));OUT.mkdir(parents=True, exist_ok=False)
assert tf.__version__ == '1.4.0' and sys.version_info[:3] == (3,5,4)


def fixture(config):
    rng=np.random.RandomState(123)
    arrays={}
    for side in ['a','b']:
        for split,n in [('train',8),('test',5)]:
            arrays[config['dataset_'+side]+'_'+split]=rng.rand(n,64,84,1)<.06
    arrays['JCP_mixed']=rng.rand(24,64,84,1)<.05
    return SimpleNamespace(arrays=arrays,manifest={'identity':'synthetic-tests-only'})


def invoke(config, path, kind='gan', steps=0, resume=False):
    tf.reset_default_graph()
    args=SimpleNamespace(kind=kind,device='cpu',resume=resume,max_steps=steps,max_seconds=0)
    with (OUT/(path.name+'-invocation.log')).open('a') as log, contextlib.redirect_stdout(log):
        runner.train(args,config,fixture(config),path)


def compare(left,right):
    a=tf.train.NewCheckpointReader(runner.checkpoint_prefix(left/'checkpoints'))
    b=tf.train.NewCheckpointReader(runner.checkpoint_prefix(right/'checkpoints'))
    names=a.get_variable_to_shape_map()
    assert names==b.get_variable_to_shape_map()
    maximum=0.
    for name in names:
        maximum=max(maximum,float(np.max(np.abs(a.get_tensor(name)-b.get_tensor(name)))))
    assert maximum==0.,maximum
    return {'variables':len(names),'max_absolute_difference':maximum,'step':int(a.get_tensor('completed_steps')),
            'replay_count':int(a.get_tensor('replay_count')) if 'replay_count' in names else None}


results={'runtime':build_info(),'device':'CPU','real_data_training':False,'resume':{},'evaluation':{}}
for run_id in ['CP-base','JC-partial','JP-full']:
    config=configuration(run_id)
    config.update(ngf=2,ndf=2,batch_size=2,epochs=3,classifier_epochs=3,decay_start_epoch=1,
                  evaluate_epochs=[2,3],replay_pool_size=2 if config['model']!='base' else 0,
                  checkpoint_every_steps=2,report_every_steps=1,classifier_robustness_sigmas=[0.,.5])
    first,second=OUT/(run_id+'-continuous'),OUT/(run_id+'-resumed')
    invoke(config,first,steps=7)
    invoke(config,second,steps=3)
    # Simulate an unsaved step tail left by a crash.
    with (second/'steps.jsonl').open('a') as f:f.write(json.dumps({'step':9999})+'\n')
    invoke(config,second,steps=4,resume=True)
    results['resume'][run_id]=compare(first,second)
    records=[json.loads(line) for line in (second/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records]==list(range(1,8))
    if config['model']!='base':
        assert all(abs(r['d_loss']-r['d_base_loss']-r['d_mixed_loss'])<1e-5 for r in records)
    # Finish tiny 3-epoch fixture; exercises epoch checkpoints and final partial test batches.
    invoke(config,second,steps=0,resume=True)
    classifier=OUT/(run_id+'-classifier')
    invoke(config,classifier,kind='classifier',steps=0)
    tf.reset_default_graph()
    args=SimpleNamespace(device='cpu',limit=0,gan_checkpoint=str(second/'epoch-03'),classifier_checkpoint=str(classifier/'epoch-03'))
    evaluation=OUT/(run_id+'-evaluation')
    with (OUT/(run_id+'-evaluation.log')).open('w') as log,contextlib.redirect_stdout(log):
        runner.evaluate(args,config,fixture(config),evaluation)
    report=json.loads((evaluation/'report.json').read_text())
    assert report['labels']==config['genres'] and report['pair']==config['pair'] and report['model']==config['model']
    assert all(report['binary_legacy_pipeline'][d]['n']==5 for d in ['AtoB','BtoA'])
    results['evaluation'][run_id]={'phrases_per_direction':5,'midi_files':len(list(evaluation.glob('*.mid'))),'labels':report['labels']}
(OUT/'native-verification.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
