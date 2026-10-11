"""Bounded CPU inference verifies compatibility with the imported JC classifier."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import contextlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tf_compat import tf
import experiment
from style_classifier import Classifer
from suite_plan import configuration
experiment.tf=tf
bundle=Path(sys.argv[1]);original=Path(sys.argv[2]);output=Path(sys.argv[3])
output.parent.mkdir(parents=True,exist_ok=True)
settings=configuration('JC-full')
options=experiment.legacy_options(settings,16,output.parent)
session_config=tf.ConfigProto(device_count={'GPU':0},intra_op_parallelism_threads=4,inter_op_parallelism_threads=2)
results={}
with tf.Session(config=session_config) as sess:
    with output.with_suffix('.log').open('w') as log,contextlib.redirect_stdout(log):model=Classifer(sess,options)
    checkpoint=experiment.checkpoint_prefix(bundle/'baseline/classifier/epoch-30')
    tf.train.Saver(var_list=model.d_vars).restore(sess,checkpoint)
    for label,(key,direction) in enumerate([('JC_J_test','AtoB'),('JC_C_test','BtoA')]):
        array=np.load(str(bundle/'data'/(key+'.npy')),mmap_mode='r')
        probabilities=np.concatenate([sess.run(model.test_result_softmax,{model.test_midi:array[i:i+32].astype(np.float32)*2-1}) for i in range(0,len(array),32)])
        reference=np.load(str(original/'evaluation-epoch-30'/(direction+'-binary_legacy_pipeline-probabilities.npy')))[:,0]
        difference=float(np.max(np.abs(probabilities-reference)))
        assert difference<.001,difference
        results[key]={'n':len(array),'correct':int((probabilities.argmax(1)==label).sum()),'max_probability_difference_cpu_vs_saved_gpu':difference}
results['device']='CPU';results['trained_updates']=0
output.write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
