"""Native TF1/CUDA/cuDNN check. Python 3.5 compatible; GPU required by default."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', required=True)
    p.add_argument('--device', choices=['gpu', 'cpu'], default='gpu',
                   help='CPU is only for local packaging verification, not Kaggle training')
    args = p.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if platform.python_version() != '3.5.4':
        raise RuntimeError('Expected Python 3.5.4, got ' + platform.python_version())
    import tensorflow as tf
    import numpy as np
    import pretty_midi
    import pkg_resources
    if tf.__version__ != '1.4.0' or not tf.test.is_built_with_cuda():
        raise RuntimeError('Expected tensorflow-gpu==1.4.0; CPU wheels/TF2 are not accepted')
    if np.__version__ != '1.14.2':
        raise RuntimeError('Expected original numpy==1.14.2')
    assert pretty_midi.__version__ == '0.2.8'
    # Video plotting imports moviepy and tries a retired ffmpeg download.
    # Training on prepared piano rolls does not use that optional code path.
    assert pkg_resources.get_distribution('pypianoroll').version == '0.1.3'
    cudnn = ctypes.CDLL('libcudnn.so.6')
    cudnn.cudnnGetVersion.restype = ctypes.c_size_t
    cudnn_version = int(cudnn.cudnnGetVersion())
    cuda = ctypes.CDLL('libcudart.so.8.0')
    cuda_version = ctypes.c_int()
    cuda_status = int(cuda.cudaRuntimeGetVersion(ctypes.byref(cuda_version)))
    cuda_package = json.loads((Path(sys.prefix) / 'conda-meta/cudatoolkit-8.0-3.json').read_text())
    if cuda_package['version'] != '8.0' or cudnn_version != 6021:
        raise RuntimeError('Wrong CUDA/cuDNN packages')
    if args.device == 'gpu' and (cuda_status != 0 or cuda_version.value != 8000):
        raise RuntimeError('Expected CUDA 8.0 and cuDNN 6.0.21; got runtime={}, cudnn={}'.format(
            cuda_version.value, cudnn_version))
    gpu_info = None
    if args.device == 'gpu':
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                                           '--format=csv,noheader']).decode().strip()
        if 'P100' not in gpu_info:
            raise RuntimeError('This notebook requires Kaggle P100. Got: ' + gpu_info +
                '. P100 retires 2026-09-15; a T4 replacement is not an accepted original-runtime test.')
    target = '/gpu:0' if args.device == 'gpu' else '/cpu:0'
    with tf.device(target):
        x = tf.constant(np.ones((1, 8, 8, 1), np.float32))
        kernel = tf.get_variable('probe_kernel', initializer=np.ones((3, 3, 1, 2), np.float32))
        y = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME')
        gradient = tf.gradients(tf.reduce_sum(y), kernel)[0]
        update = tf.train.AdamOptimizer(.0002).minimize(tf.reduce_mean(y * y))
    config = tf.ConfigProto(allow_soft_placement=False, intra_op_parallelism_threads=2,
                           inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        output, grad, _ = sess.run([y, gradient, update])
        assert np.isfinite(output).all() and np.isfinite(grad).all() and np.any(grad)
        devices = [d.name for d in sess.list_devices()]
    report = {'passed': True, 'python': platform.python_version(), 'tensorflow': tf.__version__,
              'numpy': np.__version__, 'cuda_runtime': cuda_version.value if cuda_status == 0 else None,
              'cuda_query_status': cuda_status, 'cuda_package': cuda_package['version'], 'cudnn': cudnn_version,
              'device': args.device, 'devices': devices, 'nvidia_smi': gpu_info,
              'training_device': target, 'convolution_backward_and_adam': 'passed',
              'native_tf1': True, 'cpu_only_packaging_check': args.device == 'cpu'}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
