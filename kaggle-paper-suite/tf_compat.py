"""Use native TF1 when installed, otherwise the existing TF2 graph adapter."""
import tensorflow as _tensorflow

if _tensorflow.__version__.split('.')[0] == '1':
    tf = _tensorflow
else:
    tf = _tensorflow.compat.v1
    tf.disable_v2_behavior()


def gpu_devices():
    if hasattr(_tensorflow, 'config'):
        return _tensorflow.config.list_physical_devices('GPU')
    from tensorflow.python.client import device_lib
    return [d for d in device_lib.list_local_devices() if d.device_type == 'GPU']


def build_info():
    if hasattr(tf.sysconfig, 'get_build_info'):
        return tf.sysconfig.get_build_info()
    return {'tensorflow': tf.__version__, 'cuda_build': tf.test.is_built_with_cuda(),
            'runtime': 'native TensorFlow 1 (not compat.v1)'}


def gpu_details():
    if hasattr(_tensorflow, 'config'):
        return [_tensorflow.config.experimental.get_device_details(d) for d in gpu_devices()]
    return [{'name': d.name, 'description': d.physical_device_desc} for d in gpu_devices()]
