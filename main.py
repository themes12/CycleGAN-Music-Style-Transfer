import argparse
import os
import random

from experiment import configure_device, safe_name, update_metadata, utc_now, write_metadata


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def boolean(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('use true or false')


def name(value):
    try:
        return safe_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error))


def build_parser():
    parser = argparse.ArgumentParser(description='CycleGAN symbolic music experiments (TensorFlow 1).')
    parser.add_argument('--dataset_dir', default='JAZZ2ROCK', help='legacy option; use --data_root')
    parser.add_argument('--data_root', default='./datasets', help='directory containing genre train/test folders')
    parser.add_argument('--dataset_A_dir', type=name, default='JC_J', help='domain A folder under data_root')
    parser.add_argument('--dataset_B_dir', type=name, default='JC_C', help='domain B folder under data_root')
    parser.add_argument('--run_id', type=name, default='default', help='stable experiment name shared by train and test')
    parser.add_argument('--seed', type=int, default=19, help='Python, NumPy and TensorFlow random seed')
    parser.add_argument('--device', default='auto', help='auto, cpu, or GPU indices such as 0')
    parser.add_argument('--num_threads', type=positive_int, default=1, help='TensorFlow CPU worker threads')
    parser.add_argument('--epoch', type=positive_int, default=100)
    parser.add_argument('--epoch_step', type=int, default=10, help='epoch at which learning-rate decay starts')
    parser.add_argument('--batch_size', type=positive_int, default=16)
    parser.add_argument('--train_size', type=positive_int, default=100000000, help='maximum training phrases per domain per epoch')
    parser.add_argument('--test_size', type=positive_int, default=None, help='maximum test phrases per domain (default: all)')
    parser.add_argument('--load_size', type=positive_int, default=286)
    parser.add_argument('--fine_size', type=positive_int, default=128)
    parser.add_argument('--time_step', type=positive_int, default=64)
    parser.add_argument('--pitch_range', type=positive_int, default=84)
    parser.add_argument('--ngf', type=positive_int, default=64)
    parser.add_argument('--ndf', type=positive_int, default=64)
    parser.add_argument('--input_nc', type=positive_int, default=1)
    parser.add_argument('--output_nc', type=positive_int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--which_direction', choices=('AtoB', 'BtoA'), default='AtoB')
    parser.add_argument('--phase', choices=('train', 'test'), default='train')
    parser.add_argument('--save_freq', type=positive_int, default=1000)
    parser.add_argument('--print_freq', type=positive_int, default=100, help='generate training samples every N updates')
    parser.add_argument('--continue_train', type=boolean, nargs='?', const=True, default=False)
    parser.add_argument('--checkpoint_dir', default='./checkpoint')
    parser.add_argument('--sample_dir', default='./samples')
    parser.add_argument('--test_dir', default='./test')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--L1_lambda', type=float, default=10.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    for option in ('use_midi_G', 'use_midi_D', 'use_lsgan'):
        parser.add_argument('--' + option, type=boolean, nargs='?', const=True, default=False)
    parser.add_argument('--max_size', type=int, default=50)
    parser.add_argument('--sigma_c', type=float, default=1.0)
    parser.add_argument('--sigma_d', type=float, default=1.0)
    parser.add_argument('--model', choices=('base', 'partial', 'full'), default='base')
    parser.add_argument('--type', choices=('cyclegan', 'classifier'), default='cyclegan')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error('--seed must be between 0 and 4294967295')
    if args.epoch_step < 0 or args.sigma_c < 0 or args.sigma_d < 0:
        parser.error('epoch_step and noise sigmas must be nonnegative')
    try:
        configure_device(args.device)
    except ValueError as error:
        parser.error(str(error))

    import numpy as np
    import tensorflow as tf
    if not tf.__version__.startswith('1.'):
        raise RuntimeError('This model requires TensorFlow 1. Use the Docker image or environment.yml.')
    from model import cyclegan
    from style_classifier import Classifer

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    for directory in (args.checkpoint_dir, args.sample_dir, args.test_dir, args.log_dir):
        os.makedirs(directory, exist_ok=True)
    path, record = write_metadata(args, tf.__version__)
    config = tf.ConfigProto(allow_soft_placement=True,
                           intra_op_parallelism_threads=args.num_threads,
                           inter_op_parallelism_threads=args.num_threads)
    config.gpu_options.allow_growth = True
    try:
        with tf.Session(config=config) as session:
            model = cyclegan(session, args) if args.type == 'cyclegan' else Classifer(session, args)
            model.train(args) if args.phase == 'train' else model.test(args)
        record['status'] = 'completed'
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = str(error)
        raise
    finally:
        record['finished_at'] = utc_now()
        update_metadata(path, record)
        print('Experiment metadata:', path)


if __name__ == '__main__':
    main()
