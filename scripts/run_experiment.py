"""Train and test both directions, optionally evaluating a genre classifier."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import build_parser


def run_pipeline(args):
    excluded = {'phase', 'type', 'which_direction', 'evaluate_classifier', 'skip_train'}
    common = []
    for key, value in vars(args).items():
        if key not in excluded and value is not None:
            common.extend(['--' + key, str(value)])
    steps = []
    if not args.skip_train:
        steps.append(['--type', 'cyclegan', '--phase', 'train'])
    for direction in ('AtoB', 'BtoA'):
        steps.append(['--type', 'cyclegan', '--phase', 'test', '--which_direction', direction])
    if args.evaluate_classifier:
        if not args.skip_train:
            steps.append(['--type', 'classifier', '--phase', 'train'])
        for direction in ('AtoB', 'BtoA'):
            steps.append(['--type', 'classifier', '--phase', 'test', '--which_direction', direction])
    for index, step in enumerate(steps, 1):
        command = [sys.executable, str(ROOT / 'main.py')] + common + step
        print('Step {}/{}: {}'.format(index, len(steps), subprocess.list2cmdline(command)), flush=True)
        subprocess.run(command, cwd=str(ROOT), check=True)


def main():
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument('--evaluate_classifier', action='store_true')
    parser.add_argument('--skip_train', action='store_true', help='test existing CycleGAN/classifier checkpoints')
    args = parser.parse_args()
    try:
        run_pipeline(args)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)


if __name__ == '__main__':
    main()
