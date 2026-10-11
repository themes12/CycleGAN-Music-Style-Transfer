import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from experiment import configure_device, model_directory, piano_roll_files
from main import build_parser
from scripts.prepare_data import DOMAINS, prepare


class SharingTests(unittest.TestCase):
    def test_false_flags_remain_false(self):
        args = build_parser().parse_args(['--continue_train', 'False', '--use_lsgan', 'false'])
        self.assertFalse(args.continue_train)
        self.assertFalse(args.use_lsgan)

    def test_missing_gpu_environment_is_supported(self):
        with patch.dict('os.environ', {}, clear=True):
            configure_device('auto')
            configure_device('cpu')

    def test_run_name_cannot_escape_output_directory(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(['--run_id', '../outside'])

    def test_model_and_classifier_paths_use_the_same_run(self):
        args = build_parser().parse_args(['--run_id', 'shared-run', '--sigma_d', '0', '--sigma_c', '0'])
        self.assertEqual(model_directory(args), 'JC_J2JC_C_shared-run_base_0.0')
        args.type = 'classifier'
        self.assertEqual(model_directory(args), 'classifier_JC_J2JC_C_shared-run_0.0')

    def test_data_order_is_stable_and_ignores_unrelated_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'JC_J' / 'train'
            folder.mkdir(parents=True)
            for name in ('phrase_2.npy', 'readme.txt', 'phrase_1.npy'):
                (folder / name).touch()
            self.assertEqual([Path(path).name for path in piano_roll_files(root, 'JC_J', 'train')],
                             ['phrase_1.npy', 'phrase_2.npy'])
            with self.assertRaises(ValueError):
                piano_roll_files(root, 'JC_C', 'train')


class DatasetTests(unittest.TestCase):
    def archive(self, root, unsafe=False, invalid=False):
        archive = root / 'input.zip'
        with zipfile.ZipFile(str(archive), 'w') as output:
            for domain in DOMAINS:
                splits = (None,) if domain == 'JCP_mixed' else ('train', 'test')
                for split in splits:
                    for number in (2, 1):
                        filename = '{}/{}phrase_{}.npy'.format(domain, split + '/' if split else '', number)
                        array = np.zeros((1, 64, 84, 1), dtype=np.bool_)
                        array[0, number, 24, 0] = True
                        if invalid:
                            array = array[:, :32]
                        buffer = io.BytesIO()
                        np.save(buffer, array)
                        output.writestr(filename, buffer.getvalue())
            if unsafe:
                output.writestr('../escape.npy', b'not a phrase')
        return archive

    def test_subset_and_manifest_are_repeatable(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            archive = self.archive(root)
            first = prepare(archive, root / 'one', limit=1)
            second = prepare(archive, root / 'two', limit=1)
            self.assertEqual(first, second)
            self.assertTrue(all(count == 1 for count in first['counts'].values()))
            array = np.load(str(root / 'one' / 'JC_J' / 'train' / 'phrase_1.npy'))
            self.assertEqual(array.shape, (64, 84, 1))
            self.assertTrue(array[1, 24, 0])
            with self.assertRaises(ValueError):
                prepare(archive, root / 'one')

    def test_unsafe_archive_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, unsafe=True)
            with self.assertRaisesRegex(ValueError, 'Unsafe path'):
                prepare(archive, root / 'output')
            self.assertFalse((root / 'output').exists())
            self.assertFalse((root / 'escape.npy').exists())

    def test_bad_shape_and_wrong_checksum_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = self.archive(root, invalid=True)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                prepare(archive, root / 'output', expected_sha256='0' * 64)
            with self.assertRaisesRegex(ValueError, 'binary piano roll'):
                prepare(archive, root / 'output')


if __name__ == '__main__':
    unittest.main()
