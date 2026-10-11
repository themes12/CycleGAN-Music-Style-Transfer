"""Build the self-contained private Kaggle upload using modern Python + NumPy."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
from suite_plan import PROTOCOL, PAIRS, RUNS
from suite_data import file_sha

ROOT = Path(__file__).resolve().parent
CODE = ['model.py', 'module.py', 'ops.py', 'utils.py', 'write_midi.py', 'style_classifier.py',
        'tf_compat.py', 'experiment.py', 'experiment_metrics.py', 'experiment_health.py',
        'experiment_objectives.py', 'suite_plan.py', 'suite_data.py', 'replay.py', 'baseline-provenance.json',
        'KAGGLE.th.md', 'kaggle/bootstrap.py', 'kaggle/preflight.py', 'kaggle/suite_session.py',
        'kaggle/conda-linux-64.lock', 'kaggle/pip-bootstrap.lock', 'kaggle/requirements-tf14.lock',
        'kaggle/requirements-tf14.in', 'kaggle/cyclegan_paper_suite.ipynb',
        'build_bundle.py', 'generate_notebook.py', 'tests/test_suite.py',
        'tests/native_checks.py', 'tests/check_imported_classifier.py',
        'verification/verification.json', 'verification/native-verification.json',
        'verification/imported-classifier.json']


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def import_baseline(source, output):
    manifest = json.loads((source / 'run-manifest.json').read_text())
    if manifest['protocol'] != 'native-tf14-jazz-classic-base-v1':
        raise ValueError('Expected the completed native TF1.4 JC baseline')
    preflight = json.loads((source / 'preflight.json').read_text())
    if not preflight['passed'] or preflight['tensorflow'] != '1.4.0' or preflight['device'] != 'gpu':
        raise ValueError('Baseline must have passed native TF1.4 GPU preflight')
    for name, expected in manifest['code_sha256'].items():
        if file_sha(source / 'code' / name) != expected:
            raise ValueError('Baseline code changed: ' + name)
    for name in ['model.py', 'module.py', 'ops.py', 'utils.py', 'write_midi.py', 'style_classifier.py', 'tf_compat.py']:
        if file_sha(ROOT / name) != manifest['code_sha256'][name]:
            raise ValueError('Shared graph changed; do not reuse classifier without compatibility verification: ' + name)
    c = json.loads((source / 'gan/config.json').read_text())['config']
    if any(c[k] != v for k, v in dict(model='base', sigma_d=1., cycle_loss='legacy_l1', ngf=64, ndf=64,
                                     batch_size=16, seed=19, epochs=30, classifier_epochs=30).items()):
        raise ValueError('Baseline settings differ from the registered first cell')
    for kind, steps in [('gan', 21030), ('classifier', 42060)]:
        complete = json.loads((source / kind / 'complete.json').read_text())
        if complete['state'] != 'complete' or complete['completed_steps'] != steps:
            raise ValueError('Incomplete baseline ' + kind)
        copy(source / kind / 'complete.json', output / kind / 'complete.json')
        copy(source / kind / 'config.json', output / kind / 'config.json')
        copy(source / kind / 'runtime.json', output / kind / 'runtime.json')
    for name in ['run-manifest.json', 'preflight.json', 'results.json', 'gan/objective.json']:
        copy(source / name, output / name)
    for epoch in [20, 30]:
        name = 'evaluation-epoch-{}/report.json'.format(epoch)
        report = json.loads((source / name).read_text())
        if report['limited_smoke'] or any(report['binary_legacy_pipeline'][d]['n'] != 1125 for d in ['AtoB', 'BtoA']):
            raise ValueError('Baseline evaluation is incomplete')
        copy(source / name, output / name)
    for suffix in ['.index', '.data-00000-of-00001']:
        copy(source / ('classifier/epoch-30/model-42060' + suffix), output / ('classifier/epoch-30/model-42060' + suffix))
    (output / 'classifier/epoch-30/checkpoint').write_text('model_checkpoint_path: "model-42060"\nall_model_checkpoint_paths: "model-42060"\n')


def prepare_data(archive_path, output, baseline):
    output.mkdir()
    old = json.loads((baseline / 'data/manifest.json').read_text())
    manifest = {'source': archive_path.name, 'source_zip_sha256': file_sha(archive_path), 'splits': {}}
    content_hashes = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('.npy'))
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive names')
        groups = {}
        for pair in PAIRS.values():
            for domain in [pair['a'], pair['b']]:
                for mode in ['train', 'test']:
                    groups[domain + '_' + mode] = (domain + '/' + mode + '/', pair[mode])
        groups['JCP_mixed'] = ('JCP_mixed/', 33648)
        for key, (prefix, expected) in groups.items():
            members = [n for n in names if n.startswith(prefix)]
            if len(members) != expected:
                raise ValueError('Incomplete published split {}: {} != {}'.format(key, len(members), expected))
            target = output / (key + '.npy')
            array = np.lib.format.open_memmap(target, mode='w+', dtype=np.bool_, shape=(expected, 64, 84, 1))
            entries, hashes = [], []
            for i, name in enumerate(members):
                raw = archive.read(name)
                a = np.load(io.BytesIO(raw), allow_pickle=False)
                if a.shape not in [(64, 84), (64, 84, 1), (1, 64, 84, 1)] or not np.isin(a, [0, 1]).all():
                    raise ValueError('Invalid piano roll: ' + name)
                array[i] = a.reshape(64, 84, 1)
                hashes.append(hashlib.sha256(array[i].tobytes()).hexdigest())
                entries.append({'name': name, 'sha256': hashlib.sha256(raw).hexdigest()})
            info = {'count': expected, 'density': float(array.mean()), 'samples': entries,
                    'unique_patterns': len(set(hashes)), 'silent_count': int((~array.reshape(expected, -1).any(1)).sum())}
            array.flush()
            del array
            if key.startswith('JC_'):
                if entries != old['splits'][key]['samples']:
                    raise ValueError('JC source membership/content differs from completed baseline')
                original = baseline / 'data' / (key + '.npy')
                if file_sha(original) != old['splits'][key]['sha256']:
                    raise ValueError('Baseline prepared JC data changed')
                if not np.array_equal(np.load(original, mmap_mode='r'), np.load(target, mmap_mode='r')):
                    raise ValueError('JC packed data differ')
                shutil.copy2(original, target)
            info['sha256'] = file_sha(target)
            manifest['splits'][key] = info
            content_hashes[key] = hashes
            print('Prepared', key, expected, flush=True)
    manifest['identity'] = hashlib.sha256(json.dumps({k: v['sha256'] for k, v in manifest['splits'].items()}, sort_keys=True).encode()).hexdigest()
    leakage = {}
    mixed = set(content_hashes['JCP_mixed'])
    for pair, info in PAIRS.items():
        train = set(content_hashes[info['a'] + '_train']) | set(content_hashes[info['b'] + '_train'])
        leakage[pair] = {}
        for domain in [info['a'], info['b']]:
            test = content_hashes[domain + '_test']
            leakage[pair][domain] = {'test_phrases': len(test), 'exact_match_pair_train': sum(h in train for h in test),
                                    'exact_match_full_mixed': sum(h in mixed for h in test)}
    manifest['exact_content_overlap_audit'] = leakage
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def build(args):
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / 'cyclegan-paper-suite'
    bundle.mkdir()
    for name in CODE:
        copy(ROOT / name, bundle / name)
    for path in sorted((ROOT / 'kaggle/configs').glob('*.json')):
        copy(path, bundle / 'kaggle/configs' / path.name)
    import_baseline(args.baseline, bundle / 'baseline')
    data = prepare_data(args.dataset, bundle / 'data', args.baseline)
    files = {p.relative_to(bundle).as_posix(): file_sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    manifest = {'protocol': PROTOCOL, 'scope': 'Remaining eight selected configurations; imported JC base and classifier',
                'data_identity': data['identity'], 'configurations': RUNS, 'files': files}
    (bundle / 'bundle-manifest.json').write_text(json.dumps(manifest, indent=2))
    archive_path = args.output / 'cyclegan-paper-suite.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(args.output).as_posix())
    copy(ROOT / 'kaggle/cyclegan_paper_suite.ipynb', args.output / 'cyclegan_paper_suite.ipynb')
    result = {'archive': str(archive_path.resolve()), 'bytes': archive_path.stat().st_size,
              'sha256': file_sha(archive_path), 'files': len(files), 'data_identity': data['identity'],
              'data_overlap': data['exact_content_overlap_audit']}
    (args.output / 'build-report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args())
