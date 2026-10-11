"""Resumable native-TF1.4 runner for the selected JC/CP/JP legacy variants."""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time
from types import SimpleNamespace


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def trim_checkpoint_log(path, step):
    if not path.exists():
        return
    lines = path.read_text().splitlines()
    kept = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except ValueError:
            if index != len(lines) - 1:
                raise ValueError('Corrupt non-trailing log record: ' + str(path))
            continue
        if record['step'] <= step:
            kept.append(line)
    if len(kept) != len(lines):
        path.rename(path.with_name(path.name + '.superseded-' + str(time.time())))
        path.write_text(''.join(line + '\n' for line in kept))


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['train', 'evaluate'])
    p.add_argument('--kind', choices=['gan', 'classifier'], default='gan')
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-steps', type=int, default=0, help='Bound this invocation; zero means finish training')
    p.add_argument('--max-seconds', type=float, default=0, help='Checkpoint and pause after this invocation time budget')
    p.add_argument('--gan-checkpoint')
    p.add_argument('--classifier-checkpoint')
    p.add_argument('--limit', type=int, default=0, help='Evaluation smoke limit; zero evaluates the complete test split')
    return p.parse_args()


def legacy_options(config, batch_size, run):
    return SimpleNamespace(batch_size=batch_size, fine_size=128, time_step=64, pitch_range=84,
        input_nc=1, output_nc=1, L1_lambda=config['lambda_cycle'], gamma=config['gamma'],
        sigma_d=config['sigma_d'], sigma_c=0., dataset_dir='', dataset_A_dir=config['dataset_a'],
        dataset_B_dir=config['dataset_b'], sample_dir=str(run / 'samples'), model=config['model'],
        ngf=config['ngf'], ndf=config['ndf'], phase='train', max_size=50)


def session_config(args):
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4,
                           inter_op_parallelism_threads=2)
    config.gpu_options.allow_growth = True
    if args.device == 'cpu':
        config.device_count['GPU'] = 0
    elif not gpu_devices():
        raise RuntimeError('GPU required, but TensorFlow exposes none')
    return config


def make_model(sess, config, kind, run, batch_size=None):
    tf.set_random_seed(config['seed'])
    options = legacy_options(config, batch_size or config['batch_size'], run)
    with (run / ('graph-' + kind + '.log')).open('w') as output, contextlib.redirect_stdout(output):
        return cyclegan(sess, options) if kind == 'gan' else Classifer(sess, options)


def checkpoint_prefix(path):
    path = Path(path)
    if path.is_dir():
        state = tf.train.get_checkpoint_state(str(path))
        if not state:
            raise ValueError('No checkpoint found in ' + str(path))
        path = path / Path(state.model_checkpoint_path).name
    if not Path(str(path) + '.index').exists():
        raise ValueError('Missing checkpoint: ' + str(path))
    return str(path)


def training_arrays(data, config):
    from experiment_health import split_training
    count = config.get('validation_per_domain', 0)
    a, va, ma = split_training(data.arrays[config['dataset_a'] + '_train'], count, config['seed'] + 200000)
    b, vb, mb = split_training(data.arrays[config['dataset_b'] + '_train'], count, config['seed'] + 200001)
    return a, b, va, vb, {'source': 'published training split only', 'A': ma, 'B': mb}


def training_health(sess, model, va, vb, limits):
    from experiment_health import assess_outputs
    result = {}
    for name, array, source, fetches in [
        ('AtoB', va, model.test_A, [model.testB, model.testA_]),
        ('BtoA', vb, model.test_B, [model.testA, model.testB_])]:
        outputs = [sess.run(fetches, {source: array[i:i+16].astype(np.float32)})
                   for i in range(0, len(array), 16)]
        result[name] = assess_outputs(array, np.concatenate([x[0] for x in outputs]),
                                     np.concatenate([x[1] for x in outputs]), limits)
    return {'source': 'held-out training validation; published test set unused',
            'passed': all(r['passed'] for r in result.values()), 'directions': result}


def train(args, config, data, run):
    from suite_plan import validate_config
    from suite_data import mixed_training
    from replay import ReplayPool
    validate_config(config)
    kind = args.kind
    identity = {'config': config, 'data_identity': data.manifest['identity'], 'kind': kind}
    metadata_path = run / 'config.json'
    if args.resume:
        if json.loads(metadata_path.read_text()) != identity:
            raise ValueError('Resume requires the same configuration and dataset')
    else:
        run.mkdir(parents=True, exist_ok=False)
        atomic_json(metadata_path, identity)
    (run / 'checkpoints').mkdir(exist_ok=True)
    a, b, va, vb, membership = training_arrays(data, config)
    mixed = mixed_training(data, config, a, b) if kind == 'gan' else None
    if mixed is not None:
        membership['mixed'] = {'source': config['mixed_source'], 'count': len(mixed),
            'sampling': 'independent seeded permutation per epoch; first batches*batch_size phrases'}
    atomic_json(run / 'training-split.json', membership)
    health = config.get('health_check') if kind == 'gan' else None
    if health and (not len(va) or min(health['every_steps'], health['max_consecutive_failures']) <= 0):
        raise ValueError('Health checks require validation phrases and positive intervals/patience')
    batch = config['batch_size']
    batches = min(len(a), len(b)) // batch if kind == 'gan' else (len(a) + len(b)) // batch
    epochs = config['epochs'] if kind == 'gan' else config['classifier_epochs']
    total_steps = batches * epochs
    if not batches or config['decay_start_epoch'] >= epochs:
        raise ValueError('Invalid epoch/batch schedule')
    stopping = []
    def stop(signum, _frame):
        stopping.append(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with tf.Session(config=session_config(args)) as sess:
        tf.set_random_seed(config['seed'])
        m = make_model(sess, config, kind, run)
        lr = tf.placeholder(tf.float32, (), name='learning_rate')
        def optimizer(loss, variables, name):
            opt = tf.train.AdamOptimizer(lr, beta1=config['beta1'], beta2=config['beta2'], name=name)
            gradients = opt.compute_gradients(loss, var_list=variables)
            if any(g is None for g, _ in gradients):
                raise ValueError('Disconnected trainable variable')
            return opt.apply_gradients(gradients), tf.global_norm([g for g, _ in gradients]), gradients
        if kind == 'gan':
            from experiment_objectives import generator_objective
            g_loss, cycle_loss, objective = generator_objective(m, config, {'A': float(a.mean()), 'B': float(b.mean())})
            atomic_json(run / 'objective.json', objective)
            g_op, g_norm, g_gradients = optimizer(g_loss, m.g_vars, 'AdamG')
            discriminator_loss = m.d_loss if config['model'] == 'base' else m.D_loss
            d_op, d_norm, _ = optimizer(discriminator_loss, m.d_vars, 'AdamD')
            g_direction_norms = [tf.global_norm([g for g, v in g_gradients if v.name.startswith(scope)])
                                 for scope in ['generatorA2B/', 'generatorB2A/']]
            names = ['g_loss', 'd_loss', 'cycle_weighted', 'cycle_unweighted', 'g_gradient_norm', 'd_gradient_norm']
            if mixed is not None:
                names += ['d_base_loss', 'd_mixed_loss']
            variables = m.g_vars + m.d_vars
        else:
            c_op, c_norm, _ = optimizer(m.d_loss, m.d_vars, 'AdamClassifier')
            names = ['cross_entropy', 'gradient_norm']
            variables = m.d_vars
        step = tf.get_variable('completed_steps', initializer=0, trainable=False)
        epoch_sums = tf.get_variable('epoch_sums', initializer=np.zeros(len(names), np.float64), trainable=False)
        metrics_input = tf.placeholder(tf.float64, [len(names)])
        finish_step = tf.group(step.assign_add(1), epoch_sums.assign_add(metrics_input))
        reset_sums = epoch_sums.assign(np.zeros(len(names), np.float64))
        pool = None
        if mixed is not None:
            pool = ReplayPool(config['replay_pool_size'], (batch, 64, 84, 1))
            pool_values = tf.get_variable('replay_values', shape=pool.values.shape, initializer=tf.zeros_initializer(), trainable=False)
            pool_count = tf.get_variable('replay_count', initializer=0, trainable=False)
            pool_input = tf.placeholder(tf.float32, pool.values.shape)
            pool_count_input = tf.placeholder(tf.int32, ())
            save_pool = tf.group(pool_values.assign(pool_input), pool_count.assign(pool_count_input))
        if health:
            health_failures = tf.get_variable('consecutive_health_failures', initializer=0, trainable=False)
            next_health_failures = tf.placeholder(tf.int32, (), name='next_health_failures')
            update_health_failures = health_failures.assign(next_health_failures)
        saver = tf.train.Saver(max_to_keep=2, save_relative_paths=True)
        epoch_saver = tf.train.Saver(max_to_keep=30, save_relative_paths=True)
        sess.run(tf.global_variables_initializer())
        if args.resume:
            saver.restore(sess, checkpoint_prefix(run / 'checkpoints'))
        start_step = current = int(sess.run(step))
        if pool is not None:
            pool.values, pool.count = sess.run([pool_values, pool_count])
        # A crash can leave logs ahead of the last durable optimizer checkpoint.
        # Preserve that tail, then resume with unique absolute steps.
        if args.resume:
            for filename in ['steps.jsonl', 'epochs.jsonl', 'health.jsonl']:
                trim_checkpoint_log(run / filename, current)
        if health and int(sess.run(health_failures)) >= health['max_consecutive_failures']:
            raise RuntimeError('Checkpoint stopped by quality gate; diagnose and start a new experiment')
        latest_health = None
        if health and (run / 'health.json').exists():
            previous_health = json.loads((run / 'health.json').read_text())
            if previous_health['step'] <= current:
                latest_health = previous_health
        started = time.monotonic()
        elapsed_updates = 0.
        recent_seconds = []
        atomic_json(run / 'runtime.json', {'python': sys.version, 'tensorflow': tf.__version__,
            'build': build_info(), 'platform': platform.platform(),
            'devices': [str(d) for d in sess.list_devices()], 'parameters': sum(v.shape.num_elements() for v in variables),
            'gpu_details': gpu_details(),
            'pid': os.getpid(), 'started': timestamp(), 'from_step': current})
        with (run / 'runtime-history.jsonl').open('a') as history:
            history.write(json.dumps(json.loads((run / 'runtime.json').read_text())) + '\n')
        with (run / 'steps.jsonl').open('a', buffering=1) as events:
            previous_epoch = None
            while current < total_steps:
                epoch, index = divmod(current, batches)
                if previous_epoch != epoch:
                    rng = np.random.RandomState(config['seed'] + epoch + (0 if kind == 'gan' else 100000))
                    order_a, order_b = rng.permutation(len(a)), rng.permutation(len(b))
                    order_c = rng.permutation(len(a) + len(b))
                    if mixed is not None:
                        order_mixed = np.random.RandomState(config['seed'] + 3000000 + epoch).permutation(len(mixed))
                    previous_epoch = epoch
                learning_rate = config['learning_rate']
                if epoch >= config['decay_start_epoch']:
                    learning_rate *= (epochs - epoch) / (epochs - config['decay_start_epoch'])
                t0 = time.monotonic()
                extra_metrics = {}
                if kind == 'gan':
                    ba = a[order_a[index * batch:(index + 1) * batch]].astype(np.float32)
                    bb = b[order_b[index * batch:(index + 1) * batch]].astype(np.float32)
                    noise = np.abs(np.random.RandomState(config['seed'] + 1000000 + current).normal(
                        0, config['sigma_d'], ba.shape)).astype(np.float32)
                    feed = {m.real_data: np.concatenate([ba, bb], axis=3), m.gaussian_noise: noise, lr: learning_rate}
                    if mixed is not None:
                        feed[m.real_mixed] = mixed[order_mixed[index * batch:(index + 1) * batch]].astype(np.float32)
                    fetch = [m.fake_A, m.fake_B, g_loss, cycle_loss, g_norm, g_op, g_direction_norms, m.cycle_loss]
                    if current == start_step:
                        trace = tf.RunMetadata()
                        # TF1's hardware profiler requires a separate CUDA 8 CUPTI library.
                        # Software tracing records device placement without that optional profiler.
                        trace_level = (tf.RunOptions.SOFTWARE_TRACE if tf.__version__.startswith('1.')
                                       else tf.RunOptions.FULL_TRACE)
                        fa, fb, gl, cl, gn, _, direction_norms, legacy_cycle = sess.run(fetch, feed,
                            options=tf.RunOptions(trace_level=trace_level), run_metadata=trace)
                        placement = {d.device: [n.node_name for n in d.node_stats] for d in trace.step_stats.dev_stats}
                        atomic_json(run / 'execution-devices.json', placement)
                        if args.device == 'gpu' and not any('GPU:0' in d.upper() for d in placement):
                            raise RuntimeError('No training operations executed on GPU')
                    else:
                        fa, fb, gl, cl, gn, _, direction_norms, legacy_cycle = sess.run(fetch, feed)
                    if pool is not None:
                        fa, fb = pool.query(fa, fb, config['seed'] + 2000000 + current)
                    feed.update({m.fake_A_sample: fa, m.fake_B_sample: fb})
                    d_fetch = [discriminator_loss, d_norm, d_op]
                    if mixed is not None:
                        d_fetch += [m.d_loss, m.d_all_loss]
                    d_values = sess.run(d_fetch, feed)
                    dl, dn = d_values[:2]
                    values = [gl, dl, cl, cl / config['lambda_cycle'], gn, dn]
                    if mixed is not None:
                        values += list(d_values[3:])
                    extra_metrics = {'g_a2b_gradient_norm': float(direction_norms[0]),
                        'g_b2a_gradient_norm': float(direction_norms[1]),
                        'legacy_cycle_unweighted': float(legacy_cycle / config['lambda_cycle'])}
                else:
                    ids = order_c[index * batch:(index + 1) * batch]
                    ba = np.stack([a[i] if i < len(a) else b[i - len(a)] for i in ids]).astype(np.float32)
                    labels = np.eye(2, dtype=np.float32)[(ids >= len(a)).astype(int)]
                    loss, norm, _ = sess.run([m.d_loss, c_norm, c_op],
                        {m.origin_train: ba * 2 - 1, m.label_train: labels, lr: learning_rate})
                    values = [loss, norm]
                if not np.isfinite(values).all():
                    raise FloatingPointError('Non-finite loss/gradient; retaining last completed checkpoint')
                sess.run(finish_step, {metrics_input: np.asarray(values, dtype=np.float64)})
                current += 1
                seconds = time.monotonic() - t0
                elapsed_updates += seconds
                recent_seconds.append(seconds)
                recent_seconds = recent_seconds[-50:]
                events.write(json.dumps({'step': current, 'epoch': epoch + 1, 'batch': index + 1,
                    'learning_rate': learning_rate, 'seconds': seconds, **dict(zip(names, map(float, values))),
                    **extra_metrics}) + '\n')
                epoch_end = current % batches == 0
                checked_health = False
                failed_health = False
                if health and (current % health['every_steps'] == 0 or epoch_end):
                    latest_health = training_health(sess, m, va, vb, health)
                    previous_failures = int(sess.run(health_failures))
                    failures = previous_failures + 1 if current >= health['warmup_steps'] and not latest_health['passed'] else 0
                    sess.run(update_health_failures, {next_health_failures: failures})
                    latest_health.update(step=current, consecutive_failures=failures, thresholds=health)
                    atomic_json(run / 'health.json', latest_health)
                    with (run / 'health.jsonl').open('a') as f:
                        f.write(json.dumps(latest_health) + '\n')
                    checked_health = True
                    failed_health = failures >= health['max_consecutive_failures']
                if epoch_end:
                    averages = sess.run(epoch_sums) / batches
                    epoch_result = {'epoch': epoch + 1, 'step': current, **dict(zip(names, averages.tolist()))}
                    if kind == 'classifier':
                        epoch_result['test_accuracy_diagnostic_only'] = classifier_accuracy(sess, m, data, 0., config['seed'], config)
                    with (run / 'epochs.jsonl').open('a') as f:
                        f.write(json.dumps(epoch_result) + '\n')
                    sess.run(reset_sums)
                    if kind == 'gan':
                        samples = run / 'samples'
                        samples.mkdir(exist_ok=True)
                        source = (va if health else data.arrays[config['dataset_a'] + '_test'])[:4].astype(np.float32)
                        target = (vb if health else data.arrays[config['dataset_b'] + '_test'])[:4].astype(np.float32)
                        outputs = sess.run([m.testB, m.testA_, m.testA, m.testB_], {m.test_A: source, m.test_B: target})
                        np.savez_compressed(str(samples / 'epoch-{:02d}.npz'.format(epoch+1)), origin_a=source,
                            origin_b=target, transfer_a2b=outputs[0], cycle_a=outputs[1],
                            transfer_b2a=outputs[2], cycle_b=outputs[3])
                bounded = ((args.max_steps and current - start_step >= args.max_steps) or
                           (args.max_seconds and time.monotonic() - started >= args.max_seconds))
                should_save = epoch_end or current % config['checkpoint_every_steps'] == 0 or bounded or stopping or checked_health
                if should_save:
                    if pool is not None:
                        sess.run(save_pool, {pool_input: pool.values, pool_count_input: pool.count})
                    saver.save(sess, str(run / 'checkpoints' / 'model'), global_step=step)
                    if epoch_end and ((kind == 'classifier' and epoch + 1 == config['classifier_epochs']) or
                                      (kind == 'gan' and epoch + 1 in config['evaluate_epochs'])):
                        epoch_dir = run / 'epoch-{:02d}'.format(epoch+1)
                        epoch_dir.mkdir(exist_ok=True)
                        epoch_saver.save(sess, str(epoch_dir / 'model'), global_step=step)
                if current % config['report_every_steps'] == 0 or current == start_step + 1 or should_save:
                    status = {'state': 'complete' if current == total_steps else 'running', 'kind': kind,
                        'step': current, 'total_steps': total_steps, 'epoch': epoch + 1,
                        'batch_in_epoch': index + 1, 'batches_per_epoch': batches, 'updated': timestamp(),
                        'learning_rate': learning_rate, 'recent_step_seconds': float(np.median(recent_seconds)),
                        'remaining_seconds_estimate': float(np.median(recent_seconds) * (total_steps-current)),
                        'last_metrics': {**dict(zip(names, map(float, values))), **extra_metrics}}
                    if latest_health:
                        status['health'] = latest_health
                    atomic_json(run / 'status.json', status)
                    print(json.dumps(status), flush=True)
                if failed_health:
                    status.update(state='failed', reason='quality_gate', checkpoint_step=current)
                    atomic_json(run / 'status.json', status)
                    raise RuntimeError('Quality gate failed repeatedly; checkpoint saved. See health.json')
                if (bounded or stopping) and current < total_steps:
                    status['state'] = 'paused'
                    atomic_json(run / 'status.json', status)
                    break
            result = {'state': 'complete' if current == total_steps else 'paused', 'start_step': start_step,
                'completed_steps': current, 'total_steps': total_steps, 'elapsed_seconds': time.monotonic()-started,
                'optimizer_seconds': elapsed_updates, 'stopped_by_signal': bool(stopping)}
            with (run / 'invocations.jsonl').open('a') as history:
                history.write(json.dumps(result) + '\n')
            atomic_json(run / 'last-invocation.json', result)
            if current == total_steps:
                atomic_json(run / 'complete.json', result)
            print(json.dumps(result), flush=True)


def classifier_accuracy(sess, classifier, data, sigma, seed, config, limit=0):
    rng = np.random.RandomState(seed)
    correct, count = 0, 0
    for label, key in enumerate([config['dataset_a'] + '_test', config['dataset_b'] + '_test']):
        array = data.arrays[key][:limit or None]
        for start in range(0, len(array), 32):
            x = array[start:start+32].astype(np.float32) * 2 - 1
            if sigma:
                x += rng.normal(0, sigma, x.shape).astype(np.float32)
            probabilities = sess.run(classifier.test_result_softmax, {classifier.test_midi: x})
            correct += int((probabilities.argmax(axis=1) == label).sum())
            count += len(x)
    return correct / count


def evaluate(args, config, data, run):
    from experiment_metrics import transfer_metrics
    from utils import save_midis
    import pretty_midi
    run.mkdir(parents=True, exist_ok=False)
    gan_ckpt = checkpoint_prefix(args.gan_checkpoint)
    classifier_ckpt = checkpoint_prefix(args.classifier_checkpoint)
    binary, continuous = {}, {}
    with tf.Session(config=session_config(args)) as sess:
        model = make_model(sess, config, 'gan', run, batch_size=1)
        tf.train.Saver(var_list=model.g_vars).restore(sess, gan_ckpt)
        for direction, key, source, fetches in [
            ('AtoB', config['dataset_a'] + '_test', model.test_A, [model.testB, model.testA_]),
            ('BtoA', config['dataset_b'] + '_test', model.test_B, [model.testA, model.testB_])]:
            origins = data.arrays[key][:args.limit or None]
            transfers, cycles = [], []
            for start in range(0, len(origins), 16):
                t, c = sess.run(fetches, {source: origins[start:start+16].astype(np.float32)})
                if not np.isfinite(t).all() or not np.isfinite(c).all():
                    raise FloatingPointError('Non-finite inference')
                transfers.append(t)
                cycles.append(c)
            continuous[direction] = [origins.astype(np.float32), np.concatenate(transfers), np.concatenate(cycles)]
            binary[direction] = [v > .5 for v in continuous[direction]]
            np.savez_compressed(str(run / (direction + '.npz')), origin=binary[direction][0],
                transfer=binary[direction][1], cycle=binary[direction][2],
                transfer_continuous=continuous[direction][1], cycle_continuous=continuous[direction][2])
            print('Generated', direction, len(origins), flush=True)
    tf.reset_default_graph()
    with tf.Session(config=session_config(args)) as sess:
        classifier = make_model(sess, config, 'classifier', run, batch_size=16)
        tf.train.Saver(var_list=classifier.d_vars).restore(sess, classifier_ckpt)
        report = {'protocol': config['protocol'], 'pair': config['pair'], 'model': config['model'],
            'sigma_d': config['sigma_d'], 'labels': config['genres'],
            'cycle_loss': config.get('cycle_loss', 'legacy_l1'),
            'gan_checkpoint': gan_ckpt, 'classifier_checkpoint': classifier_ckpt,
            'data_identity': data.manifest['identity'], 'limited_smoke': bool(args.limit),
            'classifier_robustness': {}, 'binary_legacy_pipeline': {}, 'continuous_diagnostic': {}}
        for sigma in config['classifier_robustness_sigmas']:
            report['classifier_robustness'][str(sigma)] = classifier_accuracy(sess, classifier, data, sigma, config['seed'], config, args.limit)
        for label, direction in enumerate(['AtoB', 'BtoA']):
            for representation, arrays in [('binary_legacy_pipeline', binary[direction]),
                                            ('continuous_diagnostic', continuous[direction])]:
                all_probs = []
                for array in arrays:
                    probs = [sess.run(classifier.test_result_softmax,
                        {classifier.test_midi: array[i:i+32].astype(np.float32)*2-1})
                        for i in range(0, len(array), 32)]
                    all_probs.append(np.concatenate(probs))
                p = np.stack(all_probs, axis=1)
                report[representation][direction] = transfer_metrics(p, *binary[direction], label)
                np.save(str(run / (direction + '-' + representation + '-probabilities.npy')), p)
            for index in range(min(4, len(binary[direction][0]))):
                for stage, array in zip(['origin', 'transfer', 'cycle'], binary[direction]):
                    path = run / '{}-{}-{}.mid'.format(direction, index+1, stage)
                    save_midis(array[index:index+1], str(path))
                    parsed = pretty_midi.PrettyMIDI(str(path))
                    assert all(n.end > n.start >= 0 for instrument in parsed.instruments for n in instrument.notes)
        for representation in ['binary_legacy_pipeline', 'continuous_diagnostic']:
            r = report[representation]
            for metric in ['signed_probability_strength_all', 'signed_probability_strength_successful',
                           'legacy_accuracy_derived_strength']:
                values = [r[direction][metric] for direction in ['AtoB', 'BtoA']]
                r['bidirectional_' + metric] = None if None in values else float(np.mean(values))
        if config.get('health_check'):
            from experiment_health import assess_outputs
            report['note_quality'] = {direction: assess_outputs(*arrays, config['health_check'])
                                      for direction, arrays in continuous.items()}
        report['interpretation'] = ('Experimental balanced-cycle objective; not a reproduction of the paper loss.'
            if config.get('cycle_loss') == 'balanced_l1' else 'Legacy compatibility protocol; see documented paper/code differences.')
        report['paper_reference'] = config['paper_reference']
        report['paper_caveat'] = 'Selected repository configurations, not a 54-model search; architecture/noise/aggregation caveats remain.'
        atomic_json(run / 'report.json', report)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    args = arguments()
    config = json.loads(Path(args.config).read_text())
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if config['deterministic']:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
    import numpy as np
    from tf_compat import tf, gpu_devices, build_info, gpu_details
    if config.get('tensorflow_required') and tf.__version__ != config['tensorflow_required']:
        raise RuntimeError('This protocol requires TensorFlow ' + config['tensorflow_required'])
    if config.get('python_required') and platform.python_version() != config['python_required']:
        raise RuntimeError('This protocol requires Python ' + config['python_required'])
    from model import cyclegan
    from style_classifier import Classifer
    from suite_data import PreparedData
    data = PreparedData(args.data, config)
    # Python 3.5 Path.resolve() requires an existing path; a fresh run must not exist.
    run = Path(os.path.realpath(args.run_dir))
    try:
        (train if args.command == 'train' else evaluate)(args, config, data, run)
    except BaseException as error:
        if run.exists():
            import traceback
            atomic_json(run / 'failure.json', {'time': timestamp(), 'error': repr(error), 'traceback': traceback.format_exc()})
        raise
