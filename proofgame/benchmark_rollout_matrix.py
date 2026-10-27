"""Read-only factorial rollout performance benchmark with resumable JSON output.

Run a frozen checkpoint through CPU worker inference, shared action caches,
centralized CPU inference, and centralized Apple MPS inference. Sweep worker
counts, per-update barriers, inference batch sizes and queue wait times.

IMPORTANT: A barrier larger than the training batch measures a *frozen-policy
scheduling upper bound*, NOT a drop-in change to the optimizer's update schedule.
The script does not train or alter the selected checkpoint/run.

Example:
  python -m proofgame.benchmark_rollout_matrix \
    --run runs/ast_branch_race_8x6_fixed --epoch 2 --machines 64 \
    --strategies worker,worker-shared,central-cpu,central-mps \
    --workers 4,8,16 --barriers 8,32,64 \
    --inference-batches 4,8 --wait-ms 0,2 --repeats 2 \
    --out runs/ast_branch_race_8x6_fixed/rollout_matrix.json
"""
import argparse
from collections import Counter
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import statistics
import tempfile
import time

import torch

from .benchmark_real_gradients import integers, select_machines
from .epoch_train import init_worker, run_machine, task_seed
from .gpu_runtime import CentralRolloutPool
from .model import Policy


def build_cases(strategies, workers, barriers, inference_batches, waits, repeats):
    cases = []
    for strategy in strategies:
        for worker_count in workers:
            for barrier in barriers:
                options = ([(None, None)] if strategy in ('worker', 'worker-shared')
                           else [(size, wait) for size in inference_batches
                                 for wait in waits])
                for infer_batch, wait in options:
                    for repeat in range(repeats):
                        cases.append({'strategy': strategy, 'workers': worker_count,
                                      'barrier': barrier, 'inference_batch': infer_batch,
                                      'wait_ms': wait, 'repeat': repeat})
    return cases


def task_for(row, config, epoch, *, central):
    common = (
        row,
        task_seed(config['seed'], 'train', epoch, row['machine_id']),
        config['attempts_per_machine'], config['budget'], 0,
        False, config.get('train_exploration_epsilon', 0.),
        config.get('balanced_initial_answers', False),
        False, True, config.get('group_relative', False),
        config.get('branch_rounds', 0))
    if central:
        return common
    # run_machine takes a frozen snapshot path before the version.
    raise RuntimeError('worker tasks need a file path')


def worker_task_for(row, config, epoch, weights_path):
    seed = task_seed(config['seed'], 'train', epoch, row['machine_id'])
    return (
        row, seed, config['attempts_per_machine'], config['budget'],
        str(weights_path), 0, False,
        config.get('train_exploration_epsilon', 0.),
        config.get('balanced_initial_answers', False),
        False, True, config.get('group_relative', False),
        config.get('branch_rounds', 0))


def digest_results(results):
    hashes = {}
    for row in results:
        # Prove that competing inference schedulers completed equivalent
        # trajectories when they do; floating probability bits are excluded.
        trails = row.get('group_trajectories') or []
        source = {
            'attempts': row['attempts'], 'actions': row['actions'],
            'statuses': row['statuses'],
            'success_actions': [
                [s['action'] for s in success['actions']]
                for success in row['successes']],
            'group_actions': [
                [s['action'] for s in trajectory['actions']]
                for trajectory in trails],
        }
        hashes[row['machine_id']] = hashlib.sha256(
            json.dumps(source, sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()
    return hashes


def run_case(case, config, rows, checkpoint):
    central = case['strategy'] in ('central-cpu', 'central-mps')
    device = 'mps' if case['strategy'] == 'central-mps' else 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available():
        return {'status': 'unavailable_mps'}
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    policy = Policy(**saved['config']).to(device).eval()
    policy.load_state_dict(saved['state_dict'])
    infer_batch = case['inference_batch'] if central else None
    with tempfile.TemporaryDirectory(prefix='proofgame-rollout-matrix-') as tmp:
        snapshot = Path(tmp) / 'policy.pt'
        if not central:
            torch.save({'version': 0, 'config': policy.config,
                        'state_dict': policy.state_dict()}, snapshot)
            runner = mp.get_context('spawn').Pool(
                case['workers'], initializer=init_worker,
                initargs=(policy.config, case['strategy'] == 'worker-shared'),
                maxtasksperchild=50)
        else:
            runner = CentralRolloutPool(
                policy, workers=case['workers'],
                inference_batch_size=infer_batch, wait_ms=case['wait_ms'])
        output = []
        with runner as pool:
            start = time.perf_counter()
            for offset in range(0, len(rows), case['barrier']):
                part = rows[offset:offset + case['barrier']]
                if central:
                    jobs = [task_for(row, config, config['benchmark_epoch'],
                                     central=True) for row in part]
                    result = pool.run(jobs)
                else:
                    jobs = [worker_task_for(
                        row, config, config['benchmark_epoch'], snapshot)
                        for row in part]
                    result = pool.map(run_machine, jobs, chunksize=1)
                output.extend(result)
            if device == 'mps':
                torch.mps.synchronize()
            elapsed = time.perf_counter() - start
            inference = pool.inference.stats() if central else None
    total_actions = sum(row['actions'] for row in output)
    total_attempts = sum(row['attempts'] for row in output)
    worker_seconds = sum(row['seconds'] for row in output)
    result = {
        'status': 'ok',
        'seconds': round(elapsed, 5),
        'machines': len(output),
        'actions': total_actions,
        'attempts': total_attempts,
        'solved_machines': sum(bool(row['successes']) for row in output),
        'verified_proofs': sum(len(row['successes']) for row in output),
        'actions_per_second': total_actions / elapsed,
        'attempts_per_second': total_attempts / elapsed,
        'machine_task_seconds_sum': worker_seconds,
        'elapsed_worker_occupancy': worker_seconds / (case['workers']*elapsed),
        'observation_cache_hits': sum(row.get('cache_hits', 0) for row in output),
        'network_evaluations': sum(row.get('network_evaluations', 0) for row in output),
        'worker_inference_time_sum': sum(
            row.get('profile', {}).get('inference_seconds', 0) for row in output),
        'worker_observation_time_sum': sum(
            row.get('profile', {}).get('observation_seconds', 0) for row in output),
        'worker_step_time_sum': sum(
            row.get('profile', {}).get('step_seconds', 0) for row in output),
        'action_feature_cache_hits': sum(
            row.get('profile', {}).get('action_feature_hits', 0) for row in output),
        'action_feature_cache_misses': sum(
            row.get('profile', {}).get('action_feature_misses', 0) for row in output),
        'trajectory_hash_by_machine': digest_results(output),
        'central_inference': inference,
        'caveats': [
            'Worker elapsed-task occupancy is not hardware CPU utilization.',
            'Different numerical inference may change trajectories even with identical RNG seeds.',
            'This is a frozen-policy benchmark without gradient learning.',
            'Barrier values greater than the actual SGD batch size are upper bounds '
            'requiring an algorithm-preserving asynchronous design before training.',
        ],
    }
    return result


def run_matrix(run, epoch, *, machines, machine_source, strategies,
               workers, barriers, inference_batches, waits, repeats, seed,
               output, resume=False, max_cases=None):
    config = json.loads((run / 'config.json').read_text())
    dataset = json.loads((run / 'dataset.json').read_text())
    selected = select_machines(run, epoch, dataset, machines, machine_source, seed)
    config = {**config, 'benchmark_epoch': epoch}
    checkpoint = run / f'model_epoch_{epoch:02d}.pt'
    if not checkpoint.exists():
        raise FileNotFoundError(str(checkpoint))
    cases = build_cases(strategies, workers, barriers, inference_batches, waits,
                        repeats)
    random.Random(seed).shuffle(cases)
    signature = {
        'run': str(run), 'epoch': epoch, 'checkpoint': str(checkpoint),
        'machine_ids': [r['machine_id'] for r in selected],
        'machine_source': machine_source, 'cases': cases,
    }
    fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()).hexdigest()
    if output.exists():
        if not resume:
            raise ValueError('output already exists; use --resume to continue')
        report = json.loads(output.read_text())
        if report['signature_sha256'] != fingerprint:
            raise ValueError('resume arguments differ from the saved matrix')
    else:
        report = {
            'note': 'Read-only frozen-policy rollout matrix; no model updates.',
            'run': str(run), 'epoch': epoch, 'signature_sha256': fingerprint,
            'experiment': signature, 'torch_version': torch.__version__,
            'results': [],
        }
    done = {r['case_id'] for r in report['results']}
    for i, case in enumerate(cases):
        case_id = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()[:16]
        if case_id in done:
            continue
        if max_cases is not None and len(report['results']) >= max_cases:
            break
        print(json.dumps({'event': 'rollout_matrix_start', 'case': i + 1,
                          'of': len(cases), 'case_id': case_id, **case}),
              flush=True)
        started = time.perf_counter()
        try:
            metrics = run_case(case, config, selected, checkpoint)
        except Exception as exc:
            metrics = {'status': 'error', 'error': repr(exc)}
        row = {'case_id': case_id, **case, **metrics}
        report['results'].append(row)
        # Flush after each possibly lengthy run for crash recovery.
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(output)
        print(json.dumps({
            'event': 'rollout_matrix_done', 'case_id': case_id,
            'status': row['status'], 'seconds': row.get('seconds'),
            'case_wall_seconds': round(time.perf_counter()-started, 3),
            'actions_per_second': row.get('actions_per_second'),
        }), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=2)
    p.add_argument('--machines', type=int, default=64)
    p.add_argument('--machine-source',
                   choices=('duration-stratified', 'saved-success', 'random'),
                   default='duration-stratified')
    p.add_argument('--strategies', default='worker,worker-shared,central-cpu,central-mps')
    p.add_argument('--workers', default='4,8,16')
    p.add_argument('--barriers', default='8,32,64')
    p.add_argument('--inference-batches', default='4,8')
    p.add_argument('--wait-ms', default='0,2')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--seed', type=int, default=37)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-cases', type=int,
                   help='stop after N completed configurations; --resume later')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    strategies = list(dict.fromkeys(s.strip() for s in args.strategies.split(',')
                                   if s.strip()))
    if not strategies or any(s not in (
            'worker', 'worker-shared', 'central-cpu', 'central-mps') for s in strategies):
        p.error('--strategies must be worker,worker-shared,central-cpu,central-mps')
    try:
        worker_counts = integers(args.workers)
        barriers = integers(args.barriers)
        inference_batches = integers(args.inference_batches)
        waits = [float(s.strip()) for s in args.wait_ms.split(',')]
        if (not waits or min(waits) < 0 or min(
                args.epoch, args.machines, args.repeats) < 1
                or any(not math.isfinite(wait) for wait in waits)):
            raise ValueError('invalid benchmark configuration')
        report = run_matrix(
            args.run, args.epoch, machines=args.machines,
            machine_source=args.machine_source, strategies=strategies,
            workers=worker_counts, barriers=barriers,
            inference_batches=inference_batches, waits=waits,
            repeats=args.repeats, seed=args.seed,
            output=args.out, resume=args.resume, max_cases=args.max_cases)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    good = [row for row in report['results'] if row['status'] == 'ok']
    print('\nROLLOUT SPEED EXPERIMENTS')
    for row in sorted(good, key=lambda r: r['seconds'])[:10]:
        print(f"{row['strategy']:<13} workers={row['workers']:2d} "
              f"barrier={row['barrier']:3d} "
              f"batch={str(row['inference_batch']):>4} "
              f"wait={str(row['wait_ms']):>4} "
              f"seconds={row['seconds']:.2f} "
              f"actions/s={row['actions_per_second']:.1f}")
    print('Saved:', args.out,
          f"({len(report['results'])}/{len(report['experiment']['cases'])} cases)")


if __name__ == '__main__':
    import math
    main()
