"""Focused CPU-rollout and end-to-end optimizer A/B for the next speed release.

Two or four meaningful CPU rollout variants, NOT a 448-case parameter grid.
All variants process the SAME frozen checkpoint, machine list, action seeds,
and training-sized machine barriers, with model-version cache invalidation.

Optionally reuse the first measured rollout results to compare a COMPLETE
production learn_batch() optimizer step on the CPU baseline vs an opt-in
two-pass CPU/MPS update, including grouping, on-policy checks, gradient
clipping, optimizer step, and post-update numerical health checks.

Example:
 python -m proofgame.benchmark_speed_v2 \
    --run runs/ast_branch_race_8x6_fixed --epoch 2 \
    --machines 16 --workers 8 --barrier 8 --cache-mb 32 \
    --repeats 2 --learn-repeats 2 --gradient-device mps \
    --out runs/ast_branch_race_8x6_fixed/speed_v2.json
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import torch

from .benchmark_real_gradients import select_machines
from .benchmark_rollout_matrix import run_case
from .epoch_train import learn_batch
from .model import Policy
from .speed_estimates import read_history, stage_projection


def _write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(path)


def _compare_vectors(baseline, candidate):
    # This is for *complete* model gradients/parameters after the optimizer,
    # not a loose comparison of average parameter magnitudes.
    old = torch.cat([
        p.grad.detach().cpu().reshape(-1).to(torch.float64)
        if p.grad is not None else torch.zeros(p.numel(), dtype=torch.float64)
        for p in baseline.parameters()])
    new = torch.cat([
        p.grad.detach().cpu().reshape(-1).to(torch.float64)
        if p.grad is not None else torch.zeros(p.numel(), dtype=torch.float64)
        for p in candidate.parameters()])
    relative = (torch.linalg.vector_norm(new - old) /
                torch.linalg.vector_norm(old).clamp_min(1e-12))
    params_old = torch.cat([p.detach().cpu().reshape(-1).to(torch.float64)
                            for p in baseline.parameters()])
    params_new = torch.cat([p.detach().cpu().reshape(-1).to(torch.float64)
                            for p in candidate.parameters()])
    return {
        'relative_gradient_l2': float(relative),
        'max_absolute_gradient_difference': float((new - old).abs().max()),
        'max_absolute_parameter_difference': float(
            (params_new - params_old).abs().max()),
    }


def _learn_once(snapshot, batch_results, config, budget, *, gpu_device,
                context_batch_size):
    cpu = Policy(**snapshot['config']).eval()
    cpu.load_state_dict(snapshot['state_dict'])
    opt_cpu = torch.optim.AdamW(cpu.parameters(), lr=config['lr'], weight_decay=0.)
    torch.set_num_threads(1)
    before = time.perf_counter()
    default = learn_batch(
        cpu, opt_cpu, batch_results, budget,
        exploration_epsilon=config.get('train_exploration_epsilon', 0.),
        balanced_initial_answers=config.get('balanced_initial_answers', False),
        length_penalty=config.get('length_penalty', 0.),
        group_relative=config.get('group_relative', False),
        branch_rounds=config.get('branch_rounds', 0))
    default_seconds = time.perf_counter() - before

    accelerated = Policy(**snapshot['config']).to(gpu_device).eval()
    accelerated.load_state_dict(snapshot['state_dict'])
    opt_acc = torch.optim.AdamW(
        accelerated.parameters(), lr=config['lr'], weight_decay=0.)
    before = time.perf_counter()
    alternative = learn_batch(
        accelerated, opt_acc, batch_results, budget,
        gpu_gradient_batch_size=context_batch_size,
        exploration_epsilon=config.get('train_exploration_epsilon', 0.),
        balanced_initial_answers=config.get('balanced_initial_answers', False),
        length_penalty=config.get('length_penalty', 0.),
        group_relative=config.get('group_relative', False),
        branch_rounds=config.get('branch_rounds', 0),
        gradient_backend='two-pass')
    if gpu_device == 'mps':
        torch.mps.synchronize()
    optimized_seconds = time.perf_counter() - before

    same_status = bool(default['updated']) == bool(alternative['updated'])
    if not same_status:
        raise RuntimeError('baseline and optimized learning disagree on optimizer update')
    if default['updated']:
        drift = _compare_vectors(cpu, accelerated)
        # AdamW's normalization of very small gradients means a tiny
        # reordering error can yield a larger parameter delta. Validate
        # the COMPLETE gradient tightly and bound weights by at most one
        # pair of learning-rate-sized updates, rather than bitwise identity.
        weight_drift_limit = max(2. * config['lr'], 1e-6)
        numerical_ok = (
            drift['relative_gradient_l2'] < 1e-4
            and drift['max_absolute_parameter_difference'] <= weight_drift_limit)
        drift['max_allowed_parameter_difference'] = weight_drift_limit
    else:
        drift = None
        numerical_ok = True
    return {
        'cpu_serial_seconds': default_seconds,
        'two_pass_seconds': optimized_seconds,
        'baseline': default,
        'optimized': alternative,
        'complete_update_numerical_error': drift,
        'passes_update_parity': numerical_ok and same_status,
    }


def benchmark(run, epoch, *, machines, source, workers, barrier, cache_mb,
              include_shared, repeats, learn_repeats, gradient_device,
              context_batch_size, seed, out, target_minutes):
    if out.exists():
        raise ValueError('output file already exists; choose a new --out')
    if min(epoch, machines, workers, barrier, repeats, context_batch_size) < 1:
        raise ValueError('positive experiment sizes required')
    if cache_mb < 1 or learn_repeats < 0:
        raise ValueError('cache must be positive; learn-repeats nonnegative')
    if gradient_device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is not available; use --gradient-device cpu')
    config = json.loads((run / 'config.json').read_text())
    dataset = json.loads((run / 'dataset.json').read_text())
    checkpoint = run / f'model_epoch_{epoch:02d}.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(str(checkpoint))
    if json.loads((run / 'config.json').read_text())['model']['representation'] != 'ast':
        raise ValueError('topology-cache A/B requires an AST model')
    rows = select_machines(run, epoch, dataset, machines, source, seed)
    if not rows:
        raise ValueError('no selected machines')
    config = {**config, 'benchmark_epoch': epoch}
    variants = [
        ('baseline-local-cpu', 'worker', 0),
        ('topology-cache-local-cpu', 'worker', cache_mb),
    ]
    if include_shared:
        variants += [
            ('shared-action-cache', 'worker-shared', 0),
            ('shared-plus-topology', 'worker-shared', cache_mb),
        ]
    report = {
        'status': 'incomplete', 'run': str(run), 'epoch': epoch,
        'checkpoint': str(checkpoint),
        'settings': {
            'machines': len(rows), 'source': source, 'seed': seed,
            'workers': workers, 'barrier': barrier, 'topology_cache_mb': cache_mb,
            'include_shared_action_cache': include_shared, 'repeats': repeats,
            'learn_repeats': learn_repeats, 'gradient_device': gradient_device,
            'context_batch_size': context_batch_size,
        },
        'results': [], 'rollout_summary': None,
        'learning_summary': None, 'epoch_projection': None,
        'caveats': [
            'All rollouts use a single frozen weight tensor, even though '
            'version IDs and action caches change at the training batch barrier.',
            'The isolated optimizer test learns from one complete machine batch; '
            'it never modifies the saved checkpoint or original run.',
            'Evaluation speed is not assumed to improve automatically.',
            'Model-version cache invalidation is simulated, not actual SGD updates.',
        ],
    }
    _write(out, report)
    original_rows = None
    for repeat in range(repeats):
        # Alternate order between repeats to reduce thermal/order bias.
        order = variants if repeat % 2 == 0 else list(reversed(variants))
        for name, strategy, megabytes in order:
            case = {
                'strategy': strategy, 'workers': workers, 'barrier': barrier,
                'inference_batch': None, 'wait_ms': None, 'repeat': repeat}
            print(json.dumps({
                'event': 'focused_rollout_start', 'variant': name,
                'repeat': repeat, 'machines':len(rows)}), flush=True)
            started = time.perf_counter()
            if original_rows is None and name == 'baseline-local-cpu':
                measurement, original_rows = run_case(
                    case, config, rows, checkpoint,
                    simulate_update_barriers=True,
                    cpu_topology_cache_mb=megabytes, return_results=True)
            else:
                measurement = run_case(
                    case, config, rows, checkpoint,
                    simulate_update_barriers=True,
                    cpu_topology_cache_mb=megabytes)
            report['results'].append({
                'variant': name, 'repeat': repeat, 'wall_with_setup_seconds':
                time.perf_counter() - started,
                **measurement})
            _write(out, report)
            print(json.dumps({
                'event': 'focused_rollout_done', 'variant': name,
                'seconds': measurement['seconds'],
                'actions_per_second':measurement['actions_per_second'],
                'topology_hits':measurement['topology_cache_hits'],
            }), flush=True)

    by_variant = defaultdict(list)
    for item in report['results']:
        by_variant[item['variant']].append(item)
    baseline = by_variant['baseline-local-cpu'][0]
    baseline_hashes = baseline['trajectory_hash_by_machine']
    rollout_summary = {}
    for name, _, _ in variants:
        measurements = by_variant[name]
        same = all(row['trajectory_hash_by_machine'] == baseline_hashes
                   for row in measurements)
        elapsed = statistics.median(row['seconds'] for row in measurements)
        rollout_summary[name] = {
            'median_seconds': elapsed,
            'median_actions_per_second':statistics.median(
                row['actions_per_second'] for row in measurements),
            'same_action_trajectories':same,
            'speedup_vs_baseline': (
                statistics.median(row['seconds'] for row in
                                  by_variant['baseline-local-cpu']) / elapsed
                if same else None),
        }
    report['rollout_summary'] = rollout_summary
    _write(out, report)

    if learn_repeats:
        # Highest-success *complete* batch; avoids reporting a no-update
        # learning "speedup" for an all-failed group.
        candidates = [original_rows[i:i + barrier]
                      for i in range(0, len(original_rows), barrier)]
        learning_rows = max(candidates, key=lambda group: sum(
            len(row['successes']) for row in group))
        frozen = torch.load(checkpoint, map_location='cpu', weights_only=True)
        trials = []
        for i in range(learn_repeats):
            print(json.dumps({'event': 'optimizer_ab_start',
                              'repeat': i, 'machines': len(learning_rows),
                              'gradient_device': gradient_device}), flush=True)
            trials.append(_learn_once(
                frozen, learning_rows, config, dataset['action_budget'],
                gpu_device=gradient_device, context_batch_size=context_batch_size))
            report['learning_trials'] = trials
            _write(out, report)
            print(json.dumps({
                'event': 'optimizer_ab_done', 'repeat': i,
                'cpu_seconds':trials[-1]['cpu_serial_seconds'],
                'optimized_seconds':trials[-1]['two_pass_seconds'],
                'updated':trials[-1]['optimized']['updated'],
                'numerically_equivalent':trials[-1]['passes_update_parity'],
            }), flush=True)
        valid = (all(row['passes_update_parity'] for row in trials)
                 and all(row['optimized']['updated'] for row in trials))
        factor = (statistics.median(row['cpu_serial_seconds'] for row in trials)
                  / statistics.median(row['two_pass_seconds'] for row in trials)
                  if valid else None)
        report['learning_summary'] = {
            'speedup_vs_cpu_serial': factor,
            'all_updates_numerically_equivalent': all(
                row['passes_update_parity'] for row in trials),
            'all_updates_applied':all(row['optimized']['updated'] for row in trials),
            'sampled_machine_batch':len(learning_rows),
            'note':'Full optimizer update, including gradient replay and numerical checks.',
        }
        _write(out, report)
    historical = read_history(run, epoch)
    rollout_factor = rollout_summary['topology-cache-local-cpu']['speedup_vs_baseline']
    learning_factor = (
        report['learning_summary']['speedup_vs_cpu_serial']
        if report['learning_summary'] else None)
    if rollout_factor is not None and learning_factor is not None:
        report['epoch_projection'] = stage_projection(
            historical, rollout_speedup=rollout_factor,
            learning_speedup=learning_factor,
            evaluation_speedup=1., target_minutes=target_minutes)
    report['status'] = 'complete'
    _write(out, report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=2)
    p.add_argument('--machines', type=int, default=16)
    p.add_argument('--machine-source', choices=('saved-success',
                                                'duration-stratified','random'),
                   default='saved-success')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--barrier', type=int, default=8,
                   help='actual training minibatch size, 8 on the benchmarked run')
    p.add_argument('--cache-mb', type=int, default=32)
    p.add_argument('--include-shared-action-cache', action='store_true')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--learn-repeats', type=int, default=2,
                   help='0 skips complete optimizer-step A/B')
    p.add_argument('--gradient-device', choices=('cpu','mps'), default='mps')
    p.add_argument('--context-batch-size', type=int, default=8)
    p.add_argument('--seed', type=int, default=37)
    p.add_argument('--target-minutes', type=float, default=30.)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    try:
        results = benchmark(
            args.run, args.epoch, machines=args.machines,
            source=args.machine_source, workers=args.workers,
            barrier=args.barrier, cache_mb=args.cache_mb,
            include_shared=args.include_shared_action_cache,
            repeats=args.repeats, learn_repeats=args.learn_repeats,
            gradient_device=args.gradient_device,
            context_batch_size=args.context_batch_size,
            seed=args.seed, out=args.out, target_minutes=args.target_minutes)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        p.error(str(exc))
    print('\nFOCUSED TRAINING-SIZED SPEED V2 BENCHMARK')
    for name, item in results['rollout_summary'].items():
        print(f"{name:<25} {item['median_seconds']:.3f}s, "
              f"{item['median_actions_per_second']:.1f} actions/s, "
              f"speedup={item['speedup_vs_baseline']}, "
              f"same trajectories={item['same_action_trajectories']}")
    if results['learning_summary']:
        print('Full optimizer-step speedup:',
              results['learning_summary']['speedup_vs_cpu_serial'])
    if results['epoch_projection']:
        print('Projected full epoch (not measured):',
              round(results['epoch_projection']['projected_epoch_minutes'], 2),
              'minutes')
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
