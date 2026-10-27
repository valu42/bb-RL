"""Benchmark *actual group-relative gradients* on freshly sampled frozen-policy rollouts.

Unlike benchmark_training_speed, this script retains failed attempt trajectories
in memory and obtains the real leave-one-out positive/negative coefficients from
prepare_group_relative_gradients(). Rollouts are regenerated with a selected
checkpoint (not the exact historical policy version at each saved batch).

No optimizer step, training checkpoint write, proof-objective change, or saved
run mutation. Gradient inputs/weights are held fixed across CPU, MPS, batching,
length bucketing and structural-caching variants.

Example:
 python -m proofgame.benchmark_real_gradients \
   --run runs/ast_branch_race_8x6_fixed --epoch 2 --machines 24 --states 128 \
   --threads 1,2,4 --batch-sizes 2,4,8,16 --devices cpu,mps \
   --repeats 3 --out runs/ast_branch_race_8x6_fixed/speed_lab_real_gradients.json
"""
import argparse
from collections import Counter, OrderedDict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch

from .epoch_train import (execute_machine, prepare_group_relative_gradients,
                          task_seed)
from .experimental_structural_cache import cached_structural_forward
from .experimental_action_replay import (replay_unique_action_gradients,
                                         two_pass_probability_disagreement)
from .exploration import behavior_log_probs
from .experimental_packed_batch import packed_score_many
from .gpu_runtime import token_count
from .model import Policy
from .network_search import FrozenSampler, encode_json


def integers(value):
    result = [int(part.strip()) for part in value.split(',')]
    if not result or min(result) <= 0:
        raise ValueError('expected comma-separated positive integers')
    return list(dict.fromkeys(result))


def select_machines(run, epoch, dataset, count, source, seed):
    """Historical solved machines are likely to yield informative fresh groups."""
    if source not in ('saved-success', 'random', 'duration-stratified'):
        raise ValueError('invalid machine source')
    by_id = {r['machine_id']: r for r in dataset['train']}
    candidates = []
    if source != 'random':
        historical = []
        for path in sorted((run / f'epoch_{epoch:02d}_train').glob('batch_*.json')):
            for row in json.loads(path.read_text())['machines']:
                if row['machine_id'] in by_id:
                    historical.append(row)
        seen = set()
        unique = []
        for row in historical:
            if row['machine_id'] not in seen:
                unique.append(row)
                seen.add(row['machine_id'])
        if source == 'saved-success':
            candidates = [by_id[row['machine_id']] for row in unique
                          if row['successes']]
        elif unique:
            unique.sort(key=lambda row: row['seconds'])
            # Sample evenly from the full observed execution-time distribution.
            picked = [unique[min(len(unique)-1, int(
                (i+0.5) * len(unique) / min(count, len(unique))))]
                for i in range(min(count, len(unique)))]
            candidates = [by_id[row['machine_id']] for row in picked]
    rng = random.Random(seed)
    if source != 'duration-stratified':
        rng.shuffle(candidates)
    seen = {row['machine_id'] for row in candidates}
    remaining = [row for row in dataset['train'] if row['machine_id'] not in seen]
    rng.shuffle(remaining)
    return (candidates + remaining)[:count]


def gather_real_groups(run, policy, epoch, machines, source, seed):
    config = json.loads((run / 'config.json').read_text())
    if not config.get('group_relative'):
        raise ValueError('real group-gradient benchmark requires a group-relative run')
    dataset = json.loads((run / 'dataset.json').read_text())
    budget = dataset['action_budget']
    attempts = config['attempts_per_machine']
    rounds = config.get('branch_rounds', 0)
    balanced = config.get('balanced_initial_answers', False)
    epsilon = config.get('train_exploration_epsilon', 0.)
    rows = select_machines(run, epoch, dataset, machines, source, seed)
    policy.eval()
    started = time.perf_counter()
    results = []
    for i, row in enumerate(rows):
        sampler = FrozenSampler(policy, capacity=10000)
        result = execute_machine(
            row, task_seed(config['seed'], 'train', epoch, row['machine_id']),
            attempts, budget, 0, False, sampler,
            exploration_epsilon=epsilon,
            balanced_initial_answers=balanced, group_relative=True,
            branch_rounds=rounds)
        results.append(result)
        print(json.dumps({
            'event': 'real_group_rollout', 'completed': i + 1, 'total': len(rows),
            'machine_id': row['machine_id'],
            'rounds_completed': result.get('rounds_completed'),
            'successes': len(result['successes']),
            'attempts': result['attempts'],
            'elapsed_seconds': round(time.perf_counter() - started, 2),
        }), flush=True)
    groups, proof_count, stats = prepare_group_relative_gradients(
        results, budget, config.get('length_penalty', 0.), balanced, rounds)
    denominator = stats['group_attempts'] if rounds else sum(
        row['attempts'] for row in results)
    if not denominator:
        raise ValueError('no contributing rollout group attempts')
    group_details = {
        'source': source, 'machines': len(rows),
        'generated_rollout_seconds': round(time.perf_counter() - started, 3),
        'attempts': sum(r['attempts'] for r in results),
        'actions': sum(r['actions'] for r in results),
        'network_evaluations': sum(r['network_evaluations'] for r in results),
        'observation_cache_hits': sum(r['cache_hits'] for r in results),
        'verified_proofs': proof_count, 'group_stats': stats,
        'actual_gradient_groups': len(groups),
        'denominator': denominator,
        'note': 'New rollouts use the selected frozen checkpoint, not each '
                'historical batch-specific checkpoint. Failed trajectories '
                'ARE retained until real leave-one-out gradients are prepared.',
    }
    return groups, group_details, epsilon, denominator


def make_samples(groups, model, cap, seed):
    rng = random.Random(seed)
    keys = list(groups)
    if cap and len(keys) > cap:
        # Deterministic random subsample of real coefficients; 0 means all.
        keys = [keys[i] for i in sorted(rng.sample(range(len(keys)), cap))]
    result = []
    for encoded in keys:
        observation = json.loads(encoded)
        context = encode_json({k: v for k, v in observation.items() if k != 'actions'})
        texts = [context] + [encode_json(action) for action in observation['actions']]
        length = max(token_count(text, model.representation) for text in texts)
        counts, expected = groups[encoded]
        result.append({'observation': observation, 'counts': counts,
                       'expected': expected, 'tokens': length,
                       'texts': texts, 'action_count': len(observation['actions'])})
    return result


def attention_elements(items, batch_size, bucketed, heads):
    ordered = sorted(items, key=lambda r: r['tokens']) if bucketed else items
    largest = 0
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start:start + batch_size]
        contexts = [row['texts'][0] for row in chunk]
        actions = list(dict.fromkeys(text for row in chunk
                                      for text in row['texts'][1:]))
        for texts in (contexts, actions):
            if texts:
                lengths = [token_count(text, 'ast') for text in texts]
                largest = max(largest, len(texts) * heads * max(lengths)**2)
    return largest


def finish_device(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def gradient_trial(policy, items, *, mode, batch_size, bucketed,
                   denominator, epsilon, defer_checks=False,
                   packed_action_batch_size=8, max_attention_elements=8000000):
    ordered = (sorted(items, key=lambda r: r['tokens']) if bucketed else items)
    policy.zero_grad(set_to_none=True)
    finish_device(next(policy.parameters()).device)
    started = time.perf_counter()
    forward = backward = total_loss = 0.
    deferred_losses = []
    for start in range(0, len(ordered), batch_size):
        chunk = ordered[start:start + batch_size]
        tick = time.perf_counter()
        if mode == 'serial':
            logits_batch = [policy(row['observation'])[0] for row in chunk]
        elif mode == 'packed-score-many':
            logits_batch = packed_score_many(
                policy, [row['observation'] for row in chunk],
                max_action_batch_size=packed_action_batch_size,
                max_attention_elements=max_attention_elements)
        elif mode == 'score_many':
            logits_batch = policy.score_many([row['observation'] for row in chunk])
        else:
            raise ValueError(f'unknown gradient trial mode: {mode}')
        losses = []
        for row, logits in zip(chunk, logits_batch):
            logps = behavior_log_probs(logits, row['observation'], epsilon)
            counts = torch.tensor(row['counts'], dtype=logps.dtype,
                                  device=logps.device)
            losses.append(-(logps * counts).sum() / denominator)
        loss = torch.stack(losses).sum()
        if defer_checks:
            # Avoid synchronizing MPS for every microbatch's finite check and
            # scalar .item(). These detached tensors have no autograd graph.
            deferred_losses.append(loss.detach())
        elif not bool(torch.isfinite(loss).all()):
            raise ValueError('nonfinite loss')
        forward += time.perf_counter() - tick
        tick = time.perf_counter()
        loss.backward()
        backward += time.perf_counter() - tick
        if not defer_checks:
            total_loss += float(loss.detach())
    finish_device(next(policy.parameters()).device)
    if defer_checks:
        aggregate = torch.stack(deferred_losses).sum()
        if not bool(torch.isfinite(aggregate).all()):
            raise ValueError('nonfinite deferred loss')
        total_loss = float(aggregate.detach())
    elapsed = time.perf_counter() - started
    gradient = torch.cat([
        p.grad.detach().cpu().reshape(-1).to(torch.float64)
        if p.grad is not None else torch.zeros(p.numel(), dtype=torch.float64)
        for p in policy.parameters()
    ]).clone()
    if not bool(torch.isfinite(gradient).all()):
        raise ValueError('nonfinite gradient')
    return gradient, {
        'wall_seconds': elapsed, 'forward_dispatch_seconds': forward,
        'backward_dispatch_seconds': backward, 'loss': total_loss,
    }


def probability_disagreement(policy, samples, mode, batch_size, bucketed, epsilon,
                             packed_action_batch_size=8,
                             max_attention_elements=8000000):
    """Not included in runtime: separate on-policy sampling agreement check."""
    rows = sorted(samples, key=lambda r: r['tokens']) if bucketed else samples
    max_error = 0.
    compared = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            if mode == 'serial':
                logits = [policy(row['observation'])[0] for row in batch]
            elif mode == 'packed-score-many':
                logits = packed_score_many(
                    policy, [row['observation'] for row in batch],
                    max_action_batch_size=packed_action_batch_size,
                    max_attention_elements=max_attention_elements)
            else:
                logits = policy.score_many([row['observation'] for row in batch])
            for row, scores in zip(batch, logits):
                probs = behavior_log_probs(scores, row['observation'], epsilon).exp()
                for idx, original in row['expected'].items():
                    max_error = max(max_error, abs(float(probs[idx]) - original))
                    compared += 1
    return max_error, compared


def benchmark_gradients(run, epoch, *, devices, cpu_threads, batch_sizes,
                        machines, source, cap, seed, repeats,
                        max_attention_elements, relation_cache_mb, tolerance,
                        methods=('score_many', 'packed-score-many', 'two-pass'),
                        defer_checks=False, packed_action_batch_size=8):
    model_path = run / f'model_epoch_{epoch:02d}.pt'
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    policy = Policy(**saved['config']).eval()
    policy.load_state_dict(saved['state_dict'])
    torch.set_num_threads(1)
    groups, workload, epsilon, denominator = gather_real_groups(
        run, policy, epoch, machines, source, seed)
    if not groups:
        raise ValueError('no informative gradient observations; use more machines')
    samples = make_samples(groups, policy, cap, seed)
    if not samples:
        raise ValueError('no sampled gradient observations')
    summary = {
        'run': str(run), 'epoch': epoch, 'checkpoint': str(model_path),
        'frozen_model': policy.config, 'device_options': devices,
        'torch_version': torch.__version__,
        'methods': list(methods),
        'deferred_microbatch_checks': defer_checks,
        'packed_action_batch_size': packed_action_batch_size,
        'workload': {**workload, 'benchmark_states': len(samples),
                     'sampled_positive_coefficients': sum(
                         sum(v > 0 for v in row['counts']) for row in samples),
                     'sampled_negative_coefficients': sum(
                         sum(v < 0 for v in row['counts']) for row in samples),
                     'sample_token_min': min(s['tokens'] for s in samples),
                     'sample_token_median': statistics.median(s['tokens'] for s in samples),
                     'sample_token_max': max(s['tokens'] for s in samples)},
        'limitations': [
            'Input rollouts are from a single frozen checkpoint, not a normal evolving training epoch.',
            'Sampling --states limits gradient comparison to a subset of real states.',
            'Timing excludes separately reported policy probability agreement checks.',
            'Microbenchmarks omit optimizer and model-snapshot costs.',
        ],
        'results': [],
    }
    if not any(v > 0 for row in samples for v in row['counts']) or not any(
            v < 0 for row in samples for v in row['counts']):
        print('WARNING: sampled gradient coefficients do not contain both signs',
              flush=True)
    for device_name in devices:
        if device_name == 'mps' and not torch.backends.mps.is_available():
            summary['results'].append({'device': 'mps', 'status': 'device_unavailable'})
            continue
        if device_name not in ('cpu', 'mps'):
            raise ValueError('devices must be cpu or mps')
        if device_name == 'mps':
            device_threads = [1]
        else:
            device_threads = cpu_threads
        frozen = Policy(**saved['config']).to(device_name).eval()
        frozen.load_state_dict(saved['state_dict'])
        initial = {name: value.detach().cpu().clone()
                   for name, value in frozen.state_dict().items()}
        for threads in device_threads:
            torch.set_num_threads(threads)
            variants = [('serial', 1, False, False)]
            for size in batch_sizes:
                for bucket in (False, True):
                    for method in ('score_many', 'packed-score-many'):
                        if method not in methods:
                            continue
                        variants.append((method, size, bucket, False))
                        if frozen.representation == 'ast':
                            variants.append((method, size, bucket, True))
                if 'two-pass' in methods:
                    variants.append(('two-pass', size, True, False))
                    if frozen.representation == 'ast':
                        variants.append(('two-pass', size, True, True))
            reference = None
            baseline_seconds = None
            for mode, size, bucketed, cache_enabled in variants:
                case = {'device': device_name, 'threads': threads,
                        'mode': mode, 'microbatch': size,
                        'length_bucketed': bucketed,
                        'structural_cache': cache_enabled,
                        'status': 'ok'}
                if mode == 'score_many' and frozen.representation == 'ast':
                    cost = attention_elements(samples, size, bucketed,
                                              frozen.config['heads'])
                    case['max_attention_elements_estimate'] = cost
                    if cost > max_attention_elements:
                        case['status'] = 'skipped_attention_memory_guard'
                        summary['results'].append(case)
                        continue
                context = (cached_structural_forward(
                    frozen, relation_cache_mb=relation_cache_mb)
                    if cache_enabled else nullcontext(None))
                with context as cache:
                    try:
                        def run_this_variant():
                            if mode == 'two-pass':
                                return replay_unique_action_gradients(
                                    frozen, samples, context_batch_size=size,
                                    action_batch_size=16,
                                    bucketed_contexts=bucketed,
                                    denominator=denominator, epsilon=epsilon,
                                    max_attention_elements=max_attention_elements,
                                    defer_checks=defer_checks)
                            return gradient_trial(
                                frozen, samples, mode=mode, batch_size=size,
                                bucketed=bucketed, denominator=denominator,
                                epsilon=epsilon, defer_checks=defer_checks,
                                packed_action_batch_size=packed_action_batch_size,
                                max_attention_elements=max_attention_elements)

                        # Warm up every input shape, including the global
                        # action cache's two encoder passes.
                        run_this_variant()
                        trials = []
                        grad = None
                        for _ in range(repeats):
                            grad, stats = run_this_variant()
                            trials.append(stats)
                        case.update({
                            'median_wall_seconds': statistics.median(
                                row['wall_seconds'] for row in trials),
                            'loss': statistics.median(row['loss'] for row in trials),
                            'trial_wall_seconds': [round(row['wall_seconds'], 6)
                                                   for row in trials],
                        })
                        if mode == 'two-pass':
                            case['two_pass_stages_median_seconds'] = {
                                key: statistics.median(row[key] for row in trials)
                                for key in ('action_feature_no_grad_seconds',
                                            'context_and_loss_backward_seconds',
                                            'action_gradient_replay_seconds')
                            }
                            case['action_reuse_factor'] = trials[0]['action_reuse_factor']
                            case['globally_unique_actions'] = trials[0]['unique_actions']
                            case['total_action_references'] = trials[0]['action_references']
                            case['cached_action_feature_bytes'] = trials[0]['feature_leaf_bytes']
                        else:
                            case['median_forward_dispatch_seconds'] = statistics.median(
                                row['forward_dispatch_seconds'] for row in trials)
                            case['median_backward_dispatch_seconds'] = statistics.median(
                                row['backward_dispatch_seconds'] for row in trials)
                        if mode == 'serial':
                            reference = grad
                            baseline_seconds = case['median_wall_seconds']
                            case['relative_gradient_l2'] = 0.
                            case['max_absolute_gradient_error'] = 0.
                        else:
                            if reference is None:
                                raise RuntimeError('serial baseline unavailable')
                            delta = grad - reference
                            case['relative_gradient_l2'] = float(
                                torch.linalg.vector_norm(delta) /
                                torch.linalg.vector_norm(reference).clamp_min(1e-12))
                            case['max_absolute_gradient_error'] = float(
                                delta.abs().max())
                        case['speedup_vs_same_device_serial'] = (
                            baseline_seconds / case['median_wall_seconds'])
                        if mode == 'two-pass':
                            error, compared = two_pass_probability_disagreement(
                                frozen, samples, epsilon)
                        else:
                            error, compared = probability_disagreement(
                                frozen, samples, mode, size, bucketed, epsilon,
                                packed_action_batch_size=packed_action_batch_size,
                                max_attention_elements=max_attention_elements)
                        case['max_sampled_probability_absolute_error'] = error
                        case['sampled_probabilities_checked'] = compared
                        case['passes_numerical_tolerance'] = (
                            case['relative_gradient_l2'] <= tolerance
                            and error <= 3e-3)
                        if cache is not None:
                            case['cache_stats'] = cache.stats()
                    except (RuntimeError, ValueError) as exc:
                        case['status'] = 'error'
                        case['error'] = str(exc)
                summary['results'].append(case)
                print(json.dumps({'event': 'gradient_variant', **case},
                                 default=str), flush=True)
        for name, value in frozen.state_dict().items():
            if not torch.equal(value.cpu(), initial[name]):
                raise RuntimeError('frozen model weights changed during benchmark')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=2)
    p.add_argument('--machines', type=int, default=24)
    p.add_argument('--machine-source', choices=('saved-success','random',
                                                 'duration-stratified'),
                   default='saved-success')
    p.add_argument('--states', type=int, default=128,
                   help='unique real gradient states; 0 to benchmark every state')
    p.add_argument('--devices', default='cpu',
                   help='comma-separated cpu,mps (MPS unavailable on Linux)')
    p.add_argument('--threads', default='1,2,4')
    p.add_argument('--batch-sizes', default='2,4,8,16')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=37)
    p.add_argument('--max-attention-elements', type=int, default=8000000)
    p.add_argument('--relation-cache-mb', type=int, default=128)
    p.add_argument('--methods', default='score_many,packed-score-many,two-pass',
                   help='comma-separated experimental methods; serial is always measured')
    p.add_argument('--packed-action-batch-size', type=int, default=8)
    p.add_argument('--defer-checks', action='store_true',
                   help='defer nonfinite loss checks and scalar transfers until '
                        'all microbatches complete (reduces MPS synchronization)')
    p.add_argument('--tolerance', type=float, default=1e-4)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if (min(args.epoch, args.machines, args.repeats, args.max_attention_elements) < 1
            or args.states < 0 or args.relation_cache_mb < 0
            or not math.isfinite(args.tolerance) or args.tolerance <= 0):
        p.error('invalid benchmark size/tolerance')
    if args.packed_action_batch_size < 1:
        p.error('--packed-action-batch-size must be positive')
    methods = list(dict.fromkeys(
        s.strip() for s in args.methods.split(',') if s.strip()))
    if not methods or any(mode not in ('score_many', 'packed-score-many', 'two-pass')
                          for mode in methods):
        p.error('--methods must contain score_many,packed-score-many,two-pass')
    devices = list(dict.fromkeys(s.strip() for s in args.devices.split(',') if s.strip()))
    if not devices or any(device not in ('cpu', 'mps') for device in devices):
        p.error('--devices requires cpu and/or mps')
    try:
        results = benchmark_gradients(
            args.run, args.epoch, devices=devices,
            cpu_threads=integers(args.threads),
            batch_sizes=integers(args.batch_sizes),
            machines=args.machines, source=args.machine_source,
            cap=args.states, seed=args.seed, repeats=args.repeats,
            max_attention_elements=args.max_attention_elements,
            relation_cache_mb=args.relation_cache_mb,
            tolerance=args.tolerance, methods=methods,
            defer_checks=args.defer_checks,
            packed_action_batch_size=args.packed_action_batch_size)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + '\n')
    good = [row for row in results['results'] if row['status'] == 'ok'
            and row['passes_numerical_tolerance']]
    print('\nREAL GROUP-RELATIVE GRADIENT SPEED EXPERIMENT')
    print('Rollout groups:', results['workload']['machines'],
          'real gradient states:', results['workload']['actual_gradient_groups'],
          'benchmarked:', results['workload']['benchmark_states'])
    for row in sorted(good, key=lambda row: row['median_wall_seconds'])[:10]:
        print(f"{row['device']:<3} method={row['mode']:<17} threads={row['threads']} "
              f"batch={row['microbatch']:<3} bucket={row['length_bucketed']} "
              f"cache={row['structural_cache']} "
              f"{row['median_wall_seconds']:.3f}s "
              f"speedup={row['speedup_vs_same_device_serial']:.2f}x "
              f"relative_grad_error={row['relative_gradient_l2']:.3g}")
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
