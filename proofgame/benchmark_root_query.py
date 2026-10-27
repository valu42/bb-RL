"""Read-only, two-version benchmark of the one-layer AST root-query shortcut.

python -m proofgame.benchmark_root_query --run runs/ast_branch_race_8x6_fixed

Default: CPU, eight freshly replayed machines, up to 256 real gradient states,
three timed repeats after one warmup. Add --devices cpu,mps to test MPS too.
No optimizer step; no saved checkpoint, training defaults or proof rules change.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch

from .experimental_root_query import root_query_encoder, validate_encoder
from .exploration import behavior_log_probs
from .model import Policy
from .network_search import FrozenSampler, encode_json
from .structural_encoding import cached_tree


SCHEMA = 'root-query-workload-1'


def emit(event, **fields):
    print(json.dumps({'event': event, **fields}, allow_nan=False), flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()


def workload(run, epoch, policy, checkpoint_hash, *, machines, states, seed,
             reuse=None):
    if reuse is not None:
        data = json.loads(reuse.read_text())
        if data.get('schema') != SCHEMA or data.get('checkpoint_sha256') != checkpoint_hash:
            raise ValueError('reused workload schema/checkpoint does not match')
        # Coefficients and samples must not be silently changed on reuse.
    else:
        from .benchmark_real_gradients import gather_real_groups, make_samples
        groups, details, epsilon, denominator = gather_real_groups(
            run, policy, epoch, machines, 'saved-success', seed)
        if not groups:
            raise ValueError('no informative groups; increase --machines')
        samples = make_samples(groups, policy, states, seed)
        data = {'schema': SCHEMA, 'checkpoint_sha256': checkpoint_hash,
                'epsilon': epsilon, 'denominator': denominator,
                'source': details, 'sampling_seed': seed,
                'samples': [{k: row[k] for k in ('observation', 'counts', 'expected')}
                            for row in samples]}
    if not data.get('samples') or not math.isfinite(data['denominator']) or data['denominator'] <= 0:
        raise ValueError('empty workload or invalid loss denominator')
    if not math.isfinite(data['epsilon']) or not 0 <= data['epsilon'] < 1:
        raise ValueError('invalid exploration epsilon')
    for row in data['samples']:
        row['expected'] = {int(k): v for k, v in row['expected'].items()}
        n = len(row['observation']['actions'])
        if n == 0 or len(row['counts']) != n or any(
                not math.isfinite(x) for x in row['counts']):
            raise ValueError('invalid action coefficients')
        if any(not 0 <= k < n or not math.isfinite(v) or not 0 <= v <= 1
               for k, v in row['expected'].items()):
            raise ValueError('invalid sampled probability')
    return data


def chunks_for(samples, batch_size, heads, device, max_elements):
    """Identical, length-ordered microbatches for BOTH encoders; no truncation.

    Bound the full baseline's padded attention size, including unique actions.
    The limit is a heuristic, not a guarantee of GPU peak memory usage. An
    individual valid long input is not dropped to make a speed result prettier.
    """
    def texts(row):
        obs = row['observation']
        return [encode_json({k: v for k, v in obs.items() if k != 'actions'})] + [
            encode_json(a) for a in obs['actions']]
    descriptions = [texts(row) for row in samples]
    lengths = {text: len(cached_tree(text)[0])
               for group in descriptions for text in group}
    def padded(n):
        return ((n + 15) // 16) * 16 if device == 'mps' else n
    order = sorted(range(len(samples)), key=lambda i: (max(lengths[t] for t in descriptions[i]), i))
    def cost(indices):
        contexts = [descriptions[i][0] for i in indices]
        actions = list(dict.fromkeys(t for i in indices for t in descriptions[i][1:]))
        return max(len(ts) * heads * padded(max(lengths[t] for t in ts)) ** 2
                   for ts in (contexts, actions))
    groups, current = [], []
    for i in order:
        if current and (len(current) >= batch_size or cost(current + [i]) > max_elements):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return [[samples[i] for i in group] for group in groups], {
        'microbatches': len(groups), 'largest_microbatch': max(map(len, groups)),
        'largest_baseline_attention_elements': max(cost(g) for g in groups),
        'context_tokens_min': min(lengths[g[0]] for g in descriptions),
        'context_tokens_median': statistics.median(lengths[g[0]] for g in descriptions),
        'context_tokens_max': max(lengths[g[0]] for g in descriptions),
    }


def logits_for(policy, group):
    observations = [s['observation'] for s in group]
    return ([policy(observations[0])[0]] if len(group) == 1
            else policy.score_many(observations))


def gradient_pass(policy, groups, epsilon, denominator):
    policy.zero_grad(set_to_none=True)
    losses = []
    for group in groups:
        terms = []
        for row, logits in zip(group, logits_for(policy, group)):
            logps = behavior_log_probs(logits, row['observation'], epsilon)
            counts = torch.tensor(row['counts'], dtype=logps.dtype, device=logps.device)
            terms.append(-(logps * counts).sum() / denominator)
        loss = torch.stack(terms).sum()
        loss.backward()
        losses.append(loss.detach())
    # One scalar transfer for the whole pass, identical in both implementations.
    value = float(torch.stack(losses).sum())
    if not math.isfinite(value):
        raise RuntimeError('nonfinite benchmark loss')
    return value


def inference_pass(policy, samples):
    # Match the existing local worker's frozen action-feature cache. Separate
    # sampler/cache per machine; no cross-machine reuse or two-pass trick.
    samplers = {}
    for row in samples:
        obs = row['observation']
        machine = encode_json(obs['machine'])
        if machine not in samplers:
            samplers[machine] = FrozenSampler(policy, capacity=0)
        samplers[machine].probabilities(obs)


def capture_gradients(policy):
    return {name: (p.grad.detach().cpu().double().clone() if p.grad is not None
                   else torch.zeros(p.shape, dtype=torch.float64))
            for name, p in policy.named_parameters()}


def gradient_comparison(reference, candidate):
    squared_error = squared_norm = 0.
    largest = 0.
    failed = []
    for name, ref in reference.items():
        actual = candidate[name]
        difference = actual - ref
        squared_error += float(difference.square().sum())
        squared_norm += float(ref.square().sum())
        largest = max(largest, float(difference.abs().max()))
        if not torch.allclose(ref, actual, rtol=1e-4, atol=2e-7):
            failed.append(name)
    relative = math.sqrt(squared_error) / max(math.sqrt(squared_norm), 1e-12)
    return {'relative_l2': relative, 'max_absolute_error': largest,
            'reference_l2': math.sqrt(squared_norm),
            'parameters_compared': len(reference), 'failed_parameters': failed,
            'passes': not failed and (relative <= 1e-4 or largest <= 1e-8)}


def validate(policy, groups, epsilon, denominator, device):
    records = []
    for optimized in (False, True):
        cached_tree.cache_clear()
        with root_query_encoder(policy) if optimized else nullcontext():
            with torch.no_grad():
                logits, probabilities = [], []
                expected_error = 0.
                for group in groups:
                    for row, scores in zip(group, logits_for(policy, group)):
                        p = behavior_log_probs(scores, row['observation'], epsilon).exp()
                        logits.append(scores.cpu().double())
                        probabilities.append(p.cpu().double())
                        for i, expected in row['expected'].items():
                            expected_error = max(expected_error, abs(float(p[i]) - expected))
            loss = gradient_pass(policy, groups, epsilon, denominator)
            records.append((torch.cat(logits), torch.cat(probabilities),
                            loss, capture_gradients(policy), expected_error))
    old, new = records
    grads = gradient_comparison(old[3], new[3])
    rtol, atol = (2e-4, 2e-5) if device == 'mps' else (2e-5, 2e-6)
    finite = all(bool(torch.isfinite(x).all()) for x in (old[0], new[0], old[1], new[1]))
    passed = (finite and torch.allclose(old[0], new[0], rtol=rtol, atol=atol)
              and torch.allclose(old[1], new[1], rtol=rtol, atol=atol)
              and math.isclose(old[2], new[2], rel_tol=1e-4, abs_tol=1e-7)
              and grads['passes'] and max(old[4], new[4]) <= 3e-5)
    return {'passes': bool(passed), 'logit_max_absolute_error': float((old[0]-new[0]).abs().max()),
            'probability_max_absolute_error': float((old[1]-new[1]).abs().max()),
            'baseline_sampled_probability_max_error': old[4],
            'root_sampled_probability_max_error': new[4],
            'baseline_loss': old[2], 'root_loss': new[2], 'gradient': grads,
            'reference': 'original encoder on the SAME device; not cross-device equivalence'}


def measure_pair(policy, samples, groups, epsilon, denominator, *, device,
                 phase, warmups, repeats):
    timings = {'baseline': [], 'root-query': []}
    warm = {'baseline': [], 'root-query': []}
    for repeat in range(warmups + repeats):
        names = ('baseline', 'root-query') if repeat % 2 == 0 else ('root-query', 'baseline')
        for name in names:
            with root_query_encoder(policy) if name == 'root-query' else nullcontext():
                # Clear parsed syntax equally. Timed calls include JSON, tree
                # parsing, tensor preparation, attention, and (if asked) backward.
                cached_tree.cache_clear()
                synchronize(device)
                start = time.perf_counter()
                if phase == 'inference':
                    inference_pass(policy, samples)
                else:
                    gradient_pass(policy, groups, epsilon, denominator)
                synchronize(device)
                seconds = time.perf_counter() - start
            target = warm if repeat < warmups else timings
            target[name].append(seconds)
            emit('root_query_timing', device=device, phase=phase, variant=name,
                 warmup=repeat < warmups, repeat=repeat, seconds=round(seconds, 6))
    medians = {key: statistics.median(values) for key, values in timings.items()}
    spread = {key: max(values) / min(values) for key, values in timings.items()}
    stable = all(x <= 1.5 for x in spread.values())
    ratio = medians['baseline'] / medians['root-query']
    return {'median_seconds': medians, 'trial_seconds': timings, 'warmup_seconds': warm,
            'max_over_min': spread, 'timing_stable': stable,
            'median_ratio': ratio, 'validated_speedup': ratio if stable else None,
            'note': 'Repeated fixed workload, explicit warmups, alternating order. '
                    'Correctness checks already warmed kernels; no cold-start claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=2)
    p.add_argument('--machines', type=int, default=8)
    p.add_argument('--states', type=int, default=256, help='0 uses all available real states')
    p.add_argument('--devices', default='cpu', help='cpu or cpu,mps; unavailable MPS is skipped')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--mps-batch-size', type=int, default=8)
    p.add_argument('--max-attention-elements', type=int, default=8000000)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=37)
    p.add_argument('--reuse-workload', type=Path)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    devices = list(dict.fromkeys(s.strip() for s in args.devices.split(',')))
    if any(s not in ('cpu', 'mps') for s in devices):
        p.error('--devices must be cpu and/or mps')
    if min(args.epoch, args.machines, args.threads, args.mps_batch_size,
           args.max_attention_elements, args.warmups) < 1 or args.states < 0 or args.repeats < 3:
        p.error('positive counts required; repeats >=3, warmups >=1, states >=0')
    out = args.out or args.run / f'root_query_epoch_{args.epoch:02d}.json'
    workfile = out.with_suffix('.workload.json')
    if out.exists() or (args.reuse_workload is None and workfile.exists()):
        p.error('output already exists; use a new --out (nothing was overwritten)')
    checkpoint = args.run / f'model_epoch_{args.epoch:02d}.pt'
    report = {'status': 'incomplete', 'devices': {}, 'limitations': [
        'This is sampled inference/gradient replay, NOT a measured full epoch.',
        'CPU sampler timing uses per-machine frozen action caches, but no proof '
        'steps, multiprocessing or scheduling. MPS sampler timing is also synchronous.',
        'Gradient timing uses ordinary autograd, no two-pass method or optimizer step.',
        'Real coefficients include failed attempts; machine selection favors historical '
        'successes and is not an unbiased population sample.',
        'No 30-minute forecast is inferred from this benchmark.',
    ]}
    try:
        torch.set_num_threads(args.threads)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        base = Policy(**saved['config']).eval()
        base.load_state_dict(saved['state_dict'])
        if base.representation != 'ast':
            raise ValueError('requires an AST checkpoint')
        validate_encoder(base.structural_encoder)
        if not all(bool(torch.isfinite(t).all()) for t in base.state_dict().values()):
            raise ValueError('nonfinite checkpoint')
        sha = digest(checkpoint)
        report.update({'checkpoint': str(checkpoint), 'checkpoint_sha256': sha,
                       'model': saved['config'], 'settings': vars(args) | {'run': str(args.run),
                           'out': str(out), 'reuse_workload': str(args.reuse_workload) if args.reuse_workload else None},
                       'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                       'torch': str(torch.__version__), 'threads': torch.get_num_threads()}})
        # Path objects have been replaced above; no opaque runtime objects in JSON.
        write_json(out, report)
        data = workload(args.run, args.epoch, base, sha, machines=args.machines,
                        states=args.states, seed=args.seed, reuse=args.reuse_workload)
        if args.reuse_workload is None:
            write_json(workfile, data)
        report['workload_file'] = str(args.reuse_workload or workfile)
        samples = data['samples']
        report['sample'] = {'states': len(samples), 'source': data['source'],
                            'positive_coefficients': sum(v > 0 for s in samples for v in s['counts']),
                            'negative_coefficients': sum(v < 0 for s in samples for v in s['counts'])}
        write_json(out, report)
        if not report['sample']['positive_coefficients'] or not report['sample']['negative_coefficients']:
            raise ValueError('sample must contain BOTH advantage signs; increase --machines/--states')
        for device in devices:
            if device == 'mps' and not torch.backends.mps.is_available():
                report['devices'][device] = {'status': 'unavailable'}
                write_json(out, report)
                continue
            model = Policy(**saved['config']).to(device).eval()
            model.load_state_dict(saved['state_dict'])
            groups, lengths = chunks_for(samples, 1 if device == 'cpu' else args.mps_batch_size,
                                         saved['config']['heads'], device, args.max_attention_elements)
            result = {'status': 'checking', 'workload_shapes': lengths}
            report['devices'][device] = result
            emit('root_query_correctness_start', device=device, states=len(samples))
            result['correctness'] = validate(model, groups, data['epsilon'], data['denominator'], device)
            write_json(out, report)
            emit('root_query_correctness_done', device=device, **result['correctness'])
            if not result['correctness']['passes']:
                result['status'] = 'failed_correctness'
                raise RuntimeError(f'{device}: correctness failed; fast path is NOT approved')
            for phase in ('inference', 'gradient'):
                result[phase] = measure_pair(model, samples, groups, data['epsilon'], data['denominator'],
                                             device=device, phase=phase, warmups=args.warmups, repeats=args.repeats)
                write_json(out, report)
            if not all(torch.equal(t.cpu(), saved['state_dict'][k].cpu()) for k, t in model.state_dict().items()):
                raise RuntimeError('benchmark changed model weights')
            result['status'] = 'passed'
            write_json(out, report)
        if digest(checkpoint) != sha:
            raise RuntimeError('checkpoint changed during the experiment')
        report['status'] = 'complete'
        write_json(out, report)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        report.update(status='failed', error=str(exc))
        write_json(out, report)
        p.exit(2, f'Benchmark failed: {exc}\nPartial report: {out}\n')
    print('\nROOT-QUERY A/B SUMMARY (not a full-epoch forecast)')
    for device, row in report['devices'].items():
        if row['status'] != 'passed':
            print(device, row['status'])
            continue
        for phase in ('inference', 'gradient'):
            trial = row[phase]
            print(f"{device} {phase}: original={trial['median_seconds']['baseline']:.4f}s "
                  f"root={trial['median_seconds']['root-query']:.4f}s "
                  f"ratio={trial['median_ratio']:.2f}x stable={trial['timing_stable']}")
    print('Report:', out)
    print('Reusable workload:', report['workload_file'])


if __name__ == '__main__':
    main()
