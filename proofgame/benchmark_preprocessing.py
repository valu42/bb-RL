"""Profile structural syntax, relation matrices, tensor preparation, and kernels.

This is deliberately more diagnostic than an epoch benchmark. It uses saved
successful-proof state shapes to answer whether optimizing the Python tree /
relation builder (potential C++ target) could beat optimizing dense attention,
or whether input caching is limited by a low reuse rate. Runs on a frozen model.

Example:
 python -m proofgame.benchmark_preprocessing \
  --run runs/ast_branch_race_8x6_fixed --epoch 2 --batches 51,58,75 \
  --states 256 --threads 1,2,4 --repeats 3 --torch-profiler \
  --out runs/ast_branch_race_8x6_fixed/preprocessing_profile.json
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import time

import torch

from .benchmark_training_speed import (
    collect_success_states, split_ints, read_existing_profiles, pick_batches)
from .experimental_structural_cache import (
    StructuralPreprocessingCache, cached_structural_forward)
from .model import Policy
from .network_search import encode_json
from .structural_encoding import cached_tree, relation_ids


def state_texts(items):
    observations = [item['observation'] for item in items]
    contexts = []
    actions = []
    for obs in observations:
        contexts.append(encode_json({k: v for k, v in obs.items()
                                     if k != 'actions'}))
        actions.extend(encode_json(a) for a in obs['actions'])
    return contexts, actions


def bench(call, repeats):
    wall = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        wall.append(time.perf_counter()-started)
    return {'median_seconds': statistics.median(wall),
            'repeat_seconds': wall}


def structural_diagnostics(texts, repeats, cache_megabytes):
    """Isolate only cached-tree parsing and CPU NxN structural relation work."""
    all_lengths = []
    for text in texts:
        all_lengths.append(len(cached_tree(text)[0]))
    counts = Counter(texts)
    unique = list(counts)
    shapes = [len(cached_tree(text)[0]) for text in unique]
    by_length = {
        '<=128': sum(n <= 128 for n in all_lengths),
        '129-256': sum(128 < n <= 256 for n in all_lengths),
        '257-512': sum(256 < n <= 512 for n in all_lengths),
        '>512': sum(n > 512 for n in all_lengths),
    }
    result = {
        'total_inputs': len(texts),
        'distinct_inputs': len(unique),
        'potential_reuse_fraction': 1.-len(unique)/len(texts) if texts else 0.,
        'length_histogram': by_length,
        'length_min': min(all_lengths) if texts else None,
        'length_median': statistics.median(all_lengths) if texts else None,
        'length_max': max(all_lengths) if texts else None,
        'dense_relation_bytes_uncached_per_full_pass': sum(n*n*8 for n in all_lengths),
        'dense_relation_bytes_unique_inputs': sum(n*n*8 for n in shapes),
    }

    def syntax_cold():
        cached_tree.cache_clear()
        for text in texts:
            cached_tree(text)

    def syntax_hot():
        for text in texts:
            cached_tree(text)

    def relations_uncached():
        for text in texts:
            _, parents, bindings = cached_tree(text)
            relation_ids(parents, bindings)

    cache = StructuralPreprocessingCache(cache_megabytes)
    def relations_cached():
        for text in texts:
            _, parents, bindings = cached_tree(text)
            cache.relation(text, parents, bindings)

    # Reset cache between test kinds rather than relying on configuration order.
    result['syntax_cold'] = bench(syntax_cold, repeats)
    result['syntax_warm'] = bench(syntax_hot, repeats)
    result['relations_uncached'] = bench(relations_uncached, repeats)
    result['relations_cached'] = bench(relations_cached, repeats)
    result['cache_stats'] = cache.stats()
    return result


def model_compute(policy, contexts, actions, batches, repeats, cache_mb):
    """Compare forward/inference encoding with bounded structural cache."""
    # Shape-bucket each category; only compare identical inputs per mode.
    cases = []
    for name, texts in (('contexts', contexts), ('actions', actions)):
        unique = list(dict.fromkeys(texts))
        unique.sort(key=lambda text: len(cached_tree(text)[0]))
        for size in batches:
            groups = [unique[i:i+size] for i in range(0, len(unique), size)]
            for cached in (False, True):
                if not groups:
                    continue
                cache_context = (
                    cached_structural_forward(
                        policy, relation_cache_mb=cache_mb)
                    if cached else None)
                if cache_context is None:
                    from contextlib import nullcontext
                    cache_context = nullcontext(None)
                with cache_context as cache:
                    with torch.inference_mode():
                        # Warmup all shapes once before measuring.
                        for group in groups:
                            policy.encode(group)
                        def run():
                            for group in groups:
                                policy.encode(group)
                        timing = bench(run, repeats)
                row = {'kind': name, 'group_size': size, 'cached': cached,
                       'unique_input_count': len(unique),
                       'number_of_encoder_calls': len(groups), **timing}
                if cache is not None:
                    row['cache_stats'] = cache.stats()
                cases.append(row)
                print(json.dumps({'event': 'model_encode_case', **row}),
                      flush=True)
    return cases


def profiler_top_ops(policy, contexts, count):
    if not contexts or count < 1:
        return []
    selected = contexts[:count]
    with torch.inference_mode():
        policy.encode([selected[0]])
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True, profile_memory=True) as prof:
            for text in selected:
                policy.encode([text])
    rows = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total,
                  reverse=True)
    return [{
        'operator': event.key,
        'calls': event.count,
        'self_cpu_time_ms': event.self_cpu_time_total / 1000,
        'total_cpu_time_ms': event.cpu_time_total / 1000,
        'self_cpu_memory_bytes': event.self_cpu_memory_usage,
    } for event in rows[:25]]


def benchmark_preprocessing(run, epoch, batches, sample_size, max_tokens,
                            threads, encoding_batches, repeats, cache_mb,
                            torch_profiler):
    config = json.loads((run / 'config.json').read_text())
    history, profiles = read_existing_profiles(run, epoch, config.get('workers', 8))
    chosen = pick_batches(profiles, tuple(batches) if batches else ())
    model_path = run / f'model_epoch_{epoch:02d}.pt'
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    policy = Policy(**saved['config']).eval()
    policy.load_state_dict(saved['state_dict'])
    if policy.representation != 'ast':
        raise ValueError('structural analysis requires AST representation')
    items, selection = collect_success_states(
        run, epoch, chosen, max_tokens, sample_size, seed=41)
    if not items:
        raise ValueError('no sample proof observations')
    contexts, actions = state_texts(items)
    report = {
        'type': 'CPU structural preprocessing vs encoder kernel experiment',
        'checkpoint': str(model_path),
        'selection': selection,
        'settings': {
            'epoch': epoch, 'repeats': repeats,
            'threads': threads, 'encoding_batches': encoding_batches,
            'cache_megabytes': cache_mb,
        },
        'historical_training_seconds': history['training_seconds'],
        'diagnostics': {},
        'encode_cases': [],
        'profiler': None,
        'note': 'torch.profiler timings have significant instrumentation '
                'overhead; use non-profiled median_seconds for speed comparisons.',
    }
    for num_threads in threads:
        torch.set_num_threads(num_threads)
        texts = contexts + actions
        print(f'Preprocessing Python/text analysis, threads={num_threads}',
              flush=True)
        report['diagnostics'][str(num_threads)] = structural_diagnostics(
            texts, repeats, cache_mb)
        report['encode_cases'].extend([
            {'threads': num_threads, **row}
            for row in model_compute(policy, contexts, actions, encoding_batches,
                                     repeats, cache_mb)
        ])
    if torch_profiler:
        torch.set_num_threads(1)
        report['profiler'] = profiler_top_ops(policy, contexts, min(12, len(contexts)))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=2)
    p.add_argument('--batches', default='')
    p.add_argument('--states', type=int, default=256)
    p.add_argument('--max-context-tokens', type=int, default=800)
    p.add_argument('--threads', default='1,2,4')
    p.add_argument('--encoding-batches', default='1,2,4,8')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--cache-mb', type=int, default=128)
    p.add_argument('--torch-profiler', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if min(args.epoch, args.states, args.max_context_tokens, args.repeats) < 1:
        p.error('positive counts required')
    if args.cache_mb < 0:
        p.error('negative cache size')
    try:
        result = benchmark_preprocessing(
            args.run, args.epoch,
            split_ints(args.batches) if args.batches else None,
            args.states, args.max_context_tokens,
            split_ints(args.threads), split_ints(args.encoding_batches),
            args.repeats, args.cache_mb, args.torch_profiler)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('\nSTRUCTURAL HOTSPOTS')
    for threads, metrics in result['diagnostics'].items():
        uncached = metrics['relations_uncached']['median_seconds']
        cached = metrics['relations_cached']['median_seconds']
        print(f"threads={threads}: relation creation {uncached:.3f}s "
              f"vs cache {cached:.3f}s ({uncached/cached:.2f}x) "
              f"potential raw-input reuse {metrics['potential_reuse_fraction']:.1%}")
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
