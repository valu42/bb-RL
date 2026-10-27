"""Read-only CPU speed experiments for a completed proofgame branch-race epoch.

This script benchmarks the existing serial CPU gradient implementation against
batched score_many() on *real saved successful-proof observations*. No optimizer
step, checkpoint write, or environment change is performed. Saved training batch
JSONs intentionally omit failed group trajectories, so this is a representative
compute microbenchmark, NOT an exact replay of the original REINFORCE update.

Example (run when the live training process is stopped to avoid CPU contention):
    python -m proofgame.benchmark_training_speed \\
      --run runs/ast_branch_race_8x6_fixed --epoch 2 \\
      --batches 51,58,75 --states 48 --threads 1,2 \\
      --batch-sizes 2,4,8 --repeats 2 \\
      --out runs/ast_branch_race_8x6_fixed/speed_benchmark_epoch2.json
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time

import torch

from .environment import Game
from .exploration import behavior_log_probs
from .logic import Machine
from .model import Policy
from .network_search import encode_json
from .structural_encoding import cached_tree


def split_ints(value):
    values = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    if not values or any(value < 1 for value in values):
        raise ValueError('expected comma-separated positive integers')
    if len(values) != len(set(values)):
        raise ValueError('duplicate values')
    return values


def read_existing_profiles(run, epoch, workers):
    directory = run / 'profiling'
    profiles = [json.loads(p.read_text()) for p in sorted(
        directory.glob(f'epoch_{epoch:02d}_train_batch_*.json'))]
    if not profiles:
        raise FileNotFoundError('missing epoch training profiles; run with --profile')
    stage_totals = Counter()
    for p in profiles:
        stage_totals.update(p['stage_wall_seconds'])
    wall = sum(stage_totals.values())
    task_seconds = 0.
    machine_times = []
    for p in profiles:
        batch_number = p['batch_number']
        batch_start = (batch_number - 1) * int(p.get('machines', 8))
        # Normally batches are full. Read start_index from the file itself for
        # irregular final batches rather than relying on this approximation.
        batch_file = run / f'epoch_{epoch:02d}_train' / f'batch_{batch_start:04d}.json'
        if not batch_file.exists():
            candidates = sorted((run / f'epoch_{epoch:02d}_train').glob('batch_*.json'))
            batch_file = candidates[batch_number-1] if batch_number <= len(candidates) else batch_file
        if batch_file.exists():
            saved = json.loads(batch_file.read_text())
            ts = [float(m['seconds']) for m in saved['machines']]
            task_seconds += sum(ts)
            machine_times.extend(ts)
    eval_path = directory / f'epoch_{epoch:02d}_eval_summary.json'
    eval_seconds = None
    if eval_path.exists():
        report = json.loads(eval_path.read_text())
        eval_seconds = sum(report['stage_wall_seconds_sum'].values())
    rollout = stage_totals['rollout']
    learning = stage_totals['learn']
    result = {
        'profiled_batches':len(profiles),
        'stage_wall_seconds':dict(stage_totals),
        'training_seconds':round(wall,3),
        'evaluation_seconds':round(eval_seconds,3) if eval_seconds is not None else None,
        'full_epoch_seconds':round(wall+eval_seconds,3) if eval_seconds is not None else None,
        'machine_task_seconds':round(task_seconds,3),
        'worker_task_occupancy':round(task_seconds/(workers*rollout),4) if rollout else None,
        'ideal_rollout_floor_seconds_from_task_times':round(task_seconds/workers,3),
        'machine_seconds_median':round(statistics.median(machine_times),4) if machine_times else None,
    }
    return result, profiles


def pick_batches(profiles, explicit, maximum=3):
    available = {p['batch_number']:p for p in profiles}
    if explicit:
        missing = sorted(set(explicit) - set(available))
        if missing:
            raise ValueError(f'unknown batch numbers: {missing}')
        return explicit
    by_learn = sorted(profiles, key=lambda p:p['stage_wall_seconds']['learn'])
    return tuple(dict.fromkeys((by_learn[-1]['batch_number'],
                                by_learn[len(by_learn)//2]['batch_number'],
                                by_learn[0]['batch_number'])))[:maximum]


def collect_success_states(run, epoch, batches, max_tokens, cap, seed):
    dataset = json.loads((run/'dataset.json').read_text())
    budget = dataset['action_budget']
    train_folder = run/f'epoch_{epoch:02d}_train'
    files = sorted(train_folder.glob('batch_*.json'))
    if not files:
        raise FileNotFoundError('no saved training batches')
    by_number = {i+1:p for i,p in enumerate(files)}
    candidates = {}
    actions_replayed = successes = long_contexts = 0
    source_machines = 0
    for batch in batches:
        path = by_number.get(batch)
        if path is None:
            raise ValueError(f'no training batch file #{batch}')
        contents = json.loads(path.read_text())
        for row in contents['machines']:
            if not row['successes']:
                continue
            source_machines += 1
            machine = Machine(row['machine']['rows'])
            for success in row['successes']:
                successes += 1
                game = Game(machine,budget=budget)
                for step in success['actions']:
                    obs = game.observation()
                    action_ids = [a['id'] for a in obs['actions']]
                    if step['action'] not in action_ids:
                        raise RuntimeError('saved successful proof no longer replays: '+step['action'])
                    index = action_ids.index(step['action'])
                    serialized = encode_json(obs)
                    if serialized not in candidates:
                        ctx = {k:v for k,v in obs.items() if k!='actions'}
                        ctx_str = encode_json(ctx)
                        n = len(cached_tree(ctx_str)[0])
                        if n <= max_tokens:
                            candidates[serialized] = (obs,index,n)
                        else:
                            long_contexts += 1
                    game.step(step['action'],observe=False)
                    actions_replayed += 1
                if game.status != 'SOLVED':
                    raise RuntimeError('saved verified proof did not replay to SOLVED')
    all_items = list(candidates.values())
    lengths = [item[2] for item in all_items]
    # Random sample, independent of the model probabilities. Preserve the same
    # state set and gradient weights for ALL benchmark configurations.
    rng = random.Random(seed)
    if len(all_items) > cap:
        all_items = rng.sample(all_items,cap)
    result = []
    for i,(obs,index,tokens) in enumerate(all_items):
        coefficient = (-1. if i%2 else 1.) * (0.5+(i%7)/7.)
        result.append({'observation':obs,'index':index,
                       'tokens':tokens,'coefficient':coefficient})
    stats = {
        'batches':list(batches),
        'source_solved_machines':source_machines,
        'saved_successes_examined':successes,
        'proof_actions_replayed':actions_replayed,
        'distinct_states_within_length_limit':len(candidates),
        'contexts_too_long':long_contexts,
        'sampled_states':len(result),
        'source_token_median':statistics.median(lengths) if lengths else None,
        'source_token_max':max(lengths) if lengths else None,
        'sample_token_median':statistics.median([x['tokens'] for x in result]) if result else None,
        'sample_token_max':max((x['tokens'] for x in result),default=None),
        'caveat':'Saved failed group trajectories are not retained; sampled states come only from successful proof attempts, and their coefficients are synthetic. Speed/gradient-equivalence microbenchmark only, not a training or generalization score.',
    }
    return result,stats


def execute_gradient(policy,items,batch_size,threads,bucket,epsilon):
    torch.set_num_threads(threads)
    ordered = sorted(items,key=lambda r:r['tokens']) if bucket else list(items)
    policy.zero_grad(set_to_none=True)
    total = 0.
    forward_time=backward_time=0.
    begin=time.perf_counter()
    for start in range(0,len(ordered),batch_size):
        chunk = ordered[start:start+batch_size]
        t0=time.perf_counter()
        if batch_size == 1:
            logits = [policy(chunk[0]['observation'])[0]]
        else:
            logits = policy.score_many([r['observation'] for r in chunk])
        losses=[-r['coefficient'] * behavior_log_probs(logit,r['observation'],epsilon)[r['index']]
                /len(items) for r,logit in zip(chunk,logits)]
        loss=torch.stack(losses).sum()
        forward_time+=time.perf_counter()-t0
        t0=time.perf_counter()
        loss.backward()
        backward_time+=time.perf_counter()-t0
        total+=float(loss.detach())
    seconds=time.perf_counter()-begin
    with torch.no_grad():
        # Individual gradient tensors are small for the current network;
        # compare every parameter, including structural relation biases.
        gradients = torch.cat([
            p.grad.detach().float().reshape(-1).clone() if p.grad is not None
            else torch.zeros_like(p, dtype=torch.float32).reshape(-1)
            for p in policy.parameters()
        ])
    return {'wall_seconds':seconds,'forward_seconds':forward_time,
            'backward_seconds':backward_time,'loss':total},gradients


def planned_chunk_size(items,batch_size,bucket):
    ordered = sorted(items,key=lambda r:r['tokens']) if bucket else list(items)
    return max((len(chunk)*(max(r['tokens'] for r in chunk)**2)
                for start in range(0,len(ordered),batch_size)
                if (chunk:=ordered[start:start+batch_size])),default=0)


def run_benchmarks(policy, items, thread_counts, batch_sizes, repeats, epsilon,
                   max_attention_elements):
    if not items:
        raise ValueError('no saved successful proof states; pick different batches')
    results=[]
    for threads in thread_counts:
        base_grad = None
        base_seconds = None
        variants = [(1,False)]+[(b,bucket) for b in batch_sizes for bucket in (False,True)]
        for batch_size,bucket in variants:
            estimate = planned_chunk_size(items,batch_size,bucket)
            label='serial' if batch_size==1 else ('bucketed' if bucket else 'unsorted')
            print(f'Benchmark: threads={threads} batch={batch_size} mode={label}',flush=True)
            if estimate > max_attention_elements:
                results.append({'threads':threads,'batch_size':batch_size,'mode':label,
                                'status':'skipped_memory_guard',
                                'max_batch_token_pairs':estimate})
                continue
            # An untimed warm-up avoids counting one-time kernel initialization.
            warmup, _ = execute_gradient(policy,items,batch_size,threads,bucket,epsilon)
            timed=[]
            gradients=None
            for _ in range(repeats):
                stat,gradients = execute_gradient(policy,items,batch_size,threads,bucket,epsilon)
                timed.append(stat)
            seconds=statistics.median(x['wall_seconds'] for x in timed)
            loss=timed[0]['loss']
            report={'threads':threads,'batch_size':batch_size,'mode':label,
                    'status':'ok','states':len(items),'median_wall_seconds':round(seconds,4),
                    'median_forward_seconds':round(statistics.median(x['forward_seconds'] for x in timed),4),
                    'median_backward_seconds':round(statistics.median(x['backward_seconds'] for x in timed),4),
                    'loss':round(loss,8),'max_batch_token_pairs':estimate,
                    'warmup_wall_seconds':round(warmup['wall_seconds'],4)}
            if batch_size==1:
                base_grad=gradients
                base_seconds=seconds
                report['relative_gradient_l2']=0.
                report['gradient_max_absolute_error']=0.
                report['speedup_vs_same_thread_serial']=1.
            else:
                delta=(gradients-base_grad).detach()
                report['relative_gradient_l2']=round(float(torch.linalg.vector_norm(delta) /
                                                    torch.linalg.vector_norm(base_grad).clamp_min(1e-12)),8)
                report['gradient_max_absolute_error']=round(float(delta.abs().max()),8)
                report['speedup_vs_same_thread_serial']=round(base_seconds/seconds,3)
            results.append(report)
            print(json.dumps({'event':'gradient_benchmark',**report}),flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--epoch',type=int,default=2)
    p.add_argument('--batches',type=str,default='',
                   help='comma-separated 1-based batch numbers, e.g. 51,58,75; default chooses slow, median and fast gradient batches')
    p.add_argument('--states',type=int,default=48,
                   help='number of distinct saved successful-proof states used for each gradient benchmark')
    p.add_argument('--max-context-tokens',type=int,default=800)
    p.add_argument('--max-attention-elements',type=int,default=4000000,
                   help='guard against large padded attention batches; skips unsafe combinations')
    p.add_argument('--threads',type=str,default='1,2')
    p.add_argument('--batch-sizes',type=str,default='2,4,8')
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--workers',type=int,default=8,
                   help='workers used in the recorded training run (for occupancy estimate)')
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--out',type=Path,
                   help='JSON output path (default: <run>/speed_benchmark_epoch_NN.json)')
    args=p.parse_args()
    if min(args.epoch,args.states,args.max_context_tokens,args.max_attention_elements,
           args.repeats,args.workers)<1:
        p.error('all count arguments must be positive')
    threads=split_ints(args.threads)
    batch_sizes=split_ints(args.batch_sizes)
    if 1 in batch_sizes:
        p.error('batch-sizes should contain values greater than one; serial is automatically measured')
    config=json.loads((args.run/'config.json').read_text())
    history,profiles=read_existing_profiles(args.run,args.epoch,args.workers)
    batch_numbers=pick_batches(profiles,split_ints(args.batches) if args.batches else ())
    items,sample=collect_success_states(args.run,args.epoch,batch_numbers,
                                        args.max_context_tokens,args.states,args.seed)
    model_file=args.run/f'model_epoch_{args.epoch:02d}.pt'
    if not model_file.exists():
        raise FileNotFoundError(f'{model_file} not found; use a fully completed epoch')
    saved=torch.load(model_file,map_location='cpu',weights_only=True)
    torch.set_num_threads(1)
    policy=Policy(**saved['config']).eval()
    policy.load_state_dict(saved['state_dict'])
    epsilon=float(config.get('train_exploration_epsilon',0.))
    print(f'Using frozen checkpoint {model_file} and {len(items)} successful-proof states',flush=True)
    measurements=run_benchmarks(policy,items,threads,batch_sizes,args.repeats,
                                epsilon,args.max_attention_elements)
    candidates=[r for r in measurements if r['status']=='ok']
    baseline=next((r for r in candidates if r['threads']==1 and r['batch_size']==1),None)
    fastest=min(candidates,key=lambda r:r['median_wall_seconds'])
    result={
        'note':'Read-only microbenchmarks: no optimizer step and no stochastic rollouts',
        'system':{'machine':platform.machine(),'python':platform.python_version(),
                  'pytorch':torch.__version__,'cpu_threads_options':list(threads)},
        'run':str(args.run),'epoch':args.epoch,'model':str(model_file),
        'history':history,'sample':sample,'gradient_benchmarks':measurements,
        'best':{'threads':fastest['threads'],'batch_size':fastest['batch_size'],
                'mode':fastest['mode'],'median_wall_seconds':fastest['median_wall_seconds']},
    }
    if baseline:
        best_ratio=baseline['median_wall_seconds']/fastest['median_wall_seconds']
        result['best']['speedup_vs_current_1_thread_serial']=round(best_ratio,3)
        # Extrapolation holds only if the sampled successful-proof states are
        # representative of the full (including failed) gradient workload.
        target = (history['stage_wall_seconds']['rollout']+
                  history['stage_wall_seconds']['learn']/best_ratio+
                  (history['evaluation_seconds'] or 0.))
        result['hypothetical_full_epoch_minutes_if_learning_only_optimized']=round(target/60,2)
    output=args.out or (args.run/f'speed_benchmark_epoch_{args.epoch:02d}.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print('\n=== SPEED BENCHMARK SUMMARY ===')
    print('Current full epoch (minutes):', round((history['full_epoch_seconds'] or
                                                   history['training_seconds'])/60,2))
    print('Benchmark input states:',len(items),'from batches:',*batch_numbers)
    print('Fastest configuration:',result['best'])
    print('\nFASTEST SIX CONFIGURATIONS')
    for row in sorted(candidates, key=lambda row:row['median_wall_seconds'])[:6]:
        rel = row.get('relative_gradient_l2',0.)
        print(f"  {row['threads']} thread(s), batch={row['batch_size']}, "
              f"{row['mode']:<8} {row['median_wall_seconds']:.3f}s "
              f"| grad relative L2={rel:.2e}")
    valid = [r for r in candidates if r['batch_size']>1]
    if valid:
        worst=max(valid, key=lambda row:row['relative_gradient_l2'])
        print('Maximum relative gradient difference:',
              f"{worst['relative_gradient_l2']:.2e}")
        if worst['relative_gradient_l2'] > 1e-4:
            print('CAUTION: A batched variant differs numerically from serial; review before enabling it.')
    print('WARNING: Saved failed trajectories are absent; predicted epoch time is only an extrapolation.')
    print('JSON REPORT:',output)


if __name__=='__main__':
    main()
