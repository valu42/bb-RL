"""Compare frozen-policy CPU rollout throughput without changing training.

Run with:
 python -m proofgame.benchmark_cpu_inference --dataset runs/unfiltered_1000_100/dataset.json \
     --model runs/ast_masked_actions/state.pt --machines 24 --attempts 10 \
     --workers 8 --batch-size 8 --representation ast

The same machines, weights and seeds are used in every mode. Worker
startup is excluded from measured rollout time; numerical differences
may produce different stochastic trajectories, so both actions/s and
proof/attempt counts are reported. No optimizer steps or checkpoint writes.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import tempfile
import time

import torch

from .epoch_train import init_worker, run_machine, task_seed
from .gpu_runtime import CentralRolloutPool
from .model import Policy


def compare_modes(dataset, policy, *, machines=24, attempts=10, workers=8,
                  batch_size=8, inference_batch_size=8, wait_ms=2.,
                  seed=7, balanced=False, modes=('baseline','shared-cache','central')):
    if min(machines,attempts,workers,batch_size,inference_batch_size)<1:
        raise ValueError('positive benchmark counts required')
    if balanced and attempts%2:
        raise ValueError('balanced attempts must be even')
    rows=dataset['test'][:machines]
    if not rows:
        raise ValueError('no benchmark machines')
    budget=dataset['action_budget']
    torch.set_num_threads(1)
    report=[]
    with tempfile.TemporaryDirectory(prefix='proofgame-benchmark-') as tmp:
        snapshot=Path(tmp)/'frozen.pt'
        torch.save({'version':0,'config':policy.config,
                    'state_dict':policy.state_dict()},snapshot)
        for mode in modes:
            if mode not in ('baseline','shared-cache','central'):
                raise ValueError('unknown mode '+mode)
            if mode=='central':
                runner=CentralRolloutPool(policy,workers=workers,
                                          inference_batch_size=inference_batch_size,
                                          wait_ms=wait_ms)
            else:
                runner=mp.get_context('spawn').Pool(
                    workers,initializer=init_worker,
                    initargs=(policy.config,mode=='shared-cache'),
                    maxtasksperchild=50)
            with runner as pool:
                start=time.perf_counter()
                results=[]
                for offset in range(0,len(rows),batch_size):
                    batch=rows[offset:offset+batch_size]
                    if mode=='central':
                        tasks=[(r,task_seed(seed,'eval',0,r['machine_id']),attempts,
                                budget,0,balanced,0.,balanced,False,True)
                               for r in batch]
                        results.extend(pool.run(tasks))
                    else:
                        tasks=[(r,task_seed(seed,'eval',0,r['machine_id']),attempts,
                                budget,str(snapshot),0,balanced,0.,balanced,False,True)
                               for r in batch]
                        results.extend(pool.map(run_machine,tasks,chunksize=1))
                elapsed=time.perf_counter()-start
                central_stats=pool.inference.stats() if mode=='central' else None
            actions=sum(r['actions'] for r in results)
            total_attempts=sum(r['attempts'] for r in results)
            proofs=sum(bool(r['successes']) for r in results)
            hits=sum(r['profile'].get('action_feature_hits',0) for r in results)
            misses=sum(r['profile'].get('action_feature_misses',0) for r in results)
            row={'mode':mode,'seconds':round(elapsed,3),'machines':len(results),
                 'attempts':total_attempts,'actions':actions,'proved_machines':proofs,
                 'actions_per_second':round(actions/elapsed,2),
                 'attempts_per_second':round(total_attempts/elapsed,2),
                 'action_embedding_cache_hits':hits,
                 'action_embedding_cache_misses':misses}
            if central_stats is not None:
                row['central_inference']=central_stats
            report.append(row)
            print(json.dumps({'event':'benchmark_mode',**row}),flush=True)
    baseline=next((r for r in report if r['mode']=='baseline'),None)
    if baseline:
        for row in report:
            row['throughput_vs_baseline']=round(
                row['actions_per_second']/baseline['actions_per_second'],3)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--model',type=Path,
                        help='existing initial_model.pt or state.pt (read-only)')
    parser.add_argument('--machines',type=int,default=24)
    parser.add_argument('--attempts',type=int,default=10)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--inference-batch-size',type=int,default=8)
    parser.add_argument('--inference-wait-ms',type=float,default=2.)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--representation',choices=('bytes','ast'),default='ast')
    parser.add_argument('--balanced-initial-answers',action='store_true',
                        help='use forced 50/50 train-style attempts and stop after first proof')
    parser.add_argument('--modes',default='baseline,shared-cache,central',
                        help='comma-separated: baseline,shared-cache,central')
    parser.add_argument('--out',type=Path,help='optional JSON summary path')
    args=parser.parse_args()
    dataset=json.loads(args.dataset.read_text())
    if args.model:
        saved=torch.load(args.model,map_location='cpu',weights_only=True)
        policy=Policy(**saved['config']).eval()
        policy.load_state_dict(saved['state_dict'])
    else:
        torch.manual_seed(args.seed)
        policy=Policy(width=32,layers=1,representation=args.representation).eval()
    result=compare_modes(
        dataset,policy,machines=args.machines,attempts=args.attempts,
        workers=args.workers,batch_size=args.batch_size,
        inference_batch_size=args.inference_batch_size,
        wait_ms=args.inference_wait_ms,seed=args.seed,
        balanced=args.balanced_initial_answers,
        modes=tuple(args.modes.split(',')))
    print('\nFROZEN CPU BENCHMARK (startup excluded, no gradient updates)')
    print(f"{'Mode':<15} {'Actions/s':>12} {'vs baseline':>12} {'Proofs':>8}")
    for row in result:
        print(f"{row['mode']:<15} {row['actions_per_second']:>12,.0f} "
              f"{row.get('throughput_vs_baseline',0):>11.2f}x "
              f"{row['proved_machines']:>8}")
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print('Saved:',args.out)


if __name__=='__main__':
    main()
