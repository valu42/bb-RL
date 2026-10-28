"""Read-only A/B of production CPU rollout + full optimizer-step execution.

Uses a checkpoint from --run, never writes into or trains that run. Both
schedulers use the new per-attempt streams. Reports a sampled batch, not an
epoch forecast. Worker startup/warmups are separate from timed repetitions.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import tempfile
import time

import torch

from .benchmark_rollout_matrix import digest_results
from .benchmark_speed_v2 import _compare_vectors
from .epoch_train import atomic_json, init_worker, learn_batch, task_seed
from .model import Policy
from .parallel_rollouts import run_local_rollouts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--epoch',type=int,default=2)
    p.add_argument('--machines',type=int,default=8)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if min(args.epoch,args.machines,args.workers,args.repeats)<1 or args.out.exists():
        p.error('positive counts and a new --out path required')
    cfg=json.loads((args.run/'config.json').read_text())
    if not cfg.get('branch_rounds') or not cfg.get('group_relative'):
        p.error('this focused A/B requires a group-relative branch-race run')
    checkpoint=args.run/f'model_epoch_{args.epoch:02d}.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['config'].get('representation')!='ast' or saved['config']['layers']!=1:
        p.error('a one-layer AST checkpoint is required')
    dataset=json.loads((args.run/'dataset.json').read_text())
    # Select uniformly from the entire training set, without favoring successes.
    import random
    rows=random.Random(cfg['seed']+args.epoch).sample(
        dataset['train'],min(args.machines,len(dataset['train'])))
    report={'status':'running','checkpoint':str(checkpoint),
            'machines':len(rows),'workers':args.workers,'warmups':[], 'trials':[],
            'note':'Sampled frozen-checkpoint batches with fresh identical AdamW states; '
                   'not a full epoch or retained-quality measurement. Snapshot writes '
                   'and pool startup are outside the timed rollout+learning stages. '
                   'Both variants use per-attempt-v1 streams, not historical streams.'}
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='proofgame-runtime-') as tmp:
        for repeat in range(-1,args.repeats):
            trial={}
            names=['baseline','optimized'] if repeat%2==0 else ['optimized','baseline']
            models={}; hashes={}
            for name in names:
                optimized=name=='optimized'
                model=Policy(**{**saved['config'],
                                'encoder_backend':'root-query' if optimized else 'full'}).eval()
                model.load_state_dict(saved['state_dict'])
                models[name]=model
                snapshot=Path(tmp)/f'{name}.pt'
                torch.save({'version':repeat,'state_dict':model.state_dict()},snapshot)
                jobs=[(row,task_seed(cfg['seed'],'train',args.epoch,row['machine_id']),
                       cfg['attempts_per_machine'],dataset['action_budget'],str(snapshot),repeat,
                       False,cfg.get('train_exploration_epsilon',0.),True,False,True,True,
                       cfg['branch_rounds']) for row in rows]
                with mp.get_context('spawn').Pool(
                        args.workers,initializer=init_worker,initargs=(model.config,True)) as pool:
                    # Warm the actual worker pool before timing. Caches are
                    # invalidated by the subsequent snapshot version change.
                    warm_snapshot=Path(tmp)/f'{name}_warm.pt'
                    torch.save({'version':-99,'state_dict':model.state_dict()},warm_snapshot)
                    warm_jobs=[(*job[:4],str(warm_snapshot),-99,*job[6:]) for job in jobs]
                    startup=time.perf_counter()
                    run_local_rollouts(pool,warm_jobs,workers=args.workers,
                                       split_attempts=optimized,independent_attempt_seeds=True)
                    startup=time.perf_counter()-startup
                    started=time.perf_counter()
                    results=run_local_rollouts(pool,jobs,workers=args.workers,
                                               split_attempts=optimized,
                                               independent_attempt_seeds=True)
                    rollout=time.perf_counter()-started
                optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=0.)
                started=time.perf_counter()
                update=learn_batch(
                    model,optimizer,results,dataset['action_budget'],
                    gpu_gradient_batch_size=8,gradient_backend='two-pass' if optimized else 'serial',
                    exploration_epsilon=cfg.get('train_exploration_epsilon',0.),
                    balanced_initial_answers=True,group_relative=True,
                    length_penalty=cfg.get('length_penalty',0.),branch_rounds=cfg['branch_rounds'])
                learning=time.perf_counter()-started
                hashes[name]=digest_results(results)
                trial[name]={'rollout_seconds':rollout,'learn_seconds':learning,
                             'total_seconds':rollout+learning,'update':update,
                             'worker_startup_and_warmup_seconds':startup,
                             'attempts':sum(r['attempts'] for r in results),
                             'actions':sum(r['actions'] for r in results)}
            trial['same_trajectories']=hashes['baseline']==hashes['optimized']
            trial['gradient_comparison']=_compare_vectors(models['baseline'],models['optimized'])
            trial['passes']=(trial['same_trajectories'] and
                             trial['baseline']['update']['updated']==trial['optimized']['update']['updated'] and
                             trial['gradient_comparison']['relative_gradient_l2']<1e-4 and
                             trial['gradient_comparison']['max_absolute_parameter_difference']<=max(2*cfg['lr'],1e-6))
            report['warmups' if repeat<0 else 'trials'].append(trial)
            atomic_json(args.out,report)
            print(json.dumps({'event':'runtime_benchmark_trial','repeat':repeat,**trial}),flush=True)
    trials=report['trials']
    medians={name:statistics.median(t[name]['total_seconds'] for t in trials)
             for name in ('baseline','optimized')}
    stable=all(max(t[n]['total_seconds'] for t in trials)/min(t[n]['total_seconds'] for t in trials)<=1.5
               for n in medians)
    updated=all(t[n]['update']['updated'] for t in trials+report['warmups'] for n in medians)
    report.update(status='complete',median_batch_seconds=medians,timing_stable=stable,
                  full_gradient_update_exercised=updated,
                  validated_speedup=(medians['baseline']/medians['optimized']
                                     if stable and updated and len(trials)>=3 and all(t['passes'] for t in trials+report['warmups']) else None))
    atomic_json(args.out,report)
    print(json.dumps({'event':'runtime_benchmark_summary',**{k:report[k] for k in
                     ('median_batch_seconds','timing_stable','validated_speedup')}}),flush=True)


if __name__=='__main__':
    main()
