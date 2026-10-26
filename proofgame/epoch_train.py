"""Resumable epoch training on fixed, disjoint train/test datasets.

Evaluation always runs the exact configured number of independent-reset
attempts per machine. Training can optionally stop on its first verified proof.
Rollout batches share frozen weights; only training successes supply REINFORCE
gradients. Evaluation never updates parameters. Selector proofs are not accepted
in the dataset. Checkpoints commit after every rollout batch.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import time

import torch

from .environment import Game
from .kernel import check
from .logic import Machine
from .model import Policy
from .network_search import FrozenSampler, attempt, encode_json
from .population_sample import from_rank
from .tasks import is_test, key
from .unfiltered_dataset import DATASET_FORMAT as UNFILTERED_FORMAT, POPULATION


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def atomic_torch(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp); temp.replace(path)


def acquire_run_lock(path):
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0'); handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('this run is already active')
    return handle


def task_seed(seed, phase, epoch, machine_id):
    # Evaluation uses the same seeds each epoch; it is a fixed monitoring set.
    epoch = epoch if phase == 'train' else 0
    return int.from_bytes(hashlib.sha256(f'{seed}:{phase}:{epoch}:{machine_id}'.encode()).digest()[:8], 'big')


_worker_policy = None


def init_worker(model_config):
    global _worker_policy
    torch.set_num_threads(1)
    _worker_policy = Policy(**model_config).eval()


def run_machine(task):
    selected, seed, attempts, budget, weights_file, version, stop_on_success = task
    saved = torch.load(weights_file, map_location='cpu', weights_only=True)
    if saved['version'] != version: raise RuntimeError('rollout policy version changed')
    _worker_policy.load_state_dict(saved['state_dict'])
    sampler = FrozenSampler(_worker_policy, capacity=10000)
    machine = Machine(selected['machine']['rows']); rng = random.Random(seed)
    successes = []; statuses = Counter(); actions = 0; start = time.monotonic()
    for i in range(attempts):
        cert, trace, status = attempt(sampler, machine, rng, budget)
        actions += len(trace); statuses[status] += 1
        if cert:
            successes.append({'attempt':i+1, 'actions':trace, 'certificate':cert})
            if stop_on_success:
                break
    return {**selected, 'action_seed':seed, 'policy_version':version,
            'attempts':sum(statuses.values()), 'actions':actions, 'statuses':dict(statuses),
            'successes':successes, 'seconds':time.monotonic()-start,
            'network_evaluations':sampler.misses, 'cache_hits':sampler.hits}


def learn_batch(policy, optimizer, results, budget):
    """Zero-reward episodes contribute zero; identical state gradients combine.

    Every success is on-policy under the same frozen batch weights. Grouping
    identical observations/actions is algebraically the same as summing their
    individual log-probability gradients, divided by all attempts in the batch.
    """
    groups = {}; proof_count = 0; numerical_fallbacks = 0
    for result in results:
        machine = Machine(result['machine']['rows'])
        for success in result['successes']:
            check(success['certificate'])
            replay = Game(machine, budget=budget); proof_count += 1
            for step in success['actions']:
                obs = replay.observation(); encoded = encode_json(obs)
                index = next(i for i,a in enumerate(obs['actions']) if a['id']==step['action'])
                if encoded not in groups:
                    groups[encoded] = ([0]*len(obs['actions']), {})
                counts, expected = groups[encoded]; counts[index] += 1
                if index in expected and not math.isclose(expected[index],step['probability'],rel_tol=1e-5,abs_tol=1e-7):
                    raise RuntimeError('inconsistent sampled probabilities')
                expected[index] = step['probability']
                replay.step(step['action'], observe=False)
            if replay.status != 'SOLVED' or json.loads(json.dumps(replay.certificate())) != json.loads(json.dumps(success['certificate'])):
                raise ValueError('incomplete or mismatched training proof')
    if not proof_count:
        return {'updated':False, 'proofs':0, 'unique_gradient_observations':0, 'loss':0.}
    optimizer.zero_grad(set_to_none=True)
    denominator = sum(r['attempts'] for r in results); total_loss = 0.
    for encoded, (counts, expected) in groups.items():
        observation = json.loads(encoded)
        logits,_ = policy(observation); logps = logits.log_softmax(-1)
        inference_probs = None
        for index, probability in expected.items():
            grad_probability = float(logps[index].detach().exp())
            if not math.isclose(grad_probability, probability, rel_tol=2e-5, abs_tol=2e-7):
                # On some PyTorch versions, the eval/no-grad transformer kernel
                # used by FrozenSampler differs slightly from the grad kernel.
                # Verify against the inference path before rejecting the rollout.
                if inference_probs is None:
                    with torch.inference_mode():
                        inference_probs = policy(observation)[0].softmax(-1).tolist()
                inference_probability = inference_probs[index]
                if not math.isclose(inference_probability, probability, rel_tol=2e-4, abs_tol=2e-6):
                    raise RuntimeError(
                        'off-policy training trajectory: '
                        f'sampled={probability:.9g}, grad={grad_probability:.9g}, '
                        f'inference={inference_probability:.9g}, action={observation["actions"][index]["id"]}')
                numerical_fallbacks += 1
        loss = -(logps*torch.tensor(counts,dtype=logps.dtype)).sum()/denominator
        if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
        loss.backward(); total_loss += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return {'updated':True, 'proofs':proof_count, 'unique_gradient_observations':len(groups),
            'loss':total_loss, 'gradient_norm':float(norm),
            'inference_gradient_probability_differences':numerical_fallbacks}


def summarize(files):
    rows = [r for file in files for r in json.loads(file.read_text())['machines']]
    successes = sum(len(r['successes']) for r in rows)
    attempts = sum(r['attempts'] for r in rows)
    proved = sum(bool(r['successes']) for r in rows)
    return {'machines':len(rows), 'machines_proved':proved,
            'machine_coverage':proved/len(rows) if rows else 0.,
            'attempts':attempts, 'successful_attempts':successes,
            'attempt_success_rate':successes/attempts if attempts else 0.,
            'actions':sum(r['actions'] for r in rows),
            'statuses':dict(sum((Counter(r['statuses']) for r in rows),Counter()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--attempts',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-first-train-proof',action='store_true',
                        help='stop each training machine after one verified proof; keep evaluation at the full attempt budget')
    args = parser.parse_args()
    if min(args.epochs,args.attempts,args.batch_size,args.workers)<1:parser.error('positive counts required')
    if args.out.exists() and not args.resume:parser.error('choose a new output directory or --resume')
    args.out.mkdir(parents=True,exist_ok=True)
    try:lock = acquire_run_lock(args.out/'run.lock')
    except RuntimeError as e:parser.error(str(e))
    dataset_bytes = args.dataset.read_bytes(); dataset = json.loads(dataset_bytes)
    dataset_format = dataset.get('format')
    if dataset_format not in ('proofgame-epoch-curriculum-1', UNFILTERED_FORMAT):
        parser.error('unsupported dataset')
    if type(dataset.get('action_budget')) is not int or dataset['action_budget'] < 1:
        parser.error('invalid action budget')
    ids = []
    for split in ['train','test']:
        if not dataset.get(split): parser.error('empty split')
        for row in dataset[split]:
            if set(row)!={'rank','machine_id','machine'}:
                parser.error('dataset must contain only machine identities and tables')
            machine=Machine(row['machine']['rows'])
            if key(machine)!=row['machine_id']:
                parser.error('invalid machine identity')
            if dataset_format=='proofgame-epoch-curriculum-1':
                if is_test(machine)!=(split=='test'):
                    parser.error('invalid filtered dataset split')
            else:
                rank=row['rank']
                if (type(rank) is not int or not 0<=rank<POPULATION
                        or from_rank(rank,states=2).to_json()!=machine.to_json()):
                    parser.error('invalid unfiltered two-state machine/rank')
            ids.append(row['machine_id'])
    if len(set(ids))!=len(ids):parser.error('duplicate/overlapping machines')
    if dataset_format==UNFILTERED_FORMAT:
        from .unfiltered_dataset import build_dataset
        seed=dataset.get('sampling_seed')
        if (type(seed) is not int or dataset.get('population')!=POPULATION
                or dataset.get('sampling')!='uniform_without_replacement_over_all_labelled_two_state_machines'):
            parser.error('invalid unfiltered sampling metadata')
        expected=build_dataset(len(dataset['train']),len(dataset['test']),
                               dataset['action_budget'],seed)
        for split in ('train','test'):
            if [row['rank'] for row in dataset[split]] != [row['rank'] for row in expected[split]]:
                parser.error('unfiltered dataset ranks do not match the claimed seed')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    policy=Policy(width=32,layers=1).eval()
    optimizer=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=0.)
    config={'dataset_sha256':hashlib.sha256(dataset_bytes).hexdigest(),
            'train_machines':len(dataset['train']),'test_machines':len(dataset['test']),
            'attempts_per_machine':args.attempts,'budget':dataset['action_budget'],
            'batch_size':args.batch_size,'seed':args.seed,'lr':args.lr,'model':policy.config,
            'parameters':sum(p.numel() for p in policy.parameters()),'epochs_requested':args.epochs,
            'algorithm':'on-policy REINFORCE batches; zero baseline; mean gradient over all attempts',
            'attempts_after_success':('training: stop after first verified proof; evaluation: full attempt budget'
                                       if args.stop_after_first_train_proof else 'continue to exact attempt budget'),
            'evaluation':('fixed unfiltered test set; fixed sampling seeds; no updates'
                          if dataset_format==UNFILTERED_FORMAT
                          else 'fixed filtered test set; fixed sampling seeds; no updates'),
            'selector_proofs_supplied':0,'initial_archive':0}
    if args.resume:
        old=json.loads((args.out/'config.json').read_text())
        changing_strategy = False
        for k in config:
            if k!='epochs_requested' and config[k]!=old[k]:
                if (k=='attempts_after_success' and args.stop_after_first_train_proof
                        and old[k]=='continue to exact attempt budget'):
                    changing_strategy = True
                else:
                    parser.error('resume configuration changed: '+k)
        state=torch.load(args.out/'state.pt',map_location='cpu',weights_only=True)
        if args.epochs < state['completed_epochs']:parser.error('requested epoch count is below completed epochs')
        changes=list(old.get('strategy_changes',[]))
        if changing_strategy:
            changes.append({'phase':state['phase'],'epoch':state['epoch'],
                            'next_machine_index':state['next_index'],
                            'from':old['attempts_after_success'],
                            'to':config['attempts_after_success']})
            print('WARNING: changing the training attempt strategy while resuming. '
                  'The current epoch may contain mixed training budgets.',flush=True)
        if changes:
            config['strategy_changes']=changes
        policy.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
        if state['phase']=='done' and state['completed_epochs']<args.epochs:
            state.update(phase='train',epoch=state['completed_epochs']+1,next_index=0)
    else:
        state={'phase':'baseline','epoch':0,'next_index':0,'updates':0,'completed_epochs':0,'active_seconds':0.}
        atomic_torch(args.out/'initial_model.pt',{'config':policy.config,'state_dict':policy.state_dict(),'seed':args.seed})
        atomic_json(args.out/'dataset.json',dataset)
    atomic_json(args.out/'config.json',config)
    segment_start=time.monotonic();previous_seconds=state['active_seconds']

    def save_state():
        state['active_seconds']=previous_seconds+time.monotonic()-segment_start
        state.update(config=policy.config,state_dict=policy.state_dict(),optimizer_state_dict=optimizer.state_dict())
        atomic_torch(args.out/'state.pt',state)

    save_state()
    if state['phase'] != 'done':
        atomic_json(args.out/'report.json',{'complete':False,'epochs_completed':state['completed_epochs'],'epochs_requested':args.epochs})
    weights_file=(args.out/'workers_policy.pt').resolve()
    with mp.get_context('spawn').Pool(args.workers,initializer=init_worker,initargs=(policy.config,),maxtasksperchild=50) as pool:
        while state['phase']!='done':
            phase=state['phase'];epoch=state['epoch']
            training=phase=='train';rows=list(dataset['train' if training else 'test'])
            if training:random.Random(args.seed+10000*epoch).shuffle(rows)
            folder=args.out/f'epoch_{epoch:02d}_{"train" if training else "eval"}'
            folder.mkdir(exist_ok=True)
            start_index=state['next_index']
            if start_index<len(rows):
                batch=rows[start_index:start_index+args.batch_size]
                atomic_torch(weights_file,{'version':state['updates'],'config':policy.config,'state_dict':policy.state_dict()})
                tasks=[(r,task_seed(args.seed,'train' if training else 'eval',epoch,r['machine_id']),
                        args.attempts,dataset['action_budget'],str(weights_file),state['updates'],
                        training and args.stop_after_first_train_proof) for r in batch]
                results=pool.map(run_machine,tasks,chunksize=1)
                update=learn_batch(policy,optimizer,results,dataset['action_budget']) if training else {'updated':False}
                state['updates']+=int(update['updated'])
                batch_file=folder/f'batch_{start_index:04d}.json'
                atomic_json(batch_file,{'epoch':epoch,'phase':phase,'start_index':start_index,'machines':results,'update':update})
                state['next_index']+=len(batch);save_state()
                progress={'phase':phase,'epoch':epoch,'target_epochs':args.epochs,'machines_done':state['next_index'],
                          'machines_total':len(rows),'policy_updates':state['updates'],
                          'last_batch_solved':sum(bool(r['successes']) for r in results),
                          'last_batch_proofs':sum(len(r['successes']) for r in results),'active_seconds':state['active_seconds']}
                atomic_json(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
                continue
            metrics=summarize(sorted(folder.glob('batch_*.json')))
            metrics.update(epoch=epoch,phase=phase,policy_updates=state['updates'],active_seconds=state['active_seconds'])
            atomic_json(args.out/f'epoch_{epoch:02d}_{"training" if training else "evaluation"}.json',metrics)
            if training:
                state.update(phase='eval',next_index=0)
            elif phase=='baseline':
                state.update(phase='train',epoch=1,next_index=0)
            else:
                state['completed_epochs']=epoch
                atomic_torch(args.out/f'model_epoch_{epoch:02d}.pt',{'config':policy.config,'state_dict':policy.state_dict(),'seed':args.seed,'epoch':epoch,'updates':state['updates']})
                state.update(phase='done' if epoch>=args.epochs else 'train',epoch=epoch+1,next_index=0)
            save_state()
    report={'complete':True,'epochs_completed':state['completed_epochs'],'policy_updates':state['updates'],
            'active_seconds':state['active_seconds'],
            'evaluations':[json.loads((args.out/f'epoch_{i:02d}_evaluation.json').read_text()) for i in range(state['completed_epochs']+1)],
            'training':[json.loads((args.out/f'epoch_{i:02d}_training.json').read_text()) for i in range(1,state['completed_epochs']+1)]}
    atomic_json(args.out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
