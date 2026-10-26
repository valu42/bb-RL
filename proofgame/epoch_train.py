"""Resumable epoch training on fixed, disjoint train/test datasets.

Evaluation always runs the exact configured number of independent-reset
attempts per machine. Training can optionally stop on its first verified proof.
Rollout batches share frozen weights; only training successes supply REINFORCE
gradients. Evaluation never updates parameters. Selector proofs are not accepted
in the dataset. Checkpoints commit after every rollout batch.
"""
import argparse
import copy
import csv
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
from .exploration import behavior_log_probs
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


def choose_run_directory(path, resume=False):
    """Create a fresh unique run directory without overwriting earlier runs.

    The exact path is used for --resume. Fresh runs first claim the requested
    path, then try numbered siblings using atomic mkdir(exist_ok=False).
    """
    if resume:
        if not path.is_dir():
            raise FileNotFoundError('resume directory does not exist: ' + str(path))
        return path
    for number in range(100000):
        candidate = path if number == 0 else path.with_name(
            f'{path.name}_run_{number:04d}')
        try:
            candidate.mkdir(parents=True,exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError('could not allocate a new output directory next to ' + str(path))


def policy_is_finite(policy):
    """Detect corrupt parameters and buffers before rollout/checkpoint."""
    for tensor in policy.state_dict().values():
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
            return False
    return True


def optimizer_is_finite(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                if not bool(torch.isfinite(value).all()):
                    return False
    return True


def assert_policy_finite(policy, where):
    if not policy_is_finite(policy):
        raise RuntimeError(
            f'non-finite model weights {where}; the saved checkpoint may be corrupted. '
            'Do not resume it without recovery. Start a fresh run or restore an earlier finite checkpoint.')


def task_seed(seed, phase, epoch, machine_id):
    # Evaluation uses the same seeds each epoch; it is a fixed monitoring set.
    epoch = epoch if phase == 'train' else 0
    return int.from_bytes(hashlib.sha256(f'{seed}:{phase}:{epoch}:{machine_id}'.encode()).digest()[:8], 'big')


_worker_policy = None


def init_worker(model_config):
    global _worker_policy
    torch.set_num_threads(1)
    _worker_policy = Policy(**model_config).eval()


def execute_machine(selected, seed, attempts, budget, version, stop_on_success, sampler,
                    exploration_epsilon=0.):
    """Execute a machine with either a CPU policy or the central GPU sampler."""
    machine = Machine(selected['machine']['rows']); rng = random.Random(seed)
    successes = []; statuses = Counter(); actions = 0; start = time.monotonic()
    first_actions=Counter(); policy_sums={'HALT':0.,'NONHALT':0.}
    behavior_sums={'HALT':0.,'NONHALT':0.}; initial_observations=0
    def record_initial(action, raw, mixed):
        nonlocal initial_observations
        first_actions[action]+=1
        if raw:
            initial_observations+=1
            for answer in ('HALT','NONHALT'):
                policy_sums[answer]+=raw[answer]
                behavior_sums[answer]+=mixed[answer]
    for i in range(attempts):
        cert, trace, status = attempt(sampler, machine, rng, budget,
                                      exploration_epsilon=exploration_epsilon,
                                      on_initial_choice=record_initial)
        actions += len(trace); statuses[status] += 1
        if cert:
            successes.append({'attempt':i+1, 'actions':trace, 'certificate':cert})
            if stop_on_success:
                break
    return {**selected, 'action_seed':seed, 'policy_version':version,
            'attempts':sum(statuses.values()), 'actions':actions, 'statuses':dict(statuses),
            'successes':successes, 'seconds':time.monotonic()-start,
            'network_evaluations':sampler.misses, 'cache_hits':sampler.hits,
            'first_action_counts':dict(first_actions),
            'initial_choice_observations':initial_observations,
            'initial_policy_probability_sums':policy_sums,
            'initial_behavior_probability_sums':behavior_sums}


def run_machine(task):
    selected, seed, attempts, budget, weights_file, version, stop_on_success, epsilon = task
    saved = torch.load(weights_file, map_location='cpu', weights_only=True)
    if saved['version'] != version: raise RuntimeError('rollout policy version changed')
    _worker_policy.load_state_dict(saved['state_dict'])
    sampler = FrozenSampler(_worker_policy, capacity=10000)
    return execute_machine(selected,seed,attempts,budget,version,stop_on_success,sampler,epsilon)


def learn_batch(policy, optimizer, results, budget, gpu_gradient_batch_size=4,
                exploration_epsilon=0.):
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
    device = next(policy.parameters()).device
    is_mps = device.type == 'mps'
    observations = list(groups.items())
    microbatch = gpu_gradient_batch_size if is_mps else 1
    for start in range(0,len(observations),microbatch):
        chunk = observations[start:start+microbatch]
        parsed = [json.loads(encoded) for encoded, _ in chunk]
        if is_mps:
            logits_batch = policy.score_many(parsed)
        else:
            logits_batch = [policy(obs)[0] for obs in parsed]
        losses = []
        for observation, logits, (_, (counts, expected)) in zip(parsed,logits_batch,chunk):
            logps = behavior_log_probs(logits,observation,exploration_epsilon)
            inference_probs = None
            for index, probability in expected.items():
                grad_probability = float(logps[index].detach().exp())
                if not math.isclose(grad_probability, probability,
                                    rel_tol=2e-4 if is_mps else 2e-5,abs_tol=2e-6 if is_mps else 2e-7):
                    # Batched inference and gradient kernels may give slightly
                    # different float32 results on Metal. Recheck against the
                    # same frozen policy, never allow a changed policy version.
                    if inference_probs is None:
                        with torch.inference_mode():
                            inference_logits=policy(observation)[0]
                            inference_probs=behavior_log_probs(
                                inference_logits,observation,exploration_epsilon).exp().tolist()
                    inference_probability = inference_probs[index]
                    if not math.isclose(inference_probability, probability,
                                        rel_tol=3e-3 if is_mps else 2e-4,
                                        abs_tol=3e-5 if is_mps else 2e-6):
                        raise RuntimeError(
                            'off-policy training trajectory: '
                            f'sampled={probability:.9g}, grad={grad_probability:.9g}, '
                            f'inference={inference_probability:.9g}, action={observation["actions"][index]["id"]}')
                    numerical_fallbacks += 1
            loss = -(logps*torch.tensor(counts,dtype=logps.dtype,
                                        device=logps.device)).sum()/denominator
            if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
            losses.append(loss)
        grouped_loss = torch.stack(losses).sum()
        grouped_loss.backward()
        total_loss += float(grouped_loss.detach())
    # A single bad REINFORCE gradient must not poison all future rollouts.
    # Reject the update without changing the policy/optimizer; keep proof logs.
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=False)
    if not bool(torch.isfinite(norm)):
        optimizer.zero_grad(set_to_none=True)
        return {'updated':False, 'proofs':proof_count,
                'unique_gradient_observations':len(groups),
                'loss':total_loss, 'rejected_update':'nonfinite_gradient_norm'}
    if not policy_is_finite(policy):
        raise RuntimeError('non-finite policy parameters before optimizer step')
    old_weights=[p.detach().clone() for p in policy.parameters()]
    old_optimizer=copy.deepcopy(optimizer.state_dict())
    try:
        optimizer.step()
        weights_ok=policy_is_finite(policy)
        optimizer_ok=optimizer_is_finite(optimizer)
        logits_ok=True
        grad_mode_logits_ok=None
        healthy=weights_ok and optimizer_ok
        if healthy:
            # A finite weight vector can still yield invalid attention logits.
            # Validate up to 8 *distinct* replay states spread across the
            # successful trajectories (not just the first and last).
            probe_count=min(8,len(observations))
            probe_indices=sorted({round(i*(len(observations)-1)/(probe_count-1))
                                  for i in range(probe_count)}) if probe_count>1 else [0]
            with torch.inference_mode():
                for index in probe_indices:
                    encoded,_=observations[index]
                    logits,_=policy(json.loads(encoded))
                    if not bool(torch.isfinite(logits).all()):
                        logits_ok=False
                        healthy=False
                        break
            if not logits_ok:
                with torch.enable_grad():
                    check_logits,_=policy(json.loads(encoded))
                    grad_mode_logits_ok=bool(torch.isfinite(check_logits).all())
    except Exception:
        with torch.no_grad():
            for p,old in zip(policy.parameters(),old_weights):
                p.copy_(old)
        optimizer.load_state_dict(old_optimizer)
        optimizer.zero_grad(set_to_none=True)
        raise
    if not healthy:
        with torch.no_grad():
            for p,old in zip(policy.parameters(),old_weights):
                p.copy_(old)
        optimizer.load_state_dict(old_optimizer)
        optimizer.zero_grad(set_to_none=True)
        return {'updated':False, 'proofs':proof_count,
                'unique_gradient_observations':len(groups),
                'loss':total_loss, 'gradient_norm':float(norm),
                'rejected_update':'nonfinite_optimizer_result',
                'numerical_health':{'weights_finite':weights_ok,
                                    'optimizer_finite':optimizer_ok,
                                    'probe_logits_finite':logits_ok,
                                    'probe_grad_mode_logits_finite':grad_mode_logits_ok}}
    return {'updated':True, 'proofs':proof_count, 'unique_gradient_observations':len(groups),
            'loss':total_loss, 'gradient_norm':float(norm),
            'inference_gradient_probability_differences':numerical_fallbacks}


def count_proofs_by_answer(rows):
    """Report certificate-backed outcomes, not inferred ground-truth labels."""
    answers = ('HALT', 'NONHALT')
    machines = {answer: 0 for answer in answers}
    proofs = {answer: 0 for answer in answers}
    first_attempts = {answer: [] for answer in answers}
    for row in rows:
        labels = set()
        for success in row['successes']:
            cert = success['certificate']
            answer = cert['answer']
            if answer not in proofs:
                raise ValueError('unexpected verified proof answer: ' + str(answer))
            proofs[answer] += 1
            labels.add(answer)
        for answer in labels:
            machines[answer] += 1
        if len(labels)>1:
            raise ValueError('a machine has contradictory accepted proofs')
        if row['successes']:
            first_answer = row['successes'][0]['certificate']['answer']
            first_attempts[first_answer].append(row['successes'][0]['attempt'])
    return {
        'machines_proved_by_answer': machines,
        'successful_attempts_by_answer': proofs,
        'machines_unproved': len(rows) - sum(machines.values()),
        'first_proof_attempt_mean_by_answer': {
            answer: sum(first_attempts[answer])/len(first_attempts[answer])
            if first_attempts[answer] else None for answer in answers
        }
    }


def summarize_initial_choices(rows):
    counts=Counter(); raw={'HALT':0.,'NONHALT':0.}
    behavior={'HALT':0.,'NONHALT':0.}; n=0
    for row in rows:
        counts.update(row.get('first_action_counts',{}))
        n+=row.get('initial_choice_observations',0)
        for answer in ('HALT','NONHALT'):
            raw[answer]+=row.get('initial_policy_probability_sums',{}).get(answer,0.)
            behavior[answer]+=row.get('initial_behavior_probability_sums',{}).get(answer,0.)
    return {
        'first_action_counts':{action:counts.get(action,0)
                               for action in ('choose:HALT','choose:NONHALT','reset')},
        'initial_choice_observations':n,
        'mean_initial_policy_probability':{
            answer:raw[answer]/n if n else None for answer in ('HALT','NONHALT')},
        'mean_initial_behavior_probability':{
            answer:behavior[answer]/n if n else None for answer in ('HALT','NONHALT')}
    }


def summarize_rows(rows):
    successes = sum(len(r['successes']) for r in rows)
    attempts = sum(r['attempts'] for r in rows)
    proved = sum(bool(r['successes']) for r in rows)
    return {'machines':len(rows), 'machines_proved':proved,
            'machine_coverage':proved/len(rows) if rows else 0.,
            'machines_solved_pct':100*proved/len(rows) if rows else 0.,
            'attempts':attempts, 'successful_attempts':successes,
            'attempt_success_rate':successes/attempts if attempts else 0.,
            'attempt_success_pct':100*successes/attempts if attempts else 0.,
            'actions':sum(r['actions'] for r in rows),
            'statuses':dict(sum((Counter(r['statuses']) for r in rows),Counter())),
            **count_proofs_by_answer(rows), **summarize_initial_choices(rows)}


def summarize(files):
    rows = [r for file in files for r in json.loads(file.read_text())['machines']]
    return summarize_rows(rows)

def evaluation_line(metrics):
    label = 'Baseline (before training)' if metrics['epoch'] == 0 else f"Epoch {metrics['epoch']} evaluation"
    return (f"{label}: machines solved {metrics['machines_proved']}/{metrics['machines']} "
            f"({100*metrics['machine_coverage']:.2f}%); successful attempts "
            f"{metrics['successful_attempts']}/{metrics['attempts']} "
            f"({100*metrics['attempt_success_rate']:.2f}%)")


def write_evaluation_csv(out):
    """Rebuild from committed summaries so resuming never duplicates rows."""
    fields = ['epoch', 'machines_proved', 'machines', 'machines_solved_pct',
              'successful_attempts', 'attempts', 'attempt_success_pct', 'policy_updates']
    temp = out/'evaluations.csv.tmp'
    summaries = sorted(out.glob('epoch_*_evaluation.json'),
                       key=lambda path: int(path.name.split('_')[1]))
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for path in summaries:
            metrics = json.loads(path.read_text())
            metrics['machines_solved_pct'] = 100*metrics['machine_coverage']
            metrics['attempt_success_pct'] = 100*metrics['attempt_success_rate']
            writer.writerow({field: metrics[field] for field in fields})
    temp.replace(out/'evaluations.csv')


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
    parser.add_argument('--train-exploration-epsilon',type=float,default=0.2,
                        help='fraction of initial training decisions sampled uniformly between HALT/NONHALT (default 0.2); evaluation unchanged')
    parser.add_argument('--representation',choices=('bytes','ast'),default='bytes',
                        help='byte-level JSON or AST-aware tokens and learnable structural attention')
    parser.add_argument('--device',choices=('cpu','mps'),default='cpu',
                        help='MPS uses centralized batched GPU inference with CPU proof workers')
    parser.add_argument('--gpu-inference-batch-size',type=int,default=8)
    parser.add_argument('--gpu-inference-wait-ms',type=float,default=2.)
    parser.add_argument('--gpu-max-attention-elements',type=int,default=8000000)
    parser.add_argument('--gpu-gradient-batch-size',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-first-train-proof',action='store_true',
                        help='stop each training machine after one verified proof; keep evaluation at the full attempt budget')
    args = parser.parse_args()
    if min(args.epochs,args.attempts,args.batch_size,args.workers,
           args.gpu_inference_batch_size,args.gpu_max_attention_elements,
           args.gpu_gradient_batch_size)<1:
        parser.error('positive counts required')
    if not math.isfinite(args.train_exploration_epsilon) or not 0<=args.train_exploration_epsilon<1:
        parser.error('train-exploration-epsilon must be in [0,1)')
    if not math.isfinite(args.gpu_inference_wait_ms) or args.gpu_inference_wait_ms<0:
        parser.error('invalid GPU batching wait')
    if args.device=='mps' and not torch.backends.mps.is_available():
        parser.error('MPS is not available in this PyTorch/macOS environment')
    try:
        args.out = choose_run_directory(args.out, resume=args.resume)
    except (OSError, RuntimeError) as e:
        parser.error(str(e))
    print(json.dumps({'event':'run_directory',
                      'path':str(args.out),
                      'resuming':args.resume}),flush=True)
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
    policy=Policy(width=32,layers=1,representation=args.representation).to(args.device).eval()
    optimizer=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=0.)
    config={'dataset_sha256':hashlib.sha256(dataset_bytes).hexdigest(),
            'train_machines':len(dataset['train']),'test_machines':len(dataset['test']),
            'attempts_per_machine':args.attempts,'budget':dataset['action_budget'],
            'batch_size':args.batch_size,'seed':args.seed,'lr':args.lr,'model':policy.config,
            'parameters':sum(p.numel() for p in policy.parameters()),'epochs_requested':args.epochs,
            'algorithm':'on-policy REINFORCE batches; zero baseline; mean gradient over all attempts',
            'train_exploration_epsilon':args.train_exploration_epsilon,
            'attempts_after_success':('training: stop after first verified proof; evaluation: full attempt budget'
                                       if args.stop_after_first_train_proof else 'continue to exact attempt budget'),
            'evaluation':('fixed unfiltered test set; fixed sampling seeds; no updates'
                          if dataset_format==UNFILTERED_FORMAT
                          else 'fixed filtered test set; fixed sampling seeds; no updates'),
            'selector_proofs_supplied':0,'initial_archive':0}
    if args.device=='mps':
        config['device']='mps'
        config['gpu_inference']={'batch_size':args.gpu_inference_batch_size,
                                 'wait_ms':args.gpu_inference_wait_ms,
                                 'max_attention_elements':args.gpu_max_attention_elements,
                                 'gradient_batch_size':args.gpu_gradient_batch_size}
    if args.resume:
        old=json.loads((args.out/'config.json').read_text())
        changing_strategy = False
        for k in config:
            if k!='epochs_requested' and config[k]!=old.get(k,0. if k=='train_exploration_epsilon' else None):
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
        assert_policy_finite(policy, 'on resume')
        if not optimizer_is_finite(optimizer):
            parser.error('non-finite optimizer state on resume; restore a finite checkpoint or start a new run')
        if state['phase']=='done' and state['completed_epochs']<args.epochs:
            state.update(phase='train',epoch=state['completed_epochs']+1,next_index=0)
    else:
        state={'phase':'baseline','epoch':0,'next_index':0,'updates':0,'completed_epochs':0,'active_seconds':0.}
        atomic_torch(args.out/'initial_model.pt',{'config':policy.config,'state_dict':policy.state_dict(),'seed':args.seed})
        atomic_json(args.out/'dataset.json',dataset)
    atomic_json(args.out/'config.json',config)
    write_evaluation_csv(args.out)
    segment_start=time.monotonic();previous_seconds=state['active_seconds']

    def save_state():
        assert_policy_finite(policy, 'before saving state')
        if not optimizer_is_finite(optimizer):
            raise RuntimeError('non-finite optimizer state before checkpoint; refusing to save it')
        state['active_seconds']=previous_seconds+time.monotonic()-segment_start
        state.update(config=policy.config,state_dict=policy.state_dict(),optimizer_state_dict=optimizer.state_dict())
        atomic_torch(args.out/'state.pt',state)

    save_state()
    if state['phase'] != 'done':
        atomic_json(args.out/'report.json',{'complete':False,'epochs_completed':state['completed_epochs'],'epochs_requested':args.epochs})
    weights_file=(args.out/'workers_policy.pt').resolve()
    if args.device=='mps':
        from .gpu_runtime import CentralRolloutPool
        runner=CentralRolloutPool(policy,workers=args.workers,
                                 inference_batch_size=args.gpu_inference_batch_size,
                                 wait_ms=args.gpu_inference_wait_ms,
                                 max_attention_elements=args.gpu_max_attention_elements)
        print(json.dumps({'device':'mps','inference':'centralized-batched',
                          'workers':args.workers,'gpu_batch_size':args.gpu_inference_batch_size}),flush=True)
    else:
        runner=mp.get_context('spawn').Pool(args.workers,initializer=init_worker,
                                           initargs=(policy.config,),maxtasksperchild=50)
    with runner as pool:
        while state['phase']!='done':
            phase=state['phase'];epoch=state['epoch']
            training=phase=='train';rows=list(dataset['train' if training else 'test'])
            if training:random.Random(args.seed+10000*epoch).shuffle(rows)
            folder=args.out/f'epoch_{epoch:02d}_{"train" if training else "eval"}'
            folder.mkdir(exist_ok=True)
            start_index=state['next_index']
            if start_index<len(rows):
                batch=rows[start_index:start_index+args.batch_size]
                if args.device=='mps':
                    tasks=[(r,task_seed(args.seed,'train' if training else 'eval',epoch,r['machine_id']),
                            args.attempts,dataset['action_budget'],state['updates'],
                            training and args.stop_after_first_train_proof,
                            args.train_exploration_epsilon if training else 0.) for r in batch]
                    results=pool.run(tasks)
                else:
                    atomic_torch(weights_file,{'version':state['updates'],'config':policy.config,
                                               'state_dict':policy.state_dict()})
                    tasks=[(r,task_seed(args.seed,'train' if training else 'eval',epoch,r['machine_id']),
                            args.attempts,dataset['action_budget'],str(weights_file),state['updates'],
                            training and args.stop_after_first_train_proof,
                            args.train_exploration_epsilon if training else 0.) for r in batch]
                    results=pool.map(run_machine,tasks,chunksize=1)
                update=learn_batch(policy,optimizer,results,dataset['action_budget'],
                                   gpu_gradient_batch_size=args.gpu_gradient_batch_size,
                                   exploration_epsilon=args.train_exploration_epsilon) if training else {'updated':False}
                state['updates']+=int(update['updated'])
                if update.get('rejected_update'):
                    state['rejected_updates']=state.get('rejected_updates',0)+1
                    print(json.dumps({'event':'training_update_rejected','phase':phase,'epoch':epoch,
                                      'machine_index':start_index,'reason':update['rejected_update'],
                                      'rejected_updates':state['rejected_updates']}),flush=True)
                batch_file=folder/f'batch_{start_index:04d}.json'
                atomic_json(batch_file,{'epoch':epoch,'phase':phase,'start_index':start_index,'machines':results,'update':update})
                state['next_index']+=len(batch);save_state()
                batch_outcomes=count_proofs_by_answer(results)
                batch_choices=summarize_initial_choices(results)
                progress={'phase':phase,'epoch':epoch,'target_epochs':args.epochs,'machines_done':state['next_index'],
                          'machines_total':len(rows),'policy_updates':state['updates'],
                          'last_batch_solved':sum(bool(r['successes']) for r in results),
                          'last_batch_proofs':sum(len(r['successes']) for r in results),
                          'rejected_updates':state.get('rejected_updates',0),
                          'last_batch_first_action_counts':batch_choices['first_action_counts'],
                          'last_batch_mean_initial_policy_probability':batch_choices['mean_initial_policy_probability'],
                          'last_batch_mean_initial_behavior_probability':batch_choices['mean_initial_behavior_probability'],
                          'last_batch_machines_proved_by_answer':batch_outcomes['machines_proved_by_answer'],
                          'last_batch_successful_attempts_by_answer':batch_outcomes['successful_attempts_by_answer'],
                          'active_seconds':state['active_seconds']}
                if args.device=='mps':
                    progress['gpu_inference']=pool.inference.stats()
                atomic_json(args.out/'progress.json',progress);print(json.dumps(progress),flush=True)
                continue
            metrics=summarize(sorted(folder.glob('batch_*.json')))
            metrics.update(epoch=epoch,phase=phase,policy_updates=state['updates'],active_seconds=state['active_seconds'])
            atomic_json(args.out/f'epoch_{epoch:02d}_{"training" if training else "evaluation"}.json',metrics)
            print(json.dumps({'event':'epoch_summary','phase':phase,'epoch':epoch,
                              'machines':metrics['machines'],
                              'machines_proved':metrics['machines_proved'],
                              'machines_unproved':metrics['machines_unproved'],
                              'machines_proved_by_answer':metrics['machines_proved_by_answer'],
                              'successful_attempts_by_answer':metrics['successful_attempts_by_answer'],
                              'machine_coverage':metrics['machine_coverage'],
                              'machines_solved_pct':metrics['machines_solved_pct'],
                              'attempts':metrics['attempts'],
                              'successful_attempts':metrics['successful_attempts'],
                              'attempt_success_pct':metrics['attempt_success_pct'],
                              'first_action_counts':metrics['first_action_counts'],
                              'mean_initial_policy_probability':metrics['mean_initial_policy_probability'],
                              'mean_initial_behavior_probability':metrics['mean_initial_behavior_probability'],
                              'train_exploration_epsilon':args.train_exploration_epsilon if training else 0.,
                              'policy_updates':metrics['policy_updates']}),
                  flush=True)
            if not training:
                print(evaluation_line(metrics),flush=True)
                write_evaluation_csv(args.out)
                # Separate fixed diagnostic machines. No gradients, no extra
                # training data, and no effect on held-out machine coverage.
                from .special_benchmarks import (
                    benchmark_rows, make_benchmark_report, write_benchmark_report
                )
                challenge_rows = benchmark_rows()
                if args.device=='mps':
                    challenge_tasks=[
                        (row,task_seed(args.seed,'eval',0,row['machine_id']),
                         args.attempts,dataset['action_budget'],state['updates'],True,0.)
                        for row in challenge_rows
                    ]
                    challenge_results=pool.run(challenge_tasks)
                else:
                    atomic_torch(weights_file,{'version':state['updates'],
                                               'config':policy.config,
                                               'state_dict':policy.state_dict()})
                    challenge_tasks=[
                        (row,task_seed(args.seed,'eval',0,row['machine_id']),
                         args.attempts,dataset['action_budget'],
                         str(weights_file),state['updates'],True,0.)
                        for row in challenge_rows
                    ]
                    challenge_results=pool.map(run_machine,challenge_tasks,chunksize=1)
                benchmark_report=make_benchmark_report(
                    epoch,challenge_results,args.attempts,dataset['action_budget'],
                    trained_ids=(row['machine_id'] for row in dataset['train'])
                )
                write_benchmark_report(args.out,benchmark_report)
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
            'training':[json.loads((args.out/f'epoch_{i:02d}_training.json').read_text()) for i in range(1,state['completed_epochs']+1)],
            'special_benchmarks':[json.loads((args.out/f'special_benchmarks_epoch_{i:02d}.json').read_text())
                                  for i in range(state['completed_epochs']+1)
                                  if (args.out/f'special_benchmarks_epoch_{i:02d}.json').exists()]}
    atomic_json(args.out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

