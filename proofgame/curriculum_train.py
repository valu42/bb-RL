"""Train on filter-selected machines, using only self-generated proof rewards.

One machine is kept for at most 10,000 twenty-action attempts. At its first
verified success, apply one zero-baseline REINFORCE update and move on. Failed
attempts leave policy and optimizer unchanged. No selector certificate is read.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import time

import torch

from .environment import Game
from .kernel import check
from .logic import Machine
from .model import Policy
from .network_search import FrozenSampler, attempt
from .tasks import is_test, key


def learn_from_verified_trace(policy, optimizer, machine, trace, budget=20):
    """Refuse incomplete/invalid trajectories before any optimizer mutation."""
    replay = Game(machine, budget=budget)
    for row in trace:
        replay.step(row['action'], observe=False)
    if replay.status != 'SOLVED':
        raise ValueError('policy updates require a fully verified self-generated proof')
    certificate = replay.certificate()
    check(certificate)
    optimizer.zero_grad(set_to_none=True)
    replay = Game(machine, budget=budget)
    loss_value = 0.
    for row in trace:
        obs = replay.observation()
        logits, _ = policy(obs)
        index = next(i for i, a in enumerate(obs['actions']) if a['id'] == row['action'])
        logp = logits.log_softmax(-1)[index]
        if 'probability' in row and not math.isclose(float(logp.detach().exp()), row['probability'], rel_tol=1e-5, abs_tol=1e-7):
            raise RuntimeError('trajectory probabilities do not match current policy')
        loss = -logp  # Terminal reward 1, baseline 0, undiscounted return.
        if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
        loss.backward()
        loss_value += float(loss.detach())
        replay.step(row['action'], observe=False)
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return {'loss': loss_value, 'gradient_norm_before_clipping': float(norm),
            'certificate': certificate}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--attempts', type=int, default=10000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('choose a fresh output directory')
    if min(args.attempts, args.threads) < 1 or args.lr <= 0: p.error('invalid limits')
    selection = json.loads(args.selection.read_text())
    if selection['format'] != 'proofgame-filtered-machines-1' or selection['split'] != 'train':
        p.error('expected training-only machine selection')
    budget = selection['action_budget']
    for row in selection['machines']:
        if set(row) != {'rank', 'machine_id', 'machine', 'network_action_seed'}:
            p.error('selection must not contain answers, certificates, or proof steps')
        machine = Machine(row['machine']['rows'])
        if is_test(machine) or key(machine) != row['machine_id']:
            p.error('invalid machine or training/test split')
    args.out.mkdir(parents=True)
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    policy = Policy(width=32, layers=1).eval()
    initial = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.)
    config = {'selection': selection, 'network_seed': args.seed, 'model': policy.config,
              'parameters': sum(p.numel() for p in policy.parameters()), 'torch': torch.__version__,
              'attempt_limit_per_machine': args.attempts, 'actions_per_attempt': budget,
              'learning_rate': args.lr, 'algorithm': 'success-triggered REINFORCE, zero baseline',
              'reward': '1 only for a complete kernel-checked proof; otherwise 0',
              'failure_update': 'none; policy and optimizer unchanged',
              'move_to_next_machine': 'first self-generated proof or attempt cap',
              'selector_proofs_supplied': 0, 'external_demonstrations': 0,
              'previous_checkpoint_loaded': False}
    (args.out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    torch.save({'config': policy.config, 'state_dict': initial, 'seed': args.seed}, args.out/'initial_model.pt')
    updates = 0; records = []; start = time.monotonic()

    def checkpoint(name, completed):
        torch.save({'config': policy.config, 'state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'seed': args.seed,
                    'completed_machines': completed, 'training_updates': updates}, args.out/name)

    checkpoint('model.pt', 0)
    with (args.out/'records.jsonl').open('w') as log:
        for j, selected in enumerate(selection['machines']):
            machine = Machine(selected['machine']['rows'])
            rng = random.Random(selected['network_action_seed'])
            # A new cache per machine also guarantees invalidation after every update.
            sampler = FrozenSampler(policy)
            success = None; action_count = 0; statuses = Counter()
            for i in range(args.attempts):
                cert, trace, status = attempt(sampler, machine, rng, budget)
                action_count += len(trace); statuses[status] += 1
                if cert:
                    update = learn_from_verified_trace(policy, optimizer, machine, trace, budget)
                    if update['certificate'] != cert: raise RuntimeError('replayed certificate mismatch')
                    updates += 1
                    success = {'attempt': i+1, 'actions': trace, 'certificate': cert,
                               'update_loss': update['loss'], 'gradient_norm': update['gradient_norm_before_clipping']}
                    checkpoint('model.pt', j+1)
                    checkpoint(f'model_after_machine_{j}.pt', j+1)
                if (i+1) % 100 == 0 or cert or i+1 == args.attempts:
                    progress = {'completed_machines': len(records), 'current_machine': j+1,
                                'total_machines': len(selection['machines']), 'rank': selected['rank'],
                                'attempts_on_current': i+1, 'updates': updates,
                                'total_attempts': sum(r['attempts_completed'] for r in records)+i+1,
                                'network_evaluations_on_current': sampler.misses,
                                'cache_hits_on_current': sampler.hits,
                                'elapsed_seconds': time.monotonic()-start}
                    (args.out/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
                    print(json.dumps(progress), flush=True)
                if cert: break
            record = {**selected, 'attempts_completed': i+1, 'actions_simulated': action_count,
                      'statuses': dict(statuses), 'success': success}
            records.append(record); log.write(json.dumps(record)+'\n'); log.flush()
            checkpoint('model.pt', j+1)
    report = {'complete': True, 'machines': len(records), 'machines_proved': sum(bool(r['success']) for r in records),
              'training_updates': updates, 'attempts_completed': sum(r['attempts_completed'] for r in records),
              'actions_simulated': sum(r['actions_simulated'] for r in records),
              'proved_answers': dict(Counter(r['success']['certificate']['answer'] for r in records if r['success'])),
              'statuses': dict(sum((Counter(r['statuses']) for r in records), Counter())),
              'parameter_l1_change': sum(float((v-initial[k]).abs().sum()) for k,v in policy.state_dict().items()),
              'elapsed_seconds': time.monotonic()-start,
              'interpretation': 'Training-machine discovery under repeated attempts. Generalization must be evaluated independently on unfiltered held-out machines.'}
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
