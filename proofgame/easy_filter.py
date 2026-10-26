"""Select training machines using a capped, uniform stochastic proof search.

The selector records its certificates for auditing, but selection.json contains
only machines and independent network-action seeds: no answers or proof steps.
"""
import argparse
import json
from pathlib import Path
import random
import time

from .cached_random import Simulator
from .kernel import check
from .population_sample import from_rank
from .tasks import is_test, key


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=4)
    p.add_argument('--attempts', type=int, default=10000)
    p.add_argument('--budget', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261028)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists(): p.error('choose a fresh output directory')
    if min(args.count, args.attempts, args.budget) < 1: p.error('positive limits required')
    args.out.mkdir(parents=True)
    rng = random.Random(args.seed)
    ranks = rng.sample(range(20736), 20736)
    search_rng = random.Random(args.seed + 1)
    network_rng = random.Random(args.seed + 2)
    selected = []; records = []; start = time.monotonic()
    with (args.out/'filter_records.jsonl').open('w') as log:
        for rank in ranks:
            machine = from_rank(rank)
            if is_test(machine): continue
            action_seed = search_rng.getrandbits(64)
            sim = Simulator(machine, max_cached_states=50000)
            action_rng = random.Random(action_seed)
            certificate = None; actions = 0
            for i in range(args.attempts):
                certificate, n = sim.attempt(action_rng, args.budget)
                actions += n
                if certificate:
                    check(certificate)
                    break
            accepted = certificate is not None
            row = {'rank': rank, 'machine_id': key(machine), 'machine': machine.to_json(),
                   'filter_action_seed': action_seed, 'attempts_completed': i+1,
                   'actions_simulated': actions, 'accepted': accepted,
                   'successful_attempt_actions': n if accepted else None,
                   'filter_certificate': certificate}
            records.append(row); log.write(json.dumps(row)+'\n'); log.flush()
            if accepted:
                selected.append({'rank': rank, 'machine_id': key(machine),
                                 'machine': machine.to_json(),
                                 'network_action_seed': network_rng.getrandbits(64)})
            progress = {'candidates_tested': len(records), 'accepted': len(selected),
                        'requested': args.count, 'elapsed_seconds': time.monotonic()-start}
            (args.out/'progress.json').write_text(json.dumps(progress, indent=2)+'\n')
            print(json.dumps(progress), flush=True)
            if len(selected) == args.count: break
    if len(selected) != args.count: raise RuntimeError('population exhausted before target count')
    selection = {'format': 'proofgame-filtered-machines-1', 'selection_seed': args.seed,
                 'filter_attempt_limit': args.attempts, 'action_budget': args.budget,
                 'split': 'train', 'machines': selected}
    (args.out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    report = {**progress, 'complete': True, 'rejected': len(records)-len(selected),
              'filter_attempts': sum(r['attempts_completed'] for r in records),
              'filter_actions': sum(r['actions_simulated'] for r in records),
              'interpretation': 'Acceptance means this random search found a verified proof within the cap. Rejection is not a proof of difficulty or unprovability.'}
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
