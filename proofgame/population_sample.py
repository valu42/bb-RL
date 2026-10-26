"""Estimate random-search coverage using uniformly sampled labelled machines.

Each machine gets an independent seeded action stream. No filtering on behavior
and no proof demonstrations. Confidence intervals describe the sampled search
procedure, not intrinsic mathematical provability.
"""
import argparse
from collections import Counter
from itertools import product
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import time
from .cached_random import Simulator
from .shared_random import SharedSimulator
from .logic import Machine
from .kernel import check
from .tasks import key


def from_rank(rank,states=2):
    options=tuple(product(tuple(f'q{i}' for i in range(states))+('qH',),('b0','b1'),('L','R')))
    total=len(options)**(2*states)
    if not 0<=rank<total: raise ValueError('machine rank out of range')
    rows=[]
    for _ in range(2*states):
        rank,digit=divmod(rank,len(options));rows.append(options[digit])
    return Machine(tuple(rows))


_shared_simulator=None


def worker(task):
    global _shared_simulator
    rank,seed,attempts,budget,cache_states=task
    machine=from_rank(rank)
    if _shared_simulator is None:_shared_simulator=SharedSimulator(machine,max(500000,cache_states))
    sim=_shared_simulator;sim.set_machine(machine);rng=random.Random(seed)
    prior_hits,prior_misses=sim.hits,sim.misses
    start=time.monotonic();success=None;steps=0
    for i in range(attempts):
        cert,n=sim.attempt(rng,budget);steps+=n
        if cert:
            check(cert)
            if cert['machine']!=machine.to_json():raise RuntimeError('wrong assigned machine')
            success={'answer':cert['answer'],'attempt':i+1,'actions_in_attempt':n,'certificate':cert}
            break
    return {'rank':rank,'machine_id':key(machine),'machine':machine.to_json(),
            'action_seed':seed,'attempts_completed':i+1,'actions_simulated':steps,
            'success':success,'seconds':time.monotonic()-start,
            'cache_hits':sim.hits-prior_hits,'cache_misses':sim.misses-prior_misses,
            'backend':'shared-derived-goals-v2'}


def interval(successes,n):
    # Wilson score interval, nominal 95%, without finite-population correction.
    z=1.959963984540054;p=successes/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=int,default=512)
    p.add_argument('--attempts',type=int,default=10000)
    p.add_argument('--budget',type=int,default=20)
    p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--cache-states',type=int,default=10000)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='resume the same fixed sample and per-machine seeds')
    args=p.parse_args();total=12**4
    if not 1<=args.sample<=total or min(args.attempts,args.budget,args.workers,args.cache_states)<1:
        p.error('invalid sample or resource limit')
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'records.jsonl').exists() and not args.resume:p.error('choose a fresh output directory or --resume')
    rng=random.Random(args.seed);ranks=rng.sample(range(total),args.sample)
    tasks=[(rank,rng.getrandbits(64),args.attempts,args.budget,args.cache_states) for rank in ranks]
    plan={'population':total,'sample_without_replacement':args.sample,'sampling_seed':args.seed,
          'attempts_per_machine':args.attempts,'actions_per_attempt':args.budget,'workers':args.workers,
          'cache_states':args.cache_states,'ranks_and_seeds':[(t[0],t[1]) for t in tasks],
          'sampling':'uniform over all 20,736 labelled transition tables, not restricted to the training/test split',
          'random_actions':'independent seeded streams, uniform over each available action menu'}
    if args.resume:
        old=json.loads((args.out/'plan.json').read_text())
        for field in ['population','sample_without_replacement','sampling_seed','attempts_per_machine','actions_per_attempt','ranks_and_seeds']:
            if json.loads(json.dumps(old[field]))!=json.loads(json.dumps(plan[field])):p.error('resume configuration differs from saved plan')
    (args.out/'process.json').write_text(json.dumps({'pid':os.getpid()},indent=2)+'\n')
    (args.out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    start=time.monotonic();records=[]
    if args.resume:
        records=[json.loads(line) for line in (args.out/'records.jsonl').read_text().splitlines()]
        if len({r['rank'] for r in records})!=len(records):p.error('duplicate machine records')
        planned={rank:seed for rank,seed,*_ in tasks}
        for row in records:
            if planned.get(row['rank'])!=row['action_seed']:p.error('record does not belong to this plan')
            if row['success']:check(row['success']['certificate'])
    completed={r['rank'] for r in records};remaining=[t for t in tasks if t[0] not in completed]
    with (args.out/'records.jsonl').open('a' if args.resume else 'w') as log,mp.get_context('spawn').Pool(args.workers) as pool:
        for row in pool.imap_unordered(worker,remaining,chunksize=1):
            records.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if len(records)%16==0 or len(records)==args.sample:
                summary={'completed':len(records),'sample':args.sample,'proved':sum(r['success'] is not None for r in records),
                         'elapsed_seconds':time.monotonic()-start}
                (args.out/'progress.json').write_text(json.dumps(summary,indent=2)+'\n')
                print(json.dumps(summary),flush=True)
    successes=sum(r['success'] is not None for r in records)
    answers=Counter(r['success']['answer'] for r in records if r['success'])
    report={**plan,'complete':True,'proved':successes,'unproved':len(records)-successes,
            'proved_answers':dict(answers),'observed_fraction':successes/len(records),
            'actions_simulated':sum(r['actions_simulated'] for r in records),
            'attempts_completed':sum(r['attempts_completed'] for r in records),
            'elapsed_seconds':time.monotonic()-start,
            'interpretation':'Raw observed Monte Carlo result on the specified sampled machines and independent random-action streams. No extrapolation. Unproved means no proof found within this budget.'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='ranks_and_seeds'},indent=2))


if __name__=='__main__':main()
