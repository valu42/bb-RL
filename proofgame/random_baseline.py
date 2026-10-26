"""Uniform-action Monte Carlo; no model, training examples, or theorem hints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
from .environment import Game
from .tasks import fixtures,sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes",type=int,default=1000)
    p.add_argument("--budget",type=int,default=64)
    p.add_argument("--task",choices=["mixed","one_step"],default="mixed")
    p.add_argument("--seed",type=int,default=213)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.episodes<1 or args.budget<1: p.error("positive episode count and budget required")
    task_rng=random.Random(args.seed); action_rng=random.Random(args.seed+1)
    counts=Counter(); successes=[]; start=time.monotonic()
    for i in range(args.episodes):
        m=sample(task_rng,(2,3),"test") if args.task=="mixed" else fixtures()["one_step"]
        g=Game(m,budget=args.budget)
        while g.status=="ACTIVE": g.step(action_rng.choice(g.available_actions()))
        counts[g.status]+=1
        if g.status=="SOLVED": successes.append({"episode":i,"certificate":g.certificate()})
        if (i+1)%100==0: print(json.dumps({"episodes":i+1,"successes":counts['SOLVED'],"seconds":time.monotonic()-start}),flush=True)
    report={"episodes":args.episodes,"budget":args.budget,"task":args.task,"seed":args.seed,
            "status":dict(counts),"success_rate":counts['SOLVED']/args.episodes,
            "seconds":time.monotonic()-start,"successes":successes,
            "zero_success_one_sided_95_percent_upper_bound":1-0.05**(1/args.episodes) if not successes else None,
            "interpretation":"Uniform choices among currently offered actions; includes rejected actions, resets, and alternate proof paths."}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!='successes'},indent=2))


if __name__=="__main__": main()
