"""Post-training diagnostics only. Never supplies data or rewards to the trainer."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import random
import torch
from .demo import run_demo
from .environment import Game
from .model import Policy
from .tasks import fixtures,sample,key
from .train import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("run",type=Path)
    args=p.parse_args(); folder=args.run
    config=json.loads((folder/"config.json").read_text())
    records=[json.loads(x) for x in (folder/"metrics.jsonl").read_text().splitlines()]
    rng=random.Random(config["seed"]); easy=[]; hist=Counter(); states=Counter()
    seven=['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']
    for row in records:
        m=sample(rng,tuple(config["states"]),'train')
        if key(m)!=row['machine_id']: raise RuntimeError('task stream does not reproduce')
        states[row['states']]+=1; hist.update(row['action_counts'])
        if m.rows[0][0]=='qH':
            g=Game(m,budget=64)
            for action in seven: g.step(action)
            if g.status!='SOLVED': raise RuntimeError('seven-action verification failed')
            easy.append(row['episode'])
    paths={}
    for name in ['one_step','two_step','three_step','blank_sweeper','two_state_forever']:
        _,trace=run_demo(name);g=Game(fixtures()[name],budget=64);prob=Fraction(1);branches=[]
        for action in trace:
            n=len(g.available_actions());branches.append(n);prob/=n;g.step(action)
        paths[name]={'actions':len(trace),'branch_counts':branches,'exact_sequence_probability':str(prob),
                     'probability_decimal':float(prob)}
    torch.set_num_threads(2); stochastic={}
    for label,name in [('untrained','initial_model.pt'),('trained','model.pt')]:
        saved=torch.load(folder/name,map_location='cpu',weights_only=True)
        policy=Policy(**saved['config']);policy.load_state_dict(saved['state_dict']);policy.eval()
        torch.manual_seed(config['seed']+200000)
        stochastic[label]=evaluate(policy,config['eval_episodes'],tuple(config['states']),config['budget'],config['seed']+100000,greedy=False)
    report={'training_episodes':len(records),'training_actions':sum(r['actions'] for r in records),
            'training_proofs':sum(int(r['reward']) for r in records),
            'state_counts':dict(states),'verified_seven_action_tasks':len(easy),'seven_action_episode_indices':easy,
            'invalid_actions':sum(r['invalid_actions'] for r in records),'action_counts':dict(hist),
            'exact_sequence_probabilities':paths,'sampled_policy_evaluation':stochastic,
            'note':'Proof-length audit performed after training. No proof fixtures or labels were provided to training.'}
    (folder/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('action_counts','seven_action_episode_indices','sampled_policy_evaluation','exact_sequence_probabilities')},indent=2))
    for label,r in stochastic.items(): print(label,'sampled policy:',r['status'])


if __name__=='__main__':main()
