"""On-policy REINFORCE + learned baseline; terminal verified proof reward only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time
import torch
from .environment import Game
from .model import Policy, ModelLimit
from .tasks import sample, key


def rollout(policy,machine,budget,rng=None,greedy=False):
    game=Game(machine,budget=budget)
    logps=[]; values=[]; entropies=[]; actions=[]; status=None
    while game.status=="ACTIVE":
        obs=game.observation()
        if policy is None:
            action=rng.choice(obs["actions"])["id"]
        else:
            try: logits,value=policy(obs)
            except ModelLimit: status="MODEL_LIMIT"; break
            distribution=torch.distributions.Categorical(logits=logits)
            choice=logits.argmax() if greedy else distribution.sample()
            action=obs["actions"][int(choice.item())]["id"]
            logps.append(distribution.log_prob(choice)); values.append(value)
            entropies.append(distribution.entropy())
        game.step(action); actions.append(action)
    reward=float(game.status=="SOLVED")
    return game,actions,logps,values,entropies,reward,status or game.status


def evaluate(policy,count,states,budget,seed,greedy=True,collect_proofs=False):
    rng=random.Random(seed); action_rng=random.Random(seed+1)
    counts=Counter(); answers=Counter(); ids=[]; successes=[]
    with torch.no_grad():
        for _ in range(count):
            m=sample(rng,states,"test"); ids.append(key(m))
            game,actions,_,_,_,reward,status=rollout(policy,m,budget,action_rng,greedy=greedy and policy is not None)
            counts[status]+=1
            if reward:
                answers[game.answer]+=1
                if collect_proofs:
                    successes.append({"episode":len(ids)-1,"machine_id":key(m),
                                      "actions":actions,"certificate":game.certificate()})
    result={"episodes":count,"unique_machines":len(set(ids)),"status":dict(counts),
            "proved_answers":dict(answers),"success_rate":counts["SOLVED"]/max(count,1),
            "machine_ids":ids}
    if collect_proofs:result["successes"]=successes
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes",type=int,default=1000)
    p.add_argument("--budget",type=int,default=128)
    p.add_argument("--states",type=int,nargs="+",default=[2,3])
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--layers",type=int,default=2)
    p.add_argument("--max-tokens",type=int,default=4096)
    p.add_argument("--lr",type=float,default=3e-4)
    p.add_argument("--entropy",type=float,default=0.,help="optional regularizer, zero for strict baseline")
    p.add_argument("--eval-episodes",type=int,default=20)
    p.add_argument("--out",type=Path,default=Path("runs/experiment"))
    p.add_argument("--threads",type=int,default=2)
    args=p.parse_args()
    if args.episodes<1 or args.eval_episodes<0 or args.budget<1: p.error("invalid episode counts or budget")
    args.out.mkdir(parents=True,exist_ok=True)
    # Refuse overwriting an experiment rather than silently mixing training records.
    if (args.out/"metrics.jsonl").exists(): p.error("output already contains a run; choose a new --out")
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); random.seed(args.seed)
    rng=random.Random(args.seed)
    policy=Policy(width=args.width,layers=args.layers,max_tokens=args.max_tokens)
    initial={k:v.detach().clone() for k,v in policy.state_dict().items()}
    opt=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=0.)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({"model":policy.config,"parameters":sum(x.numel() for x in policy.parameters()),
                   "torch":torch.__version__,"reward":"1 only for a fully checked submitted proof",
                   "training_examples":0,"initial_archive":0,"checkpoint_resumed":False})
    (args.out/"config.json").write_text(json.dumps(config,indent=2)+"\n")
    torch.save({"config":policy.config,"state_dict":policy.state_dict(),"seed":args.seed},args.out/"initial_model.pt")
    policy.eval()
    before=evaluate(policy,args.eval_episodes,args.states,args.budget,args.seed+100000)
    baseline=evaluate(None,args.eval_episodes,args.states,args.budget,args.seed+100000)
    policy.train(); solved=0; start=time.monotonic()
    with (args.out/"metrics.jsonl").open("w") as log:
        for episode in range(args.episodes):
            m=sample(rng,args.states,"train")
            # Do not retain an entire episode's attention graphs. Recompute each
            # chosen action below, before the single on-policy parameter update.
            with torch.no_grad():
                game,actions,logps,values,entropies,reward,status=rollout(policy,m,args.budget)
            loss_value=None; invalid_actions=0
            if actions:
                replay=Game(m,budget=args.budget); opt.zero_grad(); loss_value=0.
                for action in actions:
                    obs=replay.observation(); logits,value=policy(obs)
                    distribution=torch.distributions.Categorical(logits=logits)
                    chosen=next(i for i,a in enumerate(obs["actions"]) if a["id"]==action)
                    logp=distribution.log_prob(torch.tensor(chosen,device=logits.device))
                    loss=-logp*(reward-value.detach())+0.5*(value-reward).square()/len(actions)-args.entropy*distribution.entropy()/len(actions)
                    if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
                    loss.backward(); loss_value+=float(loss.detach()); replay.step(action)
                    invalid_actions+=int(replay.error is not None)
                torch.nn.utils.clip_grad_norm_(policy.parameters(),1.); opt.step()
            solved+=int(reward)
            row={"episode":episode,"machine_id":key(m),"states":len(m.rows)//2,
                 "reward":reward,"status":status,"actions":len(actions),"loss":loss_value,
                 "invalid_actions":invalid_actions,"action_counts":dict(Counter(actions))}
            log.write(json.dumps(row)+"\n"); log.flush()
            if reward:
                (args.out/f"proof_{episode}.json").write_text(json.dumps(game.certificate(),indent=2)+"\n")
            if episode%10==0 or episode==args.episodes-1: print(json.dumps(row),flush=True)
    policy.eval()
    after=evaluate(policy,args.eval_episodes,args.states,args.budget,args.seed+100000)
    change=sum(float((v-initial[k]).abs().sum()) for k,v in policy.state_dict().items())
    report={"training_episodes":args.episodes,"training_proofs":solved,
            "seconds":time.monotonic()-start,"parameter_l1_change":change,
            "uniform_random":baseline,"untrained_greedy":before,"trained_greedy":after,
            "interpretation":"Success requires an accepted submitted proof; parameter changes alone do not establish learned proof discovery."}
    torch.save({"config":policy.config,"state_dict":policy.state_dict(),"seed":args.seed},args.out/"model.pt")
    (args.out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
