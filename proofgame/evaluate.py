"""Evaluate a saved policy on held-out machines, or one supplied machine."""
import argparse
import json
from pathlib import Path
import torch
from .logic import Machine
from .model import Policy
from .train import evaluate,rollout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--states",type=int,nargs="+",default=[2,3])
    p.add_argument("--budget",type=int,default=128)
    p.add_argument("--seed",type=int,default=100007)
    p.add_argument("--sample-actions",action="store_true",help="sample the learned policy instead of greedy decoding")
    p.add_argument("--record-proofs",action="store_true",help="include successful certificates and exact action traces in evaluation JSON")
    p.add_argument("--machine",type=Path)
    p.add_argument("--proof",type=Path,default=Path("model_certificate.json"))
    args=p.parse_args()
    if args.episodes<0 or args.budget<1: p.error("invalid episode count or budget")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    policy=Policy(**checkpoint["config"]); policy.load_state_dict(checkpoint["state_dict"]); policy.eval()
    if args.machine:
        machine=Machine(json.loads(args.machine.read_text())["rows"])
        with torch.no_grad(): game,actions,_,_,_,reward,status=rollout(policy,machine,args.budget,greedy=not args.sample_actions)
        print(json.dumps({"status":status,"reward":reward,"actions":actions},indent=2))
        if reward: args.proof.write_text(json.dumps(game.certificate(),indent=2)+"\n")
    else: print(json.dumps(evaluate(policy,args.episodes,args.states,args.budget,args.seed,greedy=not args.sample_actions,collect_proofs=args.record_proofs),indent=2))


if __name__=="__main__": main()
