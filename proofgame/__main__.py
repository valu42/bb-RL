"""Run with python -m proofgame --help."""
import argparse
import json
from pathlib import Path
import random
from .logic import Machine, Invalid
from .environment import Game
from .kernel import check
from .tasks import fixtures, sample


def main():
    p=argparse.ArgumentParser(description="Proof Game: finite actions, checked proofs, no initial archive")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("demo",help="run human-written checker fixtures; not model output")
    d.add_argument("name",choices=list(fixtures()))
    d.add_argument("--out",type=Path)
    c=sub.add_parser("check",help="independently recheck a JSON certificate")
    c.add_argument("file",type=Path)
    q=sub.add_parser("play",help="interactive terminal proof editor")
    q.add_argument("--fixture",choices=list(fixtures()),default="one_step")
    q.add_argument("--machine",type=Path)
    q.add_argument("--budget",type=int,default=256)
    q.add_argument("--out",type=Path,default=Path("certificate.json"))
    s=sub.add_parser("sample",help="sample an unlabelled machine")
    s.add_argument("--states",type=int,nargs="+",default=[2,3]); s.add_argument("--seed",type=int,default=7)
    args=p.parse_args()
    if args.command=="sample": print(json.dumps(sample(random.Random(args.seed),args.states).to_json(),indent=2))
    elif args.command=="check":
        try: print("ACCEPTED",check(json.loads(args.file.read_text())))
        except (Invalid,KeyError,TypeError,ValueError) as e: p.exit(1,f"REJECTED: {e}\n")
    elif args.command=="demo":
        from .demo import run_demo
        game,trace=run_demo(args.name)
        print("Human-written verification fixture — not a learned proof.")
        for i,a in enumerate(trace,1): print(f"{i:3}  {a}")
        print(f"{game.status}: {check(game.certificate())}; {len(trace)} actions")
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(game.certificate(),indent=2)+"\n")
            args.out.with_suffix(".actions.json").write_text(json.dumps(trace,indent=2)+"\n")
    elif args.command=="play":
        m=Machine(json.loads(args.machine.read_text())["rows"]) if args.machine else fixtures()[args.fixture]
        game=Game(m,args.budget)
        while game.status=="ACTIVE":
            obs=game.observation()
            print(f"\n{game.left} actions left | answer={game.answer} | focus={game.focus}")
            if game.error: print("Last action rejected:",game.error)
            g=game.current()
            if g: print(json.dumps(g.text(),indent=2))
            if obs["draft"]: print("Building:",obs["draft"])
            for i,a in enumerate(obs["actions"]):
                print(f"{i:3}: {a['id']}"+(f" = {a['expression']}" if "expression" in a else ""))
            try: line=input("Action index or action ID (quit to exit): ").strip()
            except (EOFError,KeyboardInterrupt): print(); return
            if line=="quit": return
            if line.isdecimal():
                i=int(line)
                if i>=len(obs["actions"]): print("Index out of range"); continue
                line=obs["actions"][i]["id"]
            game.step(line)
        print(game.status)
        if game.status=="SOLVED":
            args.out.write_text(json.dumps(game.certificate(),indent=2)+"\n")
            print("Checked certificate saved to",args.out)


if __name__=="__main__": main()
