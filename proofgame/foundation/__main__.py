"""Generate checked demonstrations, pretrain, then run foundational RL."""
import argparse
import copy
import fcntl
import json
from pathlib import Path


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name in ('generate','pretrain','train','run','evaluate','verify'):
        s=sub.add_parser(name)
        if name=='verify':s.add_argument('--certificate',required=True);continue
        s.add_argument('--out',required=True)
        s.add_argument('--seed',type=int,default=42);s.add_argument('--workers',type=int,default=8)
        s.add_argument('--threads',type=int,default=1)
        if name!='evaluate':s.add_argument('--resume',action='store_true')
        if name in ('generate','run'):
            s.add_argument('--per-size',type=int,default=10000);s.add_argument('--train',type=int,default=1000)
            s.add_argument('--dev',type=int,default=400);s.add_argument('--test',type=int,default=400)
            s.add_argument('--budget',type=int,default=2048);s.add_argument('--simulation',type=int,default=0,help='0: deterministic tiered 128/1024/10k/100k budgets')
            s.add_argument('--compiler-steps',type=int,default=20000)
        if name in ('pretrain','train','evaluate'):s.add_argument('--data',required=True)
        if name in ('pretrain','train','run','evaluate'):
            s.add_argument('--device',choices=('cpu','mps'),default='cpu',help='learning/evaluation device; parallel RL inference stays on CPU')
            s.add_argument('--eval-attempts',type=int,default=100)
        if name in ('pretrain','train','run'):
            s.add_argument('--width',type=int,default=64);s.add_argument('--layers',type=int,default=2)
            s.add_argument('--max-tokens',type=int,default=4096);s.add_argument('--lr',type=float,default=3e-4)
            s.add_argument('--batch-size',type=int,default=64 if name=='pretrain' else 8)
            s.add_argument('--save-every',type=int,default=10)
        if name=='run':s.add_argument('--pretrain-batch-size',type=int,default=64)
        if name in ('pretrain','run'):
            s.add_argument('--curriculum-updates',type=int,default=1000);s.add_argument('--updates',type=int,default=15625);s.add_argument('--eval-every',type=int,default=1000)
            s.add_argument('--eval-machines',type=int,default=40)
        if name in ('train','run'):
            s.add_argument('--epochs',type=int,default=100);s.add_argument('--attempts',type=int,default=8)
            s.add_argument('--rounds',type=int,default=6);s.add_argument('--length-penalty',type=float,default=.3)
            s.add_argument('--init',default=None);s.add_argument('--skip-special',action='store_true');s.add_argument('--skip-final-test',action='store_true')
        if name=='evaluate':
            s.add_argument('--checkpoint',required=True);s.add_argument('--split',choices=('dev','test','special'),default='test')
    return p


def main():
    p=parser();a=p.parse_args()
    for k in ('workers','threads','per_size','train','dev','test','budget','compiler_steps','pretrain_batch_size','width','layers','max_tokens','batch_size','save_every','eval_every','eval_machines','eval_attempts','attempts','rounds'):
        if hasattr(a,k) and getattr(a,k)<1:p.error(k+' must be positive')
    for k in ('epochs','updates','simulation','curriculum_updates'):
        if hasattr(a,k) and getattr(a,k)<0:p.error(k+' must be nonnegative')
    if hasattr(a,'lr') and not 0<a.lr<1:p.error('lr must be in (0,1)')
    if hasattr(a,'length_penalty') and not 0<=a.length_penalty<1:p.error('length penalty must be in [0,1)')
    if a.command=='verify':
        from .kernel import check
        print(check(json.loads(Path(a.certificate).read_text())));return
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_name(out.name+'.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:p.error('another process owns this output')
        if a.command=='generate':
            from .data import generate
            generate(a)
        elif a.command=='run':
            from .data import generate
            from .training import pretrain,train
            base=Path(a.out);base.mkdir(parents=True,exist_ok=True)
            g=copy.copy(a);g.out=str(base/'data');g.resume=a.resume and (Path(g.out)/'config.json').exists();generate(g)
            s=copy.copy(a);s.command='pretrain';s.batch_size=a.pretrain_batch_size;s.data=g.out;s.out=str(base/'pretrain');s.resume=a.resume and (Path(s.out)/'checkpoint.pt').exists();pretrain(s)
            r=copy.copy(a);r.command='train';r.data=g.out;r.out=str(base/'rl');r.init=str(Path(s.out)/'best.pt' if (Path(s.out)/'best.pt').exists() else Path(s.out)/'checkpoint.pt');r.resume=a.resume and (Path(r.out)/'checkpoint.pt').exists();train(r)
        else:
            from . import training
            getattr(training,{'train':'train','pretrain':'pretrain','evaluate':'evaluate'}[a.command])(a)

if __name__=='__main__':main()
