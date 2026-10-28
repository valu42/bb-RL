"""Supervised action learning and on-policy branch-race RL in foundation v2.

No compiler, teacher, simulator, or legacy proof checker is imported here.
"""
import concurrent.futures
import copy
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import time
import torch
from ..model import Policy,ModelLimit
from .environment import Game
from .kernel import FORMAT,check
from .logic import Machine
from .data import atomic,digest,hash_file,Corpus,VERSION,language_hash


def save(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)

def load(path,device='cpu'):return torch.load(path,map_location=device,weights_only=False)

def finite_step(policy,opt):
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in policy.parameters()):
        opt.zero_grad();return False
    torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
    weights=copy.deepcopy(policy.state_dict());previous=copy.deepcopy(opt.state_dict());opt.step()
    valid=all(torch.isfinite(p).all() for p in policy.parameters())
    valid=valid and all(not isinstance(x,torch.Tensor) or torch.isfinite(x).all() for s in opt.state.values() for x in s.values())
    if not valid:policy.load_state_dict(weights);opt.load_state_dict(previous)
    return bool(valid)

def proof_text(summary):
    c=summary['machines_proved_by_answer'];a=summary['mean_first_proof_actions_by_answer']
    mean=lambda v:'n/a' if v is None else f'{v:.1f}'
    return f"HALT {c['HALT']} (avg {mean(a['HALT'])} actions) | NONHALT {c['NONHALT']} (avg {mean(a['NONHALT'])} actions) | unproved {summary['machines_unproved']}"


def log(out,row):
    with (Path(out)/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    atomic(Path(out)/'progress.json',row);print(json.dumps(row),flush=True)

def corpus_info(path):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if m['config'].get('language_hash')!=language_hash():raise ValueError('corpus language implementation changed; regenerate corpus')
    if m['version']!=VERSION or hash_file(path/'corpus.jsonl')!=m['corpus_sha256']:raise ValueError('corpus version/hash mismatch')
    machines=json.loads((path/'machines.json').read_text())
    if digest(machines)!=m['source_hash']:raise ValueError('machine manifest mismatch')
    return m,machines

def checkpoint(policy,opt,config,**state):
    return {'language':FORMAT,'config':config,'model':policy.config,'state_dict':policy.state_dict(),
            'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
            'mps_rng':torch.mps.get_rng_state() if next(policy.parameters()).device.type=='mps' else None,**state}

def restore_rng(s):
    torch.set_rng_state(s['torch_rng'].cpu())
    if s.get('mps_rng') is not None:torch.mps.set_rng_state(s['mps_rng'].cpu())

def init(args,kind):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    m,machines=corpus_info(args.data)
    cfg={k:v for k,v in vars(args).items() if k not in ('resume','updates','epochs','workers','out','command','save_every')}
    cfg.update(kind=kind,corpus_hash=m['corpus_sha256'],language=FORMAT,language_hash=language_hash(),budget=m['config']['budget'])
    if kind=='rl' and args.init:cfg['initial_sha256']=hash_file(args.init)
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    if args.resume:
        s=load(out/'checkpoint.pt')
        if s['language']!=FORMAT or s['config']!=cfg:raise ValueError('resume configuration mismatch')
        policy=Policy(**s['model']).to(args.device);policy.load_state_dict(s['state_dict'])
    else:
        if any(out.iterdir()):raise ValueError('run output must be empty or use --resume')
        if kind=='rl' and args.init:
            s0=load(args.init)
            if s0.get('language')!=FORMAT:raise ValueError('initial checkpoint is not foundation v2')
            if s0['config']['corpus_hash']!=cfg['corpus_hash']:raise ValueError('initial checkpoint corpus mismatch')
            policy=Policy(**s0['model']).to(args.device);policy.load_state_dict(s0['state_dict'])
            cfg['initial_sha256']=hash_file(args.init)
        else:policy=Policy(width=args.width,layers=args.layers,heads=4,max_tokens=args.max_tokens,representation='ast').to(args.device)
        s=None
        atomic(out/'config.json',cfg)
    # Reconstruct the source hash on resume, rather than accepting changed weights.
    if kind=='rl' and args.init:
        cfg['initial_sha256']=hash_file(args.init)
        if s and s['config']!=cfg:raise ValueError('initial checkpoint changed')
    opt=torch.optim.AdamW(policy.parameters(),lr=args.lr,weight_decay=0.)
    if s:opt.load_state_dict(s['optimizer']);restore_rng(s)
    return out,cfg,machines,policy,opt,s


def rollout(policy,row,budget,seed,forced=None):
    rng=random.Random(seed);g=Game(Machine(row['machine']['rows']),budget);actions=[];probs=[]
    stats={'inference_seconds':0.,'game_seconds':0.,'observation_seconds':0.}
    with torch.no_grad():
        while g.status=='ACTIVE':
            t=time.monotonic();obs=g.observation();stats['observation_seconds']+=time.monotonic()-t
            if not actions and forced:action='choose:'+forced;prob=1.
            else:
                t=time.monotonic()
                try:logits,_=policy(obs)
                except ModelLimit:g.status='MODEL_LIMIT';break
                ps=torch.softmax(logits,0).cpu().tolist();stats['inference_seconds']+=time.monotonic()-t
                if not all(math.isfinite(x) for x in ps):raise RuntimeError('nonfinite rollout probabilities')
                i=rng.choices(range(len(ps)),weights=ps)[0];action=obs['actions'][i]['id'];prob=ps[i]
            t=time.monotonic();g.step(action,observe=False);stats['game_seconds']+=time.monotonic()-t
            actions.append(action);probs.append(prob)
    cert=g.certificate() if g.status=='SOLVED' else None
    if cert:check(cert)
    return {'actions':actions,'probabilities':probs,'status':g.status,'certificate':cert,'forced':forced,'timing':stats}

_worker=None
_worker_version=None

def worker_init(threads):
    global _worker,_worker_version
    torch.set_num_threads(threads);_worker=None;_worker_version=None

def machine_job(task):
    global _worker,_worker_version
    snapshot,version,row,budget,seed,attempts,rounds,training=task
    if _worker_version!=version:
        s=load(snapshot)
        if _worker is None:_worker=Policy(**s['model'])
        _worker.load_state_dict(s['state_dict']);_worker.eval();_worker_version=version
    successes=[];group=[];actions=0;executed=0;statuses={};timing={'inference_seconds':0.,'game_seconds':0.,'observation_seconds':0.}
    for round_no in range(rounds if training else 1):
        groups={};winner=None
        for answer in (('HALT','NONHALT') if training else (None,)):
            trajectories=[]
            for attempt in range(attempts):
                trajectory=rollout(_worker,row,budget,seed+executed*104729,answer);executed+=1
                trajectories.append(trajectory);actions+=len(trajectory['actions']);statuses[trajectory['status']]=statuses.get(trajectory['status'],0)+1
                for k,v in trajectory['timing'].items():timing[k]+=v
                if trajectory['certificate']:
                    successes.append({'attempt':executed,'actions':trajectory['actions'],'certificate':trajectory['certificate']})
                    if winner and winner!=trajectory['certificate']['answer']:raise RuntimeError('contradictory verified proofs')
                    winner=trajectory['certificate']['answer']
                    if not training:break
            groups[answer]=trajectories
        if winner:
            if training:group=groups[winner]
            break
    return {**row,'successes':successes,'group':group,'attempts':executed,'actions':actions,'statuses':statuses,'timing':timing}


def autonomous(policy,machines,budget,attempts,seed):
    rows=[];policy.eval()
    for i,row in enumerate(machines):
        successes=[]
        for a in range(attempts):
            t=rollout(policy,row,budget,seed+i*100000+a)
            if t['certificate']:
                successes=[{'attempt':a+1,'actions':t['actions'],'certificate':t['certificate']}];break
        rows.append({**row,'successes':successes})
    return summarize(rows)


def summarize(rows,by_size=True):
    counts={'HALT':0,'NONHALT':0};lengths={'HALT':0,'NONHALT':0}
    by_states={}
    for row in rows:
        if row['successes']:
            first=row['successes'][0];answer=first['certificate']['answer'];counts[answer]+=1;lengths[answer]+=len(first['actions'])
        by_states.setdefault(str(row['states']),[]).append(row)
    return {'machines':len(rows),'machines_proved_by_answer':counts,'machines_unproved':len(rows)-sum(counts.values()),
            'mean_first_proof_actions_by_answer':{a:lengths[a]/counts[a] if counts[a] else None for a in counts},
            'by_states':{n:summarize(rs,False) for n,rs in by_states.items()} if by_size else {}}


def balanced_prefix(rows,count):
    buckets={}
    for r in rows:buckets.setdefault(r['states'],[]).append(r)
    result=[]
    while buckets and len(result)<count:
        for n in list(sorted(buckets)):
            result.append(buckets[n].pop(0))
            if not buckets[n]:del buckets[n]
            if len(result)==count:break
    return result


def pretrain(args):
    out,cfg,machines,policy,opt,s=init(args,'pretrain');budget=cfg['budget'];rng=random.Random(args.seed)
    corpus=Corpus(Path(args.data)/'corpus.jsonl');start_update=s['update'] if s else 0
    if s:rng.setstate(s['rng'])
    verified=set();best=s.get('best',-1) if s else -1
    for update in range(start_update,args.updates):
        start=time.monotonic();opt.zero_grad();loss_sum=0.;correct=0;accepted=0;skipped=0;policy.train()
        while accepted<args.batch_size:
            if skipped>args.batch_size*100:raise RuntimeError('too many context-limited examples; inspect corpus/model budgets')
            row,example=corpus.sample(rng,0 if update<args.curriculum_updates else 1 if update<2*args.curriculum_updates else 2)
            if example['proof_id'] not in verified:
                check(example['certificate']);verified.add(example['proof_id'])
            pos=rng.randrange(len(example['actions']));g=Game(Machine(row['machine']['rows']),budget)
            for a in example['actions'][:pos]:
                g.step(a,observe=False)
                if g.error:raise ValueError('invalid demonstration replay: '+g.error)
            obs=g.observation();target=example['actions'][pos];index=next(i for i,a in enumerate(obs['actions']) if a['id']==target)
            try:logits,_=policy(obs)
            except ModelLimit:skipped+=1;continue
            loss=-torch.log_softmax(logits,0)[index]/args.batch_size
            if not torch.isfinite(loss):raise RuntimeError('nonfinite supervised loss')
            loss.backward();loss_sum+=float(loss.detach());correct+=int(logits.argmax().item()==index);accepted+=1
        healthy=finite_step(policy,opt)
        record={'event':'pretrain','update':update+1,'sampled_targets':(update+1)*args.batch_size,'loss':loss_sum,'accuracy':correct/accepted,
                'skipped_model_limit':skipped,'update_accepted':healthy,'seconds':time.monotonic()-start}
        if (update+1)%args.eval_every==0 or update+1==args.updates:
            result=autonomous(policy,balanced_prefix(machines['splits']['dev'],args.eval_machines),budget,args.eval_attempts,args.seed+900000)
            record['dev']=result;score=sum(result['machines_proved_by_answer'].values())
        else:score=None
        state=checkpoint(policy,opt,cfg,update=update+1,rng=rng.getstate(),best=max(best,score if score is not None else -1))
        if score is not None and score>best:best=score;save(out/'best.pt',state)
        if (update+1)%args.save_every==0 or score is not None:save(out/'checkpoint.pt',state)
        log(out,record)
    if args.updates==0:save(out/'checkpoint.pt',checkpoint(policy,opt,cfg,update=0,rng=rng.getstate(),best=best))


def learn(policy,opt,results,budget,penalty):
    groups=[]
    for r in results:
        ts=r['group']
        if len(ts)<2:continue
        rewards=[1-penalty*len(t['actions'])/budget if t['certificate'] else 0 for t in ts]
        if max(rewards)==min(rewards):continue
        groups.append((r,ts,rewards))
    if not groups:return {'informative_groups':0,'all_failed_groups':sum(not r.get('successes',any(t['certificate'] for t in r['group'])) for r in results),'loss':0.,'update_accepted':False}
    opt.zero_grad();total=0.;policy.eval()
    for r,ts,rewards in groups:
        for t,reward in zip(ts,rewards):
            advantage=(len(ts)*reward-sum(rewards))/(len(ts)-1)
            if abs(advantage)<1e-12:continue
            g=Game(Machine(r['machine']['rows']),budget)
            for i,action in enumerate(t['actions']):
                if not (i==0 and t['forced']):
                    obs=g.observation();j=next(j for j,a in enumerate(obs['actions']) if a['id']==action)
                    logits,_=policy(obs);lp=torch.log_softmax(logits,0)[j]
                    if not math.isclose(float(lp.exp().detach()),t['probabilities'][i],rel_tol=2e-4,abs_tol=1e-6):raise RuntimeError('on-policy replay probability mismatch')
                    loss=-lp*advantage/(len(groups)*len(ts));loss.backward();total+=float(loss.detach())
                g.step(action,observe=False)
            if t['certificate']:
                check(t['certificate'])
                if g.status!='SOLVED' or digest(g.certificate())!=digest(t['certificate']):raise RuntimeError('RL replay mismatch')
    return {'informative_groups':len(groups),'all_failed_groups':sum(not r.get('successes',any(t['certificate'] for t in r['group'])) for r in results),'loss':total,'update_accepted':finite_step(policy,opt)}


def train(args):
    out,cfg,machines,policy,opt,s=init(args,'rl');budget=cfg['budget'];snapshot=out/'worker.pt'
    phase_index=s['phase_index'] if s else 0;next_index=s['next_index'] if s else 0;updates=s['updates'] if s else 0
    phases=[(0,'evaluation')]+[(e,p) for e in range(1,args.epochs+1) for p in ('training','evaluation')]
    batch_dir=out/'batches';batch_dir.mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init,initargs=(1,),mp_context=multiprocessing.get_context('spawn')) as pool:
        for pi in range(phase_index,len(phases)):
            epoch,phase=phases[pi];training=phase=='training';rows=list(machines['splits']['train' if training else 'dev'])
            if training:random.Random(args.seed+epoch).shuffle(rows)
            start_index=next_index if pi==phase_index else 0;results=[]
            for i in range(0,start_index,args.batch_size):results.extend(json.loads((batch_dir/f'{pi}_{i}.json').read_text())['results'])
            began=time.monotonic()
            for i in range(start_index,len(rows),args.batch_size):
                t0=time.monotonic();save(snapshot,{'model':policy.config,'state_dict':{k:v.detach().cpu() for k,v in policy.state_dict().items()}})
                snapshot_seconds=time.monotonic()-t0;batch=rows[i:i+args.batch_size]
                tasks=[(str(snapshot.resolve()),(pi,i),row,budget,args.seed+pi*1000000+(i+j)*10000,
                        args.attempts if training else args.eval_attempts,args.rounds,training) for j,row in enumerate(batch)]
                t=time.monotonic();batch_results=list(pool.map(machine_job,tasks));rollout_seconds=time.monotonic()-t
                t=time.monotonic();learning=learn(policy,opt,batch_results,budget,args.length_penalty) if training else {};learn_seconds=time.monotonic()-t
                updates+=int(learning.get('update_accepted',False))
                for r in batch_results:r.pop('group',None)
                atomic(batch_dir/f'{pi}_{i}.json',{'results':batch_results});results.extend(batch_results)
                t=time.monotonic();save(out/'checkpoint.pt',checkpoint(policy,opt,cfg,phase_index=pi,next_index=i+len(batch),updates=updates));checkpoint_seconds=time.monotonic()-t
                elapsed=time.monotonic()-began;done=i+len(batch);eta=elapsed*(len(rows)-done)/max(1,done-start_index)
                summary=summarize(results);latest=summarize(batch_results)
                record={'event':'batch_profile','epoch':epoch,'phase':phase,'machines_done':done,'machines_total':len(rows),'eta_seconds':eta,
                        'snapshot_seconds':snapshot_seconds,'rollout_seconds':rollout_seconds,'learn_seconds':learn_seconds,'checkpoint_seconds':checkpoint_seconds,
                        'worker_seconds_sum':{k:sum(r['timing'][k] for r in batch_results) for k in batch_results[0]['timing']},
                        'actions_per_second':sum(r['actions'] for r in batch_results)/max(rollout_seconds,1e-9),'learning':learning,'proof_progress':summary,'last_batch':latest}
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                atomic(out/'progress.json',record)
                bar='█'*int(24*done/len(rows))+'░'*(24-int(24*done/len(rows)))
                print(f'Epoch {epoch} {phase} [{bar}] {done}/{len(rows)} | ETA ~{eta/60:.1f}m | cumulative '+proof_text(summary),flush=True)
                print(f'Last batch ({len(batch)} machines): '+proof_text(latest),flush=True)
            summary=summarize(results);summary.update(epoch=epoch,phase=phase)
            atomic(out/f'epoch_{epoch:02d}_{phase}.json',summary)
            for n,part in sorted(summary['by_states'].items()):print(f'Epoch {epoch} {phase} | {n}-state: '+proof_text(part),flush=True)
            if not training:
                save(out/f'model_epoch_{epoch:02d}.pt',checkpoint(policy,opt,cfg,phase_index=pi+1,next_index=0,updates=updates))
                if not args.skip_special:
                    save(snapshot,{'model':policy.config,'state_dict':{k:v.detach().cpu() for k,v in policy.state_dict().items()}})
                    special=list(pool.map(machine_job,[(str(snapshot.resolve()),('special',epoch),r,budget,args.seed+epoch*100000+j,args.eval_attempts,1,False) for j,r in enumerate(machines['splits']['special'])]))
                    for r in special:
                        r.pop('group',None)
                        print('Special eval '+str(epoch)+': '+r.get('name','special')+': '+('PROVED' if r['successes'] else 'NOT_PROVED'),flush=True)
                        for proof in r['successes']:print(json.dumps(proof,indent=2),flush=True)
                    atomic(out/f'special_benchmarks_epoch_{epoch:02d}.json',{'epoch':epoch,'results':special,'summary':summarize(special)})
            save(out/'checkpoint.pt',checkpoint(policy,opt,cfg,phase_index=pi+1,next_index=0,updates=updates))
            next_index=0


        final_path=out/f'final_test_epoch_{args.epochs:02d}.json'
        if not args.skip_final_test and not final_path.exists():
            save(snapshot,{'model':policy.config,'state_dict':{k:v.detach().cpu() for k,v in policy.state_dict().items()}})
            tested=list(pool.map(machine_job,[(str(snapshot.resolve()),('test',args.epochs),r,budget,args.seed+70000000+j,args.eval_attempts,1,False) for j,r in enumerate(machines['splits']['test'])]))
            for r in tested:r.pop('group',None)
            atomic(final_path,{'epoch':args.epochs,'summary':summarize(tested),'results':tested})
            print(json.dumps({'event':'final_test',**summarize(tested)}),flush=True)


def evaluate(args):
    s=load(args.checkpoint)
    if s.get('language')!=FORMAT:raise ValueError('not a foundation checkpoint')
    manifest,machines=corpus_info(args.data)
    if s['config']['corpus_hash']!=manifest['corpus_sha256'] or s['config']['language_hash']!=language_hash():raise ValueError('evaluation checkpoint/data mismatch')
    torch.set_num_threads(args.threads)
    p=Policy(**s['model']).to(args.device);p.load_state_dict(s['state_dict'])
    report=autonomous(p,machines['splits'][args.split],s['config']['budget'],args.eval_attempts,args.seed)
    if Path(args.out).exists():raise ValueError('evaluation output already exists')
    atomic(args.out,report);print(json.dumps(report,indent=2))
