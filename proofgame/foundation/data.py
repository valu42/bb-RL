"""Offline random machines, family-disjoint splits, and checked demonstration jobs."""
import concurrent.futures
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import time
from .logic import Machine, Invalid, Limit

VERSION='foundation-pipeline-1'

def language_hash():
    root=Path(__file__).parent
    return hashlib.sha256(b''.join((root/name).read_bytes() for name in ('logic.py','kernel.py','definitions.py','environment.py'))).hexdigest()

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(path,x):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(path)

def family(m):
    reached={'q0'}
    while True:
        new=reached|{m.row(q,b)[0] for q in reached for b in ('b0','b1') if m.row(q,b)[0]!='qH'}
        if new==reached:break
        reached=new
    others=sorted(reached-{'q0'});variants=[]
    for order in itertools.permutations(others):
        names=('q0',)+order;rename={q:'q'+str(i) for i,q in enumerate(names)};rename['qH']='qH'
        for flip in (False,True):
            rows=[]
            for q in names:
                for bit in ('b0','b1'):
                    q2,w,d=m.row(q,bit);rows.append((rename[q2],w,('R' if d=='L' else 'L') if flip else d))
            variants.append(rows)
    return digest(min(variants))

def partition(fid,seed):
    x=int(hashlib.sha256((str(seed)+':'+fid).encode()).hexdigest()[:16],16)%10
    return 'train' if x<8 else 'dev' if x==8 else 'test'

def random_machine(rng,n):
    qs=['q'+str(i) for i in range(n)]+['qH']
    return Machine(tuple((rng.choice(qs),rng.choice(('b0','b1')),rng.choice(('L','R'))) for _ in range(n*2)))

def generate_manifest(seed,per_size,train_count,dev_count,test_count,states=(2,3,4,5)):
    from ..special_benchmarks import benchmark_definitions
    excluded={family(Machine(x['rows'])) for x in benchmark_definitions('extended')}
    # BB(5) time champion, also excluded up to the structural family fingerprint.
    bb5=Machine((('q1','b1','R'),('q2','b1','L'),('q2','b1','R'),('q1','b1','R'),('q3','b1','R'),('q4','b0','L'),('q0','b1','L'),('q3','b1','L'),('qH','b1','R'),('q0','b0','L')))
    excluded.add(family(bb5));rng=random.Random(seed);jobs=[];splits={x:[] for x in ('train','dev','test')}
    totals={'train':train_count,'dev':dev_count,'test':test_count}
    for idx,n in enumerate(states):
        print(json.dumps({'event':'sampling','states':n,'requested_teacher_machines':per_size}),flush=True)
        needed={k:v//len(states)+int(idx<v%len(states)) for k,v in totals.items()}
        got={k:0 for k in needed};seen=set();proposals=0
        # Bound search, and acknowledge exhaustion of the small two-state population.
        max_proposals=max(10000,per_size*50,sum(needed.values())*200)
        for _ in range(max_proposals):
            if proposals>=per_size and all(got[k]>=needed[k] for k in needed):break
            m=random_machine(rng,n);mid=digest(m.to_json())
            if mid in seen:continue
            seen.add(mid);fid=family(m)
            if fid in excluded:continue
            split=partition(fid,seed);row={'machine_id':mid,'family_id':fid,'states':n,'machine':m.to_json()}
            if got[split]<needed[split]:splits[split].append(row);got[split]+=1
            if split=='train' and proposals<per_size:jobs.append(row);proposals+=1
        if any(got[k]<needed[k] for k in needed):raise ValueError('insufficient split population; reduce requested counts')
    splits['special']=[{'machine_id':digest(Machine(x['rows']).to_json()),'family_id':family(Machine(x['rows'])),'states':len(x['rows'])//2,'machine':Machine(x['rows']).to_json(),'name':x.get('name',x.get('id','special'))} for x in benchmark_definitions('extended')]
    splits['special'].append({'machine_id':digest(bb5.to_json()),'family_id':family(bb5),'states':5,'machine':bb5.to_json(),'name':'BB(5) champion'})
    return {'version':VERSION,'seed':seed,'states':list(states),'excluded_families':sorted(excluded),'jobs':jobs,'splits':splits}

def teacher_job(task):
    row,budget,simulation,max_steps=task
    from .compiler import teach,replay
    start=time.monotonic();out={**row,'examples':[],'errors':[]}
    try:
        for source,cert in teach(Machine(row['machine']['rows']),simulation,max_steps):
            try:
                actions=replay(cert,budget)
                out['examples'].append({'source':source,'answer':cert['answer'],'certificate':cert,'actions':actions,'proof_id':digest(cert)})
            except (Invalid,RecursionError,ValueError) as e:out['errors'].append(str(e))
    except (Invalid,RecursionError,ValueError) as e:out['errors'].append(str(e))
    out['seconds']=time.monotonic()-start;return out

def generate(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg={k:getattr(args,k) for k in ('seed','per_size','train','dev','test','budget','simulation','compiler_steps')};cfg['version']=VERSION;cfg['language_hash']=language_hash();cfg['compiler_hash']=hash_file(Path(__file__).with_name('compiler.py'))
    if args.resume:
        old=json.loads((out/'config.json').read_text())
        if old!=cfg:raise ValueError('generation resume configuration mismatch')
        manifest=json.loads((out/'machines.json').read_text());state=json.loads((out/'progress.json').read_text())
    else:
        if any(out.iterdir()):raise ValueError('output must be empty, or use --resume')
        manifest=generate_manifest(args.seed,args.per_size,args.train,args.dev,args.test)
        atomic(out/'config.json',cfg);atomic(out/'machines.json',manifest)
        state={'processed':0,'bytes':0,'proofs':0,'actions':0,'errors':0,'coverage':{}};atomic(out/'progress.json',state)
    path=out/'corpus.jsonl'
    with path.open('a+b') as f:
        f.truncate(state['bytes'])
    jobs=manifest['jobs'][state['processed']:]
    def tasks():
        for row in jobs:
            if args.simulation>0:limit=args.simulation
            else:
                r=random.Random(int(row['machine_id'][:16],16)).random()
                limit=128 if r<.7 else 1024 if r<.9 else 10000 if r<.99 else 100000
            yield row,args.budget,limit,args.compiler_steps
    start=time.monotonic();initial=state['processed']
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool, path.open('ab') as f:
        # Bounded submission avoids retaining an entire large corpus in futures.
        iterator=iter(tasks());pending=[]
        while True:
            while len(pending)<args.workers*2:
                try:pending.append(pool.submit(teacher_job,next(iterator)))
                except StopIteration:break
            if not pending:break
            row=pending.pop(0).result();payload=(json.dumps(row,separators=(',',':'))+'\n').encode()
            f.write(payload);f.flush();os.fsync(f.fileno())
            state['processed']+=1;state['bytes']=f.tell();state['proofs']+=len(row['examples'])
            state['actions']+=sum(len(e['actions']) for e in row['examples']);state['errors']+=len(row['errors'])
            for e in row['examples']:
                k=str(row['states'])+':'+e['answer']+':'+e['source'];state['coverage'][k]=state['coverage'].get(k,0)+1
            atomic(out/'progress.json',state)
            if state['processed']%10==0 or not pending:
                elapsed=time.monotonic()-start;rate=(state['processed']-initial)/max(elapsed,1e-6)
                print(json.dumps({'event':'generation',**state,'total':len(manifest['jobs']),'machines_per_second':rate,'eta_seconds':(len(manifest['jobs'])-state['processed'])/max(rate,1e-6)}),flush=True)
    atomic(out/'manifest.json',{'version':VERSION,'config':cfg,'source_hash':digest(manifest),'state':state,'corpus_sha256':hash_file(path)})

def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

class Corpus:
    """Byte offsets and balanced strata; proof bodies are loaded on demand."""
    def __init__(self,path):
        self.path=Path(path);self.strata={};self.count=0;entries=[]
        with self.path.open('rb') as f:
            while True:
                offset=f.tell();line=f.readline()
                if not line:break
                row=json.loads(line)
                for i,e in enumerate(row['examples']):
                    bucket=self.strata.setdefault((row['states'],e['answer'],e['source']),{})
                    bucket.setdefault(row['machine_id'],[]).append((offset,i));self.count+=1
                    entries.append((len(e['actions']),(row['states'],e['answer'],e['source']),row['machine_id'],(offset,i)))
        if not self.count:raise ValueError('no replayable proofs in corpus')
        lengths=sorted(e[0] for e in entries);self.stages=[]
        for fraction in (.25,.75,1.):
            threshold=lengths[min(len(lengths)-1,int(fraction*len(lengths)))];buckets={}
            for length,key,mid,position in entries:
                if length<=threshold:buckets.setdefault(key,{}).setdefault(mid,[]).append(position)
            self.stages.append(buckets)
    def sample(self,rng,stage=2):
        strata=self.stages[stage]
        n=rng.choice(sorted({k[0] for k in strata}));answer=rng.choice(sorted({k[1] for k in strata if k[0]==n}))
        keys=[k for k in strata if k[:2]==(n,answer)];bucket=strata[rng.choice(keys)]
        offset,i=rng.choice(bucket[rng.choice(list(bucket))])
        with self.path.open('rb') as f:f.seek(offset);row=json.loads(f.readline())
        return row,row['examples'][i]
