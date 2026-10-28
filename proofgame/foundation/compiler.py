"""OFFLINE only: certificate teachers and proof compilers. Not imported by RL."""
from .logic import *
from .kernel import *
from .definitions import Theory
from .environment import Game,first_hole

class Builder:
    def __init__(self,m,answer,max_steps=20000):
        self.m=Theory(m);self.original=m;self.answer=answer;self.steps=[];self.goals=[Goal(root_goal(answer))];self.max_steps=max_steps
    @property
    def g(self):return self.goals[0]
    def emit(self,rule,data=None):
        if len(self.steps)>=self.max_steps:raise Limit('compiler step cap')
        children=expand(self.m,self.g,rule,data);self.goals=children+self.goals[1:]
        self.steps.append({'rule':rule,'data':data})
    def norm(self,where=-1):
        for _ in range(self.max_steps):
            t=self.g.target if where==-1 else self.g.facts[where]
            for path,x in reversed(list(locations(t))):
                try:unfold(x,self.m)
                except (Invalid,TypeError):continue
                self.emit('unfold',(where,path));break
            else:return
        raise Limit('normalization cap')
    def rewrite_all(self,h,where=-1):
        # Expansion produces individually checked occurrence rewrites, not a bulk rule.
        old=self.g.facts[h][1]
        for _ in range(self.max_steps):
            t=self.g.target if where==-1 else self.g.facts[where]
            found=next((p for p,x in locations(t) if x==old),None)
            if found is None:return
            self.emit('rewrite',(h,'forward',where,found))
        raise Limit('rewrite cap')
    def finish(self):
        if self.goals:raise Invalid('compiler left goals')
        c={'format':FORMAT,'machine':self.original.to_json(),'answer':self.answer,'steps':self.steps}
        check(c);return c

def disjunction(xs):
    if len(xs)==1:return xs[0]
    return ('or',xs[0],disjunction(xs[1:]))

def true_branch(t):
    if t[0]=='eq':return t[1]==t[2]
    return t[0]=='or' and (true_branch(t[1]) or true_branch(t[2]))

def close_disjunction(b):
    t=b.g.target
    if t[0]=='eq':b.emit('refl');return
    if true_branch(t[1]):b.emit('orL')
    else:b.emit('orR')
    close_disjunction(b)

def split_fact(b,h,fn):
    f=b.g.facts[h]
    if f[0]!='or':fn(h);return
    idx=len(b.g.facts);b.emit('destruct',h)
    split_fact(b,idx,fn);split_fact(b,idx,fn)

def conclude_nonhalt(b):
    b.emit('allI');n=('var',b.g.variables[-1][0]);b.emit('impI');halt=len(b.g.facts)-1
    b.emit('specialize',(0,n));membership=len(b.g.facts)-1
    def contradiction(h):
        b.rewrite_all(h,halt);b.emit('discriminate',halt)
    split_fact(b,membership,contradiction)

def compile_halt(m,n,max_steps=20000):
    b=Builder(m,'HALT',max_steps);b.emit('exI',numeral(n));b.norm();b.emit('refl');return b.finish()

def compile_closed(m,states,max_steps=20000):
    # The theorem is checked from scratch; teacher success is not a premise.
    qs=tuple(states);p=lambda x:disjunction([('eq',('qof',x),q) for q in qs])
    lemma=('all','Nat',p(('at',('bound',0),C0)))
    b=Builder(m,'NONHALT',max_steps);b.emit('cut',lemma);b.emit('natI');b.norm();close_disjunction(b)
    def branch(h):
        b.norm();b.rewrite_all(h);b.norm()
        unknown=next(x[2] for _,x in locations(b.g.target) if isinstance(x,tuple) and x[0]=='qnext')
        idx=len(b.g.facts);b.emit('casesBit',unknown)
        for _ in range(2):b.rewrite_all(idx);b.norm();close_disjunction(b)
    split_fact(b,0,branch);conclude_nonhalt(b);return b.finish()

def concrete(m,limit):
    """Independent sparse simulator: states, absolute head, nonblank cells."""
    q='q0';head=0;tape=set();seen={};configs=[]
    def stack(cells):
        out='nil'
        for bit in reversed(cells):out=('cons',bit,out)
        return out
    archive_cells=0;tracking=True
    for n in range(limit+1):
        if q=='qH':return 'HALT',n,configs
        if tracking:
            low=min(tape|{head});high=max(tape|{head})
            cost=high-low+1+len(tape)
            if n>4096 or archive_cells+cost>100000:
                tracking=False;seen.clear();configs.clear()
            else:
                l=['b1' if i in tape else 'b0' for i in range(head-1,low-1,-1)]
                r=['b1' if i in tape else 'b0' for i in range(head,high+1)]
                while l and l[-1]=='b0':l.pop()
                while r and r[-1]=='b0':r.pop()
                configs.append(('cfg',q,stack(l),stack(r)))
                identity=(q,head,frozenset(tape))
                if identity in seen:return 'CYCLE',(seen[identity],n-seen[identity]),configs
                seen[identity]=n;archive_cells+=cost
        if n==limit:break
        q,w,d=m.row(q,'b1' if head in tape else 'b0')
        if w=='b1':tape.add(head)
        else:tape.discard(head)
        head+=1 if d=='R' else -1
    return 'UNKNOWN',None,configs

def compile_cycle(m,start,period,max_steps=20000):
    outcome,data,cs=concrete(m,start+period)
    if outcome!='CYCLE' or data!=(start,period):raise Invalid('not the first exact cycle')
    # Canonical pack equations make sparse simulation and literal lists agree.
    representatives=list(dict.fromkeys(cs[:-1]))
    p=lambda x:disjunction([('eq',x,c) for c in representatives])
    invariant=('all','Nat',p(('at',('bound',0),C0)))
    b=Builder(m,'NONHALT',max_steps);b.emit('cut',invariant);b.emit('natI');b.norm();close_disjunction(b)
    def branch(h):
        b.norm();b.rewrite_all(h);b.norm();close_disjunction(b)
    split_fact(b,0,branch)
    b.emit('allI');n=('var',b.g.variables[-1][0]);b.emit('impI');halt=len(b.g.facts)-1
    b.emit('specialize',(0,n));idx=len(b.g.facts)-1
    def contradiction(h):
        b.rewrite_all(h,halt);b.norm(halt);b.emit('discriminate',halt)
    split_fact(b,idx,contradiction);return b.finish()

def replay(certificate,budget=4096,max_work=10000000):
    check(certificate,max_work=max_work)
    g=Game(Machine(certificate['machine']['rows']),budget,max_work);actions=[]
    def act(a):
        g.step(a,observe=False);actions.append(a)
        if g.error:raise Invalid('replay '+a+': '+g.error)
        if g.status!='ACTIVE' and a!='submit':raise Limit('replay budget')
    def expression(t):
        # Prefer syntactic copies, never synthesize hidden proofs.
        opts=g.expression_options()
        for a,v in opts.items():
            if a.startswith('copy:') and v==t:act(a);return
        if isinstance(t,str):act('put:'+t);return
        if t[0]=='var':act('var:v'+str(t[1]));return
        if t[0]=='bound':act('bound:'+str(t[1]));return
        if t[0]=='num':
            act('put:num')
            for digit in t[1]:act('digit:'+digit)
            act('number:done');return
        op=t[0]
        if op in ('all','ex'):
            act('put:'+op+':'+t[1]);expression(t[2]);return
        if op=='eq':
            # Bound sorts belong to the active hole's binder environment.
            _,hole=first_hole(g.draft['expr'])
            s=type_of(t[1],g.machine,g.current().variables,hole.bound)
            act('put:eq:'+s)
        else:act('put:'+op)
        for a in t[1:]:expression(a)
    act('choose:'+certificate['answer'])
    for step in certificate['steps']:
        r=step['rule'];d=freeze(step['data'])
        for action,(rule,data,sort) in g.rule_menu().items():
            if rule!=r:continue
            if sort:
                if r=='specialize' and data!=d[0]:continue
                act(action);expression(d[1] if r=='specialize' else d);act('commit');break
            if freeze(data)==d:act(action);break
        else:raise Invalid('no replay action for '+r)
    act('submit')
    if g.certificate()!=certificate: # Tuple/list JSON distinction is immaterial.
        import json
        if json.dumps(g.certificate(),sort_keys=True)!=json.dumps(certificate,sort_keys=True):raise Invalid('replay mismatch')
    return actions

def teach(m,limit=128,max_steps=20000):
    from itertools import combinations
    others=m.states[1:-1]
    for k in range(len(others)+1):
        for extra in combinations(others,k):
            qs=('q0',)+extra
            if all(m.row(q,b)[0] in qs for q in qs for b in ('b0','b1')):
                yield 'closed',compile_closed(m,qs,max_steps);break
        else:continue
        break
    kind,data,_=concrete(m,limit)
    if kind=='HALT':yield 'halt',compile_halt(m,data,max_steps)
    elif kind=='CYCLE':yield 'cycle',compile_cycle(m,*data,max_steps=max_steps)
