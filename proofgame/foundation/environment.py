"""Finite foundational action menus and a flat, replayable proof archive."""
from dataclasses import dataclass
import copy
from .logic import *
from .kernel import Goal,expand,check,FORMAT
from .definitions import Theory

@dataclass(frozen=True)
class Hole:
    sort:str
    bound:tuple=()

def first_hole(x,path=()):
    if isinstance(x,Hole):return path,x
    if isinstance(x,tuple):
        for i,y in enumerate(x[1:],1):
            h=first_hole(y,path+(i,))
            if h:return h

def draft_text(x):
    if isinstance(x,Hole):return '?'+x.sort
    if isinstance(x,tuple):return '('+' '.join(draft_text(a) for a in x)+')'
    return str(x)

class Game:
    def __init__(self,machine,budget=2048,max_work=10000000):
        if budget<1 or max_work<1:raise ValueError('positive limits required')
        self.original=machine;self.machine=Theory(machine);self.budget=budget;self.left=budget
        self.max_work=max_work;self.work=0;self.answer=None;self.steps=[];self.goals=[]
        self.draft=None;self.definition=None;self.number=None;self.status='ACTIVE';self.error=None;self.reward=0
    def certificate(self):return {'format':FORMAT,'machine':self.original.to_json(),'answer':self.answer,'steps':copy.deepcopy(self.steps)}
    def current(self):return self.goals[0] if self.goals else None
    def expression_options(self):
        if not self.draft:return {}
        h=first_hole(self.draft['expr'])
        if not h:return {}
        _,hole=h;sort=hole.sort;bound=hole.bound;g=self.draft.get('goal',self.current());menu={}
        if sort=='Formula':
            menu['put:false']='false'
            for op in ('and','or','imp'):menu['put:'+op]=(op,Hole(sort,bound),Hole(sort,bound))
            for s in SORTS:
                menu['put:eq:'+s]=('eq',Hole(s,bound),Hole(s,bound))
                for op in ('all','ex'):menu['put:'+op+':'+s]=(op,s,Hole('Formula',bound+(s,)))
            for op,(args,result) in self.machine.signature.items():
                if result=='Formula':menu['put:'+op]=(op,*(Hole(s,bound) for s in args))
        else:
            for c,s in {**CONSTS,**{q:'State' for q in self.machine.states}}.items():
                if s==sort:menu['put:'+c]=c
            for op,(args,result) in {**SIG,**self.machine.signature}.items():
                if result==sort:menu['put:'+op]=(op,*(Hole(s,bound) for s in args))
            if self.definition and self.definition.get('stage')=='step':
                d=self.definition
                if d['result']==sort:menu['put:'+d['name']]=(d['name'],*(Hole(s,bound) for s in d['args']))
            for i,s in g.variables:
                if s==sort:menu['var:v'+str(i)]=('var',i)
            for i,s in enumerate(reversed(bound)):
                if s==sort:menu['bound:'+str(i)]=('bound',i)
        seen=set()
        for f in (g.target,*g.facts):
            for _,t in locations(f):
                if t in seen:continue
                seen.add(t)
                try:
                    if sort=='Formula':wf(t,self.machine,g.variables,bound)
                    elif type_of(t,self.machine,g.variables,bound)!=sort:continue
                except Invalid:continue
                menu['copy:'+str(len(menu))]=t
        return menu
    def rule_menu(self):
        g=self.current()
        if not g:return {}
        t=g.target;tag=t[0] if isinstance(t,tuple) else t
        menu={'rule:cut':('cut',None,'Formula'),'rule:raa':('raa',None,None)}
        for rule,sort in [('casesBit','Bit'),('casesState','State'),('casesCfg','Cfg'),('casesNat','Nat'),('casesList','List')]:menu['rule:'+rule]=(rule,None,sort)
        for r in {'all':['allI'],'ex':['exI'],'imp':['impI'],'and':['andI'],'or':['orL','orR'],'eq':['refl']}.get(tag,[]):
            menu['rule:'+r]=(r,None,t[1] if r=='exI' else None)
        if tag=='eq':
            menu['rule:symm']=('symm',None,None);menu['rule:congr']=('congr',None,None)
            menu['rule:trans']=('trans',None,type_of(t[1],self.machine,g.variables))
        if tag=='all' and t[1] in ('Nat','List'):
            r='natI' if t[1]=='Nat' else 'listI';menu['rule:'+r]=(r,None,None)
        for i,f in enumerate(g.facts):
            menu[f'rule:exact:h{i}']=('exact',i,None)
            if f=='false':menu[f'rule:falseE:h{i}']=('falseE',i,None)
            if isinstance(f,tuple):
                if f[0]=='imp':menu[f'rule:apply:h{i}']=('apply',i,None)
                if f[0] in ('and','or','ex'):menu[f'rule:destruct:h{i}']=('destruct',i,None)
                if f[0]=='all':menu[f'rule:specialize:h{i}']=('specialize',i,f[1])
                if f[0]=='eq':
                    menu[f'rule:inject:h{i}']=('inject',i,None)
                    if t=='false':menu[f'rule:discriminate:h{i}']=('discriminate',i,None)
        for where,f in [(-1,t),*enumerate(g.facts)]:
            for path,x in locations(f):
                loc=str(where)+':'+'.'.join(map(str,path))
                try:unfold(x,self.machine)
                except (Invalid,TypeError):pass
                else:menu['unfold:'+loc]=('unfold',(where,path),None)
                # No hidden conversion; offer only literal equality occurrences.
                cursor=f;under=False
                for k in path:
                    if isinstance(cursor,tuple) and cursor[0] in ('all','ex'):under=True
                    cursor=cursor[k]
                if under:continue
                for h,eq in enumerate(g.facts):
                    if isinstance(eq,tuple) and eq[0]=='eq':
                        for direction,a,b in [('forward',eq[1],eq[2]),('backward',eq[2],eq[1])]:
                            if x==a and a!=b:menu[f'rewrite:{h}:{direction}:'+loc]=('rewrite',(h,direction,where,path),None)
        return menu
    def available_actions(self):
        if self.status!='ACTIVE':return []
        if self.answer is None:return ['choose:HALT','choose:NONHALT']
        if self.number is not None:
            return ([f'digit:{i}' for i in range(10)] if len(self.number['digits'])<128 else [])+(['number:done'] if self.number['digits'] else [])+['cancel','reset']
        if self.draft:
            h=first_hole(self.draft['expr'])
            return list(self.expression_options())+(['put:num'] if h and h[1].sort=='Nat' else [])+(['commit'] if not h else [])+['cancel','reset']
        if self.definition:
            d=self.definition
            if d['stage']=='signature':
                return ([f'def:param:{s}' for s in SORTS] if len(d['args'])<4 else [])+[f'def:result:{s}' for s in (*SORTS,'Formula')]+['cancel','reset']
            return ['def:plain']+[f'def:recursive:{i}' for i,s in enumerate(d['args']) if s in ('Nat','List') and d['result']!='Formula']+['cancel','reset']
        return (list(self.rule_menu())+['define','reset']) if self.goals else ['submit','reset']
    def _commit(self,rule,data):
        cost=bounded(data)+bounded((self.current().target,self.current().facts));self.work+=cost
        if self.work>self.max_work:raise Limit('game work limit')
        children=expand(self.machine,self.current(),rule,data)
        self.goals=children+self.goals[1:];self.steps.append({'rule':rule,'data':data});self.draft=None
    def _defbody(self,stage):
        d=self.definition;d['stage']=stage;vs=tuple(enumerate(d['args']))
        if stage=='step':
            extra=('Nat',) if d['args'][d['recursive']]=='Nat' else ('Bit','List')
            vs+=tuple((len(vs)+i,s) for i,s in enumerate(extra))
        self.draft={'rule':'define','expr':Hole(d['result']),'goal':Goal('false',vs,())}
    def step(self,action,observe=True):
        if self.status!='ACTIVE':raise Invalid('terminal game')
        self.left-=1;self.error=None;self.reward=0
        try:
            if action not in self.available_actions():raise Invalid('unavailable action')
            if action=='reset':
                self.machine=Theory(self.original);self.steps=[];self.goals=[Goal(root_goal(self.answer))];self.draft=None;self.definition=None;self.number=None
            elif action=='cancel':self.draft=None;self.definition=None;self.number=None
            elif action.startswith('choose:'):
                self.answer=action.split(':')[1];self.goals=[Goal(root_goal(self.answer))]
            elif action=='submit':
                check(self.certificate(),max_steps=self.budget,max_work=self.max_work);self.status='SOLVED';self.reward=1
            elif action=='define':self.definition={'name':'f'+str(len(self.machine.definitions)),'args':[],'stage':'signature'}
            elif action.startswith('def:param:'):self.definition['args'].append(action.split(':')[2])
            elif action.startswith('def:result:'):self.definition.update(result=action.split(':')[2],stage='recursion')
            elif action=='def:plain' or action.startswith('def:recursive:'):
                self.definition['recursive']=None if action=='def:plain' else int(action.split(':')[2]);self._defbody('base')
            elif action=='put:num':self.number={'path':first_hole(self.draft['expr'])[0],'digits':''}
            elif action.startswith('digit:'):self.number['digits']+=action[-1]
            elif action=='number:done':
                self.draft['expr']=put_path(self.draft['expr'],self.number['path'],numeral(int(self.number['digits'])));self.number=None
            elif action=='commit':
                d=self.draft;value=d['expr']
                if d['rule']=='define':
                    spec=self.definition;spec[spec['stage']]=value
                    if spec['stage']=='base' and spec['recursive'] is not None:self._defbody('step')
                    else:
                        data={k:v for k,v in spec.items() if k!='stage'};data.setdefault('step',None)
                        self.work+=bounded(data)
                        if self.work>self.max_work:raise Limit('game work limit')
                        self.machine.define(data);self.steps.append({'rule':'define','data':data});self.definition=None;self.draft=None
                else:self._commit(d['rule'],(d['data'],value) if d['rule']=='specialize' else value)
            elif self.draft:
                path,_=first_hole(self.draft['expr']);self.draft['expr']=put_path(self.draft['expr'],path,self.expression_options()[action])
            else:
                rule,data,sort=self.rule_menu()[action]
                if sort:self.draft={'rule':rule,'data':data,'expr':Hole(sort)}
                else:self._commit(rule,data)
        except (Invalid,RecursionError,ValueError) as e:self.error=str(e)
        if self.work>self.max_work:self.status='LIMIT'
        if self.left<=0 and self.status=='ACTIVE':self.status='EXHAUSTED'
        return (self.observation() if observe else None),self.reward,self.status!='ACTIVE'
    def observation(self):
        obs={'language':FORMAT,'machine':self.original.to_json(),'answer':self.answer,'status':self.status,'actions_left':self.left,
             'work_left':self.max_work-self.work,'goals':[g.text() for g in self.goals],
             'definitions':self.machine.definitions,'last_error':self.error,'definition_draft':self.definition,
             'draft_context':self.draft.get('goal',self.current()).text() if self.draft else None,'draft':draft_text(self.draft['expr']) if self.draft else None,'number_draft':self.number}
        opts=self.expression_options() if self.draft else {}
        obs['actions']=[{'id':a,**({'expression':show(opts[a])} if a.startswith('copy:') else {})} for a in self.available_actions()]
        return obs
