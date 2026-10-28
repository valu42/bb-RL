"""Checked, total first-order definitions; no executable user code."""
from .logic import *

class Theory:
    def __init__(self,m):
        self.rows=m.rows;self.states=m.states;self.row=m.row;self.to_json=m.to_json
        self.signature={};self.definitions={}

    def define(self,data):
        bounded(data,max_nodes=10000)
        if not isinstance(data,dict) or set(data)!={'name','args','result','recursive','base','step'}:raise Invalid('bad definition')
        name=data['name'];args=tuple(data['args']);result=data['result'];rec=data['recursive']
        if name!='f'+str(len(self.definitions)) or len(args)>4 or any(a not in SORTS for a in args) or result not in (*SORTS,'Formula'):raise Invalid('bad definition signature')
        if rec is not None and result=='Formula':raise Invalid('recursive predicates are not supported')
        if rec is not None and (type(rec)!=int or rec<0 or rec>=len(args) or args[rec] not in ('Nat','List')):raise Invalid('bad recursion argument')
        self.signature[name]=(args,result)
        try:
            vs=tuple(enumerate(args));base=freeze(data['base']);step=freeze(data['step'])
            if result=='Formula':wf(base,self,vs)
            elif type_of(base,self,vs)!=result:raise Invalid('base sort mismatch')
            if any(isinstance(t,tuple) and t[0]==name for _,t in locations(base)):raise Invalid('recursive base')
            if rec is None:
                if step is not None:raise Invalid('nonrecursive definition has step')
            else:
                extra=('Nat',) if args[rec]=='Nat' else ('Bit','List')
                vs2=vs+tuple((len(args)+i,s) for i,s in enumerate(extra))
                if type_of(step,self,vs2)!=result:raise Invalid('step sort mismatch')
                pred=('var',len(args)+len(extra)-1)
                for _,t in locations(step):
                    if isinstance(t,tuple) and t[0]==name and t[rec+1]!=pred:raise Invalid('recursion must use immediate predecessor')
            self.definitions[name]={**data,'args':args,'base':base,'step':step}
        except Exception:
            del self.signature[name];raise

    def unfold_definition(self,name,args):
        d=self.definitions[name];body=d['base'];env=dict(enumerate(args));rec=d['recursive']
        if rec is not None:
            t=args[rec]
            if d['args'][rec]=='Nat':
                v=nat_view(t)
                if v is None:raise Invalid('unknown recursive constructor')
                if v[0]:body=d['step'];env[len(args)]=v[1]
            elif t=='nil':pass
            elif isinstance(t,tuple) and t[0]=='cons':
                body=d['step'];env[len(args)]=t[1];env[len(args)+1]=t[2]
            else:raise Invalid('unknown recursive constructor')
        def sub(x,depth=0):
            if not isinstance(x,tuple):return x
            if x[0]=='var':return shift(env[x[1]],depth)
            if x[0]=='bound':return x
            if x[0] in ('all','ex'):return (x[0],x[1],sub(x[2],depth+1))
            if x[0]=='num':return x
            return (x[0],*(sub(a,depth) for a in x[1:]))
        return sub(body)
