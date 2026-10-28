"""Small natural-deduction kernel. Certificates are rechecked from an empty context.

No learned component is trusted. expand() is shared by the editor and recursive
checker, which reconstructs all contexts and never trusts saved success flags.
"""
from dataclasses import dataclass
import json
from .logic import (Invalid, Limit, Machine, SORTS, freeze, wf, type_of,
                    inst, replace, disjoint, root_goal, size, show, bounded, unfold, at_path, put_path)


@dataclass(frozen=True)
class Goal:
    target: object
    variables: tuple = ()
    facts: tuple = ()

    def fresh(self, sort):
        i = max((i for i, _ in self.variables), default=-1) + 1
        return ("var", i), self.variables + ((i, sort),)

    def text(self):
        return {"variables": {f"v{i}":s for i,s in self.variables},
                "facts": {f"h{i}":show(f) for i,f in enumerate(self.facts)},
                "goal": show(self.target)}


def validate_goal(g, m):
    if len(dict(g.variables)) != len(g.variables): raise Invalid("duplicate variable ID")
    if any(type(i) is not int or i < 0 or s not in SORTS for i,s in g.variables):
        raise Invalid("bad variable context")
    wf(g.target, m, g.variables)
    for f in g.facts: wf(f, m, g.variables)


def expand(m, g, rule, data=None):
    validate_goal(g, m)
    data = freeze(data)
    target, variables, facts = g.target, g.variables, g.facts
    def goal(t=target, vs=variables, fs=facts): return Goal(t, vs, fs)
    def shape(x, tag, arity=None):
        if not isinstance(x, tuple) or x[0] != tag or (arity and len(x) != arity):
            raise Invalid(f"expected {tag}")
        return x
    def index(i):
        if type(i) is not int or i < 0 or i >= len(facts): raise Invalid("unknown hypothesis")
        return facts[i]
    def term(t, sort):
        if type_of(t, m, variables) != sort: raise Invalid("witness has wrong sort")
    def same(a,b): return a == b
    def no_data():
        if data is not None: raise Invalid("rule takes no annotation")
    if rule == "exact":
        if not same(index(data), target): raise Invalid("hypothesis does not match goal")
        children = []
    elif rule == "refl":
        no_data(); shape(target,"eq")
        if not same(target[1],target[2]): raise Invalid("sides do not reduce to the same term")
        children = []
    elif rule == "symm":
        no_data();shape(target,"eq");children=[goal(("eq",target[2],target[1]))]
    elif rule == "trans":
        shape(target,"eq");term(data,type_of(target[1],m,variables))
        children=[goal(("eq",target[1],data)),goal(("eq",data,target[2]))]
    elif rule == "congr":
        no_data();shape(target,"eq");a,b=target[1:]
        if not isinstance(a,tuple) or not isinstance(b,tuple) or a[0]!=b[0] or a[0] in ("var","bound","num") or len(a)!=len(b):raise Invalid("matching function applications required")
        children=[goal(("eq",x,y)) for x,y in zip(a[1:],b[1:])]
    elif rule == "casesNat":
        term(data,"Nat");v,vs=g.fresh("Nat")
        children=[goal(fs=facts+(("eq",data,"z"),)),goal(vs=vs,fs=facts+(("eq",data,("s",v)),))]
    elif rule == "casesList":
        term(data,"List");b,vs=g.fresh("Bit");t,vs=Goal(target,vs,facts).fresh("List")
        children=[goal(fs=facts+(("eq",data,"nil"),)),goal(vs=vs,fs=facts+(("eq",data,("cons",b,t)),))]
    elif rule == "falseE":
        if index(data) != "false": raise Invalid("hypothesis is not false")
        children = []
    elif rule == "discriminate":
        if target != "false": raise Invalid("discriminate closes false only")
        f = shape(index(data),"eq")
        if not disjoint(f[1],f[2],m): raise Invalid("no constructor contradiction")
        children = []
    elif rule == "inject":
        f=shape(index(data),"eq"); a,b=f[1],f[2]
        if not isinstance(a,tuple) or not isinstance(b,tuple) or a[0]!=b[0] or a[0] not in ("s","cons","cfg"):
            raise Invalid("inject needs matching injective constructors")
        equations=tuple(("eq",x,y) for x,y in zip(a[1:],b[1:]))
        children=[goal(fs=facts+equations)]
    elif rule == "allI":
        no_data(); shape(target,"all")
        v, vs = g.fresh(target[1]); children = [goal(inst(target[2],v),vs)]
    elif rule == "exI":
        shape(target,"ex"); term(data,target[1]); children = [goal(inst(target[2],data))]
    elif rule == "impI":
        no_data(); shape(target,"imp"); children = [goal(target[2],fs=facts+(target[1],))]
    elif rule == "andI":
        no_data(); shape(target,"and"); children = [goal(target[1]),goal(target[2])]
    elif rule in ("orL","orR"):
        no_data(); shape(target,"or"); children = [goal(target[1 if rule=="orL" else 2])]
    elif rule == "apply":
        f=shape(index(data),"imp")
        if not same(f[2],target): raise Invalid("implication conclusion does not match")
        children=[goal(f[1])]
    elif rule == "destruct":
        f=index(data)
        if isinstance(f,tuple) and f[0]=="and": children=[goal(fs=facts+(f[1],f[2]))]
        elif isinstance(f,tuple) and f[0]=="or":
            children=[goal(fs=facts+(f[1],)),goal(fs=facts+(f[2],))]
        elif isinstance(f,tuple) and f[0]=="ex":
            v,vs=g.fresh(f[1]); children=[goal(vs=vs,fs=facts+(inst(f[2],v),))]
        else: raise Invalid("destruct needs and, or, or ex")
    elif rule == "specialize":
        if not isinstance(data,tuple) or len(data)!=2: raise Invalid("bad specialization")
        f=shape(index(data[0]),"all"); term(data[1],f[1])
        children=[goal(fs=facts+(inst(f[2],data[1]),))]
    elif rule == "cut":
        wf(data,m,variables); children=[goal(data),goal(fs=facts+(data,))]
    elif rule == "rewrite":
        if not isinstance(data,tuple) or len(data)!=4: raise Invalid("bad rewrite annotation")
        h,direction,where,path=data
        if direction not in ("forward","backward"): raise Invalid("bad direction")
        f=shape(index(h),"eq"); old,new=f[1:]
        if direction=="backward":old,new=new,old
        original=target if where==-1 else index(where)
        # Never rewrite under a binder through raw paths; use specialization first.
        cursor=original
        for i in path:
            if isinstance(cursor,tuple) and cursor[0] in ("all","ex"): raise Invalid("specialize before rewriting under binder")
            cursor=at_path(cursor,(i,))
        if at_path(original,path)!=old: raise Invalid("no matching occurrence")
        changed=put_path(original,path,new)
        if where==-1:children=[goal(changed)]
        else:children=[goal(fs=facts[:where]+(changed,)+facts[where+1:])]
    elif rule == "unfold":
        if not isinstance(data,tuple) or len(data)!=2: raise Invalid("bad occurrence")
        where,path=data
        original=target if where==-1 else index(where)
        changed=put_path(original,path,unfold(at_path(original,path),m))
        if where==-1: children=[goal(changed)]
        else: children=[goal(fs=facts[:where]+(changed,)+facts[where+1:])]
    elif rule == "natI":
        no_data(); shape(target,"all")
        if target[1]!="Nat": raise Invalid("natI needs a Nat quantifier")
        v,vs=g.fresh("Nat"); body=target[2]
        children=[goal(inst(body,"z")),goal(inst(body,("s",v)),vs,facts+(inst(body,v),))]
    elif rule == "listI":
        no_data(); shape(target,"all")
        if target[1]!="List": raise Invalid("listI needs a List quantifier")
        bit,vs=g.fresh("Bit"); tail,vs=Goal(target,vs,facts).fresh("List")
        body=target[2]
        children=[goal(inst(body,"nil")),goal(inst(body,("cons",bit,tail)),vs,facts+(inst(body,tail),))]
    elif rule == "casesCfg":
        term(data,"Cfg")
        q,vs=g.fresh("State"); l,vs=Goal(target,vs,facts).fresh("List"); r,vs=Goal(target,vs,facts).fresh("List")
        children=[goal(vs=vs,fs=facts+(("eq",data,("cfg",q,l,r)),))]
    elif rule in ("casesBit","casesState"):
        sort="Bit" if rule=="casesBit" else "State"; term(data,sort)
        values=("b0","b1") if sort=="Bit" else m.states
        children=[goal(fs=facts+(("eq",data,value),)) for value in values]
    elif rule == "raa":
        no_data(); children=[goal("false",fs=facts+(("imp",target,"false"),))]
    else: raise Invalid("unknown proof rule")
    for child in children:
        bounded((child.target,child.facts)); validate_goal(child,m)
    return children



FORMAT="proofgame-foundation-2"

def check(certificate, max_steps=100000, max_work=10000000):
    if not isinstance(certificate,dict) or set(certificate)!={"format","machine","answer","steps"}: raise Invalid("bad certificate envelope")
    if certificate["format"]!=FORMAT: raise Invalid("wrong language version")
    from .definitions import Theory
    m=Theory(Machine(certificate["machine"]["rows"]))
    goals=[Goal(root_goal(certificate["answer"]))]; work=0
    steps=certificate["steps"]
    if not isinstance(steps,list) or len(steps)>max_steps:raise Limit("certificate step limit")
    for step in steps:
        work+=bounded(step)
        if not goals:raise Invalid("steps after completed proof")
        if not isinstance(step,dict) or set(step)!={"rule","data"}:raise Invalid("bad proof step")
        if step["rule"]=="define":
            m.define(step["data"])
        else:
            g=goals.pop(0)
            work+=bounded((g.target,g.facts))
            goals[:0]=expand(m,g,step["rule"],step["data"])
        if work>max_work:raise Limit("checker work limit")
    if goals:raise Invalid("incomplete proof")
    return certificate["answer"]
