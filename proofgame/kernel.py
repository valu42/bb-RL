"""Small natural-deduction kernel. Certificates are rechecked from an empty context.

No learned component is trusted. expand() is shared by the editor and recursive
checker, which reconstructs all contexts and never trusts saved success flags.
"""
from dataclasses import dataclass
import json
from .logic import (Invalid, Limit, Machine, SORTS, freeze, wf, type_of, normal,
                    inst, replace, disjoint, root_goal, size, show)


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
    def same(a,b): return normal(a,m) == normal(b,m)
    def no_data():
        if data is not None: raise Invalid("rule takes no annotation")
    if rule == "exact":
        if not same(index(data), target): raise Invalid("hypothesis does not match goal")
        children = []
    elif rule == "refl":
        no_data(); shape(target,"eq")
        if not same(target[1],target[2]): raise Invalid("sides do not reduce to the same term")
        children = []
    elif rule == "falseE":
        if index(data) != "false": raise Invalid("hypothesis is not false")
        children = []
    elif rule == "discriminate":
        if target != "false": raise Invalid("discriminate closes false only")
        f = shape(normal(index(data),m),"eq")
        if not disjoint(f[1],f[2],m): raise Invalid("no constructor contradiction")
        children = []
    elif rule == "inject":
        f=shape(index(data),"eq"); a,b=normal(f[1],m),normal(f[2],m)
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
        if not isinstance(data,tuple) or len(data)!=2 or data[1] not in ("forward","backward"):
            raise Invalid("bad rewrite annotation")
        f=shape(index(data[0]),"eq"); old,new=f[1:]
        if data[1]=="backward": old,new=new,old
        new_facts=tuple(f if i==data[0] else replace(h,old,new) for i,h in enumerate(facts))
        t=replace(target,old,new)
        if t==target and new_facts==facts: raise Invalid("no matching occurrence")
        children=[goal(t,fs=new_facts)]
    elif rule == "reduce":
        no_data(); t=normal(target,m); fs=tuple(normal(f,m) for f in facts)
        if t==target and fs==facts: raise Invalid("nothing reduces")
        children=[goal(t,fs=fs)]
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
    elif rule in ("casesBit","casesState"):
        sort="Bit" if rule=="casesBit" else "State"; term(data,sort)
        values=("b0","b1") if sort=="Bit" else m.states
        children=[goal(fs=facts+(("eq",data,value),)) for value in values]
    elif rule == "raa":
        no_data(); children=[goal("false",fs=facts+(("imp",target,"false"),))]
    else: raise Invalid("unknown proof rule")
    for child in children: validate_goal(child,m)
    return children


def check(certificate, max_nodes=10000):
    """Return accepted answer, or raise Invalid/Limit. Checks the claimed machine too.

    Callers comparing to an assigned task must compare machine and answer; the
    environment does so by constructing the envelope itself.
    """
    if not isinstance(certificate,dict) or set(certificate)!={"format","machine","answer","proof"}:
        raise Invalid("bad certificate envelope")
    if certificate["format"]!="proofgame-0.1": raise Invalid("unsupported certificate version")
    m=Machine(certificate["machine"]["rows"])
    root=Goal(root_goal(certificate["answer"]))
    remaining=[max_nodes]
    def visit(p,g,depth=0):
        remaining[0]-=1
        if remaining[0]<0 or depth>180: raise Limit("certificate resource limit")
        if not isinstance(p,dict) or set(p)!={"rule","data","children"}:
            raise Invalid("incomplete or malformed proof node")
        if not isinstance(p["children"],list): raise Invalid("children must be a list")
        if len(json.dumps(p["data"]))>100000: raise Limit("annotation too large")
        children=expand(m,g,p["rule"],p["data"])
        if len(children)!=len(p["children"]): raise Invalid("wrong number of subproofs")
        for q,child in zip(p["children"],children): visit(q,child,depth+1)
    try: visit(certificate["proof"],root)
    except (RecursionError, MemoryError) as e: raise Limit("checker resource limit") from e
    except (KeyError,TypeError,IndexError) as e: raise Invalid("malformed certificate") from e
    return certificate["answer"]
