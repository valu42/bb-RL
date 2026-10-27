"""Finite, state-dependent action menus. Reset never refunds the action budget."""
from dataclasses import dataclass
import copy
import json
from functools import lru_cache
from types import SimpleNamespace
from .logic import (SORTS, SIG, CONSTS, Invalid, Limit, type_of, wf, show, size,
                    subterms, root_goal)
from .kernel import Goal, expand, check


@dataclass(frozen=True)
class Hole:
    sort: str
    bound: tuple = ()


def first_hole(x, path=()):
    if isinstance(x,Hole): return path,x
    if isinstance(x,tuple):
        for i,y in enumerate(x[1:],1):
            h=first_hole(y,path+(i,))
            if h: return h
    return None


def put_at(x,path,value):
    if not path: return value
    y=list(x); y[path[0]]=put_at(y[path[0]],path[1:],value); return tuple(y)


def draft_text(x):
    if isinstance(x,Hole): return f"?{x.sort}"
    if isinstance(x,str): return x
    if isinstance(x,tuple) and x[0] in ("var","bound"): return show(x)
    return "("+" ".join(draft_text(a) for a in x)+")"


@lru_cache(maxsize=100000)
def expression_options(g,states,sort,bound):
    # Typing and expression choices depend on sorts and the derived goal,
    # never on transition-table entries or the partial expression's history.
    m=SimpleNamespace(states=states);menu={}
    if sort=="Formula":
        menu["put:false"]="false"
        for op in ("and","or","imp"):
            menu[f"put:{op}"]=(op,Hole("Formula",bound),Hole("Formula",bound))
        for s in SORTS:
            menu[f"put:eq:{s}"]=("eq",Hole(s,bound),Hole(s,bound))
            for op in ("all","ex"):
                menu[f"put:{op}:{s}"]=(op,s,Hole("Formula",bound+(s,)))
    else:
        for c,s in {**CONSTS,**{q:"State" for q in states}}.items():
            if s==sort:menu[f"put:{c}"]=c
        for op,(args,result) in SIG.items():
            if result==sort:menu[f"put:{op}"]=(op,*(Hole(s,bound) for s in args))
        for i,s in g.variables:
            if s==sort:menu[f"var:v{i}"]=("var",i)
        for i,s in enumerate(reversed(bound)):
            if s==sort:menu[f"bound:{i}"]=("bound",i)
    seen=set();copies=0
    for f in (g.target,*g.facts):
        for t in subterms(f):
            if t in seen:continue
            seen.add(t)
            try:
                if sort=="Formula":wf(t,m,g.variables,bound)
                elif type_of(t,m,g.variables,bound)!=sort:continue
            except Invalid:continue
            menu[f"copy:{copies}"]=t;copies+=1
    return tuple(menu.items())


class Game:
    def __init__(self,machine,budget=256,max_nodes=2048,max_depth=80):
        if budget<1 or max_nodes<1 or max_depth<1: raise ValueError("limits must be positive")
        self.machine=machine; self.budget=budget; self.left=budget
        self.max_nodes=max_nodes; self.max_depth=max_depth
        self.answer=None; self.proof=None; self.focus=(); self.draft=None
        self.candidate=None
        # These flags describe the *current* proof/candidate, not action history.
        # They are visible in observations and determine the legal action menu.
        self.submit_failed=False
        self.status="ACTIVE"; self.error=None; self.reward=0
        self._cached_goals=None

    def goals(self):
        if self.answer is None: return []
        if self._cached_goals is not None: return self._cached_goals
        out=[]
        def walk(p,g,path):
            if p is None: out.append((path,g)); return
            children=expand(self.machine,g,p["rule"],p["data"])
            for i,(q,c) in enumerate(zip(p["children"],children)):
                walk(q,c,path+(i,))
        walk(self.proof,Goal(root_goal(self.answer)),())
        self._cached_goals=out
        return out

    def current(self):
        for path,g in self.goals():
            if path==self.focus: return g
        return None

    def rule_menu(self):
        """Map action to (rule, initial annotation, optional expression sort)."""
        g=self.current()
        if not g: return {}
        target=g.target
        menu={"rule:cut":("cut",None,"Formula"),
              "rule:reduce":("reduce",None,None),
              "rule:raa":("raa",None,None),
              "rule:casesBit":("casesBit",None,"Bit"),
              "rule:casesState":("casesState",None,"State")}
        if (self.answer=="NONHALT" and self.proof is None and
                target==root_goal("NONHALT") and not g.variables and not g.facts):
            menu["rule:closedStates"]=("closedStates",None,None)
            menu["rule:exactCycle"]=("exactCycle",None,None)
        tag=target[0] if isinstance(target,tuple) else target
        rules={"all":["allI"],"ex":["exI"],"imp":["impI"],"and":["andI"],
               "or":["orL","orR"],"eq":["refl"]}.get(tag,[])
        if tag=="all" and target[1]=="Nat": rules=rules+["natI"]
        if tag=="all" and target[1]=="List": rules=rules+["listI"]
        for r in rules: menu[f"rule:{r}"]=(r,None,target[1] if r=="exI" else None)
        for i,f in enumerate(g.facts):
            menu[f"rule:exact:h{i}"]=("exact",i,None)
            if f=="false": menu[f"rule:falseE:h{i}"]=("falseE",i,None)
            if isinstance(f,tuple):
                if f[0]=="imp": menu[f"rule:apply:h{i}"]=("apply",i,None)
                if f[0] in ("and","or","ex"): menu[f"rule:destruct:h{i}"]=("destruct",i,None)
                if f[0]=="all": menu[f"rule:specialize:h{i}"]=("specialize",i,f[1])
                if f[0]=="eq":
                    menu[f"rule:inject:h{i}"]=("inject",i,None)
                    if target=="false": menu[f"rule:discriminate:h{i}"]=("discriminate",i,None)
                    for d in ("forward","backward"):
                        menu[f"rule:rewrite:h{i}:{d}"]=("rewrite",(i,d),None)
        return menu

    def expression_menu(self):
        hole=first_hole(self.draft["expr"])
        if not hole: return {}
        _,h=hole
        return dict(expression_options(self.current(),self.machine.states,h.sort,h.bound))

    def available_actions(self):
        if self.status!="ACTIVE": return []
        if self.answer is None: return ["choose:HALT","choose:NONHALT","reset"]
        if self.candidate is not None:
            candidate=self.candidate
            if candidate["kind"]=="closedStates":
                states=candidate["states"]
                choices=[("cert:remove:" if q in states else "cert:add:")+q
                         for q in self.machine.states if q!="qH"]
                # Only structural readiness is checked here. The checker,
                # not the action menu, decides if a proposed set is closed.
                verify = ["cert:verify"] if states and not candidate["verify_failed"] else []
                return [*choices,*verify,"cert:cancel","reset"]
            if candidate["kind"]=="exactCycle":
                # Avoid actions that simply rewrite the same number. Do not
                # precheck whether the proposed recurrence is mathematically true.
                start = [f"cert:start:{n}" for n in range(33) if n != candidate["start"]]
                period = [f"cert:period:{n}" for n in range(1,33)
                          if n != candidate["period"]]
                ready = candidate["start"] is not None and candidate["period"] is not None
                verify = ["cert:verify"] if ready and not candidate["verify_failed"] else []
                return [*start,*period,*verify,"cert:cancel","reset"]
            raise Invalid("unknown certificate draft")
        if self.draft:
            return [*self.expression_menu(),*([] if first_hole(self.draft["expr"]) else ["commit"]),"cancel","reset"]
        goals=self.goals()
        # An incomplete proof cannot pass the kernel: only expose submit
        # when there are no open goals. This is syntactic, not a truth oracle.
        submit = ["submit"] if not goals and not self.submit_failed else []
        return [*self.rule_menu(),*(f"focus:{i}" for i in range(len(goals))),*submit,"reset"]

    def observation(self):
        goals=self.goals()
        obs={"machine":self.machine.to_json(),"answer":self.answer,"status":self.status,
             "actions_left":self.left,"limits":{"nodes":self.max_nodes,"depth":self.max_depth},
             "focus":self.focus,"goals":[{"path":p,**g.text()} for p,g in goals],
             "proof":self.proof,"draft":None,"candidate":self.candidate,
             "submit_failed":self.submit_failed,"last_error":self.error}
        if self.draft:
            h=first_hole(self.draft["expr"])
            obs["draft"]={"rule":self.draft["rule"],"data":self.draft["data"],
                          "expression":draft_text(self.draft["expr"]),
                          "slot":None if h is None else {"sort":h[1].sort,"bound":h[1].bound}}
        actions=self.available_actions()
        copies=self.expression_menu() if self.draft else {}
        obs["actions"]=[{"id":a,"expression":show(copies[a])} if a.startswith("copy:") else {"id":a} for a in actions]
        return obs

    def certificate(self):
        return {"format":"proofgame-0.1","machine":self.machine.to_json(),
                "answer":self.answer,"proof":copy.deepcopy(self.proof)}

    def _commit(self,rule,data):
        children=expand(self.machine,self.current(),rule,data)
        node={"rule":rule,"data":data,"children":[None]*len(children)}
        # Persistent updates keep prior boards intact without deep-copying an
        # entire proof at every action. Only the path to the changed hole moves.
        def insert(p,path):
            if not path:return node
            children=list(p['children']);children[path[0]]=insert(children[path[0]],path[1:])
            return {'rule':p['rule'],'data':p['data'],'children':children}
        self.proof=insert(self.proof,self.focus)
        self.submit_failed=False
        self.draft=None
        self._cached_goals=None
        goals=self.goals(); self.focus=goals[0][0] if goals else ()

    def _limits(self):
        count=0
        def walk(x,depth=0):
            nonlocal count
            count+=1
            if count>self.max_nodes or depth>self.max_depth: raise Limit("workspace size/depth limit")
            if isinstance(x,dict):
                for v in x.values(): walk(v,depth+1)
            elif isinstance(x,(tuple,list)):
                for v in x: walk(v,depth+1)
        walk(self.proof)
        if self.draft: walk(self.draft)
        if self.candidate: walk(self.candidate)
        for _,g in self.goals():
            walk(g.target)
            for f in g.facts: walk(f)

    def step(self,action,observe=True):
        if self.status!="ACTIVE": raise Invalid("episode is terminal")
        self.left-=1; self.error=None; self.reward=0
        old=(self.answer,self.proof,self.focus,self.draft,self.candidate,
             self.submit_failed)
        try:
            if action not in self.available_actions(): raise Invalid("action is not in current menu")
            if action=="reset":
                # Reset only the proof workspace, never the proposition being
                # proved. In particular a forced HALT/NONHALT rollout cannot
                # switch branches after a reset and learn from the other answer.
                # A new Game instance is required to choose a different answer.
                self.proof=None; self.focus=(); self.draft=None
                self.candidate=None; self.submit_failed=False; self._cached_goals=None
            elif action.startswith("choose:"):
                self.answer=action.split(":")[1]; self._cached_goals=None
            elif action=="rule:closedStates":
                self.candidate={"kind":"closedStates","states":[],"verify_failed":False}
            elif action=="rule:exactCycle":
                self.candidate={"kind":"exactCycle","start":None,"period":None,
                                "verify_failed":False}
            elif self.candidate is not None:
                candidate=self.candidate
                if action=="cert:cancel":
                    self.candidate=None
                elif action=="cert:verify":
                    if candidate["kind"]=="closedStates":
                        data=tuple(candidate["states"])
                        rule="closedStates"
                    else:
                        data=(candidate["start"],candidate["period"])
                        rule="exactCycle"
                    self._commit(rule,data)
                    self.candidate=None
                elif candidate["kind"]=="closedStates":
                    op,q=action.split(":")[1:]
                    selected=set(candidate["states"])
                    if op=="add": selected.add(q)
                    else: selected.remove(q)
                    self.candidate={**candidate,"states":sorted(selected),
                                    "verify_failed":False}
                else:
                    _,field,number=action.split(":")
                    self.candidate={**candidate,field:int(number),
                                    "verify_failed":False}
            elif action.startswith("focus:"): self.focus=self.goals()[int(action.split(":")[1])][0]
            elif action=="cancel": self.draft=None
            elif action=="submit":
                check(self.certificate()); self.status="SOLVED"; self.reward=1
            elif action=="commit":
                d=self.draft; value=d["expr"]
                data=(d["data"],value) if d["rule"]=="specialize" else value
                self._commit(d["rule"],data)
            elif self.draft:
                path,_=first_hole(self.draft["expr"])
                self.draft={**self.draft,"expr":put_at(self.draft["expr"],path,self.expression_menu()[action])}
            else:
                rule,data,sort=self.rule_menu()[action]
                if sort: self.draft={"rule":rule,"data":data,"expr":Hole(sort)}
                else: self._commit(rule,data)
            self._limits()
        except (Invalid,RecursionError) as e:
            (self.answer,self.proof,self.focus,self.draft,self.candidate,
             self.submit_failed)=old
            self._cached_goals=None
            # A failed check is a fact about this exact candidate/proof.
            # Keep the flag in the observable Markov state until it changes.
            if action=="cert:verify" and self.candidate is not None:
                self.candidate={**self.candidate,"verify_failed":True}
            if action=="submit":
                self.submit_failed=True
            self.error=str(e); self.status="ACTIVE"; self.reward=0
        if self.left==0 and self.status!="SOLVED": self.status="EXHAUSTED"
        return self.observation() if observe else None,self.reward,self.status!="ACTIVE"
