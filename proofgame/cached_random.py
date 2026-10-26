"""Exact uniform-action simulation with memoized deterministic transitions.

The cache changes neither menus nor proof rules. It is not a theorem archive:
no lemma is offered to the agent. Error messages and remaining budget are left
out of keys because they do not affect any menu or transition before timeout.
Only locally created pickle bytes are used, never user-supplied pickle files.
"""
from dataclasses import dataclass
import argparse
import copy
import json
from pathlib import Path
import pickle
import random
import time
from .environment import Game
from .kernel import check
from .logic import Machine
from .tasks import sample,key


@dataclass
class Node:
    board: bytes
    actions: tuple
    edges: list


class Simulator:
    def __init__(self,machine,max_cached_states=100000):
        self.machine=machine; self.max_cached_states=max_cached_states
        self.nodes=[]; self.lookup={}; self.hits=0; self.misses=0
        self.root=self.intern(Game(machine,budget=1000))

    def pack(self,g):
        return pickle.dumps((g.answer,g.proof,g.focus,g.draft,g.candidate),protocol=5)

    def restore(self,board):
        g=Game(self.machine,budget=1000)
        g.answer,g.proof,g.focus,g.draft,g.candidate=pickle.loads(board)
        return g

    def intern(self,g):
        board=self.pack(g)
        # Pickle byte encodings depend on object aliasing, not just values.
        # Use a canonical value representation for interning instead.
        canonical=self.state_key(g)
        if canonical in self.lookup:return self.lookup[canonical]
        if len(self.nodes)>=self.max_cached_states:return board
        idx=len(self.nodes); self.lookup[canonical]=idx
        actions=tuple(g.available_actions())
        self.nodes.append(Node(board,actions,[None]*len(actions)))
        return idx

    def state_key(self,g):
        return repr((g.answer,g.proof,g.focus,g.draft,g.candidate))

    def transition(self,state,index):
        if isinstance(state,int):
            node=self.nodes[state]
            edge=node.edges[index]
            if edge is not None:
                self.hits+=1;return edge
            board=node.board;action=node.actions[index]
        else:
            board=state;action=None
        self.misses+=1;g=self.restore(board)
        if action is None:action=g.available_actions()[index]
        g.step(action,observe=False)
        if g.status=='SOLVED':
            cert=g.certificate();check(cert);result=('SOLVED',cert)
        else:result=self.intern(g)
        if isinstance(state,int):self.nodes[state].edges[index]=result
        return result

    def menu(self,state):
        return self.nodes[state].actions if isinstance(state,int) else tuple(self.restore(state).available_actions())

    def attempt(self,rng,budget=100):
        state=self.root
        for step in range(budget):
            actions=self.menu(state)
            state=self.transition(state,rng.randrange(len(actions)))
            if isinstance(state,tuple) and state[0]=='SOLVED':return state[1],step+1
        return None,budget


def run(machine,attempts,budget,seed,max_states,out,stop_on_success=True):
    sim=Simulator(machine,max_states);rng=random.Random(seed);start=time.monotonic()
    found=[];steps=0;last=time.monotonic()
    for i in range(attempts):
        cert,n=sim.attempt(rng,budget);steps+=n
        if cert:
            found.append({'attempt':i+1,'actions_in_attempt':n,'certificate':cert})
        if time.monotonic()-last>10 or cert or i+1==attempts:
            report={'machine':machine.to_json(),'machine_id':key(machine),'seed':seed,
                    'attempt_limit':attempts,'action_budget':budget,'attempts_completed':i+1,
                    'actions_simulated':steps,'successes':found,'elapsed_seconds':time.monotonic()-start,
                    'cache_states':len(sim.nodes),'cache_hits':sim.hits,'cache_misses':sim.misses,
                    'complete':bool(cert and stop_on_success) or i+1==attempts}
            out.parent.mkdir(parents=True,exist_ok=True)
            temp=out.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(out)
            print(json.dumps({k:v for k,v in report.items() if k not in ('machine','successes')},sort_keys=True),flush=True)
            last=time.monotonic()
        if cert and stop_on_success:break
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--machine',type=Path)
    p.add_argument('--states',type=int,default=2)
    p.add_argument('--attempts',type=int,default=1000000)
    p.add_argument('--budget',type=int,default=100)
    p.add_argument('--seed',type=int,default=532)
    p.add_argument('--task-seed',type=int,default=20261025)
    p.add_argument('--cache-states',type=int,default=100000)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--continue-after-success',action='store_true')
    args=p.parse_args()
    if min(args.attempts,args.budget,args.cache_states)<1:p.error('positive limits required')
    m=Machine(json.loads(args.machine.read_text())['rows']) if args.machine else sample(random.Random(args.task_seed),(args.states,),'test')
    run(m,args.attempts,args.budget,args.seed,args.cache_states,args.out,not args.continue_after_success)


if __name__=='__main__':main()
