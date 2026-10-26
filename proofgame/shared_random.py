"""Reuse transitions across equal-size machines, guarded by every row read.

Uniform action streams remain independent per machine. Cached computations are
used only when the current machine agrees on every transition row consulted
by the original computation. This is an execution optimization, not a lemma
archive or a different proof search policy.
"""
from .cached_random import Simulator
from .environment import Game
from .logic import Machine


_reads=None
_original_row=Machine.row


def traced_row(self,q,bit):
    index=2*int(q[1:])+(bit=='b1')
    if _reads is not None:_reads[index]=self.rows[index]
    return _original_row(self,q,bit)


class SharedSimulator(Simulator):
    def __init__(self,machine,max_cached_states=100000):
        Machine.row=traced_row
        self.state_count=len(machine.rows)
        super().__init__(machine,max_cached_states)

    def set_machine(self,machine):
        if len(machine.rows)!=self.state_count:raise ValueError('cannot mix machine sizes')
        self.machine=machine

    def state_key(self,g):
        # Identical proof syntax can derive different child goals after reduce
        # on different machines. Menus belong to the derived state, not syntax.
        return repr((g.answer,g.proof,g.focus,g.draft,tuple(g.goals())))

    def transition(self,state,index):
        global _reads
        if isinstance(state,int):
            node=self.nodes[state];variants=node.edges[index]
            if variants is not None:
                for requirements,result in variants:
                    if all(self.machine.rows[i]==row for i,row in requirements):
                        self.hits+=1;return result
            board=node.board;action=node.actions[index]
        else:board=state;action=None
        self.misses+=1;g=self.restore(board)
        _reads={}
        try:
            if action is None:action=g.available_actions()[index]
            g.step(action,observe=False)
            if g.status=='SOLVED':result=('SOLVED',{'answer':g.answer,'proof':g.certificate()['proof']})
            else:result=self.intern(g)
            requirements=tuple(sorted(_reads.items()))
        finally:_reads=None
        if isinstance(state,int):
            if self.nodes[state].edges[index] is None:self.nodes[state].edges[index]=[]
            self.nodes[state].edges[index].append((requirements,result))
        return result

    def attempt(self,rng,budget=20):
        state=self.root
        for step in range(budget):
            actions=self.menu(state);state=self.transition(state,rng.randrange(len(actions)))
            if isinstance(state,tuple) and state[0]=='SOLVED':
                certificate={'format':'proofgame-0.1','machine':self.machine.to_json(),
                             'answer':state[1]['answer'],'proof':state[1]['proof']}
                return certificate,step+1
        return None,budget
