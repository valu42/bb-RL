import random
import pickle
import unittest
from proofgame.cached_random import Simulator
from proofgame.environment import Game
from proofgame.demo import run_demo
from proofgame.tasks import fixtures,sample
from proofgame.shared_random import SharedSimulator


class SimulationTests(unittest.TestCase):
    def test_shared_cache_matches_independent_machine_actions(self):
        r=random.Random(98);machines=[sample(r,(2,)) for _ in range(10)]
        sim=SharedSimulator(machines[0],max_cached_states=2000)
        for m in machines:
            sim.set_machine(m)
            for _ in range(20):
                g=Game(m,budget=20);state=sim.root
                while g.status=='ACTIVE':
                    actions=g.available_actions();self.assertEqual(tuple(actions),sim.menu(state))
                    i=r.randrange(len(actions));g.step(actions[i],observe=False)
                    state=sim.transition(state,i)
                    if isinstance(state,tuple):
                        self.assertEqual(g.status,'SOLVED')
                        self.assertEqual(state[1],{'answer':g.answer,'proof':g.certificate()['proof']});break
                    board=sim.nodes[state].board if isinstance(state,int) else state
                    self.assertEqual(pickle.loads(sim.pack(g)),pickle.loads(board))

    def test_shared_cache_guards_machine_dependent_success(self):
        from proofgame.logic import Machine
        halt=Machine((('qH','b0','R'),)*4)
        loop=Machine((('q0','b0','R'),)*4)
        sim=SharedSimulator(halt)
        trace=['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']
        for m in [halt,loop,halt,loop]:
            sim.set_machine(m);state=sim.root;g=Game(m,budget=20)
            for action in trace:
                state=sim.transition(state,sim.menu(state).index(action));g.step(action,observe=False)
            self.assertEqual(isinstance(state,tuple),g.status=='SOLVED')

    def test_shared_cache_distinguishes_reduced_goals(self):
        from proofgame.logic import Machine
        halt=Machine((('qH','b0','R'),)*4);loop=Machine((('q0','b0','R'),)*4)
        sim=SharedSimulator(halt)
        trace=['choose:HALT','rule:exI','put:s','put:z','commit','rule:reduce','rule:casesState']
        for m in [halt,loop,halt,loop]:
            sim.set_machine(m);state=sim.root;g=Game(m,budget=20)
            for action in trace:
                self.assertEqual(tuple(g.available_actions()),sim.menu(state))
                state=sim.transition(state,sim.menu(state).index(action));g.step(action,observe=False)
            self.assertEqual(tuple(g.available_actions()),sim.menu(state))

    def test_cached_matches_uncached_action_for_action(self):
        r=random.Random(812)
        for m in [sample(r,(2,)),sample(r,(3,)),fixtures()['one_step']]:
            sim=Simulator(m,max_cached_states=60) # force the uncached fallback too
            for _ in range(30):
                g=Game(m,budget=100);state=sim.root
                while g.status=='ACTIVE':
                    actions=g.available_actions()
                    self.assertEqual(tuple(actions),sim.menu(state))
                    i=r.randrange(len(actions));g.step(actions[i],observe=False)
                    state=sim.transition(state,i)
                    if isinstance(state,tuple):
                        self.assertEqual(g.status,'SOLVED');self.assertEqual(state[1],g.certificate());break
                    board=sim.nodes[state].board if isinstance(state,int) else state
                    self.assertEqual(pickle.loads(sim.pack(g)),pickle.loads(board))

    def test_cached_success_matches_all_examples(self):
        for name in fixtures():
            g,trace=run_demo(name);sim=Simulator(g.machine);state=sim.root
            for action in trace:
                state=sim.transition(state,sim.menu(state).index(action))
            self.assertEqual(state,('SOLVED',g.certificate()))

    def test_observation_optional_does_not_change_state(self):
        r=random.Random(31);m=fixtures()['one_step'];a=Game(m);b=Game(m)
        for _ in range(150):
            action=r.choice(a.available_actions());a.step(action);b.step(action,observe=False)
            self.assertEqual(a.observation(),b.observation())
            if a.status!='ACTIVE':break


if __name__=='__main__':unittest.main()
