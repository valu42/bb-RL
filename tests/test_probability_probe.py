"""One-off probability diagnostic for a known valid NONHALT proof."""
import math
import unittest
from proofgame.environment import Game
from tests.test_nonhalt_proof_feasibility import MACHINE

ACTIONS=[
 'choose:NONHALT','rule:cut',
 'put:all:Nat','put:eq:State','put:qof','put:at','bound:0',
 'put:cfg','put:q0','put:nil','put:nil','put:q0','commit',
 'rule:natI','rule:refl','rule:reduce','rule:rewrite:h0:forward',
 'rule:casesBit','put:head','put:right','put:at','var:v0',
 'put:cfg','put:q0','put:nil','put:nil','commit',
 'rule:rewrite:h1:forward','rule:refl',
 'rule:rewrite:h1:forward','rule:refl',
 'rule:allI','rule:impI','rule:specialize:h0','var:v0','commit',
 'rule:rewrite:h2:forward','rule:discriminate:h1','submit'
]

class ProbabilityTest(unittest.TestCase):
    def test_probability(self):
        game=Game(MACHINE,budget=100)
        counts=[]
        for action in ACTIONS:
            menu=game.available_actions()
            self.assertIn(action,menu)
            counts.append(len(menu))
            game.step(action,observe=False)
            self.assertFalse(game.error)
        self.assertEqual(game.status,'SOLVED')
        log10=sum(-math.log10(n) for n in counts)
        # If the initial branch is forced, exclude the 3-way choice.
        log10_forced=sum(-math.log10(n) for n in counts[1:])
        print('UNIFORM_NONHALT_PATH_PROBABILITY',{
            'actions':len(ACTIONS),'menu_sizes':counts,
            'log10_probability':log10,
            'probability':10**log10,
            'log10_given_forced_nonhalt':log10_forced,
            'probability_given_forced_nonhalt':10**log10_forced,
            'expected_attempts_given_forced_nonhalt':10**(-log10_forced)
        },flush=True)

if __name__=='__main__':
    unittest.main()
