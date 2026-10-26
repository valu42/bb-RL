"""Constructive NONHALT proof in the actual proofgame kernel and action menu.

A two-state machine with an unreachable second state stays in q0 forever.
No simulator or trusted theorem oracle is used in the certificate.
"""
import unittest

from proofgame.logic import Machine, C0
from proofgame.kernel import check
from proofgame.environment import Game


MACHINE = Machine((
    ('q0','b1','R'), ('q0','b1','R'),
    ('qH','b0','R'), ('qH','b1','R'),
))
V = ('var', 0)
Q = lambda t: ('qof', ('at', t, C0))
LEMMA = ('all','Nat',('eq',Q(('bound',0)),'q0'))
BIT = ('head',('right',('at',V,C0)))


def node(rule, data=None, *children):
    return {'rule':rule, 'data':data, 'children':list(children)}


def nonhalt_certificate():
    # Prove the invariant forall n, state(at n C0)=q0 by Nat induction.
    # In the inductive step, rewrite state(at n C0) to q0 using IH,
    # then split on the scanned bit; either transition returns q0.
    bit_branch = lambda: node('rewrite',(1,'forward'),node('refl'))
    step = node('reduce',None,
                node('rewrite',(0,'forward'),
                     node('casesBit',BIT,bit_branch(),bit_branch())))
    invariant = node('natI',None,node('refl'),step)

    # From the invariant, any assumption state(at n C0)=qH is
    # contradictory, since q0 and qH are distinct constructors.
    conclusion = node('allI',None,
                      node('impI',None,
                           node('specialize',(0,V),
                                node('rewrite',(2,'forward'),
                                     node('discriminate',1))))
    return {'format':'proofgame-0.1',
            'machine':MACHINE.to_json(),
            'answer':'NONHALT',
            'proof':node('cut',LEMMA,invariant,conclusion)}


def play_nonhalt_game(budget=100):
    game=Game(MACHINE,budget=budget)
    actions=[
        'choose:NONHALT',
        'rule:cut',
        'put:all:Nat','put:eq:State','put:qof','put:at','bound:0',
        'put:cfg','put:q0','put:nil','put:nil','put:q0','commit',
        'rule:natI',
        'rule:refl',
        'rule:reduce','rule:rewrite:h0:forward',
        'rule:casesBit','put:head','put:right','put:at','var:v0',
        'put:cfg','put:q0','put:nil','put:nil','commit',
        'rule:rewrite:h1:forward','rule:refl',
        'rule:rewrite:h1:forward','rule:refl',
        'rule:allI','rule:impI',
        'rule:specialize:h0','var:v0','commit',
        'rule:rewrite:h2:forward','rule:discriminate:h1',
        'submit',
    ]
    for i,action in enumerate(actions):
        available=game.available_actions()
        if action not in available:
            raise AssertionError(f'action {i+1} {action} unavailable; menu={available}, '
                                 f'goals={game.observation()["goals"]}, error={game.error}')
        game.step(action,observe=False)
        if game.error:
            raise AssertionError(f'action {i+1} {action} rejected: {game.error}')
    return game,actions


class NonhaltProofFeasibilityTests(unittest.TestCase):
    def test_explicit_certificate_accepted(self):
        cert=nonhalt_certificate()
        self.assertEqual(check(cert),'NONHALT')

    def test_complete_nonhalt_proof_game_under_100_actions(self):
        game,actions=play_nonhalt_game()
        self.assertLessEqual(len(actions),100)
        self.assertEqual(game.status,'SOLVED')
        self.assertEqual(game.reward,1)
        self.assertEqual(check(game.certificate()),'NONHALT')
        self.assertEqual(game.certificate(),nonhalt_certificate())
        print('NONHALT_PROOF_VERIFIED',{'states':2,'proof_actions':len(actions),
                                         'certificate_nodes':12},flush=True)


if __name__=='__main__':
    unittest.main()
