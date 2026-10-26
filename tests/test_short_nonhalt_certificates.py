"""Short sound NONHALT certificates, checked through the actual proof game."""
import copy
import unittest

from proofgame.environment import Game
from proofgame.kernel import check
from proofgame.logic import Invalid, Limit, Machine
from proofgame.nonhalt_certificates import check_closed_states, check_exact_cycle


CLOSED = Machine((
    ('q0','b1','R'),('q0','b1','R'),
    ('qH','b0','R'),('qH','b1','R'),
))
PARITY = Machine((
    ('q1','b1','R'),('q1','b1','R'),
    ('q0','b0','L'),('qH','b1','R'),
))
HALTER = Machine((
    ('qH','b1','R'),('qH','b1','R'),
    ('qH','b0','L'),('qH','b0','L'),
))


def replay(machine, actions, budget=100):
    game=Game(machine,budget=budget)
    for action in actions:
        assert action in game.available_actions(), (action,game.available_actions())
        game.step(action,observe=False)
        assert game.error is None, (action,game.error)
    assert game.status=='SOLVED',game.status
    assert game.reward==1
    assert check(game.certificate())=='NONHALT'
    return game


class ShortNonhaltProofTests(unittest.TestCase):
    def test_closed_set_five_action_certificate(self):
        actions=['choose:NONHALT','rule:closedStates',
                 'cert:add:q0','cert:verify','submit']
        game=replay(CLOSED,actions)
        self.assertEqual(len(actions),5)
        self.assertEqual(game.certificate()['proof']['rule'],'closedStates')
        self.assertEqual(tuple(game.certificate()['proof']['data']),('q0',))
        print('SHORT_NONHALT_PROOF',{'kind':'closedStates','actions':len(actions)},flush=True)

    def test_parity_trap_six_action_exact_recurrence(self):
        actions=['choose:NONHALT','rule:exactCycle',
                 'cert:start:2','cert:period:2','cert:verify','submit']
        game=replay(PARITY,actions)
        self.assertEqual(len(actions),6)
        self.assertEqual(game.certificate()['proof']['rule'],'exactCycle')
        self.assertEqual(tuple(game.certificate()['proof']['data']),(2,2))
        print('SHORT_NONHALT_PROOF',{'kind':'exactCycle','actions':len(actions)},flush=True)

    def test_halter_cannot_forge_either_certificate(self):
        self.assertRaises(Invalid,check_closed_states,HALTER,('q0',))
        self.assertRaises(Invalid,check_exact_cycle,HALTER,(0,2))
        self.assertRaises(Invalid,check_exact_cycle,HALTER,(1,1))
        for rule,data in [('closedStates',['q0']),('exactCycle',[0,1])]:
            cert={'format':'proofgame-0.1','machine':HALTER.to_json(),
                  'answer':'NONHALT',
                  'proof':{'rule':rule,'data':data,'children':[]}}
            with self.assertRaises(Invalid):
                check(cert)

    def test_rejects_nonclosed_sets_and_invalid_cycles(self):
        self.assertRaises(Invalid,check_closed_states,CLOSED,('q1',))
        self.assertRaises(Invalid,check_closed_states,CLOSED,('q0','qH'))
        self.assertRaises(Invalid,check_closed_states,CLOSED,('q0','q0'))
        self.assertRaises(Invalid,check_closed_states,PARITY,('q0',))
        self.assertRaises(Invalid,check_exact_cycle,PARITY,(0,2))
        self.assertRaises(Invalid,check_exact_cycle,PARITY,(2,1))
        self.assertRaises(Invalid,check_exact_cycle,PARITY,(True,2))
        self.assertRaises(Invalid,check_exact_cycle,PARITY,(2,0))
        self.assertRaises(Limit,check_exact_cycle,PARITY,(4096,1))

    def test_failed_verify_does_not_create_proof_or_reward(self):
        game=Game(HALTER,budget=100)
        for action in ['choose:NONHALT','rule:closedStates','cert:add:q0','cert:verify']:
            game.step(action,observe=False)
        self.assertEqual(game.status,'ACTIVE')
        self.assertEqual(game.reward,0)
        self.assertIsNone(game.proof)
        self.assertIsNotNone(game.error)
        self.assertIn('cert:cancel',game.available_actions())

    def test_tampered_machine_or_proof_is_rechecked(self):
        good=replay(PARITY,['choose:NONHALT','rule:exactCycle',
                            'cert:start:2','cert:period:2','cert:verify','submit'])
        bad=copy.deepcopy(good.certificate())
        bad['machine']=HALTER.to_json()
        with self.assertRaises(Invalid):
            check(bad)
        bad=copy.deepcopy(good.certificate())
        bad['proof']['data']=[2,3]
        with self.assertRaises(Invalid):
            check(bad)

    def test_no_certificate_action_on_halt_branch(self):
        game=Game(CLOSED,budget=20)
        game.step('choose:HALT',observe=False)
        self.assertNotIn('rule:closedStates',game.available_actions())
        self.assertNotIn('rule:exactCycle',game.available_actions())


if __name__=='__main__':
    unittest.main()
