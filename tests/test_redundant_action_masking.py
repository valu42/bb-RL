"""State-based action masking must avoid loops without leaking theorem truth."""
import unittest

from proofgame.environment import Game
from proofgame.kernel import check
from proofgame.logic import Machine
from proofgame.cached_random import Simulator
from proofgame.shared_random import SharedSimulator


CLOSED = Machine((
    ('q0','b1','R'),('q0','b1','R'),
    ('qH','b0','R'),('qH','b1','R'),
))
HALTER = Machine((
    ('qH','b1','R'),('qH','b1','R'),
    ('qH','b0','L'),('qH','b0','L'),
))
PARITY = Machine((
    ('q1','b1','R'),('q1','b1','R'),
    ('q0','b0','L'),('qH','b1','R'),
))


def play(game, actions):
    for action in actions:
        assert action in game.available_actions(), (action,game.available_actions())
        game.step(action,observe=False)
    return game


class RedundantActionMaskingTests(unittest.TestCase):
    def test_failed_closed_set_verification_masked_until_change(self):
        game=play(Game(HALTER),['choose:NONHALT','rule:closedStates'])
        self.assertNotIn('cert:verify',game.available_actions())
        game.step('cert:add:q0',observe=False)
        self.assertIn('cert:verify',game.available_actions())
        # Same syntactic proposal must be offered regardless of whether
        # the particular machine is actually closed.
        valid=play(Game(CLOSED),['choose:NONHALT','rule:closedStates','cert:add:q0'])
        self.assertEqual(game.available_actions(),valid.available_actions())
        game.step('cert:verify',observe=False)
        self.assertIsNotNone(game.error)
        self.assertEqual(game.reward,0)
        self.assertIsNone(game.proof)
        self.assertTrue(game.observation()['candidate']['verify_failed'])
        self.assertNotIn('cert:verify',game.available_actions())
        self.assertIn('cert:remove:q0',game.available_actions())
        game.step('cert:remove:q0',observe=False)
        self.assertNotIn('cert:verify',game.available_actions())  # empty again
        game.step('cert:add:q0',observe=False)
        self.assertFalse(game.observation()['candidate']['verify_failed'])
        self.assertIn('cert:verify',game.available_actions())
        game.step('cert:cancel',observe=False)
        game.step('rule:closedStates',observe=False)
        self.assertNotIn('cert:verify',game.available_actions())
        game.step('reset',observe=False)
        self.assertIsNone(game.observation()['candidate'])

    def test_exact_cycle_requires_both_fields_and_changes_after_failure(self):
        game=play(Game(HALTER),['choose:NONHALT','rule:exactCycle'])
        self.assertNotIn('cert:verify',game.available_actions())
        game.step('cert:start:0',observe=False)
        self.assertNotIn('cert:start:0',game.available_actions())
        self.assertNotIn('cert:verify',game.available_actions())
        game.step('cert:period:1',observe=False)
        self.assertNotIn('cert:period:1',game.available_actions())
        self.assertIn('cert:verify',game.available_actions())
        game.step('cert:verify',observe=False)
        self.assertEqual(game.status,'ACTIVE')
        self.assertTrue(game.observation()['candidate']['verify_failed'])
        self.assertNotIn('cert:verify',game.available_actions())
        game.step('cert:period:2',observe=False)
        self.assertIn('cert:verify',game.available_actions())
        self.assertFalse(game.observation()['candidate']['verify_failed'])

    def test_complete_certificates_still_verify_and_submit(self):
        for machine,actions in [
            (CLOSED,['choose:NONHALT','rule:closedStates','cert:add:q0','cert:verify']),
            (PARITY,['choose:NONHALT','rule:exactCycle','cert:start:2',
                     'cert:period:2','cert:verify']),
        ]:
            game=play(Game(machine),actions)
            self.assertEqual(game.goals(),[])
            self.assertIn('submit',game.available_actions())
            game.step('submit',observe=False)
            self.assertEqual(game.status,'SOLVED')
            self.assertEqual(game.reward,1)
            self.assertEqual(check(game.certificate()),'NONHALT')

    def test_submit_only_when_proof_complete(self):
        game=Game(HALTER)
        self.assertNotIn('submit',game.available_actions())
        game.step('choose:HALT',observe=False)
        self.assertNotIn('submit',game.available_actions())
        game.step('rule:exI',observe=False)
        self.assertNotIn('submit',game.available_actions())  # incomplete witness
        # An invalid explicit submit cannot yield a reward.
        game.step('submit',observe=False)
        self.assertEqual(game.reward,0)
        self.assertEqual(game.status,'ACTIVE')
        self.assertNotIn('submit',game.available_actions())
        game.step('reset',observe=False)
        self.assertFalse(game.observation()['submit_failed'])

    def test_cached_simulator_preserves_failed_candidate_flag(self):
        simulator=Simulator(HALTER)
        state=simulator.root
        for action in ['choose:NONHALT','rule:closedStates','cert:add:q0','cert:verify']:
            state=simulator.transition(state,simulator.menu(state).index(action))
        self.assertNotIn('cert:verify',simulator.menu(state))
        restored=simulator.restore(simulator.nodes[state].board)
        self.assertTrue(restored.candidate['verify_failed'])
        self.assertIn('cert:remove:q0',simulator.menu(state))
        state=simulator.transition(state,simulator.menu(state).index('cert:remove:q0'))
        state=simulator.transition(state,simulator.menu(state).index('cert:add:q0'))
        self.assertIn('cert:verify',simulator.menu(state))

    def test_shared_cache_never_confuses_masked_states(self):
        sim=SharedSimulator(HALTER,max_cached_states=500)
        state=sim.root
        for action in ['choose:NONHALT','rule:closedStates','cert:add:q0','cert:verify']:
            state=sim.transition(state,sim.menu(state).index(action))
        self.assertNotIn('cert:verify',sim.menu(state))
        sim.set_machine(CLOSED)
        fresh=sim.root
        for action in ['choose:NONHALT','rule:closedStates','cert:add:q0']:
            fresh=sim.transition(fresh,sim.menu(fresh).index(action))
        self.assertIn('cert:verify',sim.menu(fresh))


if __name__=='__main__':
    unittest.main()
