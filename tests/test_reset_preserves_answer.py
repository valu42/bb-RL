"""A proof reset must not turn a HALT attempt into a NONHALT attempt."""
import random
import unittest

from proofgame.environment import Game
from proofgame.kernel import check
from proofgame.logic import Machine
from proofgame.network_search import attempt
from proofgame.population_sample import from_rank


CLOSED = Machine((
    ('q0','b1','R'),('q0','b1','R'),
    ('qH','b0','R'),('qH','b1','R'),
))


class ScriptedSampler:
    def __init__(self, actions):
        self.actions = iter(actions)

    def probabilities(self, observation):
        desired = next(self.actions)
        choices = [item['id'] for item in observation['actions']]
        if desired not in choices:
            raise AssertionError(f'{desired} unavailable after reset: {choices}')
        return tuple(float(a == desired) for a in choices)


class ResetPreservesAnswerTests(unittest.TestCase):
    def test_reset_clears_workspace_but_preserves_selected_answer(self):
        for answer in ('HALT','NONHALT'):
            with self.subTest(answer=answer):
                game = Game(from_rank(8), budget=30)
                game.step('choose:' + answer, observe=False)
                game.step('reset', observe=False)
                obs = game.observation()
                self.assertEqual(obs['answer'], answer)
                self.assertIsNone(obs['proof'])
                self.assertIsNone(obs['candidate'])
                self.assertIsNone(obs['draft'])
                self.assertEqual(obs['actions_left'], 28)
                self.assertNotIn('choose:HALT', game.available_actions())
                self.assertNotIn('choose:NONHALT', game.available_actions())
                self.assertIn('reset', game.available_actions())

    def test_reset_clears_failed_certificate_without_switching_answer(self):
        game = Game(CLOSED, budget=30)
        for action in ('choose:NONHALT','rule:closedStates','cert:add:q0'):
            game.step(action, observe=False)
        self.assertIsNotNone(game.candidate)
        game.step('reset', observe=False)
        self.assertEqual(game.answer,'NONHALT')
        self.assertIsNone(game.candidate)
        self.assertIsNone(game.proof)
        self.assertIn('rule:closedStates', game.available_actions())
        self.assertNotIn('choose:HALT', game.available_actions())

    def test_reset_after_partial_halt_proof_does_not_switch_answer(self):
        game=Game(from_rank(8),budget=30)
        game.step('choose:HALT',observe=False)
        game.step('rule:exI',observe=False)
        self.assertIsNotNone(game.draft)
        game.step('reset',observe=False)
        self.assertIsNone(game.draft)
        self.assertIsNone(game.proof)
        self.assertEqual(game.answer,'HALT')
        self.assertNotIn('rule:closedStates',game.available_actions())

    def test_real_forced_rollouts_can_reset_and_still_prove_each_answer(self):
        cases=(
            (from_rank(8),'HALT',
             ('choose:HALT','reset','rule:exI','put:s','put:z',
              'commit','rule:refl','submit')),
            (CLOSED,'NONHALT',
             ('choose:NONHALT','reset','rule:closedStates',
              'cert:add:q0','cert:verify','submit')),
        )
        for machine,answer,actions in cases:
            with self.subTest(answer=answer):
                cert,trace,status=attempt(
                    ScriptedSampler(actions),machine,random.Random(19),
                    budget=20,forced_initial_answer=answer)
                self.assertEqual(status,'SOLVED')
                self.assertEqual([t['action'] for t in trace],list(actions))
                self.assertEqual(cert['answer'],answer)
                self.assertEqual(check(cert),answer)
                self.assertEqual(trace[0]['forced_initial_answer'],answer)
                self.assertFalse(any('forced_initial_answer' in step
                                     for step in trace[1:]))

    def test_new_game_still_allows_other_answer(self):
        first=Game(CLOSED)
        first.step('choose:HALT',observe=False)
        first.step('reset',observe=False)
        self.assertEqual(first.answer,'HALT')
        second=Game(CLOSED)
        self.assertIn('choose:NONHALT',second.available_actions())
        self.assertIsNone(second.answer)


if __name__ == '__main__':
    unittest.main()
