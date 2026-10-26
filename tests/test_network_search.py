import importlib.util
import unittest

from proofgame.environment import Game
from proofgame.population_sample import from_rank


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency is not installed")
class FrozenSamplerTests(unittest.TestCase):
    def test_cached_probabilities_match_direct_policy_and_include_full_context(self):
        import torch
        from proofgame.model import Policy
        from proofgame.network_search import FrozenSampler
        torch.set_num_threads(1)
        torch.manual_seed(7)
        policy = Policy(width=32, layers=1).eval()
        sampler = FrozenSampler(policy, capacity=2)
        game = Game(from_rank(8), budget=20)
        with torch.inference_mode():
            for action in [None, "choose:HALT", "submit", "rule:exI", "s", "z", "commit"]:
                if action is not None:
                    game.step(action, observe=False)
                obs = game.observation()
                expected = policy(obs)[0].softmax(-1)
                actual = torch.tensor(sampler.probabilities(obs))
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)
                misses = sampler.misses
                self.assertEqual(tuple(actual.tolist()), sampler.probabilities(obs))
                self.assertEqual(sampler.misses, misses)
            # Neither changed remaining budget nor another machine may reuse an observation key.
            for changed in [Game(from_rank(8), budget=19), Game(from_rank(0), budget=20)]:
                misses = sampler.misses
                obs = changed.observation()
                actual = torch.tensor(sampler.probabilities(obs))
                self.assertEqual(sampler.misses, misses + 1)
                torch.testing.assert_close(actual, policy(obs)[0].softmax(-1), rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
