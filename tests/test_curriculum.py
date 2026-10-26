import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional PyTorch dependency is not installed')
class CurriculumTests(unittest.TestCase):
    def test_only_verified_trajectory_can_update_policy(self):
        import torch
        from proofgame.curriculum_train import learn_from_verified_trace
        from proofgame.environment import Game
        from proofgame.model import Policy
        from proofgame.population_sample import from_rank
        torch.set_num_threads(1); torch.manual_seed(7)
        policy = Policy(width=32, layers=1).eval()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0.)
        machine = from_rank(8)
        initial = {k: v.clone() for k,v in policy.state_dict().items()}
        with self.assertRaises(ValueError):
            learn_from_verified_trace(policy, optimizer, machine, [{'action':'choose:HALT'}])
        self.assertTrue(all(torch.equal(initial[k],v) for k,v in policy.state_dict().items()))
        self.assertFalse(optimizer.state)
        actions = ['choose:HALT','rule:exI','put:s','put:z','commit','rule:refl','submit']

        def log_probability():
            g = Game(machine, budget=20); total = 0.
            with torch.no_grad():
                for action in actions:
                    obs = g.observation(); logits,_ = policy(obs)
                    index = next(i for i,a in enumerate(obs['actions']) if a['id']==action)
                    total += float(logits.log_softmax(-1)[index])
                    g.step(action, observe=False)
            self.assertEqual(g.status,'SOLVED')
            return total

        before = log_probability()
        result = learn_from_verified_trace(policy, optimizer, machine, [{'action':a} for a in actions])
        self.assertEqual(result['certificate']['answer'],'HALT')
        self.assertGreater(result['gradient_norm_before_clipping'],0)
        self.assertGreater(log_probability(),before)


if __name__ == '__main__': unittest.main()
