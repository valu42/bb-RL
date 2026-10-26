import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"),"optional PyTorch dependency is not installed")
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        from proofgame.model import Policy
        torch.set_num_threads(2); torch.manual_seed(31)
        self.policy=Policy(width=32,layers=1,max_tokens=4096)

    def test_dynamic_actions_gradient_and_roundtrip(self):
        import io
        import torch
        from proofgame.environment import Game
        from proofgame.model import Policy
        from proofgame.tasks import fixtures
        g=Game(fixtures()["one_step"])
        obs=g.observation(); logits,value=self.policy(obs)
        self.assertEqual(tuple(logits.shape),(len(obs["actions"]),))
        distribution=torch.distributions.Categorical(logits=logits)
        loss=-distribution.log_prob(torch.tensor(0))*(1-value.detach())+0.5*(value-1).square()
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.policy.parameters() if p.grad is not None),0)
        buffer=io.BytesIO(); torch.save(self.policy.state_dict(),buffer); buffer.seek(0)
        other=Policy(**self.policy.config); other.load_state_dict(torch.load(buffer,weights_only=True))
        with torch.no_grad(): self.assertTrue(torch.equal(self.policy(obs)[0],other(obs)[0]))
        g.step("choose:HALT"); obs=g.observation(); logits,_=self.policy(obs)
        self.assertEqual(logits.numel(),len(obs["actions"]))

    def test_no_silent_input_truncation(self):
        from proofgame.model import ModelLimit
        self.policy.config["max_tokens"]=4
        with self.assertRaises(ModelLimit): self.policy.encode(["ten variables"])

    def test_baselines_share_exact_test_machines(self):
        from proofgame.train import evaluate
        learned=evaluate(self.policy,3,(2,3),4,101)
        random=evaluate(None,3,(2,3),4,101)
        self.assertEqual(learned["machine_ids"],random["machine_ids"])


if __name__=="__main__": unittest.main()
