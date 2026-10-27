"""Root-only shortcut: structural semantics, full derivatives and CLI safety."""
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import torch
from torch import nn

from proofgame.experimental_root_query import (
    root_layer_forward, root_query_encoder, root_relation_ids, validate_layer,
)
from proofgame.structural_encoding import relation_ids, cached_tree
from proofgame.model import Policy, ModelLimit
from proofgame.environment import Game
from proofgame.network_search import encode_json, FrozenSampler
from proofgame.population_sample import from_rank


class RootQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)

    def policy(self, dtype=torch.float32):
        torch.manual_seed(312)
        model = Policy(width=32, layers=1, representation='ast').to(dtype=dtype).eval()
        with torch.no_grad():
            model.structural_encoder.relationship.weight.uniform_(-.4, .4)
        return model

    def texts(self):
        return [encode_json({'budget': 12, 'goal': '(all Nat (eq (bound 0) (bound 0)))',
                             'facts': [], 'proof': None}),
                encode_json({'goal': '(and false false)', 'facts': ['false'] * 7,
                             'nested': {'description': 'different length 12345'}}),
                encode_json({'id': 'reset'})]

    def test_root_relations_equal_full_row_random_trees_and_binding_precedence(self):
        rng = random.Random(10)
        for count in (1, 2, 7, 31, 128):
            for _ in range(8):
                parents = tuple([-1] + [rng.randrange(i) for i in range(1, count)])
                bindings = tuple((rng.randrange(count), rng.randrange(count))
                                 for _ in range(5))
                torch.testing.assert_close(root_relation_ids(parents, bindings),
                                           relation_ids(parents, bindings)[0],
                                           atol=0, rtol=0)
        for text in self.texts():
            _, parents, bindings = cached_tree(text)
            torch.testing.assert_close(root_relation_ids(parents, bindings),
                                       relation_ids(parents, bindings)[0], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            root_relation_ids((-1, -1), ())

    def test_layer_input_mask_and_every_parameter_gradient_double_precision(self):
        for pre_norm in (False, True):
            for activation in ('relu', 'gelu'):
                torch.manual_seed(31)
                old = nn.TransformerEncoderLayer(
                    32, 4, 128, dropout=0., activation=activation,
                    batch_first=True, norm_first=pre_norm).double().eval()
                new = copy.deepcopy(old)
                x = torch.randn(2, 19, 32, dtype=torch.float64, requires_grad=True)
                x2 = x.detach().clone().requires_grad_()
                bias = torch.randn(2, 4, 19, 19, dtype=torch.float64)
                bias[1, :, :, -5:] = -1e4
                bias.requires_grad_()
                bias2 = bias.detach().clone().requires_grad_()
                expected = old(x, src_mask=bias.reshape(8, 19, 19))[:, :1]
                actual = root_layer_forward(new, x2, bias2[:, :, :1])
                weights = torch.randn_like(expected)
                (expected * weights).sum().backward()
                (actual * weights).sum().backward()
                torch.testing.assert_close(expected, actual, rtol=2e-12, atol=2e-12)
                torch.testing.assert_close(x.grad, x2.grad, rtol=2e-12, atol=2e-12)
                torch.testing.assert_close(bias.grad, bias2.grad, rtol=2e-12, atol=2e-12)
                for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
                    torch.testing.assert_close(p.grad, q.grad, rtol=2e-12, atol=2e-12,
                                               msg=lambda message: name + ': ' + message)

    def test_encoder_padding_learned_relations_and_gradients(self):
        for dtype in (torch.float64, torch.float32):
            model = self.policy(dtype)
            texts = self.texts()
            expected = model.encode(texts)
            torch.manual_seed(82)
            weights = torch.randn_like(expected)
            (expected * weights).sum().backward()
            baseline = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            with root_query_encoder(model):
                actual = model.encode(texts)
                (actual * weights).sum().backward()
            tolerance = 3e-5 if dtype == torch.float32 else 3e-12
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            for name, p in model.named_parameters():
                if name in baseline:
                    torch.testing.assert_close(p.grad, baseline[name], rtol=tolerance, atol=tolerance)
            self.assertTrue(torch.isfinite(model.structural_encoder.relationship.weight.grad).all())

    def test_policy_logits_value_score_many_and_signed_loss_gradients(self):
        model = self.policy()
        game = Game(from_rank(8), budget=12)
        first = game.observation()
        game.step('choose:HALT', observe=False)
        second = game.observation()
        observations = [first, second, first]
        reference_logits = model.score_many(observations)
        reference_value = model(first)[1]
        loss = sum((scores.log_softmax(-1) * torch.linspace(-.8, 1.3, len(scores))).sum()
                   for scores in reference_logits) + reference_value.square()
        loss.backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        with root_query_encoder(model):
            actual_logits = model.score_many(observations)
            actual_value = model(first)[1]
            actual_loss = sum((scores.log_softmax(-1) * torch.linspace(-.8, 1.3, len(scores))).sum()
                              for scores in actual_logits) + actual_value.square()
            actual_loss.backward()
        for actual, expected in zip(actual_logits, reference_logits):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(actual_value, reference_value, rtol=3e-5, atol=3e-5)
        for name, p in model.named_parameters():
            if name in grads:
                torch.testing.assert_close(p.grad, grads[name], rtol=5e-5, atol=3e-5)

    def test_no_default_change_restores_forward_and_rejects_unsupported_models(self):
        model = self.policy()
        before = {n: p.detach().clone() for n, p in model.state_dict().items()}
        original = model.structural_encoder.forward
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with root_query_encoder(model):
                model.encode(self.texts())
                raise RuntimeError('deliberate')
        self.assertEqual(original, model.structural_encoder.forward)
        self.assertEqual(set(before), set(model.state_dict()))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        for unsupported in (Policy(width=32, layers=2, representation='ast'),
                            Policy(width=32, layers=1, representation='bytes')):
            with self.assertRaises(ValueError):
                with root_query_encoder(unsupported):
                    pass
        layer = nn.TransformerEncoderLayer(32, 4, dropout=.1, batch_first=True)
        with self.assertRaises(ValueError):
            validate_layer(layer)
        model.structural_encoder.max_tokens = 2
        with root_query_encoder(model):
            with self.assertRaises(ModelLimit):
                model.encode(self.texts())

    def test_cli_on_reusable_workload_and_checkpoint_not_overwritten(self):
        # This CLI fixture is synthetic. Real CLI collection uses production
        # group-relative advantages, including failed trajectories.
        model = self.policy()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt = root / 'model_epoch_02.pt'
            torch.save({'config': model.config, 'state_dict': model.state_dict()}, ckpt)
            digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
            game = Game(from_rank(8), budget=8)
            observations = [game.observation()]
            game.step('choose:HALT', observe=False)
            observations.append(game.observation())
            sampler = FrozenSampler(model)
            samples = []
            for obs in observations:
                probabilities = sampler.probabilities(obs)
                counts = [0.] * len(probabilities)
                counts[0], counts[-1] = 1., -.8
                samples.append({'observation': obs, 'counts': counts,
                                'expected': {0: probabilities[0], len(counts)-1: probabilities[-1]}})
            work = root / 'fixture.json'
            work.write_text(json.dumps({'schema': 'root-query-workload-1',
                                        'checkpoint_sha256': digest, 'epsilon': 0., 'denominator': 8,
                                        'source': {'synthetic_test_fixture': True}, 'samples': samples}))
            report = root / 'report.json'
            command = [sys.executable, '-m', 'proofgame.benchmark_root_query', '--run', str(root),
                       '--reuse-workload', str(work), '--devices', 'cpu', '--out', str(report)]
            result = subprocess.run(command, text=True, capture_output=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            data = json.loads(report.read_text())
            self.assertEqual(data['status'], 'complete')
            self.assertTrue(data['devices']['cpu']['correctness']['passes'])
            self.assertEqual(len(data['devices']['cpu']['gradient']['trial_seconds']['baseline']), 3)
            self.assertEqual(hashlib.sha256(ckpt.read_bytes()).hexdigest(), digest)
            rerun = subprocess.run(command, text=True, capture_output=True, timeout=30)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn('output already exists', rerun.stderr)

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS hardware required')
    def test_mps_outputs_and_parameter_gradients(self):
        model = self.policy().to('mps')
        texts = self.texts()
        expected = model.encode(texts)
        weights = torch.randn_like(expected)
        (expected * weights).sum().backward()
        old = {n: p.grad.cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        with root_query_encoder(model):
            actual = model.encode(texts)
            (actual * weights).sum().backward()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-4, atol=2e-5)
        for name, p in model.named_parameters():
            if name in old:
                torch.testing.assert_close(p.grad.cpu(), old[name], rtol=3e-4, atol=3e-5)


if __name__ == '__main__':
    unittest.main()
