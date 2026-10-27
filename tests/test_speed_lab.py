"""CPU speed-lab experiments must preserve frozen-model values and gradients."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from proofgame.benchmark_real_gradients import (
    attention_elements, gradient_trial, make_samples, probability_disagreement,
    select_machines)
from proofgame.benchmark_rollout_matrix import build_cases
from proofgame.experimental_action_replay import (
    replay_unique_action_gradients, two_pass_probability_disagreement)
from proofgame.experimental_structural_cache import (
    StructuralPreprocessingCache, cached_structural_forward)
from proofgame.environment import Game
from proofgame.model import Policy
from proofgame.network_search import encode_json
from proofgame.population_sample import from_rank


class SpeedLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @staticmethod
    def example():
        torch.manual_seed(11)
        model = Policy(width=32, layers=1, representation='ast').eval()
        game = Game(from_rank(8), budget=12)
        obs = game.observation()
        return model, obs

    def test_cached_structural_encoder_has_matching_forward_and_backward(self):
        model, observation = self.example()
        context = {k: v for k, v in observation.items() if k != 'actions'}
        texts = [encode_json(context)] + [
            encode_json(a) for a in observation['actions'][:2]]
        model.zero_grad(set_to_none=True)
        baseline = model.encode(texts)
        baseline.sum().backward()
        baseline_grads = [p.grad.clone() if p.grad is not None else None
                          for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        with cached_structural_forward(model, relation_cache_mb=1,
                                       position_cache_mb=1) as cache:
            output = model.encode(texts)
            output.sum().backward()
            torch.testing.assert_close(output, baseline, rtol=0, atol=0)
            self.assertGreater(cache.stats()['relation_entries'], 0)
            self.assertGreater(cache.stats()['position_entries'], 0)
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            for old, p in zip(baseline_grads, model.parameters()):
                if old is not None:
                    torch.testing.assert_close(old, p.grad, rtol=0, atol=0)
            # Reuse all of the same serialized descriptions.
            with torch.inference_mode():
                model.encode(texts)
            self.assertGreater(cache.stats()['hits']['relation'], 0)
            self.assertGreater(cache.stats()['hits']['position'], 0)
            self.assertLessEqual(cache.stats()['relation_cache_bytes'], 1024**2)
        torch.testing.assert_close(model.encode(texts), baseline,
                                   rtol=0, atol=0)

    def test_zero_cache_does_not_store_unbounded_relation_matrices(self):
        model, obs = self.example()
        context = encode_json({k: v for k, v in obs.items() if k != 'actions'})
        with cached_structural_forward(model, relation_cache_mb=0,
                                       position_cache_mb=0) as cache:
            with torch.inference_mode():
                model.encode([context, context])
            self.assertEqual(cache.stats()['relation_entries'], 0)
            self.assertEqual(cache.stats()['position_entries'], 0)
            self.assertEqual(cache.stats()['relation_cache_bytes'], 0)
            self.assertEqual(cache.stats()['position_cache_bytes'], 0)

    def test_exact_loss_coefficient_weights_and_probability_parity(self):
        model, observation = self.example()
        length = len(observation['actions'])
        with torch.inference_mode():
            probs = model(observation)[0].softmax(-1).tolist()
        weights = [0.] * length
        weights[0] = 1.25
        weights[-1] = -0.75
        group = {encode_json(observation): (
            weights, {0: probs[0], length-1: probs[-1]})}
        samples = make_samples(group, model, 0, 44)
        self.assertEqual(len(samples), 1)
        self.assertGreater(attention_elements(samples, 2, True, 4), 0)
        ref, _ = gradient_trial(model, samples, mode='serial',
                                batch_size=1, bucketed=False,
                                denominator=8, epsilon=0.)
        with cached_structural_forward(model, relation_cache_mb=4):
            other, _ = gradient_trial(model, samples, mode='score_many',
                                      batch_size=2, bucketed=True,
                                      denominator=8, epsilon=0.)
            probability_error, compared = probability_disagreement(
                model, samples, 'score_many', 2, True, 0.)
        self.assertEqual(compared, 2)
        self.assertLess(probability_error, 1e-4)
        relative = float(torch.linalg.vector_norm(ref-other) /
                         torch.linalg.vector_norm(ref).clamp_min(1e-12))
        self.assertLess(relative, 1e-4)

    def test_two_pass_global_action_cache_preserves_complete_gradients(self):
        model, first = self.example()
        game = Game(from_rank(8), budget=12)
        game.step('choose:HALT', observe=False)
        second = game.observation()
        samples = []
        for observation in (first, second):
            with torch.inference_mode():
                logits = model(observation)[0]
                probs = logits.softmax(-1).tolist()
            coefficients = [0.] * len(observation['actions'])
            coefficients[0] = 1.5
            coefficients[-1] = -0.4
            groups = {encode_json(observation): (
                coefficients, {0: probs[0], len(probs)-1: probs[-1]})}
            samples += make_samples(groups, model, 0, 19)
        reference, reference_stats = gradient_trial(
            model, samples, mode='serial', batch_size=1, bucketed=False,
            denominator=8, epsilon=0.)
        two_pass, stats = replay_unique_action_gradients(
            model, samples, context_batch_size=2, action_batch_size=4,
            denominator=8, epsilon=0.)
        relative = float(torch.linalg.vector_norm(reference-two_pass) /
                         torch.linalg.vector_norm(reference).clamp_min(1e-12))
        self.assertLess(relative, 1e-4)
        self.assertAlmostEqual(reference_stats['loss'], stats['loss'], places=5)
        self.assertGreater(stats['action_references'], stats['unique_actions'])
        max_error, checked = two_pass_probability_disagreement(model, samples, 0.)
        self.assertLess(max_error, 1e-4)
        self.assertEqual(checked, 4)
        with cached_structural_forward(model, relation_cache_mb=4):
            cached, _ = replay_unique_action_gradients(
                model, samples, context_batch_size=2, action_batch_size=4,
                denominator=8, epsilon=0.)
        torch.testing.assert_close(two_pass, cached, rtol=0, atol=0)

    def test_rollout_matrix_cases_and_selection_are_deterministic(self):
        cases = build_cases(
            ['worker', 'central-cpu'], [4, 8], [8, 32], [4, 8], [0., 2.], 1)
        self.assertEqual(len(cases), 20)
        ids = [{'machine_id': str(i)} for i in range(20)]
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = select_machines(root, 2, {'train': ids}, 8, 'random', 123)
            second = select_machines(root, 2, {'train': ids}, 8, 'random', 123)
            self.assertEqual(first, second)
            self.assertEqual(len(set(r['machine_id'] for r in first)), 8)


if __name__ == '__main__':
    unittest.main()
