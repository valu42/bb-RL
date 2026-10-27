"""CPU speed-lab experiments must preserve frozen-model values and gradients."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from proofgame.benchmark_real_gradients import (
    attention_elements, gradient_trial, make_samples, probability_disagreement,
    select_machines)
from proofgame.benchmark_rollout_matrix import (
    build_cases, run_case, task_for, worker_task_for)
from proofgame.experimental_packed_batch import packed_score_many
from proofgame.experimental_action_replay import (
    replay_unique_action_gradients, two_pass_probability_disagreement)
from proofgame.experimental_structural_cache import (
    StructuralPreprocessingCache, cached_structural_forward)
from proofgame.environment import Game
from proofgame.model import Policy
from proofgame.network_search import encode_json
from proofgame.population_sample import from_rank
from proofgame.unfiltered_dataset import build_dataset


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

    def test_packed_action_encoding_and_deferred_checks_preserve_gradients(self):
        model, first = self.example()
        game = Game(from_rank(8), budget=12)
        game.step('choose:HALT', observe=False)
        second = game.observation()
        observations = [first, second]
        original = model.score_many(observations)
        packed = packed_score_many(
            model, observations, max_action_batch_size=2,
            max_attention_elements=1000000)
        for a, b in zip(original, packed):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        samples = []
        for i, obs in enumerate(observations):
            with torch.no_grad():
                probs = model(obs)[0].softmax(-1).tolist()
            counts = [0.] * len(obs['actions'])
            counts[0] = 1. if i == 0 else -0.8
            sample = make_samples(
                {encode_json(obs): (counts, {0: probs[0]})}, model, 0, 91)
            samples.extend(sample)
        ref, ref_stats = gradient_trial(
            model, samples, mode='serial', batch_size=1,
            bucketed=False, denominator=8, epsilon=0.)
        ref_deferred, deferred_stats = gradient_trial(
            model, samples, mode='serial', batch_size=1,
            bucketed=False, denominator=8, epsilon=0., defer_checks=True)
        torch.testing.assert_close(ref_deferred, ref, atol=1e-7, rtol=1e-5)
        self.assertAlmostEqual(ref_stats['loss'], deferred_stats['loss'], places=5)
        variant, variant_stats = gradient_trial(
            model, samples, mode='packed-score-many', batch_size=2,
            bucketed=True, denominator=8, epsilon=0., defer_checks=True,
            packed_action_batch_size=2, max_attention_elements=1000000)
        difference = float(torch.linalg.vector_norm(variant-ref) /
                           torch.linalg.vector_norm(ref).clamp_min(1e-12))
        self.assertLess(difference, 1e-4)
        self.assertAlmostEqual(ref_stats['loss'], variant_stats['loss'], places=5)
        plain, _ = replay_unique_action_gradients(
            model, samples, context_batch_size=2, action_batch_size=2,
            denominator=8, epsilon=0.)
        deferred, _ = replay_unique_action_gradients(
            model, samples, context_batch_size=2, action_batch_size=2,
            denominator=8, epsilon=0., defer_checks=True)
        torch.testing.assert_close(plain, deferred, atol=1e-7, rtol=1e-5)
        max_error, checked = probability_disagreement(
            model, samples, 'packed-score-many', 2, True, 0.,
            packed_action_batch_size=2, max_attention_elements=1000000)
        self.assertEqual(checked, 2)
        self.assertLess(max_error, 1e-5)

    def test_topology_cache_reuses_structure_with_changed_token_values(self):
        model, first = self.example()
        # Two distinct observations whose JSON numeric values differ, but
        # whose AST parent/binder structure is identical.
        second = {**first, 'actions_left': first['actions_left'] - 1}
        first_context = encode_json({k: v for k, v in first.items()
                                     if k != 'actions'})
        second_context = encode_json({k: v for k, v in second.items()
                                      if k != 'actions'})
        self.assertNotEqual(first_context, second_context)
        texts = [first_context, second_context]
        model.zero_grad(set_to_none=True)
        expected = model.encode(texts)
        expected.sum().backward()
        original = [p.grad.detach().clone() if p.grad is not None else None
                    for p in model.parameters()]
        state_keys = list(model.state_dict())
        model.structural_encoder.enable_input_cache(
            relation_cache_mb=1, position_cache_mb=1)
        model.zero_grad(set_to_none=True)
        actual = model.encode(texts)
        actual.sum().backward()
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(state_keys, list(model.state_dict()))
        for prior, p in zip(original, model.parameters()):
            if prior is not None:
                torch.testing.assert_close(prior, p.grad, rtol=0, atol=0)
        stats = model.structural_encoder._input_cache.stats()
        self.assertTrue(stats['topology_keys'])
        self.assertTrue(stats['compact_relations'])
        self.assertEqual(stats['relation_entries'], 1)
        self.assertGreaterEqual(stats['hits']['relation'], 1)
        self.assertLessEqual(stats['relation_cache_bytes'], 1024**2)
        model.structural_encoder.enable_input_cache(relation_cache_mb=0)
        self.assertIsNone(model.structural_encoder._input_cache)

    def test_rollout_frozen_version_barrier_changes_without_trajectory_drift(self):
        model, _ = self.example()
        data = build_dataset(4, 1, 4, 23)
        rows = data['train'][:4]
        config = {
            'seed': 23, 'benchmark_epoch': 1,
            'attempts_per_machine': 2, 'budget': 4,
            'train_exploration_epsilon': 0., 'group_relative': False,
            'balanced_initial_answers': False, 'branch_rounds': 0,
        }
        sample = task_for(rows[0], config, 1, central=True, version=5)
        self.assertEqual(sample[4], 5)
        remote = worker_task_for(rows[0], config, 1, '/tmp/model.pt', version=7)
        self.assertEqual(remote[5], 7)
        case = {
            'strategy': 'worker-shared', 'workers': 2, 'barrier': 2,
            'inference_batch': None, 'wait_ms': None, 'repeat': 0,
        }
        with TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'weights.pt'
            torch.save({'config': model.config, 'state_dict': model.state_dict()},
                       checkpoint)
            frozen = run_case(
                case, config, rows, checkpoint,
                simulate_update_barriers=False)
            versioned = run_case(
                case, config, rows, checkpoint,
                simulate_update_barriers=True)
        self.assertEqual(frozen['status'], 'ok')
        self.assertEqual(versioned['status'], 'ok')
        self.assertEqual(frozen['frozen_weight_versions'], 1)
        self.assertEqual(versioned['frozen_weight_versions'], 2)
        self.assertEqual(frozen['trajectory_hash_by_machine'],
                         versioned['trajectory_hash_by_machine'])
        self.assertEqual(frozen['actions'], versioned['actions'])
        self.assertEqual(frozen['attempts'], versioned['attempts'])

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
