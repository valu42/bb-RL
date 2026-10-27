"""Stage-aware speed projections must not confuse wall time with worker CPU sums."""
import unittest

from proofgame.speed_estimates import stage_projection


class SpeedProjectionTests(unittest.TestCase):
    def setUp(self):
        self.history = {
            'stage_wall_seconds': {
                'rollout': 120.,
                'learn': 240.,
                'worker_snapshot': 15.,
                'checkpoint': 5.,
            },
            'evaluation_seconds': 60.,
            'machine_task_seconds': 1000.,  # parallel CPU seconds, NOT wall
        }

    def test_stage_arithmetic_and_required_speedups(self):
        result = stage_projection(
            self.history, rollout_speedup=2.,
            learning_speedup=4., evaluation_speedup=2.,
            target_minutes=3.)
        self.assertAlmostEqual(result['historical']['full_epoch_seconds'], 440.)
        self.assertAlmostEqual(result['projected_epoch_seconds'], 170.)
        self.assertAlmostEqual(result['projected_full_epoch_speedup'], 440. / 170.)
        self.assertTrue(result['target_met_by_projection'])
        self.assertAlmostEqual(
            result['additional_rollout_speedup_needed_at_given_other_factors'],
            120. / 70.)
        self.assertAlmostEqual(
            result['additional_learning_speedup_needed_at_given_other_factors'],
            240. / 70.)
        self.assertEqual(result['projected_stages_seconds']['other_training_seconds'], 20.)

    def test_unreachable_target_is_reported_not_invented(self):
        result = stage_projection(self.history, target_minutes=.3)
        self.assertIsNone(result[
            'additional_rollout_speedup_needed_at_given_other_factors'])
        self.assertIsNone(result[
            'additional_learning_speedup_needed_at_given_other_factors'])
        self.assertFalse(result['target_met_by_projection'])

    def test_invalid_inputs_and_missing_eval(self):
        for factor in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                stage_projection(self.history, rollout_speedup=factor)
        with self.assertRaises(ValueError):
            stage_projection({'stage_wall_seconds':self.history[
                'stage_wall_seconds'], 'evaluation_seconds':None})


if __name__ == '__main__':
    unittest.main()
