"""Benchmark must distinguish MPS first-use overhead from repeatable training speed."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from proofgame.benchmark_speed_v2 import summarize_learning_trials


def trial(cpu, mps, *, updated=True, parity=True):
    return {
        'cpu_serial_seconds': cpu,
        'two_pass_seconds': mps,
        'optimized': {'updated': updated},
        'passes_update_parity': parity,
    }


class OptimizerBenchReportingTests(unittest.TestCase):
    def test_actual_two_trial_regression_cannot_project_an_epoch(self):
        # From the Mac: same 369 gradient states, MPS initially 6x slower
        # than its second execution, while gradients passed parity.
        first = trial(1.539, 6.469)
        second = trial(1.479, 1.075)
        old = summarize_learning_trials([first, second])
        self.assertEqual(old['stability_status'], 'missing_explicit_warmup')
        self.assertIsNone(old['speedup_vs_cpu_serial'])
        self.assertAlmostEqual(old['median_ratio_candidate'], .400, places=2)
        self.assertLess(old['per_timed_trial_speedups'][0], .25)
        self.assertGreater(old['per_timed_trial_speedups'][1], 1.35)

    def test_explicit_warmup_and_three_stable_repeats_can_report_factor(self):
        cold = trial(1.539, 6.469)
        warmed = [
            trial(1.479, 1.075),
            trial(1.501, 1.090),
            trial(1.492, 1.080),
        ]
        report = summarize_learning_trials(warmed, warmups=[cold])
        self.assertEqual(report['stability_status'], 'stable_warmed_sample')
        self.assertAlmostEqual(
            report['speedup_vs_cpu_serial'], 1.492 / 1.080, places=5)
        self.assertEqual(report['initial_accelerated_seconds'], 6.469)
        self.assertEqual(report['warmup_trials'], 1)
        self.assertEqual(report['timed_trials'], 3)

    def test_missing_or_unstable_trials_suppress_speedup(self):
        warm = [trial(1.5, 6.)]
        two = [trial(1.5, 1.1), trial(1.49, 1.08)]
        report = summarize_learning_trials(two, warmups=warm)
        self.assertEqual(report['stability_status'], 'too_few_timed_trials')
        self.assertIsNone(report['speedup_vs_cpu_serial'])
        unstable = summarize_learning_trials(
            two + [trial(1.5, 4.)], warmups=warm)
        self.assertEqual(unstable['stability_status'], 'timings_unstable')
        self.assertIsNone(unstable['speedup_vs_cpu_serial'])
        mismatch = summarize_learning_trials(
            two + [trial(1.5, 1.07, parity=False)], warmups=warm)
        self.assertEqual(
            mismatch['stability_status'], 'numerical_or_optimizer_mismatch')
        self.assertIsNone(mismatch['speedup_vs_cpu_serial'])
        self.assertIsNone(summarize_learning_trials([]))
        with self.assertRaises(ValueError):
            summarize_learning_trials([], max_duration_spread=.9)

    def test_inspect_existing_json_does_not_run_benchmark(self):
        with TemporaryDirectory() as tmp:
            file = Path(tmp) / 'old_speed.json'
            file.write_text(json.dumps({
                'learning_trials': [trial(1.539, 6.469), trial(1.479, 1.075)],
                'epoch_projection': {'projected_epoch_minutes': 152.43},
            }))
            process = subprocess.run(
                [sys.executable, '-m', 'proofgame.benchmark_speed_v2',
                 '--inspect-existing', str(file)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            result = json.loads(process.stdout)
            report = result['optimizer_reevaluation']
            self.assertEqual(report['stability_status'],
                             'missing_explicit_warmup')
            self.assertIsNone(report['speedup_vs_cpu_serial'])
            self.assertEqual(
                result['previous_epoch_projection']['projected_epoch_minutes'],
                152.43)
            self.assertIn('before startup/steady-state', result[
                'previous_projection_warning'])


if __name__ == '__main__':
    unittest.main()
