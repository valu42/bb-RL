import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from proofgame.epoch_train import learn_batch
from proofgame.model import Policy
from proofgame.proof_progress import ProofProgress
from proofgame.unfiltered_dataset import build_dataset
from tests.test_branch_rounds import winning_group


class ModelCapacityLoggingTests(unittest.TestCase):
    def test_machine_weighted_first_proofs_and_resume(self):
        def proof(answer, attempt, length):
            return {'certificate': {'answer': answer}, 'attempt': attempt,
                    'actions': ['x'] * length}
        rows = [
            {'successes': [proof('HALT', 3, 2), proof('HALT', 1, 10)]},
            {'successes': [proof('HALT', 1, 20)]},
            {'successes': [proof('NONHALT', 2, 6)]},
            {'successes': []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'batch_0000.json').write_text(json.dumps(
                {'start_index': 0, 'machines': rows[:2]}))
            # Written before checkpoint, so this must not count on restore.
            (folder/'batch_0002.json').write_text(json.dumps(
                {'start_index': 2, 'machines': rows[2:]}))
            restored = ProofProgress.restore(folder, 2)
            self.assertEqual(restored.completed, 2)
            restored.add(rows[2:])
            fresh = ProofProgress()
            self.assertIn('avg n/a', fresh.text())
            fresh.add(rows)
            self.assertEqual(restored.snapshot(), fresh.snapshot())
            self.assertEqual(fresh.snapshot()['machines_proved_by_answer'],
                             {'HALT': 2, 'NONHALT': 1})
            self.assertEqual(fresh.snapshot()['mean_first_proof_actions_by_answer'],
                             {'HALT': 15., 'NONHALT': 6.})
            self.assertEqual(fresh.snapshot()['machines_unproved'], 1)

    def test_wider_and_deeper_models_learn_verified_proofs(self):
        torch.set_num_threads(1)
        for layers, backend in [(1, 'root-query'), (2, 'full')]:
            with self.subTest(layers=layers):
                torch.manual_seed(7)
                model = Policy(width=64, layers=layers, representation='ast',
                               encoder_backend=backend).eval()
                rows = [winning_group(model)]
                before = {k: v.clone() for k, v in model.state_dict().items()}
                result = learn_batch(model, torch.optim.AdamW(model.parameters(), lr=3e-4),
                                     rows, 12, group_relative=True, branch_rounds=6,
                                     balanced_initial_answers=True, length_penalty=.3,
                                     gradient_backend='two-pass', gpu_gradient_batch_size=8)
                self.assertTrue(result['updated'])
                self.assertTrue(any(not torch.equal(v, before[k])
                                    for k, v in model.state_dict().items()))
                self.assertTrue(all(torch.isfinite(v).all() for v in model.state_dict().values()))

    def test_compact_cli_profiles_and_resumes_and_rejects_architecture_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root/'dataset.json'
            dataset.write_text(json.dumps(build_dataset(2, 1, 8, 17)))
            out = root/'run'
            cmd = [sys.executable, '-m', 'proofgame.epoch_train', '--dataset', str(dataset),
                   '--out', str(out), '--epochs', '1', '--attempts', '2', '--eval-attempts', '2',
                   '--batch-size', '1', '--workers', '1', '--representation', 'ast',
                   '--width', '64', '--layers', '2', '--log-mode', 'compact',
                   '--group-relative', '--balanced-initial-answers', '--branch-rounds', '1']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('encoder full', result.stdout)
            self.assertIn('HALT ', result.stderr)
            self.assertIn('NONHALT ', result.stderr)
            self.assertIn('proof actions', result.stderr)
            self.assertIn('train ETA', result.stderr)
            self.assertNotIn('"event":', result.stdout)
            config = json.loads((out/'config.json').read_text())
            self.assertEqual(config['model']['width'], 64)
            self.assertEqual(config['model']['layers'], 2)
            progress = json.loads((out/'progress.json').read_text())
            self.assertIn('proof_progress', progress)
            self.assertTrue(list((out/'profiling').glob('*train_batch*.json')))
            self.assertTrue((out/'epoch_01_training.json').exists())
            self.assertTrue((out/'special_benchmarks_epoch_01.json').exists())
            # Verbosity can change on resume, architecture cannot.
            resumed = subprocess.run(cmd + ['--resume', '--log-mode', 'detailed'],
                                     capture_output=True, text=True, timeout=60)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            wrong = subprocess.run(cmd + ['--resume', '--width', '32'],
                                   capture_output=True, text=True, timeout=60)
            self.assertNotEqual(wrong.returncode, 0)
            self.assertIn('resume configuration changed: model', wrong.stderr)
            invalid = subprocess.run(cmd + ['--encoder-backend', 'root-query'],
                                     capture_output=True, text=True, timeout=60)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn('root-query requires', invalid.stderr)


if __name__ == '__main__':
    unittest.main()
