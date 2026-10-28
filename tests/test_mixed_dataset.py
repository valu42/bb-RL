import copy
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from proofgame.mixed_dataset import build_dataset, population, validate_dataset
from proofgame.population_sample import from_rank
from proofgame.proof_progress import ProofProgress


class MixedDatasetTests(unittest.TestCase):
    def test_reproducible_disjoint_balanced_unlabelled_tables(self):
        dataset = build_dataset()
        self.assertEqual(dataset, build_dataset())
        self.assertNotEqual(dataset['train'], build_dataset(seed=19)['train'])
        validate_dataset(dataset)
        validate_dataset(json.loads(json.dumps(dataset)))
        self.assertEqual(population(4), 25600000000)
        self.assertEqual(dataset['train_counts_by_states'], {'2':334, '3':333, '4':333})
        self.assertEqual(dataset['test_counts_by_states'], {'2':100, '3':100, '4':100})
        rows = dataset['train'] + dataset['test']
        self.assertEqual(len({r['machine_id'] for r in rows}), 1300)
        self.assertFalse(set(dataset['excluded_benchmark_ids']) & {r['machine_id'] for r in rows})
        for split in ('train', 'test'):
            counts = Counter()
            for row in dataset[split]:
                self.assertEqual(set(row), {'rank', 'machine_id', 'machine'})
                states = len(row['machine']['rows']) // 2
                counts[str(states)] += 1
                self.assertEqual(from_rank(row['rank'], states).to_json(), row['machine'])
            self.assertEqual(counts, dataset[split + '_counts_by_states'])

    def test_rejects_corrupt_metadata_tables_and_overlap(self):
        original = build_dataset(6, 3, 8, 17)
        for field, value in [('sampling_seed', 99), ('action_budget', 0),
                             ('train_counts_by_states', {'2':6}), ('states', [2,3])]:
            bad = copy.deepcopy(original)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_dataset(bad)
        bad = copy.deepcopy(original)
        bad['train'][0] = bad['test'][0]
        with self.assertRaises(ValueError):
            validate_dataset(bad)
        for args in [(2, 3, 10), (3, 2, 10), (3, 3, 0), (70000, 3, 10)]:
            with self.assertRaises(ValueError):
                build_dataset(*args)

    def test_per_size_proof_means_survive_resume(self):
        rows = []
        for states, answer, length in [(2, 'HALT', 7), (3, 'NONHALT', 9), (4, 'HALT', 11)]:
            rows.append({'machine': from_rank(8, states).to_json(), 'successes': [
                {'attempt': 1, 'certificate': {'answer': answer}, 'actions': ['x']*length}]})
        rows.append({'machine': from_rank(10, 4).to_json(), 'successes': []})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'batch_0000.json').write_text(json.dumps({'start_index':0, 'machines':rows}))
            summary = ProofProgress.restore(folder, 4)
            groups = summary.snapshot()['by_states']
            self.assertEqual(groups['4']['machines'], 2)
            self.assertEqual(groups['4']['machines_unproved'], 1)
            self.assertEqual(groups['4']['mean_first_proof_actions_by_answer']['HALT'], 11)
            self.assertIsNone(groups['3']['mean_first_proof_actions_by_answer']['HALT'])
            self.assertEqual(len(summary.state_lines()), 3)

    def test_mixed_cli_training_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            dataset = folder/'dataset.json'
            generate = [sys.executable, '-m', 'proofgame.mixed_dataset', '--train', '3',
                        '--test', '3', '--budget', '8', '--out', str(dataset)]
            created = subprocess.run(generate, capture_output=True, text=True, timeout=30)
            self.assertEqual(created.returncode, 0, created.stderr)
            original = dataset.read_bytes()
            repeated = subprocess.run(generate, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(dataset.read_bytes(), original)
            out = folder/'run'
            command = [sys.executable, '-m', 'proofgame.epoch_train', '--dataset', str(dataset),
                       '--out', str(out), '--epochs', '1', '--attempts', '2', '--eval-attempts', '2',
                       '--batch-size', '3', '--workers', '1', '--representation', 'ast',
                       '--width', '64', '--layers', '2', '--log-mode', 'compact',
                       '--balanced-initial-answers', '--group-relative', '--branch-rounds', '1']
            result = subprocess.run(command, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for states in (2, 3, 4):
                self.assertIn(f'{states}-state (1 machines)', result.stdout)
            for phase in ('training', 'evaluation'):
                metrics = json.loads((out/f'epoch_01_{phase}.json').read_text())
                self.assertEqual(set(metrics['proof_progress']['by_states']), {'2', '3', '4'})
                self.assertNotIn('MODEL_LIMIT', metrics['statuses'])
            resumed = subprocess.run(command + ['--resume', '--epochs', '2'],
                                     capture_output=True, text=True, timeout=120)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertTrue((out/'model_epoch_02.pt').exists())
            config = json.loads((out/'config.json').read_text())
            self.assertIn('unfiltered', config['evaluation'])


if __name__ == '__main__':
    unittest.main()
