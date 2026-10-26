import unittest

from proofgame.population_sample import from_rank
from proofgame.tasks import key
from proofgame.unfiltered_dataset import (
    DATASET_FORMAT, POPULATION, build_dataset
)


class UnfilteredDatasetTests(unittest.TestCase):
    def test_full_size_dataset_is_disjoint_reproducible_and_unfiltered(self):
        dataset = build_dataset(train_count=1000, test_count=100, action_budget=100, seed=42)
        self.assertEqual(dataset['format'], DATASET_FORMAT)
        self.assertEqual(dataset['action_budget'], 100)
        self.assertEqual(len(dataset['train']), 1000)
        self.assertEqual(len(dataset['test']), 100)
        ranks = [row['rank'] for row in dataset['train'] + dataset['test']]
        self.assertEqual(len(ranks), len(set(ranks)))
        self.assertTrue(all(0 <= rank < POPULATION for rank in ranks))
        self.assertEqual(build_dataset(1000, 100, 100, 42), dataset)
        self.assertNotEqual(build_dataset(1000, 100, 100, 43), dataset)
        self.assertNotIn('filter_attempt_limit', dataset)
        for row in dataset['train'] + dataset['test']:
            self.assertEqual(set(row), {'rank', 'machine_id', 'machine'})
            m = from_rank(row['rank'], states=2)
            self.assertEqual(row['machine'], m.to_json())
            self.assertEqual(row['machine_id'], key(m))
            self.assertEqual(len(row['machine']['rows']), 4)

    def test_small_and_invalid_samples(self):
        self.assertEqual(len(build_dataset(1, 1)['train']), 1)
        for train, test, budget in ((0, 1, 100), (1, 0, 100),
                                    (1, 1, 0), (POPULATION, 1, 100)):
            with self.subTest(train=train, test=test, budget=budget):
                with self.assertRaises(ValueError):
                    build_dataset(train, test, budget)


if __name__ == '__main__':
    unittest.main()
