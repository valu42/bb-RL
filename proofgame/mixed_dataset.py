"""Fixed, disjoint random datasets with equal shares of 2-, 3-, and 4-state machines.

Sample labelled transition tables without any behavior/provability filtering.
The existing separate special benchmarks are excluded from both splits.
"""
import argparse
import json
from pathlib import Path
import random

from .logic import Machine
from .population_sample import from_rank
from .special_benchmarks import BENCHMARKS
from .tasks import key

DATASET_FORMAT = 'proofgame-unfiltered-mixed-states-1'
STATES = (2, 3, 4)


def population(states):
    return (4 * (states + 1)) ** (2 * states)


def split_counts(total):
    return {str(states): total // 3 + int(index < total % 3)
            for index, states in enumerate(STATES)}


def build_dataset(train_count=1000, test_count=300, action_budget=100, seed=20261028):
    if any(type(v) is not int for v in (train_count, test_count, action_budget, seed)):
        raise ValueError('counts, budget and seed must be integers')
    if min(train_count, test_count) < 3 or action_budget < 1:
        raise ValueError('each split needs at least three machines and a positive budget')
    train_counts, test_counts = split_counts(train_count), split_counts(test_count)
    excluded = {key(Machine(item['rows'])) for item in BENCHMARKS}
    rng = random.Random(seed)
    splits = {'train': [], 'test': []}
    for states in STATES:
        n_train, n_test = train_counts[str(states)], test_counts[str(states)]
        size = population(states)
        excluded_count = sum(len(item['rows']) == 2 * states for item in BENCHMARKS)
        if n_train + n_test > size - excluded_count:
            raise ValueError(f'too many distinct {states}-state machines requested')
        # Sample integer ranks directly; never enumerate the 25.6 billion
        # four-state tables. Rejection of duplicates/exclusions is symmetric.
        selected, seen = [], set()
        while len(selected) < n_train + n_test:
            rank = rng.randrange(size)
            if rank in seen:
                continue
            seen.add(rank)
            machine = from_rank(rank, states=states)
            machine_id = key(machine)
            if machine_id in excluded:
                continue
            selected.append({'rank': rank, 'machine_id': machine_id,
                             'machine': machine.to_json()})
        splits['train'].extend(selected[:n_train])
        splits['test'].extend(selected[n_train:])
    for rows in splits.values():
        rng.shuffle(rows)
    return {
        'format': DATASET_FORMAT, 'states': list(STATES),
        'sampling': 'uniform_labelled_tables_without_replacement_per_size_excluding_benchmarks',
        'sampling_seed': seed, 'action_budget': action_budget,
        'population_by_states': {str(n): population(n) for n in STATES},
        'train_counts_by_states': train_counts, 'test_counts_by_states': test_counts,
        'excluded_benchmark_ids': sorted(excluded), **splits,
    }


def validate_dataset(dataset):
    """Reproduce sampling, identities, ranks, split membership and metadata."""
    try:
        expected = build_dataset(len(dataset['train']), len(dataset['test']),
                                 dataset['action_budget'], dataset['sampling_seed'])
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid mixed-state dataset metadata') from exc
    # Machine.to_json() retains tuples in memory; files contain JSON arrays.
    if json.dumps(dataset, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('mixed-state dataset does not match its declared sampling seed and counts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=int, default=1000)
    parser.add_argument('--test', type=int, default=300)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        dataset = build_dataset(args.train, args.test, args.budget, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            stream.write(json.dumps(dataset, indent=2) + '\n')
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({'dataset': str(args.out),
                      'train': dataset['train_counts_by_states'],
                      'test': dataset['test_counts_by_states'],
                      'action_budget': args.budget}, indent=2))


if __name__ == '__main__':
    main()
