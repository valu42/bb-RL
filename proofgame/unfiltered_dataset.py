"""Reproducible, unfiltered two-state train/test dataset.

Choose distinct labelled transition tables uniformly without replacement from
the complete 12**4 population. No proof search, halting test or proof labels
are used to select machines.
"""
import argparse
import json
from pathlib import Path
import random

from .population_sample import from_rank
from .tasks import key

DATASET_FORMAT = "proofgame-unfiltered-two-state-1"
POPULATION = 12 ** 4


def build_dataset(train_count=1000, test_count=100, action_budget=100, seed=20261026,
                  fixed_test_ranks=None):
    if any(type(v) is not int for v in (train_count, test_count, action_budget, seed)):
        raise ValueError("counts, budget and seed must be integers")
    if min(train_count, test_count, action_budget) < 1:
        raise ValueError("train, test and budget must be positive")
    if train_count + test_count > POPULATION:
        raise ValueError("requested more distinct machines than exist")

    if fixed_test_ranks is None:
        ranks = random.Random(seed).sample(range(POPULATION), train_count + test_count)
        train_ranks, test_ranks = ranks[:train_count], ranks[train_count:]
    else:
        if (len(fixed_test_ranks) != test_count or
                any(type(r) is not int or not 0 <= r < POPULATION
                    for r in fixed_test_ranks) or
                len(set(fixed_test_ranks)) != test_count):
            raise ValueError('invalid fixed held-out machine ranks')
        excluded = set(fixed_test_ranks)
        # Keep the two separately reported challenge machines out of the
        # larger training sample too, otherwise 12,500 / 20,736 sampling
        # would frequently leak the parity trap or Busy Beaver benchmark.
        from itertools import product
        from .special_benchmarks import BENCHMARKS
        options = tuple(product(('q0','q1','qH'),('b0','b1'),('L','R')))
        digits = {row:i for i,row in enumerate(options)}
        benchmark_ranks = [
            sum(digits[tuple(row)] * (12**j) for j,row in enumerate(item['rows']))
            for item in BENCHMARKS
        ]
        excluded.update(benchmark_ranks)
        if train_count > POPULATION - len(excluded):
            raise ValueError('not enough machines outside held-out and benchmark sets')
        # Uniformly sample distinct training machines from the complement.
        candidates = [r for r in range(POPULATION) if r not in excluded]
        train_ranks = random.Random(seed).sample(candidates, train_count)
        test_ranks = list(fixed_test_ranks)

    def row(rank):
        machine = from_rank(rank, states=2)
        return {"rank": rank, "machine_id": key(machine), "machine": machine.to_json()}

    return {
        "format": DATASET_FORMAT,
        "population": POPULATION,
        "sampling": ("uniform_without_replacement_over_all_labelled_two_state_machines"
                     if fixed_test_ranks is None else
                     "uniform_without_replacement_excluding_fixed_test_and_benchmarks"),
        "sampling_seed": seed,
        "action_budget": action_budget,
        "train": [row(r) for r in train_ranks],
        "test": [row(r) for r in test_ranks],
        **({"fixed_test_ranks":test_ranks,
            "excluded_benchmark_ranks":benchmark_ranks}
           if fixed_test_ranks is not None else {}),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=1000, help="distinct training machines")
    p.add_argument("--test", type=int, default=100, help="distinct held-out machines")
    p.add_argument("--budget", type=int, default=100, help="maximum proof-game actions per attempt")
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--test-from", type=Path,
                   help="preserve exact held-out machines from an existing unfiltered dataset")
    p.add_argument("--out", type=Path, required=True, help="new dataset JSON path")
    args = p.parse_args()
    try:
        fixed_test = None
        if args.test_from is not None:
            previous = json.loads(args.test_from.read_text())
            if previous.get("format") != DATASET_FORMAT:
                raise ValueError('--test-from must be an unfiltered two-state dataset')
            fixed_test = [row['rank'] for row in previous['test']]
            if args.test != len(fixed_test):
                raise ValueError('--test must equal the existing held-out set size')
        dataset = build_dataset(args.train, args.test, args.budget, args.seed,
                                fixed_test_ranks=fixed_test)
    except ValueError as e:
        p.error(str(e))
    if args.out.exists():
        p.error("choose a new dataset file; refusing to overwrite an existing dataset")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dataset, indent=2) + "\n")
    print(json.dumps({
        "dataset": str(args.out), "population": POPULATION,
        "training_machines": len(dataset["train"]),
        "test_machines": len(dataset["test"]),
        "action_budget": dataset["action_budget"],
        "seed": dataset["sampling_seed"],
        "selection_filter": "none",
        "fixed_test_from": str(args.test_from) if args.test_from else None,
    }, indent=2))


if __name__ == "__main__":
    main()
