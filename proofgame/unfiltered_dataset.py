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


def build_dataset(train_count=1000, test_count=100, action_budget=100, seed=20261026):
    if any(type(v) is not int for v in (train_count, test_count, action_budget, seed)):
        raise ValueError("counts, budget and seed must be integers")
    if min(train_count, test_count, action_budget) < 1:
        raise ValueError("train, test and budget must be positive")
    if train_count + test_count > POPULATION:
        raise ValueError("requested more distinct machines than exist")

    ranks = random.Random(seed).sample(range(POPULATION), train_count + test_count)

    def row(rank):
        machine = from_rank(rank, states=2)
        return {"rank": rank, "machine_id": key(machine), "machine": machine.to_json()}

    return {
        "format": DATASET_FORMAT,
        "population": POPULATION,
        "sampling": "uniform_without_replacement_over_all_labelled_two_state_machines",
        "sampling_seed": seed,
        "action_budget": action_budget,
        "train": [row(r) for r in ranks[:train_count]],
        "test": [row(r) for r in ranks[train_count:]],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=int, default=1000, help="distinct training machines")
    p.add_argument("--test", type=int, default=100, help="distinct held-out machines")
    p.add_argument("--budget", type=int, default=100, help="maximum proof-game actions per attempt")
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--out", type=Path, required=True, help="new dataset JSON path")
    args = p.parse_args()
    try:
        dataset = build_dataset(args.train, args.test, args.budget, args.seed)
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
    }, indent=2))


if __name__ == "__main__":
    main()
