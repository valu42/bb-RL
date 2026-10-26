"""Repeated proof search using a fresh, frozen, randomly initialized policy.

Cache only complete observation-to-probability computations. The cache includes
the machine, remaining budget, error, proof state, and full ordered action menu.
It supplies no proof examples or lemmas and never changes the policy weights.
"""
import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import random
import time

import torch

from .environment import Game
from .exploration import answer_indices, mix_probabilities
from .kernel import check
from .model import Policy, ModelLimit
from .population_sample import from_rank
from .tasks import key


def encode_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class FrozenSampler:
    def __init__(self, policy, capacity=50000):
        self.policy = policy.eval()
        self.capacity = capacity
        self.cache = OrderedDict()
        self.action_features = {}
        self.hits = self.misses = 0

    @torch.inference_mode()
    def probabilities(self, observation):
        full_key = encode_json(observation)
        if full_key in self.cache:
            self.hits += 1
            result = self.cache.pop(full_key)
            self.cache[full_key] = result
            return result
        self.misses += 1
        actions = [encode_json(a) for a in observation["actions"]]
        missing = list(dict.fromkeys(a for a in actions if a not in self.action_features))
        if missing:
            embeddings = self.policy.encode(missing)
            keys = self.policy.key(embeddings)
            biases = self.policy.action_bias(embeddings).squeeze(-1)
            for a, k, b in zip(missing, keys, biases):
                self.action_features[a] = (k.clone(), b.clone())
        context = {k: v for k, v in observation.items() if k != "actions"}
        h = self.policy.encode([encode_json(context)])[0]
        keys = torch.stack([self.action_features[a][0] for a in actions])
        biases = torch.stack([self.action_features[a][1] for a in actions])
        logits = (keys * self.policy.query(h)).sum(-1) / h.shape[-1] ** 0.5 + biases
        if not bool(torch.isfinite(logits).all()):
            raise RuntimeError(
                'non-finite policy logits during CPU rollout; inspect model weights and '
                'the last training update (the current checkpoint may be corrupted)')
        probabilities = logits.float().softmax(-1)
        if (not bool(torch.isfinite(probabilities).all()) or
                not bool((probabilities >= 0).all()) or
                not bool(probabilities.sum() > 0)):
            raise RuntimeError('invalid policy probabilities during CPU rollout')
        result = tuple(probabilities.tolist())
        if self.capacity:
            self.cache[full_key] = result
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return result


def attempt(sampler, machine, rng, budget=20, exploration_epsilon=0., on_initial_choice=None,
            forced_initial_answer=None, diagnostic_steps=None):
    game = Game(machine, budget=budget)
    trace = []
    while game.status == "ACTIVE":
        observation = game.observation()
        try:
            probabilities = sampler.probabilities(observation)
        except ModelLimit:
            return None, trace, "MODEL_LIMIT"
        behavior = mix_probabilities(observation, probabilities, exploration_epsilon)
        forced = not trace and forced_initial_answer is not None
        if forced:
            if forced_initial_answer not in ('HALT','NONHALT'):
                raise ValueError('invalid forced initial answer')
            indices = answer_indices(observation)
            if indices is None:
                raise RuntimeError('initial answer choices missing from proof game')
            index = indices[0 if forced_initial_answer == 'HALT' else 1]
            # Conditional training: the scheduled initial action has probability
            # one, and therefore no policy-gradient contribution.
            behavior = tuple(1. if j == index else 0. for j in range(len(probabilities)))
        else:
            index = rng.choices(range(len(behavior)), weights=behavior, k=1)[0]
        action = observation["actions"][index]["id"]
        if not trace and on_initial_choice is not None:
            branch_indices = answer_indices(observation)
            raw = {'HALT': probabilities[branch_indices[0]],
                   'NONHALT': probabilities[branch_indices[1]]} if branch_indices else {}
            mixed = {'HALT': behavior[branch_indices[0]],
                     'NONHALT': behavior[branch_indices[1]]} if branch_indices else {}
            on_initial_choice(action, raw, mixed)
        step = {"action": action, "probability": behavior[index]}
        if forced:
            step['forced_initial_answer'] = forced_initial_answer
        trace.append(step)
        if diagnostic_steps is not None:
            # Capture only a compact human-readable view, not the entire
            # observation tensor or all possible actions.
            top = sorted(range(len(behavior)), key=lambda j: behavior[j], reverse=True)[:3]
            detail = {
                'step': len(trace), 'action': action,
                'probability': float(behavior[index]),
                'raw_policy_probability': float(probabilities[index]),
                'forced': bool(forced),
                'alternatives': [
                    {'action': observation['actions'][j]['id'],
                     'probability': float(behavior[j])} for j in top
                ],
                'actions_left_before': observation['actions_left'],
                'goal': [g['goal'] for g in observation['goals'][:2]],
                'goal_count': len(observation['goals']),
                'draft': observation.get('draft'),
                'answer': observation['answer'],
            }
        game.step(action, observe=False)
        if diagnostic_steps is not None:
            detail['status_after'] = game.status
            detail['error_after'] = game.error
            diagnostic_steps.append(detail)
    if game.status == "SOLVED":
        certificate = game.certificate()
        check(certificate)
        return certificate, trace, game.status
    return None, trace, game.status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=10000)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--sampling-seed", type=int, default=20261026)
    parser.add_argument("--network-seed", type=int, default=7)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache-states", type=int, default=50000)
    parser.add_argument("--task-index", type=int, help="run just this index of the fixed sampled task list")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.sample <= 20736 or min(args.attempts, args.budget, args.threads) < 1:
        parser.error("invalid experiment size")
    if args.task_index is not None and not 0 <= args.task_index < args.sample:
        parser.error("task index outside sampled task list")
    if args.out.exists():
        parser.error("choose a new output directory")
    args.out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.network_seed)
    policy = Policy(width=args.width, layers=args.layers)
    initial = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    torch.save({"config": policy.config, "state_dict": initial, "seed": args.network_seed},
               args.out / "initial_model.pt")
    rng = random.Random(args.sampling_seed)
    ranks = rng.sample(range(20736), args.sample)
    tasks = [(rank, rng.getrandbits(64)) for rank in ranks]
    if args.task_index is not None:
        tasks = [tasks[args.task_index]]
    plan = {"sample": len(tasks), "full_sample_size": args.sample, "task_index": args.task_index,
            "attempt_limit": args.attempts, "action_budget": args.budget,
            "sampling_seed": args.sampling_seed, "network_seed": args.network_seed,
            "ranks_and_seeds": tasks, "model": policy.config,
            "parameters": sum(p.numel() for p in policy.parameters()),
            "torch": torch.__version__, "training_updates": 0,
            "proof_examples_supplied": 0, "initial_theorem_archive": 0,
            "policy": "softmax of frozen network logits; temperature 1; no uniform mixture",
            "sampling": "uniform without replacement over all 20,736 labelled two-state machines"}
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sampler = FrozenSampler(policy, args.cache_states)
    records = []
    start = time.monotonic()
    with (args.out / "records.jsonl").open("w") as log:
        for machine_index, (rank, seed) in enumerate(tasks):
            machine = from_rank(rank)
            action_rng = random.Random(seed)
            statuses = Counter()
            success = None
            action_count = 0
            for i in range(args.attempts):
                cert, trace, status = attempt(sampler, machine, action_rng, args.budget)
                statuses[status] += 1
                action_count += len(trace)
                if cert:
                    # Replay the exact actions through a fresh game before recording success.
                    replay = Game(machine, budget=args.budget)
                    for row in trace:
                        replay.step(row["action"], observe=False)
                    assert replay.status == "SOLVED" and replay.certificate() == cert
                    check(cert)
                    success = {"attempt": i + 1, "actions": trace, "certificate": cert}
                if (i + 1) % 100 == 0 or cert or i + 1 == args.attempts:
                    progress = {"completed_machines": len(records), "current_machine": machine_index + 1,
                                "rank": rank, "attempts_on_current": i + 1,
                                "proofs": sum(bool(r["success"]) for r in records) + int(bool(cert)),
                                "cache_hits": sampler.hits, "network_evaluations": sampler.misses,
                                "elapsed_seconds": time.monotonic() - start}
                    (args.out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                    print(json.dumps(progress), flush=True)
                if cert:
                    break
            record = {"rank": rank, "machine": machine.to_json(), "machine_id": key(machine),
                      "action_seed": seed, "attempts_completed": i + 1,
                      "actions_simulated": action_count, "statuses": dict(statuses), "success": success}
            records.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
    unchanged = all(torch.equal(initial[k], v) for k, v in policy.state_dict().items())
    assert unchanged
    report = {**plan, "complete": True, "proved": sum(bool(r["success"]) for r in records),
              "proved_answers": dict(Counter(r["success"]["certificate"]["answer"] for r in records if r["success"])),
              "attempts_completed": sum(r["attempts_completed"] for r in records),
              "actions_simulated": sum(r["actions_simulated"] for r in records),
              "statuses": dict(sum((Counter(r["statuses"]) for r in records), Counter())),
              "parameters_unchanged": unchanged, "cache_hits": sampler.hits,
              "network_evaluations": sampler.misses, "elapsed_seconds": time.monotonic() - start}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
