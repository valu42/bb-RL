"""Run independent machines concurrently with identical frozen network weights."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', type=int, default=8)
    p.add_argument('--attempts', type=int, default=10000)
    p.add_argument('--budget', type=int, default=20)
    p.add_argument('--sampling-seed', type=int, default=20261026)
    p.add_argument('--network-seed', type=int, default=7)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if not 1 <= args.sample <= 20736 or min(args.attempts, args.budget, args.workers) < 1:
        p.error('invalid experiment size')
    if args.out.exists(): p.error('choose a new output directory')
    args.out.mkdir(parents=True)
    rng = random.Random(args.sampling_seed)
    ranks = rng.sample(range(20736), args.sample)
    tasks = [(r, rng.getrandbits(64)) for r in ranks]
    plan = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.update(ranks_and_seeds=tasks, training_updates=0, proof_examples_supplied=0,
                initial_theorem_archive=0, sampling='uniform without replacement over all 20,736 labelled two-state machines')
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    start = time.monotonic()

    def run(index):
        out = args.out/f'task_{index}'
        cmd = [sys.executable, '-m', 'proofgame.network_search', '--sample', str(args.sample),
               '--attempts', str(args.attempts), '--budget', str(args.budget),
               '--sampling-seed', str(args.sampling_seed), '--network-seed', str(args.network_seed),
               '--task-index', str(index), '--threads', '1', '--out', str(out)]
        with (args.out/f'task_{index}.log').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((out/'report.json').read_text())
        record = json.loads((out/'records.jsonl').read_text())
        assert (record['rank'], record['action_seed']) == tasks[index]
        assert report['parameters_unchanged'] and report['sample'] == 1
        return index, report, record

    reports = []; records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool, (args.out/'records.jsonl').open('w') as log:
        for future in as_completed([pool.submit(run, i) for i in range(args.sample)]):
            index, report, record = future.result()
            reports.append(report); records.append(record)
            log.write(json.dumps(record)+'\n'); log.flush()
            print(json.dumps({'finished_task': index, 'completed_machines': len(records),
                              'proved': sum(bool(r['success']) for r in records)}), flush=True)
    result = {**plan, 'complete': True, 'proved': sum(bool(r['success']) for r in records),
              'proved_answers': dict(sum((Counter(r['proved_answers']) for r in reports), Counter())),
              'attempts_completed': sum(r['attempts_completed'] for r in records),
              'actions_simulated': sum(r['actions_simulated'] for r in records),
              'statuses': dict(sum((Counter(r['statuses']) for r in records), Counter())),
              'parameters_unchanged': all(r['parameters_unchanged'] for r in reports),
              'parameters': reports[0]['parameters'], 'model': reports[0]['model'],
              'elapsed_seconds': time.monotonic()-start}
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
