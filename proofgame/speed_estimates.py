"""Conservative, stage-aware epoch speedup calculator.

No code is benchmarked here. The numbers are only a projection based on the
completed epoch's *measured* stage wall times and explicitly supplied speedup
factors. Worker CPU-second sums must not be added to elapsed wall time.

Example:
  python -m proofgame.speed_estimates \
    --run runs/ast_branch_race_8x6_fixed --epoch 2 \
    --rollout-speedup 1.5 --learning-speedup 2.0 \
    --evaluation-speedup 1.0 \
    --out runs/ast_branch_race_8x6_fixed/epoch_projection_v2.json
"""
import argparse
import json
import math
from pathlib import Path


def stage_projection(history, *, rollout_speedup=1., learning_speedup=1.,
                     evaluation_speedup=1., target_minutes=30.):
    """Project wall time without double-counting concurrent worker seconds."""
    factors = (rollout_speedup, learning_speedup,
               evaluation_speedup, target_minutes)
    if any(not math.isfinite(float(x)) or float(x) <= 0 for x in factors):
        raise ValueError('speedups and target minutes must be finite, positive numbers')
    stages = history['stage_wall_seconds']
    rollout = float(stages['rollout'])
    learning = float(stages['learn'])
    other_train = sum(float(value) for stage, value in stages.items()
                      if stage not in ('rollout', 'learn'))
    evaluation = history.get('evaluation_seconds')
    if evaluation is None:
        raise ValueError('historical evaluation stage missing; cannot project full epoch')
    evaluation = float(evaluation)
    original = rollout + learning + other_train + evaluation
    forecast = {
        'rollout_seconds': rollout / rollout_speedup,
        'learning_seconds': learning / learning_speedup,
        'other_training_seconds': other_train,
        'evaluation_seconds': evaluation / evaluation_speedup,
    }
    predicted = sum(forecast.values())
    target_seconds = target_minutes * 60.
    fixed_without_rollout = predicted - forecast['rollout_seconds']
    fixed_without_learning = predicted - forecast['learning_seconds']
    remaining_rollout_budget = target_seconds - fixed_without_rollout
    remaining_learning_budget = target_seconds - fixed_without_learning
    return {
        'historical': {
            'rollout_seconds': rollout,
            'learning_seconds': learning,
            'other_training_seconds': other_train,
            'evaluation_seconds': evaluation,
            'full_epoch_seconds': original,
            'full_epoch_minutes': original / 60.,
        },
        'speedup_inputs': {
            'rollout': rollout_speedup,
            'learning': learning_speedup,
            'evaluation': evaluation_speedup,
        },
        'projected_stages_seconds': forecast,
        'projected_epoch_seconds': predicted,
        'projected_epoch_minutes': predicted / 60.,
        'projected_full_epoch_speedup': original / predicted,
        'target_minutes': target_minutes,
        'target_met_by_projection': predicted <= target_seconds,
        'additional_rollout_speedup_needed_at_given_other_factors': (
            rollout / remaining_rollout_budget if remaining_rollout_budget > 0
            else None),
        'additional_learning_speedup_needed_at_given_other_factors': (
            learning / remaining_learning_budget if remaining_learning_budget > 0
            else None),
        'interpretation': [
            'A scenario, not an observed full-epoch runtime.',
            'All speedup inputs must refer to equal workloads and identical training semantics.',
            'Learning gradient microbenchmarks omit grouping/optimizer/checkpoint overhead.',
            'Frozen-policy rollout benchmarks do not model changed trajectories or optimizer updates.',
            'Evaluation stays unchanged unless an independent evaluation speedup is supplied.',
            'Other training costs, including snapshots and checkpoint writes, remain unchanged.',
            'Worker task-second sums are parallel and never added to wall time.',
        ],
    }


def read_history(run, epoch):
    from .benchmark_training_speed import read_existing_profiles
    config = json.loads((run / 'config.json').read_text())
    summary, _ = read_existing_profiles(run, epoch, config.get('workers', 8))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--epoch', type=int, default=2)
    p.add_argument('--rollout-speedup', type=float, default=1.)
    p.add_argument('--learning-speedup', type=float, default=1.)
    p.add_argument('--evaluation-speedup', type=float, default=1.)
    p.add_argument('--target-minutes', type=float, default=30.)
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    try:
        result = stage_projection(
            read_history(args.run, args.epoch),
            rollout_speedup=args.rollout_speedup,
            learning_speedup=args.learning_speedup,
            evaluation_speedup=args.evaluation_speedup,
            target_minutes=args.target_minutes)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
