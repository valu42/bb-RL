"""Optional, low-overhead wall-clock profiling for proofgame epoch training.

All measurements use perf_counter; worker CPU/IPC times are summed over
parallel machines and MUST NOT be added to the main-process wall clock.
No MPS synchronization is added solely for profiling. Central inference
includes the existing CPU transfer synchronization for probabilities.
"""
import json
from pathlib import Path


STAGES = ('worker_snapshot', 'rollout', 'learn', 'trace', 'batch_json', 'checkpoint')


def make_batch_profile(*, phase, epoch, batch_number, results, stages, device,
                       gpu_before=None, gpu_after=None, workers=None, learning=None):
    if not results:
        raise ValueError('profile requires a nonempty batch')
    totals = {}
    for row in results:
        for name, value in row.get('profile', {}).items():
            totals[name] = totals.get(name, 0) + value
    actions = sum(row['actions'] for row in results)
    attempts = sum(row['attempts'] for row in results)
    elapsed = stages.get('rollout', 0.)
    wall_total = sum(stages.get(name, 0.) for name in STAGES)
    task_seconds = sum(row.get('worker_task_seconds', row.get('seconds', 0.))
                       for row in results)
    inference_calls = totals.get('cache_hits', 0) + totals.get('cache_misses', 0)
    report = {
        'phase':phase,'epoch':epoch,'batch_number':batch_number,
        'device':device,'machines':len(results),'attempts':attempts,'actions':actions,
        'stage_wall_seconds':{name:round(stages.get(name,0.),6) for name in STAGES},
        'worker_sum_seconds':{name:round(value,6) for name,value in sorted(totals.items())
                              if name.endswith('_seconds')},
        'worker_counts':{name:value for name,value in sorted(totals.items())
                         if not name.endswith('_seconds')},
        'rollout_actions_per_wall_second':round(actions/elapsed,2) if elapsed else None,
        'rollout_attempts_per_wall_second':round(attempts/elapsed,2) if elapsed else None,
        'rollout_machines_per_wall_second':round(len(results)/elapsed,3) if elapsed else None,
        'stage_wall_percent':{name:round(100 * stages.get(name, 0.) / wall_total, 2)
                              if wall_total else 0. for name in STAGES},
        'worker_task_seconds_sum':task_seconds,
        'elapsed_worker_occupancy':task_seconds / (workers * elapsed)
                                  if workers and elapsed else None,
        'occupancy_note':'Elapsed task occupancy, not hardware CPU utilization; worker sums overlap.',
        'observation_cache_hit_rate':totals.get('cache_hits', 0) / inference_calls
                                     if inference_calls else None,
        'learning':{key:value for key,value in (learning or {}).items()
                    if key.startswith(('timing_', 'two_pass_')) or
                    key in ('unique_gradient_observations','gradient_backend','numerical_fallbacks')},
    }
    if gpu_before is not None and gpu_after is not None:
        names=('inference_requests','observation_cache_hits',
               'gpu_context_batches','inference_wall_seconds')
        report['central_inference_delta']={
            name:round(gpu_after[name]-gpu_before[name],6)
            for name in names}
    return report


def save_batch_profile(directory, report):
    folder=Path(directory)/'profiling'
    folder.mkdir(parents=True,exist_ok=True)
    filename=f"epoch_{report['epoch']:02d}_{report['phase']}_batch_{report['batch_number']:04d}.json"
    path=folder/filename
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report,indent=2)+'\n')
    tmp.replace(path)
    stages=report['stage_wall_seconds']
    worker=report['worker_sum_seconds']
    print(json.dumps({
        'event':'batch_profile',
        'phase':report['phase'],'epoch':report['epoch'],'batch':report['batch_number'],
        'device':report['device'],'machines':report['machines'],
        'actions':report['actions'],'attempts':report['attempts'],
        'actions_per_second':report['rollout_actions_per_wall_second'],
        'attempts_per_second':report['rollout_attempts_per_wall_second'],
        'rollout_seconds':stages['rollout'],
        'learn_seconds':stages['learn'],
        'snapshot_seconds':stages['worker_snapshot'],
        'batch_json_seconds':stages['batch_json'],
        'trace_seconds':stages['trace'],
        'checkpoint_seconds':stages['checkpoint'],
        'worker_model_load_seconds_sum':worker.get('worker_model_load_seconds',0.),
        'worker_inference_seconds_sum':worker.get('inference_seconds',0.),
        'worker_observation_seconds_sum':worker.get('observation_seconds',0.),
        'worker_step_seconds_sum':worker.get('step_seconds',0.),
        'worker_verification_seconds_sum':worker.get('verification_seconds',0.),
        'stage_wall_percent':report['stage_wall_percent'],
        'elapsed_worker_occupancy':report['elapsed_worker_occupancy'],
        'observation_cache_hit_rate':report['observation_cache_hit_rate'],
        'learning':report['learning'],
        'profile_path':str(path),
    }),flush=True)
    return path


def summarize_epoch_profiles(directory, epoch, phase):
    folder=Path(directory)/'profiling'
    rows=[json.loads(p.read_text()) for p in
          sorted(folder.glob(f'epoch_{epoch:02d}_{phase}_batch_*.json'))]
    if not rows:
        return None
    stage_sums={name:round(sum(row['stage_wall_seconds'][name] for row in rows),4)
                for name in STAGES}
    worker_names=set().union(*(row['worker_sum_seconds'].keys() for row in rows))
    worker_sums={name:round(sum(row['worker_sum_seconds'].get(name,0.) for row in rows),4)
                 for name in sorted(worker_names)}
    actions=sum(row['actions'] for row in rows)
    report={'epoch':epoch,'phase':phase,'profiled_batches':len(rows),
            'machines':sum(row['machines'] for row in rows),
            'attempts':sum(row['attempts'] for row in rows),
            'actions':actions,'stage_wall_seconds_sum':stage_sums,
            'worker_sum_seconds':worker_sums,
            'rollout_actions_per_wall_second':round(actions/stage_sums['rollout'],2)
            if stage_sums['rollout'] else None,
            'note':'Worker sums overlap in parallel; do not add them to wall-clock stage times.'}
    total=sum(stage_sums.values())
    report['stage_wall_percent']={name:round(100*value/total,2) if total else 0.
                                  for name,value in stage_sums.items()}
    path=folder/f'epoch_{epoch:02d}_{phase}_summary.json'
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report,indent=2)+'\n')
    tmp.replace(path)
    print(json.dumps({'event':'epoch_profile_summary','path':str(path),
                      'phase':phase,'epoch':epoch,
                      'actions_per_second':report['rollout_actions_per_wall_second'],
                      'stage_wall_seconds_sum':stage_sums}),flush=True)
    return report
