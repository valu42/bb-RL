"""Fine-grained, version-frozen branch races on the existing CPU process pool.

A machine only queues its current round. Completion order never selects a
winner or changes its random stream. The parent reuses execute_machine's
round/group/diagnostic reducer, then performs the original single batch update.
"""
import hashlib
import json
import queue
import random
import time
from types import SimpleNamespace


def attempt_seed(machine_seed, index):
    return int.from_bytes(hashlib.sha256(
        f'attempt-v1:{machine_seed}:{index}'.encode()).digest()[:8], 'big')


def branch_schedule(seed, attempts, rounds):
    rng = random.Random(seed ^ 0xA5A5F00D)
    schedule = []
    for number in range(1, rounds + 1):
        answers = ['HALT'] * attempts + ['NONHALT'] * attempts
        rng.shuffle(answers)
        schedule.extend((number, answer) for answer in answers)
    return schedule


def run_one_attempt(job):
    from . import epoch_train as et
    from .logic import Machine
    from .network_search import FrozenSampler, attempt
    task, index, answer = job
    selected, seed, _, budget, weights, version, _, epsilon, _, diagnostics, profiling, _, _ = task
    started = time.perf_counter()
    et.load_worker_snapshot(weights, version)
    load_seconds = time.perf_counter() - started
    # Keep bounded observation caches per machine, on each worker, for this
    # frozen version. Splitting jobs must not discard all repeat-state reuse.
    key = selected['machine_id']
    if key not in et._worker_samplers:
        et._worker_samplers[key] = FrozenSampler(
            et._worker_policy, capacity=10000,
            shared_action_features=et._worker_action_features,
            action_cache_capacity=4096)
        if len(et._worker_samplers) > 16:
            et._worker_samplers.popitem(last=False)
    et._worker_samplers.move_to_end(key)
    sampler = et._worker_samplers[key]
    before = (sampler.hits, sampler.misses,
              sampler.action_feature_hits, sampler.action_feature_misses)
    profile = ({'observation_seconds':0., 'inference_seconds':0.,
                'step_seconds':0., 'verification_seconds':0.,
                'inference_calls':0, 'actions':0} if profiling else None)
    initial = []
    detail = [] if diagnostics else None
    cert, trace, status = attempt(
        sampler, Machine(selected['machine']['rows']),
        random.Random(attempt_seed(seed, index)), budget,
        exploration_epsilon=epsilon, forced_initial_answer=answer,
        on_initial_choice=lambda *values: initial.append(values),
        diagnostic_steps=detail, profile=profile)
    if profile is not None:
        profile.update(worker_model_load_seconds=load_seconds,
                       cache_hits=sampler.hits-before[0],
                       cache_misses=sampler.misses-before[1],
                       action_feature_hits=sampler.action_feature_hits-before[2],
                       action_feature_misses=sampler.action_feature_misses-before[3])
    return {'certificate':cert, 'trace':trace, 'status':status,
            'initial':initial, 'detail':detail, 'profile':profile,
            'hits':sampler.hits-before[0], 'misses':sampler.misses-before[1],
            'seconds':time.perf_counter()-started}


def reduce_attempts(task, outcomes):
    """Use the production reducer, in scheduled order, regardless of finish order."""
    from .epoch_train import execute_machine
    selected, seed, attempts, budget, _, version, stop, epsilon, balanced, diagnostics, profiling, relative, rounds = task
    cursor = iter(outcomes)
    sampler = SimpleNamespace(hits=sum(x['hits'] for x in outcomes),
                              misses=sum(x['misses'] for x in outcomes))
    def replay(_sampler, _machine, _rng, _budget, **kwargs):
        row = next(cursor)
        for values in row['initial']:
            kwargs['on_initial_choice'](*values)
        if kwargs['diagnostic_steps'] is not None:
            kwargs['diagnostic_steps'].extend(row['detail'])
        if kwargs['profile'] is not None:
            for name, value in row['profile'].items():
                kwargs['profile'][name] = kwargs['profile'].get(name, 0) + value
        return row['certificate'], row['trace'], row['status']
    result = execute_machine(
        selected, seed, attempts, budget, version, stop, sampler,
        epsilon, balanced, diagnostics, profiling, relative, rounds,
        attempt_runner=replay, independent_attempt_seeds=True)
    # Sum of actual worker task durations, NOT elapsed machine wall time.
    result['seconds'] = sum(x['seconds'] for x in outcomes)
    result['worker_task_seconds'] = result['seconds']
    result['attempt_rng'] = 'per-attempt-v1'
    return result


def run_local_rollouts(pool, tasks, *, workers, split_attempts=False,
                       log_interval=10., progress_context=None,
                       independent_attempt_seeds=False):
    """Run a bounded machine batch with live heartbeats and ordered results.

    Async callbacks only enqueue results. All scheduling and reduction happen
    in the parent thread. Exceptions are propagated before any optimizer step.
    """
    from .epoch_train import run_machine
    if not tasks:
        return []
    if workers < 1 or log_interval <= 0:
        raise ValueError('positive workers and log interval required')
    if len({(task[4], task[5]) for task in tasks}) != 1:
        raise ValueError('all rollout tasks must use one frozen snapshot/version')
    if split_attempts and any(not t[8] or not t[11] or not t[12] or t[6] for t in tasks):
        raise ValueError('attempt scheduling requires complete balanced branch groups')
    events = queue.Queue()
    results = [None] * len(tasks)
    schedules = [branch_schedule(t[1],t[2],t[12]) for t in tasks] if split_attempts else []
    outcomes = [dict() for _ in tasks]
    round_pending = [0] * len(tasks)
    outstanding = completed = actions = attempts_done = 0
    started = last_log = time.perf_counter()
    worker_pids = {getattr(p, 'pid', None) for p in pool._pool} - {None}

    def submit(machine, index=None):
        nonlocal outstanding
        callback = lambda result: events.put(('ok', machine, index, result))
        error_callback = lambda error: events.put(('error', machine, index, error))
        job = tasks[machine] if index is None else (
            tasks[machine], index, schedules[machine][index][1])
        pool.apply_async(run_machine if index is None else run_one_attempt,
                         (job,), kwds=({'independent_attempt_seeds':True}
                                      if index is None and independent_attempt_seeds else {}),
                         callback=callback, error_callback=error_callback)
        outstanding += 1

    def submit_round(machine, offset):
        count = 2 * tasks[machine][2]
        round_pending[machine] = count
        for index in range(offset, offset + count):
            submit(machine, index)

    def progress():
        elapsed = time.perf_counter() - started
        print(json.dumps({'event':'rollout_progress', **(progress_context or {}),
                          'scheduler':'attempt' if split_attempts else 'machine',
                          'machines_done':completed, 'machines_total':len(tasks),
                          'attempts_done':attempts_done, 'actions_done':actions,
                          'outstanding_jobs':outstanding,
                          'elapsed_seconds':round(elapsed,3),
                          'actions_per_second':round(actions/elapsed,2) if elapsed else 0.}),flush=True)

    for machine in range(len(tasks)):
        if split_attempts:
            submit_round(machine, 0)
        else:
            submit(machine)
    while completed < len(tasks):
        try:
            kind, machine, index, row = events.get(timeout=min(log_interval, 1.))
        except queue.Empty:
            # Match Pool failure behavior but don't wait forever on a dead
            # process whose callback can never arrive.
            if (any(getattr(p, 'exitcode', None) not in (None, 0) for p in pool._pool)
                    or (worker_pids and worker_pids != {p.pid for p in pool._pool})):
                raise RuntimeError('rollout worker exited unexpectedly')
        else:
            if kind == 'error':
                raise row
            outstanding -= 1
            if split_attempts:
                outcomes[machine][index] = row
                round_pending[machine] -= 1
                actions += len(row['trace'])
                attempts_done += 1
                if round_pending[machine] == 0:
                    rows = [outcomes[machine][i] for i in range(len(outcomes[machine]))]
                    if any(r['certificate'] for r in rows) or len(rows) == len(schedules[machine]):
                        results[machine] = reduce_attempts(tasks[machine], rows)
                        outcomes[machine].clear()
                        completed += 1
                    else:
                        submit_round(machine, len(rows))
            else:
                results[machine] = row
                completed += 1
                actions += row['actions']
                attempts_done += row['attempts']
        now = time.perf_counter()
        if now - last_log >= log_interval:
            progress()
            last_log = now
    progress()
    return results
