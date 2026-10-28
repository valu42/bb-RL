"""Autonomous evaluation with bounded CPU workers and live progress.

The seeds and per-machine stopping rules match the original serial evaluator.
Only the parent writes logs. Workers report partial attempts so progress remains
visible even when no machine has finished its search allowance yet.
"""
import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import queue
import signal
import tempfile
import time

import torch


TIMINGS = ('inference_seconds', 'game_seconds', 'observation_seconds')
_policy = None
_progress_queue = None


def _init_worker(snapshot, progress_queue, threads):
    from ..model import Policy
    from .training import load
    global _policy, _progress_queue
    # Let the parent handle Ctrl-C and terminate its workers as one operation.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(threads)
    state = load(snapshot)
    _policy = Policy(**state['model'])
    _policy.load_state_dict(state['state_dict'])
    _policy.eval()
    _progress_queue = progress_queue


def _evaluate_one(policy, index, row, budget, attempts, seed, report):
    from .training import rollout
    successes = []
    actions = 0
    statuses = {}
    timing = dict.fromkeys(TIMINGS, 0.)
    last_report = 0.
    executed = 0

    def pulse(partial_actions=0, partial_timing=None, *, force=False, done=False):
        nonlocal last_report
        now = time.monotonic()
        if not force and now-last_report < 1.:
            return
        last_report = now
        report({'index': index, 'done': done, 'attempts_done': executed,
                'actions_done': actions+partial_actions,
                'answer': successes[0]['certificate']['answer'] if successes else None,
                'statuses': dict(statuses),
                'timing': {k: timing[k]+(partial_timing or {}).get(k, 0.) for k in TIMINGS}})

    pulse(force=True)
    for a in range(attempts):
        trajectory = rollout(policy, row, budget, seed+index*100000+a, progress=pulse)
        executed += 1
        actions += len(trajectory['actions'])
        status = trajectory['status']
        statuses[status] = statuses.get(status, 0)+1
        for k in TIMINGS:
            timing[k] += trajectory['timing'][k]
        if trajectory['certificate']:
            successes = [{'attempt': executed, 'actions': trajectory['actions'],
                          'certificate': trajectory['certificate']}]
            break
        pulse()
    pulse(force=True, done=True)
    return {**row, 'successes': successes, 'attempts': executed, 'actions': actions,
            'statuses': statuses, 'timing': timing}


def _worker_job(task):
    return _evaluate_one(_policy, *task, report=_progress_queue.put)


def evaluate_rows(policy, machines, budget, attempts, seed, *, workers=1,
                  threads=1, on_progress=None, progress_interval=5.):
    """Evaluate on CPU processes, or directly on the requested non-CPU device.

    Results retain input order; completion order does not affect seeds or scores.
    ETA is an estimate from completed machines, not a guaranteed time bound.
    """
    if workers < 1 or attempts < 1 or budget < 1 or threads < 1 or progress_interval <= 0:
        raise ValueError('positive evaluation limits required')
    machines = list(machines)
    policy.eval()
    device = next(policy.parameters()).device.type
    count = len(machines)
    effective_workers = min(workers, count) if device == 'cpu' else min(1, count)
    began = time.monotonic()
    last_log = began
    latest = {}
    finished = set()
    results = [None]*count

    def emit(event):
        nonlocal last_log
        now = time.monotonic()
        last_log = now
        elapsed = now-began
        done = sum(bool(p['done']) for p in latest.values())
        actions = sum(p['actions_done'] for p in latest.values())
        counts = {a: sum(p['answer'] == a for p in latest.values()) for a in ('HALT', 'NONHALT')}
        statuses = {}
        for p in latest.values():
            for status, n in p['statuses'].items():
                statuses[status] = statuses.get(status, 0)+n
        worker_seconds = {k: sum(p['timing'][k] for p in latest.values()) for k in TIMINGS}
        row = {'event': event, 'device': device, 'workers': effective_workers,
               'requested_workers': workers, 'machines_done': done, 'machines_total': count,
               'machines_in_progress': sum(not p['done'] for p in latest.values()),
               'attempts_done': sum(p['attempts_done'] for p in latest.values()),
               'attempts_max': count*attempts, 'actions_done': actions,
               'actions_max': count*attempts*budget, 'budget_per_attempt': budget,
               'actions_per_second': round(actions/max(elapsed, 1e-9), 2),
               'elapsed_seconds': round(elapsed, 2),
               'eta_seconds': round(elapsed*(count-done)/done, 2) if done else (0. if not count else None),
               'eta_basis': 'completed machines; search difficulty varies',
               'machines_proved_by_answer': counts,
               'machines_unproved_completed': done-sum(counts.values()),
               'attempt_status_counts': statuses,
               'worker_seconds_sum': {k: round(v, 3) for k, v in worker_seconds.items()}}
        if on_progress is None:
            print(json.dumps(row), flush=True)
        else:
            on_progress(row)

    def receive(message):
        if message['index'] not in finished:
            latest[message['index']] = message
        if time.monotonic()-last_log >= progress_interval:
            emit('evaluation_progress')

    def complete(index, result):
        results[index] = result
        latest[index] = {'index': index, 'done': True, 'attempts_done': result['attempts'],
                         'actions_done': result['actions'], 'statuses': result['statuses'],
                         'timing': result['timing'],
                         'answer': result['successes'][0]['certificate']['answer'] if result['successes'] else None}
        finished.add(index)

    emit('evaluation_start')
    if effective_workers <= 1:
        for i, machine in enumerate(machines):
            complete(i, _evaluate_one(policy, i, machine, budget, attempts, seed, receive))
    else:
        from .training import save
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory(prefix='foundation-eval-') as directory:
            snapshot = Path(directory)/'model.pt'
            save(snapshot, {'model': policy.config,
                            'state_dict': {k: v.detach().cpu() for k, v in policy.state_dict().items()}})
            messages = context.Queue()
            pool = concurrent.futures.ProcessPoolExecutor(
                max_workers=effective_workers, mp_context=context,
                initializer=_init_worker, initargs=(str(snapshot), messages, threads))
            pending = {}
            next_index = 0
            try:
                while next_index < count or pending:
                    while next_index < count and len(pending) < effective_workers:
                        task = (next_index, machines[next_index], budget, attempts, seed)
                        pending[pool.submit(_worker_job, task)] = next_index
                        next_index += 1
                    for future in list(pending):
                        if future.done():
                            complete(pending.pop(future), future.result())
                    try:
                        receive(messages.get(timeout=min(.2, progress_interval)))
                    except queue.Empty:
                        pass
                    if time.monotonic()-last_log >= progress_interval:
                        emit('evaluation_progress')
            except BaseException:
                # Python <3.14 has no public terminate_workers(). Waiting for all
                # 100 attempts during Ctrl-C would make interruption look hung.
                for process in list((getattr(pool, '_processes', None) or {}).values()):
                    process.terminate()
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
            finally:
                messages.close()
                messages.join_thread()
    emit('evaluation_complete')
    return results
