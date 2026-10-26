"""Single-device batched inference for parallel proof workers.

Only the main process owns the torch model / MPS device. Spawned workers
execute symbolic proof games and send observation JSON through multiprocessing
queues. Inference batches requests from *different* workers and caches action
embeddings, avoiding one GPU context per proof worker.

The same coordinator also runs on CPU for tests and apples-to-apples timing.
"""
from collections import OrderedDict
from functools import lru_cache
import json
import math
import multiprocessing as mp
import queue
import time
import traceback

import torch

from .model import ModelLimit
from .network_search import encode_json


@lru_cache(maxsize=16384)
def token_count(serialized, representation):
    if representation == 'ast':
        from .structural_encoding import cached_tree
        return len(cached_tree(serialized)[0])
    return len(serialized.encode('utf-8')) + 1


class RemoteSampler:
    """A per-machine observation cache with a synchronous remote policy API."""
    def __init__(self, worker_id, request_queue, reply_queue, capacity=10000):
        self.worker_id = worker_id
        self.requests = request_queue
        self.replies = reply_queue
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def probabilities(self, observation):
        key = encode_json(observation)
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        self.requests.put(('infer', self.worker_id, key))
        status, value = self.replies.get()
        if status == 'limit':
            raise ModelLimit(value)
        if status != 'ok':
            raise RuntimeError('central inference failed: ' + str(value))
        probs = tuple(value)
        if len(probs) != len(observation['actions']):
            raise RuntimeError('central inference returned wrong action count')
        if self.capacity:
            self.cache[key] = probs
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        return probs


def _worker_loop(worker_id, jobs, events, replies):
    try:
        torch.set_num_threads(1)
        # Do not construct or move a policy in a worker: Metal is owned only
        # by the main process, and proof execution remains on CPU.
        from .epoch_train import execute_machine
        while True:
            item = jobs.get()
            if item is None:
                return
            task_id, task = item
            try:
                selected, seed, attempts, budget, version, stop_on_success, epsilon = task
                sampler = RemoteSampler(worker_id, events, replies)
                result = execute_machine(selected, seed, attempts, budget,
                                         version, stop_on_success, sampler, epsilon)
                events.put(('done', task_id, result))
            except BaseException:
                events.put(('error', worker_id, traceback.format_exc()))
    except BaseException:
        events.put(('error', worker_id, traceback.format_exc()))


class BatchedPolicyInference:
    """GPU-owned, versioned action-feature and observation probability caches."""
    def __init__(self, policy, batch_size=8, max_attention_elements=8000000,
                 action_cache_size=4096, state_cache_size=8192):
        self.policy = policy
        self.device = next(policy.parameters()).device
        self.batch_size = batch_size
        self.max_attention_elements = max_attention_elements
        self.action_cache_size = action_cache_size
        self.state_cache_size = state_cache_size
        self.version = None
        self.actions = OrderedDict()
        self.states = OrderedDict()
        self.cache_hits = 0
        self.network_batches = 0
        self.requests = 0
        self.inference_seconds = 0.
        self.batch_sizes = []

    def begin_version(self, version):
        if version != self.version:
            self.version = version
            self.actions.clear()
            self.states.clear()

    @torch.inference_mode()
    def infer(self, request_strings):
        started = time.monotonic()
        self.policy.eval()
        outputs = [None] * len(request_strings)
        pending = OrderedDict()
        for i, key in enumerate(request_strings):
            self.requests += 1
            if key in self.states:
                self.cache_hits += 1
                self.states.move_to_end(key)
                outputs[i] = ('ok', self.states[key])
            else:
                pending.setdefault(key, []).append(i)

        records = []
        for key, positions in pending.items():
            try:
                observation = json.loads(key)
                context = {k: v for k, v in observation.items() if k != 'actions'}
                context_text = encode_json(context)
                actions = [encode_json(a) for a in observation['actions']]
                if not actions:
                    raise ValueError('observation has no candidate actions')
                for text in (context_text, *actions):
                    if token_count(text, self.policy.representation) > self.policy.config['max_tokens']:
                        raise ModelLimit('input exceeds model token limit')
                records.append((key, positions, context_text, actions,
                                token_count(context_text, self.policy.representation)))
            except ModelLimit as exc:
                for position in positions:
                    outputs[position] = ('limit', str(exc))

        # Encode each distinct action only once per frozen weight version.
        needed = list(dict.fromkeys(action for record in records for action in record[3]
                                    if action not in self.actions))
        needed.sort(key=lambda text: token_count(text, self.policy.representation))
        at = 0
        while at < len(needed):
            # Sort and bucket action lengths rather than padding many short
            # candidates up to one long copy-expression action.
            texts = []
            longest = 0
            heads = self.policy.config['heads']
            while at < len(needed) and len(texts) < 16:
                candidate = needed[at]
                length = token_count(candidate, self.policy.representation)
                worst = max(longest, length)
                if texts and (len(texts)+1)*heads*worst*worst > self.max_attention_elements:
                    break
                texts.append(candidate)
                longest = worst
                at += 1
            embeddings = self.policy.encode(texts)
            keys = self.policy.key(embeddings)
            biases = self.policy.action_bias(embeddings).squeeze(-1)
            for text, key_tensor, bias_tensor in zip(texts, keys, biases):
                self.actions[text] = (key_tensor, bias_tensor)
                self.actions.move_to_end(text)

        # Similar lengths are grouped to avoid MPS quadratic padding waste.
        records.sort(key=lambda record: record[4])
        position = 0
        heads = self.policy.config['heads']
        while position < len(records):
            longest = records[position][4]
            group = []
            while position < len(records) and len(group) < self.batch_size:
                row = records[position]
                worst = max(longest, row[4])
                attention_elements = (len(group) + 1) * heads * worst * worst
                if group and attention_elements > self.max_attention_elements:
                    break
                group.append(row)
                longest = worst
                position += 1
            self.network_batches += 1
            self.batch_sizes.append(len(group))
            contexts = self.policy.encode([row[2] for row in group])
            queries = self.policy.query(contexts)
            probabilities = []
            lengths = []
            for query, row in zip(queries, group):
                features = [self.actions[action] for action in row[3]]
                keys = torch.stack([item[0] for item in features])
                biases = torch.stack([item[1] for item in features])
                logits = (keys * query).sum(-1) / math.sqrt(query.shape[-1]) + biases
                # Avoid a per-request Metal -> CPU synchronization here.
                probs = logits.float().softmax(-1)
                probabilities.append(probs)
                lengths.append(len(row[3]))
            flattened = torch.cat(probabilities).float().cpu().tolist()
            if any(not math.isfinite(value) or value < 0 for value in flattened):
                raise RuntimeError('non-finite policy probabilities in central GPU inference; '
                                   'inspect the previous optimizer update')
            offset = 0
            for row, count in zip(group, lengths):
                result = tuple(flattened[offset:offset + count])
                offset += count
                self.states[row[0]] = result
                self.states.move_to_end(row[0])
                if len(self.states) > self.state_cache_size:
                    self.states.popitem(last=False)
                for i in row[1]:
                    outputs[i] = ('ok', result)
        # Prune only after this inference group has consumed its embeddings:
        # evicting during filling could remove actions still needed by a row.
        while len(self.actions) > self.action_cache_size:
            self.actions.popitem(last=False)
        assert all(item is not None for item in outputs)
        self.inference_seconds += time.monotonic() - started
        return outputs

    def stats(self):
        return {'inference_requests': self.requests,
                'observation_cache_hits': self.cache_hits,
                'gpu_context_batches': self.network_batches,
                'cached_action_embeddings': len(self.actions),
                'mean_context_batch_size': (sum(self.batch_sizes)/len(self.batch_sizes)
                                            if self.batch_sizes else 0.),
                'inference_wall_seconds': round(self.inference_seconds,3)}


class CentralRolloutPool:
    """Persistent spawned CPU workers + synchronous main-process GPU scheduler."""
    def __init__(self, policy, workers=8, inference_batch_size=8, wait_ms=2.,
                 max_attention_elements=8000000):
        if min(workers, inference_batch_size, max_attention_elements) <= 0 or wait_ms < 0:
            raise ValueError('invalid inference pool settings')
        self.policy = policy
        self.workers = workers
        self.inference_batch_size = inference_batch_size
        self.wait_s = wait_ms / 1000.
        self.ctx = mp.get_context('spawn')
        self.jobs = self.ctx.Queue()
        self.events = self.ctx.Queue()
        self.replies = [self.ctx.Queue() for _ in range(workers)]
        self.processes = []
        self.inference = BatchedPolicyInference(policy, inference_batch_size,
                                               max_attention_elements)

    def __enter__(self):
        self.processes = [
            self.ctx.Process(target=_worker_loop,
                             args=(i, self.jobs, self.events, self.replies[i]),
                             daemon=True)
            for i in range(self.workers)
        ]
        for proc in self.processes:
            proc.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        for _ in self.processes:
            self.jobs.put(None)
        for proc in self.processes:
            proc.join(timeout=2.)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2.)
        for q in (self.jobs, self.events, *self.replies):
            q.close()
        return False

    def _recv(self, timeout):
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            dead = [p.exitcode for p in self.processes if p.exitcode not in (None, 0)]
            if dead:
                raise RuntimeError('proof worker exited unexpectedly: ' + str(dead))
            return None

    def run(self, tasks):
        """Tasks: selected, seed, max_attempts, budget, version, stop_on_success, epsilon."""
        if not tasks:
            return []
        versions = {task[4] for task in tasks}
        if len(versions) != 1:
            raise ValueError('all rollouts must use a single frozen policy version')
        self.inference.begin_version(versions.pop())
        for index, task in enumerate(tasks):
            self.jobs.put((index, task))
        results = [None] * len(tasks)
        finished = 0
        pending = []

        def handle(event):
            nonlocal finished
            if event is None:
                return
            kind, ident, payload = event
            if kind == 'infer':
                pending.append((ident, payload))
            elif kind == 'done':
                if results[ident] is not None:
                    raise RuntimeError('duplicate proof task result')
                results[ident] = payload
                finished += 1
            elif kind == 'error':
                raise RuntimeError('proof worker failure:\n' + payload)
            else:
                raise RuntimeError('unknown proof worker event ' + str(kind))

        while finished < len(tasks):
            handle(self._recv(.5 if not pending else 0.))
            if not pending:
                continue
            deadline = time.monotonic() + self.wait_s
            while len(pending) < self.inference_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                handle(self._recv(remaining))
            batch = pending[:self.inference_batch_size]
            del pending[:len(batch)]
            answers = self.inference.infer([item[1] for item in batch])
            for (worker_id, _), answer in zip(batch, answers):
                self.replies[worker_id].put(answer)
        return results
