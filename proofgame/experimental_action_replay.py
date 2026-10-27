"""Exact-gradient experiment: deduplicate action encoder work across an update.

The policy encoder sees the same JSON action descriptions many times. The
standard CPU training path recomputes them for every state. Instead:

  (1) Encode each distinct action ONCE with torch.no_grad(), then project to
      key/bias features held as independent leaf tensors.
  (2) Score all contexts in length-bucketed microbatches, accumulating
      gradients into the cached key/bias leaves and the real context network.
  (3) Recompute each unique action ONCE *with autograd*, and inject the leaf
      gradients through key/bias into the real encoder parameters.

This is a two-pass, constant-activation-memory chain-rule calculation, not a
detached action-encoder approximation. Exactness requires deterministic eval
mode (no active dropout), identical float32 model weights, and no optimizer
steps during the calculation. CPU/MPS numerical parity MUST be measured.
"""
import math
import time

import torch

from .exploration import behavior_log_probs
from .gpu_runtime import token_count
from .network_search import encode_json


def _sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def _batch_indices(action_texts, representation, width, size, max_attention_elements):
    if size < 1 or max_attention_elements < 1:
        raise ValueError('positive action microbatch and attention limit required')
    length = [token_count(s, representation) for s in action_texts]
    order = sorted(range(len(action_texts)), key=lambda i: (length[i], i))
    result = []
    at = 0
    while at < len(order):
        chunk = []
        longest = 0
        while at < len(order) and len(chunk) < size:
            idx = order[at]
            next_length = max(longest, length[idx])
            # 4 heads for the default policy; head count supplied by caller.
            if (chunk and (len(chunk) + 1) * width * next_length**2 >
                    max_attention_elements):
                break
            if not chunk and width * next_length**2 > max_attention_elements:
                raise ValueError('single action exceeds attention element limit')
            chunk.append(idx)
            longest = next_length
            at += 1
        result.append(chunk)
    return result


def replay_unique_action_gradients(
        policy, samples, *, context_batch_size=4, action_batch_size=16,
        bucketed_contexts=True, denominator=1, epsilon=0.,
        max_attention_elements=12000000):
    """Return (full gradient vector, loss, timing/stats), WITHOUT optimizer.step.

    Each sample is {'observation': dict, 'counts': list[float], ...}. All
    gradients, including the action encoder, are computed. Preserves the
    denominator and sign of the original group-relative coefficients.
    """
    if not samples or context_batch_size < 1 or action_batch_size < 1:
        raise ValueError('nonempty samples and positive microbatches required')
    if denominator <= 0:
        raise ValueError('denominator must be positive')
    if policy.training:
        raise ValueError('two-pass replay requires policy.eval() (dropout disabled)')
    device = next(policy.parameters()).device
    policy.zero_grad(set_to_none=True)
    started = time.perf_counter()

    action_texts = list(dict.fromkeys(
        encode_json(action) for sample in samples
        for action in sample['observation']['actions']))
    action_positions = {s: i for i, s in enumerate(action_texts)}
    row_indices = [
        [action_positions[encode_json(action)] for action in sample['observation']['actions']]
        for sample in samples
    ]
    chunks = _batch_indices(
        action_texts, policy.representation, policy.config['heads'],
        action_batch_size, max_attention_elements)
    with torch.no_grad():
        cached_keys = torch.empty(
            (len(action_texts), policy.config['width']),
            device=device, dtype=next(policy.parameters()).dtype)
        cached_bias = torch.empty((len(action_texts),), device=device,
                                  dtype=cached_keys.dtype)
        for indices in chunks:
            embedding = policy.encode([action_texts[i] for i in indices])
            key = policy.key(embedding)
            bias = policy.action_bias(embedding).squeeze(-1)
            loc = torch.tensor(indices, device=device, dtype=torch.long)
            cached_keys.index_copy_(0, loc, key)
            cached_bias.index_copy_(0, loc, bias)
    # The feature leaves accumulate all dL/dKey and dL/dBias with O(U*width)
    # storage; the large action Transformer graph is NOT held across states.
    keys = cached_keys.detach().requires_grad_(True)
    biases = cached_bias.detach().requires_grad_(True)
    _sync(device)
    feature_seconds = time.perf_counter() - started

    order = list(range(len(samples)))
    if bucketed_contexts:
        order.sort(key=lambda i: (samples[i]['tokens'], i))
    total_loss = 0.
    t_context = time.perf_counter()
    for offset in range(0, len(order), context_batch_size):
        ids = order[offset:offset + context_batch_size]
        batch = [samples[i] for i in ids]
        # Only contexts are encoded during the full group-relative replay.
        contexts = [encode_json({k: v for k, v in row['observation'].items()
                                 if k != 'actions'}) for row in batch]
        queries = policy.query(policy.encode(contexts))
        losses = []
        for query, idx, row in zip(queries, ids, batch):
            loc = torch.tensor(row_indices[idx], device=device, dtype=torch.long)
            logits = ((keys.index_select(0, loc) * query).sum(-1)
                      / math.sqrt(query.shape[-1])
                      + biases.index_select(0, loc))
            logps = behavior_log_probs(logits, row['observation'], epsilon)
            counts = torch.tensor(row['counts'], device=device, dtype=logps.dtype)
            losses.append(-(logps * counts).sum() / denominator)
        loss = torch.stack(losses).sum()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('non-finite two-pass loss')
        loss.backward()
        total_loss += float(loss.detach())
    _sync(device)
    context_seconds = time.perf_counter() - t_context
    if keys.grad is None or biases.grad is None:
        raise RuntimeError('cached action feature gradients missing')

    t_action = time.perf_counter()
    # Only NOW build and discard a small real action-encoder autograd graph.
    # torch.autograd.backward accepts non-scalar output tensors plus upstream
    # dL/dkeys and dL/dbias values for exactly one group of unique actions.
    with torch.enable_grad():
        for indices in chunks:
            embedding = policy.encode([action_texts[i] for i in indices])
            key = policy.key(embedding)
            bias = policy.action_bias(embedding).squeeze(-1)
            loc = torch.tensor(indices, device=device, dtype=torch.long)
            torch.autograd.backward(
                (key, bias),
                (keys.grad.index_select(0, loc),
                 biases.grad.index_select(0, loc)))
    _sync(device)
    action_seconds = time.perf_counter() - t_action

    vector = torch.cat([
        p.grad.detach().cpu().reshape(-1).to(torch.float64)
        if p.grad is not None else torch.zeros(p.numel(), dtype=torch.float64)
        for p in policy.parameters()
    ]).clone()
    if not bool(torch.isfinite(vector).all()):
        raise RuntimeError('non-finite two-pass gradients')
    result = {
        'wall_seconds': time.perf_counter() - started,
        'action_feature_no_grad_seconds': feature_seconds,
        'context_and_loss_backward_seconds': context_seconds,
        'action_gradient_replay_seconds': action_seconds,
        'loss': total_loss,
        'unique_actions': len(action_texts),
        'action_references': sum(len(v) for v in row_indices),
        'action_reuse_factor': sum(len(v) for v in row_indices) / len(action_texts),
        'distinct_action_groups': len(chunks),
        'feature_leaf_bytes': sum(t.numel() * t.element_size()
                                  for t in (keys, biases)),
        'note': 'Full gradient through both the context and action encoders; '
                'no optimizer step or parameter modification.',
    }
    return vector, result


def two_pass_probability_disagreement(policy, samples, epsilon):
    """Numerical agreement with sampled on-policy probabilities (untimed)."""
    actions = list(dict.fromkeys(
        encode_json(a) for row in samples for a in row['observation']['actions']))
    places = {text: i for i, text in enumerate(actions)}
    device = next(policy.parameters()).device
    with torch.no_grad():
        # Small independent encoder groups avoid a huge padded forward.
        chunks = _batch_indices(
            actions, policy.representation, policy.config['heads'], 8, 12000000)
        keys = torch.empty((len(actions), policy.config['width']),
                           device=device, dtype=next(policy.parameters()).dtype)
        biases = torch.empty((len(actions),), device=device, dtype=keys.dtype)
        for group in chunks:
            vectors = policy.encode([actions[i] for i in group])
            loc = torch.tensor(group, dtype=torch.long, device=device)
            keys.index_copy_(0, loc, policy.key(vectors))
            biases.index_copy_(0, loc, policy.action_bias(vectors).squeeze(-1))
        max_error = 0.
        checked = 0
        for row in samples:
            observation = row['observation']
            context = {k: v for k, v in observation.items() if k != 'actions'}
            query = policy.query(policy.encode([encode_json(context)]))[0]
            loc = torch.tensor([places[encode_json(a)] for a in observation['actions']],
                               dtype=torch.long, device=device)
            logits = ((keys.index_select(0, loc) * query).sum(-1) /
                      math.sqrt(query.shape[-1]) + biases.index_select(0, loc))
            probs = behavior_log_probs(logits, observation, epsilon).exp()
            for idx, expected in row['expected'].items():
                max_error = max(max_error, abs(float(probs[idx]) - expected))
                checked += 1
    return max_error, checked
