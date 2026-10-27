"""Experimental length-packed score_many for exact gradient comparison.

The existing Policy.score_many batches ALL distinct action descriptions together.
A single long action therefore pads every other action to that length, even if
the *contexts* are length bucketed. This module additionally sorts and packs
the action encoder inputs and, when needed, context encoder inputs.

All embeddings remain attached to autograd. Unlike the two-pass method, this
does not detach/recompute action embeddings. It is an experiment only and does
not alter the production training path.
"""
import json
import math

import torch

from .gpu_runtime import token_count


def _pack_lengths(texts, representation, heads, limit, max_batch_size):
    """Return original indices in length-ordered, attention-budgeted chunks."""
    if not texts:
        return []
    if min(heads, limit, max_batch_size) < 1:
        raise ValueError('positive heads, attention budget and batch size required')
    lengths = [token_count(text, representation) for text in texts]
    indices = sorted(range(len(texts)), key=lambda i: (lengths[i], i))
    chunks = []
    current = []
    longest = 0
    for index in indices:
        length = lengths[index]
        if heads * length**2 > limit:
            raise ValueError('single encoding exceeds packed attention-element limit')
        padded = max(longest, length)
        if current and (len(current) == max_batch_size or
                        (len(current) + 1) * heads * padded**2 > limit):
            chunks.append(current)
            current = []
            longest = 0
        current.append(index)
        longest = max(longest, length)
    if current:
        chunks.append(current)
    return chunks


def _encode_packed(policy, texts, *, max_batch_size, max_attention_elements):
    """Autograd-preserving packed embeddings in the requested input order."""
    if not texts:
        return None
    groups = _pack_lengths(
        texts, policy.representation, policy.config['heads'],
        max_attention_elements, max_batch_size)
    outputs = [None] * len(texts)
    for indices in groups:
        embedded = policy.encode([texts[i] for i in indices])
        # Unbind/index into the actual differentiable embeddings. torch.stack
        # after reordering retains the complete gradient for every action.
        for index, value in zip(indices, embedded.unbind(0)):
            outputs[index] = value
    return torch.stack(outputs)


def packed_score_many(policy, observations, *, max_action_batch_size=8,
                      max_attention_elements=8000000):
    """Differentiate through both state and action encoders with less padding.

    Returns list[Tensor] with the same action ordering as Policy.score_many.
    No model parameters, output numerics, or forward semantics are changed.
    """
    if not observations:
        return []
    contexts = [{k: v for k, v in observation.items() if k != 'actions'}
                for observation in observations]
    context_texts = [json.dumps(ctx, sort_keys=True, separators=(',', ':'))
                     for ctx in contexts]
    action_groups = [
        [json.dumps(action, sort_keys=True, separators=(',', ':'))
         for action in observation['actions']]
        for observation in observations
    ]
    if any(not actions for actions in action_groups):
        raise ValueError('terminal observation has no actions')
    unique = list(dict.fromkeys(action for group in action_groups for action in group))
    positions = {value: index for index, value in enumerate(unique)}
    context_embeddings = _encode_packed(
        policy, context_texts,
        max_batch_size=len(context_texts),
        max_attention_elements=max_attention_elements)
    action_embeddings = _encode_packed(
        policy, unique,
        max_batch_size=max_action_batch_size,
        max_attention_elements=max_attention_elements)
    queries = policy.query(context_embeddings)
    keys = policy.key(action_embeddings)
    biases = policy.action_bias(action_embeddings).squeeze(-1)
    output = []
    for query, group in zip(queries, action_groups):
        indices = torch.tensor([positions[action] for action in group],
                               dtype=torch.long, device=keys.device)
        logits = ((keys.index_select(0, indices) * query).sum(-1)
                  / math.sqrt(query.shape[-1])
                  + biases.index_select(0, indices))
        output.append(logits)
    return output
