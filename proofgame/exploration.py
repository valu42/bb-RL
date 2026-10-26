"""On-policy exploration at answer-selection states.

Training behavior is (1-epsilon)*policy + epsilon*Uniform(HALT, NONHALT).
The same distribution is used for sampling and REINFORCE log probabilities.
Evaluation always uses the unmodified policy. No machine labels or proof
examples are supplied to the policy.
"""
import math

import torch


def answer_indices(observation):
    if observation.get('answer', object()) is not None:
        return None
    actions = observation.get('actions', ())
    ids = [a['id'] for a in actions]
    if ids.count('choose:HALT') != 1 or ids.count('choose:NONHALT') != 1:
        return None
    return ids.index('choose:HALT'), ids.index('choose:NONHALT')


def mix_probabilities(observation, probabilities, epsilon):
    if not 0 <= epsilon <= 1 or not math.isfinite(epsilon):
        raise ValueError('exploration epsilon must be finite and between 0 and 1')
    indices = answer_indices(observation)
    if epsilon == 0 or indices is None:
        return tuple(probabilities)
    values = [(1-epsilon)*float(p) for p in probabilities]
    for index in indices:
        values[index] += epsilon/2
    return tuple(values)


def behavior_log_probs(logits, observation, epsilon):
    """Differentiable log of the *actual* sampling distribution."""
    base = logits.log_softmax(-1)
    indices = answer_indices(observation)
    if epsilon == 0 or indices is None:
        return base
    if not 0 <= epsilon <= 1 or not math.isfinite(epsilon):
        raise ValueError('invalid exploration epsilon')
    # logaddexp prevents underflow when the policy is extremely confident.
    log_policy = base + math.log1p(-epsilon) if epsilon < 1 else torch.full_like(base,-float('inf'))
    log_uniform = torch.full_like(base, -float('inf'))
    for index in indices:
        log_uniform[index] = math.log(epsilon/2)
    return torch.logaddexp(log_policy,log_uniform)
