"""Evaluation panels and the settings that may change on training resume."""
from .data import digest


# The full/final-test allowance (eval_attempts) stays immutable on resume.
# Learning, proof budgets, datasets, and model settings are never relaxed here.
MUTABLE_EVALUATION = frozenset({
    'eval_every', 'eval_machines', 'pretrain_eval_attempts',
    'rl_eval_machines', 'rl_eval_attempts', 'full_eval_every',
})


def training_config(config):
    return {k: v for k, v in config.items() if k not in MUTABLE_EVALUATION}


def evaluation_changes(old, new):
    return {k: {'old': old.get(k), 'new': new.get(k)}
            for k in sorted(MUTABLE_EVALUATION) if old.get(k) != new.get(k)}


def balanced_prefix(rows, count):
    buckets = {}
    for row in rows:
        buckets.setdefault(row['states'], []).append(row)
    result = []
    while buckets and len(result) < count:
        for n in sorted(buckets):
            result.append(buckets[n].pop(0))
            if not buckets[n]:
                del buckets[n]
            if len(result) == count:
                break
    return result


def profile(config, kind, rows, attempts):
    return {'kind': kind, 'machines': len(rows), 'attempts_per_machine': attempts,
            'action_budget': config['budget'],
            'panel_sha256': digest([r.get('machine_id', digest(r['machine'])) for r in rows])}


def pretrain_panel(config, dev):
    rows = balanced_prefix(dev, config['eval_machines'])
    # Missing fields denote the original shared-budget checkpoint format.
    attempts = config.get('pretrain_eval_attempts', config['eval_attempts'])
    return rows, profile(config, 'pretrain_dev', rows, attempts)


def rl_panel(config, dev, epoch):
    interval = config.get('full_eval_every')
    full = interval is None or (epoch > 0 and epoch % interval == 0)
    rows = list(dev) if full else balanced_prefix(dev, config['rl_eval_machines'])
    attempts = config['eval_attempts'] if full else config['rl_eval_attempts']
    return rows, profile(config, 'full' if full else 'quick', rows, attempts)
