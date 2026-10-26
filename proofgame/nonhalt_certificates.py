"""Small, sound, independently checked NONHALT certificate fragments.

The policy proposes finite state sets or a repeated *complete* configuration.
No simulation cutoff is ever treated as evidence of nonhalting. The verifier
checks the exact inductive closure or a deterministic repeated configuration.
"""
from .logic import Invalid, Limit


MAX_CYCLE_STEPS = 4096


def check_closed_states(machine, states):
    if not isinstance(states, tuple) or not states or len(states) > len(machine.states):
        raise Invalid('closed-state certificate needs a nonempty tuple of states')
    if any(type(q) is not str for q in states) or len(set(states)) != len(states):
        raise Invalid('closed-state certificate has invalid or duplicate states')
    selected = set(states)
    if 'q0' not in selected or 'qH' in selected or not selected.issubset(machine.states):
        raise Invalid('closed-state certificate must include q0, exclude qH, and use machine states')
    for q in selected:
        for bit in ('b0', 'b1'):
            if machine.row(q, bit)[0] not in selected:
                raise Invalid('state set is not closed under both transition symbols')
    return True


def check_exact_cycle(machine, data):
    if not isinstance(data, tuple) or len(data) != 2:
        raise Invalid('cycle certificate needs (start, period)')
    start, period = data
    if (type(start) is not int or type(period) is not int or
            start < 0 or period <= 0):
        raise Invalid('cycle start must be nonnegative and period positive integers')
    if start + period > MAX_CYCLE_STEPS:
        raise Limit('cycle verification step limit exceeded')

    # Canonical complete configuration: state, absolute head coordinate,
    # finite set of nonblank tape cells. Blank cells are not stored.
    q, head, tape = 'q0', 0, set()
    before = None
    for step in range(start + period + 1):
        if q == 'qH':
            raise Invalid('machine halts before the claimed cycle closes')
        if step == start:
            before = (q, head, frozenset(tape))
        if step == start + period:
            if before != (q, head, frozenset(tape)):
                raise Invalid('complete configurations do not repeat')
            return True
        read = 'b1' if head in tape else 'b0'
        q2, write, direction = machine.row(q, read)
        if write == 'b1':
            tape.add(head)
        else:
            tape.discard(head)
        head += 1 if direction == 'R' else -1
        q = q2
    raise Invalid('incomplete cycle certificate')
