"""Additional fixed diagnostics. Reference information is never sent to the policy.

Cycles below were selected deterministically from random labelled tables and
verified by exact configuration recurrence. All their states are visited and
there is a halting edge, so choosing all states is not a nonhalting proof.
Growth/drift examples have the stated inductive phase invariants; we do not
claim the current policy or kernel has constructed those induction proofs.
"""


def case(identifier, name, answer, rows, explanation, **metadata):
    return dict(id=identifier, name=name, expected_answer=answer,
                rows=rows, explanation=explanation, **metadata)


FRONTIER_3 = (
    ('q1','b1','R'), ('q1','b1','R'),
    ('q2','b1','L'), ('qH','b0','L'),
    ('qH','b0','L'), ('q0','b1','R'))
FRONTIER_4 = (
    ('q1','b1','R'), ('qH','b0','L'),
    ('q2','b1','L'), ('qH','b0','L'),
    ('qH','b0','L'), ('q3','b1','R'),
    ('qH','b0','L'), ('q0','b1','R'))
NO_REFERENCE = 'Human-described invariant; no kernel-accepted reference proof is supplied.'

ADDITIONAL_BENCHMARKS = (
    case('busy_beaver_3_21', 'Busy Beaver (3-state, HALT in 21 steps)', 'HALT', (
        ('q1','b1','R'), ('qH','b1','R'),
        ('q1','b1','L'), ('q2','b0','R'),
        ('q2','b1','L'), ('q0','b1','L')),
        'First halts at transition 21.', halt_steps=21),
    case('busy_beaver_4_107', 'Busy Beaver (4-state, HALT in 107 steps)', 'HALT', (
        ('q1','b1','R'), ('q1','b1','L'),
        ('q0','b1','L'), ('q2','b0','L'),
        ('qH','b1','R'), ('q3','b1','L'),
        ('q3','b1','R'), ('q0','b0','R')),
        'First halts at transition 107.', halt_steps=107,
        reference_note='The known 37-action witness 10*10+7 exceeds the current definition-reduction budget. This does not rule out other proofs.'),
    case('growing_frontier_3', 'Growing frontier (3-state, NONHALT)', 'NONHALT', FRONTIER_3,
        'At time 3n+1: q1 at n+1, ones 0..n. Then q2 at n with ones 0..n+1; then q0 at n+1 with the same tape. The q1/1 and q2/0 halt edges are unreachable.',
        reference_note=NO_REFERENCE),
    case('guarded_cycle_3_6', 'Guarded cycle (3-state, period 6)', 'NONHALT', (
        ('q2','b1','L'), ('qH','b0','L'),
        ('q2','b0','R'), ('q0','b1','R'),
        ('q2','b1','R'), ('q1','b0','L')),
        'The complete configuration at time 1 repeats at time 7.', cycle=(1,6)),
    case('guarded_cycle_4_8', 'Guarded cycle (4-state, period 8)', 'NONHALT', (
        ('q3','b1','L'), ('q3','b1','L'),
        ('q2','b0','L'), ('q3','b1','R'),
        ('q1','b1','L'), ('qH','b1','R'),
        ('q1','b0','R'), ('q0','b0','L')),
        'The complete configuration at time 1 repeats at time 9.', cycle=(1,8)),
    case('guarded_cycle_4_10', 'Guarded cycle (4-state, period 10)', 'NONHALT', (
        ('q1','b0','R'), ('q1','b0','L'),
        ('q2','b0','R'), ('q3','b1','L'),
        ('q0','b1','L'), ('q3','b0','L'),
        ('q0','b0','L'), ('qH','b1','R')),
        'The initial complete configuration repeats at time 10.', cycle=(0,10)),
    case('growing_frontier_4', 'Paired-cell frontier (4-state, NONHALT)', 'NONHALT', FRONTIER_4,
        'At time 4n: q0 at 2n, ones 0..2n-1. Four transitions visit q1 at 2n+1 reading 0, q2 at 2n reading 1, q3 at 2n+1 reading 1, then q0 at 2n+2 with two more ones. All halt guards remain false.',
        reference_note=NO_REFERENCE),
    case('translated_cycle_3', 'Erasing drift (3-state, NONHALT)', 'NONHALT', (
        ('q1','b1','R'), ('qH','b0','L'),
        ('q2','b0','L'), ('qH','b0','L'),
        ('qH','b0','L'), ('q0','b0','R')),
        'At time 3n: q0 at n on an entirely blank tape. Three transitions write a marker, return to it, erase it, and move one cell right. This is translation, not exact recurrence.',
        reference_note=NO_REFERENCE),
    case('broken_frontier_3', 'Broken frontier (3-state, HALT in 3 steps)', 'HALT',
        FRONTIER_3[:-1] + (('qH','b1','R'),),
        'Changes the frontier q2/1 transition to HALT; first halts at transition 3.', halt_steps=3),
    case('broken_frontier_4', 'Broken frontier (4-state, HALT in 4 steps)', 'HALT',
        FRONTIER_4[:-1] + (('qH','b1','R'),),
        'Changes the paired-cell frontier q3/1 transition to HALT; first halts at transition 4.', halt_steps=4),
)
