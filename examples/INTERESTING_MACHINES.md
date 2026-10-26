# Interesting Turing machines for bb-RL

The fixture catalogue is [interesting_machines.json](interesting_machines.json).
Each machine is also available as its own `examples/interesting/*.json`
file, directly usable by `python -m proofgame play --machine ...`
and `python -m proofgame.evaluate ... --machine ...`.

Conventions: all tapes initially contain `b0`, the head starts at position
0 in `q0`, `qH` is the halt state, and rows are `q0/b0`, `q0/b1`,
`q1/b0`, `q1/b1`, etc. A row is
`[next_state, written_symbol, L_or_R]`.

| Example | States | Observed result | Challenge |
| --- | ---: | --- | --- |
| `parity_trap_2` | 2 | Does not halt (invariant below) | A 2-phase orbit; the q1/b1 halt case is syntactically available but unreachable |
| `growing_frontier_3` | 3 | Does not halt (invariant below) | A 3-phase induction; the marked tape grows without bound |
| `busy_beaver_2_6` | 2 | HALT after 6 transitions | Short verified HALT proof control |
| `busy_beaver_3_21` | 3 | HALT after 21 transitions | Longer verified HALT proof control |
| `busy_beaver_4_107` | 4 | HALT after 107 transitions | More than 100 machine steps; a compact proof expression is required |

## Two-state parity trap

| State, read | Write | Move | Next |
| --- | --- | --- | --- |
| q0, b0 | b1 | R | q1 |
| q0, b1 | b1 | R | q1 |
| q1, b0 | b0 | L | q0 |
| q1, b1 | b1 | R | qH |

After step 1, the tape has a single `b1` at position 0 and the
machine is `(q1, head=1)`. After step 2, the tape remains unchanged and
the machine is `(q0, head=0)`. These **entire configurations alternate
forever**. The only halting transition is `q1` reading `b1`, but q1
always scans the untouched cell at position 1, which is `b0`.

This is an exact 2-periodic orbit after initialization. The parity proof is
a straightforward two-case induction. Simulating 2,000 steps checks the
claim empirically; the argument above proves it for all steps.

## Three-state growing frontier

| State, read | Write | Move | Next |
| --- | --- | --- | --- |
| q0, b0 | b1 | R | q1 |
| q0, b1 | b1 | R | q1 |
| q1, b0 | b1 | L | q2 |
| q1, b1 | b0 | L | qH |
| q2, b0 | b0 | L | qH |
| q2, b1 | b1 | R | q0 |

For every integer `n >= 0`, the complete configuration follows these
three cases:

* After `3n+1` transitions: state `q1`, head `n+1`, exactly the tape
  cells `0,...,n` contain `b1`.
* After `3n+2` transitions: state `q2`, head `n`, exactly the cells
  `0,...,n+1` contain `b1`.
* After `3n+3` transitions: state `q0`, head `n+1`, exactly the cells
  `0,...,n+1` contain `b1`.

One three-transition cycle advances the frontier by one tape cell, so
these conditions hold by induction. Thus **q1 always reads b0, q2
always reads b1**, and neither halting transition is reachable. Unlike
the two-state trap, the full machine configuration never repeats:
the marked tape grows indefinitely.

## Busy beavers: 6, 21, and 107 transitions

The two- and three-state cases can be proved with a direct unary time
witness: a proof takes `t+6` proof actions, respectively **12** and
**27**. The tests construct these trajectories and submit them to the
project's formal certificate checker.

The four-state busy beaver genuinely halts on transition **107**, producing
13 ones on the tape. A fully unary witness would need **113 proof-game
actions**, so it will not fit the 100-action budget.

The existing language does have `add`, `mul`, `s`, and `z`.
A direct attempt to express the time witness as
`107 = 10 * 10 + 7` requires just **37 proof actions**, including
submission. This is **not automatically a successful proof**: the
checker must reduce the expression and the 107-step machine execution
within its own limits. The test probes the attempt and reports checker
errors. Don't count this machine as solved under the 100-action budget
unless the formal kernel accepts the certificate.

## Verified NONHALT proof: a two-state control

The project now has an **actual kernel-checked NONHALT proof**, not just a
mathematical explanation. The regression
`tests/test_nonhalt_proof_feasibility.py` constructs a proof certificate and
replays its **39 proof-game actions** against the real action menu, with
`budget=100`. Both `Game` and `kernel.check` accept it.

The control machine has two states:

| State, read | Write | Move | Next |
| --- | --- | --- | --- |
| q0, b0 | b1 | R | q0 |
| q0, b1 | b1 | R | q0 |
| q1, b0 | b0 | R | qH |
| q1, b1 | b1 | R | qH |

It proves the stronger invariant `forall n, qof(at n C0) = q0` using
`natI`, `casesBit`, `rewrite`, and `refl`, then proves NONHALT
using `cut`, `allI`, `impI`, `specialize`, and `discriminate`.
This demonstrates that NONHALT proofs **are expressible, constructible and
rewarded** in the current environment.

It does **not** demonstrate that the random policy discovers such proofs,
nor that the more challenging **parity trap** or **growing frontier**
invariants have been formalized under the same action budget. Those remain
separate research questions.

Run the exact constructive test:

```bash
python -m unittest discover -s tests -p 'test_nonhalt_proof_feasibility.py' -v
```

## Formal-proof status versus mathematical nonhalting

The two NONHALT machines have complete **human mathematical invariants**
given above, but we have **not** constructed replayable NONHALT proof
certificates in the current bb-RL proof language. They are valuable
challenges for whether the learned agent can build inductive proofs.

The first two busy-beaver examples **do** have tested, kernel-checked
HALT certificates. The 4-state one is confirmed by direct simulation;
its 100-action proof status must be evaluated separately.

Run all checks:

```bash
python -m unittest discover -s tests -p 'test_interesting_machines.py' -v
```

Run one machine interactively:

```bash
python -m proofgame play --machine examples/interesting/growing_frontier_3.json
```

Evaluate an existing checkpoint on a machine:

```bash
python -m proofgame.evaluate runs/my_run/model_epoch_01.pt \
  --machine examples/interesting/busy_beaver_4_107.json \
  --budget 100 --sample-actions
```

Sources for the busy-beaver machines:
[2-state notation](https://wiki.bbchallenge.org/wiki/Introduction_to_the_Busy_Beaver_Function),
[3-state 21-step trace](https://turbotm.de/~heiner/BB/simTM32_bbS.html),
[4-state 107-step champion](https://wiki.bbchallenge.org/wiki/BB%284%29).

