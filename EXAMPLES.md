# Machines with short checked proofs

All examples start in state A on an all-zero tape. A, B, C correspond to `q0`,
`q1`, `q2`; H corresponds to `qH`. A row gives the symbol written, head movement,
and next state. H is not counted as a working state.

These are hand-written explanation and checker examples, not model discoveries.
The learning code never reads their proofs or action sequences. The action
counts below are lengths of these checked constructions, not proven minima.

## Two states, halts after two steps

| Current state | Read | Write | Move | Next state |
|---|---|---|---|---|
| A | 0 | 1 | Right | B |
| A | 1 | 1 | Right | B |
| B | 0 | 0 | Left | H |
| B | 1 | 1 | Left | H |

Initially A reads 0. It writes 1, moves right onto another blank cell, and enters
B. B reads 0, writes 0, moves left, and enters H. Thus the control states at
times 0, 1, and 2 are A, B, H.

The proof supplies the existential time witness `s(s(z))`, then checks
`qof(at(s(s(z)), C0)) = qH` by unfolding the fixed definitions.

```text
choose:HALT
rule:exI
put:s
put:s
put:z
commit
rule:refl
submit
```

**8 actions.** Run `python -m proofgame demo two_step`.

## Three states, halts after three steps

Each row below covers both read symbols, 0 and 1.

| Current state | Read | Write | Move | Next state |
|---|---|---|---|---|
| A | 0 or 1 | 1 | Right | B |
| B | 0 or 1 | 1 | Right | C |
| C | 0 or 1 | 1 | Right | H |

The control-state sequence is A, B, C, H. Supply `s(s(s(z)))` as the time
witness; the remaining equality follows by fixed computation.

**9 actions.** The sequence is the previous example with one additional `put:s`.
Run `python -m proofgame demo three_step`.

## One state, runs forever despite having a halt transition

| Current state | Read | Write | Move | Next state |
|---|---|---|---|---|
| A | 0 | 1 | Right | A |
| A | 1 | 1 | Right | H |

The machine leaves a growing string of ones behind its head. It always moves
right, so its next scanned cell is fresh blank tape. It never reads one of the
ones it wrote. The second row is therefore unreachable from the blank start.

The precise invariant used by the checked proof is:

```text
For every n, there exists a list L such that
    at(n, C0) = cfg(A, L, nil).
```

`nil` on the right means the scanned cell and all cells to its right are blank.
The list L records cells behind the head; the invariant does not need to say
exactly how many ones it contains.

- At time zero, choose L = nil.
- Suppose the invariant holds at time n. The machine is in A and reads zero,
  so the first row applies. Its next configuration is `cfg(A, cons(1,L), nil)`.
  Choose `cons(1,L)` as the next witness. This proves the induction step.
- Every finite-time configuration therefore has state A, which is distinct
  from H. Hence the machine never halts.

**39 actions**, including constructing the invariant from syntax, proving it,
and deriving the final non-halting statement. Run
`python -m proofgame demo blank_sweeper`.

The proof is short even though the execution is infinite. The model must still
discover the invariant and the choices needed to prove it.

## Two states, alternates forever

Each row covers both read symbols.

| Current state | Read | Write | Move | Next state |
|---|---|---|---|---|
| A | 0 or 1 | 1 | Right | B |
| B | 0 or 1 | 1 | Right | A |

Initially the state is A. Every transition from A or B returns to A or B.
Induction therefore keeps the state in {A,B}, and H is never reached.

The current checked construction proves the non-halting goal directly by
induction, then explicitly splits on the possible previous state and scanned
bit. Its H branch contradicts the induction hypothesis.

**49 actions.** Run `python -m proofgame demo two_state_forever`.

## Short in which sense?

| Example | Checked editor actions | Reason the proof finishes |
|---|---:|---|
| One-step halt, original fixture | 7 | Concrete time witness 1 |
| Two-state halt | 8 | Concrete time witness 2 |
| Three-state halt | 9 | Concrete time witness 3 |
| Blank sweeper | 39 | Inductive configuration invariant |
| Two-state alternation | 49 | Induction and finite cases |

These counts use this prototype's definitional reduction: `refl` may perform
several fixed computation steps internally. They count model decisions, not
individual kernel reductions or CPU instructions. They are useful for judging
the agent's exploration problem, but should not be mistaken for a universal
measure of mathematical proof complexity.

All serialized certificates and exact action traces are in `examples/`.
