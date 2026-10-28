# Foundational proof game v2

Status: proposed language specification, not an implemented training mode.
The v1 checker, action space, checkpoints, and training commands remain v1.

See [the full pretraining-to-RL pipeline](foundational_pipeline_v2.md) for data
generation, size/split balancing, supervised training, run budgets, and evaluation.

## 1. Objective and experiment

Train a model to construct general mathematical proofs about Turing machines.
Offline generators may use simulation, cycle detection, and closed-state-set
certificates to find examples. They must compile successes into this language.
The model sees only the compiled proofs during pretraining. Subsequent RL uses
the same foundational actions, with no access to those generators.

The foundation is classical, many-sorted first-order natural deduction with
equality, natural-number induction, and finite-list induction. It is PA-style,
not a claim of a formally established equivalence to single-sorted PA. Adding
typed tapes avoids making the agent manipulate arithmetic encodings of lists.
No effective sound arithmetic foundation can prove every true statement.

The design deliberately permits the model to discover induction arguments and
intermediate lemmas. It does not require supplying hand-authored behavioral
lemmas to the model at RL time. Whether compiled demonstrations induce that
ability is the experiment, not an assumed result.

## 2. What changes from v1

In v1, `kernel.expand.same` calls `logic.normal` on both expressions. Thus
`refl`, `exact`, and `apply` can implicitly run machine computations. `reduce`,
`inject`, and `discriminate` also normalize expressions. Removing just
`closedStates` and `exactCycle` would leave this simulation route intact.

V2 has no implicit normalization in any logical rule. `refl` checks syntactic
identity. `exact` matches a hypothesis syntactically. Constructor reasoning
works only on exposed constructors. Computation requires an explicit proof step
at a specified expression occurrence.

The assertion "at time n the machine is halted" remains expressible. Removing
that assertion would remove the natural HALT theorem itself. What disappears
is automatic verification by an n-step interpreter hidden inside a proof rule.

There are no `simulate`, `closedStates`, `exactCycle`, `normalize`, `auto`, or
machine-specific invariant rules in the v2 kernel or online action space.
There is also no compiler-backed action that expands one of these on demand.

## 3. Terms, formulas, and definitions

Sorts: `Nat`, `Bit`, `List`, `State`, and `Cfg`.

Constructors:

```text
0, S(n)                  : Nat
b0, b1                   : Bit
nil, cons(b, xs)          : List
q0, ..., qH              : State  (fixed by the task)
cfg(q, left, right)       : Cfg
```

`left` lists cells left of the head, nearest first. `right` starts with the
scanned cell. Unrepresented cells are blank. Both lists are finite; trailing
blanks are allowed. This keeps the existing machine semantics.

Formulas are equality of same-sort terms, falsehood, conjunction, disjunction,
implication, and universal/existential quantification over a sort. Negation is
`P -> false`. Internally, retain de Bruijn indices and capture-avoiding
substitution; user-facing bound-variable names are display labels only.

Numerals use canonical decimal strings in serialized ASTs, stored internally
as arbitrary-precision integers; no floating-point JSON numbers. Their storage
cost is proportional to bit length. `S(17)` and `18` are not silently equal:
an explicit `numeral` equality step connects them. It checks only one successor
equation, never evaluates an arbitrary term, and charges its bit-operation cost.
Induction is still over zero/successor, not over numeral digits.

Initial definitions include addition, multiplication, head/tail with blank
defaults, list append/repeat/length/read, configuration projections, and the
machine transition and iteration functions. They provide vocabulary, not
pre-proved behavioral results. Example defining equations:

```text
add(0, y)       = y
add(S(x), y)    = S(add(x, y))
mul(0, y)       = 0
mul(S(x), y)    = add(y, mul(x, y))
append(nil, y)  = y
append(cons(b,x), y) = cons(b, append(x,y))
repeat(b, 0)    = nil
repeat(b, S(n)) = cons(b, repeat(b,n))
head(nil)      = b0
head(cons(b,t)) = b
tail(nil)      = nil
tail(cons(b,t)) = t
read(xs, 0)    = head(xs)
read(xs, S(n)) = read(tail(xs), n)
at(0, c)       = c
at(S(n), c)    = next(at(n,c))
```

The machine supplies deterministic equations for `next`, not arbitrary axioms:

```text
next(cfg(qH,l,r)) = cfg(qH,l,r)
next(cfg(qi,l,r)) = dispatch_i(head(r),l,r)     -- each non-halting state

dispatch_i(b,l,r) = cfg(qj,cons(w,l),tail(r))  -- when table[i,b]=(qj,w,R)
dispatch_i(b,l,r) = cfg(qj,tail(l),cons(head(l),cons(w,tail(r))))
                                             -- when table[i,b]=(qj,w,L)
```

The checker reconstructs these equations from the assigned transition table.
A certificate cannot override them. A transition-table lookup reveals one
transition, not the result of a sequence of transitions.

### User definitions

The model can introduce typed, parameterized names for terms or formulas.
Nonrecursive definitions may reference only earlier definitions. Structurally
recursive function definitions must designate one `Nat` or `List` argument,
cover its constructors exactly once, and recurse only on its direct predecessor
or tail. Other arguments may change. The first version forbids mutual recursion,
general fixpoints, recursive predicates, and recursion hidden behind aliases.
Definitions cannot contain free variables except their declared parameters.

This admits `repeat` and `append` without permitting `f(n)=f(n)` or a definition
that asserts its own theorem. No new axiom may be proposed by the model.
An arbitrary function identifier does not imply constructor injectivity.

## 4. Proof kernel and game actions

The following names describe the proposed interface; they are not existing CLI
commands. Every completed game action produces a bounded kernel inference or
edits a typed draft. Annotation construction uses finite menus and typed holes.

| Action family | Meaning |
| --- | --- |
| `intro`, `witness`, `split`, `left`, `right` | Standard logical introductions |
| `exact`, `apply`, `specialize`, `destruct` | Use explicit hypotheses or previously proved local lemmas |
| `refl`, `symm`, `trans`, `congr` | Syntactic reflexivity, symmetry, transitivity, congruence |
| `rewrite(h, direction, location)` | Substitute one equality at one explicitly selected occurrence |
| `unfold(definition, equation, location)` | Instantiate one defining equation at one occurrence |
| `numeral(location)` | Expose/fold one successor of a numeral using a checked equality |
| `induct_nat`, `induct_list` | Base and step subgoals, with fresh variables and induction hypothesis |
| `cases` | Constructor cases for a selected typed term |
| `inject`, `discriminate`, `false_elim`, `raa` | Constructor facts, contradiction, and classical reasoning |
| `have(statement)` | Prove an intermediate statement, then use it as a local lemma |
| `define(signature, body)` | Register a well-formed, terminating definition |
| `focus`, draft edits, `reset`, `submit` | Editor control; reset never refunds resources |

`unfold` never normalizes arguments, children, other hypotheses, or the result.
It may inspect a numeral's zero/successor shape without expanding the numeral
into a unary tree. An equation match cannot recursively evaluate a subterm to
make the match succeed. Reverse use is an explicitly instantiated equality,
not search for a matching expression. Logical matching allows only bound-name
equivalence via the internal representation, not computational conversion.

An implementation can represent `unfold` as a checked defining-equation equality
plus substitution. A kernel convenience node may fuse these two operations,
but must record and charge both. It must not fuse an unbounded sequence.

An induction hypothesis is available only in its inductive step. `have` cannot
use its own unfinished proof. The continuation gets the lemma only after that
proof closes. Lemmas proved under assumptions retain those assumptions; names
do not allow a lemma to escape its variable or hypothesis scope.

Initially, lemmas are episode-local. Every task begins with definitions but no
behavioral theorem library. Cross-episode theorem retrieval is a separate future
experiment; it must not become an unreported source of teacher assistance.

## 5. Tasks and worked examples

```text
HALT:    exists n:Nat, state(at(n,C0)) = qH
NONHALT: forall n:Nat, state(at(n,C0)) = qH -> false
C0:      cfg(q0,nil,nil)
```

These examples specify intended derivations. They are not yet executable v2
certificates, and no v2 verification or BB(5) feasibility is claimed here.

### A one-step HALT proof

Suppose the q0/blank table entry writes b1, moves right, and enters qH.
Choose witness 1. Unfold the left-hand side of the resulting equality, one
selected occurrence at a time:

```text
state(at(1,C0))
state(next(at(0,C0)))
state(next(C0))
state(dispatch_0(head(nil),nil,nil))
state(dispatch_0(b0,nil,nil))
state(cfg(qH,cons(b1,nil),tail(nil)))
state(cfg(qH,cons(b1,nil),nil))
qH
```

Here `C0` is display notation for its literal configuration. Each adjacent line
uses one defining equation. The goal is now `qH=qH`, closed by `refl`.
The checker did not interpret the witness as a request to run the machine.

### An abstraction the model can invent

Using the displayed definition of `append`, the model can propose:

```text
have forall xs:List, append(xs,nil) = xs
  induct xs
  base: unfold append(nil,nil); refl
  step b tail, IH: append(tail,nil)=tail
    unfold append(cons(b,tail),nil)
    rewrite IH inside cons(b,append(tail,nil))
    refl
```

This is a genuine all-length lemma with a constant-size inductive argument.
No `append_right_identity` rule is built in. Tape sweeps can use the same
induction structure, with more intermediate equations and hypotheses.

### A NONHALT invariant

For a machine whose q0 transitions both return to q0, propose:
`forall n, state(at(n,C0))=q0`. Prove the base using `at(0,C0)=C0`.
In the step, expose `at(S(n),C0)`. To reason about its unknown configuration,
case-split that configuration into `cfg(q,l,r)`, rewrite the induction hypothesis
to determine q, and split the scanned bit. Unfold the applicable transition
equation in each branch; both resulting state projections are q0.
Finally specialize the invariant at arbitrary n and contradict `q0=qH`.

This is ordinary induction and equality reasoning, not a privileged state-set
rule. The compiler may generate it offline; the policy must construct these
foundational steps itself online.

## 6. Offline compilation

The old checker is a data-source validator, never a v2 proof premise. Compilers
are untrusted programs: a compiler error may lose an example, but cannot certify
a theorem. The standalone v2 checker must operate with all teacher modules
uninstalled and with no dependency on `logic.normal` or `nonhalt_certificates`.

| Teacher result | Required compiled construction |
| --- | --- |
| HALT witness | Derive individual transitions using defining equations; compose their equalities into the root theorem |
| Closed state set S | Define membership as a finite disjunction; prove initial membership, preservation by state/bit cases, and exclusion of qH; use induction |
| Exact cycle with prefix | Produce checked transitions for the prefix and cycle; prove a finite configuration invariant and its preservation, then exclusion of qH by induction |

The compiler may use a known generic lemma internally only if it emits its proof
and the student's trace includes constructing that lemma. A named external
theorem, opaque proof hash, or successful v1 certificate is not a foundation.
This applies equally to trace composition and invariant induction lemmas.

Important cycle detail: v1 detects cycles with a sparse physical tape, while
list configurations can acquire trailing blanks. Equal physical tapes need not
be syntactically equal lists. V2 must not silently identify `nil` with
`cons(b0,nil)`. Define list equivalence by equality of `read(xs,i)` for all i,
and configuration equivalence by equal state and equivalent left/right lists.
The cycle compiler must include proofs of any needed tape equivalence and its
preservation by `next`. Its finite invariant uses that equivalence when literal
equality is unavailable. This is an implementation acceptance requirement,
not an assumed existing lemma. Translation-invariant relative configurations
are sufficient for NONHALT; the teacher's absolute-head cycle is stronger.

Initial compiler support is explicitly these three certificate families.
Arbitrary v1 proofs using implicit normalization are not automatically supported.
Any unsupported proof, compiler limit, or failed v2 check is recorded and excluded
from demonstrations. No HALT/NONHALT conclusion follows from a resource limit.

## 7. Certificates, accounting, and observations

Use a separate `proofgame-foundation-2` certificate format. Bind each certificate
to the full machine table, language version, definition manifest, and root
answer. Task verification also compares its table with the assigned machine.
Reconstruct all contexts and validate all definitions on standalone checking.

Allow proof DAG sharing with an acyclic dependency graph. A reference must match
the checked sequent, including context, machine, and definitions. Reuse under
different parameters requires explicit instantiation. Check a dependency before
using it. No success flags, cached booleans, or unverified theorem bodies count
as evidence. Structural sharing is an implementation technique, not an axiom.

Use separate monotonic budgets for:

- Game actions, including failed actions, edits, and resets.
- Kernel inference/equation applications, including explicit machine transitions.
- Term nodes, numeral bits, substitution/matching work, and certificate bytes.
- Observation tokens and active goals; exceeding a limit has an explicit status.

An optional 100k transition-equation budget limits direct simulation work; it is
not a maximum time that may appear in a proved theorem. Induction can prove facts
about arbitrarily large times. Transitions used symbolically count too. Failed
attempts consume work. Final standalone verification starts its own bounded
check and does not trust an in-game success cache. Report both work totals.

Observations contain the machine, definitions, visible goals, hypotheses, local
lemmas, typed draft, available structural actions, errors, and remaining budgets.
Keep a proof archive outside the attention window while exposing the statements
and scope of usable lemmas; an inspect/focus action can reveal a selected body.
Do not truncate a hypothesis needed to interpret an action. Cache keys include
all relevant context and definition versions, not just rendered goal strings.

Legal-action generation is based on syntax, sorts, scope, and equation patterns.
It must not run a simulator, search for an invariant, test proposed witnesses,
or rank candidates by a teacher's answer. Generic syntactic applicability checks
are allowed; mathematical proof search is the policy's responsibility.

## 8. Pretraining and RL

Offline sequence: generate a v1 success -> compile -> independently check v2 ->
replay as legal v2 game actions -> export observation/action training pairs.
The replayed game and the RL game must be the same implementation. Serialize
language/action versions and the initial definition manifest in the dataset.

Pretraining targets are successful foundational actions, not the teacher's
failed guesses, resets, helper commands, or hidden metadata. Teacher provenance
is stored for auditing but never passed to the model. Every proof branch is
included. A proof is not a demonstration merely because its final theorem is
known to be true: the full trace must replay and be rechecked.

Split by machine before generation, deduplicate proofs, and report near-duplicate
machine families. Keep special evaluation machines out of teacher data. Balance
short HALT, inductive NONHALT, and cycle examples so long simulation traces do
not dominate token counts. This is dataset weighting, not helper access at RL
time. Keep a record of how many successes fail compilation or exceed limits.

After pretraining, run the current group-relative RL idea against the v2 game.
Use verified terminal success and a bounded proof-length preference. Count
lemma construction in proof cost; an empty name must not make a long proof free.
Log raw success separately from length reward. Preserve sufficient action and
context budgets for compiled examples; v1's 100 actions are not an equivalent
difficulty setting after expanding helpers. Calibrate budgets using actual
compiled lengths, without changing them between experimental arms.

The primary comparison is pretrained-v2 versus randomly initialized-v2 under
identical RL/evaluation budgets. Track HALT/NONHALT coverage, all-failed groups,
proof length, verifier cost, induction use, and lemma construction/reuse.
Induction use alone is not evidence of generalization. Test held-out families
and proofs beyond the teacher generators' coverage. There is no guaranteed
emergence of abstractions, and no promised BB(5) proof within the game limits.

## 9. Implementation order and acceptance gates

1. Implement isolated `foundation` syntax, scope/type checking, small-step
   equations, and standalone kernel. First gate: explicit one-step HALT and a
   symbolic inductive list lemma; implicit HALT-via-refl must fail.
2. Implement the finite-action editor and stable action replay. Gate: the same
   foundational certificates are reachable through the actual game menus.
3. Implement offline HALT and closed-state compilers. Then implement exact-cycle
   compilation, including trailing-blank equivalence. Gate: replayed examples
   pass with teacher imports disabled; mutated tables/witnesses/proofs fail.
4. Export versioned demonstrations and implement supervised pretraining using the
   same policy/action scoring interface. Gate: reload, replay, and overfit a tiny
   held-out-from-eval training sample before large generation.
5. Add an explicit v2 epoch-training mode, compatible profiling, and the above
   controlled comparison. Existing v1 runs remain resumable with v1 semantics.

Adversarial kernel tests must reject self-referential lemmas, cyclic DAGs,
nonterminating definitions, out-of-scope variables, changed machine tables,
wrong bit branches, unproved imported lemmas, and forged equivalence of tapes.
Resource tests include huge numeral annotations and exponential term growth.
Exercise successor arithmetic, binder substitution, and scope with randomized
small cases. Check transition equations against a separate simulator on finite
examples, and replay all compiler output in the standalone kernel.

V2 policy checkpoints are versioned separately: the old action vocabulary and
proof semantics are not compatible. Do not silently resume a v1 run as v2.
This specification changes no current training command or default behavior.
