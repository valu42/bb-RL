# Random proof pretraining followed by RL

Status: design specification, accompanying [language v2](foundational_language_v2.md).
An implementation is now available; see [the usage guide](foundation_usage.md)
for executable commands, actual artifacts, and implementation differences.
Numerical design settings below remain experimental choices, not measured results.

## 1. End-to-end contract

```text
Random 2–5-state machines, assigned to immutable splits
  -> offline teacher search using simulation / state sets / exact cycles
  -> successful teacher certificates
  -> compilation into foundational proofs
  -> independent checking
  -> replay through the actual foundational game
  -> supervised next-action pretraining
  -> pretrained policy weights
  -> existing style of group-relative RL, using only foundational actions
  -> fixed, independent evaluation machines
```

No pretrained language model or human-written proof demonstrations are required.
Hand-written compiler code necessarily encodes how the three certificate families
become proofs. This experiment tests learning from those generated proofs, not
learning from zero mathematical guidance. The student never calls the compiler,
teacher search, or a v1 certificate rule during pretraining inference or RL.

Definitions and actions are identical during demonstration replay and RL.
Behavioral lemmas start empty each episode. Any helper lemma used by a generated
proof must itself appear as foundational construction in that demonstration.
Successful teacher searches are a data source; acceptance always belongs to the
independent v2 checker. Failure to find or compile a proof is never a NONHALT label.

## 2. Machines and leakage control

Each raw transition independently chooses a next state (including HALT), a
written bit, and a direction uniformly. The labelled population for n states is
`(4*(n+1))**(2*n)`. In particular there are only 20,736 two-state tables, so a
million unique two-state examples is impossible. Varying proofs can increase
demonstrations, not the count of distinct machines.

Use four size buckets, 2/3/4/5, and separate three identities:

- `machine_id`: exact transition table, as in v1.
- `family_id`: a structural equivalence fingerprint for splitting.
- `proof_id`: language version, definition manifest, machine, and certificate.

For the family fingerprint, drop states unreachable in the transition graph
from q0 using both read symbols; canonically rename remaining states with q0 and
qH fixed; take the minimum encoding over these renamings and global L/R reflection.
This groups simple unreachable-state embeddings and mirror/renaming copies
across sizes. It is not a decision procedure for semantic equivalence. Never
flip the blank symbol. At five states there are at most 24 q0-fixing renamings
before reflection. Implement and test this fingerprint before any data generation.

Assign families by a versioned hash to train/dev/test at 80/10/10, before running
teachers. Reserve every special benchmark's family outside these partitions.
Freeze the hash salt, benchmark manifest, machine seeds, and fingerprint version.
Changing them creates a new experiment rather than silently refreshing a split.

Inside a partition, preserve uniform labelled-table sampling conditional on
partition membership and benchmark exclusions. Deduplicate exact tables, but
do not pretend that uniform tables are uniform behaviors. Report structurally
reachable state counts and repeated families as well as nominal state counts.

### Fixed RL and evaluation sets

Initial scale:

| Set | Machines | Selection |
| --- | ---: | --- |
| RL train | 1,000: 250 per size | Unfiltered train-partition tables |
| Per-epoch development eval | 400: 100 per size | Unfiltered dev-partition tables |
| Final generalization test | 400: 100 per size | Unfiltered test-partition tables |
| Special challenge eval | Existing suite plus separately frozen 5-state cases | Reserved families |

Do not require teacher success when selecting any of these sets. Pretraining may
include the RL training machines; dev/test and special families are excluded.
The final test is not used to pick a checkpoint or adjust settings. Per-epoch
"evaluation" in the existing style of logs means the development set here.
Include the BB(5) champion as an aspirational challenge, not a release gate or
a promised solvable example. Adding 5-state challenge cases requires recording
their provenance and independently verifying any claimed reference proofs.

## 3. Offline random teacher generation

Use a reproducible stream of randomly sampled machines rather than waiting for
uniformly random granular actions to stumble onto complete proofs. "Random"
describes machine sampling, search scheduling, and proof variation; deterministic
teacher algorithms are deliberately allowed offline. No student model guides
the initial corpus. Keep an optional uniform-random v1 game walk as an ablation,
not the production bottleneck.

Run cheap structural search first and a bounded simulator second. Attempt to
find multiple proof styles for a machine even after the first success, subject
to the per-machine work cap. All sources are checked and deduplicated.

### Closed-state teacher

At n <= 5, at most 16 sets contain q0 and exclude qH. Enumerate those candidates
in a seeded random order, verify closure under both symbols using the v1 helper,
and retain a small number of distinct successful sets. Do not select sets using
the eventual evaluation answer. Search produces no conclusion if no set works.

### Simulation and exact-cycle teacher

Run a reference simulator with explicit step counts, stop at HALT, and record a
complete configuration trace sufficient to reproduce a certificate. Exact-cycle
detection uses state, absolute head, and the full nonblank tape; a hash hit must
be confirmed by exact equality. A nonrepeated prefix is not evidence of NONHALT.

Use deterministic per-machine sampling of a simulation allowance:

| Maximum simulated transitions | Fraction of machines |
| ---: | ---: |
| 128 | 70% |
| 1,024 | 20% |
| 10,000 | 9% |
| 100,000 | 1% |

These are offline teacher budgets, not online proof rules. The cheap-heavy mix
avoids paying 100k steps for every unhelpful table. Track cycle candidates within
the existing v1 bound of prefix + period <= 4,096; larger cycles are unsupported
in the initial compiler rather than silently exempted from the bound. Bound
cycle-detection storage and drop detection on exhaustion while simulation may
continue; store the reason. A rolling hash alone never certifies a cycle.

On HALT, prefer the earliest halting witness to avoid teaching unnecessary
post-halt computation. Keep longer witnesses only in a small explicit diversity
ablation. On a verified cycle, emit its start/period certificate. If simulation
times out, retain an unlabeled failure record, not a negative training example.

### Scale and stopping rules

Pilot: propose up to 10,000 distinct tables per size, subject to the available
train-partition population, then compile/replay enough successes to measure the
full pipeline. The small two-state train partition may be exhausted earlier;
report that fact rather than change split assignments.

Main generation: exhaust the chosen two-state training pool and propose up to
100,000 tables each for sizes 3, 4, and 5. Scale those larger pools toward one
million proposals each only if held-out results improve and measured costs fit.
These are proposal caps, not promises of that many accepted proofs. End a job
at its configured proposal, CPU-work, or disk cap; preserve partial valid shards.

Retain at most three distinct accepted proof variants per machine initially.
Seeded choices can vary disjunction order, equivalent witnesses, and independent
subgoal order. Exclude resets and failed teacher actions from successful traces.
Do not insert meaningless proof padding to inflate the corpus.

## 4. Compilation, verification, and replay

The compiler interface returns a certificate or a typed failure reason. HALT,
closed-state, and exact-cycle compilation follow the obligations in language v2,
including explicit proofs for trailing-blank tape equivalence in cycles.
Unsupported v1 proof rules are not copied through as opaque nodes.

For every accepted item:

1. Validate the teacher certificate against the exact machine.
2. Compile into the v2 definition and proof format.
3. Independently check from the empty root context without teacher imports.
4. Linearize the proof into legal typed game edits and inference actions.
5. Replay from a fresh v2 game using the same limits as the declared corpus profile.
6. Independently recheck the submitted certificate and compare the root theorem.
7. Atomically commit the item to a shard only after all stages succeed.

Step 4 must include writing witnesses, lemma statements, and definitions. A
certificate containing a large annotation is not an executable demonstration
unless the model could actually construct that annotation through its menus.
Unsupported/over-budget items remain auditable but never enter supervised loss.
Do not shorten a proof by replacing a subproof with a teacher-only command.

Store the compact certificate, action stream, hashes, length/cost summary, and
periodic replay snapshots. Reconstruct observations near sampled positions on
demand with a bounded LRU cache. Do not store a full growing proof tree at every
step, which would multiply storage and encoding costs. Snapshots are acceleration
artifacts bound to the game version and checked replay, not trusted proof inputs.

Manifest fields include split and seed versions, compiler/kernel/editor versions,
definition and action-schema hashes, all resource limits, counts by size/source/
answer/length, failed-stage counts, and unique machine/family/proof counts.
Teacher metadata is accessible to the data pipeline for sampling and auditing,
but absent from model observations.

## 5. Budgets and curriculum calibration

First collect compiled trace lengths, maximum active observation sizes, and
verification work. Select the smallest action cap from 256/512/1024/2048/4096
covering at least 90% of the pilot's shortest accepted proof per machine, using
equal size weighting. If even 4096 does not cover it, retain the cap and report
excluded coverage; do not silently expand the model or shrink the task set.
Recheck and replay the selected corpus under the frozen profile.

Start with the existing 4096-token model input cap. The v2 observation design
keeps completed proof bodies archived, with visible lemma statements and scopes.
If the selected goal/context still cannot fit, report an observation-limit
failure. No silent truncation of variables, hypotheses, or legal actions.

Freeze the action, token, numeral-bit, term-size, verifier-work, and transition
budgets before comparisons. Training-time failures and resets consume work.
Longer teacher simulations will often compile to inadmissibly long proofs; record
that yield explicitly. A 100k teacher allowance does not make a 100k-step trace
fit a 100-action or 4096-action student episode.

Pretraining curriculum: start with the shortest admissible quartile, add the
middle half, then train on the full admissible corpus. Preserve size and proof
family diversity in each stage where data exists. This curriculum changes
sampling, not the language or the final evaluation budget.

## 6. Supervised pretraining

Reuse the AST policy's state/candidate-action scoring interface with a v2
vocabulary and observation encoder. Start with width 64, two layers, four heads
as an experimental baseline; these are not validated capacity requirements.
Every prediction is over the legal v2 menu, including expression-edit actions.
The target is the next replayed action. Optimize categorical cross-entropy.

Sample size uniformly over 2/3/4/5, then HALT/NONHALT approximately evenly within
available strata. Within NONHALT, balance closed-state and cycle sources when
both are available. Sample machines uniformly in the chosen stratum, then one
of their proofs, then a position. This stops a long trace or a machine with many
variants from dominating training. Report missing strata and redistribution;
never fabricate examples to satisfy a quota.

Ensure terminal submit steps and completed lemma constructions are encountered
through ordinary replay-position sampling; log their exposure counts. Report
accuracy separately for rule choice, expression construction, and initial
HALT/NONHALT choice. High teacher-forced accuracy is not proof-search success.

Initial optimizer: AdamW, learning rate 3e-4, weight decay 0, gradient clipping
at norm 1; use the existing finite-gradient/update guards. Effective batch size
64 observations via memory-bounded microbatches. Pilot for 1 million sampled
action targets; main run for 10 million, then optionally 100 million if autonomous
dev performance continues improving. These count sampled exposures, not unique
proofs or machines. Save data coverage and repeat rates alongside the count.

At fixed update intervals, measure both loss on a disjoint compiled dev corpus
and autonomous v2 proof attempts on the unfiltered dev machines. Dev teacher
proofs may be generated for diagnostic loss but never train the policy or appear
in autonomous observations. Choose the checkpoint by autonomous dev coverage,
with a fixed tie-break by proof cost; retain the final checkpoint too. Use equal
evaluation budgets across checkpoints and experimental arms.

## 7. Handoff to the normal RL loop

Start a new v2 RL run initialized from the selected pretrained policy weights.
Check architecture, language, action-schema, and definition-manifest compatibility.
Initialize a fresh RL optimizer; do not carry supervised optimizer moments into
the different objective. Record the exact pretraining checkpoint hash. Resuming
that RL run restores its own optimizer, RNG, data order, and scheduler state.

Retain the existing overall setup initially:

| Setting | Initial choice |
| --- | --- |
| Training machines | Fixed 1,000, balanced across 2–5 states |
| Machine batch size | 8 |
| Rollout workers | 8, adjusted after measured profiling |
| Branch exploration | 8 attempts per answer per round, up to 6 rounds |
| Objective | Current group-relative verified-proof reward |
| Length penalty | 0.3 initially, using the frozen v2 action budget |
| Epoch target | 100 as a maximum, with saved checkpoints and dev monitoring |
| Dev attempts | 100 per machine, same cap for every compared policy |
| Terminal output | Compact cumulative and latest-batch proof statistics |

The first answer is behavior-balanced in training as in the current branch race;
record raw model probabilities separately. Autonomous dev/test uses the policy's
answer choices. Group-relative advantages must use the actual foundational
trajectories, including failed actions; no teacher success labels or compiler
calls enter rollout or reward. A completed proof is rewarded only after checking.

Preserve existing branch-round stopping/group semantics, checkpointing, progress
bars, and profiling. The language, dataset state range, action budget, and
pretrained initialization are the intended changes. No supervised replay loss
during this first RL experiment: it is pretraining followed by RL, as requested.
If all-failed groups dominate, report that explicitly before considering a
separately versioned curriculum or replay-assisted variant.

## 8. Evaluation and evidence of abstraction

Run at least these two arms with the same v2 language and RL budgets:

- Random initialization -> RL.
- Compiled-proof pretraining -> RL.

Also evaluate the pretrained checkpoint before RL, so a gain from imitation is
not misattributed to RL. Use three seeds for a substantive comparison after the
single-seed engineering pilot. Report the pretraining compute separately from
RL compute; this is not an equal-total-compute comparison unless explicitly run
as one. Optionally give the scratch arm extra compute for that separate question.

Report per size and overall: HALT/NONHALT proved, unproved, first verified proof
actions, verifier work/time, proofs per wall-clock hour, and unique machines.
Report all-failed/informative groups, rejection/limit reasons, and the distribution
of proof lengths. Nonproof is not a known ground-truth negative.

Track induction, lemma creation/reuse, and the fraction of successful proofs
using fewer explicit transitions than a direct trace. These are diagnostics,
not rewards or sufficient evidence of reasoning. Stronger evidence comes from
held-out behavior families and success where the bounded teachers fail. Record
teacher failure by the exact search budget; it is not a claim of impossibility.

Final test and frozen special evaluation are run at predefined checkpoints
(pretrained baseline and chosen final RL checkpoint). The per-epoch dev curve is
the monitoring tool. Report separately whether a result concerns novel tables,
structural families, or previously unseen behavior; do not conflate them.

## 9. Execution, storage, and monitoring

The offline stages use CPU process pools with bounded queues:
generation -> teacher search -> compilation -> check/replay -> shard writer.
Assign deterministic job IDs/seeds so worker count and completion order do not
change split membership or samples. Each job has a work budget and idempotent
output identity. The main writer commits manifests atomically; interrupted jobs
resume without duplicating accepted examples or forgetting failures.

Do not run saturated teacher generation beside a measured RL benchmark. Allow
explicit worker caps when sharing a machine. SFT device choice is a measured
CPU/MPS comparison using the same batches; do not assume MPS wins on small states.

Proposed run artifacts:

```text
experiment.json             # frozen configuration and source versions
splits/                     # immutable RL/dev/test machine manifests
teacher/jobs.jsonl           # successes, failures, resource usage
corpus/manifest.json         # committed shards and coverage summaries
corpus/shard_*.jsonl.gz      # certificates, actions, bounded replay snapshots
pretrain/progress.json
pretrain/metrics.jsonl
pretrain/checkpoint_*.pt
rl/config.json
rl/progress.json
rl/epoch_*_training.json
rl/epoch_*_evaluation.json
rl/checkpoint files          # existing atomic run/checkpoint conventions
```

Generation logs: proposed tables, unique families, successes by source, compile
and replay acceptance, actions stored, bytes, queue sizes, CPU time by stage,
throughput and ETA. Use explicit counters for distinct data versus sampled
training exposures. SFT logs: throughput, loss/accuracy by action category,
data reuse, autonomous coverage, learning rate, and encoding/backward time.
RL retains existing profiles plus v2 checker-work and proof-limit statistics.

## 10. Deliverable order and release gates

1. V2 kernel/editor and independent verifier, with adversarial scope/definition/
   resource tests and explicit inductive examples from the language specification.
2. HALT and closed-state compilers with exact game replay; exact-cycle compiler
   follows once tape-equivalence proofs are verified. No opaque placeholder rules.
3. Versioned sampler/splits and resumable generation; pilot measured end to end.
4. Streaming demonstrations, supervised trainer, compatible initialization path,
   and an overfit-small-corpus test plus autonomous evaluation.
5. V2 adapter for the existing epoch loop, independent worker modes, profiling,
   checkpoint/resume equivalence, and tests that teacher modules are unavailable.
6. A reproducible pilot report, then the larger generation/pretraining/RL run.

Every accepted demonstration must be independently checked and legally replayed.
Before scaling, measure generation yield, memory/disk use, maximum proof/context
size, autonomous success, and whole-pipeline timing. Do not invent a training
duration until those measurements exist. Old v1 runs continue to use their old
language and cannot silently load into v2. The implemented `python -m proofgame.foundation run` orchestrates the stages;
see the usage guide for its explicit configuration and resume contract.
