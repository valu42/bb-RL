# Proof Game — runnable research prototype

This is the first executable version of the idea we discussed: a model starts
with random weights, sees a small Turing machine, and builds a checked proof of
halting or non-halting through a finite menu of actions.

The proof environment works. The Transformer and training loop run. **This is
not yet evidence that the model learns to prove machines.** Included example
proofs are human-written verification fixtures; the trainer never loads them.


## Clone, install, and track evaluation

```bash
git clone https://github.com/valu42/bb-RL.git
cd bb-RL
poetry install --extras train --no-root
poetry run python -m proofgame.epoch_train --dataset runs/epoch_dataset_1000_100/dataset.json --out runs/my_10_epoch_run --epochs 10 --attempts 100 --batch-size 8 --workers 8
```

Use Poetry 2.x. Once cloned, use `git pull --ff-only` to get updates, then rerun
`poetry install --extras train --no-root` if dependencies changed.

A fresh run evaluates the untrained model **before epoch 1**, then again after
every training epoch. Evaluation never updates weights. Each evaluation prints:

```text
Baseline (before training): machines solved 4/100 (4.00%); successful attempts 4/10000 (0.04%)
```

This is an example of the output format. **Machines solved (%)** is the fraction
of held-out machines with at least one checked proof within the full attempt
budget. **Successful attempts (%)** counts individual successful attempts, so it
also measures consistency. Repeated successes count once in the first metric
and individually in the second. Neither metric includes the separate diagnostic
challenge machines below.

`evaluations.csv` in the run directory records baseline (epoch 0) and every
completed evaluation. It is updated immediately after each evaluation and rebuilt
without duplicate rows on resume. Per-epoch JSON files and `epoch_summary` log
records also include `machines_solved_pct` and `attempt_success_pct` on a 0–100
scale; existing fraction fields retain their 0–1 scale. A resumed run does not
repeat an already completed baseline.

## Inspect what the policy actually does (every 10th batch)

The epoch trainer now automatically captures **one verified successful
proof** and **one failed attempt on an unproved machine** from every
10th batch in both training and evaluation (when such examples exist).
If every machine in the batch was proved, the failure sample is instead
a failed attempt from a proved machine and is explicitly labeled. If no
example exists, the report says so rather than fabricating one.

The complete action-by-action trace includes:

- The machine's transition table, attempt number and final status.
- Every selected action, its **actual sampling probability**, and the
  model's raw probability (different for forced initial HALT/NONHALT).
- The top three alternative actions with probabilities at every step.
- The current proof goal or partially built expression, checker errors,
  and final verified certificate answer for successful proofs.

These are the **actual trajectories under the frozen rollout policy before
the optimizer update**, not trajectories reconstructed afterward using
different weights. Trace capture runs only on every 10th batch to limit
overhead; the diagnostics never enter gradient updates.

The human-readable report is printed in the terminal and saved to
`policy_traces/epoch_01_train_batch_0010.txt`, with matching JSON for
programmatic analysis. Evaluation batches use `eval` in the filename.
For example:

```text
  1. choose:NONHALT — 100.00% [FORCED] (raw model 1.30%)
     Top choices: choose:NONHALT 100.0%, choose:HALT 0.0%, reset 0.0%
  2. rule:closedStates — 28.40% (raw model 28.40%)
     Goal: (all Nat ...)
     Top choices: rule:closedStates 28.4%, rule:cut 21.0%, rule:exactCycle 18.1%
```

These example numbers are illustrative. The trace probabilities are
action-selection probabilities, **not** probabilities that a proof claim is
mathematically true. No new CLI flag is required.

## Eight-attempt group-relative learning across 12,500 machines

This is an **opt-in experimental learning algorithm**. Each machine gets a
complete group of eight attempts under the same frozen policy, even if an
early attempt succeeds. Unlike the old balanced mode, group-relative training
**does not stop after the first verified proof**. With
`--balanced-initial-answers`, four attempts are forced to start with HALT
and four with NONHALT; these forced first actions have no policy gradient.

A verified proof of length `L` gets reward `1 - alpha * L / B`, where
`B` is the action budget and `alpha` is `--length-penalty`.
Failures receive reward zero. For a group of `G=8` attempts, each
trajectory gets the leave-one-out advantage

```text
advantage_i = reward_i - (sum(rewards) - reward_i) / (G - 1)
```

Successful trajectories may have positive or negative advantage relative to
other attempts; failed trajectories in mixed groups have negative advantage.
The on-policy REINFORCE gradient includes **both successful and failed
trajectories** with their signed advantages. Groups where every attempt fails
or all rewards are equal produce no update. There is no PPO clipping,
learned value function, supervised proof, or curriculum. The trusted
checker and proof language remain unchanged.

To preserve the **exact same 100 held-out machines** from your previous
1,000/100 dataset while sampling 12,500 distinct training machines,
generate a new dataset (the original file is never overwritten):

```bash
python -m proofgame.unfiltered_dataset \
  --train 12500 --test 100 --budget 100 --seed 20261026 \
  --test-from runs/unfiltered_1000_100/dataset.json \
  --out runs/group_12500_100/dataset.json
```

Training is eight attempts per machine, but **evaluation remains 100
unforced attempts per machine** (including parity trap and Busy Beaver):

```bash
python -m proofgame.epoch_train \
  --dataset runs/group_12500_100/dataset.json \
  --out runs/ast_group_relative_8 \
  --epochs 100 --attempts 8 --eval-attempts 100 \
  --batch-size 8 --workers 8 \
  --representation ast --device cpu \
  --balanced-initial-answers --group-relative \
  --cpu-shared-action-cache --length-penalty 0.3 --profile
```

The fixed test machines **and the separate parity-trap and Busy Beaver
benchmark machines** are excluded from training. The 12,500 train
machines are sampled uniformly without replacement from the complement.
The dataset generator stores and the trainer revalidates the held-out
ranks and deterministic sampling seed. This keeps evaluation directly
comparable to previous runs without leaking true machine labels.

Every training batch logs the number of informative groups, all-failure
groups, and mean group reward. The training epoch summary also contains
`group_relative_stats`, including informative-group percentage and
positive/negative-advantage attempt counts. Complete failed trajectories
are used for gradient replay and then **discarded before batch checkpoint
serialization** to avoid enormous files. The normal 10th-batch human
readable policy traces remain available.

This changes the objective and exploration schedule: **start a fresh
training run** rather than resuming a 100-attempt REINFORCE experiment.
The older balanced REINFORCE mode stops after its first success,
so simply removing `--group-relative` does **not** produce a
matched-eight-attempt control. Compare by total proof-game actions or
wall time, not only epochs, and interpret algorithmic comparisons
accordingly. The new objective may perform worse on rare, hard proofs.

Tests:

```bash
python -m unittest discover -s tests -p 'test_group_relative.py' -v
python -m unittest discover -s tests -v
```

## Reward shorter verified proof trajectories

The epoch trainer supports an optional terminal-only length penalty:

```text
--length-penalty 0.3
```

For a verified proof with `L` actions and a per-attempt action budget
`B`, the **training** reward is `1 - 0.3 * L / B`.
Unsuccessful attempts still receive **zero** reward. For a budget of 100,
a five-action verified proof receives 0.985 and a 90-action proof
receives 0.73. The formal checker remains the sole source of proof
validity; there are no intermediate rewards, hand-written demonstrations,
or changes to proof search. The action count is the **whole trajectory**
including detours, resets, and failed checks, not the final certificate
tree size.

This scales each successful trajectory's on-policy REINFORCE log-probability
contribution by its own reward. Shared observations are aggregated using
reward-weighted counts, and the loss remains normalized by the number of
attempts. Forced balanced first actions continue to contribute zero
gradient. The coefficient must be finite and in `[0,1)`; default
`--length-penalty 0` preserves the original reward and optimizer.
The selected value is stored in `config.json`, and changing it on
resume is refused to keep experiments comparable.

To combine the previously measured faster CPU inference mode with a
shorter-proof incentive, **start a new run**:

```bash
python -m proofgame.epoch_train \
  --dataset runs/unfiltered_1000_100/dataset.json \
  --out runs/ast_short_reward_cache \
  --epochs 100 --attempts 100 --batch-size 8 --workers 8 \
  --representation ast --device cpu --balanced-initial-answers \
  --cpu-shared-action-cache --length-penalty 0.3 --profile
```

Training batch JSON and progress logs include
`mean_verified_proof_actions` and `mean_verified_proof_reward`
for verified training successes; evaluation coverage remains measured
using the unchanged verifier. Compare this new run against a
`--length-penalty 0` control with the same dataset, seed, and inference
mode. Shorter proofs are encouraged, **not guaranteed**; coverage on
harder machines could decrease if the coefficient is too strong.

Tests:

```bash
python -m unittest discover -s tests -p 'test_short_proof_reward.py' -v
python -m unittest discover -s tests -v
```

## Experimental CPU inference optimizations (benchmark first)

The CPU rollout path now offers two **opt-in**, mutually exclusive
performance modes. The default runner and reward/optimizer are unchanged.

1. `--cpu-shared-action-cache`: each CPU worker retains up to 4,096
   encoded action features across machines **only while the frozen policy
   version is unchanged**. On a policy update, the cache is cleared
   before loading the new weights. Observation probabilities are still
   cached per machine. This mainly helps evaluation or training batches
   that do not update the policy; when each worker sees just one machine
   per policy version, the benefit may be small.
2. `--cpu-central-inference`: reuses the existing central batched
   inference engine on **CPU**, collecting simultaneous observations
   from workers, encoding identical action descriptions only once
   per policy version, and batching context embeddings. This can help
   across machines in the same batch, but queues/IPC can make it
   slower on small networks. Use `--gpu-inference-batch-size` to set
   the central CPU batch size (default 8); the historical flag name
   is retained for compatibility.

First benchmark all three paths against the **same frozen checkpoint**,
machines, and random seeds, without modifying the training run:

```bash
python -m unittest discover -s tests -p 'test_cpu_inference_optimizations.py' -v

python -m proofgame.benchmark_cpu_inference \
  --dataset runs/unfiltered_1000_100/dataset.json \
  --model runs/ast_masked_actions/state.pt \
  --machines 24 --attempts 10 --workers 8 \
  --batch-size 8 --inference-batch-size 8 \
  --out runs/cpu_inference_benchmark.json
```

The benchmark prints each mode's actions/second, proof counts, attempt
counts, and speed relative to baseline. It excludes worker startup from
the measured rollout wall time, but includes queue and inference work.
Stochastic trajectories can differ slightly because batched numerical
operations are not guaranteed bitwise identical; compare both throughput
and work counts. Run multiple times and prefer the **measured** winner.

To run a new CPU training experiment with the winner, add exactly one
of these flags to your existing `epoch_train` command:

```text
--cpu-shared-action-cache
--cpu-central-inference --gpu-inference-batch-size 8
```

The selected mode is saved in `config.json`; use a **new output
directory** to compare it with an existing experiment. Do not enable
a different inference mode partway through a resumed run. Neither
optimization changes proof verification, training rewards, or model
architecture. Existing MPS behavior remains unchanged.

## Profile training before optimizing

Pass `--profile` to `proofgame.epoch_train` to measure where time goes,
without changing the policy, reward, action sampling, or optimizer.
This is optional and can be enabled when **resuming an existing run**
with `--resume`, using all the same training arguments:

```bash
python -m proofgame.epoch_train \
  --dataset runs/unfiltered_1000_100/dataset.json \
  --out runs/ast_masked_actions --resume \
  --epochs 100 --attempts 100 --batch-size 8 --workers 8 \
  --representation ast --device cpu --balanced-initial-answers --profile
```

A new process is needed to pick up code changes. The existing process
cannot enable profiling without a restart. Do not run two processes
against the same output directory.

Each batch prints a `batch_profile` JSON event and saves detailed data
under `<run>/profiling/epoch_01_train_batch_0010.json`. Each phase also
writes an `epoch_01_train_summary.json`. Baseline files use phase
`baseline`, and evaluation files use `eval`.

- **Stage wall time:** worker model snapshot, rollout, gradient learning,
  policy-trace output, batch JSON, and checkpoint writing.
- **Worker sum time:** constructing observations, waiting for model
  inference, executing proof actions, verifying successful certificates,
  and loading the CPU worker's policy.
- **Throughput:** actual proof-game actions, attempts, and machines per
  rollout wall-clock second.
- **MPS only:** changes in centralized inference requests, cache hits,
  network batches, and central inference wall time.

Important interpretation: worker time is **summed across parallel
machines**, so it may exceed the batch's rollout wall time. On MPS,
`worker_inference_seconds_sum` includes queue/IPC waiting, while
`central_inference_delta.inference_wall_seconds` measures the main
process's inference path. Do not add these to the wall-clock totals.
GPU execution can be asynchronous; the central inference path includes
its existing transfer/synchronization, but this is **not a device kernel
profiler**.

Profiling adds a small amount of overhead from per-action timers and
file writes. Compare throughput with the flag enabled under consistent
workloads, and use the timings to choose which optimization to attempt
next. It does not alter existing training batch JSON files or checkpoint
formats. CPU workers currently reload frozen model weights for each
machine; the `worker_model_load_seconds` counter helps quantify this.

## State-based masking of redundant proof checks

The proof game prevents repeated failed verification of an **unchanged**
candidate, without giving the model an action history or revealing the
correct answer.

- `cert:verify` appears only when a state-set candidate is nonempty, or
  when both recurrence fields have been supplied. These are syntactic
  readiness checks; the menu does **not** test whether the certificate is
  mathematically valid.
- If the kernel rejects a candidate, the current candidate receives an
  observable `verify_failed: true` flag and `cert:verify` is hidden
  until the candidate is edited. Changing a state-set member or a
  recurrence parameter resets that flag. Cancelling or resetting
  discards the candidate. Choosing the same recurrence number twice is
  also suppressed as a no-op.
- `submit` appears only after all proof goals are closed. If a fully
  assembled proof is rejected, an observable `submit_failed` flag
  prevents resubmitting the identical proof until the proof changes or
  the episode resets.

These flags describe the **current proof state**, not a list of prior
actions. They are included in observations and in the random simulator's
cache keys. A failed verification still consumes one proof-game action
and returns its normal checker error, but repeating the same check
without editing is no longer a legal action.

No reward or proof is granted unless the unchanged kernel accepts the
certificate. This reduces useless loops; it does not guarantee that the
model will learn new proof methods. Because the available action menus
change, begin a **fresh run** for a clean before/after comparison; a
previously running Python process will not pick up this code automatically.

## Short, verified NONHALT certificates

The proof game now supports two additional **formally checked** ways to
prove NONHALT without constructing a long induction proof.

**Closed state set:** the agent proposes a set of non-halting states
containing `q0`. The checker verifies that for *both* read symbols, every
transition from every state in the set remains in the set. `qH` must not
appear. This is an inductive invariant, not a simulation heuristic.

Example (5 proof-game actions on a two-state machine whose `q0` rows
both return to `q0`):

```text
choose:NONHALT
rule:closedStates
cert:add:q0
cert:verify
submit
```

**Exact recurrence:** the agent proposes `start` and positive `period`.
The checker simulates only the proposed finite segment and verifies that
the **entire** machine configuration at `start` and `start+period`
matches: state, absolute head position, and all nonblank tape cells.
It also rejects a halt anywhere in the checked prefix. Determinism
then proves the machine runs forever. Merely not halting within a
finite number of steps is **never** sufficient.

Example (6 proof-game actions for the two-state parity trap):

```text
choose:NONHALT
rule:exactCycle
cert:start:2
cert:period:2
cert:verify
submit
```

The candidate editor currently exposes starts 0..32 and periods 1..32,
while the trusted checker supports proposals with up to 4,096 simulated
transitions. The editor range is deliberately small to keep action
menus manageable; this restricts which recurrence certificates the
policy can propose, not the logical soundness of accepted proofs.
Exact recurrence cannot prove non-halting machines with endlessly
growing tapes and no repeated complete configuration.

These rules are only available for an unproved NONHALT root. Failed
candidates produce an error and **zero reward**; the kernel independently
rechecks successful certificates, including after saving/loading.
Existing general natural-deduction proof rules remain available.

Run the constructive and adversarial tests:

```bash
python -m unittest discover -s tests -p 'test_short_nonhalt_certificates.py' -v
```

## Balanced HALT/NONHALT attempts until the first verified proof

With `--balanced-initial-answers --attempts 100`, each training machine
gets a reproducibly shuffled schedule containing **up to 50 HALT-first and
50 NONHALT-first attempts**. The machine stops **as soon as either branch
finds its first formally verified proof**. If neither succeeds, all 100
attempts run. Only attempts actually executed are counted in the logs.

```bash
python -m proofgame.epoch_train \
  --dataset runs/unfiltered_1000_100/dataset.json \
  --out runs/ast_cpu_balanced_until_proof \
  --epochs 100 --attempts 100 --batch-size 8 --workers 8 \
  --representation ast --device cpu --balanced-initial-answers
```

The optional `--stop-after-first-train-proof` is redundant here: balanced
mode already stops at the first success. The schedule is balanced over
the **maximum** attempts; early stopping means the number of executed
HALT/NONHALT attempts may not be exactly equal.

Scheduled first actions have probability 1 and do not contribute a
gradient. The model learns how to construct a proof *given* the initial
answer, but this does not directly teach it which answer to select at
evaluation. The regular held-out evaluations and the parity/Busy Beaver
diagnostics remain unforced. The per-batch and per-epoch
`scheduled_initial_answer_counts` report the **actual executed**
attempts, not the full planned schedule.

The schedule requires an even maximum number of attempts, and the
existing `--train-exploration-epsilon` setting is replaced by the
balanced schedule. Use a new output directory rather than switching
strategies inside an existing checkpoint.

## Balanced answer exploration and choice logging

The epoch trainer now uses **on-policy epsilon-mixture exploration only during
training** (default `--train-exploration-epsilon 0.2`). Whenever the model
must select an answer, the action distribution is

```text
behavior(a | state) = 0.8 * policy(a | state)
                      + 0.2 * Uniform({choose:HALT, choose:NONHALT})(a)
```

Thus both answer choices retain at least 10% probability on each
answer-selection step. This does not guarantee a successful NONHALT proof,
nor does it force the network to predict 50/50: it preserves exploration
while the network learns which machines need which proof. The reset action
retains 80% of its original probability. The **actual mixed probability**
is recorded in successful trajectories, and REINFORCE differentiates the
**log of the mixed behavior distribution**, not the unmodified policy.
This avoids the off-policy bias from forcing an answer and pretending the
policy chose it. No proof examples or true machine labels are supplied.

**Evaluation and special parity/Busy Beaver benchmarks always use the
unmodified policy** (epsilon 0), so they measure what the model learned
without forced exploration.

The per-batch `progress.json` and per-epoch training/evaluation JSON now
include:

- `first_action_counts`: actual first actions `choose:HALT`,
  `choose:NONHALT`, and `reset`, across **all attempts**, including
  failures (not just successful proofs).
- `mean_initial_policy_probability`: average raw policy probabilities
  for HALT and NONHALT at the beginning of an attempt.
- `mean_initial_behavior_probability`: average probabilities after
  training exploration is applied (identical to raw values in evaluation).
- `initial_choice_observations`: denominator for those averages.

These diagnostics help distinguish **policy collapse** (raw NONHALT
probability near zero) from **proof-construction difficulty** (NONHALT
still chosen, but never successfully proved). Results are logged after
each epoch and saved in the usual files.

Example fresh training command (output directory automatically receives
a new run ID if already present):

```bash
python -m proofgame.epoch_train \
  --dataset runs/unfiltered_1000_100/dataset.json \
  --out runs/ast_cpu_exploration \
  --epochs 100 --attempts 100 --batch-size 8 --workers 8 \
  --representation ast --device cpu \
  --stop-after-first-train-proof --train-exploration-epsilon 0.2
```

To resume a run created before this option existed, pass
`--train-exploration-epsilon 0` to preserve its original behavior.
Changing the sampling strategy mid-checkpoint is intentionally rejected.

## Separate two-state diagnostic evaluations after every epoch

The epoch runner automatically tests two fixed challenge machines at **baseline
(epoch 0)** and after **every subsequent evaluation epoch**, on both CPU and
MPS:

1. **Parity trap (2-state)** — the correct conclusion is NONHALT, requiring
   reasoning about an alternating two-phase invariant.
2. **2-state Busy Beaver** — the correct conclusion is HALT (6 Turing-machine
   transitions), with a 12-action direct halting proof.

These tasks **never generate training gradients**, are **not inserted into**
the training or test dataset, and **do not inflate the main machine-coverage
metric**. Each diagnostic uses the current policy checkpoint and the configured
`--attempts` and dataset action budget (100 attempts × 100 proof-game
actions in the current unfiltered experiment). The action-sampling seeds are
fixed across epochs, and diagnostic search stops as soon as a proof is found.
The labels are for reporting only, not supplied to the model.

After the regular evaluation summary, the terminal prints human-readable
results such as:

```text
Special benchmarks — after epoch 3
Parity trap (2-state, NONHALT): NOT PROVED (no verified NONHALT proof in 100 attempts, up to 100 proof-game actions each)
Busy Beaver (2-state, HALT in 6 steps): PROVED HALT (first verified proof on attempt 9/100; 12 proof-game actions)
These are separate checks and do not change the held-out test score.
```

The above is **illustrative**, not a measured model result. A "NOT PROVED"
result means the model did not sample an accepted proof within this budget;
it does **not** mean the machine is unprovable.

Each epoch creates `special_benchmarks_epoch_00.json`,
`special_benchmarks_epoch_00.txt`, and corresponding files for later epochs.
`special_benchmarks_history.txt` keeps one readable timeline.
The final `report.json` also includes `special_benchmarks` per epoch.

Changes to a checked-out Python file do not affect a process that is already
running. Pull the new version and start a fresh run, or resume from a saved
checkpoint for subsequent evaluations to include these diagnostics.

## Automatic fresh-run IDs and MPS throughput

Running the **same** `python -m proofgame.epoch_train ... --out runs/my_run`
command repeatedly now **creates a new experiment** instead of refusing to
overwrite an existing directory. The first run uses `runs/my_run`; subsequent
runs use `runs/my_run_run_0001`, `runs/my_run_run_0002`, and so on.
The actual output path is printed as an `event: run_directory` JSON line.
All older checkpoints and reports remain untouched. **Each fresh run still
starts with randomly initialized weights and epoch 0.** To continue an
existing run, pass `--resume` with the **exact** printed run directory; no
automatic suffix is used when resuming.

The AST encoder now caches parsed syntax trees and builds attention relations
using batched CPU tensor indexing rather than individual scalar writes. MPS
progress includes `mean_context_batch_size` and `inference_wall_seconds`
so you can see whether the GPU is receiving real batches. This is an
incremental preprocessing optimization, **not a guaranteed MPS speedup**.
With a tiny width-32, one-layer Transformer, dispatching many small GPU
operations, synchronizing probabilities back to CPU, and waiting on worker
queues can still be slower than direct CPU inference. Compare wall-clock
time and proof actions per second before committing to a long run. To
benchmark the CPU path, keep `--representation ast` and use `--device cpu`
with the same dataset and attempts in a separate fresh run.

## Numerical stability and failed-run recovery

REINFORCE gradients can be unstable, especially after a rare first success.
The epoch trainer now checks the gradient norm, parameters, optimizer state,
and a sample of post-update logits. A non-finite update is **rejected and
rolled back** (including AdamW state), with a
`training_update_rejected` log event and a `rejected_updates` counter.
This avoids silently saving a corrupted model. A non-finite distribution
from inference raises an explicit diagnostic rather than the opaque
`random.choices: Total of weights must be finite` exception.

The AST encoder also disables PyTorch's native inference-only MHA fastpath,
which can produce NaN logits from otherwise finite post-update weights when
using our learned per-head structural masks. This keeps rollout inference and
gradient replay on the same attention implementation.

**Existing checkpoints may still be usable** if their model and optimizer
tensors are finite: the new attention path can fix an inference-only NaN
without changing the saved weights. Resume with the exact same run settings
and `--resume` after updating the repository. If resume instead reports
non-finite model weights or optimizer state, keep the old run directory for
inspection and start a fresh run (the unique-run suffix feature preserves
earlier outputs). Don't assume a successful `policy_updates: 1` log means
that particular update was numerically valid on every future observation.

For the current tiny model, **CPU may outperform MPS**. CPU rollout workers
make policy calls locally, while MPS needs worker queues, JSON parsing,
per-action GPU dispatch, and CPU synchronization of action probabilities.
MPS is more promising for larger networks and genuinely large inference
batches. Benchmark throughput rather than assuming the GPU will be faster.

## Apple Silicon GPU training (MPS, experimental)

The epoch trainer now supports `--device mps` on an Apple Silicon Mac with an
MPS-capable version of PyTorch. Only the **main process** owns the Metal
device and the policy. CPU processes run the symbolic proof environments and
send observations to a central **batched inference server**, which caches
frozen-policy action embeddings and observation probabilities. GPU gradients
are accumulated using microbatched observations and deduplicated actions.
Tokens are length-bucketed on MPS to reduce repeated shape compilation.

First confirm `python -c "import torch; print(torch.backends.mps.is_available())"`
prints `True`. Keep the dataset from the earlier unfiltered experiment.

Start a **new run from fresh weights** (no `--resume`):

```bash
python -m proofgame.epoch_train \
  --dataset runs/unfiltered_1000_100/dataset.json \
  --out runs/unfiltered_100_epochs_ast_mps \
  --epochs 100 --attempts 100 --batch-size 8 --workers 8 \
  --stop-after-first-train-proof --representation ast --device mps \
  --gpu-inference-batch-size 8 --gpu-inference-wait-ms 2 \
  --gpu-gradient-batch-size 4
```

MPS is **opt-in**: omitting `--device mps` preserves the existing
multiprocessing CPU runner. The GPU settings are saved in `config.json` to
protect checkpoint compatibility. If memory use is excessive, first reduce
`--gpu-inference-batch-size` to 4 and `--gpu-gradient-batch-size` to 1,
or reduce `--workers` to 4. This architecture does not guarantee speedups
for such a small Transformer; benchmark completed proof-game actions per
second and wall-clock time against CPU on the same dataset and model.

Every training and evaluation epoch writes `epoch_NN_training.json` or
`epoch_NN_evaluation.json` and prints an `event: epoch_summary` line.
In addition to overall machine coverage, these include:

- `machines_proved_by_answer.HALT` and `.NONHALT`: **distinct machines**
  with at least one formally checked proof of that answer
- `successful_attempts_by_answer.HALT` and `.NONHALT`: total verified
  successful attempts, including repeated proofs of one evaluation machine
- `machines_unproved`: machines with **no proof found within the budget**
- `first_proof_attempt_mean_by_answer`: mean attempt index of the first
  success per solved machine, or `null` when no success exists

These are **proved claims only**, not omniscient counts of truly halting or
nonhalting machines. The filtered experiment contains mostly easy halting
machines; the unfiltered dataset can contain both kinds. The per-batch
`progress.json` also includes answer-specific counts. The final
`report.json` retains the complete epoch-by-epoch training and evaluation
summary. Evaluation remains at all 100 attempts per machine.

MPS functionality requires a real Apple GPU for hardware validation. Run
`python -m unittest discover -s tests -v` before long experiments; the
optional Metal test runs only when MPS is available.

## Optional AST-aware Transformer (experimental)

The epoch trainer can use `--representation ast` instead of its default
`--representation bytes`. This changes **only the observation/action encoder**,
not the proof checker, transition rules, reward, optimizer or dataset.

AST mode parses existing JSON into an ordered field/list tree, and complete
displayed goal/hypothesis expressions into operator/argument trees. Bound
variable occurrences retain links to their quantifier via existing de Bruijn
indices. Common formal-language symbols use atomic token IDs; unseen names
fall back to UTF-8 byte tokens without hash collisions. The network still
attends globally, but its attention logits include nine **learned** directed
relationship biases (e.g. parent/child, siblings, binder/occurrence). These
start at zero: no relationship is initially assumed to be useful. No
simulator-derived halting labels, proof traces or theorem hints enter the
encoder. It is a syntax-aware model, **not** a formally invariant graph model.

To compare fairly with the existing byte encoder, **keep the same dataset,
seed, training budget, and model dimensions** but use a separate new output
directory. For example, after generating the unfiltered dataset above:

```bash
python -m proofgame.epoch_train \
  --dataset runs/unfiltered_1000_100/dataset.json \
  --out runs/unfiltered_100_epochs_ast \
  --epochs 100 --attempts 100 --batch-size 8 --workers 8 \
  --stop-after-first-train-proof --representation ast
```

The original default command still uses byte tokens, so old saved checkpoints
continue to load. AST checkpoints record `representation: ast` in their model
config; do not mix checkpoints from different representations when resuming.
The architecture has a different parameter count and may have different
memory/CPU characteristics. Treat it as a controlled research hypothesis,
not an established improvement. Use `python -m unittest discover -s tests -v`
before starting long training runs.

## Unfiltered two-state experiment: 1,000 train, 100 test, 100 epochs

This harder experiment **does not preselect machines based on stochastic proof
search**. It samples 1,100 distinct labelled two-state Turing machines uniformly
without replacement from all 20,736 possible transition tables. The first
1,000 form the fixed training set and the remaining 100 form the fixed
held-out evaluation set. Neither split contains proof labels or certificates.
The seed fixes the complete dataset. No halting/non-halting behavior is checked
when selecting machines.

Create the dataset once (choose a path that does not already exist):

```bash
python -m proofgame.unfiltered_dataset \
  --train 1000 --test 100 --budget 100 --seed 20261026 \
  --out runs/unfiltered_1000_100/dataset.json
```

Train a **new** randomly initialized model, evaluate before training (epoch 0)
and after each of 100 epochs:

```bash
python -m proofgame.epoch_train \
  --dataset runs/unfiltered_1000_100/dataset.json \
  --out runs/unfiltered_100_epochs \
  --epochs 100 --attempts 100 --batch-size 8 --workers 8 \
  --stop-after-first-train-proof
```

Each machine gets up to **100 independent proof attempts**, with up to **100
proof-game actions per attempt**. Training attempts stop early when the first
verified proof is found on that machine. Evaluation uses all 100 attempts for
every held-out machine and never updates the model. Epochs re-use the same
training machines, with a new deterministic shuffle and training action seeds
each epoch. There is no `--resume` flag in the command, and the output path
must be new to start from epoch 0.

This is potentially **very CPU-intensive**: in the worst case, 100 epochs x
1,000 training machines x 100 attempts x 100 actions = 1 billion training
actions, plus baseline and per-epoch evaluations. Training may have no useful
gradient updates if no valid proof is discovered; zero successes do not imply
unprovability. Also, because the unfiltered sample may contain non-halting
machines, the earlier filtered benchmark's success rates cannot be compared
directly to this benchmark. Measure held-out machine coverage as well as
training policy updates.

To run tests:

```bash
python -m unittest discover -s tests -v
```

## Earlier filtered experiment: 1,000 training machines, 100 test machines, 10 epochs

Everything needed for this experiment is in this archive. The main runner is
`proofgame/epoch_train.py`. The fixed, disjoint dataset is
`runs/epoch_dataset_1000_100/dataset.json`. Every selected machine had a
kernel-checked proof found within 10,000 uniform-random attempts of at most
20 actions. Eligibility certificates are stored separately for auditing and
are never loaded by the epoch trainer. All selected eligibility proofs establish
halting; this is a filtered halting benchmark, not an unfiltered test of all machines.

Use Python 3.12 if available (the version tested here). In the extracted
`proof_game` directory, set up an environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -e ".[train]"
```

On Windows PowerShell, activation is optional; use the environment's Python
explicitly to avoid changing PowerShell execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[train]"
```

Start the full experiment (one line):

```bash
python -m proofgame.epoch_train --dataset runs/epoch_dataset_1000_100/dataset.json --out runs/my_10_epoch_run --epochs 10 --attempts 100 --batch-size 8 --workers 8
```

In PowerShell without activation, replace the leading `python` with
`.\.venv\Scripts\python.exe`. The CPU runner and resume path were tested on
Linux. The Windows locking path is implemented but has not been tested on a
Windows host.

The runner evaluates the fresh network once, then runs ten training epochs with
evaluation after every epoch. Each machine receives exactly 100 reset attempts,
including attempts after a success. Every attempt allows at most 20 actions.
Training machines are shuffled each epoch; test machines and their sampling
seeds stay fixed. Evaluation never updates parameters. Each rollout batch uses
frozen weights, then a zero-baseline REINFORCE update uses only the network's
own verified successes; zero-reward attempts contribute zero gradients.

Resume after an interruption with the same command plus `--resume`:

```bash
python -m proofgame.epoch_train --dataset runs/epoch_dataset_1000_100/dataset.json --out runs/my_10_epoch_run --epochs 10 --attempts 100 --batch-size 8 --workers 8 --resume
```

Checkpoints commit after every batch. An interrupted, uncommitted batch is
recomputed from its saved weights and seeds. Only one process may own a run
directory. To continue past ten epochs, resume with a larger `--epochs` value.
Keep the dataset, seed, attempt budget, and batch size unchanged when resuming;
the number of workers may change. `--workers 4` reduces concurrent CPU/RAM use.
More than eight workers offers no benefit with a batch size of eight; changing
the batch size also changes the optimization experiment.

Outputs inside the chosen run directory:

- `progress.json`: current epoch, phase, and completed machines.
- `epoch_00_evaluation.json`: the untrained baseline on the filtered test set.
- `epoch_01_training.json` and `epoch_01_evaluation.json`, etc.: per-epoch metrics.
- `model_epoch_01.pt`, etc.: evaluated checkpoints.
- `state.pt`: the resumable model, optimizer, and progress state.
- `epoch_01_train/batch_*.json`, etc.: sampled attempts, counts, and successful proof traces.
- `report.json`: `complete: false` while active; complete learning curve at the end.

The main metric is **machine coverage**: the fraction of test machines with at
least one verified proof in their 100 attempts. The runner also reports proof
success per attempt. Do not compare this filtered 100-attempt coverage directly
to the earlier unfiltered single-attempt scores.

### CPU and GPU support

The current epoch runner is **CPU-only**. Each worker uses one PyTorch CPU thread.
A powerful multicore CPU can run several rollout workers. Merely installing a
CUDA build of PyTorch will not make this script use the GPU: the model and tensors
are not moved to CUDA, and there is no `--device cuda` option.

A useful GPU extension would batch neural inference across many simultaneous
games while leaving the Python proof engine on CPU. A GPU speedup has not been
implemented or measured here; this tiny model and per-action control flow make
it unsafe to promise one without benchmarking. Use PyTorch's official install
selector for a CUDA-compatible build if developing that extension:
https://pytorch.org/get-started/locally/

### Tests

```bash
python -m unittest discover -s tests -v
```

The epoch runner has a test for zero-reward updates, grouped-gradient equivalence,
and fixed evaluation seeds. A small end-to-end run and continuation to a second
epoch were also exercised before launching the full experiment.

## Try it

Python 3.10 or newer is sufficient for the checker, terminal interface, and core
tests. From this directory:

```bash
python -m proofgame demo one_step
python -m proofgame demo forever
python -m proofgame demo blank_sweeper
python -m proofgame play --fixture one_step
python -m proofgame check examples/forever.json
python -m unittest discover -s tests -v
```

The terminal interface prints the current goal, variables, hypotheses, and
numbered actions. Enter an action number or its exact ID. `quit` exits.

To prove the one-step example yourself, choose these actions:

```text
choose:HALT
rule:exI
put:s
put:z
commit
rule:refl
submit
```

You have selected the claim “the machine halts,” supplied the time witness
`s(z)` (one), and checked that its state at that time is the halt state.

## Original single-attempt trainer

The following commands describe the earlier `proofgame.train` baseline, rather than the filtered epoch experiment above.

Use a virtual environment and install the optional training dependency:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'
python -m proofgame.train --episodes 1000 --states 2 3 --out runs/my_first_run
```

On Windows, activate with `.venv\Scripts\activate` instead. A CPU build of
PyTorch is enough for a small smoke test; installation options depend on your
machine. The tested environment used Python 3.12 and PyTorch 2.14.1+cpu.

A shorter plumbing check:

```bash
python -m proofgame.train --episodes 5 --eval-episodes 3 --budget 24 --width 32 --layers 1 --out runs/my_smoke
```

The bundle now also includes a completed 200-episode run in `runs/longer_200`.
See `EXAMPLES.md` for written machines and short proofs, and `EXPERIMENT_RESULTS.md`
for the training results, random-search measurements, and their limitations.

Each run must use a fresh output directory. The command writes its configuration,
episode metrics, checkpoint, evaluation report, and any accepted proofs it
actually discovers. It does not resume or import existing model weights.

```bash
python -m proofgame.evaluate runs/my_first_run/model.pt --episodes 100
python -m proofgame.evaluate runs/my_first_run/model.pt --episodes 100 --sample-actions
python -m proofgame sample --states 3 --seed 42 > machine.json
python -m proofgame play --machine machine.json
python -m proofgame.evaluate runs/my_first_run/model.pt --machine machine.json
```

Sampling produces unlabelled machines. No simulator preclassifies the training
tasks. A deterministic hash partitions exact machine descriptions into train
and test pools. This prevents exact overlap; it does not exclude isomorphic
machines. All three evaluation policies receive the same test machines.

## What is fixed, and what is learned?

Fixed: the logical rules, types, machine semantics, expression editor, action
budget, and proof checker. Learned: which answer to attempt, which rule to use,
which expressions and lemmas to construct, which hypothesis to select, and
whether to abandon the current proof.

By default, the training reward is exactly 1 for a submitted proof
accepted by the checker, and 0 otherwise. The epoch trainer optionally
scales verified terminal rewards by trajectory length with
`--length-penalty`; failures still receive zero. There is no intermediate
reward for filling holes, writing lemmas, or constructing long expressions. There are no supplied demonstrations,
pretrained embeddings, theorem archives, hindsight relabeling, or curriculum in the original trainer. The epoch runner above adds machine selection, without supplying proof demonstrations.
An optional entropy regularizer is available but defaults to zero.

Learning uses on-policy REINFORCE with a learned value baseline. An episode's
actions are replayed to recompute gradients before its single parameter update;
this keeps attention graphs from accumulating across the whole episode. This
replay is of the agent's own trajectory, not supervised demonstration training.

The strict terminal-only reward makes discovery difficult. Even lengthy runs
may remain at zero success. The prototype lets us measure that instead of
assuming it will work.

## The model

The default policy is a randomly initialized, two-layer Transformer with width
64, four attention heads, and a value head. It reads UTF-8 bytes, so variable
names, machine-state names, and action IDs need no pretrained tokenizer or
fixed inventory of identifiers. The byte vocabulary has 256 byte tokens plus
padding and a pooling token.

The full current proof state is encoded once. Candidate actions are encoded
separately with the same Transformer and scored against the state embedding.
A categorical distribution selects one candidate. Thus the number of actions
can change as variables, hypotheses, and expression slots change.

Position encodings are sinusoidal. Inputs exceeding the configured token
budget end that rollout as `MODEL_LIMIT`; they are never silently truncated.
An unbounded symbolic language does not imply an unbounded neural context.

## Variables and numbers

The sorts are `Nat`, `Bit`, `List`, `State`, and `Cfg`. Free variables have stable
IDs such as `v0` and `v17`, with their sorts in the local context. IDs are local
references, not natural-number values. Introducing ten variables creates ten
references; no ten-variable feature or vocabulary change is necessary.

Quantified formulas internally use bound references. `bvar0` refers to the
nearest enclosing quantifier, `bvar1` to the next, and so on. When `allI`
introduces an arbitrary variable, the checker replaces the corresponding bound
reference with a fresh stable free ID. The model does not implement renaming or
substitution. Existential elimination introduces a witness only within its
child proof; that witness cannot escape to the parent or a sibling branch.

Natural numbers use `z` and `s`, with recursively defined `add` and `mul`.
There are no thousand initial numerals and no fixed vocabulary of integers.
Binary numeral notation is not implemented in this first version.

## Machine semantics and the two goals

A machine has complete binary transition rows, starting in `q0` on a blank
tape. `qH` is the halt state. Rows are ordered `q0/b0`, `q0/b1`, `q1/b0`, etc.:

```json
{"rows": [["qH", "b0", "R"], ["qH", "b1", "R"]]}
```

A configuration is `cfg(q, L, R)`. The left list is nearest-cell first; the
right list includes the scanned cell. Missing cells contain `b0`. Writing,
moving, and entering a new state count as one transition. Literal tape lists
are not silently trimmed or identified with each other.

This version uses a total function `next(C)`, which leaves a halted configuration
unchanged, and a finite-time function:

```text
at(z, C)     = C
at(s(n), C)  = next(at(n, C))
```

With `C0 = cfg(q0, nil, nil)`, the goals are:

```text
HALT:     exists n:Nat. qof(at(n, C0)) = qH
NONHALT:  forall n:Nat. (qof(at(n, C0)) = qH) -> false
```

Freezing after halting does not change these two properties. This is an explicit
simplification of the earlier relational `run(n,C,D)` design. It is not an
implementation of that earlier document verbatim.

The checker computes fixed defining equations, including finite runs when a
concrete time is supplied. This computation is part of proof checking. No
general programming language is offered to the model. Computation can still
be expensive: reductions have a fixed fuel limit, and exceeding it rejects
that attempted step. It never establishes non-halting.

For symbolic non-halting arguments, the equation
`qof(next(C)) = qnext(qof(C), head(right(C)))` exposes the next control state.
`qnext` is defined directly by the transition table, including `qnext(qH,b)=qH`.
It does not infer an invariant or tell the model whether the machine halts.

## Proof rules

The foundation is standard classical natural deduction with sorted equality,
natural-number and list induction, and the machine definitions above. This
prototype implements its own small checker; it does not embed Peano, Lean, or
an externally verified kernel.

| Rule | Effect |
|---|---|
| `allI`, `exI` | Introduce an arbitrary variable, or choose an existential witness. |
| `impI`, `apply` | Assume an implication's premise, or use a known implication. |
| `andI`, `orL`, `orR` | Prove a conjunction or a selected disjunct. |
| `destruct` | Unpack a conjunction or existential; split an assumed disjunction. |
| `specialize` | Instantiate a universally quantified hypothesis. |
| `cut` | Propose a lemma, prove it, then use it in a separate continuation. |
| `natI`, `listI` | Generate base and induction-step obligations. |
| `casesBit`, `casesState` | Split on all constructors of a finite type. |
| `rewrite` | Substitute using an equality hypothesis, forward or backward. |
| `inject`, `discriminate` | Use constructor injectivity or a constructor contradiction. |
| `refl`, `exact`, `falseE` | Close by definitional equality, an existing fact, or false. |
| `reduce` | Normalize the goal and hypotheses using fixed definitions. |
| `raa` | Prove the goal classically by contradiction. |

`refl` and `exact` allow definitional conversion; other rule shapes are generally
syntactic, so `reduce` may be needed first. `cut` never puts a lemma among its
own assumptions. Every generated child is checked for type and scope.

The supplied `forever` fixture demonstrates a genuine induction proof, not
infinite simulation or a cycle-detection shortcut. Its machine always stays in
`q0`, while writing and moving right. The proof proposes the invariant
`forall n. qof(at(n,C0)) = q0`, proves it by induction and a split on the scanned
bit, then derives that the machine never reaches `qH`. The fixture's invariant
is hand-written for validation and is not supplied to the learning agent.

## How one action works

The environment offers a finite list appropriate to the current state:

1. Select `HALT` or `NONHALT`.
2. Select a proof rule for the focused goal.
3. If the rule needs an expression, fill its typed holes one constructor at a
   time. For example, `put:s` creates a new `Nat` hole and `put:z` fills one.
4. `commit` checks that annotation and installs the rule's subgoals.
5. Repeat, then `submit` when the complete proof is ready.

Rules needing no annotation execute in one action. A rule can appear in the
menu and still fail semantically: offering `refl` on an equality does not tell
the agent that its sides are equal. Typed expression slots exclude ill-sorted
choices. `copy:N` copies an existing, well-scoped expression; the observation
shows exactly what will be copied.

`focus:N` selects an open goal. `cancel` abandons the expression draft.
`reset` clears the proof and its local lemmas, preserving the same machine.
Every attempted action consumes budget, including invalid actions and reset.
A failed rule restores the previous board. A successful submission on the last
available action still wins. There is no undo history in v0.1.

The observation includes all open goals and their contexts, the proof tree,
focus, expression draft, resource limits, last error, and actions remaining.
The model has no recurrent memory beyond its current input. Reset therefore
also removes old proof clutter from its observation.

## Certificates, limits, and what we have verified

Certificates are JSON trees carrying the machine, claimed answer, rule names,
annotations, and child proofs. `check` reconstructs every context from the
root. It never trusts a saved `SOLVED` flag or a model's assertion. The editor
and checker share rule-expansion code; this avoids divergent semantics but
means tests are not an independent formal verification of that code.

Finite menus do not mean a finite mathematical language. Runtime caps keep
each experiment finite: action budget, workspace size/depth, checker visits,
definition-reduction fuel, Transformer context, and currently at most 32 machine
states. These are implementation limits, not claims of mathematical
impossibility. Failure or exhaustion always means **unproved**.

Tests cover accepted halting and inductive non-halting certificates, wrong time
witnesses, a false non-halting claim on a changed machine, malformed proof
trees, scope isolation, induction with outer parameters, ten live variables,
constructor injectivity, random legal-action walks, resource behavior, and
agreement with a separately written concrete one-step simulator. Optional
PyTorch tests cover dynamic action scoring, backpropagation, checkpoint reload,
input limits, and equal evaluation task sets.

`runs/verified_smoke/report.json` records a five-episode execution test. It found
zero proofs. The hand-written examples and passing tests establish that the
mechanism works on those cases; they do not establish learned proof discovery,
coverage of all two- or three-state machines, or logical completeness.

The longer run likewise found no proofs during its 200 episodes. A separate
uniform-random search found one accepted proof in 1,000 attempts on the fixed
one-step machine. That random discovery was not fed back into training.

You can reproduce the random baseline and the post-training audit with:

```bash
python -m proofgame.random_baseline --task one_step --episodes 1000 --out runs/my_random.json
python -m proofgame.audit_run runs/longer_200
```

`audit_run` uses hand-written examples to measure known proof lengths only after
training. It also verifies which sampled training tasks have seven-action
proofs; these labels and proofs never enter the learning loop.

## Where to modify things

| File | Responsibility |
|---|---|
| `proofgame/logic.py` | Syntax, types, binding, machine semantics, reduction. |
| `proofgame/kernel.py` | Logical rules and complete certificate checking. |
| `proofgame/environment.py` | Action menus, expression builder, resets, limits. |
| `proofgame/model.py` | Byte Transformer and variable-size action scoring. |
| `proofgame/train.py` | Terminal-reward training and baseline evaluation. |
| `proofgame/tasks.py` | Machine sampling and partitioning. |
| `proofgame/demo.py`, `examples/` | Explicitly separate verification fixtures. |

Useful background: [MINIMO](https://arxiv.org/abs/2407.00695),
[Peano](https://arxiv.org/abs/2211.15864), and the PyTorch documentation for
[TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
and [policy-gradient distributions](https://docs.pytorch.org/docs/stable/distributions.html).
This prototype does not reproduce MINIMO's training algorithm or its results.

