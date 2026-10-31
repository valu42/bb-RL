# Running the foundational pipeline

The implemented entry point is `python -m proofgame.foundation`. It provides
`generate`, `pretrain`, `train`, `evaluate`, `verify`, and an end-to-end `run`.
V1 commands and checkpoints remain unchanged; v1 policy checkpoints cannot be
used to initialize or resume v2. Install training dependencies with
`python -m pip install -e '.[train]'`.

## Full workflow

From the repository root:

```bash
python -u -m proofgame.foundation run \
  --out runs/foundation_2_to_5 \
  --per-size 10000 --train 1000 --dev 400 --test 400 \
  --budget 512 --simulation 0 \
  --updates 15625 --pretrain-batch-size 64 \
  --eval-every 2000 --eval-machines 40 --pretrain-eval-attempts 8 \
  --width 64 --layers 2 --max-tokens 4096 \
  --epochs 100 --batch-size 8 --workers 8 \
  --attempts 8 --rounds 6 --eval-attempts 100 \
  --rl-eval-machines 100 --rl-eval-attempts 8 --full-eval-every 10 \
  --device cpu
```

This command samples up to 10,000 distinct training-partition machines of each
size, not 10,000 guaranteed proofs. It compiles all supported teacher successes
that fit the chosen action budget. Pretraining performs 15,625 updates of 64
sampled action targets (one million exposures, not unique proofs), followed by
100 maximum RL epochs. The 512-action budget is a starting experiment; corpus
acceptance and model-limit statistics reveal whether it needs changing. Choose
a new output for a different budget. No large training-quality result or wall
clock estimate is implied by these defaults.

`--simulation 0` uses seeded teacher allowances of 128/1,024/10,000/100,000
transitions with probabilities 70/20/9/1 percent. A positive value supplies one
fixed teacher allowance. These allowances affect offline generation only.
The online model has explicit equations and induction, not a simulate command.

Run the **same command with `--resume` appended** after interruption. Generation
recovers its committed byte offset, SFT restores optimizer and RNG state, and RL
restores its phase/batch and optimizer. Resume permits changing worker count and
extending `--updates` or `--epochs`, plus the quick-evaluation settings described
below; learning configuration and source hashes must otherwise match. Once RL has started, changing its pretrained checkpoint
requires a new RL output, even if the new weights came from extending SFT.

### Evaluation schedule

| Check | Machines | Attempts per machine | When |
|---|---:|---:|---|
| Pretraining dev | Fixed balanced subset, up to 40 | 8 | Every 2,000 updates and at the end |
| RL quick dev | Fixed balanced subset, up to 100 | 8 | Baseline and ordinary epochs |
| RL full dev | All dev machines (400 in the full command) | 100 | Every tenth completed epoch |
| Special machines | All 13 | 100 | After every completed RL epoch |
| Held-out test | All test machines (400 in the full command) | 100 | Once at the end |

A full dev check replaces that epoch's quick check. The baseline at epoch 0 is
quick and has no special-machine pass. Special machines run after every trained
epoch, independently of the quick/full dev schedule. A run ending before epoch
10 still gets its per-epoch special checks and full held-out test. Set
`--full-eval-every 1` to also run full dev checks after every trained epoch.
All searches stop at the first verified proof for each machine and retain the
corpus's action limit (512 above). RL training still uses `--attempts` and
`--rounds`; these evaluation settings never change its exploration allowance.

The controls are `--eval-every`, `--eval-machines`, and
`--pretrain-eval-attempts` for pretraining, and `--rl-eval-machines`,
`--rl-eval-attempts`, and `--full-eval-every` for RL. These six settings can change
on resume. `--eval-attempts` controls **full, special, final-test, and standalone**
evaluation only; it remains fixed when resuming training. Model architecture,
learning settings, proof budgets, and datasets also retain strict resume checks.

Every new result carries an `evaluation` object with kind, panel hash, machine
count, attempts per machine, and action budget. Compare scores with the same
profile: quick and full coverage are different measurements. `best.pt` is
selected by the pretraining dev profile only, with ties retaining the earlier
checkpoint. Full RL results do not compete against quick scores.

For an existing run, interrupt with Ctrl-C, update the code, and use the full
command above with `--resume`. The generated corpus, trained weights, optimizer,
and committed RL training batches are preserved. A pending evaluation restarts
under the new allowance. If the pretraining panel/attempt budget changes, the
old `best.pt` is copied to `best_before_eval_change_*.pt`; selection restarts from
a fresh check of the current weights, without comparing against the old score.
Even if pretraining was already finished, this check runs before handing off to
RL. If RL has already started, `run --resume` skips finished pretraining and
preserves the initial checkpoint already consumed by RL.

CPU pretraining evaluation and the standalone `evaluate` command use
`--workers` processes (up to the number of machines), with deterministic
per-machine seeds and the same first-proof stopping rule as serial evaluation.
MPS evaluation stays on the requested device in one process; its progress log
reports the effective worker count. The action and attempt allowances are not
reduced by parallel execution.

Evaluation prints `evaluation_start` immediately, `evaluation_progress` about
every five seconds, and `evaluation_complete` at the end. Progress includes
completed machines, partial-attempt actions, completed attempts, actions/sec,
HALT/NONHALT counts, elapsed time, attempt exit statuses, and accumulated worker
inference/game/observation times. ETA is initially `null`, then estimated from
completed machines; it can change substantially as harder machines finish.
CPU progress continues even while workers are starting or a model call is slow.
Serial/MPS progress is emitted between model actions, so a slow single action
can delay a log. Pending machines are not counted as unproved.

For example, the full command's periodic pretraining check uses 40 dev machines,
up to 8 attempts each, and up to 512 actions per attempt: at most 163,840
actions. Ordinary RL dev checks allow at most 409,600 actions. Eight workers
distribute that search; they do not guarantee an eightfold speedup. Full checks
are deliberately more expensive and less frequent.

To apply an evaluation fix to an existing process, interrupt it with Ctrl-C,
update the repository, and restart the same command with `--resume`. New code
cannot change an already-running Python process. Older checkpoints resume from
their last saved update (typically 990 when waiting at the first evaluation).
New checkpoints save the completed supervised update **before** evaluation;
interruption then repeats that evaluation without repeating the optimizer
update. In-flight evaluation search itself is not checkpointed.

## Small installation check

This validates operation, not learning quality:

```bash
python -u -m proofgame.foundation run \
  --out runs/foundation_smoke \
  --per-size 3 --train 4 --dev 4 --test 4 \
  --budget 128 --simulation 16 --compiler-steps 2000 \
  --updates 2 --pretrain-batch-size 2 --eval-every 2 \
  --width 16 --layers 1 --eval-machines 4 \
  --epochs 1 --batch-size 2 --workers 2 \
  --attempts 2 --rounds 1 --eval-attempts 1 \
  --pretrain-eval-attempts 1 --rl-eval-attempts 1 \
  --skip-special
```

Special eval is enabled after every completed RL epoch by default; the smoke
command disables it explicitly. Final held-out test evaluation still runs.
`--skip-final-test` explicitly disables that end-of-run test when desired.

## Run stages separately

```bash
python -m proofgame.foundation generate \
  --out runs/foundation_data --per-size 1000 \
  --train 1000 --dev 400 --test 400 --budget 512 --workers 8

python -u -m proofgame.foundation pretrain \
  --data runs/foundation_data --out runs/foundation_pretrain \
  --updates 2000 --batch-size 64 --width 64 --layers 2

python -u -m proofgame.foundation train \
  --data runs/foundation_data --out runs/foundation_rl \
  --init runs/foundation_pretrain/best.pt \
  --epochs 100 --workers 8 --batch-size 8 \
  --attempts 8 --rounds 6 --eval-attempts 100
```

Omit `--init` on a fresh `train` run for the random-initialization control.
The RL action budget comes from the corpus profile, not an independently
changeable training flag. A new RL optimizer is created on initialization;
resuming restores that RL optimizer. SFT's optimizer is not transferred.

Pretraining's first 1,000 updates sample the shortest proof-length quartile,
the next 1,000 sample through the third quartile, and subsequent updates use
all admissible proofs. Set `--curriculum-updates 0` for the full corpus from
the beginning. Sampling balances available machine sizes, answer classes,
and NONHALT sources, then machines, proof variants, and replay positions.
Empty strata cannot be balanced and are visible in generation coverage logs.

## Results, proofs, and profiling

A `run` output contains `data/`, `pretrain/`, and `rl/`. Separate-stage commands
write those stages directly to their respective `--out` paths.

- `data/machines.json`: fixed family-disjoint train/dev/test/special sets and
  teacher job list. Related renamings, reflections, and unreachable-state
  embeddings share a split. This is structural, not complete semantic deduplication.
- `data/corpus.jsonl`: accepted certificates and legal action demonstrations,
  plus unsuccessful generation/compilation reasons. The index loads offsets,
  not all proof bodies. `manifest.json` hashes the completed corpus.
- `data/progress.json`: proposed/processed jobs, accepted proofs/actions, coverage
  by size/answer/source, errors, and the committed corpus byte offset.
- `pretrain/metrics.jsonl`: sampled exposures, supervised loss/accuracy, context
  exclusions, update rejection, timing, autonomous dev results, and live
  evaluation events tagged with their update number. `seconds` measures the
  supervised update; `evaluation_seconds` measures evaluation separately, and
  `total_seconds` is their sum. `pretrain/progress.json` holds the latest event.
- `pretrain/checkpoint.pt`: latest saved training state; `best.pt`: first best
  autonomous dev coverage checkpoint. Ties currently retain the earlier model.
- `rl/epoch_XX_training.json` and `epoch_XX_evaluation.json`: cumulative HALT,
  NONHALT, unproved, first-proof action means, and state-size breakdowns. Dev
  summaries include the evaluation profile and also have a
  `epoch_XX_evaluation_KIND_PROFILE.json` copy for separate quick/full histories.
- `rl/metrics.jsonl`: batch rollout/learning/checkpoint/snapshot times, summed
  worker inference/observation/game times, actions/sec, and informative groups.
- `rl/batches/`: committed machine results and successful proof action traces.
  Evaluation batches are isolated by profile so resuming after a setting change
  cannot combine results from different search budgets.
- `rl/special_benchmarks_epoch_XX.json`: the original extended suite plus BB(5).
  Successful special proofs are also printed. BB(5) is an aspirational challenge.
- `rl/final_test_epoch_XX.json`: separate held-out test results after training.

Terminal RL progress shows cumulative and most-recent-batch counts and proof
lengths, an ETA, and per-size phase summaries. A proof action is an editor or
inference action, not a Turing-machine transition.

Explicit evaluation and certificate verification:

```bash
python -m proofgame.foundation evaluate \
  --data runs/foundation_data \
  --checkpoint runs/foundation_rl/model_epoch_01.pt \
  --split test --eval-attempts 100 --out runs/foundation_test_01.json

python -m proofgame.foundation verify --certificate certificate.json
```

Evaluation refuses to overwrite its output. Extract the `certificate` object
from a stored success to use `verify`.

## Implementation boundaries

The student has logical rules, typed expression construction, natural/list
induction, explicit occurrence rewriting/unfolding, equality reasoning, case
splits, scoped intermediate lemmas, and checked definitions. Function definitions
may recurse structurally on one Nat/List argument; predicate definitions must
be nonrecursive. Definitions have at most four parameters. Completed proof
steps live in a flat archive outside the model observation. Current goals,
hypotheses, definitions, and draft context remain visible; overlong observations
produce `MODEL_LIMIT`, never silent truncation.

The three offline compiler families are HALT simulation, closed state sets, and
exact cycles. Arbitrary v1 certificates are not accepted as opaque imports.
Every emitted equation/inference is checked and replayed through the v2 game.
The online game/kernel/trainer do not import the offline compiler or v1 helpers.

Compared with the original design, canonical tape updates use explicit `pack`
equations that eliminate trailing zero cells when a stack is updated. This makes
exact cycles compile to literal configuration invariants without an implicit
extensional-equivalence oracle. `pack` is ordinary visible computation; `refl`
does not execute it. The fixed `qof(next(c))` equation exposes the one-transition
state projection. Neither equation summarizes a cycle or an invariant.

Generation has bounded in-flight worker jobs and a cycle-history cell cap; after
history exhaustion, cycle detection stops while bounded simulation can continue.
Large traces and compiler/editor limit failures are recorded and excluded.
The initial release uses a fixed user-selected action budget rather than an
automatic 90th-percentile calibration; inspect the pilot coverage before scaling.
The corpus stores compact flat certificates and action streams, replaying sampled
prefixes on demand rather than persisting large observation snapshots.

Parallel RL inference runs on CPU workers. `--device mps` moves supervised
training and RL gradient computation to MPS, making RL hybrid CPU/MPS; it does
not enable the v1 central full-MPS inference server. CPU and MPS throughput must
be measured on the user's machine. There is no proven learning or BB(5) success
claim from the engineering smoke tests.

Language/editor source hashes bind corpora and checkpoints to their semantics.
If those source files change, regeneration is required instead of silently
mixing action meanings. Trusted local PyTorch checkpoints use normal Python
serialization and should not be loaded from untrusted sources.
