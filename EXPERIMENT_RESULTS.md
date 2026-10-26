# Experiment results

## Latest pilot: filter for stochastic-search-solvable machines

The fast uniform proof search tested nine randomly ordered training machines and accepted four whose proof it found within 10,000 attempts of at most 20 actions. The neural policy independently proved two of those four machines and made two updates from its own verified trajectories. It received no selector proof steps, certificates, or answers.

| Selected machine | Fast filter: first proof attempt | Network: first proof attempt | Network proof actions |
| --- | ---: | ---: | ---: |
| 1 (rank 20252) | 1,670 | Not found in 10,000 | — |
| 2 (rank 19125) | 1,372 | 1,743 | 8 |
| 3 (rank 1231) | 3,930 | 9,564 | 15 |
| 4 (rank 4651) | 5,731 | Not found in 10,000 | — |

There were 31,307 neural attempts and 626,123 actions in total. Both discovered training proofs establish halting. Each machine was abandoned at the first neural proof or after 10,000 attempts, as specified before the pilot; this produced two optimizer updates, not 10,000 updates per machine.

The policy was freshly initialized with seed 7, using the same 23,138-parameter Transformer. In addition to selecting easier tasks and allowing repeated attempts, this pilot changed the update rule to success-triggered REINFORCE with baseline zero: failed attempts leave both policy and optimizer unchanged; a self-generated verified proof receives reward 1 and one policy-gradient update. AdamW uses learning rate 0.0003 and zero weight decay, with gradient-norm clipping at 1. There is no entropy bonus or partial-proof reward. The finite observation cache is discarded after updates. This is an exploratory combined change, not an isolated test of the filter alone.

The update raised the probability of the first successful training action sequence by a factor of 1.148, and the second by 1.172, measured immediately before and after their respective updates. These are training-trajectory diagnostics, not generalization scores.

### Unfiltered held-out evaluation

The initial and final policies were tested on the same 1,000 held-out episodes (901 distinct machines), sampling actions with one attempt of at most 20 actions per episode. The test set was not filtered for ease and had no exact overlap with the training machines.

| Policy | Verified proofs | Single-attempt success rate |
| --- | ---: | ---: |
| Initial random policy | 0 / 1,000 | 0% |
| After the two updates | 1 / 1,000 | 0.1% |

The final test result reproduced when rerun to capture the proof. The success occurred at zero-based test episode 742 and used 13 actions. Its certificate was independently checked and its complete action sequence replayed. One held-out success is too little to establish a reliable improvement or broad generalization.

The initial evaluation was reused only after verifying that the independently created initial parameters were identical to the earlier baseline. The final parameters also match the checkpoint after the last successful update, confirming that the remaining unsuccessful attempts did not change them.

The selector audit is in `runs/easy_filter_4/`; its `selection.json` intentionally contains no proof or answer fields. Training checkpoints, trajectories, gradient diagnostics, and final evaluation certificates are in `runs/curriculum_4/`. `training_audit.json` and `evaluation_summary.json` record the checks. A rejected selector candidate merely failed this seeded search within its cap; it was not proved intrinsically difficult or unprovable.

Reproduce:

```bash
python -m proofgame.easy_filter --count 4 --attempts 10000 --budget 20 --seed 20261028 --out runs/fresh_filter
python -m proofgame.curriculum_train --selection runs/fresh_filter/selection.json --attempts 10000 --seed 7 --threads 1 --out runs/fresh_curriculum
python -m proofgame.evaluate runs/fresh_curriculum/initial_model.pt --episodes 1000 --states 2 --budget 20 --seed 100007 --sample-actions --record-proofs
python -m proofgame.evaluate runs/fresh_curriculum/model.pt --episodes 1000 --states 2 --budget 20 --seed 100007 --sample-actions --record-proofs
```


## Earlier run: one attempt per random machine, 20 actions

A fresh 23,138-parameter Transformer trained for **5,000 episodes**, with one independently sampled two-state machine per episode, one attempt, and a maximum of 20 proof-game actions. Sampling was with replacement; 4,317 distinct training machines appeared. No failed machine was deliberately retried. Training produced **0 verified proofs in 100,000 actions**.

We evaluated the saved initial and final checkpoints by sampling their action probabilities on the same 1,000 held-out test episodes (901 distinct machines), with one attempt of at most 20 actions per episode. The training and test machine sets were disjoint.

| Checkpoint | Verified proofs | Single-attempt success rate |
| --- | ---: | ---: |
| Initial random network | 0 / 1,000 | 0% |
| After 5,000 training episodes | 0 / 1,000 | 0% |

All test episodes exhausted the action budget. There were no model-context-limit failures. The parameters changed, but no improvement in proof discovery was measured. No successful proof reward was encountered during training.

Training used the existing terminal-reward REINFORCE trainer with a learned value baseline: reward 1 for a kernel-checked submitted proof and 0 otherwise; learning rate 0.0003; entropy coefficient 0. Model initialization seed 7, width 32, one Transformer layer. No earlier proofs, demonstrations, or theorem archive were supplied. A separate frozen-network pilot was interrupted when the user requested this training run; its discovered proofs were not training inputs.

The audit reconstructed every sampled training machine, checked all episode budgets and counts, verified fresh initialization and finite final parameters, and confirmed zero train/test overlap. See `runs/train_random_once_5000/training_audit.json` and `evaluation_summary.json`. The run directory contains both checkpoints and all 5,000 episode records. This single-attempt metric has a different budget from the earlier Monte Carlo experiment with up to 10,000 attempts per machine.

Reproduce training and the two stochastic evaluations:

```bash
python -m proofgame.train --episodes 5000 --budget 20 --states 2 --seed 7 --width 32 --layers 1 --eval-episodes 0 --threads 1 --out runs/fresh_training
python -m proofgame.evaluate runs/fresh_training/initial_model.pt --episodes 1000 --states 2 --budget 20 --seed 100007 --sample-actions
python -m proofgame.evaluate runs/fresh_training/model.pt --episodes 1000 --states 2 --budget 20 --seed 100007 --sample-actions
```


These are measured results for this prototype, not claims about all small Turing machines.

## Training from random initialization

Trained a fresh 23,138-parameter Transformer for 200 episodes, using 12,800 actions in total. Each attempt had 64 actions. The stream contained 115 two-state and 85 three-state machines. Reward was terminal-only; no proof examples or archive were supplied.

Accepted training proofs: **0**. Greedy evaluation found 0/30 before training and 0/30 afterward. Sampled-policy evaluation likewise found 0/30 before and 0/30 afterward. Uniform random actions found 0/30 on the same evaluation machines.

A separate audit verified that **55 of the 200 training tasks admit a seven-action proof**. Those audit proofs were not supplied to training. Parameter updates alone are not evidence that the model learned proof discovery.

## Random actions: 64-action attempts

| Task distribution | Attempts | Accepted proofs |
|---|---:|---:|
| Fixed one-step-halting machine | 1,000 | 1 |
| Random held-out two-/three-state machines | 1,000 | 0 |

The observed 0.1% success rate on the fixed easiest machine is a noisy estimate. Zero successes on the mixed distribution do not mean its true success probability is zero. The one-sided 95% binomial upper bound from 0/1,000 is about 0.299% per attempt.

The successful random certificate was independently checked. It used a redundant contradiction argument and time witness 2 for a machine that halts after one step. The successful attempt took 27 actions including an abandoned start and rejected moves; it was not the canonical seven-action sequence.

## Exact probability of a particular action sequence

| Hand-written example | Sequence length | Uniform probability from the initial state |
|---|---:|---:|
| one_step | 7 | 1/52488 |
| two_step | 8 | 1/314928 |
| three_step | 9 | 1/1889568 |

These probabilities concern those exact sequences from the beginning. A longer attempt can reset or find another valid proof, so they are not the total success probabilities per attempt.

## Tighter experiment: 10,000 attempts, at most 20 actions each

The task seed was 20261025. Two machines of each size were sampled from the held-out hash partition without filtering on behavior. Random action seeds were 532, 533, 534, and 535. Each experiment stopped at its first accepted proof or 10,000 attempts. An additional one-state machine was an explicitly selected easy control.

| Task | Working states | Outcome | First successful attempt | Actions in successful attempt |
|---|---:|---|---:|---:|
| Sample 1 | 2 | No proof found | — | — |
| Sample 2 | 2 | No proof found | — | — |
| Sample 3 | 3 | HALT proved | 2591 | 13 |
| Sample 4 | 3 | No proof found | — | — |
| Easy control | 1 | HALT proved | 3547 | 20 |

**Yes: random search found a proof of a sampled three-state machine within 10,000 attempts and 20 actions per attempt.** This particular machine halts immediately from the blank start; its other working states are not visited on that run. The result does not demonstrate discovery of a non-halting proof or reasoning through all three states.

The successful three-state certificate selected the time expression `s(add(z, s(mul(z,z))))`, which evaluates to 2. Because a halted configuration is frozen by this prototype, time 2 is a valid witness even though the first halt occurs at time 1.

## Earlier experiment: up to one million attempts, 100 actions each

The same four machines were used. Three searches completed early with checked halting proofs: sample 1 at attempt 2,385; sample 3 at 613; sample 4 at 8,951. The fourth search (sample 2) was stopped to run the requested 10,000-machine experiment. Its last saved checkpoint was 344,950 attempts without a proof. **The million-attempt experiment is incomplete.**

The cache only reuses deterministic editor transitions. It does not add proof rules, remove actions, supply lemmas, or learn. Tests compare it against ordinary execution action by action, including its uncached fallback, and compare all example certificates. Simulation omits rendering observations. Existing reset/cancel actions remain available; an automatic reset also occurs at the end of each failed attempt.

## How many machines exist under these conventions?

For n labelled working states, each of 2n rows chooses a written bit, a direction, and one of n+1 next states including halt. There are `[4(n+1)]^(2n)` complete transition tables: **20,736** for two working states and **16,777,216** for three. Equivalent descriptions and unreachable states are counted separately.

The four-machine pilot is far too small to estimate the fraction of all machines that random proof search can solve. None of the unsuccessful searches proves unprovability.

Written machine tables and hand-written proofs are in `EXAMPLES.md`. Exact certificates, action traces, training metrics, model checkpoints, seeds, and simulator source are included.


## Completed 10,000-machine Monte Carlo experiment

**3,629 of 10,000 machines (36.29%) yielded a kernel-checked proof.** All 3,629 proofs established halting. No non-halting proofs were found. The other 6,371 machines exhausted their search budget without a proof.

We sampled 10,000 distinct labelled two-state, binary-symbol transition tables uniformly without replacement from the 20,736-table population. Every machine starts on a blank tape. The sampling seed was 20261027; the complete sample and independent per-machine action seeds are in `runs/two_state_monte_carlo_10000/plan.json`.

Each machine received at most 10,000 independent-reset attempts, with at most 20 proof-game actions per attempt, stopping at its first accepted proof. Actions were selected uniformly from the current available menu. Reset, cancel, and rejected actions consumed the budget. These are proof-game actions, not Turing-machine execution steps. No learned policy, proof demonstrations, or theorem archive was supplied.

The completed run used 74,759,653 attempts and 1,495,169,436 proof-game actions. This is the raw Monte Carlo result; it is not an estimate of how many machines possess a short proof.

Every successful certificate was checked during the run and independently rechecked in the final audit. The audit also verified all 10,000 distinct machine ranks, the seeded sample and action seeds, machine/certificate agreement, attempt limits, action limits, and aggregate totals. See `final_audit.json`, `report.json`, and `records.jsonl` in the run directory.

A cross-machine cache defect was caught during development: stored syntax alone did not identify machine-dependent derived goals. All 2,996 affected records were discarded and rerun with the same seeds. The final dataset contains 256 records from the original per-machine simulator and 9,744 from the corrected simulator, whose keys include derived goals. Differential checks against the original simulator passed. See `correction.json` and `cache_differential_validation.json`. The cache reuses deterministic computations only; it adds no proof actions or lemmas.

The run was resumed from completed records several times. The `elapsed_seconds` field in `report.json` measures only the final resumed segment, not total runtime. The effective shared-cache capacity was 500,000 states per worker (the implementation applies this floor to the CLI cache setting).

Reproduce with:

```bash
python -m proofgame.population_sample --sample 10000 --attempts 10000 --budget 20 --workers 6 --seed 20261027 --out runs/fresh_monte_carlo
```
