# Validation of this prototype

Executed on 25 October 2026 with Python 3.12 and PyTorch 2.14.1+cpu.

- `python -m unittest discover -s tests -v`: **24 tests passed**, including the
  three optional model tests. Without PyTorch, those model tests are skipped.
- `python -m proofgame demo one_step`: accepted HALT, 7 public actions.
- `python -m proofgame demo two_step`: accepted HALT, 8 public actions.
- `python -m proofgame demo forever`: accepted NONHALT, 39 public actions,
  including a lemma, natural induction, and a case split on the scanned bit.
- Complete certificates are independently rechecked from their serialized
  trees. The demonstrations are human-written validation cases, not discoveries
  by a trained model and not inputs to training.
- Five training episodes on uniformly sampled two- and three-state machines,
  with a 24-action episode budget, width 32, and one Transformer layer,
  completed. Parameters changed and losses remained finite.
- The uniform-random baseline, untrained greedy policy, and trained greedy
  policy were evaluated on the same three held-out machine IDs. Each found
  zero proofs within that budget.
- Reloading the saved checkpoint and evaluating it through the standalone
  evaluation command completed successfully.

See `runs/verified_smoke/config.json`, `metrics.jsonl`, and `report.json` for the
exact run. The included checkpoint is only a smoke-test artifact. The training
command always initializes fresh random weights.

These checks validate the implemented paths, not formal soundness of the
implementation, broad machine coverage, or successful learned proof search.

## Follow-up experiment

- All 24 tests passed again with the added three-state halting, two-state
  non-halting, and blank-sweeper certificates included in the example test.
- The three-state halting fixture takes 9 actions; the two-state non-halting
  fixture takes 49; the blank-sweeper invariant proof takes 39.
- A fresh random initialization was trained for 200 episodes, each with a
  64-action budget, on randomly sampled two- and three-state machines. No
  demonstration, initial theorem archive, or intermediate reward was used.
  It found zero proofs. The report and checkpoint are in `runs/longer_200`.
- Independent uniform-action trials found 0/1,000 proofs on the mixed held-out
  machine distribution, and 1/1,000 on the fixed one-step-halting machine.
  The successful random certificate was rechecked and its 27-action trajectory
  reproduced from the seed. It was not supplied to training.

## Random-search simulator and tighter budget

- Latest full suite: **27 tests passed**. The three added tests compare cached
  transitions with ordinary execution, exercise uncached fallback, replay every
  proof fixture, and check that suppressing observations changes no proof state.
- The same simulator tests also pass under PyPy 3.11, used for the longer run.
- With 10,000 attempts and a 20-action cap per attempt, sample 3 produced a
  checked HALT certificate on attempt 2,591 using 13 actions. The easy control
  succeeded on attempt 3,547 using 20 actions. All three other sampled machines
  exhausted their 10,000 attempts without a proof.
- The separate one-million-attempt/100-action experiment is partially complete
  at this snapshot. Do not treat its unfinished case as a completed failure.


The completed 10,000-machine Monte Carlo dataset passed its final audit: all 10,000 unique assignments and budgets checked, and all 3,629 successful certificates rechecked. See `runs/two_state_monte_carlo_10000/final_audit.json`.


The new frozen-network sampler was checked against direct network probabilities, including cache changes in machine, remaining budget, and error state. The suite passed all 31 tests with PyTorch; the dependency-free run passed with four optional neural tests skipped. The 5,000-episode training run and paired 1,000-episode evaluations passed their recorded audits.


The curriculum extension passed all 32 tests. Its update test refuses incomplete trajectories without optimizer mutation and checks that a verified trajectory can increase in probability. The actual pilot rechecked all four selector certificates, replayed both neural training proofs, verified budgets and checkpoint consistency, and replayed its held-out proof. The evaluator now optionally records successful certificates and exact action traces; its existing model tests passed after that addition.
