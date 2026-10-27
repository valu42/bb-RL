# Follow-up speed experiments after the 512-state gradient results

The longer benchmark reported 19,053 real gradient observations (512 sampled
for the timed tests), 5,598 action references, 191 distinct actions (~29.3x
reuse), token lengths 130-1070, and excellent numerical agreement
(~2e-6 relative gradient L2). The best measured MPS two-pass gradient
was **1.692 s** for 512 states; the one-thread CPU serial baseline was
**3.364 s** (approximately 1.99x), while the best CPU two-pass variant
was **2.236 s** (MPS is approximately 1.32x faster). The much larger
"15.76x" refers only to the unusually slow MPS serial baseline.

We have now added two *experimental*, read-only follow-ups:

- **packed-score-many**: keeps regular, end-to-end PyTorch autograd through both
  encoders but separates unique actions into length-matched encoding batches
  instead of padding an entire action collection to the longest action.
  Contexts are packed too, under a per-encoding memory guard. This makes
  possible the comparisons previously skipped by the coarse attention guard.
- **--defer-checks**: transfers scalar loss results from MPS to CPU once
  after the entire gradient pass, instead of synchronizing on every single
  microbatch. It still checks final loss and gradient finiteness. The old
  per-batch behavior remains the default.

The two-pass method is still present. No production training code changes.

## Run on the same Mac, from the repository root

~~~bash
git pull origin main

python -m proofgame.benchmark_real_gradients \
  --run runs/ast_branch_race_8x6_fixed --epoch 2 \
  --machines 48 --machine-source saved-success --states 512 \
  --devices cpu,mps --threads 1,4 \
  --methods packed-score-many,two-pass \
  --batch-sizes 2,4,8 --packed-action-batch-size 8 \
  --max-attention-elements 12000000 \
  --relation-cache-mb 256 --repeats 5 --defer-checks \
  --out runs/ast_branch_race_8x6_fixed/packed_deferred_512.json
~~~

This reruns frozen-policy rollouts and uses real positive/negative
group-relative advantages. It logs all tested combinations. MPS and CPU
serial baselines in the same report use the *same deferred loss accounting*.

To isolate the effect of avoiding per-microbatch MPS synchronizations, run
the **exact same command** once more WITHOUT --defer-checks and with
a distinct output name (for example packed_synchronized_512.json).
The seed, selected machines, reward coefficients and checkpoint remain
fixed. Compare the results using the *CPU serial baseline* as the common
reference, not MPS serial vs MPS batched alone.

If packed-score-many wins with low gradient/probability error, it can be
evaluated as an optional production path. If two-pass remains faster, the
largest remaining target is context-encoder computation/preparation.

For a stress test, use --states 1024 and/or a larger machine sample. Avoid
blindly raising max-attention-elements: it bounds each dense structural
attention encoder batch; high values can cause large memory use or swap.

## Next after this

Run benchmark_preprocessing as described in SPEED_LAB.md. The longer
512-state cache experiment recorded ~134 MB of relationship matrices and
thousands of misses with only hundreds of hits. Caching isn't universally
helpful on long heterogeneous observations. Test larger cache budgets,
compact relation-ID storage, and input preparation (CPU -> MPS copies) **only
after** measuring a representative input-length distribution.

Finally, prioritize the separate frozen rollout matrix to test worker
occupancy and central GPU inference. Scheduling a larger frozen batch is an
*upper bound*, not a drop-in training update change.

Please send packed_deferred_512.json and packed_synchronized_512.json, along
with preprocessing_profile.json if available. Avoid treating a microbenchmark
speedup as a prediction for the full 89.8-minute epoch.


## Rollout matrix follow-up: simulate optimizer-version cache resets

The first seven 64-machine runs revealed a major result: local CPU inference
with eight workers and shared action features ran in 73.51 seconds
(3,959 actions/s), while sampled centralized CPU configurations needed
369-477 seconds and centralized MPS configurations needed 656-1,157 seconds.
All reported 291,053 actions, but **worker count, attention batch size and
barrier size also differed**. There is not yet a controlled isolated gain
attributable solely to sharing action features.

IMPORTANT COMPARABILITY ISSUE: the original matrix uses a single frozen
version ID 0 for all 64 machines. In production, the training loop changes
weights and invalidates action embedding caches after EVERY 8-machine SGD
batch. Thus the impressive 64-machine shared-cache result may benefit from
reuse across machine boundaries that production cannot retain.

The matrix now has a read-only --simulate-update-barriers flag, which keeps
weights frozen for apples-to-apples action trajectories but increments the
frozen version at each configured barrier. Both worker-local feature caches
and central inference caches must invalidate old versions. It reports
simulated_update_barriers and frozen_weight_versions. No model updates occur.

**After the ongoing old matrix finishes or is stopped**, perform a focused,
training-representative CPU comparison (four configurations, NOT 448):

~~~bash
git pull origin main

python -m proofgame.benchmark_rollout_matrix \
  --run runs/ast_branch_race_8x6_fixed --epoch 2 \
  --machines 64 --machine-source duration-stratified \
  --strategies worker,worker-shared \
  --workers 8 --barriers 8,64 \
  --inference-batches 8 --wait-ms 0 --repeats 1 \
  --simulate-update-barriers \
  --out runs/ast_branch_race_8x6_fixed/rollout_versions_cpu.json
~~~

This tests isolated per-machine versus shared action caches under identical
worker count, with both training-sized and intentionally large barriers.
The order of four cases is shuffled. The 64-machine barrier is still an
upper-bound scheduling experiment; the **8-machine** barrier is the more
relevant production configuration. The cache-version resets at barrier 8
closely model the per-minibatch invalidation cost but NOT changed action
probabilities from real gradient updates.

If CPU-local paths remain ahead, a further four-case focused comparison
should include central-cpu and central-mps at **workers=8, barrier=8,
inference_batch=8, wait=0**. Do not use cross-configuration speedups
with different barriers/worker counts to predict full-epoch timing.

Existing rollout_matrix.json remains resumable with its old signature when
--simulate-update-barriers is omitted. Use a DIFFERENT --out for the new
matrix, as in the example. The original 448-case Cartesian product is not a
requirement; --max-cases 12 stops after 12 cases.
