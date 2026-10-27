# Speed v2: implement the low-risk architectural wins first

The previous 448-case rollout matrix demonstrated that local CPU inference can be
much faster than centralized CPU/MPS inference. The 512-state real-gradient
benchmark demonstrated that exact two-pass action-gradient replay on MPS can
be ~2x faster than serial CPU learning for that *sample*.

**Speed v2 combines these independently measured strengths** without changing
the proof checker, exploration rate, sampled branches, group-relative rewards,
per-batch optimizer update cadence, or model architecture.

Everything below is opt-in. Existing training defaults remain unchanged.

## What's new

1. **Hybrid execution:** --device mps --mps-local-cpu-rollouts runs the
   *policy optimizer* on MPS, but keeps the eight symbolic/search/inference
   workers independent on CPU. The central GPU inference queue is not used
   for rollouts. Each frozen model version is snapshotted for CPU workers.
2. **Production two-pass group gradients:** --gradient-backend two-pass
   globally deduplicates action descriptions within an update, accumulates
   downstream action-feature gradients, and replays each unique action
   encoder with autograd once. Positive and negative group-relative
   coefficients are unchanged, and all sampled probabilities are checked
   against the frozen model, with a serial numerical fallback when needed.
   --gpu-gradient-batch-size controls context microbatches (8 recommended for
   the first Mac experiment). The original serial path remains default.
3. **Bounded compact structural topology cache:** --cpu-topology-cache-mb 32
   keeps input-only relation matrices in uint8, keyed by the exact parent
   tree and binder links rather than observation JSON. It can therefore
   reuse relations across different numeric budgets or lexical tokens
   with the same AST topology. Every forward still embeds current lexical
   tokens and current learned relation weights. The cache is independent
   of model version and contains no neural outputs.
4. **Focused benchmark** tests ordinary local CPU rollouts vs the topology
   cache at the **real 8-machine update boundary**, with cache-version
   resets, exact trajectory hashes and a full optimizer-step CPU-vs-MPS A/B.
5. **Stage-aware speed projections** report how the measured stage factors
   would affect the historical full epoch, with the remaining improvement
   required to hit the 30-minute goal. These are *projections*, never claims
   that a new complete epoch has been observed.

## Run the focused A/B first

Stop the old matrix/training job before benchmarking. This script runs a tiny
number of meaningful cases rather than 448 mostly dominated combinations.

~~~bash
git pull origin main

python -m proofgame.benchmark_speed_v2 \
  --run runs/ast_branch_race_8x6_fixed \
  --epoch 2 \
  --machines 16 \
  --machine-source saved-success \
  --workers 8 --barrier 8 \
  --cache-mb 32 \
  --repeats 2 \
  --learn-repeats 2 \
  --gradient-device mps \
  --context-batch-size 8 \
  --out runs/ast_branch_race_8x6_fixed/speed_v2_ab.json
~~~

Results are written to the output JSON after *each* completed rollout case.
The full optimizer A/B uses the exact same frozen-policy trajectories from one
complete machine batch. Both CPU serial and optimized MPS paths perform
group-gradient preparation, the optimizer step, clipping, and numerical checks.
The full resulting gradients and model parameter values are compared.

On a machine without MPS, use --gradient-device cpu to compare CPU serial with
CPU two-pass. Use --learn-repeats 0 for a rollout-only test.

**Important:** 16 saved-success machines are not statistically representative
of all training machines. This is a high-signal correctness and throughput
screen. Increase to 64 and select duration-stratified only if the first A/B
passes. The benchmark also checks *all* sampled trajectory hashes; drift
invalidates the rollout speedup comparison.

To separately isolate action-feature reuse *between machines*, add
--include-shared-action-cache (two more strategies). With exactly eight
workers and an eight-machine update boundary, each worker normally handles
only one machine per version, so we do not expect much cross-machine reuse.

## Calculate full-epoch speedup and the remaining gap to 30 minutes

The standalone calculator reads the actual recorded stage wall times and
held-out evaluation duration from the previous epoch. It never adds worker
CPU-second sums to the main-process wall clock.

~~~bash
python -m proofgame.speed_estimates \
  --run runs/ast_branch_race_8x6_fixed \
  --epoch 2 \
  --rollout-speedup 1.5 \
  --learning-speedup 2.0 \
  --evaluation-speedup 1.0 \
  --target-minutes 30 \
  --out runs/ast_branch_race_8x6_fixed/epoch_projection_v2.json
~~~

The speedup values above are **illustrative**, NOT measured v2 results.
Use the measured rollout/optimizer factors in speed_v2_ab.json once it is
available. The estimate holds checkpoint/snapshot overhead unchanged and
does not assume that faster rollout automatically makes evaluation faster.

## Optional new end-to-end training run

Once the A/B tests pass on the Mac, create a **new run directory** with the
original dataset and exactly the same attempts/branch rounds/group-relative
flags, learning rate, seed, and held-out evaluation procedure as the old run.
Do NOT resume the old directory with changed execution settings: strict resume
compatibility checks intentionally reject that.

Add these v2 flags to the otherwise identical original training command:

~~~text
--representation ast
--device mps
--mps-local-cpu-rollouts
--gradient-backend two-pass
--gpu-gradient-batch-size 8
--cpu-topology-cache-mb 32
--profile
~~~

Optionally add --cpu-shared-action-cache, but at eight workers/eight machines
per update it may provide little benefit. A new run should validate both
wall time and held-out quality before any speed settings become defaults.

For CPU-only training, use --device cpu and omit
--mps-local-cpu-rollouts; --gradient-backend two-pass and
--cpu-topology-cache-mb work on AST CPU runs too.

## What this does NOT attempt yet

- Rewriting the proof engine in C++: old profiles measured ~95% of worker
  task seconds inside inference, not proof-step execution.
- Breaking the eight-machine optimizer barrier: changes worker scheduling
  and must maintain exact branch-round/advantage semantics.
- Centralizing CPU/MPS rollout inference: already performed poorly on the
  observed 64-machine configurations.
- Changing AST token limits, attention bias, reward/exploration, or theorem
  verification to improve speed.
- Claiming a sample speedup is an epoch speedup. The focused report labels
  all projections and includes measured correctness/trajectory comparisons.

The biggest remaining potential gains are likely in local per-context
inference, worker scheduling at fixed optimizer boundaries, and low-overhead
structural matrix/tensor preparation. Follow up based on the focused results,
not a brute-force parameter matrix.
