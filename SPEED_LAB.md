# Speed lab: multi-hour experiments to find major bb-RL speedups

All experiments use a frozen checkpoint. They never call optimizer.step(),
modify training weights, change the proof objective, or overwrite a training run.
Run on the actual Mac hardware (including Apple MPS when available), without
another training job competing for CPU/GPU resources.

The previous 64-state synthetic-gradient microbenchmark found **1.46x** for
2-thread length-bucketed size-4 CPU gradients. The objective now is to discover
much larger speedups without weakening proof verification or changing the
group-relative gradient mathematics.

## High-value hypotheses

| Priority | Hypothesis | Test |
| --- | --- | --- |
| 1 | The same action Transformer encodings are needlessly recomputed for thousands of gradient states | Global unique-action, two-pass gradient replay with correct action-side gradients |
| 2 | CPU syntax/relationship construction is a major cost | Cached relation ID matrices vs fresh construction, cold/hot syntax timings |
| 3 | Dense padded attention is the neural bottleneck | Token-length bucketing at batch sizes 2/4/8/16; memory guard |
| 4 | Apple MPS is underused | CPU vs MPS *full forward and backward* (not inference only) |
| 5 | A machine-sized SGD batch creates a worker barrier | Worker counts 4/8/16/24 and frozen-policy barriers 8/16/32/64 |
| 6 | Repeated action embeddings dominate rollout inference | Per-worker shared action cache vs isolated caches |
| 7 | A central inference server enables better batching | Central CPU/MPS inference vs separate worker inference |
| 8 | GPU inference batching is limited by queue delays | Central batch sizes 4/8/16 and wait times 0/2/5 ms |
| 9 | Python overhead justifies C++/Rust | Python syntax/relations vs CPU PyTorch operator profiler |
| 10 | Long inputs dominate runtime | Include 512+/800+ token samples and report the length distribution |
| 11 | Better tensor kernels might dominate C++ gains | Follow up on SDPA/attention backend and fixed-shape torch.compile tests |
| 12 | Independent attempt parallelism can remove barriers | Measure frozen scheduling gain first; implement reproducible per-attempt streams only after proving group assembly invariants |

See the three scripts below. Start with the small real-gradient test; its
saved JSON explicitly flags numerical disagreement.

## 1. Real full-group gradients, with failed trajectories

This reruns actual complete group-relative attempts at the *frozen epoch-2*
checkpoint and calls the production group-gradient preparation function. It
retains unsuccessful trajectories **in memory**, so negative leave-one-out
advantages are represented. It tests serial, score_many, length buckets,
bounded structural caches, and an experimental global-action two-pass replay.

The two-pass algorithm computes each action's neural features once without
autograd, accumulates their exact downstream gradients across context groups,
then recomputes each unique action *once with autograd* and injects the
accumulated gradients. This retains gradients through the action encoder;
it is not a detached approximation. It is benchmark-only for now.

First trial:

~~~bash
python -m proofgame.benchmark_real_gradients \
  --run runs/ast_branch_race_8x6_fixed --epoch 2 \
  --machines 8 --states 64 --machine-source saved-success \
  --threads 1,2 --batch-sizes 2,4 --devices cpu \
  --repeats 2 \
  --out runs/ast_branch_race_8x6_fixed/real_gradients_small.json
~~~

Long trial (MPS is marked unavailable when unsupported):

~~~bash
python -m proofgame.benchmark_real_gradients \
  --run runs/ast_branch_race_8x6_fixed --epoch 2 \
  --machines 48 --states 512 --machine-source duration-stratified \
  --threads 1,2,4,8 --batch-sizes 2,4,8,16 --devices cpu,mps \
  --repeats 3 --relation-cache-mb 256 \
  --max-attention-elements 12000000 \
  --out runs/ast_branch_race_8x6_fixed/real_gradients_long.json
~~~

The original number of attempts/rounds and exploration strategy come from
the run's config. The script logs each completed machine. Use --states 0 to
benchmark all freshly generated gradient states instead of subsampling them.

Look for these JSON fields: median_wall_seconds, relative_gradient_l2,
max_absolute_gradient_error, max_sampled_probability_absolute_error,
passes_numerical_tolerance, action_reuse_factor, globally_unique_actions,
cached_action_feature_bytes, and two_pass_stages_median_seconds.

The newly sampled rollouts use the epoch checkpoint, NOT the evolving model
weights of the historical batch. An exact historical replay is impossible from
the saved training batch files because failed trajectories were removed.

## 2. Python structural preprocessing vs neural compute

~~~bash
python -m proofgame.benchmark_preprocessing \
  --run runs/ast_branch_race_8x6_fixed --epoch 2 \
  --batches 51,58,75 --states 256 \
  --max-context-tokens 1200 \
  --threads 1,2,4,8 --encoding-batches 1,2,4,8 \
  --cache-mb 256 --repeats 3 --torch-profiler \
  --out runs/ast_branch_race_8x6_fixed/preprocessing_profile.json
~~~

This isolates cached_tree parsing, relation_ids dense matrix creation,
length-bucketed policy.encode calls, and the heaviest CPU Torch operators.
Cache sizes are byte-bounded. Cached values are input-only integer relationship
matrices and deterministic positional tensors, never neural logits,
learnable relation biases, embeddings, or policy gradients.

**C++ should not be the first move without timing evidence.** If Python
relationship generation dominates, native code is worth testing. If attention
and matmul dominate, prioritize better batching, optimized Torch backends,
and/or GPU work first. PyTorch operator profiler timings include substantial
instrumentation overhead; compare unprofiled wall-clock medians instead.

## 3. Actual rollout workers, MPS, scheduling and cache matrix

Small smoke run:

~~~bash
python -m proofgame.benchmark_rollout_matrix \
  --run runs/ast_branch_race_8x6_fixed --epoch 2 \
  --machines 8 --strategies worker,worker-shared,central-cpu \
  --workers 4 --barriers 8 --inference-batches 4 \
  --wait-ms 0 --repeats 1 \
  --out runs/ast_branch_race_8x6_fixed/rollout_smoke.json
~~~

Long sweep (intentionally large factorial; can take hours):

~~~bash
python -m proofgame.benchmark_rollout_matrix \
  --run runs/ast_branch_race_8x6_fixed --epoch 2 \
  --machines 64 --machine-source duration-stratified \
  --strategies worker,worker-shared,central-cpu,central-mps \
  --workers 4,8,16,24 --barriers 8,16,32,64 \
  --inference-batches 4,8,16 --wait-ms 0,2 \
  --repeats 2 --max-cases 12 \
  --out runs/ast_branch_race_8x6_fixed/rollout_matrix.json
~~~

Every case is saved atomically. To continue after 12 cases, rerun with the
same arguments plus --resume and a larger --max-cases (or omit that cap).
Case order is deterministically shuffled to reduce order/thermal bias.

The output includes actions/s, attempts/s, worker elapsed-time occupancy,
central inference statistics and per-machine action-path hashes. Worker
occupancy is **not CPU utilization**.

Important: current training updates model weights after each 8-machine batch.
A frozen 32/64-machine barrier measures scheduling opportunity, *not a safe
drop-in algorithm*. Attempt-level distribution or updates that span barriers
must preserve the group-relative normalization, proof-round semantics, and
version-frozen on-policy probabilities, with explicitly tested RNG behavior.

## Decision rules

1. Compare the same rollout machines and fixed seeds, model checkpoint,
   coefficient set, float32 type, and observation ordering unless the
   experiment intentionally changes the ordering. Changed trajectories
   should be reported, not hidden.
2. Do not promote a faster gradient method if the complete gradient differs
   materially from serial, or if sampled policy probabilities fail on-policy
   checks. Real groups must include both positive and negative advantages.
3. Compare actual full backward pass plus preprocessing, not forward only.
   Monitor RAM/swap and thermal throttling on long Mac runs.
4. No new caching may hold stale weights or make loss depend on earlier
   updates. Reusing *input structure* across updates is safe; action neural
   features must be versioned or discarded.
5. Only enable the fastest passing method *optionally* in production training,
   then re-run held-out quality evaluation before changing any defaults.
   The prior approximately 72% evaluation is the quality reference.

## Send back

Upload real_gradients_small.json or real_gradients_long.json,
preprocessing_profile.json, and rollout_matrix.json. Use them to select
the fastest validated paths instead of guessing from theoretical bounds.
