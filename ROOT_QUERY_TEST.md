# First-output-only Transformer: one controlled A/B test

This experiment compares **the existing encoder** with **an encoder that computes only the first output token**. Everything else stays the same: checkpoint weights, input tokens, positional encoding, learned relationship biases, action ordering, gradient coefficients and ordinary autograd. There is no two-pass gradient method, additional topology cache, C++ dependency, new training architecture, or parameter grid.

## Run from the repository root

Stop other training/benchmark processes first.

```bash
git pull origin main

python -m proofgame.benchmark_root_query \
  --run runs/ast_branch_race_8x6_fixed \
  --epoch 2 \
  --out runs/ast_branch_race_8x6_fixed/root_query_test.json
```

Defaults: one CPU thread, eight machines, up to 256 real gradient states, one explicit warmup and three timed repeats per implementation. This is the first test to run because independent CPU inference was the useful rollout architecture in the earlier measurements.

The script first regenerates complete group-relative attempts at the frozen epoch checkpoint, retaining failed trajectories, and uses production gradient preparation. The timed sample must contain both positive and negative coefficients. It prints collection progress, numerical-check results, and every completed timing.

It saves:

* `root_query_test.json`: settings, checkpoint SHA-256, workload statistics, numerical errors, individual timings, warmups, speed ratios and stability flags. **Send back this file.**
* `root_query_test.workload.json`: reusable observations and exact coefficients, tied to the checkpoint hash. Keep this locally for further tests; it avoids regenerating the proof attempts.

Existing files are never silently overwritten. Choose a new `--out` for subsequent runs.

## What a pass means

Before timing, the script compares all sampled action logits and probabilities, the signed loss, and gradients for every named parameter. It additionally compares the sampled probabilities with their original frozen-rollout values. A mismatch stops the test and saves the partial report. No optimizer step is performed, and the original checkpoint hash and model tensors are checked for modification.

Timings include JSON/syntax processing, input tensors, the model, and backward where applicable. Parsed-syntax caches are cleared equally before each pass; frozen action caches start empty for each inference pass and are separate per machine. Variant order alternates, MPS is explicitly synchronized when used, and warmup timings are recorded separately. A max/min duration ratio above 1.5 marks a result unstable and suppresses `validated_speedup`; raw measurements remain visible.

The final output is shaped like:

```text
ROOT-QUERY A/B SUMMARY (not a full-epoch forecast)
cpu inference: original=...s root=...s ratio=...x stable=True
cpu gradient: original=...s root=...s ratio=...x stable=True
Report: .../root_query_test.json
Reusable workload: .../root_query_test.workload.json
```

The inference row replays the existing frozen sampler on fixed observations. It does **not** include proof-state transitions, worker communication, scheduling or end-to-end proof-search wall time. The gradient row includes full ordinary backward, but not group preparation, AdamW or checkpoint writing. These are useful component measurements, not proof of a particular epoch runtime or retained held-out accuracy.

## Optional: test MPS on exactly the same observations

After the CPU test, without repeating proof collection:

```bash
python -m proofgame.benchmark_root_query \
  --run runs/ast_branch_race_8x6_fixed \
  --epoch 2 \
  --devices mps \
  --reuse-workload runs/ast_branch_race_8x6_fixed/root_query_test.workload.json \
  --out runs/ast_branch_race_8x6_fixed/root_query_mps.json
```

MPS gradients use length-ordered microbatches up to eight, identical for original/root variants. The baseline's padded attention budget can reduce a microbatch; valid long inputs are not silently dropped. MPS inference deliberately measures the same synchronous per-state sampler, **not** a newly batched inference coordinator. Unsupported MPS is reported as unavailable. Compare each root implementation against its own same-device, same-batching baseline; do not call a ratio against serial MPS a CPU speedup.

To test CPU and MPS in one initial command, add `--devices cpu,mps`. To use every generated gradient state, add `--states 0` before generating the workload. To repeat an existing workload, use `--reuse-workload`; `--states` does not silently resample a reused file.

## Why the shortcut is valid here

For one Transformer encoder layer, output token zero consumes all **input** key/value tokens, but not the other tokens' **output** states. The discarded query rows and their residual/norm/feed-forward outputs are unnecessary. The implementation computes every input embedding and key/value projection, only the first query, and only the first output's residual/norm/feed-forward path. Learned relation weights remain differentiable; only their root-query row is constructed.

`encode_tree` emits one connected tree with root token zero. Its relation row is SELF for zero, PARENT for direct children, ANCESTOR for other descendants, with the original binding precedence retained. Tests compare this row exactly against the complete relation matrix. No observation is shortened, and token limits/padding match the original.

The temporary override is rejected for byte models, multiple encoder layers, nonzero dropout, unsupported attention configurations, or an enabled topology cache. It is restored even after exceptions. Production forward methods, checkpoint format and training defaults are unchanged.

## Regression tests

```bash
python -m unittest discover -s tests -p 'test_root_query.py' -v
```

Coverage includes nonzero learned relation weights, variable lengths and padding, float32/float64 encoder checks, double-precision input/mask/all-parameter gradients, pre/post normalization, ReLU/GELU, complete policy scores and value head, signed losses, restoration/architecture guards, checkpoint immutability, and a complete CLI round trip. Hardware-gated MPS derivative tests run on the Mac and are skipped on CPU-only CI.

The collected workload favors historically successful machines and may not represent the entire training distribution. New sequence shapes can incur new accelerator setup costs. A successful microbenchmark should be followed by a small end-to-end rollout/training comparison before making this the default.
