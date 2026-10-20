# DESIGN

This document separates implemented behavior, measured evidence, and proposed improvements.
Numbers quoted come from the saved files in [results/](results/) (raw CSV, JSON, plots).
Reproduction commands and the challenge-to-code map are in [README.md](README.md).
The serving design in section 5 is a proposal, not an implemented service.

## Implementation overview and execution order

The NumPy-based core consists of the [autodiff engine](engine/tensor.py),
[layers](nn/layers.py), [optimizers](nn/optim.py), and [pruning controller](prune/pruner.py).
Scikit-learn is used only to load digits; Matplotlib produces plots.
The default model is `64 -> 256 -> 128 -> 10`, with ReLU hidden layers and no
activation after the output logits. The 256/128 widths are experimental design choices.

Each training batch follows this order in [the shared loop](train/loop.py):

```text
zero_grad -> forward -> mean cross-entropy -> backward
          -> observe statistics -> maybe update masks -> optimizer step
```

The two pruning calls are omitted for dense training. Gradients are calculated
before mask changes, then the optimizer synchronizes its state with the new mask.
Newly disabled entries are excluded from the update. A revived entry has a zero
parameter gradient from that masked forward pass, while its effective-weight gradient
remains available to rank regrowth; it learns through subsequent active forward passes.
The graph is rebuilt every batch. Mask changes alter the effective function but do not
physically delete nodes or switch training to sparse kernels.

He initialization uses standard deviation `sqrt(2/fan_in)` in the default ReLU hidden
layers. The output layer uses Xavier, `sqrt(2/(fan_in+fan_out))`, when He is requested.
The initialization study compares these with standard-deviation-1 and 0.01 baselines;
it does not establish that one initialization is universally optimal.

## 1. Importance criterion: why `1/2 * F * w^2` approximates "loss increase if removed"

Removing connection `i` means the perturbation `dw_i = -w_i`. Expand the training loss around the
current weights:

    L(w - w_i e_i) - L(w)  =  -g_i w_i  +  1/2 H_ii w_i^2  +  O(w_i^3)        (g = dL/dw, H = d2L/dw2)

* **Magnitude** `|w|` has the same ranking as a positive quadratic removal cost if
  the linear term is negligible and every `H_ii` is the same positive constant. In general they are not:
  a weight feeding many strongly-active inputs has large curvature, one feeding a mostly-zero
  input has almost none (`H_ii ~ E[x_i^2 * ...]`). Magnitude also is not comparable across layers
  with different fan-in, which matters whenever sparsity is allocated by one global threshold.
* **First-order Taylor** `|E[g w]|` keeps the linear term. It is informative early in training,
  but where training has converged `E[g] -> 0` and the term vanishes into noise. We implement it
  (`taylor1`) as a baseline/ablation.
* **Saliency used here (`fisher`)** uses a second-order-style score and substitutes the
  *empirical Fisher diagonal*: `F_ii = E_n[(dl_n/dw_eff_i)^2]`, so
  `score_i = 1/2 F_ii w_i^2`. This is an Optimal Brain Damage-inspired approximation,
  not an exact Hessian calculation. The model Fisher, which averages over model-generated
  labels, has an expected negative-log-likelihood Hessian identity under regularity
  assumptions. Our empirical Fisher instead uses observed labels and need not equal
  the model Fisher, generalized Gauss-Newton matrix, or observed-loss Hessian, even at
  an empirical optimum. The implementation ranks within each layer and does not rely
  on calibrated cross-layer score comparability.
* **Exact statistic, approximate curvature.** For `y = xW_eff + b`,
  `dl_n/dW_eff_ij = x_ni * d_nj`. Because the batch loss is a
  mean, `y.grad = d / N`, hence `F = N * (x**2)^T @ (y.grad**2)`: one extra matmul per layer, no
  per-sample backward passes. `tests/test_prune.py::test_empirical_fisher_matches_bruteforce_per_sample_gradients`
  checks this against N separate size-1 backward passes. Statistics are smoothed with an EMA
  (beta = 0.9) so a mask decision does not hang on one mini-batch.

**Caveats we are not hiding.** (a) The empirical Fisher is only a curvature proxy
(Kunstner et al. 2019); convergence alone does not make it the observed-loss Hessian.
(b) Single-weight scores ignore interactions: removing the lowest-scoring 10% of weights is not
the sum of 10% single removals. Gradual pruning with retraining between steps is what makes this
tolerable. (c) `results/criteria_correlation.json` measures, rather than assumes, how well each
score rank-correlates with the *true* single-weight loss increase; read it before trusting (a).

## 2. What is "the gradient of a masked weight"?

The mask is a *constant multiplication inside the graph*: `W_eff = W * M`, `y = x @ W_eff + b`.
Two different gradients exist and the engine keeps them separate:

| quantity | value at a masked entry | meaning |
|---|---|---|
| `W.grad` (parameter gradient) | **exactly 0** | the true derivative of `L(W * M)` w.r.t. `W`: wiggling a masked weight changes nothing |
| `W_eff.grad` (`Linear.dense_grad`) | `x_i * delta_j` summed over the batch; may be zero or nonzero | local sensitivity to the effective connection; retained for observer statistics and regrowth ranking |

Why this is the right choice: the optimizer must see the first one, because it is the honest
derivative of the function being trained; a straight-through estimator (letting the dense
gradient flow into the masked weight) makes the hidden weight drift while it is masked, so a
revival teleports it to a value nothing ever validated. The second one is what lets regrowth
rank dormant connections without any special-case autodiff op.

**Optimizer consequences** (`nn/optim.py`, each demonstrated by a test in `tests/test_masked.py`):

1. Pruned weights are set to an exact `0.0` (not near zero) and never updated.
2. When an entry's mask flips in either direction, its Adam `m`, `v` and step counter are reset to 0.
   Otherwise stale moments (non-zero, from a different regime) would kick a revived weight.
3. Bias correction uses a **per-element step count**. With a supplied nonzero gradient and
   fresh state, the first update has magnitude `lr * |g|/(|g| + eps)`, approximately `lr`
   when epsilon is negligible. With fresh moments but a sufficiently old global counter,
   the magnitude instead approaches `lr * 0.1/sqrt(0.001)`, about `3.16 lr`, ignoring
   epsilon. The tests exercise nonzero-gradient revival explicitly and compare both counters.
   In the normal training loop, revival follows backward, so that entry's parameter
   gradient is still zero for the current batch. Its counter becomes 1 on that active
   optimizer step, with no movement if its fresh moments and gradient are zero.
4. A revived weight starts at 0, so revival alone does not change the network function.
   Other removals in the same event can change predictions.

## 3. Pruning schedule and regrowth

* **Cubic ramp** (Zhu & Gupta 2017), `s_t = s_f (1 - (1 - (t - t0)/(t_end - t0))^3)`: prunes
  fastest when the net is most redundant and slows near the target where each removal costs more.
  A dense warm-up (10% of steps) precedes it and the last 40% of training is fine-tuning on the
  frozen mask. One-shot pruning is kept as an ablation (`fisher_oneshot`).
* **Allocation:** every layer is pruned to the same density (per-layer uniform), identical for all
  criteria, so the comparison isolates the *ranking* function. A learned/global allocation is
  a possible improvement to evaluate, not an established result here.
* **Regrowth (optional, `fisher_regrow`):** at each event a fraction (cosine-decayed to 0 by
  `t_end`) of the budget is given to masked connections with the largest `|EMA(dL/dW_eff)|`; they
  re-enter at 0. Connections dropped in the same event are not eligible (no flip-flop), and the
  decay to 0 is the stability argument: the final mask stops changing before fine-tuning, so the
  last phase optimizes a fixed function class.

The hard budget is an active-weight count, not a hardware runtime guarantee. At the
final target, each layer retains `round((1-s_final) * weight_count)` connections.
At 90%, the three counts are 1,638, 3,277, and 128: 5,043 of 50,432 weights,
or about 90.0004% achieved sparsity. Biases are not pruned.
With 1,258 training examples, batch 64, and 60 epochs, there are 1,200 planned
steps. The default mask events run from step 120 through 720 inclusive, every
10 steps: 61 events, including the initial zero-sparsity event.

## 4. Performance costs and candidate optimizations

* Potential per-op Python overhead: operations create Tensors, backward closures, and
  array results. The sweep's total `seconds` measurements are not a profiler and do
  not establish whether Python, allocation, or BLAS dominates. Profile before choosing
  an optimization. Candidates include fusing `Linear + bias + activation`, reusing
  buffers, or replaying a traced graph, each with fresh gradient checks.
* Masked training is dense: `W * M` is recomputed every forward, and the observer adds one extra
  matmul per layer for the Fisher. Training therefore gets *no* speedup from sparsity here; only
  inference can benefit in some measured regimes (section 6). A possible future
  optimization is sparse computation during the frozen-mask phase, subject to benchmarking.
* float64 everywhere doubles memory traffic versus float32 (kept for clean gradient checks).
* The graph is rebuilt every step; no checkpointing/recompute (irrelevant at this size).

## 5. Serving a self-pruned model in a multi-tenant service at thousands of requests/s

**Proposed architecture only:** this repository does not implement an inference API,
checkpoint export/loading, tenant isolation, quantization, or a throughput load test.
Thousands of requests/s is a target to validate, not a measured capability.

1. **Unstructured sparsity is not free speed.** BLAS/GPU kernels are dense; our own NumPy
   sparse path (see `results/cost.md`) shows the FLOP reduction does not automatically become a
   wall-clock reduction. For production, decide by measured p50/p99 on the target hardware, not
   by FLOPs. Options: sparse kernels only past a high-sparsity breakeven; **structured** variants
   (N:M such as 2:4 on NVIDIA Sparse Tensor Cores, block sparsity, or removing whole dead neurons
   and folding their constant output into the next bias) which dense hardware can exploit.
2. **Memory is a potential multi-tenant win.** The current float64 sparse representation
   reduces measured model-array storage. A production artifact would also need architecture,
   training-derived preprocessing statistics, label mapping, version, and checksums. Evaluate
   CSR/CSC and lower-precision values separately for accuracy and hardware support. Actual
   resident memory and cold-start latency require measurement including runtime overhead.
3. **Request path:** dynamic micro-batching per model (bounded by a latency deadline), stateless
   workers, immutable versioned artifacts (weights + masks + checksums), LRU model cache with
   per-tenant memory quotas, admission control and per-tenant rate limits so one tenant cannot
   starve another.
4. **Release safety:** a pruned model is a new model. Gate promotion on an offline accuracy
   regression budget, shadow traffic, then canary with automatic rollback on accuracy proxy / error
   rate / p99. Keep the dense checkpoint to roll back to.
5. **Observability:** per-model latency histograms, achieved sparsity and a checksum of the mask
   in every response header/log line, so "which model served this" is always answerable.

## 6. Results and what they do / do not show

The accuracy sweep below uses 8 seeds; seed `s` fixes the train/test split (70/30 of the 1,797 digits, 539 test
images), initialisation and batch order, so methods are compared on **paired** runs. Network
64-256-128-10 (50,432 prunable weights). Reported accuracy is the **final epoch** (no checkpoint
selection). `results/pareto_table.md`, `pareto_summary.json`, `pareto_raw.csv`, `pareto.png`.

| target | magnitude | taylor1 | fisher (ours) | fisher - magnitude (paired, 95% CI) | W/T/L |
|---|---|---|---|---|---|
| 50% | 97.63 | 97.82 | 97.91 | +0.28 +/- 0.68 pt (n.s.) | 5/1/2 |
| 75% | 97.59 | 97.91 | 97.84 | +0.26 +/- 0.26 pt (n.s.) | 5/2/1 |
| 90% | 97.54 | 97.70 | 97.68 | +0.14 +/- 0.49 pt (n.s.) | 6/0/2 |
| 95% | 97.56 | 97.43 | 97.29 | **-0.28** +/- 0.33 pt (n.s.) | 1/2/5 |
| 98% | 95.73 | 96.29 | 96.57 | +0.83 +/- 0.40 pt (significant) | 8/0/0 |
| 99% | 82.31 (sd 10.3) | 84.04 | 95.83 (sd 0.7) | +13.52 +/- 8.39 pt (significant) | 8/0/0 |

Dense: 97.50% +/- 1.03 (sd over seeds). W/T/L means paired wins/ties/losses.
The paired +/- values are 95% t-interval half-widths; they are not the sample
standard deviations shown in [the full table](results/pareto_table.md). These are
per-comparison intervals, not a multiple-comparison-adjusted analysis.

**Honest reading.** Up to 95% sparsity, importance-based pruning is *not* distinguishable from
magnitude pruning in this setup (differences are inside the seed noise; at 95% magnitude is
nominally ahead). The three main gradual criteria at <=95% are within about 0.33 pt of dense, inside the
split-to-split noise (sd 1.0 pt), so we do not claim pruning "improved" accuracy even where the mean is
slightly above dense. The criterion matters only at the extreme end (98-99%), where the compared
magnitude baseline becomes unstable (sd 10 pt across seeds). We did not investigate *why*; one
hypothesis (untested) is that uniform per-layer allocation leaves the 128x10 output layer with ~13
weights at 99% and magnitude ranking breaks classes apart more often than a curvature-aware score.

Ablations (paired vs `fisher`): one-shot pruning is worse at every level where it was tried
(-0.58 pt at 90%, significant; -25 pt at 98%); regrowth (`fisher_regrow`) gave no significant
benefit at any level (-0.23 / -0.14 / +0.19 / -0.37 pt at 90/95/98/99%).

**Criterion validity** (`results/criteria_correlation.json`): Spearman correlation between each
score and the *measured* loss increase from zeroing a single weight (500 random weights per
layer). After 40 epochs `fisher` correlates best in layers 0 and 2 (0.79, 0.90 vs magnitude 0.57,
0.72), but in layer 1 the signed first-order term `-g*w` beats it (0.66 vs 0.52). The Fisher
score is a useful proxy here, not a faithful predictor.

**Cost** (`results/cost.md`, seed-0 models): active weights and MACs drop exactly as the mask says
(90% -> 10.0x fewer MACs, 98% -> 50.0x) and our sparse forward path matches the dense masked
forward to ~1e-14. **Wall-clock is a different story:** the NumPy sparse path is faster than the
dense BLAS path only at batch size 1 (1.4-2.2x); at batch 32 it is 0.2-1.0x and at batch 539 it is
0.06-0.27x, i.e. *slower*. The dense-times-zero path does the same FLOPs as the unpruned network and
is not counted as a saving. The reliable wins here are counted inference MACs, active
parameter count and model-array storage (406,608 -> 67,860 bytes at 90%), not batched latency.
The MAC count excludes bias additions, activations and sparse indexing overhead. Storage
includes sparse index arrays and biases, but excludes Python objects, intermediate buffers,
and process/runtime memory. Timings use the minimum of seven rounds of 200 forwards
after warm-up, not an end-to-end request benchmark or a p99 service measurement.

## 7. Known limitations

* Small dataset (8x8 digits), one architecture; conclusions may not transfer to larger models.
* The split is train/test only (no validation set), and hyperparameters (lr 2e-3, 60 epochs, EMA beta
  0.9) were fixed once for all methods rather than tuned per method; the test accuracy was logged each
  epoch for the curves but only the final-epoch value is reported. A cleaner protocol would add a
  validation split.
* n = 8 seeds with t-based intervals; the 99% comparison has high variance, so its CI is wide.
* Per-layer-uniform sparsity only; the 0% point is the dense baseline.
* Dependencies are not fully version-locked, and clean-clone reproduction remains a
  submission check. No production throughput or tenant-isolation claim has been validated.

## 8. Correctness evidence and live-defense boundaries

The [gradient tests](tests/test_gradcheck.py) exercise supported operations and broadcasting
against central finite differences. The [mask tests](tests/test_masked.py) distinguish raw
and effective gradients, check frozen data/state, and exercise Adam revival and intentionally
naive alternatives. The [pruning tests](tests/test_prune.py) cover budget behavior,
regrowth, the per-sample Fisher formula, and sparse/dense output equivalence.
The full suite passed locally (37 tests); this supports the tested cases, not a proof
for every input, architecture, or floating-point regime.

For a live extension, add the operation and backward rule in the engine and test
broadcasting and finite differences, or add a score to the existing criterion registry
and test its ranking and budget interaction. No such extra operation or criterion
is claimed to have been implemented live. The candidate must explain and extend
the code independently; passing tests and written explanations do not replace that defense.
