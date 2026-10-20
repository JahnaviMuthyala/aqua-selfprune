# AQUA self-pruning network (NumPy only)

A from-scratch reverse-mode autodiff engine, MLP, Adam optimizer and a network that **prunes
itself during training** under a cubic sparsity schedule, with saliency (empirical-Fisher / OBD)
importance, optional regrowth, a sparse-aware inference path, and an honest evidence pipeline.

**Allowed-library statement:** the autodiff engine, layers, optimizer, pruner and training loop use
only NumPy. `scikit-learn` is used in exactly one place, `train/data.py`, to *load* the bundled 8x8
digits dataset (no sklearn models, training or gradients). Matplotlib is used for plots.

## Layout
```
engine/  tensor.py (autodiff), gradcheck.py        nn/     layers.py (masked Linear, MLP), optim.py (Adam, SGD)
prune/   criteria.py, schedule.py, pruner.py, sparse.py (sparse-aware forward)
train/   data.py, loop.py, run_part2.py, run_part3.py, run_part4.py, run_cost.py, run_criteria_check.py, analysis.py
tests/   test_gradcheck.py  test_masked.py  test_prune.py        results/  committed plots + raw numbers
DESIGN.md  (derivations, design answers, results, limitations)
```

## Install (Python >= 3.10)
```
pip install -r requirements.txt          # or: uv pip install -r requirements.txt
```

## Commands
| what | command | time (1 CPU) |
|---|---|---|
| gradient-check suite (Part 1) | `pytest tests/test_gradcheck.py` | seconds |
| masked-weight correctness (Part 1) | `pytest tests/test_masked.py` | seconds |
| all tests (37) | `pytest` | seconds |
| Part 2: dense training + learning curve + init study | `python -m train.run_part2` | ~1 min |
| Part 3: one self-pruning run (90%) | `python -m train.run_part3` | ~10 s |
| Part 4: Pareto sweep, baselines, ablations (8 seeds) | `python -m train.run_part4 --seeds 8` | ~25 min |
| Part 4: re-analyse without retraining | `python -m train.run_part4 --analyze-only` | seconds |
| Part 4: cost measurement (run on an idle machine) | `python -m train.run_cost` | ~1 min |
| criterion vs true loss increase | `python -m train.run_criteria_check` | ~1 min |

Seeds are fixed (`seed` controls split, init and batch order). The sweep is resumable.

## Headline results (details and caveats in DESIGN.md section 6)
* **Part 2:** dense 64-256-128-10 MLP reaches 98.3% test accuracy (seed 0), NaN-free; He init vs
  Xavier vs deliberately bad inits compared in `results/part2_init_ablation.json`.
* **Part 3 (seed 0, 90% target):** achieved sparsity 90.0004%, test accuracy 97.96% vs 98.33% dense
  (single run: 2 more mistakes out of 539 images); masked weights verified to be exact zeros.
* **Falsifiable claim (8 seeds, paired):** at **99% sparsity** (about 504 of 50,432 weights
  kept), saliency pruning retains **95.83% +/- 0.72** test accuracy (98.3% of the dense 97.50%)
  versus **82.31% +/- 10.29** for magnitude pruning at the same per-layer sparsity: paired
  difference **+13.5 pt, 95% CI +/- 8.4, 8 of 8 seeds favour saliency**. At 98%: **+0.83 pt +/- 0.40,
  8/8**. At **<= 95% the two are not distinguishable** (at 95% magnitude is nominally ahead by
  0.28 +/- 0.33 pt, not significant). Reproduce with `python -m train.run_part4 --seeds 8`; numbers in
  `results/pareto_summary.json`.
* **Cost:** 90% / 95% / 98% sparsity cuts MACs per sample 10x / 20x / 50x and storage
  406,608 -> 67,860 / 36,364 / 17,052 bytes. Our NumPy sparse forward is faster than dense BLAS only at
  batch size 1 (1.4-2.2x) and **slower** at batch 32 and 539; the dense-times-zero path is the same
  FLOPs as the unpruned net and is not counted as a saving. See `results/cost.md`.

## What the masked-weight tests prove (`tests/test_masked.py`)
Masked weights contribute exactly zero; their parameter gradient is exactly zero while the gradient
w.r.t. the effective weight matches finite differences; Adam never moves a masked weight and keeps its
state at zero; a revived connection's first step is exactly `lr`; and three naive alternatives
(updating under the mask, global-step bias correction, stale momentum) are shown to be wrong.
