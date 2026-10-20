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
Run commands from the project root: the directory containing this file and the package folders.
For an isolated environment on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m train.run_part2
```

In the commands below, replace `python` with `.\.venv\Scripts\python.exe` when using this
environment. On Linux/macOS the equivalent interpreter is `.venv/bin/python`.
Alternatively, install dependencies into your selected interpreter with
`python -m pip install -r requirements.txt`.

## Challenge requirements and implementation

| Part | Implementation | Evidence |
|---|---|---|
| 1: reverse-mode autodiff and changing masks | [Tensor engine](engine/tensor.py), [finite-difference checker](engine/gradcheck.py), [mask-aware optimizers](nn/optim.py) | [Operation tests](tests/test_gradcheck.py) and [masked-weight tests](tests/test_masked.py) |
| 2: train a real classifier | [Composable layers](nn/layers.py), [mini-batch loop](train/loop.py), [dense experiment](train/run_part2.py) | [Learning curve](results/part2_curve.png), [raw history](results/part2_curve.json), [initialization study](results/part2_init_ablation.json) |
| 3: self-pruning under a target budget | [Importance criteria](prune/criteria.py), [schedule](prune/schedule.py), [controller](prune/pruner.py) | [Single-run measurements](results/part3_run.json), [curve](results/part3_run.png), [pruning tests](tests/test_prune.py) |
| 4: honest comparison and cost evidence | [Multi-seed sweep](train/run_part4.py), [analysis](train/analysis.py), [sparse inference](prune/sparse.py), [cost benchmark](train/run_cost.py) | [Raw CSV](results/pareto_raw.csv), [paired statistics](results/pareto_summary.json), [Pareto plot](results/pareto.png), [cost table](results/cost.md) |

The task is 10-class classification of 1,797 bundled 8x8 digit images: 1,258 training
and 539 test examples. The MLP is `64 -> 256 -> 128 -> 10`, with ReLU hidden
activations, 50,432 prunable weights, and 394 unpruned biases. The loss is not an
additional output layer. Defaults are 60 epochs, batch size 64, and Adam learning
rate 0.002. Hidden widths are design choices, not dataset requirements.

## Commands
Times are approximate historical observations, not guarantees for your hardware or BLAS configuration.

| what | command | approximate time |
|---|---|---|
| gradient-check suite (Part 1) | `python -m pytest tests/test_gradcheck.py` | seconds |
| masked-weight correctness (Part 1) | `python -m pytest tests/test_masked.py` | seconds |
| all tests (37) | `python -m pytest -q` | seconds |
| Part 2: dense training + learning curve + init study | `python -m train.run_part2` | ~20 s |
| Part 3: one self-pruning run (90%) | `python -m train.run_part3` | ~20 s |
| Part 4: Pareto sweep, baselines, ablations (8 seeds) | `python -m train.run_part4 --seeds 8` | ~25 min |
| Part 4: re-analyse without retraining | `python -m train.run_part4 --analyze-only` | seconds |
| Part 4: cost measurement (run on an idle machine) | `python -m train.run_cost` | ~1 min |
| criterion vs true loss increase | `python -m train.run_criteria_check` | ~1 min |

Seeds are fixed (`seed` controls split, init and batch order). The sweep resumes completed
method/target/seed jobs from its CSV, not partially trained models. The default sweep is
216 training runs. To independently regenerate the sweep without reusing the included CSV:

```powershell
python -m train.run_part4 --seeds 8 --out results_reproduced
```

Choose a new output directory for a fresh sweep; an existing directory may resume its CSV.
Other experiment scripts write to `results` relative to the current working directory
and can overwrite existing outputs. They save plots and measurements, not trained-model
checkpoints. Figures are saved rather than displayed in a GUI.

## Reproducibility and evidence provenance

The full 37-test suite passed during local verification. Dense and Fisher-90% seed-0
training were also rerun and matched the headline accuracies below. The included eight-seed
sweep and cost table are saved experiments; they were not rerun as part of that local check.
For independent verification, run the commands above and compare final metrics and masks,
not training-time fields. Fixed seeds do not guarantee bit-for-bit agreement across all
NumPy/BLAS versions or platforms. Dependencies are lower-bounded, not fully locked.

For timing comparisons, use an idle machine, record Python/NumPy versions, CPU and BLAS
configuration, and report batch size. The cost script requests single-threaded OpenMP/
OpenBLAS defaults when those environment settings are not already set; this does not
guarantee that every possible BLAS backend uses one thread.

## Headline results (details and caveats in DESIGN.md section 6)
* **Part 2:** dense 64-256-128-10 MLP reaches 98.3% test accuracy (seed 0), NaN-free; He init vs
  Xavier vs deliberately bad inits compared in `results/part2_init_ablation.json`.
* **Part 3 (seed 0, 90% target):** achieved sparsity 90.0004%, test accuracy 97.96% vs 98.33% dense
  (single run: 2 more mistakes out of 539 images); masked weights verified to be exact zeros.
* **Falsifiable claim (8 seeds, paired):** at **99% sparsity** (about 504 of 50,432 weights
  kept), saliency pruning retains **95.83% +/- 0.72** test accuracy (98.3% of the dense 97.50%)
  versus **82.31% +/- 10.29** for magnitude pruning at the same per-layer sparsity: paired
  difference **+13.5 pt, 95% CI +/- 8.4, 8 of 8 seeds favour saliency**. Accuracy +/- values
  above are sample standard deviations; paired intervals are per-comparison and not
  corrected for multiple comparisons. At 98%: **+0.83 pt +/- 0.40,
  8/8**. At **<= 95% the two are not distinguishable** (at 95% magnitude is nominally ahead by
  0.28 +/- 0.33 pt, not significant). Reproduce with `python -m train.run_part4 --seeds 8`; numbers in
  `results/pareto_summary.json`.
* **Cost:** 90% / 95% / 98% sparsity cuts MACs per sample 10x / 20x / 50x and storage
  406,608 -> 67,860 / 36,364 / 17,052 bytes. Our NumPy sparse forward is faster than dense BLAS only at
  batch size 1 (1.4-2.2x) and **slower** at batch 32 and 539; the dense-times-zero path is the same
  FLOPs as the unpruned net and is not counted as a saving. Storage means model-array
  bytes including indices and biases, not whole-process memory; MAC counts exclude
  bias/activation/indexing overhead. See [the measured cost table](results/cost.md).

## What the masked-weight tests prove (`tests/test_masked.py`)
Masked weights contribute exactly zero; their parameter gradient is exactly zero while the gradient
w.r.t. the effective weight matches finite differences; Adam never moves a masked weight and keeps its
state at zero; with the tested nonzero gradient, a revived connection's first step has magnitude
approximately `lr` (within tolerance, accounting for epsilon); and three naive alternatives
(updating under the mask, global-step bias correction, stale momentum) are shown to be wrong.

## Scope and limitations

* The connection budget is applied independently per layer, with integer rounding. It is
  not a measured hardware-latency budget.
* Masked training remains dense and includes extra Fisher-statistic work. Lower inference
  MAC counts must not be presented as a training speedup.
* The empirical Fisher is a curvature proxy, not generally the true Hessian or a guaranteed
  prediction of removal cost.
* There is no validation split; test curves are logged each epoch, but final-epoch accuracy
  is reported. Future tuning should use a separate validation set.
* The sparse inference helpers support ReLU hidden activations and copy the trained weights.
  They are not a general activation-preserving converter or a live view of later training.
* There is no model export/loading pipeline, inference API, or implemented multi-tenant
  service. The serving discussion in [DESIGN.md](DESIGN.md) is a proposal.

## Submission and technical-defense checklist

* Push source, tests, [design reasoning](DESIGN.md), raw measurements, and plots to the
  submission GitHub repository; preserve genuine commit history.
* Verify installation, tests, and representative experiments from a clean clone.
* For a private repository, grant the specified reviewers access and share its URL.
* Be ready to derive the masked gradient, explain fresh Adam state and per-entry counters,
  implement and gradient-check a new operation, and defend the measured cost trade-offs.

The supplied challenge excerpt has incomplete README/test/design sublists. Confirm those
details against the original specification before treating this checklist as exhaustive.
