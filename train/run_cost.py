"""Part 4 (cost): does pruning actually reduce work?  Measures, for trained pruned models:
  * active parameters, MACs/sample (dense-times-zero vs sparse-aware), storage bytes
  * wall-clock of (a) dense path on the pruned weights  [dense x zeros: NOT a real saving]
                  (b) our sparse-aware path (CSC gather + segmented sum, written in NumPy)
  * equivalence of the two outputs
Wall-clock is reported as measured, including when the sparse NumPy path is *slower* than BLAS."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import json
import timeit

import numpy as np

from prune import SparseMLP, dense_forward_numpy, dense_macs_per_sample
from train.data import load_digits_split
from train.loop import train


def best_time(fn, x, reps=200, rounds=7):
    fn(x)
    return min(timeit.repeat(lambda: fn(x), number=reps, repeat=rounds)) / reps


def dead_hidden_units(model):
    lins = model.linears()
    dead = 0
    for k in range(len(lins) - 1):
        fan_in = lins[k].mask.sum(axis=0)
        fan_out = lins[k + 1].mask.sum(axis=1)
        dead += int(((fan_in == 0) | (fan_out == 0)).sum())
    return dead


def main():
    d = load_digits_split(0)
    out = {}
    for name, cfg in [("fisher_90", dict(criterion="fisher", sparsity=0.90)),
                      ("fisher_95", dict(criterion="fisher", sparsity=0.95)),
                      ("fisher_98", dict(criterion="fisher", sparsity=0.98)),
                      ("magnitude_90", dict(criterion="magnitude", sparsity=0.90))]:
        r = train(d, seed=0, prune=cfg)
        model, sp = r["model"], None
        sp = SparseMLP(model)
        lins = model.linears()
        n_w = sum(l.mask.size for l in lins)
        n_b = sum(l.b.data.size for l in lins)
        nnz = sum(l.n_active for l in lins)
        X = d["Xte"]
        maxdiff = float(np.abs(sp(X) - dense_forward_numpy(model, X)).max())
        row = dict(test_acc=r["history"]["test_acc"][-1], total_weights=n_w, active_weights=nnz,
                   achieved_sparsity=1 - nnz / n_w, active_params_incl_bias=nnz + n_b, total_params_incl_bias=n_w + n_b,
                   macs_dense=dense_macs_per_sample(model), macs_sparse=sp.macs_per_sample(),
                   mac_reduction=dense_macs_per_sample(model) / sp.macs_per_sample(),
                   bytes_dense_float64=8 * (n_w + n_b), bytes_csc_float64_int32=sp.storage_bytes(),
                   dead_hidden_units=dead_hidden_units(model), max_abs_diff_sparse_vs_dense=maxdiff, timing_us={})
        for bs in (1, 32, 539):
            xb = X[:bs]
            td = best_time(lambda x: dense_forward_numpy(model, x), xb)
            ts = best_time(sp, xb)
            row["timing_us"][bs] = dict(dense_times_zero=td * 1e6, sparse_aware=ts * 1e6, sparse_speedup=td / ts)
        out[name] = row
        print(name, json.dumps(row, indent=1, default=float))
    json.dump(out, open("results/cost.json", "w"), indent=1, default=float)
    L = ["| model | active weights | MACs/sample dense -> sparse | reduction | bytes dense -> CSC | wall-clock speedup (bs=1 / 32 / 539) |", "|---|---|---|---|---|---|"]
    for k, v in out.items():
        sp_ = " / ".join(f"{v['timing_us'][b]['sparse_speedup']:.2f}x" for b in (1, 32, 539))
        L.append(f"| {k} | {v['active_weights']}/{v['total_weights']} | {v['macs_dense']} -> {v['macs_sparse']} | {v['mac_reduction']:.1f}x | "
                 f"{v['bytes_dense_float64']} -> {v['bytes_csc_float64_int32']} | {sp_} |")
    open("results/cost.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
