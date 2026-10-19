"""Part 3: one self-pruning run (default: 90% sparsity, Fisher/OBD saliency, cubic ramp)."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from train.data import load_digits_split
from train.loop import train

OUT = "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparsity", type=float, default=0.9)
    ap.add_argument("--criterion", default="fisher", choices=["magnitude", "taylor1", "fisher"])
    ap.add_argument("--regrow", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    d = load_digits_split(a.seed)
    dense = train(d, seed=a.seed)
    r = train(d, seed=a.seed, prune=dict(criterion=a.criterion, sparsity=a.sparsity, regrow_frac=a.regrow), verbose=True)
    h, p = r["history"], r["pruner"]
    summary = dict(criterion=a.criterion, target_sparsity=a.sparsity, achieved_sparsity=p.achieved_sparsity(),
                   layer_densities=p.layer_densities(), dense_test_acc=dense["history"]["test_acc"][-1],
                   pruned_test_acc=h["test_acc"][-1], accuracy_cost=dense["history"]["test_acc"][-1] - h["test_acc"][-1],
                   nan_free=bool(np.all(np.isfinite(h["train_loss"]))), prune_events=len(p.log),
                   masked_weights_exact_zero=bool(all(np.all(l.W.data[l.mask == 0] == 0) for l in r["model"].linears())))
    print(json.dumps(summary, indent=1))
    json.dump(dict(summary=summary, history=h, prune_log=p.log), open(f"{OUT}/part3_run.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(h["epoch"], h["sparsity"]); ax[0].set(xlabel="epoch", ylabel="sparsity", title="cubic sparsity ramp")
    ax[1].semilogy(h["epoch"], h["train_loss"], label="pruned train")
    ax[1].semilogy(dense["history"]["epoch"], dense["history"]["train_loss"], "--", label="dense train")
    ax[1].set(xlabel="epoch", ylabel="loss", title="loss (bumps = pruning events)"); ax[1].legend()
    ax[2].plot(h["epoch"], h["test_acc"], label=f"{a.criterion} {a.sparsity:.0%}")
    ax[2].plot(dense["history"]["epoch"], dense["history"]["test_acc"], "--", label="dense")
    ax[2].set(xlabel="epoch", ylabel="test accuracy", ylim=(0.9, 1.0), title="accuracy during self-pruning"); ax[2].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/part3_run.png", dpi=130)


if __name__ == "__main__":
    main()
