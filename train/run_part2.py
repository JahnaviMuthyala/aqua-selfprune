"""Part 2: dense MLP + Adam learning curve, and the weight-initialisation ablation."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import Tensor, no_grad
from nn import make_mlp
from train.data import load_digits_split
from train.loop import train

OUT = "results"


def activation_stds(init, seed=0):
    """Std of each layer's pre-activation on the first data batch at initialisation."""
    d = load_digits_split(0)
    model = make_mlp([64, 256, 128, 10], np.random.RandomState(seed), init=init)
    x, stds = Tensor(d["Xtr"][:256]), []
    with no_grad():
        for l in model.layers:
            x = l(x)
            if type(l).__name__ == "Linear":
                stds.append(float(x.data.std()))
    return stds


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load_digits_split(0)
    r = train(d, seed=0, verbose=True)
    h = r["history"]
    assert not r["diverged"] and np.all(np.isfinite(h["train_loss"]))
    json.dump(h, open(f"{OUT}/part2_curve.json", "w"), indent=1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].semilogy(h["epoch"], h["train_loss"], label="train loss")
    ax[0].semilogy(h["epoch"], h["test_loss"], label="test loss")
    ax[0].set(xlabel="epoch", ylabel="cross-entropy", title="Dense MLP 64-256-128-10, Adam")
    ax[0].legend()
    ax[1].plot(h["epoch"], h["train_acc"], label="train acc")
    ax[1].plot(h["epoch"], h["test_acc"], label="test acc")
    ax[1].set(xlabel="epoch", ylabel="accuracy", ylim=(0.9, 1.005))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/part2_curve.png", dpi=130)

    abl = {}
    for init in ("he", "xavier", "normal1", "tiny"):
        rr = train(d, seed=0, init=init, epochs=20)
        hh = rr["history"]
        abl[init] = dict(preact_std_per_layer_at_init=activation_stds(init), diverged=rr["diverged"],
                         nan_free=bool(np.all(np.isfinite(hh["train_loss"]))),
                         loss_after_epoch1=hh["train_loss"][0] if hh["train_loss"] else None,
                         final_test_acc=hh["test_acc"][-1] if hh["test_acc"] else None)
        print(init, abl[init])
    json.dump(abl, open(f"{OUT}/part2_init_ablation.json", "w"), indent=1)
    print(f"\nfinal dense: train_acc={h['train_acc'][-1]:.4f} test_acc={h['test_acc'][-1]:.4f} NaN-free=True")


if __name__ == "__main__":
    main()
