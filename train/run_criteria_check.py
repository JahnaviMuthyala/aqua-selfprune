"""Empirical check of the derivation: does each score rank-correlate with the *true* loss increase
Delta L when that single weight is set to zero (measured on the full training set)?"""
import json

import numpy as np

from prune import criteria
from train.data import load_digits_split
from train.loop import evaluate, train


def rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(k_per_layer=500, seed=0):
    d = load_digits_split(seed)
    res = {}
    for epochs in (3, 40):
        r = train(d, seed=seed, epochs=epochs, prune=dict(criterion="fisher", sparsity=0.0))  # observe only
        model, pr = r["model"], r["pruner"]
        base = evaluate(model, d["Xtr"], d["ytr"])[0]
        rng = np.random.RandomState(1)
        out = {}
        for li, (layer, st) in enumerate(zip(model.linears(), pr.st)):
            idx = rng.choice(layer.W.data.size, size=min(k_per_layer, layer.W.data.size), replace=False)
            dl = []
            for f in idx:
                i = np.unravel_index(f, layer.W.data.shape)
                old = layer.W.data[i]
                layer.W.data[i] = 0.0
                dl.append(evaluate(model, d["Xtr"], d["ytr"])[0] - base)
                layer.W.data[i] = old
            dl = np.array(dl)
            w = layer.W.data
            sc = {n: criteria.score(n, w, st).ravel()[idx] for n in criteria.CRITERIA}
            sc["neg_g_times_w (signed 1st order)"] = (-st["ema_wg"]).ravel()[idx]
            out[f"layer{li}"] = {n: spearman(v, dl) for n, v in sc.items()}
            out[f"layer{li}"]["mean_true_dL"] = float(dl.mean())
        res[f"epochs={epochs}"] = out
        print(f"epochs={epochs}", json.dumps(out, indent=1))
    json.dump(res, open("results/criteria_correlation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
