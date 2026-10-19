"""Mini-batch training loop (own code) with optional self-pruning."""
import time

import numpy as np

from engine import Tensor, no_grad, softmax_cross_entropy
from nn import Adam, ReLU, make_mlp
from prune import SelfPruner

DEFAULTS = dict(hidden=(256, 128), epochs=60, batch=64, lr=2e-3, init="he")


def evaluate(model, X, y):
    with no_grad():
        logits = model(Tensor(X))
        loss = float(softmax_cross_entropy(logits, y).data)
    return loss, float((logits.data.argmax(1) == y).mean())


def train(data, seed=0, prune=None, act=ReLU, verbose=False, **overrides):
    """prune: None (dense) or dict(criterion, sparsity, schedule='cubic', regrow_frac=0.0,
    every=10, t0_frac=0.1, t_end_frac=0.6). Returns dict(model, pruner, history, ...)."""
    cfg = {**DEFAULTS, **overrides}
    rng = np.random.RandomState(seed)
    Xtr, ytr = data["Xtr"], data["ytr"]
    n = len(Xtr)
    model = make_mlp([Xtr.shape[1], *cfg["hidden"], 10], rng, act=act, init=cfg["init"])
    opt = Adam(model.parameters(), lr=cfg["lr"])
    steps_per_epoch = int(np.ceil(n / cfg["batch"]))
    total = steps_per_epoch * cfg["epochs"]
    pruner = None
    if prune:
        p = dict(prune)
        pruner = SelfPruner(model.linears(), criterion=p["criterion"], final_sparsity=p["sparsity"],
                            t0=int(p.get("t0_frac", 0.1) * total), t_end=int(p.get("t_end_frac", 0.6) * total),
                            every=p.get("every", 10), schedule=p.get("schedule", "cubic"),
                            regrow_frac=p.get("regrow_frac", 0.0))
    hist = dict(epoch=[], train_loss=[], train_acc=[], test_loss=[], test_acc=[], sparsity=[])
    diverged, step, t_start = False, 0, time.time()
    for ep in range(cfg["epochs"]):
        perm, run_loss = rng.permutation(n), 0.0
        for i in range(0, n, cfg["batch"]):
            idx = perm[i:i + cfg["batch"]]
            opt.zero_grad()
            loss = softmax_cross_entropy(model(Tensor(Xtr[idx])), ytr[idx])
            if not np.isfinite(loss.data):
                diverged = True
                break
            loss.backward()
            if pruner:
                pruner.observe()
                pruner.maybe_update(step)
            opt.step()
            run_loss += float(loss.data) * len(idx)
            step += 1
        if diverged:
            break
        trl, tra = evaluate(model, Xtr, ytr)
        tel, tea = evaluate(model, data["Xte"], data["yte"])
        for k, v in zip(hist, (ep + 1, run_loss / n, tra, tel, tea, pruner.achieved_sparsity() if pruner else 0.0)):
            hist[k].append(v)
        if verbose and (ep % 5 == 4 or ep == 0):
            print(f"ep {ep+1:3d} loss {run_loss/n:.4f} train_acc {tra:.4f} test_acc {tea:.4f} sparsity {hist['sparsity'][-1]:.3f}")
    # invariants for honest masking
    for l in model.linears():
        assert np.all(l.W.data[l.mask == 0] == 0.0), "masked weight is not an exact zero"
    return dict(model=model, pruner=pruner, history=hist, diverged=diverged, cfg=cfg,
                seconds=time.time() - t_start, total_steps=total)
