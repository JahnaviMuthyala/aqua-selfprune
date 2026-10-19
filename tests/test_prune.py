"""Part 3 components: schedule, Fisher formula, pruner invariants, regrowth, sparse forward."""
import numpy as np

from engine import Tensor, softmax_cross_entropy
from nn import Adam, make_mlp
from prune import SelfPruner, SparseMLP, cubic_sparsity, dense_forward_numpy
from prune.sparse import SparseLinear


def test_cubic_schedule_shape():
    s = [cubic_sparsity(t, 100, 700, 0.9) for t in range(0, 800)]
    assert s[0] == 0.0 and s[100] == 0.0 and np.isclose(s[700], 0.9) and s[799] == 0.9
    assert np.all(np.diff(s) >= -1e-12)
    assert s[200] - s[100] > s[700] - s[600]       # prunes fastest early, slowest near the target


def test_empirical_fisher_matches_bruteforce_per_sample_gradients():
    rng = np.random.RandomState(0)
    model = make_mlp([6, 5, 3], rng)
    X, y = rng.randn(9, 6), rng.randint(0, 3, 9)
    softmax_cross_entropy(model(Tensor(X)), y).backward()
    fast = [l.empirical_fisher_diag() for l in model.linears()]
    brute = [np.zeros_like(l.W.data) for l in model.linears()]
    for n in range(len(X)):                         # true per-sample gradients, one forward/backward each
        model.zero_grad()
        softmax_cross_entropy(model(Tensor(X[n:n + 1])), y[n:n + 1]).backward()
        for k, l in enumerate(model.linears()):
            brute[k] += l.W.grad ** 2 / len(X)
    for f, b in zip(fast, brute):
        assert np.allclose(f, b, rtol=1e-9, atol=1e-12)


def _toy(seed=0, sizes=(8, 16, 12, 3)):
    rng = np.random.RandomState(seed)
    model = make_mlp(list(sizes), rng)
    X, y = rng.randn(64, sizes[0]), rng.randint(0, sizes[-1], 64)
    return model, X, y, rng


def _run(pruner, model, X, y, steps, rng, record=None):
    opt = Adam(model.parameters(), lr=5e-3)
    for s in range(steps):
        idx = rng.choice(len(X), 16, replace=False)
        opt.zero_grad()
        softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
        pruner.observe()
        pruner.maybe_update(s)
        opt.step()
        if record is not None:
            record.append([l.mask.copy() for l in model.linears()])
        for l in model.linears():                   # honest masking at every single step
            assert np.all(l.W.data[l.mask == 0] == 0.0)
    return opt


def test_pruner_hits_exact_target_per_layer_and_masks_are_monotone_without_regrowth():
    for crit in ("magnitude", "taylor1", "fisher"):
        model, X, y, rng = _toy()
        p = SelfPruner(model.linears(), criterion=crit, final_sparsity=0.75, t0=5, t_end=45, every=5)
        masks = []
        _run(p, model, X, y, 80, rng, masks)
        for l in model.linears():
            assert l.n_active == int(round(0.25 * l.mask.size))
        for a, b in zip(masks[:-1], masks[1:]):     # without regrowth a pruned weight never returns
            for ma, mb in zip(a, b):
                assert np.all(mb <= ma)


def test_pruner_actually_changes_the_function_only_through_the_mask():
    model, X, y, rng = _toy()
    p = SelfPruner(model.linears(), criterion="fisher", final_sparsity=0.5, t0=0, t_end=20, every=5)
    _run(p, model, X, y, 30, rng)
    out = model(Tensor(X)).data
    for l in model.linears():
        l.W.data[l.mask == 0] = 123.0               # corrupt everything under the mask
    assert np.array_equal(model(Tensor(X)).data, out)


def test_regrowth_keeps_sparsity_exact_and_revived_weights_start_from_zero_with_fresh_state():
    model, X, y, rng = _toy()
    p = SelfPruner(model.linears(), criterion="fisher", final_sparsity=0.7, t0=5, t_end=60, every=5, regrow_frac=0.3)
    masks = []
    opt = Adam(model.parameters(), lr=5e-3)
    revived_seen = 0
    prev = [l.mask.copy() for l in model.linears()]
    for s in range(90):
        idx = rng.choice(len(X), 16, replace=False)
        opt.zero_grad()
        softmax_cross_entropy(model(Tensor(X[idx])), y[idx]).backward()
        p.observe()
        ev = p.maybe_update(s)
        if ev is not None:
            for l, pm in zip(model.linears(), prev):
                rev = (l.mask > 0) & (pm == 0)
                revived_seen += int(rev.sum())
                assert np.all(l.W.data[rev] == 0.0)            # function-preserving revival
                assert l.n_active == int(round((1 - ev[1]) * l.mask.size))   # budget exact despite churn
            snap = [l.mask.copy() for l in model.linears()]
        else:
            snap = None
        opt.step()
        if ev is not None:
            for l, pm in zip(model.linears(), prev):
                rev = (l.mask > 0) & (pm == 0)
                st = opt.state[id(l.W)]
                assert np.all(st.t[rev] == 1)                   # fresh per-element step counter
        prev = [l.mask.copy() for l in model.linears()]
    assert revived_seen > 0, "regrowth never triggered: test is vacuous"
    for l in model.linears():                                    # mask is frozen after t_end
        assert l.n_active == int(round(0.3 * l.mask.size))


# ---------------- sparse forward ---------------------------------------------------------------
def test_sparse_forward_equals_dense_masked_forward():
    for density in (1.0, 0.3, 0.05):
        model, X, y, rng = _toy(seed=2)
        for l in model.linears():
            l.mask[:] = (rng.rand(*l.mask.shape) < density).astype(float)
            l.mask[:, 0] = 0.0                                   # force an empty output column
            l.apply_mask()
        sp = SparseMLP(model)
        for n in (1, 7, 64):
            assert np.allclose(sp(X[:n]), dense_forward_numpy(model, X[:n]), atol=1e-12)
            assert np.allclose(sp(X[:n]), model(Tensor(X[:n])).data, atol=1e-12)


def test_sparse_linear_all_zero_layer():
    W, b = np.ones((4, 3)), np.arange(3.0)
    out = SparseLinear(W, np.zeros((4, 3)), b)(np.ones((2, 4)))
    assert np.array_equal(out, np.tile(b, (2, 1)))
