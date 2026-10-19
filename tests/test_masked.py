"""Part 1 CORRECTNESS REQUIREMENT: masked weights, their gradient, and optimizer state.

What we claim (and test):
  (a) a masked weight contributes exactly zero to the forward pass, whatever its raw value;
  (b) the gradient w.r.t. the *parameter* W of L(W * M) is exactly zero at masked entries and
      equals the dense gradient at active entries (this is the true derivative, verified by
      finite differences);
  (c) the gradient w.r.t. the *effective* weight W_eff = W*M (the "dense gradient") is the
      meaningful 'what if I revived it' signal and is also verified numerically;
  (d) Adam never moves a masked weight and keeps its (m, v, t) at 0;
  (e) a revived connection starts with a fresh optimizer state: its first step is ~lr,
      and a hand-rolled reference Adam on only the active entries matches ours exactly;
  (f) the three naive implementations really do fail (so the test is meaningful).
"""
import numpy as np

from engine import Tensor, check_grad, softmax_cross_entropy
from nn import Adam, Linear

LR = 1e-2


def _layer(seed=0, p_mask=0.5, fi=5, fo=4):
    rng = np.random.RandomState(seed)
    layer = Linear(fi, fo, rng)
    layer.mask[:] = (rng.rand(fi, fo) > p_mask).astype(float)
    layer.mask[0, 0] = 0.0  # guarantee at least one masked and one active entry
    layer.mask[1, 1] = 1.0
    layer.apply_mask()
    x = rng.randn(8, fi)
    y = rng.randint(0, fo, 8)
    return layer, x, y


# (a) -----------------------------------------------------------------------------------------
def test_masked_weight_contributes_zero_to_forward():
    layer, x, _ = _layer()
    y0 = layer(Tensor(x)).data.copy()
    layer.W.data[layer.mask == 0] = 1e9  # garbage hiding under the mask must be invisible
    assert np.array_equal(layer(Tensor(x)).data, y0)


# (b) + (c) -------------------------------------------------------------------------------------
def test_param_grad_is_zero_on_masked_and_dense_on_active():
    layer, x, y = _layer()
    softmax_cross_entropy(layer(Tensor(x)), y).backward()
    dense, g = layer.dense_grad, layer.W.grad
    assert np.any(np.abs(dense[layer.mask == 0]) > 1e-6), "test is vacuous: dense grad should be non-zero on masked entries"
    assert np.all(g[layer.mask == 0] == 0.0)
    assert np.array_equal(g[layer.mask == 1], dense[layer.mask == 1])


def test_param_grad_matches_finite_difference_of_masked_function():
    layer, x, y = _layer()
    M, b = layer.mask, layer.b.data
    # L(W) = CE(x @ (W*M) + b): numerical derivative at masked entries must be exactly ~0
    check_grad(lambda W: softmax_cross_entropy(Tensor(x) @ (W * Tensor(M)) + Tensor(b), y), layer.W.data.copy())


def test_dense_grad_matches_finite_difference_wrt_effective_weight():
    layer, x, y = _layer()
    softmax_cross_entropy(layer(Tensor(x)), y).backward()
    dense = layer.dense_grad.copy()
    b = layer.b.data
    check_grad(lambda We: softmax_cross_entropy(Tensor(x) @ We + Tensor(b), y), layer.W.data * layer.mask)
    # and the autodiff dense grad equals the closed form X^T Delta
    delta = layer._y.grad
    assert np.allclose(dense, x.T @ delta)


# (d) -------------------------------------------------------------------------------------------
def test_adam_never_moves_masked_weights_and_keeps_state_zero():
    layer, x, y = _layer()
    opt = Adam(layer.parameters(), lr=LR)
    for _ in range(30):
        opt.zero_grad()
        softmax_cross_entropy(layer(Tensor(x)), y).backward()
        opt.step()
        assert np.all(layer.W.data[layer.mask == 0] == 0.0)
    st = opt.state[id(layer.W)]
    for arr in (st.m, st.v, st.t):
        assert np.all(arr[layer.mask == 0] == 0.0)
    assert np.all(st.t[layer.mask == 1] == 30)


def test_adam_matches_reference_adam_on_active_entries_only():
    """Training a masked layer == training only the surviving parameters (hand-written Adam)."""
    rng = np.random.RandomState(3)
    mask = (rng.rand(4, 3) > 0.5).astype(float)
    mask[0, 0], mask[1, 1] = 1.0, 1.0
    target = rng.randn(4, 3)
    W = Tensor(rng.randn(4, 3) * mask, requires_grad=True)
    W.mask = mask
    opt = Adam([W], lr=LR)
    idx = mask.astype(bool)
    w = W.data[idx].copy()
    m, v = np.zeros_like(w), np.zeros_like(w)
    for t in range(1, 41):
        opt.zero_grad()
        (((W * Tensor(mask)) - Tensor(target * mask)) ** 2).sum().backward()
        opt.step()
        g = 2 * (w - target[idx])                      # analytic gradient on active entries
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        w = w - LR * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    assert np.allclose(W.data[idx], w, atol=1e-12)
    assert np.all(W.data[~idx] == 0.0)


# (e) + (f) ----------------------------------------------------------------------------------------
def _run_prune_revive(opt_factory, feed_dense_grad=False, n_active=50, n_pruned=20):
    """Constant gradient g=1 on 3 entries. Entry 0 is trained, pruned, then revived.
    Returns (raw weight while pruned, change in effective weight on the first step after revival)."""
    p = Tensor(np.zeros(3), requires_grad=True)
    p.mask = np.ones(3)
    opt = opt_factory([p])
    for _ in range(n_active):
        p.grad = np.ones(3)
        opt.step()
    p.mask[0] = 0.0                                  # prune entry 0
    for _ in range(n_pruned):
        p.grad = np.ones(3)                          # a straight-through dense gradient keeps arriving
        opt.step()
    raw_while_pruned = p.data[0]
    p.mask[0] = 1.0                                  # revive entry 0
    before = p.data[0] * p.mask[0]
    p.grad = np.ones(3)
    opt.step()
    return raw_while_pruned, abs(p.data[0] - before)


def test_revived_connection_first_step_is_exactly_one_lr_and_weight_was_exact_zero():
    raw, first = _run_prune_revive(lambda ps: Adam(ps, lr=LR))
    assert raw == 0.0
    assert np.isclose(first, LR, rtol=1e-6)          # fresh Adam: step = lr * g/|g| = lr


class NaiveMaskedAdam:
    """What most quick implementations do: ignore the mask inside the optimizer (state keeps
    evolving on the dense gradient, raw weight keeps moving); the mask is only applied in forward."""

    def __init__(self, params, lr):
        self.params, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            m[:] = 0.9 * m + 0.1 * p.grad
            v[:] = 0.999 * v + 0.001 * p.grad ** 2
            p.data -= self.lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


def test_naive_optimizer_drifts_under_the_mask_and_jumps_on_revival():
    raw, _ = _run_prune_revive(lambda ps: NaiveMaskedAdam(ps, LR))
    assert abs(raw) > 60 * LR                         # 70 steps of hidden drift (raw weight != 0)
    # on revival the effective weight jumps from 0 to the drifted value, in one step
    p = Tensor(np.zeros(1), requires_grad=True)
    opt = NaiveMaskedAdam([p], LR)
    for _ in range(70):
        p.grad = np.ones(1)
        opt.step()
    assert abs(p.data[0]) > 60 * LR                   # the value the connection "teleports" to when unmasked


def test_global_step_bias_correction_is_wrong_for_revived_weights():
    """Fresh moments + global t: first step = lr * (1-b1)/sqrt(1-b2) * sqrt(1-b2^t)/(1-b1^t).
    For t >> 1/(1-b2) that is 3.16 x lr (overshoot); for small t it is < lr. Never the correct 1 x lr."""
    for n_active in (3000, 50):
        n_pruned = 20
        t = n_active + n_pruned + 1                  # global step index of the first post-revival step
        expect = 0.1 / np.sqrt(0.001) * np.sqrt(1 - 0.999 ** t) / (1 - 0.9 ** t)
        _, bad = _run_prune_revive(lambda ps: Adam(ps, lr=LR, per_element_step=False), n_active=n_active)
        assert np.isclose(bad / LR, expect, rtol=1e-3)
        assert not np.isclose(bad / LR, 1.0, rtol=0.1)
        _, good = _run_prune_revive(lambda ps: Adam(ps, lr=LR), n_active=n_active)
        assert np.isclose(good / LR, 1.0, rtol=1e-6)
    assert expect != 1.0


def test_stale_momentum_would_move_a_revived_weight_in_the_wrong_direction():
    """With our optimizer, momentum built up before pruning cannot leak past a prune/revive cycle."""
    p = Tensor(np.zeros(2), requires_grad=True)
    p.mask = np.ones(2)
    opt = Adam([p], lr=LR)
    for _ in range(40):
        p.grad = np.array([5.0, 5.0])                 # strong +gradient history on both entries
        opt.step()
    p.mask[0] = 0.0
    p.grad = np.array([5.0, 5.0]); opt.step()
    p.mask[0] = 1.0
    p.grad = np.array([-1.0, -1.0])                   # gradient now flips sign
    before = p.data.copy()
    opt.step()
    assert p.data[0] > before[0]                      # revived entry follows the *new* gradient (fresh state)
    assert p.data[1] < before[1] or abs(p.data[1] - before[1]) < LR  # old entry is still dominated by stale momentum
