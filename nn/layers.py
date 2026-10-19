"""Composable layers. Linear carries a {0,1} connection mask (all ones = dense)."""
import numpy as np

from engine import Tensor


class Module:
    def parameters(self):
        return []

    def __call__(self, x):
        return self.forward(x)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class GELU(Module):
    def forward(self, x):
        return x.gelu()


INITS = ("he", "xavier", "normal1", "tiny")


def init_weight(fan_in, fan_out, scheme, rng):
    """He:    N(0, 2/fan_in)         -- preserves E[h^2] through ReLU (half the units are zeroed).
    Xavier: N(0, 2/(fan_in+fan_out)) -- preserves variance for tanh-like / linear layers.
    normal1 / tiny are deliberately bad baselines used only in the init ablation."""
    if scheme == "he":
        std = np.sqrt(2.0 / fan_in)
    elif scheme == "xavier":
        std = np.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "normal1":
        std = 1.0
    elif scheme == "tiny":
        std = 0.01
    else:
        raise ValueError(scheme)
    return rng.randn(fan_in, fan_out) * std


class Linear(Module):
    """y = x @ (W * M) + b.

    The mask is applied *inside the graph* as an ordinary multiplication by a constant, so
        W.grad = dL/dW_eff * M      (exactly 0 where M == 0: the true derivative of L(W*M))
    and, because intermediates keep their grads, `dense_grad` (= dL/dW_eff, nonzero at masked
    positions) is available for regrowth / importance without any special-case op.
    """

    def __init__(self, fan_in, fan_out, rng, init="he"):
        self.W = Tensor(init_weight(fan_in, fan_out, init, rng), requires_grad=True)
        self.b = Tensor(np.zeros(fan_out), requires_grad=True)
        self.mask = np.ones((fan_in, fan_out))
        self.W.mask = self.mask  # same object: in-place edits by the pruner are seen by the optimizer
        self._x = self._weff = self._y = None

    def parameters(self):
        return [self.W, self.b]

    def forward(self, x):
        w_eff = self.W * Tensor(self.mask)
        y = x @ w_eff + self.b
        self._x, self._weff, self._y = x.data, w_eff, y
        return y

    # ---- quantities captured by the last forward/backward ----------------------------
    @property
    def dense_grad(self):
        """dL/dW_eff: the gradient w.r.t. the *effective* weight, including masked entries."""
        return self._weff.grad

    def empirical_fisher_diag(self):
        """Per-sample empirical Fisher diagonal  F_ij = (1/N) sum_n (dl_n/dW_ij)^2.

        With y = xW + b and per-sample loss l_n: dl_n/dW_ij = x_ni * d_nj, d_n = dl_n/dy_n.
        The batch loss is a mean, so y.grad = d/N  =>  F = N * (x^2)^T (y.grad^2). One matmul.
        """
        n = self._x.shape[0]
        return n * (self._x ** 2).T @ (self._y.grad ** 2)

    def apply_mask(self):
        """Make masked weights *real* zeros (not just hidden behind the mask)."""
        self.W.data *= self.mask

    @property
    def n_active(self):
        return int(self.mask.sum())


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]

    def linears(self):
        return [l for l in self.layers if isinstance(l, Linear)]


def make_mlp(sizes, rng, act=ReLU, init="he"):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        layers.append(Linear(a, b, rng, init="xavier" if (last and init == "he") else init))
        if not last:
            layers.append(act())
    return Sequential(*layers)
