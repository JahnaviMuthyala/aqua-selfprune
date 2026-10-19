"""Minimal reverse-mode autodiff on top of NumPy.

Design notes
------------
* Every Tensor wraps a float64 ndarray. float64 keeps finite-difference gradient
  checks tight; the library is tiny so speed is not the constraint.
* A node created by an op stores its parents and a closure `_backward` that reads
  `self.grad` and *accumulates* into each parent via `_accum`.
* `backward()` builds a topological order with an iterative DFS (no recursion
  limit on deep graphs) and walks it in reverse.
* Broadcasting: every binary op un-broadcasts the incoming gradient back to the
  operand's original shape (`_unbroadcast`), so (N, D) + (D,) is correct.
* Graphs are single-use. Intermediate (non-leaf) grads are reset at the start of
  `backward()`, leaf grads accumulate until the optimizer calls zero_grad().
* Intermediate nodes KEEP their .grad after backward(). The pruning code relies on
  this: for Linear, `W_eff.grad` is the *dense* gradient w.r.t. the masked weight.
"""
from __future__ import annotations

import contextlib

import numpy as np

_GRAD_ENABLED = True


@contextlib.contextmanager
def no_grad():
    """Disable graph construction (used for evaluation / inference)."""
    global _GRAD_ENABLED
    prev, _GRAD_ENABLED = _GRAD_ENABLED, False
    try:
        yield
    finally:
        _GRAD_ENABLED = prev


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `grad` down to `shape`, undoing NumPy broadcasting."""
    while grad.ndim > len(shape):  # extra leading dims
        grad = grad.sum(axis=0)
    for ax, s in enumerate(shape):  # dims that were size-1 and got stretched
        if s == 1 and grad.shape[ax] != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad


def _wrap(x) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


def _make(data, parents, op) -> "Tensor":
    rg = _GRAD_ENABLED and any(p.requires_grad for p in parents)
    return Tensor(data, requires_grad=rg, _parents=tuple(parents) if rg else (), _op=op)


class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.mask = None  # optional {0,1} array; the optimizer treats 0-entries as frozen
        self._parents = _parents
        self._op = _op
        self._backward = lambda: None

    # ---- bookkeeping -------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    def _accum(self, g: np.ndarray):
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.array(g, dtype=np.float64)  # copy: never alias another node's grad
        else:
            self.grad += g

    # ---- arithmetic --------------------------------------------------------
    def __add__(self, other):
        other = _wrap(other)
        out = _make(self.data + other.data, (self, other), "add")
        if out.requires_grad:
            def _bw():
                self._accum(_unbroadcast(out.grad, self.shape))
                other._accum(_unbroadcast(out.grad, other.shape))
            out._backward = _bw
        return out

    def __neg__(self):
        out = _make(-self.data, (self,), "neg")
        if out.requires_grad:
            out._backward = lambda: self._accum(-out.grad)
        return out

    def __sub__(self, other):
        other = _wrap(other)
        out = _make(self.data - other.data, (self, other), "sub")
        if out.requires_grad:
            def _bw():
                self._accum(_unbroadcast(out.grad, self.shape))
                other._accum(_unbroadcast(-out.grad, other.shape))
            out._backward = _bw
        return out

    def __mul__(self, other):
        other = _wrap(other)
        out = _make(self.data * other.data, (self, other), "mul")
        if out.requires_grad:
            def _bw():
                self._accum(_unbroadcast(out.grad * other.data, self.shape))
                other._accum(_unbroadcast(out.grad * self.data, other.shape))
            out._backward = _bw
        return out

    def __truediv__(self, other):
        other = _wrap(other)
        out = _make(self.data / other.data, (self, other), "div")
        if out.requires_grad:
            def _bw():
                self._accum(_unbroadcast(out.grad / other.data, self.shape))
                other._accum(_unbroadcast(-out.grad * self.data / other.data ** 2, other.shape))
            out._backward = _bw
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only constant exponents supported"
        out = _make(self.data ** p, (self,), "pow")
        if out.requires_grad:
            out._backward = lambda: self._accum(out.grad * p * self.data ** (p - 1))
        return out

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        return _wrap(other) - self

    def __rtruediv__(self, other):
        return _wrap(other) / self

    def matmul(self, other):
        other = _wrap(other)
        assert self.data.ndim == 2 and other.data.ndim == 2, "matmul supports 2-D only"
        out = _make(self.data @ other.data, (self, other), "matmul")
        if out.requires_grad:
            def _bw():
                self._accum(out.grad @ other.data.T)
                other._accum(self.data.T @ out.grad)
            out._backward = _bw
        return out

    __matmul__ = matmul

    # ---- reductions --------------------------------------------------------
    def sum(self, axis=None, keepdims=False):
        out = _make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")
        if out.requires_grad:
            def _bw():
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self._accum(np.broadcast_to(g, self.shape))
            out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size / np.asarray(self.data.sum(axis=axis, keepdims=keepdims)).size
        return self.sum(axis=axis, keepdims=keepdims) / float(n)

    # ---- non-linearities ---------------------------------------------------
    def relu(self):
        out = _make(np.maximum(self.data, 0.0), (self,), "relu")
        if out.requires_grad:
            out._backward = lambda: self._accum(out.grad * (self.data > 0))  # subgradient 0 at 0
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = _make(t, (self,), "tanh")
        if out.requires_grad:
            out._backward = lambda: self._accum(out.grad * (1.0 - t * t))
        return out

    def sigmoid(self):
        s = 0.5 * (1.0 + np.tanh(0.5 * self.data))  # overflow-free sigmoid
        out = _make(s, (self,), "sigmoid")
        if out.requires_grad:
            out._backward = lambda: self._accum(out.grad * s * (1.0 - s))
        return out

    def gelu(self):
        # tanh approximation; the backward is the exact derivative of *this* formula.
        c, a = np.sqrt(2.0 / np.pi), 0.044715
        x = self.data
        t = np.tanh(c * (x + a * x ** 3))
        out = _make(0.5 * x * (1.0 + t), (self,), "gelu")
        if out.requires_grad:
            def _bw():
                d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * c * (1.0 + 3.0 * a * x * x)
                self._accum(out.grad * d)
            out._backward = _bw
        return out

    def exp(self):
        e = np.exp(self.data)
        out = _make(e, (self,), "exp")
        if out.requires_grad:
            out._backward = lambda: self._accum(out.grad * e)
        return out

    def log(self):
        out = _make(np.log(self.data), (self,), "log")
        if out.requires_grad:
            out._backward = lambda: self._accum(out.grad / self.data)
        return out

    # ---- backward ----------------------------------------------------------
    def backward(self, grad=None):
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        # iterative post-order DFS -> topological order
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        for node in topo:
            if node._parents:  # non-leaf: reset so a node's grad is exactly this pass's
                node.grad = None
        self.grad = np.array(grad, dtype=np.float64)
        for node in reversed(topo):
            node._backward()


def softmax_cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy of softmax(logits) vs integer class targets (fused, stable).

    loss = mean_n [ logsumexp(z_n) - z_{n,y_n} ],  computed with the max-shift trick.
    d loss / d z = (softmax(z) - onehot(y)) / N.
    """
    z = logits.data
    n = z.shape[0]
    shifted = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    logp = shifted - lse
    loss = -logp[np.arange(n), targets].mean()
    out = _make(loss, (logits,), "softmax_ce")
    if out.requires_grad:
        def _bw():
            p = np.exp(logp)
            p[np.arange(n), targets] -= 1.0
            logits._accum(out.grad * p / n)
        out._backward = _bw
    return out


def softmax(z: np.ndarray) -> np.ndarray:
    s = z - z.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)
