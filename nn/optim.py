"""Optimizers with *mask-aware* state.

Failure modes a naive implementation has when connections are pruned and revived:
  1. Stale moments:   m, v of a pruned weight keep decaying but are non-zero; on revival the
                      weight is kicked by a gradient history from a different regime.
  2. Hidden drift:    the optimizer keeps updating the raw weight under the mask (from a
                      straight-through gradient); on revival the weight jumps to a drifted value.
  3. Bias correction: Adam's 1/(1-beta^t) uses a global step t. A revived weight has fresh
                      moments (m=v=0) but a large global t, so the correction is ~1 and the first step is
                      (1-b1)g / sqrt((1-b2)g^2) = 3.16 x lr once t >> 1/(1-b2) = 1000
                      (and <1 x lr for small t) instead of the correct 1 x lr.
Our rule: whenever an entry's mask flips, its (m, v, t) are reset to zero, a pruned weight is
set to an exact 0, and bias correction uses a per-element step counter.
"""
import numpy as np


class _State:
    def __init__(self, shape):
        self.m = np.zeros(shape)
        self.v = np.zeros(shape)
        self.t = np.zeros(shape)
        self.active = np.ones(shape, dtype=bool)


def _sync_mask(p, st):
    """Detect prune/revive events since the last step; reset optimizer state accordingly."""
    if p.mask is None:
        return None
    mask = p.mask > 0
    changed = mask != st.active
    if changed.any():
        for arr in (st.m, st.v, st.t):
            arr[changed] = 0.0          # fresh state for both newly pruned and newly revived entries
        st.active = mask.copy()
    p.data[~mask] = 0.0                 # a pruned weight is a *real* zero
    return mask


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, per_element_step=True):
        self.params, self.lr, self.b1, self.b2, self.eps = list(params), lr, betas[0], betas[1], eps
        self.per_element_step = per_element_step  # False reproduces bug #3 (used in a test)
        self.state = {id(p): _State(p.data.shape) for p in self.params}
        self.global_t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.global_t += 1
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[id(p)]
            mask = _sync_mask(p, st)
            g = p.grad if mask is None else p.grad * mask
            if mask is None:
                st.t += 1
            else:
                st.t[mask] += 1
            t = st.t if self.per_element_step else self.global_t
            st.m = self.b1 * st.m + (1 - self.b1) * g
            st.v = self.b2 * st.v + (1 - self.b2) * g * g
            if mask is not None:
                st.m *= mask
                st.v *= mask
            # clip t at 1 only guards the 0/0 on frozen entries; their update is masked out below
            tt = np.maximum(t, 1.0) if self.per_element_step else t
            mhat = st.m / (1 - self.b1 ** tt)
            vhat = st.v / (1 - self.b2 ** tt)
            upd = self.lr * mhat / (np.sqrt(vhat) + self.eps)
            if mask is not None:
                upd *= mask
            p.data -= upd


class SGD:
    """SGD + momentum with the same mask-aware state handling."""

    def __init__(self, params, lr=1e-2, momentum=0.9):
        self.params, self.lr, self.mu = list(params), lr, momentum
        self.state = {id(p): _State(p.data.shape) for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[id(p)]
            mask = _sync_mask(p, st)
            g = p.grad if mask is None else p.grad * mask
            st.m = self.mu * st.m + g
            if mask is not None:
                st.m *= mask
            p.data -= self.lr * st.m
