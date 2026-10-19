"""Finite-difference gradient checking."""
import numpy as np

from .tensor import Tensor


def numerical_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Central differences of scalar f at x (f maps ndarray -> float)."""
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        fp = f(x)
        x[i] = old - eps
        fm = f(x)
        x[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def check_grad(fn, *arrays, eps=1e-6, rtol=1e-5, atol=1e-7, seed=0, wrt=None):
    """Compare autodiff vs numerical gradients of fn w.r.t. every array in `arrays`.

    fn takes Tensors and returns a Tensor of any shape; the output is reduced to a scalar
    with a fixed random projection so that every output element is exercised with a
    different weight (catches transposed/permuted gradients that a plain .sum() hides).
    Returns the max abs error; raises AssertionError on mismatch.
    """
    arrays = [np.array(a, dtype=np.float64) for a in arrays]
    wrt = range(len(arrays)) if wrt is None else wrt
    probe = fn(*[Tensor(a) for a in arrays]).data
    proj = np.random.RandomState(seed).randn(*probe.shape) if probe.shape else np.array(1.0)

    def scalar(*arrs):
        return (fn(*[Tensor(a, requires_grad=True) for a in arrs]) * Tensor(proj)).sum()

    ts = [Tensor(a, requires_grad=True) for a in arrays]
    (fn(*ts) * Tensor(proj)).sum().backward()
    worst = 0.0
    for k in wrt:
        def f_k(x, k=k):
            args = list(arrays)
            args[k] = x
            return float(scalar(*args).data)
        num = numerical_grad(f_k, arrays[k].copy(), eps)
        ana = ts[k].grad if ts[k].grad is not None else np.zeros_like(num)
        err = np.abs(ana - num)
        worst = max(worst, float(err.max()) if err.size else 0.0)
        if not np.all(err <= atol + rtol * np.abs(num)):
            raise AssertionError(f"grad mismatch for arg {k}: max err {err.max():.3e}\nana={ana}\nnum={num}")
    return worst
