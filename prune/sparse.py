"""A genuinely sparse-aware forward path (no dense x zeros) + cost accounting.

SparseLinear stores only surviving weights in CSC order (grouped by output column) and computes
y[:, j] = sum over stored (i, w_ij) of x[:, i] * w_ij  with one gather and one segmented sum.
Work is O(N * nnz) instead of O(N * in * out).
"""
import numpy as np


class SparseLinear:
    def __init__(self, W, mask, b):
        keep = mask > 0
        rows, cols = np.nonzero(keep.T)[1], np.nonzero(keep.T)[0]   # iterate column-major
        self.rows = rows.astype(np.int32)
        self.vals = W[rows, cols].copy()
        self.n_out, self.n_in = W.shape[1], W.shape[0]
        counts = keep.sum(axis=0)
        self.nonempty = np.flatnonzero(counts)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.starts = starts[self.nonempty].astype(np.int64)
        self.b = b.copy()
        self.nnz = int(self.vals.size)

    def __call__(self, x):
        out = np.zeros((x.shape[0], self.n_out))
        if self.nnz:
            prod = x[:, self.rows] * self.vals           # (N, nnz): only surviving connections
            out[:, self.nonempty] = np.add.reduceat(prod, self.starts, axis=1)
        return out + self.b

    def storage_bytes(self):
        return self.vals.nbytes + self.rows.nbytes + self.starts.nbytes + self.nonempty.nbytes + self.b.nbytes


class SparseMLP:
    """Inference-only twin of a trained masked Sequential (ReLU between layers)."""

    def __init__(self, model):
        from nn import Linear
        self.layers = [SparseLinear(l.W.data, l.mask, l.b.data) for l in model.layers if isinstance(l, Linear)]

    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return x

    def macs_per_sample(self):
        return sum(l.nnz for l in self.layers)

    def storage_bytes(self):
        return sum(l.storage_bytes() for l in self.layers)


def dense_macs_per_sample(model):
    from nn import Linear
    return sum(l.mask.size for l in model.layers if isinstance(l, Linear))


def dense_forward_numpy(model, x):
    """Dense path on the SAME pruned weights: x @ W (zeros included). This is the dense-times-zero
    shortcut: identical FLOPs to the unpruned net, so it cannot count as a real saving."""
    from nn import Linear
    lins = [l for l in model.layers if isinstance(l, Linear)]
    for i, l in enumerate(lins):
        x = x @ (l.W.data * l.mask) + l.b.data
        if i < len(lins) - 1:
            x = np.maximum(x, 0.0)
    return x
