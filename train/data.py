"""Data loading. scikit-learn is used ONLY to load the bundled 8x8 digits dataset (no training,
no models, no gradients). The split and standardisation are our own NumPy code."""
import numpy as np


def load_digits_split(split_seed=0, test_frac=0.3):
    from sklearn.datasets import load_digits  # dataset loader only

    d = load_digits()
    X, y = d.data.astype(np.float64) / 16.0, d.target.astype(np.int64)
    perm = np.random.RandomState(split_seed).permutation(len(X))
    n_test = int(round(test_frac * len(X)))
    te, tr = perm[:n_test], perm[n_test:]
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)  # constant pixels stay 0 after centring
    return dict(Xtr=(X[tr] - mu) / sd, ytr=y[tr], Xte=(X[te] - mu) / sd, yte=y[te])
