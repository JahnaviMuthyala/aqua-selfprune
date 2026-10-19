"""Importance criteria. Each returns a score per weight; LOW score => prune first.

Notation: weight w, removing it means the perturbation  dw = -w  (set to zero).
Taylor expansion of the loss around the current weights:

    L(w + dw) - L(w)  ~=  g * dw  +  1/2 * H * dw^2                      (g = dL/dw, H = d2L/dw2)
    removing w (dw = -w):  Delta L(w)  ~=  -g * w  +  1/2 * H * w^2

  * magnitude : |w|.  Equals sqrt(2*DeltaL/H) if H were the same for every weight (it is not).
  * taylor1   : |E[g * w]|.  First-order term only (Molchanov et al. 2017). Accurate far from a
                minimum, but vanishes where training has converged (g -> 0) and is noisy.
  * obd/fisher: 1/2 * F * w^2, with F the per-sample *empirical Fisher* diagonal
                F = E_n[(dl_n/dw)^2]. Near a minimum of a negative-log-likelihood the Fisher
                matches the Hessian (Gauss-Newton), so this is Optimal Brain Damage (LeCun 1990)
                with a cheap, always-positive curvature estimate: it is the second-order term of
                the expansion above, and unlike taylor1 it is still informative at convergence.
"""
import numpy as np

CRITERIA = ("magnitude", "taylor1", "fisher")


def score(name, w, st):
    if name == "magnitude":
        return np.abs(w)
    if name == "taylor1":
        return np.abs(st["ema_wg"])
    if name == "fisher":
        return 0.5 * st["ema_fisher"] * w * w
    raise ValueError(name)


def regrow_score(st):
    """Which *masked* connection would help most if revived (at w=0)? First order: |dL/dw_eff|.
    We use |EMA(g)| so noise averages out and only consistently-signed gradients count."""
    return np.abs(st["ema_g"])
