"""Self-pruning controller: observe -> (maybe) update masks -> optimizer step."""
import numpy as np

from . import criteria
from .schedule import cubic_sparsity, oneshot_sparsity


class SelfPruner:
    def __init__(self, linears, criterion="fisher", final_sparsity=0.9, t0=100, t_end=700, every=10,
                 schedule="cubic", regrow_frac=0.0, beta=0.9):
        assert criterion in criteria.CRITERIA
        self.layers, self.criterion, self.s_final = list(linears), criterion, final_sparsity
        self.t0, self.t_end, self.every = t0, t_end, every
        self.schedule = cubic_sparsity if schedule == "cubic" else oneshot_sparsity
        self.regrow_frac, self.beta = regrow_frac, beta
        self.st = [dict(ema_g=0.0, ema_wg=0.0, ema_fisher=0.0, n=0) for _ in self.layers]
        self.log = []  # (step, target_sparsity, achieved_sparsity, n_regrown)

    # ---- statistics (cheap: one extra matmul per layer) -------------------------------------
    def observe(self):
        """Call after loss.backward(), before the optimizer step."""
        b = self.beta
        for layer, st in zip(self.layers, self.st):
            g = layer.dense_grad                       # dL/dW_eff, defined at masked entries too
            wg, fisher = layer.W.data * g, layer.empirical_fisher_diag()
            if st["n"] == 0:
                st["ema_g"], st["ema_wg"], st["ema_fisher"] = g.copy(), wg, fisher
            else:
                st["ema_g"] = b * st["ema_g"] + (1 - b) * g
                st["ema_wg"] = b * st["ema_wg"] + (1 - b) * wg
                st["ema_fisher"] = b * st["ema_fisher"] + (1 - b) * fisher
            st["n"] += 1

    # ---- schedule ------------------------------------------------------------------------------
    def target_sparsity(self, step):
        return self.schedule(step, self.t0, self.t_end, self.s_final)

    def due(self, step):
        if step < self.t0:
            return False
        if step >= self.t_end:
            return step == self.t_end            # one last update lands exactly on the target
        return (step - self.t0) % self.every == 0

    def regrow_fraction(self, step):
        """Cosine-decayed to 0 at t_end so the final mask is frozen for fine-tuning."""
        if self.regrow_frac <= 0 or step >= self.t_end:
            return 0.0
        prog = (step - self.t0) / max(1, self.t_end - self.t0)
        return self.regrow_frac * 0.5 * (1 + np.cos(np.pi * prog))

    # ---- mask update -----------------------------------------------------------------------------
    def maybe_update(self, step):
        if not self.due(step):
            return None
        s_t = self.target_sparsity(step)
        regrown_total = 0
        for layer, st in zip(self.layers, self.st):
            size = layer.mask.size
            n_target = int(round((1.0 - s_t) * size))
            old = layer.mask > 0
            # cap regrowth at the number of currently-masked connections that can be revived;
            # otherwise the layer would silently end up *below* its density budget.
            n_regrow = min(int(self.regrow_fraction(step) * n_target), int((~old).sum()))
            n_keep = max(n_target - n_regrow, 0)
            if old.sum() < n_keep:                       # never *add* connections via the schedule
                n_keep, n_regrow = int(old.sum()), 0
            sc = np.where(old, criteria.score(self.criterion, layer.W.data, st), -np.inf).ravel()
            keep = np.zeros(size, dtype=bool)
            if n_keep > 0:
                keep[np.argpartition(sc, size - n_keep)[size - n_keep:]] = True
            new = keep.reshape(layer.mask.shape)
            if n_regrow > 0:
                dropped_now = old & ~new                  # don't resurrect what we just dropped
                rs = np.where(~new & ~dropped_now, criteria.regrow_score(st), -np.inf).ravel()
                grow = np.argpartition(rs, size - n_regrow)[size - n_regrow:]
                ok = np.isfinite(rs[grow])
                flat = new.ravel().copy()
                flat[grow[ok]] = True
                new = flat.reshape(layer.mask.shape)
                regrown_total += int(ok.sum())
            layer.mask[:] = new.astype(float)             # in-place: optimizer shares this array
            layer.apply_mask()                            # pruned weights become exact zeros;
            # revived weights (mask 0->1) are already 0 => function-preserving at the instant of revival.
        self.log.append((step, s_t, self.achieved_sparsity(), regrown_total))
        return self.log[-1]

    # ---- reporting ---------------------------------------------------------------------------------
    def achieved_sparsity(self):
        tot = sum(l.mask.size for l in self.layers)
        return 1.0 - sum(l.n_active for l in self.layers) / tot

    def layer_densities(self):
        return [l.n_active / l.mask.size for l in self.layers]
