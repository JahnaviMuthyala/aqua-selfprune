"""Aggregate raw sweep rows -> summary table, paired statistics, Pareto plot, falsifiable claim."""
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
       11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045}  # two-sided 95% t


def tcrit(df):
    return T95[max(k for k in T95 if k <= df)] if df >= 1 else float("nan")


def read_rows(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ("target", "seed", "achieved", "test_acc", "test_loss", "train_acc"):
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
    return rows


def paired(rows, a, b, target, key="test_acc"):
    """Per-seed paired difference method a minus method b (same seed => same split, init, batches)."""
    da = {r["seed"]: r[key] for r in rows if r["method"] == a and abs(r["target"] - target) < 1e-9}
    db = {r["seed"]: r[key] for r in rows if r["method"] == b and abs(r["target"] - target) < 1e-9}
    seeds = sorted(set(da) & set(db))
    diff = np.array([da[s] - db[s] for s in seeds])
    n = len(diff)
    if n < 2:
        return None
    sd = diff.std(ddof=1)
    se = sd / np.sqrt(n)
    ci = tcrit(n - 1) * se
    return dict(n=n, mean_diff=float(diff.mean()), sd=float(sd), ci95=float(ci), t=float(diff.mean() / se) if se > 0 else float("inf"),
                wins=int((diff > 0).sum()), ties=int((diff == 0).sum()), losses=int((diff < 0).sum()),
                significant=bool(abs(diff.mean()) > ci))


def analyze(raw_csv, out_dir):
    rows = read_rows(raw_csv)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["target"])].append(r)
    dense = [r["test_acc"] for r in rows if r["method"] == "dense"]
    dense_mean = float(np.mean(dense))
    summary = {}
    for (m, t), rs in sorted(groups.items()):
        acc = np.array([r["test_acc"] for r in rs]); loss = np.array([r["test_loss"] for r in rs])
        summary[f"{m}@{t}"] = dict(method=m, target=t, n=len(rs), achieved=float(np.mean([r["achieved"] for r in rs])),
                                   acc_mean=float(acc.mean()), acc_std=float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                                   loss_mean=float(loss.mean()), retention_vs_dense=float(acc.mean() / dense_mean))
    paired_stats = {}
    for t in sorted({r["target"] for r in rows if r["method"] != "dense"}):
        for a, b in (("fisher", "magnitude"), ("taylor1", "magnitude"), ("fisher", "taylor1"),
                     ("fisher_regrow", "fisher"), ("fisher_oneshot", "fisher")):
            for key in ("test_acc", "test_loss"):
                s = paired(rows, a, b, t, key)
                if s:
                    paired_stats[f"{a}-{b}@{t}:{key}"] = s
    json.dump(dict(dense_acc_mean=dense_mean, dense_acc_std=float(np.std(dense, ddof=1)), n_seeds_dense=len(dense),
                   summary=summary, paired=paired_stats), open(f"{out_dir}/pareto_summary.json", "w"), indent=1)

    # ---- markdown table ----
    methods = ["magnitude", "taylor1", "fisher", "fisher_regrow", "fisher_oneshot"]
    targets = sorted({r["target"] for r in rows if r["method"] != "dense"})
    lines = ["| target sparsity | " + " | ".join(methods) + " |", "|---|" + "---|" * len(methods)]
    lines.insert(0, f"Dense baseline: {100*dense_mean:.2f}% +/- {100*np.std(dense, ddof=1):.2f} (n={len(dense)})\n")
    for t in targets:
        cells = []
        for m in methods:
            s = summary.get(f"{m}@{t}")
            cells.append(f"{100*s['acc_mean']:.2f} +/- {100*s['acc_std']:.2f}" if s else "-")
        lines.append(f"| {t:.0%} | " + " | ".join(cells) + " |")
    open(f"{out_dir}/pareto_table.md", "w").write("\n".join(lines) + "\n")

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    sty = {"magnitude": ("tab:gray", "o"), "taylor1": ("tab:orange", "s"), "fisher": ("tab:blue", "^"),
           "fisher_regrow": ("tab:green", "v"), "fisher_oneshot": ("tab:red", "x")}
    for axi, zoom in zip(ax, (False, True)):
        for m in methods:
            pts = [summary[f"{m}@{t}"] for t in targets if f"{m}@{t}" in summary]
            if not pts:
                continue
            x = [100 * p["achieved"] for p in pts]
            y = [100 * p["acc_mean"] for p in pts]
            e = [100 * p["acc_std"] for p in pts]
            axi.errorbar(x, y, yerr=e, label=m, color=sty[m][0], marker=sty[m][1], capsize=3, lw=1.3,
                         ls="--" if "oneshot" in m or "regrow" in m else "-")
        axi.axhline(100 * dense_mean, color="k", ls=":", lw=1, label="dense")
        axi.set(xlabel="achieved sparsity (% of weights removed)", ylabel="test accuracy (%)")
        axi.grid(alpha=.3)
        if zoom:
            axi.set_xlim(88, 100); axi.set_title("zoom: high sparsity (mean +/- std across seeds)")
        else:
            axi.set_title("Sparsity-accuracy Pareto (mean +/- std across seeds)")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pareto.png", dpi=130)
    return summary, paired_stats, dense_mean
