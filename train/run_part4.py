"""Part 4: sparsity sweep x method x seeds -> raw CSV, summary, Pareto plot, paired statistics.

Seed s controls the train/test split, the initialisation and the batch order, so the methods
are compared on *paired* runs (identical until the first pruning event)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import csv
import time

from train.analysis import analyze
from train.data import load_digits_split
from train.loop import train

METHODS = {
    "magnitude": dict(criterion="magnitude"),
    "taylor1": dict(criterion="taylor1"),
    "fisher": dict(criterion="fisher"),
    "fisher_regrow": dict(criterion="fisher", regrow_frac=0.1),                 # ablation: allow revival
    "fisher_oneshot": dict(criterion="fisher", schedule="oneshot", t0_frac=0.6, t_end_frac=0.6),  # ablation: no ramp
}
MAIN = ["magnitude", "taylor1", "fisher"]
FIELDS = ["method", "target", "seed", "achieved", "test_acc", "test_loss", "train_acc", "diverged", "seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--levels", type=float, nargs="*", default=[0.5, 0.75, 0.9, 0.95, 0.98, 0.99])
    ap.add_argument("--ablation-levels", type=float, nargs="*", default=[0.9, 0.95, 0.98, 0.99])
    ap.add_argument("--out", default="results")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = f"{a.out}/pareto_raw.csv"
    if not a.analyze_only:
        done = set()
        if os.path.exists(raw):
            done = {(r["method"], float(r["target"]), int(r["seed"])) for r in csv.DictReader(open(raw))}
        new = not os.path.exists(raw)
        f = open(raw, "a", newline="")
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        t0 = time.time()
        for seed in range(a.seeds):
            d = load_digits_split(seed)
            jobs = [("dense", 0.0, None)]
            jobs += [(m, s, METHODS[m]) for s in a.levels for m in MAIN]
            jobs += [(m, s, METHODS[m]) for s in a.ablation_levels for m in METHODS if m not in MAIN]
            for m, s, cfg in jobs:
                if (m, s, seed) in done:
                    continue
                r = train(d, seed=seed, prune=None if cfg is None else {**cfg, "sparsity": s})
                h = r["history"]
                ach = r["pruner"].achieved_sparsity() if r["pruner"] else 0.0
                w.writerow(dict(method=m, target=s, seed=seed, achieved=f"{ach:.6f}", test_acc=f"{h['test_acc'][-1]:.6f}",
                                test_loss=f"{h['test_loss'][-1]:.6f}", train_acc=f"{h['train_acc'][-1]:.6f}",
                                diverged=int(r["diverged"]), seconds=f"{r['seconds']:.1f}"))
                f.flush()
            print(f"seed {seed} done  ({time.time()-t0:.0f}s elapsed)", flush=True)
        f.close()
    summary, paired_stats, dense = analyze(raw, a.out)
    print(open(f"{a.out}/pareto_table.md").read())
    for k, v in paired_stats.items():
        if k.endswith("test_acc") and k.split("@")[0] in ("fisher-magnitude",):
            print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})


if __name__ == "__main__":
    main()
