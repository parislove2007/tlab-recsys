"""How large a difference can appear between two models on the same features?

A paired bootstrap over users measures sampling noise but not training noise.
LightGBM samples features and rows, so two runs on an identical feature set
differ, and that difference sets the floor below which no comparison means
anything. This trains the reference set several times and measures it.
"""
import argparse
import itertools
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from tlab import dataset as D, rank
from tlab.config import RUNS, SCALES
from tlab.human import human_programs
from tlab.metrics import paired_bootstrap
from tlab.pipeline import Bundle

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="main")
ap.add_argument("--seeds", type=int, default=4)
ap.add_argument("--users", type=int, default=0)
a = ap.parse_args()

cached = RUNS / f"ranker_params_{a.scale}.json"
params = json.loads(cached.read_text())["params"] if cached.exists() else {}
users = D.sample_users(a.users or SCALES[a.scale])
b = Bundle(users, verbose=False)
parts = b.matrix(human_programs())
print(f"{len(users)} users, reference feature set, {a.seeds} booster seeds", flush=True)

runs, per_user = [], {}
for sd in range(a.seeds):
    p = dict(params or {})
    # data_random_seed cannot be changed after the Dataset is built, so the
    # run-to-run variation comes from the sampling seeds
    p.update({"seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd})
    model, _, _ = rank.train_rank(parts, params=p)
    test, _, pu = rank.evaluate(model, b, parts, "test")
    runs.append({"seed": sd, "test_ndcg": test["ndcg@10"], "test_auc": test["auc"],
                 "iterations": model.best_iteration})
    per_user[sd] = pu.group_by("user_id").agg(pl.col("ndcg").mean())
    print(f"  seed {sd}: test ndcg@10 = {test['ndcg@10']:.4f}  auc = {test['auc']:.4f}", flush=True)
    del model

vals = np.array([r["test_ndcg"] for r in runs])
deltas = []
for i, j in itertools.combinations(range(a.seeds), 2):
    d = paired_bootstrap(per_user[i], per_user[j])
    d["pair"] = f"{i} vs {j}"
    deltas.append(d)
    print(f"  seeds {i} vs {j}: {d['delta']:+.4f} "
          f"[{d['ci_low']:+.4f}; {d['ci_high']:+.4f}]  доля побед {d['p_win']:.3f}", flush=True)

worst = max(deltas, key=lambda x: abs(x["delta"]))
signif = [d for d in deltas if d["ci_low"] > 0 or d["ci_high"] < 0]
out = {
    "scale": a.scale, "n_users": int(len(users)), "seeds": a.seeds,
    "runs": runs,
    "spread": {"min": float(vals.min()), "max": float(vals.max()),
               "range": float(vals.max() - vals.min()), "std": float(vals.std())},
    "pairwise": deltas,
    "largest_pair_delta": abs(worst["delta"]),
    "n_pairs_called_significant": len(signif),
    "n_pairs": len(deltas),
}
(RUNS / f"noise_floor_{a.scale}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"\nразброс по сидам: {vals.min():.4f}..{vals.max():.4f} "
      f"(размах {vals.max()-vals.min():.4f}, sd {vals.std():.4f})")
print(f"пар, которые бутстрэп назвал значимыми: {len(signif)} из {len(deltas)} "
      f"— при том что набор признаков во всех одинаковый")
