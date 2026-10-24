"""Reference points and the hand-written baseline, at the reporting scale."""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from tlab import dataset as D, rank
from tlab.config import RUNS, SCALES
from tlab.human import human_programs
from tlab.metrics import ranking_metrics
from tlab.pipeline import Bundle

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="main")
ap.add_argument("--users", type=int, default=0, help="override the scale, for smoke tests")
a = ap.parse_args()

users = D.sample_users(a.users or SCALES[a.scale])
print(f"scale={a.scale}: {len(users)} users", flush=True)
b = Bundle(users)

out = {"scale": a.scale, "n_users": int(len(users)), "heuristics": {}, "models": {}}

# no-learning reference points: rank by a single candidate-generator signal
for label, col, desc in [("rrf", "rrf", "rank fusion of the four generators"),
                         ("als", "als_score", "ALS score alone"),
                         ("popularity", "rank_pop", "14-day popularity alone")]:
    rows = []
    rng = np.random.default_rng(0)
    for T in b.dates:
        cand = b.folds[T]["cand"]
        s = cand[col].cast(pl.Float64).fill_null(-1e9)
        if col == "rank_pop":
            s = -cand[col].cast(pl.Float64).fill_null(1e9)
        # these signals have many ties; breaking them by the stored order would
        # quietly borrow the ranking of the other generators
        s = s + rng.random(cand.height) * 1e-6
        df = cand.select("user_id", "label").with_columns(s.alias("score"))
        if T in b.dates[-2:]:
            m, _ = ranking_metrics(df, pos_table=b.folds[T]["pos"])
            rows.append(m)
    out["heuristics"][label] = {
        "desc": desc,
        "ndcg@10": float(np.mean([r["ndcg@10"] for r in rows])),
        "ndcg@10_abs": float(np.mean([r["ndcg@10_abs"] for r in rows])),
        "recall@10": float(np.mean([r["recall@10"] for r in rows])),
    }
    print(f"  heuristic {label:11} test ndcg@10 = {out['heuristics'][label]['ndcg@10']:.4f}", flush=True)

# Ranker settings are chosen here, on the validation fold with the reference
# feature set, and every later script reuses the cached choice so that no arm
# gets a tuning budget the others did not.
GRID = [
    {},
    {"num_leaves": 127, "min_data_in_leaf": 40},
    {"num_leaves": 31, "min_data_in_leaf": 120, "learning_rate": 0.08},
    {"num_leaves": 63, "learning_rate": 0.04, "lambda_l2": 8.0},
    {"num_leaves": 63, "lambda_l2": 0.5, "feature_fraction": 0.6},
]
cached = RUNS / f"ranker_params_{a.scale}.json"
if cached.exists():
    saved = json.loads(cached.read_text())
    best_params, trials = saved["params"], saved["trials"]
else:
    ref_parts = b.matrix(human_programs())
    trials = []
    for cfg in GRID:
        m, tt, _ = rank.train_rank(ref_parts, params=cfg or None)
        v, _, _ = rank.evaluate(m, b, ref_parts, "val")
        trials.append({"params": cfg, "val_ndcg": v["ndcg@10"],
                       "iters": m.best_iteration, "train_seconds": tt})
        print(f"  tune {cfg} -> val {v['ndcg@10']:.4f}", flush=True)
        del m
    best_params = max(trials, key=lambda x: x["val_ndcg"])["params"] or None
    cached.write_text(json.dumps({"params": best_params, "trials": trials},
                                 indent=1, ensure_ascii=False))
    del ref_parts
print("ranker settings chosen on validation:", best_params, flush=True)
out["ranker_params"] = best_params
out["tuning_trials"] = trials

for label, progs in [("BASE", []), ("BASE+HUMAN", human_programs())]:
    t0 = time.time()
    parts = b.matrix(progs)
    tfeat = time.time() - t0
    model, ttrain, cats = rank.train_rank(parts, params=best_params)
    val, _, _ = rank.evaluate(model, b, parts, "val")
    test, per_fold, per_user = rank.evaluate(model, b, parts, "test")
    imp = rank.feature_importance(model)
    out["models"][label] = {
        "n_features": len(model.feature_name()), "n_programs": len(progs),
        "best_iteration": model.best_iteration,
        "feature_seconds": round(tfeat, 1), "train_seconds": ttrain,
        "val": val, "test": test, "test_per_fold": per_fold,
        "top_features": imp.head(30).to_dicts(),
    }
    per_user.write_parquet(RUNS / f"peruser_{label.replace('+','_')}_{a.scale}.parquet")
    imp.write_csv(RUNS / f"importance_{label.replace('+','_')}_{a.scale}.csv")
    print(f"  {label:11} nf={len(model.feature_name()):3d} val={val['ndcg@10']:.4f} "
          f"test={test['ndcg@10']:.4f} abs={test['ndcg@10_abs']:.4f} "
          f"auc={test['auc']:.4f} feat={tfeat:.0f}s train={ttrain}s", flush=True)

(RUNS / f"baselines_{a.scale}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print("saved", RUNS / f"baselines_{a.scale}.json")
