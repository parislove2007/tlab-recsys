"""The one script that touches the test folds.

Takes the feature sets each search run selected on validation, retrains the
full ranker at the reporting scale, and evaluates once on the held-out weeks.
Also measures what the searched features add on top of the hand-written ones,
and whether a set discovered on a small user sample still helps on a large one.
"""
import argparse
import glob
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from tlab import dataset as D, rank
from tlab.config import RUNS, SCALES
from tlab.dsl import canon, name
from tlab.human import human_programs, grid_programs
from tlab.metrics import paired_bootstrap
from tlab.pipeline import Bundle

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="main")
ap.add_argument("--runs", default="search_*_none_s*.json")
ap.add_argument("--budget-match", type=int, default=0,
                help="keep only the N highest-gain programs of each searched set")
ap.add_argument("--out", default=None)
ap.add_argument("--only", default=None, help="comma separated label prefixes to keep")
ap.add_argument("--skip", default=None, help="comma separated label prefixes to drop")
ap.add_argument("--best-seed-only", action="store_true",
                help="keep only the best validation seed of each search strategy")
a = ap.parse_args()

HUMAN = human_programs()
GRID = grid_programs()
sets = {"BASE": [], "BASE+HUMAN": HUMAN, "BASE+GRID": GRID}
meta = {"BASE": {},
        "BASE+HUMAN": {"n_programs": len(HUMAN), "source": "hand-written"},
        "BASE+GRID": {"n_programs": len(GRID), "source": "systematic grid, no search"}}

for f in sorted(glob.glob(str(RUNS / a.runs))):
    r = json.loads(pathlib.Path(f).read_text())
    progs = r["accepted"]
    if a.budget_match:
        progs = sorted(progs, key=lambda p: -r["gains"].get(canon(p), 0.0))[: a.budget_match]
    label = f"AUTO:{r['strategy']}:s{r['seed']}"
    sets[label] = progs
    meta[label] = {"n_programs": len(progs), "strategy": r["strategy"], "seed": r["seed"],
                   "search_val_ndcg": r["val_trace"][-1]["val_ndcg"],
                   "search_wall_seconds": r["wall_seconds"],
                   "evaluator_cost": r["evaluator_cost"], "proposer_cost": r["proposer_cost"],
                   "programs_evaluated": r["evaluator_cost"]["programs"]}
    print(f"loaded {label}: {len(progs)} programs", flush=True)

# the best searched set per strategy, by the validation score the search itself saw
by_strategy = {}
for label, m in meta.items():
    if not label.startswith("AUTO:"):
        continue
    by_strategy.setdefault(m["strategy"], []).append((m["search_val_ndcg"], label))
for strat, lst in by_strategy.items():
    best = max(lst)[1]
    sets[f"BASE+HUMAN+AUTO:{strat}"] = HUMAN + [
        p for p in sets[best] if canon(p) not in {canon(q) for q in HUMAN}
    ]
    meta[f"BASE+HUMAN+AUTO:{strat}"] = {"n_programs": len(sets[f"BASE+HUMAN+AUTO:{strat}"]),
                                        "strategy": strat, "from": best}

if a.best_seed_only:
    best_labels = {max(lst)[1] for lst in by_strategy.values()}
    sets = {k: v for k, v in sets.items()
            if not k.startswith("AUTO:") or k in best_labels}
if a.only:
    keep = set(a.only.split(","))
    sets = {k: v for k, v in sets.items() if any(k.startswith(x) for x in keep)}
if a.skip:
    drop = set(a.skip.split(","))
    sets = {k: v for k, v in sets.items() if not any(k.startswith(x) for x in drop)}
print("arms:", ", ".join(sets))

users = D.sample_users(SCALES[a.scale])
print(f"\nscale={a.scale}: {len(users)} users", flush=True)
b = Bundle(users)

results, per_user = {}, {}
for label, progs in sets.items():
    t0 = time.time()
    parts = b.matrix(progs)
    tfeat = time.time() - t0
    model, ttrain, _ = rank.train_rank(parts)
    val, _, _ = rank.evaluate(model, b, parts, "val")
    test, per_fold, pu = rank.evaluate(model, b, parts, "test")
    imp = rank.feature_importance(model)
    searched = [name(p) for p in progs]
    share = float(imp.filter(pl.col("feature").is_in(searched))["gain"].sum()
                  / max(imp["gain"].sum(), 1e-9))
    results[label] = {
        **meta.get(label, {}),
        "n_features": len(model.feature_name()), "best_iteration": model.best_iteration,
        "feature_seconds": round(tfeat, 1), "train_seconds": ttrain,
        "gain_share_of_dsl_features": round(share, 4),
        "val": val, "test": test, "test_per_fold": per_fold,
        "top_features": imp.head(25).to_dicts(),
    }
    per_user[label] = pu.group_by("user_id").agg(pl.col("ndcg").mean())
    print(f"  {label:28} nf={len(model.feature_name()):3d} val={val['ndcg@10']:.4f} "
          f"test={test['ndcg@10']:.4f} abs={test['ndcg@10_abs']:.4f} auc={test['auc']:.4f} "
          f"feat={tfeat:.0f}s train={ttrain}s", flush=True)

ref = "BASE+HUMAN"
pairs = {}
for label in results:
    if label == ref:
        continue
    pairs[label] = paired_bootstrap(per_user[ref], per_user[label])
    print(f"  delta vs {ref}: {label:28} {pairs[label]['delta']:+.4f} "
          f"[{pairs[label]['ci_low']:+.4f}, {pairs[label]['ci_high']:+.4f}] "
          f"p_win={pairs[label]['p_win']:.3f}", flush=True)

out = a.out or f"final_{a.scale}{'_bm' + str(a.budget_match) if a.budget_match else ''}.json"
(RUNS / out).write_text(json.dumps(
    {"scale": a.scale, "n_users": int(len(users)), "reference": ref,
     "results": results, "paired_vs_reference": pairs}, indent=1, ensure_ascii=False))
print("saved", RUNS / out)
