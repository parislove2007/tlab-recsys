"""Does the missing expressiveness matter?

The search plateaued inside the flat language, and the shape it could not ask
for is an aggregate of an aggregate: "is this item more popular than what this
user usually watches". The language is extended with exactly that family and
the gain is measured on top of both the hand-written and the searched sets.
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
from tlab.human import human_programs, nested_programs, segment_programs
from tlab.metrics import paired_bootstrap
from tlab.pipeline import Bundle

ap = argparse.ArgumentParser()
ap.add_argument("--scale", default="main")
ap.add_argument("--auto-run", default="search_llm_recorded_none_s0.json")
a = ap.parse_args()

HUMAN = human_programs()
NEST = nested_programs() + segment_programs()
print(f"{len(nested_programs())} nested + {len(segment_programs())} segment programs")
auto = []
f = RUNS / a.auto_run
if f.exists():
    auto = json.loads(f.read_text())["accepted"]
    print(f"searched set: {len(auto)} programs from {a.auto_run}")

sets = {"BASE+HUMAN": HUMAN, "BASE+HUMAN+EXT": HUMAN + NEST}
if auto:
    sets["BASE+AUTO"] = auto
    sets["BASE+AUTO+EXT"] = auto + NEST
    merged = HUMAN + [p for p in auto if canon(p) not in {canon(q) for q in HUMAN}]
    sets["BASE+HUMAN+AUTO"] = merged
    sets["BASE+HUMAN+AUTO+EXT"] = merged + NEST

users = D.sample_users(SCALES[a.scale])
b = Bundle(users)
cached = RUNS / f"ranker_params_{a.scale}.json"
best_params = json.loads(cached.read_text())["params"] if cached.exists() else None
print("ranker settings:", best_params)
res, pu = {}, {}
for label, progs in sets.items():
    t0 = time.time()
    parts = b.matrix(progs)
    tf = time.time() - t0
    m, tt, _ = rank.train_rank(parts, params=best_params)
    val, _, _ = rank.evaluate(m, b, parts, "val")
    test, _, per = rank.evaluate(m, b, parts, "test")
    imp = rank.feature_importance(m)
    nn = [name(p) for p in NEST]
    share = float(imp.filter(pl.col("feature").is_in(nn))["gain"].sum()
                  / max(imp["gain"].sum(), 1e-9)) if label.endswith("EXT") else 0.0
    res[label] = {"n_programs": len(progs), "n_features": len(m.feature_name()),
                  "val": val, "test": test, "feature_seconds": round(tf, 1),
                  "train_seconds": tt, "gain_share_of_nested": round(share, 4),
                  "top_nested": imp.filter(pl.col("feature").is_in(nn)).head(10).to_dicts()}
    pu[label] = per.group_by("user_id").agg(pl.col("ndcg").mean())
    print(f"  {label:26} nf={len(m.feature_name()):3d} val={val['ndcg@10']:.4f} "
          f"test={test['ndcg@10']:.4f} nested_gain_share={share:.2f}", flush=True)

pairs = {}
for base, ext in [("BASE+HUMAN", "BASE+HUMAN+EXT"), ("BASE+AUTO", "BASE+AUTO+EXT"),
                  ("BASE+HUMAN+AUTO", "BASE+HUMAN+AUTO+EXT")]:
    if base in pu and ext in pu:
        pairs[f"{ext} vs {base}"] = paired_bootstrap(pu[base], pu[ext])
        d = pairs[f"{ext} vs {base}"]
        print(f"  {ext} vs {base}: {d['delta']:+.4f} [{d['ci_low']:+.4f}; {d['ci_high']:+.4f}]")

(RUNS / f"nested_{a.scale}.json").write_text(json.dumps(
    {"scale": a.scale, "n_users": int(len(users)), "n_nested": len(NEST),
     "results": res, "paired": pairs}, indent=1, ensure_ascii=False))
print("saved", RUNS / f"nested_{a.scale}.json")
