"""How much does the choice of time window matter, relation by relation?

For each relation the same counting feature is added to the base model with
every available window, one at a time, and the validation gain is recorded.
This is the empirical version of the question the task asks: why seven days?
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from tlab import dataset as D, search
from tlab.config import RUNS, SCALES
from tlab.dsl import WINDOWS, name
from tlab.pipeline import Bundle

RELATIONS = [
    ("user activity", dict(anchor="user", via=None, pool=None, event="start")),
    ("user watches", dict(anchor="user", via=None, pool=None, event="watch")),
    ("item popularity", dict(anchor="item", via=None, pool=None, event="start")),
    ("item watches", dict(anchor="item", via=None, pool=None, event="watch")),
    ("user x genre", dict(anchor="attr", via="genres", pool="sum", event="start")),
    ("user x director", dict(anchor="attr", via="directors", pool="sum", event="start")),
    ("user x keyword", dict(anchor="attr", via="keywords", pool="sum", event="start")),
    ("genre popularity", dict(anchor="attr_item", via="genres", pool="mean", event="start")),
]

users = D.sample_users(SCALES["search"])
b = Bundle(users, verbose=False)
ev = search.Evaluator(b)
base, _ = ev.val_ndcg([])
print(f"base val ndcg@10 = {base:.4f}", flush=True)

rows = []
for label, spec in RELATIONS:
    for w in WINDOWS:
        p = dict(kind="agg", agg="count", window=w, norm=None, **spec)
        t0 = time.time()
        s, _ = ev.val_ndcg([p])
        rows.append({"relation": label, "window": w, "val_ndcg": s,
                     "gain": s - base, "feature": name(p),
                     "seconds": round(time.time() - t0, 1)})
        print(f"  {label:18} w={str(w):>4}  val={s:.4f}  gain={s-base:+.4f}", flush=True)

best = {}
for label, _ in RELATIONS:
    r = [x for x in rows if x["relation"] == label]
    best[label] = max(r, key=lambda x: x["val_ndcg"])
(RUNS / "window_sweep.json").write_text(json.dumps(
    {"base_val_ndcg": base, "rows": rows, "best_per_relation": best}, indent=1, ensure_ascii=False))
print("\nbest window per relation:")
for k, v in best.items():
    print(f"  {k:18} -> {v['window']} days (gain {v['gain']:+.4f})")
