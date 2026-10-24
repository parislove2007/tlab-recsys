"""Materialise the candidate set once, for every fold, over all eligible users.

Smaller experiment scales are user subsets of this table, so every arm and
every scale sees literally the same candidates.
"""
import json
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl
from tlab import dataset as D
from tlab.config import DATA, ALL_DATES, N_CAND

out = DATA / "folds"
out.mkdir(exist_ok=True)
ev = D.load_events()
users = D.eligible_users()
print("eligible users", len(users))

all_stats = []
for T in ALL_DATES:
    t0 = time.time()
    cand, stats, model, items, ranked = D.build_fold(ev, users, T, n_cand=N_CAND)
    tag = T.isoformat()

    # per-user positive accounting, needed for absolute (not candidate-restricted) metrics
    tgt = D.target(ev, T).filter(pl.col("user_id").is_in(users)).select("user_id", "item_id").unique()
    seen = D.history(ev, T).filter(pl.col("user_id").is_in(users)).select("user_id", "item_id").unique()
    reach = (tgt.join(seen, on=["user_id", "item_id"], how="anti")
                .group_by("user_id").agg(pl.len().alias("n_reachable")))
    got = (cand.filter(pl.col("label") == 1).group_by("user_id").agg(pl.len().alias("n_in_cand")))
    reach.join(got, on="user_id", how="left").with_columns(
        pl.col("n_in_cand").fill_null(0)
    ).write_parquet(out / f"pos_{tag}.parquet")

    cand.write_parquet(out / f"cand_{tag}.parquet")
    np.save(out / f"als_user_{tag}.npy", model.user_factors.astype(np.float32))
    np.save(out / f"als_item_{tag}.npy", model.item_factors.astype(np.float32))
    np.save(out / f"als_items_{tag}.npy", items)
    np.save(out / f"als_users_{tag}.npy", ranked)

    stats["seconds"] = round(time.time() - t0, 1)
    all_stats.append(stats)
    print(json.dumps(stats, ensure_ascii=False))

(DATA / "fold_stats.json").write_text(json.dumps(all_stats, indent=2, ensure_ascii=False))
