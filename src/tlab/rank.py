import gc
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .config import TOPK, SEED
from .metrics import ranking_metrics, pointwise_metrics, paired_bootstrap
from .pipeline import CAT_FEATURES

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [TOPK],
    "lambdarank_truncation_level": 30,
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "num_threads": 8,
    "verbosity": -1,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
}


def _stack(parts, split):
    Xs, ys, groups, keys = [], [], [], []
    for T, X, cand in parts[split]:
        Xs.append(X)
        ys.append(cand["label"].to_numpy())
        sizes = cand.group_by("user_id", maintain_order=True).len()["len"].to_numpy()
        groups.append(sizes)
        keys.append(cand.select("user_id", "item_id").with_columns(pl.lit(str(T)).alias("T")))
    return (pl.concat(Xs), np.concatenate(ys), np.concatenate(groups), pl.concat(keys))


def train_rank(parts, params=None, num_boost_round=800, early_stopping=60, quiet=True):
    p = dict(PARAMS)
    if params:
        p.update(params)
    Xtr, ytr, gtr, _ = _stack(parts, "train")
    Xva, yva, gva, kva = _stack(parts, "val")
    cats = [c for c in CAT_FEATURES if c in Xtr.columns]
    names = Xtr.columns
    t0 = time.time()
    # the frames are released as soon as LightGBM has binned them: a wide
    # feature set is several gigabytes and two live copies do not fit
    dtr = lgb.Dataset(Xtr.to_pandas(), label=ytr, group=gtr, feature_name=names,
                      categorical_feature=cats, free_raw_data=True)
    del Xtr
    gc.collect()
    dtr.construct()
    dva = lgb.Dataset(Xva.to_pandas(), label=yva, group=gva, feature_name=names,
                      categorical_feature=cats, reference=dtr, free_raw_data=True)
    del Xva
    gc.collect()
    dva.construct()
    cb = [lgb.early_stopping(early_stopping, verbose=False)]
    model = lgb.train(p, dtr, num_boost_round=num_boost_round, valid_sets=[dva], callbacks=cb)
    del dtr, dva
    gc.collect()
    return model, round(time.time() - t0, 2), cats


def score_split(model, parts, split, with_pos=True):
    res, per_user = {}, {}
    for T, X, cand in parts[split]:
        s = model.predict(X.to_pandas(), num_iteration=model.best_iteration)
        df = cand.select("user_id", "label").with_columns(pl.Series("score", s))
        pos = None
        if with_pos:
            pos = [f for f in parts[split] if f[0] == T][0][2]
        res[T] = df
    return res


def evaluate(model, bundle, parts, split, k=TOPK):
    agg, pu_all = [], []
    for T, X, cand in parts[split]:
        s = model.predict(X.to_pandas(), num_iteration=model.best_iteration)
        df = cand.select("user_id", "label").with_columns(pl.Series("score", s.astype(np.float32)))
        m, pu = ranking_metrics(df, k=k, pos_table=bundle.folds[T]["pos"])
        m.update(pointwise_metrics(df))
        m["T"] = str(T)
        agg.append(m)
        pu_all.append(pu.with_columns(pl.lit(str(T)).alias("T")))
    keys = [c for c in agg[0] if c != "T"]
    mean = {c: float(np.mean([a[c] for a in agg])) for c in keys if isinstance(agg[0][c], float)}
    mean["groups"] = int(sum(a["groups"] for a in agg))
    per_user = pl.concat(pu_all)
    return mean, agg, per_user


def feature_importance(model, kind="gain"):
    names = model.feature_name()
    vals = model.feature_importance(importance_type=kind)
    return (pl.DataFrame({"feature": names, kind: vals.astype(np.float64)})
            .sort(kind, descending=True))
