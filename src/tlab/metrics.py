import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss

from .config import TOPK


def _dcg(rel):
    return np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))


def ranking_metrics(df, k=TOPK, pos_table=None):
    """Ranking quality of `score` against `label`, grouped by user.

    Two views are reported. The candidate-restricted view treats the candidate
    list as the universe, so it isolates the ranker. The absolute view charges
    the ranker for positives the candidate stage never retrieved, which is the
    number a product would actually see.
    """
    d = df.select("user_id", "label", "score").sort(["user_id", "score"], descending=[False, True])
    users = d["user_id"].to_numpy()
    labels = d["label"].to_numpy().astype(np.float64)
    bounds = np.flatnonzero(np.diff(users)) + 1
    groups = np.split(np.arange(len(users)), bounds)

    uids, ndcg, ap, rec, hit, npos = [], [], [], [], [], []
    for g in groups:
        rel = labels[g]
        n = rel.sum()
        if n == 0:
            continue
        top = rel[:k]
        ideal = np.zeros(k)
        ideal[: int(min(n, k))] = 1.0
        ndcg.append(_dcg(top) / _dcg(ideal))
        prec = np.cumsum(top) / np.arange(1, len(top) + 1)
        ap.append(float((prec * top).sum() / min(n, k)))
        rec.append(float(top.sum() / n))
        hit.append(float(top.sum() > 0))
        uids.append(users[g[0]])
        npos.append(n)

    out = {
        f"ndcg@{k}": float(np.mean(ndcg)), f"map@{k}": float(np.mean(ap)),
        f"recall@{k}": float(np.mean(rec)), f"hit@{k}": float(np.mean(hit)),
        "groups": len(ndcg),
    }
    per_user = pl.DataFrame({"user_id": uids, "ndcg": ndcg, "ap": ap,
                             "recall": rec, "hit": hit, "n_pos": npos})

    if pos_table is not None:
        # users whose positives were all lost by the candidate stage score zero
        full = pos_table.join(per_user, on="user_id", how="left")
        n_all = full.height
        ideal_k = np.minimum(full["n_reachable"].to_numpy(), k)
        idcg = np.array([_dcg(np.r_[np.ones(int(i)), np.zeros(k - int(i))]) for i in ideal_k])
        got = full["ndcg"].fill_null(0.0).to_numpy()
        got_pos = full["n_pos"].fill_null(0).to_numpy()
        # rescale candidate-restricted ndcg onto the absolute ideal
        idcg_cand = np.array([_dcg(np.r_[np.ones(int(min(p, k))), np.zeros(k - int(min(p, k)))])
                              if p > 0 else 1.0 for p in got_pos])
        out[f"ndcg@{k}_abs"] = float(np.mean(got * idcg_cand / np.maximum(idcg, 1e-9)))
        out[f"recall@{k}_abs"] = float(
            np.mean(full["recall"].fill_null(0.0).to_numpy() * got_pos
                    / np.maximum(full["n_reachable"].to_numpy(), 1))
        )
        out["groups_abs"] = n_all
    return out, per_user


def pointwise_metrics(df):
    y = df["label"].to_numpy()
    s = df["score"].to_numpy()
    if y.min() == y.max():
        return {}
    p = 1.0 / (1.0 + np.exp(-np.clip(s, -30, 30)))
    return {"auc": float(roc_auc_score(y, s)), "logloss": float(log_loss(y, p, labels=[0, 1]))}


def paired_bootstrap(a, b, n=2000, seed=0):
    """Paired bootstrap over users for the difference b - a on a shared metric."""
    j = a.join(b, on="user_id", suffix="_b")
    x = j["ndcg"].to_numpy()
    y = j["ndcg_b"].to_numpy()
    d = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    boot = d[idx].mean(axis=1)
    return {"delta": float(d.mean()), "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "p_win": float((boot > 0).mean()), "n_users": len(d)}
