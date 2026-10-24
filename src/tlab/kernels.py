"""Learned temporal kernels.

The hand-written features pick a window and count events inside it. The window
is a guess: why seven days and not ten? Here the hard window is replaced by a
smooth kernel exp(-lambda * age) whose decay rate is a parameter, so the time
horizon of each relation is fitted from the data instead of chosen.

Everything factorises. For a relation r the daily event counts live in a small
key-level matrix S_r (user x day, item x day, (user, value) x day, ...), and a
sparse assignment P_r maps each candidate row to the keys it draws on. The
feature is then

    f_r = P_r @ (S_r @ exp(-lambda_r * d))

which is cheap, exact, and differentiable in lambda. The learned half-life
ln 2 / lambda is directly comparable to the windows a human would have picked.
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
import torch.nn as nn

MAX_AGE = 200

RELATIONS = [
    ("user_start", "user", None, "start"),
    ("user_watch", "user", None, "watch"),
    ("item_start", "item", None, "start"),
    ("item_watch", "item", None, "watch"),
    ("item_finish", "item", None, "finish"),
    ("ugenre", "attr", "genres", "start"),
    ("udirector", "attr", "directors", "start"),
    ("ukeyword", "attr", "keywords", "start"),
    ("igenre", "attr_item", "genres", "start"),
]


def _coo(rows, cols, n_rows, n_cols, vals=None):
    v = np.ones(len(rows), dtype=np.float32) if vals is None else vals.astype(np.float32)
    m = sp.coo_matrix((v, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    m.sum_duplicates()
    return m


def build_relation(engine, cand, anchor, via, event):
    """Key-level daily counts S and the row-to-key assignment P."""
    df = engine.u[event] if anchor in ("user", "attr") else engine.g[event]
    df = df.filter(pl.col("age") < MAX_AGE)
    rows = cand.with_row_index("_rid").select(
        "_rid", pl.col("user_id").cast(pl.Int32), pl.col("item_id").cast(pl.Int32))

    if anchor == "user":
        keys = rows["user_id"].unique().sort()
        kpos = {k: i for i, k in enumerate(keys)}
        cnt = df.filter(pl.col("user_id").is_in(keys)).group_by("user_id", "age").agg(pl.len().alias("n"))
        S = _coo(np.fromiter((kpos[u] for u in cnt["user_id"]), np.int32, cnt.height),
                 cnt["age"].to_numpy(), len(keys), MAX_AGE, cnt["n"].to_numpy())
        P = _coo(rows["_rid"].to_numpy(),
                 np.fromiter((kpos[u] for u in rows["user_id"]), np.int32, rows.height),
                 rows.height, len(keys))
        return S, P

    if anchor == "item":
        keys = rows["item_id"].unique().sort()
        kpos = {k: i for i, k in enumerate(keys)}
        cnt = df.filter(pl.col("item_id").is_in(keys)).group_by("item_id", "age").agg(pl.len().alias("n"))
        S = _coo(np.fromiter((kpos[i] for i in cnt["item_id"]), np.int32, cnt.height),
                 cnt["age"].to_numpy(), len(keys), MAX_AGE, cnt["n"].to_numpy())
        P = _coo(rows["_rid"].to_numpy(),
                 np.fromiter((kpos[i] for i in rows["item_id"]), np.int32, rows.height),
                 rows.height, len(keys))
        return S, P

    amap = engine._attr_map(via)
    joined = df.join(amap, on="item_id")
    row_keys = rows.join(amap, on="item_id")
    if anchor == "attr":
        cnt = joined.group_by("user_id", "value", "age").agg(pl.len().alias("n"))
        pair = pl.concat([cnt.select("user_id", "value"), row_keys.select("user_id", "value")]).unique()
        pair = pair.with_row_index("_kid")
        cnt = cnt.join(pair, on=["user_id", "value"])
        row_keys = row_keys.join(pair, on=["user_id", "value"])
        nk = pair.height
    else:
        cnt = joined.group_by("value", "age").agg(pl.len().alias("n"))
        pair = pl.concat([cnt.select("value"), row_keys.select("value")]).unique().with_row_index("_kid")
        cnt = cnt.join(pair, on="value")
        row_keys = row_keys.join(pair, on="value")
        nk = pair.height

    S = _coo(cnt["_kid"].to_numpy(), cnt["age"].to_numpy(), nk, MAX_AGE, cnt["n"].to_numpy())
    P = _coo(row_keys["_rid"].to_numpy(), row_keys["_kid"].to_numpy(), rows.height, nk)
    return S, P


def _to_torch(m):
    m = m.tocoo()
    return torch.sparse_coo_tensor(
        np.vstack([m.row, m.col]), m.data.astype(np.float32), m.shape
    ).coalesce()


class FoldTensors:
    def __init__(self, engine, cand, relations=RELATIONS):
        self.S, self.P = [], []
        for _, anchor, via, event in relations:
            S, P = build_relation(engine, cand, anchor, via, event)
            self.S.append(_to_torch(S))
            self.P.append(_to_torch(P))
        self.y = torch.tensor(cand["label"].to_numpy().astype(np.float32))
        uid = cand["user_id"].to_numpy()
        _, inv = np.unique(uid, return_inverse=True)
        self.gid = torch.tensor(inv.astype(np.int64))
        self.n_groups = int(inv.max()) + 1
        self.n_rows = cand.height


class KernelRanker(nn.Module):
    def __init__(self, n_rel, n_kernels=2, hidden=48, init_halflife=(7.0, 60.0)):
        super().__init__()
        init = np.log(np.expm1(np.log(2.0) / np.asarray(init_halflife, dtype=np.float64)))
        self.raw = nn.Parameter(
            torch.tensor(np.tile(init, (n_rel, 1)), dtype=torch.float32))
        d = n_rel * n_kernels
        self.head = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.ages = torch.arange(MAX_AGE, dtype=torch.float32)

    def lam(self):
        return nn.functional.softplus(self.raw) + 1e-4

    def features(self, t):
        lam = self.lam()
        out = []
        for r, (S, P) in enumerate(zip(t.S, t.P)):
            k = torch.exp(-lam[r].unsqueeze(1) * self.ages.unsqueeze(0)).T   # (D, M)
            out.append(torch.sparse.mm(P, torch.sparse.mm(S, k)))
        return torch.log1p(torch.cat(out, dim=1))

    def forward(self, t):
        return self.head(self.features(t)).squeeze(1)

    def half_lives(self):
        return (np.log(2.0) / self.lam().detach().numpy())


def listwise_loss(score, y, gid, n_groups):
    """Softmax cross-entropy within each candidate list."""
    m = torch.full((n_groups,), -1e30).index_reduce_(0, gid, score, "amax", include_self=True)
    ex = torch.exp(score - m[gid])
    denom = torch.zeros(n_groups).index_add_(0, gid, ex)
    logp = score - m[gid] - torch.log(denom[gid] + 1e-12)
    npos = torch.zeros(n_groups).index_add_(0, gid, y).clamp(min=1.0)
    return -(logp * y / npos[gid]).sum() / n_groups


POOL_FOR = {"user": None, "item": None, "attr": "sum", "attr_item": "mean"}


def to_decay_programs(relations, half_lives, norms=(None, "z_in_group")):
    """Turns fitted decay rates into ordinary DSL programs.

    After this the learned horizons are nothing special: they compile, cache and
    pass the leak tests like any other feature, and can be measured at any scale.
    """
    out = []
    for i, (_, anchor, via, event) in enumerate(relations):
        for h in half_lives[i]:
            for nrm in norms:
                if anchor in ("user", "attr") and nrm == "z_in_group":
                    # a user-keyed value is constant inside a candidate list
                    if anchor == "user":
                        continue
                out.append({"kind": "decay", "anchor": anchor, "via": via,
                            "event": event, "halflife": round(float(h), 3),
                            "pool": POOL_FOR[anchor], "norm": nrm})
    seen, uniq = set(), []
    for q in out:
        k = str(sorted(q.items(), key=lambda kv: kv[0]))
        if k not in seen:
            seen.add(k)
            uniq.append(q)
    return uniq
