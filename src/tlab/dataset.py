import datetime as dt
import numpy as np
import polars as pl
import scipy.sparse as sp

from .config import DATA, TARGET_DAYS, SEED

# only the measurement script flips this; the pipeline always drops seen items
_DROP_SEEN = True


def load_events():
    return pl.read_parquet(DATA / "events.parquet")


def eligible_users():
    return np.sort(pl.read_parquet(DATA / "eligible_users.parquet")["user_id"].to_numpy())


def sample_users(n, seed=SEED):
    pool = eligible_users()
    if n >= len(pool):
        return pool
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool, size=n, replace=False))


def history(events, T):
    return events.filter(pl.col("ts") < T)


def target(events, T, days=TARGET_DAYS):
    return events.filter((pl.col("ts") >= T) & (pl.col("ts") < T + dt.timedelta(days=days)))


def _csr(hist, users, items):
    uidx = {u: k for k, u in enumerate(users)}
    iidx = {i: k for k, i in enumerate(items)}
    r = np.fromiter((uidx[u] for u in hist["user_id"]), dtype=np.int32, count=hist.height)
    c = np.fromiter((iidx[i] for i in hist["item_id"]), dtype=np.int32, count=hist.height)
    m = sp.csr_matrix((np.ones(hist.height, dtype=np.float32), (r, c)),
                      shape=(len(users), len(items)))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def _popular(hist, T, window_days, topn):
    lo = T - dt.timedelta(days=window_days)
    return (
        hist.filter(pl.col("ts") >= lo)
        .group_by("item_id").agg(pl.len().alias("n"))
        .sort(["n", "item_id"], descending=[True, False]).head(topn)
        .with_row_index("rank_pop").cast({"rank_pop": pl.Int32})
        .select("item_id", "rank_pop")
    )


def _genre_popular(uhist, hist, T, attrs, window_days=14, n_genre=3, per_genre=30):
    lo = T - dt.timedelta(days=window_days)
    gen = attrs.filter(pl.col("attr") == "genres").select("item_id", "value")
    gpop = (
        hist.filter(pl.col("ts") >= lo).join(gen, on="item_id")
        .group_by("value", "item_id").agg(pl.len().alias("n"))
        .sort(["value", "n", "item_id"], descending=[False, True, False])
        .group_by("value", maintain_order=True).head(per_genre)
    )
    ugen = (
        uhist.join(gen, on="item_id")
        .group_by("user_id", "value").agg(pl.len().alias("w"))
        .sort(["user_id", "w", "value"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True).head(n_genre)
    )
    joined = (
        ugen.join(gpop, on="value")
        .group_by("user_id", "item_id").agg((pl.col("w") * pl.col("n")).max().alias("gp"))
    )
    return (
        joined.sort(["user_id", "gp", "item_id"], descending=[False, True, False])
        .with_columns(pl.int_range(pl.len()).over("user_id").cast(pl.Int32).alias("rank_gpop"))
        .select("user_id", "item_id", "rank_gpop")
    )


def _als(mat, users, items, topn, factors=64, iters=15, seed=SEED):
    from implicit.als import AlternatingLeastSquares
    model = AlternatingLeastSquares(factors=factors, regularization=0.05, alpha=10.0,
                                    iterations=iters, random_state=seed,
                                    calculate_training_loss=False)
    model.fit(mat, show_progress=False)
    ids, scores = model.recommend(np.arange(len(users)), mat, N=topn,
                                  filter_already_liked_items=True)
    return pl.DataFrame({
        "user_id": np.repeat(users, topn),
        "item_id": items[ids.ravel()],
        "als_score": scores.ravel().astype(np.float32),
        "rank_als": np.tile(np.arange(topn, dtype=np.int32), len(users)),
    }), model


def _item_sim(mat, neigh=50):
    co = (mat.T @ mat).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    pop = np.asarray(mat.sum(axis=0)).ravel() + 1.0
    rows, cols, vals = [], [], []
    for i in range(co.shape[0]):
        s, e = co.indptr[i], co.indptr[i + 1]
        if s == e:
            continue
        idx, v = co.indices[s:e], co.data[s:e]
        v = v / np.sqrt(pop[i] * pop[idx])          # popularity-normalised cosine
        if len(v) > neigh:
            keep = np.argpartition(-v, neigh)[:neigh]
            idx, v = idx[keep], v[keep]
        rows.append(np.full(len(idx), i, dtype=np.int32))
        cols.append(idx.astype(np.int32))
        vals.append(v.astype(np.float32))
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=co.shape)


def _covisit(mat, sim, uhist, users, items, topn, n_seed=8):
    # the n_seed most recently watched items of each user, propagated through
    # the item-item similarity
    seeds = (
        uhist.sort(["user_id", "ts", "item_id"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True).head(n_seed)
    )
    uidx = {u: k for k, u in enumerate(users)}
    iidx = {i: k for k, i in enumerate(items)}
    r = np.fromiter((uidx[u] for u in seeds["user_id"]), dtype=np.int32, count=seeds.height)
    c = np.fromiter((iidx[i] for i in seeds["item_id"]), dtype=np.int32, count=seeds.height)
    seed_mat = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=mat.shape)
    scored = (seed_mat @ sim).tocsr()
    scored = scored - scored.multiply(mat)          # drop items already in the history
    scored.eliminate_zeros()

    ru, ri, rs, rr = [], [], [], []
    for k in range(scored.shape[0]):
        s, e = scored.indptr[k], scored.indptr[k + 1]
        if s == e:
            continue
        idx, v = scored.indices[s:e], scored.data[s:e]
        if len(v) > topn:
            keep = np.argpartition(-v, topn)[:topn]
            idx, v = idx[keep], v[keep]
        order = np.lexsort((idx, -v))
        ru.append(np.full(len(idx), users[k]))
        ri.append(items[idx[order]])
        rs.append(v[order])
        rr.append(np.arange(len(idx), dtype=np.int32))
    return pl.DataFrame({
        "user_id": np.concatenate(ru), "item_id": np.concatenate(ri),
        "covis_score": np.concatenate(rs).astype(np.float32),
        "rank_covis": np.concatenate(rr),
    })


def build_fold(events, users, T, n_cand, keep_only_positive_groups=True):
    """Candidates and labels for prediction date T.

    Item-side generators run on the whole population, which is what a live
    system would have; the ranked rows are restricted to `users`. Nothing here
    touches events at or after T except to write the label column.
    """
    hist = history(events, T)
    uhist = hist.filter(pl.col("user_id").is_in(users))
    tgt = target(events, T).filter(pl.col("user_id").is_in(users))
    items = np.sort(hist["item_id"].unique().to_numpy())
    mat = _csr(uhist, users, items)

    keep = np.asarray(mat.sum(axis=1)).ravel() > 0
    users, mat = users[keep], mat[keep]

    attrs = pl.read_parquet(DATA / "item_attrs.parquet")
    pop = _popular(hist, T, window_days=14, topn=150)
    sim = _item_sim(mat)
    als, als_model = _als(mat, users, items, topn=80)
    cov = _covisit(mat, sim, uhist, users, items, topn=80)
    gpop = _genre_popular(uhist, hist, T, attrs)

    cand = (
        pl.DataFrame({"user_id": users}).join(pop, how="cross")
        .join(als, on=["user_id", "item_id"], how="full", coalesce=True)
        .join(cov, on=["user_id", "item_id"], how="full", coalesce=True)
        .join(gpop, on=["user_id", "item_id"], how="full", coalesce=True)
    )
    # one row per (user, item) in KION, so an item already watched can never be
    # a future target; dropping seen items removes guaranteed negatives
    seen = uhist.select("user_id", "item_id").unique()
    if _DROP_SEEN:
        cand = cand.join(seen, on=["user_id", "item_id"], how="anti")
    cand = cand.with_columns(
        (1.0 / (20 + pl.col("rank_pop").fill_null(10_000))
         + 1.0 / (20 + pl.col("rank_als").fill_null(10_000))
         + 1.0 / (20 + pl.col("rank_covis").fill_null(10_000))
         + 1.0 / (20 + pl.col("rank_gpop").fill_null(10_000))).alias("rrf")
    )
    cand = (
        cand.sort(["user_id", "rrf", "item_id"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True).head(n_cand)
    )

    pos = tgt.select("user_id", "item_id").unique().with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    cand = cand.join(pos, on=["user_id", "item_id"], how="left").with_columns(
        pl.col("label").fill_null(0)
    )

    reachable = pos.join(seen, on=["user_id", "item_id"], how="anti")
    n_cand_pos = int(cand["label"].sum())
    stats = {
        "T": str(T), "users_ranked": len(users), "rows": cand.height,
        "cand_pos": n_cand_pos,
        "reachable_pos": reachable.height,
        "cand_recall": round(n_cand_pos / max(1, reachable.height), 4),
        "users_with_target": int(reachable["user_id"].n_unique()),
    }

    if keep_only_positive_groups:
        hot = cand.filter(pl.col("label") == 1).select("user_id").unique()
        cand = cand.join(hot, on="user_id")
        stats["groups_kept"] = int(hot.height)
        stats["rows_kept"] = cand.height

    cand = cand.with_columns(pl.lit(T).alias("T"))
    return cand, stats, als_model, items, users


def load_attrs():
    """Item attribute table in long form, including the single-valued columns
    that the DSL can route an indirect link through."""
    multi = pl.read_parquet(DATA / "item_attrs.parquet")
    it = pl.read_parquet(DATA / "items.parquet")
    singles = [
        it.select("item_id", pl.col("content_type").alias("value"),
                  pl.lit("content_type").alias("attr")),
        it.select("item_id",
                  pl.when(pl.col("release_year") > 0)
                    .then((pl.col("release_year") // 10 * 10).cast(pl.Utf8))
                    .otherwise(pl.lit("__na__")).alias("value"),
                  pl.lit("decade").alias("attr")),
        it.select("item_id", pl.col("age_rating").cast(pl.Utf8).alias("value"),
                  pl.lit("age_rating").alias("attr")),
    ]
    return pl.concat([multi.select("item_id", "value", "attr")] + singles)
