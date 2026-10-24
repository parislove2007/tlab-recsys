import gc

import numpy as np
import polars as pl

from . import dataset as D
from .config import DATA, TRAIN_DATES, VAL_DATES, ALL_DATES
from .dsl import FoldEngine, canon, name

CAT_FEATURES = ["u_age", "u_income", "u_sex", "i_content_type", "i_genre"]


def _vocab(values):
    return {v: k for k, v in enumerate(sorted(set(values)))}


class Bundle:
    """Everything needed to train and score one experiment at a fixed user scale.

    Candidates come from the shared materialised tables, so every arm sees the
    same lists. Features are compiled per fold and cached by program, which is
    what keeps the search loop cheap.
    """

    def __init__(self, users, dates=ALL_DATES, verbose=True):
        self.users = np.asarray(users)
        self.dates = list(dates)
        uset = pl.Series("user_id", self.users)
        ev = D.load_events()
        attrs = D.load_attrs()
        items = pl.read_parquet(DATA / "items.parquet")
        usr = pl.read_parquet(DATA / "users.parquet")

        self.vocab = {
            "u_age": _vocab(usr["age"].unique().to_list()),
            "u_income": _vocab(usr["income"].unique().to_list()),
            "u_sex": _vocab(usr["sex"].unique().to_list()),
            "i_content_type": _vocab(items["content_type"].unique().to_list()),
            "i_genre": _vocab(items["primary_genre"].unique().to_list()),
        }

        self.folds = {}
        for T in self.dates:
            tag = T.isoformat()
            cand = (
                pl.read_parquet(DATA / "folds" / f"cand_{tag}.parquet")
                .filter(pl.col("user_id").is_in(uset))
            )
            hot = cand.filter(pl.col("label") == 1).select("user_id").unique()
            cand = cand.join(hot, on="user_id").sort(["user_id", "rrf", "item_id"], descending=[False, True, False])
            pos = (pl.read_parquet(DATA / "folds" / f"pos_{tag}.parquet")
                   .filter(pl.col("user_id").is_in(uset)))
            engine = FoldEngine(D.history(ev, T), T, attrs, users=self.users, user_attrs=usr)
            self.folds[T] = {
                "cand": cand,
                "pos": pos,
                "base": self._base(cand, T, usr, items),
                "engine": engine,
            }
            if verbose:
                print(f"  fold {tag}: {cand.height} rows, {hot.height} groups", flush=True)
        self._cache = {}

    def _base(self, cand, T, usr, items):
        tag = T.isoformat()
        uf = np.load(DATA / "folds" / f"als_user_{tag}.npy")
        itf = np.load(DATA / "folds" / f"als_item_{tag}.npy")
        uids = np.load(DATA / "folds" / f"als_users_{tag}.npy")
        iids = np.load(DATA / "folds" / f"als_items_{tag}.npy")
        upos = {u: k for k, u in enumerate(uids)}
        ipos = {i: k for k, i in enumerate(iids)}
        ur = np.fromiter((upos.get(u, -1) for u in cand["user_id"]), dtype=np.int64, count=cand.height)
        ir = np.fromiter((ipos.get(i, -1) for i in cand["item_id"]), dtype=np.int64, count=cand.height)
        ok = (ur >= 0) & (ir >= 0)
        dot = np.full(cand.height, np.nan, dtype=np.float32)
        dot[ok] = np.einsum("ij,ij->i", uf[ur[ok]], itf[ir[ok]])

        df = cand.select(
            "user_id", "item_id",
            pl.col("rank_pop").fill_null(9999).cast(pl.Float32).alias("b_rank_pop"),
            pl.col("rank_als").fill_null(9999).cast(pl.Float32).alias("b_rank_als"),
            pl.col("rank_covis").fill_null(9999).cast(pl.Float32).alias("b_rank_covis"),
            pl.col("rank_gpop").fill_null(9999).cast(pl.Float32).alias("b_rank_gpop"),
            pl.col("als_score").cast(pl.Float32).alias("b_als_score"),
            pl.col("covis_score").cast(pl.Float32).alias("b_covis_score"),
            pl.col("rrf").cast(pl.Float32).alias("b_rrf"),
            (pl.col("rank_pop").is_not_null().cast(pl.Int8)
             + pl.col("rank_als").is_not_null().cast(pl.Int8)
             + pl.col("rank_covis").is_not_null().cast(pl.Int8)
             + pl.col("rank_gpop").is_not_null().cast(pl.Int8)).cast(pl.Float32).alias("b_n_sources"),
        ).with_columns(pl.Series("b_als_dot", dot))

        df = (
            df.join(usr, on="user_id", how="left")
            .with_columns(
                pl.col("age").fill_null("__na__").replace_strict(self.vocab["u_age"], default=-1).cast(pl.Int32).alias("u_age"),
                pl.col("income").fill_null("__na__").replace_strict(self.vocab["u_income"], default=-1).cast(pl.Int32).alias("u_income"),
                pl.col("sex").fill_null("__na__").replace_strict(self.vocab["u_sex"], default=-1).cast(pl.Int32).alias("u_sex"),
                pl.col("kids_flg").fill_null(0).cast(pl.Float32).alias("u_kids"),
            )
            .drop("age", "income", "sex", "kids_flg")
            .join(items, on="item_id", how="left")
            .with_columns(
                pl.col("content_type").replace_strict(self.vocab["i_content_type"], default=-1).cast(pl.Int32).alias("i_content_type"),
                pl.col("primary_genre").replace_strict(self.vocab["i_genre"], default=-1).cast(pl.Int32).alias("i_genre"),
                pl.col("release_year").cast(pl.Float32).alias("i_year"),
                pl.when(pl.col("release_year") > 0).then(2021 - pl.col("release_year")).otherwise(None).cast(pl.Float32).alias("i_years_old"),
                pl.col("age_rating").cast(pl.Float32).alias("i_age_rating"),
                pl.col("for_kids").cast(pl.Float32).alias("i_for_kids"),
                pl.col("n_genres").cast(pl.Float32).alias("i_n_genres"),
                pl.col("title_len").cast(pl.Float32).alias("i_title_len"),
                pl.col("descr_len").cast(pl.Float32).alias("i_descr_len"),
                pl.col("n_keywords").cast(pl.Float32).alias("i_n_keywords"),
            )
            .drop("content_type", "primary_genre", "release_year", "age_rating",
                  "for_kids", "n_genres", "title_len", "descr_len", "n_keywords")
        )
        return df.drop("user_id", "item_id")

    def base_cols(self):
        return list(self.folds[self.dates[0]]["base"].columns)

    def features(self, progs, max_cache=100):
        """Per-fold feature frames for the given programs, caching by program.

        The engines keep their intermediate groupings only while those stay
        small; the large ones, which is what exhausts memory over a long search,
        are rebuilt on demand. Finished columns stay in `_cache`, which is what
        the search actually reuses.
        """
        out = {}
        for T in self.dates:
            f = self.folds[T]
            need, cols = [], {}
            for p in progs:
                key = (T, canon(p))
                if key in self._cache:
                    cols[name(p)] = self._cache[key]
                else:
                    need.append(p)
            if need:
                got = f["engine"].compute(need, f["cand"])
                for p in need:
                    nm = name(p)
                    col = got[nm]
                    self._cache[(T, canon(p))] = col
                    cols[nm] = col
                del got
            out[T] = pl.DataFrame({name(p): cols[name(p)] for p in progs})
        if len(self._cache) > max_cache * len(self.dates):
            self._cache.clear()
            gc.collect()
        return out

    def matrix(self, progs, use_base=True):
        """Assembled (X, y, group sizes) per split."""
        feats = self.features(progs) if progs else {T: None for T in self.dates}
        parts = {"train": [], "val": [], "test": []}
        for T in self.dates:
            split = ("train" if T in TRAIN_DATES else "val" if T in VAL_DATES else "test")
            f = self.folds[T]
            pieces = []
            if use_base:
                pieces.append(f["base"])
            if feats[T] is not None and feats[T].width:
                pieces.append(feats[T])
            X = pl.concat(pieces, how="horizontal") if pieces else pl.DataFrame()
            parts[split].append((T, X, f["cand"]))
        return parts
