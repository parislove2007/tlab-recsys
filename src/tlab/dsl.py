"""A small typed language for point-in-time features over an event log.

A feature is a JSON program, not code. The compiler is handed only the events
that happened strictly before the prediction date, so no program can reach
into the future: leak-freeness is a property of the engine, not of the
individual feature. That is what makes it safe to let a search procedure (or a
language model) invent features without reviewing each one by hand.
"""
import json
import math

import polars as pl

ANCHORS = ["user", "item", "attr", "attr_item"]
VIA = ["genres", "countries", "studios", "directors", "actors", "keywords",
       "content_type", "decade", "age_rating"]
EVENTS = ["start", "watch", "finish"]
WINDOWS = [1, 3, 7, 14, 30, 60, 90, 180, None]
AGGS = ["count", "nuniq_item", "nuniq_user", "nuniq_day", "mean_pct", "max_pct",
        "std_pct", "sum_dur", "mean_dur", "recency", "mean_age", "trend"]
NORMS = [None, "log1p", "per_day", "share_user", "share_item", "z_in_group", "rank_in_group"]
POOLS = ["max", "sum", "mean", "min"]
OUTERS = ["mean", "max", "min", "std"]
COMPARES = ["profile", "diff", "ratio"]
INNER_NORMS = [None, "log1p", "per_day", "share_item"]
SEG_DIMS = ["age", "income", "sex", "kids_flg"]
SEG_AGGS = ["count", "nuniq_user", "mean_pct", "trend"]
SEG_NORMS = [None, "log1p", "z_in_group", "rank_in_group", "share_item"]

# Aggregates are computed in float64 and narrowed once, at the end: watch-time
# sums over millions of events do not survive float32.
_PCT = pl.col("watched_pct").cast(pl.Float64)
_DUR = pl.col("total_dur").cast(pl.Float64)

AGG_EXPR = {
    "count": lambda w: pl.len().cast(pl.Float64),
    "nuniq_item": lambda w: pl.col("item_id").n_unique().cast(pl.Float64),
    "nuniq_user": lambda w: pl.col("user_id").n_unique().cast(pl.Float64),
    "nuniq_day": lambda w: pl.col("age").n_unique().cast(pl.Float64),
    "mean_pct": lambda w: _PCT.mean(),
    "max_pct": lambda w: _PCT.max(),
    "std_pct": lambda w: _PCT.std(),
    "sum_dur": lambda w: _DUR.sum(),
    "mean_dur": lambda w: _DUR.mean(),
    "recency": lambda w: pl.col("age").min().cast(pl.Float64),
    "mean_age": lambda w: pl.col("age").cast(pl.Float64).mean(),
    "trend": lambda w: (
        (pl.col("age") < (w or 60) / 2).sum().cast(pl.Float64)
        / ((pl.col("age") >= (w or 60) / 2).sum().cast(pl.Float64) + 1.0)
    ),
}
RATE_AGGS = {"count", "nuniq_item", "nuniq_user", "nuniq_day", "sum_dur"}


class InvalidProgram(Exception):
    pass


def validate(p):
    k = p.get("kind")
    if k == "agg":
        _validate_agg(p)
    elif k == "decay":
        _validate_decay(p)
    elif k == "segment":
        _validate_segment(p)
    elif k == "nested":
        _validate_nested(p)
    elif k in ("ratio", "mul"):
        for side in ("a", "b"):
            if side not in p:
                raise InvalidProgram(f"{k} needs 'a' and 'b'")
            _validate_agg(p[side])
        if k == "ratio" and p["a"] == p["b"]:
            raise InvalidProgram("ratio of a program with itself is constant")
    else:
        raise InvalidProgram(f"unknown kind {k!r}")
    return p


def _validate_decay(p):
    """A soft window: every past event contributes exp(-ln2 * age / halflife).

    This is how a horizon fitted by gradient descent comes back into the
    language as an ordinary program, so it is cached, compiled and checked for
    leaks exactly like any hand-written feature.
    """
    a, via, pool = p.get("anchor"), p.get("via"), p.get("pool")
    if a not in ANCHORS:
        raise InvalidProgram(f"anchor {a!r}")
    if p.get("event") not in EVENTS:
        raise InvalidProgram(f"event {p.get('event')!r}")
    hl = p.get("halflife")
    if not isinstance(hl, (int, float)) or not (0.05 <= float(hl) <= 2000):
        raise InvalidProgram(f"halflife {hl!r}")
    if p.get("norm") not in (None, "log1p", "z_in_group", "rank_in_group"):
        raise InvalidProgram(f"norm {p.get('norm')!r}")
    if a in ("attr", "attr_item"):
        if via not in VIA:
            raise InvalidProgram(f"anchor {a} needs a via")
        if pool not in POOLS:
            raise InvalidProgram(f"anchor {a} needs a pool")
    elif via is not None or pool is not None:
        raise InvalidProgram(f"anchor {a} takes no via/pool")


def _validate_segment(p):
    """Demand for the candidate among users demographically like this one.

    The flat language has no user-side attribute, so "popular among women aged
    25-34" cannot be written in it at all.
    """
    if p.get("dim") not in SEG_DIMS:
        raise InvalidProgram(f"dim {p.get('dim')!r}")
    if p.get("agg") not in SEG_AGGS:
        raise InvalidProgram(f"agg {p.get('agg')!r}")
    if p.get("event") not in EVENTS:
        raise InvalidProgram(f"event {p.get('event')!r}")
    if p.get("window") not in WINDOWS:
        raise InvalidProgram(f"window {p.get('window')!r}")
    if p.get("norm") not in SEG_NORMS:
        raise InvalidProgram(f"norm {p.get('norm')!r}")


def _validate_nested(p):
    """A profile of the user's history measured with an item-level feature.

    This is the one shape the flat language cannot express: an aggregate of an
    aggregate, such as "how popular is the content this user usually watches".
    """
    inner = p.get("inner")
    if not isinstance(inner, dict):
        raise InvalidProgram("nested needs an inner agg program")
    _validate_agg(inner)
    if inner["anchor"] != "item":
        raise InvalidProgram("the inner program must be item-keyed")
    if inner.get("norm") not in INNER_NORMS:
        raise InvalidProgram("inner norm must not depend on the candidate list")
    if p.get("outer") not in OUTERS:
        raise InvalidProgram(f"outer {p.get('outer')!r}")
    if p.get("compare") not in COMPARES:
        raise InvalidProgram(f"compare {p.get('compare')!r}")
    if p.get("event") not in EVENTS:
        raise InvalidProgram(f"event {p.get('event')!r}")
    if p.get("window") not in WINDOWS:
        raise InvalidProgram(f"window {p.get('window')!r}")
    if p["compare"] != "profile" and p["outer"] == "std":
        raise InvalidProgram("comparing against a standard deviation is not meaningful")


def _validate_agg(p):
    if p.get("kind") != "agg":
        raise InvalidProgram("expected an agg program")
    for f in ("anchor", "event", "agg"):
        if f not in p:
            raise InvalidProgram(f"missing {f}")
    a, via, ev, w, ag = p["anchor"], p.get("via"), p["event"], p.get("window"), p["agg"]
    nrm, pool = p.get("norm"), p.get("pool")
    if a not in ANCHORS:
        raise InvalidProgram(f"anchor {a!r}")
    if ev not in EVENTS:
        raise InvalidProgram(f"event {ev!r}")
    if ag not in AGGS:
        raise InvalidProgram(f"agg {ag!r}")
    if w not in WINDOWS:
        raise InvalidProgram(f"window {w!r}")
    if nrm not in NORMS:
        raise InvalidProgram(f"norm {nrm!r}")
    if a in ("attr", "attr_item"):
        if via not in VIA:
            raise InvalidProgram(f"anchor {a} needs a via from {VIA}")
        if pool not in POOLS:
            raise InvalidProgram(f"anchor {a} needs a pool from {POOLS}")
    else:
        if via is not None or pool is not None:
            raise InvalidProgram(f"anchor {a} takes no via/pool")
    # aggregates that are constant by construction for a given anchor
    if ag == "nuniq_user" and a in ("user", "attr"):
        raise InvalidProgram("nuniq_user is always 1 when the group is a single user")
    if ag == "nuniq_item" and a == "item":
        raise InvalidProgram("nuniq_item is always 1 when the group is a single item")
    if nrm == "per_day" and ag not in RATE_AGGS:
        raise InvalidProgram("per_day only makes sense for counting aggregates")
    if nrm == "share_user" and a not in ("user", "attr"):
        raise InvalidProgram("share_user needs a user-keyed anchor")
    if nrm == "share_item" and a not in ("item", "attr_item"):
        raise InvalidProgram("share_item needs an item-keyed anchor")
    if nrm in ("share_user", "share_item") and ag not in RATE_AGGS:
        raise InvalidProgram("shares only make sense for counting aggregates")


def name(p):
    if p["kind"] == "decay":
        bits = ["dec", p["anchor"]]
        if p.get("via"):
            bits.append(p["via"][:4])
        bits += [p["event"][:3], f"h{p['halflife']:g}"]
        if p.get("pool"):
            bits.append(p["pool"][:3])
        if p.get("norm"):
            bits.append(p["norm"])
        return "_".join(bits)
    if p["kind"] == "segment":
        return ("seg_" + p["dim"][:4] + "_" + p["event"][:3] + "_"
                + ("all" if p.get("window") is None else f"w{p['window']}")
                + "_" + p["agg"] + ("_" + p["norm"] if p.get("norm") else ""))
    if p["kind"] == "nested":
        return ("nest_" + name(p["inner"]) + f"_{p['outer']}_{p['event'][:3]}_"
                + ("all" if p.get("window") is None else f"w{p['window']}")
                + f"_{p['compare']}")
    if p["kind"] == "agg":
        bits = [p["anchor"]]
        if p.get("via"):
            bits.append(p["via"][:4])
        bits.append(p["event"][:3])
        bits.append("all" if p.get("window") is None else f"w{p['window']}")
        bits.append(p["agg"])
        if p.get("pool"):
            bits.append(p["pool"][:3])
        if p.get("norm"):
            bits.append(p["norm"])
        return "_".join(bits)
    return f"{p['kind']}__{name(p['a'])}__{name(p['b'])}"


def canon(p):
    return json.dumps(p, sort_keys=True, separators=(",", ":"))



# Item-level components that every decomposable aggregate can be rebuilt from.
# Aggregating these over the items carrying an attribute value gives exactly the
# same number as grouping the raw events by that value, but costs one pass over
# 16k items instead of a join against millions of events.
def _item_components(window):
    half = (window or 60) / 2
    return [
        pl.len().cast(pl.Float64).alias("cnt"),
        _PCT.sum().alias("sum_pct"),
        (_PCT ** 2).sum().alias("sumsq_pct"),
        _PCT.max().alias("max_pct"),
        _DUR.sum().alias("sum_dur"),
        pl.col("age").min().cast(pl.Float64).alias("min_age"),
        pl.col("age").sum().cast(pl.Float64).alias("sum_age"),
        (pl.col("age") < half).sum().cast(pl.Float64).alias("cnt_recent"),
        pl.col("user_id").n_unique().cast(pl.Float64).alias("nu_user"),
        pl.col("age").n_unique().cast(pl.Float64).alias("nu_day"),
    ]


_COMBINE = {
    "cnt": "sum", "sum_pct": "sum", "sumsq_pct": "sum", "max_pct": "max",
    "sum_dur": "sum", "min_age": "min", "sum_age": "sum", "cnt_recent": "sum",
}

DECOMPOSABLE = {"count", "nuniq_item", "mean_pct", "max_pct", "std_pct",
                "sum_dur", "mean_dur", "recency", "mean_age", "trend"}


def _derive(agg, window):
    c = pl.col("cnt")
    if agg == "count":
        return c
    if agg == "nuniq_item":
        return pl.col("n_items")
    if agg == "mean_pct":
        return pl.col("sum_pct") / c
    if agg == "max_pct":
        return pl.col("max_pct")
    if agg == "std_pct":
        var = (pl.col("sumsq_pct") - pl.col("sum_pct") ** 2 / c) / (c - 1)
        return pl.when(c > 1).then(var.clip(lower_bound=0).sqrt()).otherwise(None)
    if agg == "sum_dur":
        return pl.col("sum_dur")
    if agg == "mean_dur":
        return pl.col("sum_dur") / c
    if agg == "recency":
        return pl.col("min_age")
    if agg == "mean_age":
        return pl.col("sum_age") / c
    if agg == "trend":
        return pl.col("cnt_recent") / (c - pl.col("cnt_recent") + 1.0)
    raise InvalidProgram(f"{agg} is not decomposable")


class FoldEngine:
    """Compiles DSL programs against one fold.

    `events` must already be restricted to ts < T by the caller; the constructor
    asserts it, and that single assertion is what makes every program in this
    language leak-free. User-keyed aggregates read only the sampled users'
    histories, item-keyed ones read the whole population, which is what a live
    system would have available.
    """

    # Grouped tables are kept only while they are small. An item-level table is
    # 16k rows and expensive to rebuild, so caching it is pure gain; a
    # (user, keyword) table is millions of rows and is what exhausts memory.
    MAX_CACHED_ROWS = 300_000
    MAX_CACHED_JOIN_ROWS = 1_200_000

    def __init__(self, events, T, item_attrs, users=None, join_cache=6, grp_cache=72,
                 user_attrs=None):
        if events.height and events["ts"].max() >= T:
            raise AssertionError("engine was given events at or after the prediction date")
        self.T = T
        self.attrs = item_attrs.with_columns(pl.col("item_id").cast(pl.Int32))
        self.join_cache = join_cache
        self.grp_cache = grp_cache
        # narrow dtypes and no date column: `age` carries the same information
        # and the whole log has to stay resident for every fold at once
        base = events.select(
            pl.col("user_id").cast(pl.Int32),
            pl.col("item_id").cast(pl.Int32),
            (pl.lit(T) - pl.col("ts")).dt.total_days().cast(pl.Int32).alias("age"),
            pl.col("watched_pct").cast(pl.Float32),
            pl.col("total_dur").cast(pl.Int32),
            pl.col("is_watch"), pl.col("is_finish"),
        )
        own = base if users is None else base.filter(
            pl.col("user_id").is_in(pl.Series(users).cast(pl.Int32)))
        # only the full log is kept for the global side; the watch and finish
        # subsets are taken on demand and the results are what gets cached
        self.base = base
        self.u = {"start": own, "watch": own.filter(pl.col("is_watch")),
                  "finish": own.filter(pl.col("is_finish"))}
        self.hist_days = max(1, int(base["age"].max()) if base.height else 1)
        self.user_attrs = user_attrs
        self._attr_cache = {}
        self._own_attr = {}
        self._istats = {}
        self._grp = {}
        self._seg = {}
        self.calls = 0

    @property
    def g(self):
        return _GlobalView(self)

    def _global(self, event):
        if event == "start":
            return self.base
        return self.base.filter(pl.col("is_watch" if event == "watch" else "is_finish"))

    def _attr_map(self, via):
        if via not in self._attr_cache:
            self._attr_cache[via] = (self.attrs.filter(pl.col("attr") == via)
                                     .select("item_id", "value"))
        return self._attr_cache[via]

    def _own_joined(self, event, via):
        key = (event, via)
        if key in self._own_attr:
            return self._own_attr[key]
        joined = self.u[event].join(self._attr_map(via), on="item_id")
        if joined.height <= self.MAX_CACHED_JOIN_ROWS:
            if len(self._own_attr) >= self.join_cache:
                self._own_attr.pop(next(iter(self._own_attr)))
            self._own_attr[key] = joined
        return joined

    def _item_stats(self, event, window):
        key = (event, window)
        if key not in self._istats:
            if len(self._istats) >= self.grp_cache:
                self._istats.pop(next(iter(self._istats)))
            df = self._global(event)
            if window is not None:
                df = df.filter(pl.col("age") < window)
            self._istats[key] = df.group_by("item_id").agg(_item_components(window))
        return self._istats[key]

    def _grouped(self, anchor, via, event, window, agg):
        key = (anchor, via, event, window, agg)
        if key in self._grp:
            return self._grp[key]

        if anchor == "item":
            st = self._item_stats(event, window)
            if agg == "nuniq_user":
                out = st.select("item_id", pl.col("nu_user").alias("v"))
            elif agg == "nuniq_day":
                out = st.select("item_id", pl.col("nu_day").alias("v"))
            else:
                out = st.select("item_id", _derive(agg, window).alias("v"))
        elif anchor == "attr_item" and agg in DECOMPOSABLE:
            st = self._item_stats(event, window).join(self._attr_map(via), on="item_id")
            agg_exprs = [getattr(pl.col(c), op)().alias(c) for c, op in _COMBINE.items()]
            agg_exprs.append(pl.len().cast(pl.Float64).alias("n_items"))
            out = (st.group_by("value").agg(agg_exprs)
                     .select("value", _derive(agg, window).alias("v")))
        else:
            df = (self._own_joined(event, via) if (anchor == "attr")
                  else self.u[event] if anchor == "user"
                  else self._global(event).join(self._attr_map(via), on="item_id"))
            if window is not None:
                df = df.filter(pl.col("age") < window)
            keys = {"user": ["user_id"], "attr": ["user_id", "value"],
                    "attr_item": ["value"]}[anchor]
            out = df.group_by(keys).agg(AGG_EXPR[agg](window).alias("v"))

        out = out.with_columns(pl.col("v").cast(pl.Float32))
        if out.height <= self.MAX_CACHED_ROWS:
            if len(self._grp) >= self.grp_cache:
                self._grp.pop(next(iter(self._grp)))
            self._grp[key] = out
        return out

    def _denominator(self, p, cand):
        # shares are always against the total number of events of that entity in
        # the same window, which keeps the quantity well defined whatever the
        # numerator's aggregate is
        if p.get("norm") == "share_user":
            g = self._grouped("user", None, p["event"], p.get("window"), "count")
            return cand.join(g.rename({"v": "den"}), on="user_id", how="left")["den"]
        g = self._grouped("item", None, p["event"], p.get("window"), "count")
        return cand.join(g.rename({"v": "den"}), on="item_id", how="left")["den"]

    def _eval_agg(self, p, cand):
        self.calls += 1
        a, via, pool = p["anchor"], p.get("via"), p.get("pool")
        g = self._grouped(a, via, p["event"], p.get("window"), p["agg"])
        if a == "user":
            s = cand.join(g, on="user_id", how="left")["v"]
        elif a == "item":
            s = cand.join(g, on="item_id", how="left")["v"]
        else:
            on = ["user_id", "value"] if a == "attr" else ["value"]
            pooled = (
                cand.with_row_index("_rid")
                .join(self._attr_map(via), on="item_id")
                .join(g, on=on, how="left")
                .group_by("_rid")
                .agg(getattr(pl.col("v"), pool)().alias("v"))
            )
            s = (cand.with_row_index("_rid")
                     .join(pooled, on="_rid", how="left")
                     .sort("_rid")["v"])
        s = s.cast(pl.Float32)

        nrm = p.get("norm")
        if nrm == "log1p":
            s = s.clip(lower_bound=0).log1p()
        elif nrm == "per_day":
            s = s / float(p.get("window") or self.hist_days)
        elif nrm in ("share_user", "share_item"):
            d = self._denominator(p, cand).cast(pl.Float32)
            s = s / (d.fill_null(0.0) + 1.0)
        elif nrm in ("z_in_group", "rank_in_group"):
            tmp = cand.select("user_id").with_columns(s.alias("v"))
            if nrm == "z_in_group":
                s = tmp.select(
                    ((pl.col("v") - pl.col("v").mean().over("user_id"))
                     / (pl.col("v").std().over("user_id") + 1e-6)).alias("v")
                )["v"]
            else:
                s = tmp.select(
                    (pl.col("v").rank("average", descending=True).over("user_id")
                     / pl.len().over("user_id")).alias("v")
                )["v"]
        return s.cast(pl.Float32)

    def _item_table(self, inner):
        """Item-level values of the inner program, as a joinable table."""
        g = self._grouped("item", None, inner["event"], inner.get("window"), inner["agg"])
        nrm = inner.get("norm")
        if nrm == "log1p":
            g = g.with_columns(pl.col("v").clip(lower_bound=0).log1p())
        elif nrm == "per_day":
            g = g.with_columns(pl.col("v") / float(inner.get("window") or self.hist_days))
        elif nrm == "share_item":
            den = self._grouped("item", None, inner["event"], inner.get("window"), "count")
            g = (g.join(den.rename({"v": "den"}), on="item_id", how="left")
                  .with_columns(pl.col("v") / (pl.col("den").fill_null(0.0) + 1.0))
                  .drop("den"))
        return g

    def _decay_grouped(self, anchor, via, event, halflife):
        key = ("decay", anchor, via, event, round(float(halflife), 4))
        if key in self._grp:
            return self._grp[key]
        lam = math.log(2.0) / float(halflife)
        df = (self._own_joined(event, via) if anchor == "attr"
              else self.u[event] if anchor == "user"
              else self._global(event) if anchor == "item"
              else self._global(event).join(self._attr_map(via), on="item_id"))
        keys = {"user": ["user_id"], "item": ["item_id"],
                "attr": ["user_id", "value"], "attr_item": ["value"]}[anchor]
        out = (df.group_by(keys)
                 .agg((pl.col("age").cast(pl.Float64) * (-lam)).exp().sum().alias("v"))
                 .with_columns(pl.col("v").cast(pl.Float32)))
        if out.height <= self.MAX_CACHED_ROWS:
            if len(self._grp) >= self.grp_cache:
                self._grp.pop(next(iter(self._grp)))
            self._grp[key] = out
        return out

    def _eval_decay(self, p, cand):
        self.calls += 1
        a, via, pool = p["anchor"], p.get("via"), p.get("pool")
        g = self._decay_grouped(a, via, p["event"], p["halflife"])
        if a == "user":
            s = cand.join(g, on="user_id", how="left")["v"]
        elif a == "item":
            s = cand.join(g, on="item_id", how="left")["v"]
        else:
            on = ["user_id", "value"] if a == "attr" else ["value"]
            pooled = (cand.with_row_index("_rid")
                      .join(self._attr_map(via), on="item_id")
                      .join(g, on=on, how="left")
                      .group_by("_rid")
                      .agg(getattr(pl.col("v"), pool)().alias("v")))
            s = (cand.with_row_index("_rid").join(pooled, on="_rid", how="left")
                 .sort("_rid")["v"])
        s = s.cast(pl.Float32)
        nrm = p.get("norm")
        if nrm == "log1p":
            s = s.clip(lower_bound=0).log1p()
        elif nrm in ("z_in_group", "rank_in_group"):
            tmp = cand.select("user_id").with_columns(s.alias("v"))
            if nrm == "z_in_group":
                s = tmp.select(((pl.col("v") - pl.col("v").mean().over("user_id"))
                                / (pl.col("v").std().over("user_id") + 1e-6)).alias("v"))["v"]
            else:
                s = tmp.select((pl.col("v").rank("average", descending=True).over("user_id")
                                / pl.len().over("user_id")).alias("v"))["v"]
        return s.cast(pl.Float32)

    def _seg_table(self, dim, event, window, agg):
        key = (dim, event, window, agg)
        if key in self._seg:
            return self._seg[key]
        if self.user_attrs is None:
            raise InvalidProgram("segment programs need the user attribute table")
        umap = self.user_attrs.select(
            pl.col("user_id").cast(pl.Int32), pl.col(dim).cast(pl.Utf8).alias("seg"))
        df = self._global(event)
        if window is not None:
            df = df.filter(pl.col("age") < window)
        df = df.join(umap, on="user_id", how="left").with_columns(
            pl.col("seg").fill_null("__na__"))
        out = (df.group_by("seg", "item_id")
                 .agg(AGG_EXPR[agg](window).alias("v"))
                 .with_columns(pl.col("v").cast(pl.Float32)))
        if out.height <= self.MAX_CACHED_ROWS:
            if len(self._seg) >= 24:
                self._seg.pop(next(iter(self._seg)))
            self._seg[key] = out
        return out

    def _eval_segment(self, p, cand):
        self.calls += 1
        g = self._seg_table(p["dim"], p["event"], p.get("window"), p["agg"])
        umap = self.user_attrs.select(
            pl.col("user_id").cast(pl.Int32), pl.col(p["dim"]).cast(pl.Utf8).alias("seg"))
        c = cand.join(umap, on="user_id", how="left").with_columns(
            pl.col("seg").fill_null("__na__"))
        s = c.join(g, on=["seg", "item_id"], how="left")["v"].cast(pl.Float32)
        nrm = p.get("norm")
        if nrm == "log1p":
            s = s.clip(lower_bound=0).log1p()
        elif nrm == "share_item":
            den = self._grouped("item", None, p["event"], p.get("window"), "count")
            d = cand.join(den.rename({"v": "den"}), on="item_id", how="left")["den"]
            s = s / (d.fill_null(0.0).cast(pl.Float32) + 1.0)
        elif nrm in ("z_in_group", "rank_in_group"):
            tmp = cand.select("user_id").with_columns(s.alias("v"))
            if nrm == "z_in_group":
                s = tmp.select(((pl.col("v") - pl.col("v").mean().over("user_id"))
                                / (pl.col("v").std().over("user_id") + 1e-6)).alias("v"))["v"]
            else:
                s = tmp.select((pl.col("v").rank("average", descending=True).over("user_id")
                                / pl.len().over("user_id")).alias("v"))["v"]
        return s.cast(pl.Float32)

    def _eval_nested(self, p, cand):
        self.calls += 1
        itab = self._item_table(p["inner"])
        own = self.u[p["event"]]
        if p.get("window") is not None:
            own = own.filter(pl.col("age") < p["window"])
        prof = (own.join(itab, on="item_id")
                   .group_by("user_id")
                   .agg(getattr(pl.col("v"), p["outer"])().alias("prof")))
        joined = cand.join(prof, on="user_id", how="left")
        if p["compare"] == "profile":
            return joined["prof"].cast(pl.Float32)
        here = cand.join(itab, on="item_id", how="left")["v"].cast(pl.Float32)
        pr = joined["prof"].cast(pl.Float32)
        if p["compare"] == "diff":
            return (here - pr).cast(pl.Float32)
        return (here / (pr.fill_null(0.0) + 1.0)).cast(pl.Float32)

    def compute(self, progs, cand):
        cand = cand.select(pl.col("user_id").cast(pl.Int32), pl.col("item_id").cast(pl.Int32))
        cols = {}
        for p in progs:
            if p["kind"] == "decay":
                v = self._eval_decay(p, cand)
            elif p["kind"] == "segment":
                v = self._eval_segment(p, cand)
            elif p["kind"] == "nested":
                v = self._eval_nested(p, cand)
            elif p["kind"] == "agg":
                v = self._eval_agg(p, cand)
            else:
                a = self._eval_agg(p["a"], cand)
                b = self._eval_agg(p["b"], cand)
                if p["kind"] == "ratio":
                    v = a / (b.fill_null(0.0) + float(p.get("smooth", 1.0)))
                else:
                    v = a * b
            cols[name(p)] = v.fill_nan(None).cast(pl.Float32)
        return pl.DataFrame(cols)


class SlowEngine(FoldEngine):
    """Reference implementation: every aggregate is computed by grouping the raw
    event rows. Only used by the test that checks the fast path against it."""

    def _grouped(self, anchor, via, event, window, agg):
        key = ("slow", anchor, via, event, window, agg)
        if key in self._grp:
            return self._grp[key]
        own = anchor in ("user", "attr")
        df = self.u[event] if own else self._global(event)
        if window is not None:
            df = df.filter(pl.col("age") < window)
        if via is not None:
            df = df.join(self._attr_map(via), on="item_id")
        keys = {"user": ["user_id"], "item": ["item_id"],
                "attr": ["user_id", "value"], "attr_item": ["value"]}[anchor]
        out = (df.group_by(keys).agg(AGG_EXPR[agg](window).alias("v"))
                 .with_columns(pl.col("v").cast(pl.Float32)))
        self._grp[key] = out
        return out


class _GlobalView:
    """Keeps `engine.g[event]` working for callers that expect a mapping while
    the subsets are produced on demand rather than stored."""

    def __init__(self, engine):
        self._e = engine

    def __getitem__(self, event):
        return self._e._global(event)


# --------------------------------------------------------------------------
# Rendering programs as SQL.
#
# The task description frames hand-made features as SQL over the event table.
# A program in this language is exactly that, so it can be printed back as a
# query. Useful for two things: a reviewer can read a generated feature and see
# for themselves that the time filter only looks backwards, and the same query
# can be run against a warehouse without this codebase.
# --------------------------------------------------------------------------

_SQL_AGG = {
    "count": "COUNT(*)",
    "nuniq_item": "COUNT(DISTINCT e.item_id)",
    "nuniq_user": "COUNT(DISTINCT e.user_id)",
    "nuniq_day": "COUNT(DISTINCT e.ts)",
    "mean_pct": "AVG(e.watched_pct)",
    "max_pct": "MAX(e.watched_pct)",
    "std_pct": "STDDEV_SAMP(e.watched_pct)",
    "sum_dur": "SUM(e.total_dur)",
    "mean_dur": "AVG(e.total_dur)",
    "recency": "MIN(DATE_DIFF('day', e.ts, :T))",
    "mean_age": "AVG(DATE_DIFF('day', e.ts, :T))",
}
_SQL_EVENT = {"start": None, "watch": "e.watched_pct >= 30", "finish": "e.watched_pct >= 90"}


def _sql_agg(p, alias="v"):
    a, via, w, ag = p["anchor"], p.get("via"), p.get("window"), p["agg"]
    where = ["e.ts < :T"]
    if w is not None:
        where.append(f"e.ts >= :T - INTERVAL '{w}' DAY")
    if _SQL_EVENT[p["event"]]:
        where.append(_SQL_EVENT[p["event"]])

    if ag == "trend":
        half = (w or 60) / 2
        expr = (f"CAST(COUNT_IF(DATE_DIFF('day', e.ts, :T) < {half:g}) AS DOUBLE)"
                f" / (COUNT_IF(DATE_DIFF('day', e.ts, :T) >= {half:g}) + 1)")
    else:
        expr = _SQL_AGG[ag]

    joins, keys = [], []
    if a == "user":
        keys = ["e.user_id"]
    elif a == "item":
        keys = ["e.item_id"]
    else:
        joins.append(f"JOIN item_attr a ON a.item_id = e.item_id AND a.attr = '{via}'")
        keys = ["e.user_id", "a.value"] if a == "attr" else ["a.value"]

    sql = (f"SELECT {', '.join(keys)}, {expr} AS {alias}\n"
           f"FROM events e\n"
           + ("".join(f"{j}\n" for j in joins))
           + "WHERE " + "\n  AND ".join(where) + "\n"
           + f"GROUP BY {', '.join(keys)}")
    return sql, keys


def _join_on(p, alias):
    """How a sub-select attaches to the candidate table."""
    a = p["anchor"]
    if a == "user":
        return f"{alias}.user_id = c.user_id"
    if a == "item":
        return f"{alias}.item_id = c.item_id"
    if a == "attr":
        return (f"{alias}.user_id = c.user_id AND {alias}.value = ca.value"
                f"  -- ca: the candidate item's '{p['via']}' values, pooled with {p['pool'].upper()}")
    return (f"{alias}.value = ca.value"
            f"  -- ca: the candidate item's '{p['via']}' values, pooled with {p['pool'].upper()}")


def to_sql(p, indent=""):
    """A readable SQL rendering of a program. `:T` is the prediction date."""
    k = p["kind"]
    if k in ("ratio", "mul"):
        expr = "l.v / (r.v + 1)" if k == "ratio" else "l.v * r.v"
        needs_attr = any(q["anchor"] in ("attr", "attr_item") for q in (p["a"], p["b"]))
        attr_join = ""
        if needs_attr:
            via = next(q["via"] for q in (p["a"], p["b"])
                       if q["anchor"] in ("attr", "attr_item"))
            attr_join = f"\nJOIN item_attr ca ON ca.item_id = c.item_id AND ca.attr = '{via}'"
        return (f"-- {name(p)}\n"
                f"WITH l AS (\n{to_sql(p['a'])}\n), r AS (\n{to_sql(p['b'])}\n)\n"
                f"SELECT c.user_id, c.item_id, {expr} AS value\n"
                f"FROM candidates c{attr_join}\n"
                f"LEFT JOIN l ON {_join_on(p['a'], 'l')}\n"
                f"LEFT JOIN r ON {_join_on(p['b'], 'r')}")
    if k == "decay":
        a, via, w = p["anchor"], p.get("via"), None
        base, keys = _sql_agg({**p, "agg": "count", "window": None}, "v")
        lam = math.log(2.0) / float(p["halflife"])
        base = base.replace("COUNT(*)",
                            f"SUM(EXP(-{lam:.6g} * DATE_DIFF('day', e.ts, :T)))")
        return f"-- {name(p)}\n{base}"
    if k == "segment":
        return (f"-- {name(p)}\n"
                f"SELECT u.{p['dim']} AS seg, e.item_id, {_SQL_AGG.get(p['agg'], 'COUNT(*)')} AS v\n"
                f"FROM events e JOIN users u ON u.user_id = e.user_id\n"
                f"WHERE e.ts < :T"
                + (f"\n  AND e.ts >= :T - INTERVAL '{p['window']}' DAY" if p.get("window") else "")
                + (f"\n  AND {_SQL_EVENT[p['event']]}" if _SQL_EVENT[p["event"]] else "")
                + f"\nGROUP BY u.{p['dim']}, e.item_id")
    if k == "nested":
        inner, _ = _sql_agg(p["inner"], "iv")
        return (f"-- {name(p)}\n"
                f"WITH item_level AS (\n{inner}\n), profile AS (\n"
                f"  SELECT e.user_id, {p['outer'].upper()}(i.iv) AS prof\n"
                f"  FROM events e JOIN item_level i ON i.item_id = e.item_id\n"
                f"  WHERE e.ts < :T"
                + (f" AND e.ts >= :T - INTERVAL '{p['window']}' DAY" if p.get("window") else "")
                + "\n  GROUP BY e.user_id\n)\n"
                f"SELECT c.user_id, c.item_id, "
                + {"profile": "p.prof",
                   "diff": "i.iv - p.prof",
                   "ratio": "i.iv / (p.prof + 1)"}[p["compare"]]
                + " AS value\nFROM candidates c\n"
                "LEFT JOIN item_level i ON i.item_id = c.item_id\n"
                "LEFT JOIN profile p ON p.user_id = c.user_id")
    sql, keys = _sql_agg(p)
    note = ""
    if p.get("pool"):
        note = (f"\n-- then, per candidate (user, item): "
                f"{p['pool'].upper()} of v over the item's '{p['via']}' values")
    if p.get("norm"):
        note += f"\n-- then normalised with: {p['norm']}"
    return f"-- {name(p)}\n{sql}{note}"
