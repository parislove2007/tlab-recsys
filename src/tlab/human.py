"""The hand-written feature set: what a competent engineer would ship.

Written in the same DSL as the searched features so the comparison is between
two ways of choosing feature specifications, not between two engines.
"""


def A(anchor, event, agg, window=None, via=None, pool=None, norm=None):
    return {"kind": "agg", "anchor": anchor, "via": via, "event": event,
            "window": window, "agg": agg, "norm": norm, "pool": pool}


def R(a, b):
    return {"kind": "ratio", "a": a, "b": b, "smooth": 1.0}


def human_programs():
    p = []

    # how active the user is, and how deeply they watch
    for w in (7, 30, None):
        p.append(A("user", "start", "count", w))
        p.append(A("user", "start", "nuniq_item", w))
    for w in (7, 30):
        p.append(A("user", "start", "count", w, norm="per_day"))
    for w in (7, 30, None):
        p.append(A("user", "watch", "count", w))
    p += [
        A("user", "finish", "count", 30),
        A("user", "finish", "count", None),
        A("user", "start", "nuniq_day", 30),
        A("user", "start", "nuniq_day", None),
        A("user", "start", "mean_pct", None),
        A("user", "start", "mean_pct", 30),
        A("user", "start", "max_pct", None),
        A("user", "start", "mean_dur", None),
        A("user", "start", "sum_dur", 30),
        A("user", "start", "recency", None),
        A("user", "start", "trend", 30),
        R(A("user", "watch", "count", 30), A("user", "start", "count", 30)),
        R(A("user", "finish", "count", None), A("user", "start", "count", None)),
    ]

    # how popular and how engaging the item is
    for w in (1, 7, 14, 30, 90, None):
        p.append(A("item", "start", "count", w))
    for w in (7, 30):
        p.append(A("item", "start", "count", w, norm="per_day"))
        p.append(A("item", "watch", "count", w))
    p += [
        A("item", "start", "nuniq_user", 7),
        A("item", "start", "nuniq_user", None),
        A("item", "start", "mean_pct", None),
        A("item", "start", "mean_dur", None),
        A("item", "start", "recency", None),
        A("item", "start", "trend", 14),
        A("item", "start", "count", 7, norm="z_in_group"),
        A("item", "start", "count", 7, norm="rank_in_group"),
        R(A("item", "watch", "count", 7), A("item", "start", "count", 7)),
        R(A("item", "finish", "count", 30), A("item", "start", "count", 30)),
    ]

    # indirect user-item links: the user's taste for what this item is made of
    for pool in ("sum", "max"):
        for w in (7, 30, None):
            p.append(A("attr", "start", "count", w, via="genres", pool=pool))
    p += [
        A("attr", "start", "count", None, via="genres", pool="sum", norm="share_user"),
        A("attr", "watch", "count", 30, via="genres", pool="sum"),
        A("attr", "start", "mean_pct", None, via="genres", pool="mean"),
        A("attr", "start", "recency", None, via="genres", pool="min"),
        A("attr", "start", "count", None, via="directors", pool="sum"),
        A("attr", "start", "count", None, via="directors", pool="max"),
        A("attr", "start", "count", None, via="actors", pool="sum"),
        A("attr", "start", "count", None, via="keywords", pool="sum"),
        A("attr", "start", "count", None, via="studios", pool="max"),
        A("attr", "start", "count", None, via="countries", pool="sum", norm="share_user"),
        A("attr", "start", "count", None, via="content_type", pool="max", norm="share_user"),
        A("attr", "start", "count", None, via="decade", pool="max", norm="share_user"),
        A("attr", "start", "count", None, via="age_rating", pool="max", norm="share_user"),
    ]

    # what the crowd is watching inside each attribute right now
    p += [
        A("attr_item", "start", "count", 7, via="genres", pool="mean"),
        A("attr_item", "start", "count", 30, via="genres", pool="mean"),
        A("attr_item", "start", "count", 7, via="genres", pool="max"),
        A("attr_item", "start", "count", 30, via="decade", pool="mean"),
        A("attr_item", "start", "nuniq_user", 7, via="genres", pool="mean"),
        R(A("attr_item", "watch", "count", 30, via="genres", pool="mean"),
          A("attr_item", "start", "count", 30, via="genres", pool="mean")),
    ]
    return p


def grid_programs(limit=480, seed=0):
    """The dumbest automation that could work: a systematic grid over the same
    relations a human would think of, every window and several aggregates, with
    no search at all. This is the reference that says whether an intelligent
    search is needed or whether one can simply compute everything and let the
    booster regularise.
    """
    import random as _random
    from .dsl import WINDOWS, validate, InvalidProgram, canon

    out = []
    for ev in ("start", "watch", "finish"):
        for w in WINDOWS:
            for ag in ("count", "nuniq_item", "mean_pct", "sum_dur", "recency", "trend"):
                out.append(A("user", ev, ag, w))
            for ag in ("count", "nuniq_user", "mean_pct", "sum_dur", "recency", "trend"):
                out.append(A("item", ev, ag, w))
            out.append(A("user", ev, "count", w, norm="per_day"))
            out.append(A("item", ev, "count", w, norm="per_day"))
            for via in ("genres", "directors", "keywords"):
                for pool in ("sum", "max"):
                    out.append(A("attr", ev, "count", w, via=via, pool=pool))
                out.append(A("attr", ev, "mean_pct", w, via=via, pool="mean"))
            for pool in ("mean", "max"):
                out.append(A("attr_item", ev, "count", w, via="genres", pool=pool))

    keep, seen = [], set()
    for p in out:
        try:
            validate(p)
        except InvalidProgram:
            continue
        c = canon(p)
        if c in seen:
            continue
        seen.add(c)
        keep.append(p)
    if len(keep) > limit:
        _random.Random(seed).shuffle(keep)
        keep = keep[:limit]
    return keep


def nested_programs():
    """The feature family the flat language cannot express.

    An aggregate of an aggregate: measure every item in the user's history with
    an item-level feature, summarise that into a profile, and compare the
    candidate against it. "Is this more popular than what this person usually
    watches" is a different question from "is this popular", and the flat
    language has no way to ask it.
    """
    def inner(agg, window, event="start", norm=None):
        return {"kind": "agg", "anchor": "item", "via": None, "event": event,
                "window": window, "agg": agg, "norm": norm, "pool": None}

    def N(inner_p, outer, compare, window=None, event="start"):
        return {"kind": "nested", "inner": inner_p, "outer": outer,
                "event": event, "window": window, "compare": compare}

    pop7 = inner("count", 7, norm="log1p")
    pop30 = inner("count", 30, norm="log1p")
    popall = inner("count", None, norm="log1p")
    qual = inner("mean_pct", None)
    conv = inner("count", 30, event="watch", norm="share_item")
    dur = inner("mean_dur", None)
    age = inner("mean_age", None)

    out = []
    for ip in (pop7, pop30, popall):
        out += [N(ip, "mean", "profile"), N(ip, "mean", "diff"), N(ip, "mean", "ratio"),
                N(ip, "max", "diff"), N(ip, "std", "profile")]
    for ip in (qual, conv, dur, age):
        out += [N(ip, "mean", "profile"), N(ip, "mean", "diff")]
    # a profile built from recent history only, and one from deep watches only
    out += [N(pop7, "mean", "diff", window=30), N(pop7, "mean", "diff", window=7),
            N(popall, "mean", "diff", window=30),
            N(qual, "mean", "diff", event="watch"),
            N(pop7, "mean", "diff", event="finish"),
            N(popall, "min", "diff"), N(popall, "max", "ratio")]
    return out


def segment_programs():
    """The second family the flat language cannot express: demand for the
    candidate among users demographically similar to this one."""
    out = []
    for dim in ("age", "income", "sex", "kids_flg"):
        for w in (7, 30, None):
            out.append({"kind": "segment", "dim": dim, "event": "start",
                        "window": w, "agg": "count", "norm": "share_item"})
        out.append({"kind": "segment", "dim": dim, "event": "start",
                    "window": 7, "agg": "count", "norm": "z_in_group"})
        out.append({"kind": "segment", "dim": dim, "event": "start",
                    "window": 30, "agg": "count", "norm": "rank_in_group"})
        out.append({"kind": "segment", "dim": dim, "event": "watch",
                    "window": 30, "agg": "count", "norm": "share_item"})
        out.append({"kind": "segment", "dim": dim, "event": "start",
                    "window": None, "agg": "mean_pct", "norm": None})
    for dim in ("age", "sex"):
        out.append({"kind": "segment", "dim": dim, "event": "start",
                    "window": 7, "agg": "nuniq_user", "norm": "z_in_group"})
        out.append({"kind": "segment", "dim": dim, "event": "start",
                    "window": 14, "agg": "trend", "norm": None})
    return out
