"""Leak tests.

The pipeline's claim is that a feature can only ever see events strictly before
the prediction date, and that this holds for every program in the language
rather than for a reviewed list of features. These tests attack that claim from
five directions: poisoning the event log with future rows, handing the engine
future rows directly, probing the window boundary on a synthetic log, checking
the candidate generator is a function of the past only, and destroying the
label to confirm nothing in the feature set encodes it.
"""
import datetime as dt
import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl

from tlab import dataset as D, space, rank
from tlab.config import DATA, TEST_DATES, TRAIN_DATES
from tlab.dsl import FoldEngine, name
from tlab.human import human_programs

T = TEST_DATES[0]
N_USERS = 1500


def _sample():
    ev = D.load_events()
    users = D.sample_users(N_USERS)
    cand = (pl.read_parquet(DATA / "folds" / f"cand_{T.isoformat()}.parquet")
            .filter(pl.col("user_id").is_in(pl.Series(users)))
            .select("user_id", "item_id").head(20000))
    return ev, users, cand


def _poison(ev, users):
    """Add rows the model must never be able to see: the target week repeated
    with extreme values, and events far in the future."""
    tgt = D.target(ev, T).filter(pl.col("user_id").is_in(pl.Series(users)))
    loud = tgt.with_columns(
        pl.lit(100.0).cast(pl.Float64).alias("watched_pct"),
        pl.lit(999999).cast(tgt.schema["total_dur"]).alias("total_dur"),
        pl.lit(True).alias("is_watch"), pl.lit(True).alias("is_finish"),
    )
    far = loud.with_columns((pl.col("ts") + dt.timedelta(days=100)).alias("ts"))
    return pl.concat([ev, loud, loud, far]).sort("ts")


def test_features_invariant_to_future_events():
    ev, users, cand = _sample()
    attrs = D.load_attrs()
    rng = random.Random(11)
    progs = human_programs() + [space.random_program(rng) for _ in range(150)]

    clean = FoldEngine(D.history(ev, T), T, attrs, users=users).compute(progs, cand)
    dirty = FoldEngine(D.history(_poison(ev, users), T), T, attrs, users=users).compute(progs, cand)

    assert clean.columns == dirty.columns
    for c in clean.columns:
        x, y = clean[c].to_numpy(), dirty[c].to_numpy()
        assert np.array_equal(np.isnan(x), np.isnan(y)), f"{c}: null pattern changed"
        assert np.array_equal(x[~np.isnan(x)], y[~np.isnan(y)]), f"{c}: values changed"


def test_engine_refuses_future_events():
    ev, users, _ = _sample()
    attrs = D.load_attrs()
    try:
        FoldEngine(ev, T, attrs, users=users)
    except AssertionError:
        return
    raise AssertionError("engine accepted an event log that reaches past T")


def test_window_boundary_is_half_open():
    """On a synthetic log with exactly one event per day, a window of w days
    must count the w-1 events at ages 1..w-1. The event at age 0 falls on the
    prediction date itself, which belongs to the target window, and the events
    after T must be invisible: both are checked here."""
    rows = []
    for age in range(-5, 40):
        rows.append({"user_id": 1, "item_id": 100 + age, "ts": T - dt.timedelta(days=age),
                     "total_dur": 10, "watched_pct": 100.0, "is_watch": True, "is_finish": True})
    ev = pl.DataFrame(rows).with_columns(pl.col("ts").cast(pl.Date))
    attrs = pl.DataFrame({"item_id": [1], "value": ["x"], "attr": ["genres"]})
    eng = FoldEngine(D.history(ev, T), T, attrs, users=np.array([1]))
    cand = pl.DataFrame({"user_id": [1], "item_id": [100]})
    for w, expect in [(1, None), (3, 2.0), (7, 6.0), (30, 29.0), (None, 39.0)]:
        p = {"kind": "agg", "anchor": "user", "via": None, "event": "start",
             "window": w, "agg": "count", "norm": None, "pool": None}
        got = eng.compute([p], cand)[name(p)][0]
        assert got == expect, f"window {w} counted {got}, expected {expect}"

    # the most recent visible event is yesterday, never today or later
    p = {"kind": "agg", "anchor": "user", "via": None, "event": "start",
         "window": None, "agg": "recency", "norm": None, "pool": None}
    assert eng.compute([p], cand)[name(p)][0] == 1.0


def test_candidates_are_a_function_of_the_past():
    ev, users, _ = _sample()
    a, _, _, _, _ = D.build_fold(ev, users, T, n_cand=50)
    b, _, _, _, _ = D.build_fold(_poison(ev, users), users, T, n_cand=50)
    ka = a.select("user_id", "item_id").sort(["user_id", "item_id"])
    kb = b.select("user_id", "item_id").sort(["user_id", "item_id"])
    assert ka.equals(kb), "candidate list changed when future events were injected"


def test_shuffled_labels_destroy_the_signal():
    """With labels permuted inside each group, no feature set should beat a
    random ordering of the candidates."""
    from tlab.pipeline import Bundle
    users = D.sample_users(6000)
    b = Bundle(users, verbose=False)
    parts = b.matrix(human_programs())
    rng = np.random.default_rng(0)
    for split in ("train", "val", "test"):
        out = []
        for Tf, X, cand in parts[split]:
            # an independent permutation inside every group: the number of
            # positives per user survives, any link to the item does not
            shuffled = (cand.with_columns(pl.Series("_r", rng.random(cand.height)))
                        .with_columns(pl.col("label").sort_by("_r").over("user_id").alias("label"))
                        .drop("_r"))
            out.append((Tf, X, shuffled))
        parts[split] = out

    model, _, _ = rank.train_rank(parts, num_boost_round=200, early_stopping=25)
    m, _, _ = rank.evaluate(model, b, parts, "test")

    rnd = []
    for Tf, X, cand in parts["test"]:
        df = cand.select("user_id", "label").with_columns(
            pl.Series("score", rng.random(cand.height)))
        from tlab.metrics import ranking_metrics
        rnd.append(ranking_metrics(df)[0]["ndcg@10"])
    chance = float(np.mean(rnd))
    assert m["ndcg@10"] < chance * 1.15, (
        f"model reached ndcg {m['ndcg@10']:.4f} on shuffled labels, chance is {chance:.4f}")
    return m["ndcg@10"], chance


if __name__ == "__main__":
    for fn in [test_engine_refuses_future_events, test_window_boundary_is_half_open,
               test_features_invariant_to_future_events,
               test_candidates_are_a_function_of_the_past]:
        fn()
        print(f"  ok  {fn.__name__}")
    got, chance = test_shuffled_labels_destroy_the_signal()
    print(f"  ok  test_shuffled_labels_destroy_the_signal (ndcg {got:.4f} vs chance {chance:.4f})")
