"""The fast engine rebuilds item and attribute aggregates from decomposable
components. This checks it against the straightforward implementation."""
import random
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import polars as pl

from tlab import dataset as D, space
from tlab.config import DATA, TEST_DATES
from tlab.dsl import FoldEngine, SlowEngine, name


def test_fast_matches_slow():
    ev = D.load_events()
    attrs = D.load_attrs()
    users = D.sample_users(1500)
    T = TEST_DATES[0]
    hist = D.history(ev, T)
    cand = (pl.read_parquet(DATA / "folds" / f"cand_{T.isoformat()}.parquet")
            .filter(pl.col("user_id").is_in(pl.Series(users)))
            .select("user_id", "item_id").head(20000))

    fast = FoldEngine(hist, T, attrs, users=users)
    slow = SlowEngine(hist, T, attrs, users=users)

    rng = random.Random(7)
    progs = [space.random_program(rng) for _ in range(250)]
    a = fast.compute(progs, cand)
    b = slow.compute(progs, cand)

    bad = []
    for p in progs:
        c = name(p)
        x, y = a[c].to_numpy(), b[c].to_numpy()
        nx, ny = np.isnan(x), np.isnan(y)
        if not np.array_equal(nx, ny):
            bad.append((c, "null pattern"))
            continue
        d = np.abs(x[~nx] - y[~ny])
        scale = np.maximum(np.abs(y[~ny]), 1.0)
        if d.size and (d / scale).max() > 1e-4:
            bad.append((c, float((d / scale).max())))
    assert not bad, f"{len(bad)} of {len(progs)} programs disagree: {bad[:6]}"


if __name__ == "__main__":
    test_fast_matches_slow()
    print("fast engine matches the reference on 250 random programs")
