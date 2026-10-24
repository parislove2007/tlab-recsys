"""How much does dropping already-watched items buy, at a fixed list size?

KION holds one row per (user, item), so an item in the history can never be a
future target. This measures the two variants against the same denominator.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl
from tlab import dataset as D
from tlab.config import RUNS, TEST_DATES, N_CAND

users = D.sample_users(20000)
ev = D.load_events()
rows = []
for T in TEST_DATES:
    for drop_seen in (False, True):
        D._DROP_SEEN = drop_seen
        cand, stats, _, _, _ = D.build_fold(ev, users, T, n_cand=N_CAND,
                                            keep_only_positive_groups=False)
        rows.append({"T": str(T), "drop_seen": drop_seen, **stats})
        print(rows[-1], flush=True)
D._DROP_SEEN = True

summary = {}
for drop in (False, True):
    r = [x for x in rows if x["drop_seen"] == drop]
    summary["with_seen" if not drop else "without_seen"] = {
        "cand_recall": round(sum(x["cand_recall"] for x in r) / len(r), 4),
        "cand_pos": sum(x["cand_pos"] for x in r),
        "reachable_pos": sum(x["reachable_pos"] for x in r),
    }
(RUNS / "seen_exclusion.json").write_text(json.dumps(
    {"n_users": int(len(users)), "n_cand": N_CAND, "folds": rows, "summary": summary},
    indent=1, ensure_ascii=False))
print(json.dumps(summary, indent=1))
