"""Paired bootstrap against the reference arm, from the saved per-user metrics.

The comparison normally happens at the end of run_final; this recovers it when
that run was cut short, and it is exactly the same computation.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import polars as pl
from tlab.config import RUNS
from tlab.metrics import paired_bootstrap

scale = sys.argv[1] if len(sys.argv) > 1 else "main"
out_name = sys.argv[2] if len(sys.argv) > 2 else f"final_{scale}.json"

d = json.loads((RUNS / out_name).read_text())
ref = d.get("reference", "BASE+HUMAN")


def load(label):
    f = RUNS / f"pufinal_{label.replace('+', '_').replace(':', '-')}_{scale}.parquet"
    return pl.read_parquet(f) if f.exists() else None


base = load(ref)
if base is None:
    raise SystemExit(f"no per-user file for the reference arm {ref}")

pairs = {}
for label in d["results"]:
    if label == ref:
        continue
    cur = load(label)
    if cur is None:
        continue
    pairs[label] = paired_bootstrap(base, cur)
    p = pairs[label]
    print(f"  {label:34} {p['delta']:+.4f} [{p['ci_low']:+.4f}; {p['ci_high']:+.4f}] "
          f"доля побед {p['p_win']:.3f}  n={p['n_users']}")

d["paired_vs_reference"] = pairs
(RUNS / out_name).write_text(json.dumps(d, indent=1, ensure_ascii=False))
print(f"\n{len(pairs)} сравнений записано в {out_name}")
