"""Prints the discovered features as SQL, for reading and for auditing."""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from tlab.config import RUNS, REPORTS
from tlab.dsl import canon, name, to_sql

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="search_llm_recorded_none_s0.json")
ap.add_argument("--top", type=int, default=12)
a = ap.parse_args()

d = json.loads((RUNS / a.run).read_text())
acc = sorted(d["accepted"], key=lambda p: -d["gains"].get(canon(p), 0.0))
lines = [f"# Признаки, найденные поиском ({d['strategy']}, сид {d['seed']})",
         "",
         f"Принято {len(d['accepted'])} программ из {d['evaluator_cost']['programs']} "
         f"предложенных. Ниже — {min(a.top, len(acc))} с наибольшим вкладом, в виде SQL. "
         "`:T` — дата предсказания; обратите внимание, что во всех запросах условие "
         "по времени строго одностороннее.", ""]
for p in acc[: a.top]:
    lines += [f"### `{name(p)}`",
              f"вклад по сплитам: {d['gains'].get(canon(p), 0.0):,.0f}".replace(",", " "),
              "", "```sql", to_sql(p), "```", ""]
(REPORTS / "discovered_features.md").write_text("\n".join(lines))
print("\n".join(lines[:40]))
print(f"... saved {REPORTS / 'discovered_features.md'}")
