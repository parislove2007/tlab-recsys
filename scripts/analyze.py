"""Three analyses over the saved artefacts.

Who the searched features help (by how much history a user has), what horizons
the two approaches actually chose, and whether the cheap proxy used inside the
search ranked the feature sets the same way the full-scale test does.
"""
import collections
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from tlab import dataset as D
from tlab.config import RUNS, REPORTS, TEST_DATES
from tlab.dsl import WINDOWS
from tlab.human import human_programs

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
out = {}

# ---------------------------------------------------------------- who benefits
WANT = ["BASE_main", "BASE_HUMAN_main", "AUTO-llm_recorded-s0_main",
        "BASE_HUMAN_AUTO-llm_recorded_main", "BASE_KERNEL_main"]
NICE = {"BASE_main": "базовые признаки",
        "BASE_HUMAN_main": "+ ручной набор",
        "AUTO-llm_recorded-s0_main": "+ найденный поиском",
        "BASE_HUMAN_AUTO-llm_recorded_main": "+ ручной и найденный",
        "BASE_KERNEL_main": "+ выученные горизонты"}
peruser = {}
for pat in ("peruser_*_main.parquet", "pufinal_*_main.parquet"):
    for f in RUNS.glob(pat):
        peruser[f.stem.split("_", 1)[1]] = f
peruser = {NICE.get(k, k): v for k, v in peruser.items() if k in WANT}
if len(peruser) >= 2:
    ev = D.load_events()
    hist = (D.history(ev, TEST_DATES[0]).group_by("user_id")
            .agg(pl.len().alias("n_hist")))
    edges = [8, 15, 30, 60, 10 ** 9]
    labels = ["5-7", "8-14", "15-29", "30-59", "60+"]

    rows, series = [], {}
    order = [NICE[k] for k in WANT if NICE.get(k) in peruser]
    for tag in order:
        path = peruser[tag]
        d = (pl.read_parquet(path).group_by("user_id").agg(pl.col("ndcg").mean())
             .join(hist, on="user_id", how="left").with_columns(pl.col("n_hist").fill_null(0)))
        buckets = []
        for lo, hi in zip([0] + edges[:-1], edges):
            sel = d.filter((pl.col("n_hist") >= max(lo, 5)) & (pl.col("n_hist") < hi))
            buckets.append((len(sel), float(sel["ndcg"].mean()) if len(sel) else float("nan")))
        series[tag] = buckets
        rows.append([tag] + [f"{m:.4f} (n={n})" for n, m in buckets])
    out["by_history"] = {"labels": labels, "series": {k: [(n, m) for n, m in v]
                                                      for k, v in series.items()}}

    if len(series) >= 2:
        fig, ax = plt.subplots(figsize=(6.8, 3.6))
        x = np.arange(len(labels))
        w = 0.8 / len(series)
        for i, tag in enumerate(order):
            b = series[tag]
            ax.bar(x + i * w - 0.4 + w / 2, [m for _, m in b], width=w, label=tag)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("длина истории пользователя на дату предсказания, событий")
        ax.set_ylabel("NDCG@10 на тесте")
        ax.set_title("Кому помогают признаки: разбивка по длине истории")
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(REPORTS / "fig_by_history.png")
        plt.close(fig)

# ------------------------------------------------------- which windows chosen
def windows_of(progs):
    c = collections.Counter()
    for p in progs:
        for q in ([p] if p["kind"] == "agg" else
                  [p[k] for k in ("a", "b", "inner") if isinstance(p.get(k), dict)]):
            if q.get("kind") == "agg":
                c[q.get("window")] += 1
    return c


chosen = {"ручной набор": windows_of(human_programs())}
for f in sorted(glob.glob(str(RUNS / "search_*.json"))):
    d = json.loads(pathlib.Path(f).read_text())
    chosen[d["strategy"]] = windows_of(d["accepted"])
out["windows_chosen"] = {k: {str(w): v.get(w, 0) for w in WINDOWS} for k, v in chosen.items()}

xs = [w if w is not None else 365 for w in WINDOWS]
fig, ax = plt.subplots(figsize=(7.0, 3.6))
for k, c in chosen.items():
    tot = max(1, sum(c.values()))
    ax.plot(xs, [c.get(w, 0) / tot for w in WINDOWS], marker="o", ms=3.5, lw=1.4, label=k)
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels(["1", "3", "7", "14", "30", "60", "90", "180", "вся"])
ax.set_xlabel("окно агрегации, дней")
ax.set_ylabel("доля признаков в наборе")
ax.set_title("Какие горизонты выбирает человек и какие — поиск")
ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(REPORTS / "fig_windows_chosen.png")
plt.close(fig)

# ------------------------------------------------------------- proxy fidelity
fin = RUNS / "final_main.json"
if fin.exists():
    d = json.loads(fin.read_text())
    pairs = [(v["search_val_ndcg"], v["test"]["ndcg@10"], k)
             for k, v in d["results"].items() if "search_val_ndcg" in v]
    if len(pairs) >= 3:
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        rho, pv = spearmanr(a, b)
        out["proxy_fidelity"] = {"n_sets": len(pairs), "spearman": float(rho),
                                 "p_value": float(pv),
                                 "points": [{"set": k, "search_val": x, "main_test": y}
                                            for x, y, k in pairs]}
        fig, ax = plt.subplots(figsize=(5.0, 4.0))
        ax.scatter(a, b, s=28)
        for x, y, k in pairs:
            ax.annotate(k.replace("AUTO:", ""), (x, y), fontsize=6,
                        textcoords="offset points", xytext=(4, 3))
        ax.set_xlabel("NDCG@10 на валидации, выборка поиска (10k юзеров)")
        ax.set_ylabel("NDCG@10 на тесте, основная выборка")
        ax.set_title(f"Верность дешёвого прокси, Спирмен {rho:+.2f}")
        fig.tight_layout()
        fig.savefig(REPORTS / "fig_proxy_fidelity.png")
        plt.close(fig)

(RUNS / "analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: (list(v)[:3] if isinstance(v, dict) else v) for k, v in out.items()},
                 indent=1, ensure_ascii=False)[:1200])
print("saved", RUNS / "analysis.json")
