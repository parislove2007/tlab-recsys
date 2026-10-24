"""Figures for the report."""
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tlab.config import RUNS, REPORTS

plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
COLORS = {"random": "#8c8c8c", "evolution": "#1f77b4", "llm_local": "#ff7f0e",
          "llm_recorded": "#d62728", "human_pool": "#2ca02c", "grid": "#9467bd"}
NAMES = {"random": "случайный поиск", "evolution": "эволюционный поиск",
         "llm_local": "LLM-поиск (Qwen2.5-Coder 7B)", "llm_recorded": "LLM-поиск (Opus 5)",
         "human_pool": "ручной набор через тот же отбор (69 программ)"}


def load_searches(base="none"):
    """Runs grouped by strategy. A run that started from the hand-written set
    is a different experiment and is kept apart."""
    runs = {}
    for f in sorted(glob.glob(str(RUNS / "search_*.json"))):
        r = json.loads(pathlib.Path(f).read_text())
        if r.get("base", "none") != base:
            continue
        runs.setdefault(r["strategy"], []).append(r)
    return runs


def fig_sample_efficiency(runs):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for strat, lst in runs.items():
        grid = np.arange(0, max(t["programs"] for r in lst for t in r["val_trace"]) + 1, 16)
        curves = []
        for r in lst:
            x = np.array([t["programs"] for t in r["val_trace"]], dtype=float)
            y = np.array([t["val_ndcg"] for t in r["val_trace"]], dtype=float)
            curves.append(np.interp(grid, x, y, left=y[0], right=y[-1]))
        c = np.vstack(curves)
        col = COLORS.get(strat, "k")
        ax.plot(grid, c.mean(0), color=col, lw=1.8, label=f"{NAMES.get(strat, strat)} (n={len(lst)})")
        if len(lst) > 1:
            ax.fill_between(grid, c.min(0), c.max(0), color=col, alpha=0.15, lw=0)
    base = min(t["val_ndcg"] for lst in runs.values() for r in lst for t in r["val_trace"])
    ax.axhline(base, color="k", ls=":", lw=1, label="базовая модель без DSL-признаков")
    ax.set_xlabel("программ предложено и скомпилировано (единица бюджета)")
    ax.set_ylabel("NDCG@10 на валидации")
    ax.set_title("Эффективность расходования бюджета разными стратегиями поиска")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_sample_efficiency.png")
    plt.close(fig)


def fig_window_sweep():
    f = RUNS / "window_sweep.json"
    if not f.exists():
        return
    d = json.loads(f.read_text())
    rows = d["rows"]
    rels = list(dict.fromkeys(r["relation"] for r in rows))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xs = [1, 3, 7, 14, 30, 60, 90, 180, 365]
    labels = ["1", "3", "7", "14", "30", "60", "90", "180", "вся\nистория"]
    cmap = plt.get_cmap("tab10")
    for i, rel in enumerate(rels):
        r = [x for x in rows if x["relation"] == rel]
        y = np.array([next(z["gain"] for z in r if (z["window"] or 365) == w) for w in xs])
        sd = np.array([next(z.get("val_sd", 0.0) for z in r if (z["window"] or 365) == w)
                       for w in xs])
        ax.plot(xs, y, marker="o", ms=3.5, lw=1.4, color=cmap(i % 10), label=rel)
        if sd.any():
            ax.fill_between(xs, y - sd, y + sd, color=cmap(i % 10), alpha=0.12, lw=0)
        b = max(r, key=lambda z: z["gain"])
        ax.plot([b["window"] or 365], [b["gain"]], marker="*", ms=12,
                color=cmap(i % 10), mec="k", mew=0.4, zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.axvline(7, color="k", ls=":", lw=1)
    ax.axvline(30, color="k", ls=":", lw=1)
    ax.set_xlabel("окно агрегации, дней (пунктир — типичный ручной выбор 7 и 30)")
    ax.set_ylabel("прирост NDCG@10 на валидации")
    rep = d.get("repeats", 1)
    ax.set_title("Какое временное окно на самом деле нужно каждому отношению\n"
                 f"(звезда — оптимум; один признак добавляется к базовой модели, "
                 f"среднее по {rep} сидам бустинга)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_window_sweep.png")
    plt.close(fig)


REL_RU = {
    "user_start": "активность пользователя", "user_watch": "глубокие просмотры\nпользователя",
    "item_start": "спрос на объект", "item_watch": "глубокие просмотры\nобъекта",
    "item_finish": "досмотры объекта", "ugenre": "вкус к жанру",
    "udirector": "вкус к режиссёру", "ukeyword": "вкус к теме",
    "igenre": "спрос на жанр у всех",
}


def fig_half_lives():
    cands = sorted(glob.glob(str(RUNS / "kernels_*.json")))
    if not cands:
        return
    d = json.loads(pathlib.Path(cands[0]).read_text())
    hl = d["learned_half_lives"]
    names = list(hl)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(names))
    fast = [hl[n]["half_life_days"][0] for n in names]
    slow = [hl[n]["half_life_days"][1] for n in names]
    ax.barh(y - 0.19, fast, height=0.36, color="#1f77b4", label="быстрое ядро")
    ax.barh(y + 0.19, slow, height=0.36, color="#aec7e8", label="медленное ядро")
    for v, ls, lab in [(7, ":", "7 дней"), (30, "--", "30 дней")]:
        ax.axvline(v, color="k", ls=ls, lw=1)
        ax.text(v, len(names) - 0.4, lab, fontsize=7, ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([REL_RU.get(n, n) for n in names], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("выученный период полураспада, дней (log)")
    ax.set_title("Горизонт, который градиентный спуск выбирает сам,\nпротив окон, которые выбирает человек")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(REPORTS / "fig_half_lives.png")
    plt.close(fig)


ARM_RU = {
    "BASE": "базовые признаки (кандидаты и метаданные)",
    "BASE+HUMAN": "ручной набор, 69 программ",
    "BASE+GRID": "систематическая сетка, без поиска",
    "BASE+KERNEL": "выученные горизонты",
    "BASE+HUMAN+KERNEL": "ручной набор + выученные горизонты",
    "BASE+AUTO+KERNEL": "найденный поиском + выученные горизонты",
}
STRAT_RU = {"evolution": "эволюционный поиск", "random": "случайный поиск",
            "llm_recorded": "LLM-поиск, Opus 5", "llm_local": "LLM-поиск, 7B локально",
            "human_pool": "ручной набор, другой порядок колонок"}


def arm_label(k):
    if k in ARM_RU:
        return ARM_RU[k]
    if k.startswith("AUTO:"):
        _, strat, seed = k.split(":")
        return f"{STRAT_RU.get(strat, strat)}, сид {seed[1:]}"
    if k.startswith("HUMAN+SEARCH-ON-TOP:"):
        strat = k.split(":")[1]
        return f"ручной набор + поиск поверх него ({STRAT_RU.get(strat, strat)})"
    if k.startswith("BASE+HUMAN+AUTO:"):
        strat = k.split(":")[1]
        if strat == "human_pool":
            return "ручной набор сам с собой (вырожденный случай)"
        return f"ручной + найденный ({STRAT_RU.get(strat, strat)})"
    return k


def arm_colour(k, ref):
    if k == ref:
        return "#2ca02c"
    if k == "BASE":
        return "#bdbdbd"
    if "KERNEL" in k:
        return "#9467bd"
    if "GRID" in k:
        return "#8c564b"
    if "llm_recorded" in k:
        return "#d62728"
    if "llm_local" in k:
        return "#ff7f0e"
    if "evolution" in k:
        return "#1f77b4"
    if "random" in k:
        return "#7f7f7f"
    if "human_pool" in k:
        return "#17becf"
    return "#4c72b0"


def fig_final(path="final_main.json", noise=0.006):
    f = RUNS / path
    if not f.exists():
        return
    d = json.loads(f.read_text())
    res, pair = d["results"], d.get("paired_vs_reference", {})
    ref = d.get("reference", "BASE+HUMAN")
    order = sorted(res, key=lambda k: res[k]["test"]["ndcg@10"])
    vals = np.array([res[k]["test"]["ndcg@10"] for k in order])
    err = np.array([[0.0, 0.0] if k not in pair else
                    [max(0, pair[k]["delta"] - pair[k]["ci_low"]),
                     max(0, pair[k]["ci_high"] - pair[k]["delta"])] for k in order]).T

    fig, ax = plt.subplots(figsize=(10.4, 0.40 * len(order) + 1.9))
    y = np.arange(len(order))
    base = res[ref]["test"]["ndcg@10"] if ref in res else None
    if base is not None:
        ax.axvspan(base - noise, base + noise, color="#2ca02c", alpha=0.10, lw=0,
                   label=f"ручной набор ± измеренный шум ({noise:.3f})")
        ax.axvline(base, color="#2ca02c", lw=1.2, ls="--")
    ax.barh(y, vals, color=[arm_colour(k, ref) for k in order], height=0.64,
            xerr=err, error_kw=dict(lw=0.9, capsize=2, ecolor="#333333"))
    span = vals.max() - vals.min()
    for i, k in enumerate(order):
        ax.text(vals[i] + err[1][i] + span * 0.02, i,
                f"{vals[i]:.4f}   {res[k].get('n_programs', 0)} прогр. / "
                f"{res[k]['n_features']} призн.", va="center", fontsize=7.2)
    ax.set_yticks(y)
    ax.set_yticklabels([arm_label(k) for k in order], fontsize=8)
    ax.set_xlim(vals.min() - span * 0.08, vals.max() + span * 0.62)
    ax.set_xlabel("NDCG@10 на тесте, две отложенные недели")
    ax.set_title("Одинаковые кандидаты, одинаковое разбиение, одинаковые базовые признаки\n"
                 "усы — 95% интервал парного бутстрэпа относительно ручного набора",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS / f"fig_{pathlib.Path(path).stem}.png")
    plt.close(fig)


if __name__ == "__main__":
    runs = load_searches()
    if runs:
        fig_sample_efficiency(runs)
        print("fig_sample_efficiency.png")
    fig_window_sweep(); print("fig_window_sweep.png (if data)")
    fig_half_lives(); print("fig_half_lives.png (if data)")
    for p in ("final_main.json", "final_xl.json", "final_main_bm69.json"):
        fig_final(p)
    print("done ->", REPORTS)
