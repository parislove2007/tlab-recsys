"""Turns the run artefacts into the markdown tables the report quotes."""
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from tlab.config import RUNS, REPORTS, DATA

NAMES = {"random": "случайный поиск", "evolution": "эволюционный поиск",
         "llm_local": "LLM-поиск, Qwen2.5-Coder 7B (локально)",
         "llm_recorded": "LLM-поиск, Opus 5", "human_pool": "ручной набор, тот же протокол отбора"}
out = []


def h(t, lvl=2):
    out.append(f"\n{'#' * lvl} {t}\n")


def table(head, rows):
    out.append("| " + " | ".join(head) + " |")
    out.append("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    out.append("")


def f4(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


# ---- data and folds
fs = json.loads((DATA / "fold_stats.json").read_text())
h("Фолды и кандидаты")
table(["дата предсказания", "роль", "ранжируемых юзеров", "достижимых позитивов",
       "попало в кандидаты", "recall кандидатов", "групп с позитивом", "строк"],
      [[r["T"],
        "train" if r["T"] <= "2021-07-26" else ("val" if r["T"] == "2021-08-02" else "test"),
        f"{r['users_ranked']:,}".replace(",", " "),
        f"{r['reachable_pos']:,}".replace(",", " "),
        f"{r['cand_pos']:,}".replace(",", " "),
        f"{r['cand_recall']:.3f}",
        f"{r['groups_kept']:,}".replace(",", " "),
        f"{r['rows_kept']:,}".replace(",", " ")] for r in fs])

# ---- window sweep
wf = RUNS / "window_sweep.json"
if wf.exists():
    d = json.loads(wf.read_text())
    h("Какое окно нужно каждому отношению")
    rows = []
    for rel, b in d["best_per_relation"].items():
        r = [x for x in d["rows"] if x["relation"] == rel]
        g7 = next((x["gain"] for x in r if x["window"] == 7), None)
        g30 = next((x["gain"] for x in r if x["window"] == 30), None)
        rows.append([rel, b["window"] if b["window"] else "вся история", f"{b['gain']:+.4f}",
                     f"{g7:+.4f}" if g7 is not None else "—",
                     f"{g30:+.4f}" if g30 is not None else "—",
                     f"{(b['gain'] - max(g7 or -9, g30 or -9)):+.4f}"])
    table(["отношение", "лучшее окно, дней", "прирост NDCG@10",
           "прирост при 7 днях", "прирост при 30 днях", "что теряет ручной выбор"], rows)
    out.append(f"Базовая модель без DSL-признаков: NDCG@10 = {d['base_val_ndcg']:.4f} на валидации.\n")

# ---- search efficiency
runs = {}
for f in sorted(glob.glob(str(RUNS / "search_*.json"))):
    r = json.loads(pathlib.Path(f).read_text())
    runs.setdefault(r["strategy"], []).append(r)
if runs:
    h("Стратегии поиска при равном бюджете")
    rows = []
    for strat, lst in sorted(runs.items()):
        best = [t["val_ndcg"] for r in lst for t in r["val_trace"][-1:]]
        at128, at256 = [], []
        for r in lst:
            tr = r["val_trace"]
            at128.append(max([t["val_ndcg"] for t in tr if t["programs"] <= 128] or [0]))
            at256.append(max([t["val_ndcg"] for t in tr if t["programs"] <= 256] or [0]))
        nf = [t["n_features"] for r in lst for t in r["val_trace"][-1:]]
        wall = [r["wall_seconds"] for r in lst]
        progs = [r["evaluator_cost"]["programs"] for r in lst]
        llm = [r["proposer_cost"] for r in lst]
        tok = sum(c.get("tokens_out", 0) for c in llm)
        rows.append([NAMES.get(strat, strat), len(lst), int(np.mean(progs)),
                     f"{np.mean(at128):.4f}", f"{np.mean(at256):.4f}",
                     f"{np.mean(best):.4f} ± {np.std(best):.4f}",
                     f"{np.mean(nf):.0f}", f"{np.mean(wall)/60:.1f}",
                     f"{tok/max(1,len(lst)):.0f}" if tok else "—"])
    table(["стратегия", "сидов", "программ", "val@128", "val@256", "val финал",
           "признаков", "мин. поиска", "ток. вывода LLM"], rows)

# ---- final comparison
for fname, title in [("final_main.json", "Итоговое сравнение на тесте (масштаб main)"),
                     ("final_xl.json", "Перенос на все 146 747 юзеров"),
                     ("final_main_bm69.json", "То же при равном числе программ (69)")]:
    f = RUNS / fname
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    h(title)
    res, pair = d["results"], d["paired_vs_reference"]
    rows = []
    for k in sorted(res, key=lambda x: -res[x]["test"]["ndcg@10"]):
        r = res[k]
        p = pair.get(k)
        rows.append([
            k, r.get("n_programs", 0), r["n_features"],
            f4(r["val"]["ndcg@10"]), f4(r["test"]["ndcg@10"]),
            f4(r["test"]["ndcg@10_abs"]), f4(r["test"]["map@10"]),
            f4(r["test"]["recall@10"]), f4(r["test"]["auc"]),
            f"{p['delta']:+.4f} [{p['ci_low']:+.4f}; {p['ci_high']:+.4f}]" if p else "— (эталон)",
            f"{r.get('gain_share_of_dsl_features', 0):.2f}",
            f"{r['feature_seconds']:.0f}/{r['train_seconds']:.0f}",
        ])
    table(["набор", "прогр.", "призн.", "val NDCG@10", "test NDCG@10", "test NDCG@10 абс.",
           "MAP@10", "Recall@10", "AUC", "Δ к BASE+HUMAN, 95% ДИ",
           "доля gain у DSL-призн.", "сек. призн./обуч."], rows)
    if "heuristics" in d:
        pass

bl = RUNS / "baselines_main.json"
if bl.exists():
    d = json.loads(bl.read_text())
    h("Опорные точки без обучения ранкера")
    table(["правило", "что это", "test NDCG@10", "test NDCG@10 абс.", "Recall@10"],
          [[k, v["desc"], f4(v["ndcg@10"]), f4(v["ndcg@10_abs"]), f4(v["recall@10"])]
           for k, v in d["heuristics"].items()])

# ---- kernels
for f in sorted(glob.glob(str(RUNS / "kernels_*.json"))):
    d = json.loads(pathlib.Path(f).read_text())
    h("Выученные временные горизонты")
    table(["отношение", "быстрое ядро, дней полураспада", "медленное ядро, дней"],
          [[k, v["half_life_days"][0], v["half_life_days"][1]]
           for k, v in d["learned_half_lives"].items()])
    h("Ядра внутри градиентного бустинга", 3)
    table(["набор", "признаков", "val NDCG@10", "test NDCG@10", "test абс."],
          [[k, v["n_features"], f4(v["val"]["ndcg@10"]), f4(v["test"]["ndcg@10"]),
            f4(v["test"]["ndcg@10_abs"])] for k, v in d["gbdt"].items()])
    out.append(f"Модель только на ядрах, без бустинга: test NDCG@10 = "
               f"{d['standalone']['test']['ndcg@10']:.4f}.\n")

REPORTS.mkdir(exist_ok=True)
(REPORTS / "tables.md").write_text("\n".join(out))
print("\n".join(out[:40]))
print(f"\n... saved {REPORTS / 'tables.md'}")
