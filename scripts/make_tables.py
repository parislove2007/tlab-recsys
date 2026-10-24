"""Turns the run artefacts into the markdown tables the report quotes."""
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from tlab import llm
from tlab.config import RUNS, REPORTS, DATA


def grammar_rate(run):
    """How much of what the model proposed survived the validator.

    Recomputed from the stored transcripts with the same parser the search
    used, so it is available even for runs recorded before the counter existed.
    """
    d = RUNS / f"{run['strategy']}_{run['base']}_s{run['seed']}"
    if run["strategy"] == "llm_recorded":
        d = RUNS / f"llm_opus5_{run['base']}_s{run['seed']}"
    if not d.is_dir():
        return None
    ok = bad = 0
    for f in sorted(d.glob("gen*_response.txt")):
        errs = []
        ok += len(llm.parse_programs(f.read_text(), errs))
        bad += len(errs)
    return (ok, bad) if ok + bad else None

REL_RU = {
    "user_start": "активность пользователя", "user_watch": "глубокие просмотры пользователя",
    "item_start": "спрос на объект", "item_watch": "глубокие просмотры объекта",
    "item_finish": "досмотры объекта", "ugenre": "вкус к жанру",
    "udirector": "вкус к режиссёру", "ukeyword": "вкус к теме",
    "igenre": "спрос на жанр у всех",
}
NAMES = {"random": "случайный поиск", "evolution": "эволюционный поиск",
         "llm_local": "LLM-поиск, Qwen2.5-Coder 7B (локально)",
         "llm_recorded": "LLM-поиск, Opus 5", "human_pool": "ручной набор, тот же протокол отбора"}
out = []


def h(t, lvl=3):
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
runs, on_human = {}, {}
for f in sorted(glob.glob(str(RUNS / "search_*.json"))):
    r = json.loads(pathlib.Path(f).read_text())
    (runs if r.get("base", "none") == "none" else on_human).setdefault(
        r["strategy"], []).append(r)
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
        llm_costs = [r["proposer_cost"] for r in lst]
        calls = sum(c.get("calls", 0) for c in llm_costs)
        rates = [grammar_rate(r) for r in lst]
        rates = [x for x in rates if x]
        ok = sum(x[0] for x in rates)
        bad = sum(x[1] for x in rates)
        final = (f"{np.mean(best):.4f} ± {np.std(best):.4f}" if len(lst) > 1
                 else f"{np.mean(best):.4f}")
        rows.append([NAMES.get(strat, strat), len(lst), int(np.mean(progs)),
                     f"{np.mean(at128):.4f}", f"{np.mean(at256):.4f}", final,
                     f"{np.mean(nf):.0f}", f"{np.mean(wall)/60:.1f}",
                     f"{calls / len(lst):.0f}" if calls else "—",
                     f"{bad / (ok + bad) * 100:.0f}%" if rates else "—"])
    table(["стратегия", "сидов", "программ", "val@128", "val@256", "val финал",
           "признаков", "мин. поиска", "вызовов LLM", "отбраковано грамматикой"], rows)
    out.append("Бюджет у всех стратегий одинаковый: 15 поколений по 32 программы "
               "и ровно 30 обучений прокси-модели. Ручной набор прогнан через тот же "
               "протокол, но у него всего 69 программ, поэтому его строка короче.\n")

if on_human:
    h("Поиск, начинающий с ручного набора")
    rows = []
    for strat, lst in sorted(on_human.items()):
        for r in lst:
            tr = r["val_trace"]
            rows.append([NAMES.get(strat, strat), r["seed"], tr[0]["val_ndcg"],
                         f"{tr[-1]['val_ndcg']:.4f}",
                         f"{tr[-1]['val_ndcg'] - tr[0]['val_ndcg']:+.4f}",
                         tr[-1]["n_features"], r["evaluator_cost"]["programs"]])
    table(["стратегия", "сид", "val с ручным набором", "val после поиска",
           "прирост", "добавлено признаков", "программ"], rows)

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
    h(f"Выученные временные горизонты (выборка {d['n_users']} юзеров)")
    table(["отношение", "быстрое ядро, дней полураспада", "медленное ядро, дней"],
          [[REL_RU.get(k, k), v["half_life_days"][0], v["half_life_days"][1]]
           for k, v in d["learned_half_lives"].items()])
    h(f"Ядра внутри градиентного бустинга (та же выборка {d['n_users']} юзеров, "
      "не сравнивать напрямую с основной таблицей)", 4)
    table(["набор", "признаков", "val NDCG@10", "test NDCG@10", "test абс."],
          [[k, v["n_features"], f4(v["val"]["ndcg@10"]), f4(v["test"]["ndcg@10"]),
            f4(v["test"]["ndcg@10_abs"])] for k, v in d["gbdt"].items()])
    out.append(f"Модель только на ядрах, без бустинга: test NDCG@10 = "
               f"{d['standalone']['test']['ndcg@10']:.4f}.\n")

# ---- scale transfer
xl = [("runs/final_xl70k.json", "70 000"), ("runs/final_xl100k_partial.json", "100 000")]
rows40 = {}
if (RUNS / "final_main.json").exists():
    rows40 = {k: v["test"]["ndcg@10"]
              for k, v in json.loads((RUNS / "final_main.json").read_text())["results"].items()}
have = [(pathlib.Path(f), lbl) for f, lbl in xl if (REPORTS.parent / f).exists()]
if have and rows40:
    h("Переносится ли найденный набор на большую выборку")
    cols, data = [], {}
    for f, lbl in have:
        d = json.loads((REPORTS.parent / f).read_text())
        cols.append(lbl)
        for k, v in d["results"].items():
            data.setdefault(k, {})[lbl] = v["test"]["ndcg@10"]
    names = [k for k in data if k in rows40]
    table(["набор", "40 000 (основная)"] + [f"{c} юзеров" for c in cols],
          [[k, f"{rows40[k]:.4f}"] + [f"{data[k].get(c, float('nan')):.4f}" for c in cols]
           for k in names])
    out.append("Выборки вложены не полностью, поэтому колонки сравнивать между собой "
               "нужно только внутри одной строки относительно ручного набора. "
               "Полный набор из 16 плеч на 70 000 и 100 000 не поместился в память "
               "(процесс доходил до 25 ГБ), поэтому здесь только четыре ключевых: "
               "база, ручной набор, лучший найденный и контроль на шум.\n")

# ---- brute-force reference
gr = RUNS / "final_gridref.json"
if gr.exists():
    d = json.loads(gr.read_text())
    keep = {k: v for k, v in d["results"].items() if k in ("BASE", "BASE+HUMAN", "BASE+GRID")}
    if "BASE+GRID" in keep:
        h("Нужен ли поиск, если можно посчитать всё подряд")
        table(["набор", "прогр.", "призн.", "val NDCG@10", "test NDCG@10", "призн., с"],
              [[k, v.get("n_programs", 0), v["n_features"], f4(v["val"]["ndcg@10"]),
                f4(v["test"]["ndcg@10"]), f"{v['feature_seconds']:.0f}"]
               for k, v in keep.items()])
        out.append(f"Измерено на поисковой выборке {d['n_users']} пользователей, где "
                   "сетка помещается по времени и памяти. Сетка содержит те же 480 "
                   "программ, что составляют бюджет одной стратегии поиска, но "
                   "перечисленные систематически, без отбора.\n")

# ---- cost of each approach
fin = RUNS / "final_main.json"
if fin.exists() and runs:
    d = json.loads(fin.read_text())
    h("Вычислительные затраты")
    rows = []
    label_for = {"BASE+HUMAN": "ручной набор"}
    for k, v in d["results"].items():
        if k == "BASE":
            continue
        src = None
        if k.startswith("AUTO:"):
            src = k.split(":")[1]
        name_ru = (NAMES.get(src, k) if src else label_for.get(k, k))
        search_min = v.get("search_wall_seconds")
        ec = v.get("evaluator_cost") or {}
        pc = v.get("proposer_cost") or {}
        rows.append([
            name_ru + (f" (сид {v['seed']})" if "seed" in v else ""),
            v.get("n_programs", 0), v["n_features"],
            f"{search_min / 60:.1f}" if search_min else "—",
            ec.get("programs", "—"),
            f"{pc.get('calls', 0)}/{pc.get('tokens_out', 0) or '—'}" if pc else "—",
            f"{v['feature_seconds']:.0f}", f"{v['train_seconds']:.0f}",
            v.get("best_iteration", "—"),
        ])
    table(["подход", "прогр.", "призн.", "поиск, мин", "программ оценено",
           "вызовов LLM / ток.", "призн., с", "обучение, с", "деревьев"], rows)
    out.append("Счёт признаков и обучение измерены на основной выборке, поиск — "
               "на дешёвой выборке 10 000 пользователей. Время счёта признаков — "
               "это и есть стоимость подхода в продакшене: его платят на каждом "
               "перестроении витрины.\n")

# ---- language extension
ext = RUNS / "nested_main.json"
if ext.exists():
    d = json.loads(ext.read_text())
    h("Расширение языка: то, что плоская грамматика не выражает")
    out.append(f"Добавлено {d['n_nested']} программ двух семейств — агрегат от агрегата "
               "(«популярнее ли этот объект того, что пользователь обычно смотрит») и "
               "якорь по сегменту пользователя («популярно среди похожих по демографии»).\n")
    table(["набор", "прогр.", "призн.", "val NDCG@10", "test NDCG@10",
           "доля gain у новых семейств"],
          [[k, v["n_programs"], v["n_features"], f4(v["val"]["ndcg@10"]),
            f4(v["test"]["ndcg@10"]), f"{v.get('gain_share_of_nested', 0):.2f}"]
           for k, v in d["results"].items()])
    if d.get("paired"):
        table(["сравнение", "Δ NDCG@10", "95% ДИ", "доля побед"],
              [[k, f"{v['delta']:+.4f}", f"[{v['ci_low']:+.4f}; {v['ci_high']:+.4f}]",
                f"{v['p_win']:.3f}"] for k, v in d["paired"].items()])
    top = next((v["top_nested"] for v in d["results"].values() if v.get("top_nested")), [])
    if top:
        h("Самые полезные программы новых семейств", 4)
        table(["признак", "вклад по сплитам"],
              [[t["feature"], f"{t['gain']:,.0f}".replace(",", " ")] for t in top[:8]])

# ---- extra analyses
an = RUNS / "analysis.json"
if an.exists():
    d = json.loads(an.read_text())
    if "by_history" in d:
        h("Кому помогают признаки: разбивка по длине истории")
        labels = d["by_history"]["labels"]
        table(["набор"] + [f"{l} событий" for l in labels],
              [[k] + [f"{m:.4f}" if m == m else "—" for _, m in v]
               for k, v in d["by_history"]["series"].items()])
        n0 = next(iter(d["by_history"]["series"].values()))
        out.append("Размеры групп: " + ", ".join(
            f"{labels[i]} — {int(n)}" for i, (n, _) in enumerate(n0)) + ".\n")
    if "windows_chosen" in d:
        h("Распределение выбранных горизонтов")
        wk = ["1", "3", "7", "14", "30", "60", "90", "180", "None"]
        table(["набор"] + [w if w != "None" else "вся история" for w in wk],
              [[k] + [str(v.get(w, 0)) for w in wk] for k, v in d["windows_chosen"].items()])
    if "proxy_fidelity" in d:
        pf = d["proxy_fidelity"]
        h("Верность дешёвого прокси")
        out.append(f"Корреляция Спирмена между NDCG@10 на валидации выборки поиска "
                   f"(10 000 юзеров) и NDCG@10 на тесте основной выборки по "
                   f"{pf['n_sets']} найденным наборам: **{pf['spearman']:+.2f}** "
                   f"(p = {pf['p_value']:.3f}).\n")

REPORTS.mkdir(exist_ok=True)
(REPORTS / "tables.md").write_text("\n".join(out))
print("\n".join(out[:40]))
print(f"\n... saved {REPORTS / 'tables.md'}")
