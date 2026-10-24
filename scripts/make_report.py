"""Assembles README.md from the written sections and the generated tables."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from tlab.config import REPORTS, ROOT

FIGURES = [
    ("fig_sample_efficiency.png",
     "Рост качества на валидации по мере расходования бюджета. Полоса — разброс по сидам."),
    ("fig_window_sweep.png",
     "Прирост от одного счётного признака при разных окнах, по отношениям. Звезда — оптимум."),
    ("fig_windows_chosen.png",
     "Распределение выбранных горизонтов: ручной набор против найденных поиском."),
    ("fig_half_lives.png",
     "Периоды полураспада, выученные градиентным спуском, против окон 7 и 30 дней."),
    ("fig_final_main.png",
     "Итоговое сравнение на двух отложенных неделях. Усы — 95% доверительный интервал парного бутстрэпа относительно ручного набора."),
    ("fig_by_history.png",
     "NDCG@10 в разбивке по длине истории пользователя."),
    ("fig_proxy_fidelity.png",
     "Насколько валидация на выборке поиска предсказывает тест на основной выборке."),
]

parts = []
for f in ("part1_method.md", "part2_research.md"):
    p = REPORTS / f
    if p.exists():
        parts.append(p.read_text().rstrip())

tables = REPORTS / "tables.md"
if tables.exists():
    parts.append("## Результаты\n" + tables.read_text().rstrip())

figs = ["\n## Рисунки\n"]
for name, cap in FIGURES:
    if (REPORTS / name).exists():
        figs.append(f"![{cap}](reports/{name})\n\n*{cap}*\n")
if len(figs) > 1:
    parts.append("\n".join(figs).rstrip())

for f in ("part3_analysis.md", "part4_repro.md"):
    p = REPORTS / f
    if p.exists():
        parts.append(p.read_text().rstrip())

(ROOT / "README.md").write_text("\n\n".join(parts) + "\n")
print(f"README.md: {len((ROOT / 'README.md').read_text().splitlines())} lines")
