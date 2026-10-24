from pathlib import Path
import datetime as dt

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data_raw" / "kion_train"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
for p in (DATA, RUNS, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

# Event taxonomy derived from watched_pct. The levels are nested
# (finish subset of watch subset of start), which mirrors the
# impression / click / purchase funnel the task description talks about.
WATCH_PCT = 30.0
FINISH_PCT = 90.0

TARGET_DAYS = 7

# Prediction dates. History for a fold is everything strictly before T,
# the label window is [T, T + TARGET_DAYS).
TRAIN_DATES = [dt.date(2021, 6, 28), dt.date(2021, 7, 5), dt.date(2021, 7, 12),
               dt.date(2021, 7, 19), dt.date(2021, 7, 26)]
VAL_DATES = [dt.date(2021, 8, 2)]
TEST_DATES = [dt.date(2021, 8, 9), dt.date(2021, 8, 16)]
ALL_DATES = TRAIN_DATES + VAL_DATES + TEST_DATES

# Users are sampled among those already active before the earliest fold, so
# the sampling rule never looks at anything later than the first history cut.
SAMPLE_CUTOFF = TRAIN_DATES[0]
MIN_HISTORY = 5

# Chosen so that the widest feature set (480 grid programs) still fits in the
# 30 GB of this machine once LightGBM has its own copy of the matrix.
SCALES = {
    "search": 10_000,    # cheap proxy used inside the feature-search loop
    "main": 40_000,      # the setting every reported number is measured on
    "xl": 100_000,       # scale-transfer check for the discovered features
}

N_CAND = 100

TOPK = 10
SEED = 42
