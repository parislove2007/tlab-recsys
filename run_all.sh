#!/usr/bin/env bash
# Full reproduction. Expects an activated environment built from requirements.txt.
#
# The whole thing takes about six hours on an eight-core laptop. Run the stages
# one at a time: a single search run can reach 15-20 GB, and two of them at once
# will not fit in 30 GB.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
PY=${PY:-python}

mkdir -p data_raw
if [ ! -d data_raw/kion_train ]; then
  curl -L -o data_raw/kion_train.zip \
    https://storage.yandexcloud.net/itmo-recsys-public-data/kion_train.zip
  (cd data_raw && unzip -q -o kion_train.zip)
fi

$PY scripts/prepare.py
$PY scripts/build_candidates.py
$PY scripts/measure_seen_exclusion.py

$PY tests/test_engine_equivalence.py
$PY tests/test_no_leakage.py

for s in 0 1 2; do
  $PY scripts/run_search.py --strategy random    --seed $s
  $PY scripts/run_search.py --strategy evolution --seed $s
done
$PY scripts/run_search.py --strategy human_pool --seed 0 --generations 3 --block 23 --keep 23
$PY scripts/run_search.py --strategy evolution --base human --seed 0

# the Opus 5 arm replays the transcripts committed under runs/llm_opus5_none_s0
$PY scripts/run_search.py --strategy llm_recorded --seed 0 --transcripts runs/llm_opus5_none_s0

# the local-model arm needs an ollama server with qwen2.5-coder:7b-instruct
if curl -sf -m 3 http://127.0.0.1:11435/api/tags > /dev/null; then
  $PY scripts/run_search.py --strategy llm_local --seed 0 --generations 6
fi

$PY scripts/run_window_sweep.py --repeats 1
$PY scripts/run_kernels.py --scale search
$PY scripts/run_baselines.py --scale main
$PY scripts/measure_noise_floor.py --scale main --seeds 4

# the only scripts that read the test folds
$PY scripts/run_final.py --scale main
$PY scripts/run_nested_extension.py --scale main
$PY scripts/run_final.py --users 70000 --best-seed-only --skip BASE+GRID --out final_xl70k.json
$PY scripts/run_final.py --users 10000 --no-tune --only BASE,BASE+HUMAN,BASE+GRID --out final_gridref.json
$PY scripts/run_final.py --users 10000 --no-tune --only AUTO: --out final_autoref.json

$PY scripts/make_tables.py
$PY scripts/make_figures.py
$PY scripts/analyze.py
$PY scripts/show_features.py
$PY scripts/make_report.py
$PY scripts/make_pdf.py
