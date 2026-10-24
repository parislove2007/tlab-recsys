#!/usr/bin/env bash
# Full reproduction. Expects an activated environment built from requirements.txt.
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

$PY tests/test_engine_equivalence.py
$PY tests/test_no_leakage.py

for s in 0 1 2; do
  $PY scripts/run_search.py --strategy random    --seed $s
  $PY scripts/run_search.py --strategy evolution --seed $s
done
$PY scripts/run_search.py --strategy human_pool --seed 0 --generations 3 --block 23 --keep 23

# the Opus 5 arm replays the transcripts committed under runs/llm_opus5_none_s0
$PY scripts/run_search.py --strategy llm_recorded --seed 0 \
    --transcripts runs/llm_opus5_none_s0

# the local-model arm needs an ollama server with qwen2.5-coder:7b-instruct
if curl -sf -m 3 http://127.0.0.1:11435/api/tags > /dev/null; then
  for s in 0 1; do $PY scripts/run_search.py --strategy llm_local --seed $s; done
fi

$PY scripts/run_window_sweep.py
$PY scripts/run_kernels.py --scale search
$PY scripts/run_baselines.py --scale main
$PY scripts/run_final.py --scale main
$PY scripts/run_final.py --scale xl --only BASE,BASE+HUMAN,AUTO,BASE+HUMAN+AUTO --out final_xl.json

$PY scripts/make_tables.py
$PY scripts/make_figures.py
