#!/usr/bin/env bash
# Stage 1 (ComputeSE, shard-parallel) then stage 2 (ashr, chunk-parallel) per context.
# Resumable: both stages skip work whose output already exists.
set -u

PY=${PY:-/home/beraslan/miniconda/envs/py312/bin/python}
ROOT=${ROOT:-/home/beraslan/Projects/ModuleFinder}
SCREEN=${SCREEN:-/processed_datasets/VCI/ChemoGenetic_H1_Basak}
OUT=${OUT:-/large_storage/ctc/beraslan/ModuleFinder/computese}
ASHR=$ROOT/src/cryptex/de/run_ashr_on_chunk.R

SE_JOBS=${SE_JOBS:-3}       # ComputeSE is memory-bound (~60 GB/shard)
ASHR_JOBS=${ASHR_JOBS:-6}   # ashr is time-bound (~30-50 GB/shard)
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

for ctx in "$@"; do
  echo "=============================================================="
  echo "[$(date '+%F %T')] $ctx"
  echo "=============================================================="

  echo "--- stage 1: ComputeSE (${SE_JOBS} shards at a time) ---"
  # -u and --line-buffered: the previous run died mid-context and the log told us
  # nothing because both python and grep were block-buffering.
  $PY -u "$ROOT/scripts/compute_se_shards.py" \
      --screen-dir "$SCREEN" --out-dir "$OUT" --contexts "$ctx" \
      --n-control-cells 100000 --n-subsamples 10 --spacing linear ${SE_EXTRA:-} \
      --shard-jobs "$SE_JOBS" --summary "$OUT/summary_${ctx}.csv" \
      2>&1 | grep --line-buffered -vE "ImplicitModification|warnings.warn|utils.warn_names"
  echo "[$(date '+%F %T')] $ctx stage 1 exited (pipeline status: ${PIPESTATUS[0]:-?})"

  echo "--- stage 2: ashr (${ASHR_JOBS} chunks at a time) ---"
  todo=$(mktemp)
  for d in "$OUT/$ctx"/ad_*/; do
      [ -f "$d/chunk_000000.csv" ] && [ ! -f "$d/AshrResult_chunk_000000.csv" ] \
          && echo "$d/chunk_000000.csv"
  done > "$todo"
  echo "    $(wc -l < "$todo") chunks to shrink"
  [ -s "$todo" ] && xargs -a "$todo" -n 1 -P "$ASHR_JOBS" Rscript "$ASHR"
  echo "[$(date '+%F %T')] $ctx stage 2 exited (status $?)"
  rm -f "$todo"

  n_chunk=$(ls "$OUT/$ctx"/*/chunk_000000.csv 2>/dev/null | wc -l)
  n_ashr=$(ls "$OUT/$ctx"/*/AshrResult_chunk_000000.csv 2>/dev/null | wc -l)
  echo "[$(date '+%F %T')] $ctx done: $n_chunk chunks, $n_ashr ashr results"
done
echo "[$(date '+%F %T')] ALL CONTEXTS COMPLETE"
