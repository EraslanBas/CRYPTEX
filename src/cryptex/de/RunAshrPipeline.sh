#!/usr/bin/env bash
# End-to-end SE → ashr → posterior-mean-matrix pipeline for one or more
# AnnData (.h5ad) inputs.
#
# Per input adata:
#   1. ComputeSE.py             : cell-level → per-pert mean_diff + se, chunked
#   2. run_ashr_on_chunk.R      : ashr adaptive shrinkage on each chunk (parallel)
#   3. MergeAshrRes.py          : merge pert/gene metadata back into ashr output
#   4. AshrGenerateMatrix.py    : pivot to wide pert × gene PosteriorMean matrix
#
# Usage:
#   ./RunAshrPipeline.sh /path/adata1.h5ad /path/adata2.h5ad ...
#       Output dirs land under $OUT_BASE/<basename_without_h5ad>/.
#
#   ./RunAshrPipeline.sh --dir /path/to/some_dir
#       Runs the pipeline on every *.h5ad found one level deep under that dir.
#
# Config (edit at the top or override via environment variables):
#   OUT_BASE          where per-adata output dirs are created
#   N_ASHR_JOBS       parallel R workers for the ashr step
#   GROUP_KEY         adata.obs column holding perturbation labels
#   CONTROL_LABEL     label denoting controls
#   LAYER             layer to score (empty = adata.X)
#   MIN_CELLS         minimum cells per perturbation
#   CHUNK_PERTS       perturbations per chunk CSV
#   MATRIX_FORMAT     parquet or csv (default: parquet)

set -euo pipefail

# -------------------- config (override via env) --------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/beraslan/miniconda/envs/py312/bin/python}"
RSCRIPT="${RSCRIPT:-Rscript}"

OUT_BASE="${OUT_BASE:-${HERE}/AshrPipelineOutputs}"
N_ASHR_JOBS="${N_ASHR_JOBS:-6}"

GROUP_KEY="${GROUP_KEY:-target_gene}"
CONTROL_LABEL="${CONTROL_LABEL:-non-targeting}"
LAYER="${LAYER:-}"
MIN_CELLS="${MIN_CELLS:-20}"
CHUNK_PERTS="${CHUNK_PERTS:-100}"
MATRIX_FORMAT="${MATRIX_FORMAT:-parquet}"

# -------------------- script paths --------------------
COMPUTE_SE="${HERE}/ComputeSE.py"
RUN_ASHR="${HERE}/run_ashr_on_chunk.R"
MERGE_ASHR="${HERE}/MergeAshrRes.py"
ASHR_MATRIX="${HERE}/AshrGenerateMatrix.py"

for f in "$COMPUTE_SE" "$RUN_ASHR" "$MERGE_ASHR" "$ASHR_MATRIX"; do
    [ -f "$f" ] || { echo "[error] missing pipeline script: $f" >&2; exit 1; }
done
chmod +x "$RUN_ASHR" 2>/dev/null || true

mkdir -p "$OUT_BASE"

# -------------------- collect input adatas --------------------
ADATAS=""
if [ "${1:-}" = "--dir" ]; then
    [ -d "${2:-}" ] || { echo "[error] --dir argument is not a directory: ${2:-}" >&2; exit 1; }
    ADATAS=$(find "$2" -maxdepth 1 -type f -name "*.h5ad" | sort | tr '\n' ' ')
else
    ADATAS="$*"
fi

if [ -z "$ADATAS" ]; then
    echo "Usage:"
    echo "  $0 /path/a1.h5ad /path/a2.h5ad ..."
    echo "  $0 --dir /path/to/dir_with_h5ad_files"
    exit 1
fi

echo "===================================================================="
echo "RunAshrPipeline"
echo "  out_base       : $OUT_BASE"
echo "  group_key      : $GROUP_KEY"
echo "  control_label  : $CONTROL_LABEL"
echo "  layer          : ${LAYER:-<adata.X>}"
echo "  min_cells      : $MIN_CELLS"
echo "  chunk_perts    : $CHUNK_PERTS"
echo "  ashr jobs      : $N_ASHR_JOBS"
echo "  matrix format  : $MATRIX_FORMAT"
echo "===================================================================="

# -------------------- per-adata pipeline --------------------
run_one_adata() {
    local adata="$1"
    local name; name="$(basename "$adata" .h5ad)"
    local out_dir="$OUT_BASE/$name"

    if [ ! -f "$adata" ]; then
        echo "[skip] not a file: $adata" >&2
        return 0
    fi

    mkdir -p "$out_dir"
    echo
    echo "--------------------------------------------------------------------"
    echo "[$(date '+%F %T')] processing $name"
    echo "  adata   : $adata"
    echo "  out_dir : $out_dir"
    echo "--------------------------------------------------------------------"

    # -------- step 1: ComputeSE --------
    if compgen -G "$out_dir/chunk_*.csv" > /dev/null; then
        echo "  [step 1] chunks already exist — skipping ComputeSE"
    else
        echo "  [step 1] ComputeSE.py"
        SE_CMD="$PYTHON $COMPUTE_SE \
            --adata $adata \
            --out-dir $out_dir \
            --group-key $GROUP_KEY \
            --control-label $CONTROL_LABEL \
            --min-cells $MIN_CELLS \
            --chunk-perts $CHUNK_PERTS"
        if [ -n "$LAYER" ]; then
            SE_CMD="$SE_CMD --layer $LAYER"
        fi
        eval "$SE_CMD"
    fi

    # -------- step 2: ashr per chunk (parallel) --------
    echo "  [step 2] run_ashr_on_chunk.R  (N_ASHR_JOBS=$N_ASHR_JOBS)"
    find "$out_dir" -maxdepth 1 -type f -name 'chunk_*.csv' -print0 \
        | sort -z \
        | xargs -0 -n 1 -P "$N_ASHR_JOBS" "$RSCRIPT" "$RUN_ASHR"

    # -------- step 3: merge metadata back into ashr outputs --------
    echo "  [step 3] MergeAshrRes.py"
    $PYTHON "$MERGE_ASHR" --dir "$out_dir" --no-combine

    # -------- step 4: pivot to wide posterior-mean matrix --------
    local ext="$MATRIX_FORMAT"
    [ "$ext" = "parquet" ] && ext="parquet" || ext="csv"
    local out_matrix="$out_dir/PosteriorMean_matrix_${name}.${ext}"
    echo "  [step 4] AshrGenerateMatrix.py → $out_matrix"
    $PYTHON "$ASHR_MATRIX" --dir "$out_dir" --out "$out_matrix" --format "$MATRIX_FORMAT"

    echo "  [done] $name"
}

for adata in $ADATAS; do
    run_one_adata "$adata"
done

echo
echo "===================================================================="
echo "[$(date '+%F %T')] All adata processed."
echo "Outputs under: $OUT_BASE"
echo "===================================================================="
