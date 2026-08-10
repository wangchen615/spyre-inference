#!/usr/bin/env bash
# ==============================================================================
# run_matrix.sh -- the prefix-caching x kv-offload matrix, ALL on Yue's PoC branch.
#
# Runs ttft_evict.py four ways at one length, changing exactly one flag at a time.
# Everything else -- script, block size, pool sizing, engine config, code -- is
# held constant, which is what makes the comparison fair. (An earlier attempt
# compared against the phase-1 recompute stack; that had five simultaneous
# differences and was meaningless.)
#
#   cell 1  --no-prefix-caching --no-kv-offload   every run recomputes in full
#   cell 2  --prefix-caching    --no-kv-offload   FAIR CONTROL for cell 3
#   cell 3  --prefix-caching    --kv-offload      the reload path
#   cell 4  --no-prefix-caching --kv-offload      DEGENERATE, see below
#
# Cell 4 cannot produce a reload measurement. Offload tracks *reusable* blocks;
# with prefix caching off there are none, so nothing is ever stored. ttft_evict.py
# warns about exactly this. It is run anyway to confirm the warning empirically
# (expect stored=0 loaded=0), not because it yields a number.
#
# Usage:
#   ./run_matrix.sh                    # length 1024, runs 5
#   LENGTH=4096 RUNS=5 ./run_matrix.sh
# ==============================================================================
set -uo pipefail

NEW=/tmp/work-kvoffload
LENGTH="${LENGTH:-1024}"
RUNS="${RUNS:-5}"
OUTDIR="${OUTDIR:-$NEW/results/matrix-${LENGTH}}"

export BUILD_ROOT=$NEW
# shellcheck disable=SC1091
source ~/spyre-build/repro/scripts/env.sh
export DTI_PROJECT_ROOT=$NEW
export SENLIB_DEVEL_CONFIG_FILE=/etc/ibm/spyre/senlib_config.json
export HF_HOME=/workspace/.cache/huggingface
# Thread pinning: the pod's cgroup quota is 8 but the container sees 192 CPUs.
# Worth ~13%; established in phase 1.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
ulimit -s 524288 2>/dev/null || echo "warn: could not raise stack"

mkdir -p "$OUTDIR"
cd "$NEW/spyre-inference"

run_cell() {
    local name="$1"; shift
    local log="$OUTDIR/${name}.log"
    printf '\n=== %s: %s\n' "$name" "$*"
    # Sequential by necessity: Spyre is contested by one process at a time; two
    # concurrent Spyre processes hang or corrupt the compile cache.
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
        uv run --no-sync --directory "$NEW/spyre-inference" python \
        tests/v1/kv_offload/ttft_evict.py --length "$LENGTH" --runs "$RUNS" "$@" \
        >"$log" 2>&1
    printf '    exit=%s\n' "$?"
    grep -E "steady-state reload|run1\(cold\)|cumulative BLOCKS|^\[verdict\]" "$log" | sed 's/^/    /'
}

echo "matrix at length=$LENGTH runs=$RUNS -> $OUTDIR"

run_cell cell1-nocache-nooffload  --no-prefix-caching --no-kv-offload
run_cell cell2-cache-nooffload    --prefix-caching    --no-kv-offload
run_cell cell3-cache-offload      --prefix-caching    --kv-offload
run_cell cell4-nocache-offload    --no-prefix-caching --kv-offload

echo
echo "================ SUMMARY (steady state = run3+) ================"
printf '%-28s %12s %12s %14s\n' cell cold_s steady_s blocks_s/l
for f in "$OUTDIR"/cell*.log; do
    n=$(basename "$f" .log)
    cold=$(grep -oP 'run1\(cold\)=\K[0-9.]+' "$f" | tail -1)
    steady=$(grep -oP 'steady-state reload \(run3\+\): median=\K[0-9.]+' "$f" | tail -1)
    blks=$(grep -oP 'cumulative BLOCKS stored=\K[0-9]+ loaded=[0-9]+' "$f" | tail -1 | tr ' ' '/' | sed 's/loaded=//')
    printf '%-28s %12s %12s %14s\n' "$n" "${cold:-n/a}" "${steady:-n/a}" "${blks:-n/a}"
done
echo
echo "The meaningful comparison is cell2 vs cell3: identical setup, offload is the"
echo "only difference. cell1 is the no-caching floor. cell4 should show stored=0."
