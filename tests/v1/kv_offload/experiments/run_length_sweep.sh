#!/usr/bin/env bash
# ==============================================================================
# run_length_sweep.sh -- offload ON vs OFF across context lengths.
#
# THE PHASE-1 ANALOGUE. Phase 1 established that recompute cost is SUPERLINEAR in
# prompt length (5.83s at 1024 -> 714.25s at 65536: 122x over a 64x length range),
# while a block copy is linear in bytes. So the value of reloading instead of
# recomputing should GROW with context length. This sweep measures that curve.
#
# It also fixes a measurement artifact of the single-length runs. The pool is sized
# `ceil(L/128) + 2`, and at L=1024 that is 10 blocks for an 8-block prompt -- 20%
# slack, enough that the scheduler keeps a prompt partially resident and eviction
# is never total (observed: run3+ stored=0, loaded=7 of 8). The slack RATIO shrinks
# with length:
#
#     L=1024   8 blocks/prompt   pool 10   slack 20%
#     L=4096  32 blocks/prompt   pool 34   slack 5.9%
#     L=16384 128 blocks/prompt  pool 130  slack 1.5%
#     L=65536 512 blocks/prompt  pool 514  slack 0.4%
#
# so eviction becomes near-total at length without contriving anything. Note that
# forcing zero slack is IMPOSSIBLE: max_model_len = L + 1 generated token rounded
# up to a block, so the engine needs ceil((L+1)/128) blocks minimum and refuses to
# boot with fewer (verified at L=1024: --gpu-blocks 8 -> ValueError).
#
# WHY OFFLOAD-OFF IS CAPPED. With offload off every alternating run recomputes in
# full, so a length costs ~RUNS x cold. At 65536 that is ~5 x 714s = 1 hour for one
# cell. The curve is already established by phase 1's recompute baseline, so the
# control is capped at OFF_MAX_LEN (default 16384) and longer points are compared
# against phase 1's measured cold numbers instead.
#
# Usage:
#   ./run_length_sweep.sh
#   LENGTHS="1024 4096 16384" RUNS=5 ./run_length_sweep.sh
#   OFF_MAX_LEN=8192 ./run_length_sweep.sh     # cheaper control
# ==============================================================================
set -uo pipefail

NEW=/tmp/work-kvoffload
LENGTHS="${LENGTHS:-1024 4096 16384 65536}"
RUNS="${RUNS:-5}"
OFF_MAX_LEN="${OFF_MAX_LEN:-16384}"
OUTDIR="${OUTDIR:-$NEW/results/lengthsweep-$(date +%Y%m%d-%H%M%S)}"

export BUILD_ROOT=$NEW
# shellcheck disable=SC1091
source ~/spyre-build/repro/scripts/env.sh
export DTI_PROJECT_ROOT=$NEW
export SENLIB_DEVEL_CONFIG_FILE=/etc/ibm/spyre/senlib_config.json
export HF_HOME=/workspace/.cache/huggingface
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
ulimit -s 524288 2>/dev/null || echo "warn: could not raise stack"

mkdir -p "$OUTDIR"
cd "$NEW/spyre-inference"
echo "length sweep -> $OUTDIR"
echo "lengths=$LENGTHS runs=$RUNS offload-off capped at $OFF_MAX_LEN"

# one() <tag> <length> <extra flags...>
one() {
    local tag="$1" n="$2"; shift 2
    local log="$OUTDIR/${tag}_${n}.log"
    printf '\n==> %s length=%s\n' "$tag" "$n"
    local t0 t1
    t0=$(date +%s)
    # Sequential only: Spyre is contested by one process at a time.
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
        uv run --no-sync --directory "$NEW/spyre-inference" python \
        tests/v1/kv_offload/ttft_evict.py --length "$n" --runs "$RUNS" "$@" \
        >"$log" 2>&1
    local rc=$?
    t1=$(date +%s)
    printf '    rc=%s in %ss\n' "$rc" "$((t1 - t0))"
    if [[ $rc -ne 0 ]]; then
        # An engine-construction failure is swallowed by the harness's own fd
        # redirection (TransferCounter dups fd 1/2), so the visible log can be
        # nearly empty. Point at the temp file that holds the real traceback.
        printf '    FAILED -- real traceback may be in the newest /tmp/tmp*.ttft_evict.log\n'
        grep -m3 -E "ValueError|RuntimeError|Error" "$log" | sed 's/^/      /'
    fi
    grep -E "run1\(cold\)|steady-state reload|cumulative BLOCKS|^\[verdict\]" "$log" | sed 's/^/    /'
}

for n in $LENGTHS; do
    one offload-on "$n" --prefix-caching --kv-offload
    if [[ "$n" -le "$OFF_MAX_LEN" ]]; then
        one offload-off "$n" --prefix-caching --no-kv-offload
    else
        printf '\n==> offload-off length=%s SKIPPED (> OFF_MAX_LEN=%s; compare against\n' "$n" "$OFF_MAX_LEN"
        printf '    phase-1 recompute baseline instead)\n'
    fi
done

echo
echo "=================== LENGTH SWEEP SUMMARY ==================="
printf '%8s  %12s  %12s  %12s  %10s  %14s\n' \
    length on_cold_s on_steady_s off_steady_s speedup blocks_s/l
printf '%8s  %12s  %12s  %12s  %10s  %14s\n' \
    -------- ------------ ------------ ------------ ---------- --------------
for n in $LENGTHS; do
    on="$OUTDIR/offload-on_${n}.log"
    off="$OUTDIR/offload-off_${n}.log"
    oncold=$(grep -oP 'run1\(cold\)=\K[0-9.]+' "$on" 2>/dev/null | tail -1)
    onst=$(grep -oP 'steady-state reload \(run3\+\): median=\K[0-9.]+' "$on" 2>/dev/null | tail -1)
    offst=$(grep -oP 'steady-state reload \(run3\+\): median=\K[0-9.]+' "$off" 2>/dev/null | tail -1)
    blks=$(grep -oP 'cumulative BLOCKS stored=\K[0-9]+ loaded=[0-9]+' "$on" 2>/dev/null | tail -1 \
           | sed 's/ loaded=/\//')
    sp="n/a"
    if [[ -n "${onst:-}" && -n "${offst:-}" ]]; then
        sp=$(python3 -c "print(f'{${offst}/${onst}:.1f}x')" 2>/dev/null || echo n/a)
    fi
    printf '%8s  %12s  %12s  %12s  %10s  %14s\n' \
        "$n" "${oncold:-n/a}" "${onst:-n/a}" "${offst:-skipped}" "$sp" "${blks:-n/a}"
done
echo
echo "speedup = off_steady / on_steady  (higher = offload wins by more)"
echo "Phase-1 recompute cold, for the skipped control points:"
echo "  1024=5.83s  4096=24.88s  16384=117.48s  32768=274.15s  65536=714.25s"
