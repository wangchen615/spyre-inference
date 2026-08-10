#!/usr/bin/env bash
# ==============================================================================
# run_tight_hbm.sh -- squeeze device KV (HBM) so prefix caching CANNOT help, and
# give the host (DRAM) tier plenty of room so reload ALWAYS can.
#
# THE POINT. With a device pool sized for a single prompt, an alternating A/B
# workload can never serve the incoming prompt from device-resident blocks -- the
# other prompt has to go. That removes prefix caching as a confound and leaves one
# question: when the blocks are needed again, are they RELOADED from DRAM or
# RECOMPUTED? Offload on vs off answers exactly that, with nothing else differing.
#
# TWO POOLS, SIZED IN OPPOSITE DIRECTIONS
#
#   device (HBM)  = ceil(L/128) + 1 blocks  <- the TIGHTEST legal size.
#       max_model_len must cover the prompt plus the one generated token, so the
#       engine needs ceil((L+1)/128) blocks and REFUSES to boot with fewer:
#           --gpu-blocks 8 at L=1024 -> ValueError, "estimated maximum model
#           length is 1024" (verified).
#       That makes blk+1 the floor. The harness default is blk+2, one block
#       looser, and at short lengths that spare block is a large fraction of a
#       prompt (25% at L=1024) -- enough for the scheduler to keep part of the
#       other prompt resident. Observed at the default: run3+ stored=0,
#       loaded=7-of-8, i.e. A was never actually displaced. blk+1 removes that
#       slack entirely.
#
#   host (DRAM)   = 16 GB, far more than any length here needs.
#       Measured 2.10 MB per block, so two prompts need:
#           L=16384 -> 0.54 GB    L=32768 -> 1.08 GB    L=65536 -> 2.15 GB
#       The harness default --cpu-bytes 2e9 is BELOW the 65536 requirement, which
#       is why that length thrashed: only 440 of 512 blocks reloaded, `stored`
#       climbed to 1301 cumulative from repeated store jobs, and the run sequence
#       (360 -> 148 -> 147 s) never reached a plateau. 16 GB removes the host tier
#       as a variable so the reload path is measured, not host eviction.
#
# WHAT TO EXPECT
#   offload ON  -> reload from DRAM; loaded_blocks should be ~all of a prompt
#   offload OFF -> full recompute every run; cold/steady ~= 1.0x
#   speedup = off_steady / on_steady, and it should GROW with length, because
#   recompute is superlinear in prompt length while a block copy is linear.
#
# Usage:
#   ./run_tight_hbm.sh
#   LENGTHS="16384 65536" RUNS=7 ./run_tight_hbm.sh
#   HOST_GB=32 ./run_tight_hbm.sh
# ==============================================================================
set -uo pipefail

NEW=/tmp/work-kvoffload
LENGTHS="${LENGTHS:-4096 16384 65536}"
RUNS="${RUNS:-7}"
HOST_GB="${HOST_GB:-16}"
# Cap the offload-OFF control: it recomputes in full every run, so a length costs
# ~RUNS x cold. At 65536 that is 7 x 717 s = 84 min for one cell, and phase 1
# already measured the recompute curve.
OFF_MAX_LEN="${OFF_MAX_LEN:-16384}"
OUTDIR="${OUTDIR:-$NEW/results/tighthbm-$(date +%Y%m%d-%H%M%S)}"

export BUILD_ROOT=$NEW
# shellcheck disable=SC1091
source ~/spyre-build/repro/scripts/env.sh
export DTI_PROJECT_ROOT=$NEW
export SENLIB_DEVEL_CONFIG_FILE=/etc/ibm/spyre/senlib_config.json
export HF_HOME=/workspace/.cache/huggingface
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
ulimit -s 524288 2>/dev/null || echo "warn: could not raise stack"

CPU_BYTES=$((HOST_GB * 1000000000))
mkdir -p "$OUTDIR"
cd "$NEW/spyre-inference"

echo "tight-HBM sweep -> $OUTDIR"
echo "lengths=$LENGTHS runs=$RUNS host_pool=${HOST_GB}GB offload-off capped at $OFF_MAX_LEN"
printf '\n%8s  %10s  %12s  %14s\n' length blk/prompt device_pool host_needed_2p
for n in $LENGTHS; do
    blk=$(( (n + 127) / 128 ))
    printf '%8s  %10s  %12s  %13.2fGB\n' "$n" "$blk" "$((blk + 1))" \
        "$(python3 -c "print(2*$blk*2.10e6/1e9)")"
done

one() {
    local tag="$1" n="$2"; shift 2
    local blk=$(( (n + 127) / 128 ))
    local pool=$(( blk + 1 ))          # tightest legal device pool
    local log="$OUTDIR/${tag}_${n}.log"
    printf '\n==> %s L=%s device_pool=%s (blk=%s, zero spare beyond the gen token)\n' \
        "$tag" "$n" "$pool" "$blk"
    local t0; t0=$(date +%s)
    # Sequential only: Spyre is contested by one process at a time; concurrent
    # Spyre processes hang or corrupt the compile cache.
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
        uv run --no-sync --directory "$NEW/spyre-inference" python \
        tests/v1/kv_offload/ttft_evict.py \
        --length "$n" --runs "$RUNS" --gpu-blocks "$pool" --cpu-bytes "$CPU_BYTES" "$@" \
        >"$log" 2>&1
    local rc=$?
    printf '    rc=%s in %ss\n' "$rc" "$(( $(date +%s) - t0 ))"
    if [[ $rc -ne 0 ]]; then
        # The harness dups fd 1/2 in TransferCounter BEFORE building the engine and
        # only restores them in close(), so an engine-construction failure leaves
        # the visible log nearly empty and the traceback in a temp file.
        printf '    FAILED -- traceback likely in the newest /tmp/tmp*.ttft_evict.log\n'
        grep -m3 -E "ValueError|RuntimeError" "$log" | sed 's/^/      /'
    fi
    grep -E "run1\(cold\)|steady-state reload|cumulative BLOCKS|^\[verdict\]" "$log" | sed 's/^/    /'
}

for n in $LENGTHS; do
    one offload-on "$n" --prefix-caching --kv-offload
    if [[ "$n" -le "$OFF_MAX_LEN" ]]; then
        one offload-off "$n" --prefix-caching --no-kv-offload
    else
        printf '\n==> offload-off L=%s SKIPPED (> %s); compare against phase-1 recompute\n' \
            "$n" "$OFF_MAX_LEN"
    fi
done

echo
echo "============== TIGHT-HBM SUMMARY (device = blk+1, host = ${HOST_GB}GB) =============="
printf '%8s  %11s  %12s  %12s  %9s  %16s\n' \
    length on_cold_s on_steady_s off_steady_s speedup blocks_stored/loaded
printf '%8s  %11s  %12s  %12s  %9s  %16s\n' \
    -------- ----------- ------------ ------------ --------- ----------------
for n in $LENGTHS; do
    on="$OUTDIR/offload-on_${n}.log"; off="$OUTDIR/offload-off_${n}.log"
    oc=$(grep -oP 'run1\(cold\)=\K[0-9.]+' "$on" 2>/dev/null | tail -1)
    os=$(grep -oP 'steady-state reload \(run3\+\): median=\K[0-9.]+' "$on" 2>/dev/null | tail -1)
    fs=$(grep -oP 'steady-state reload \(run3\+\): median=\K[0-9.]+' "$off" 2>/dev/null | tail -1)
    bl=$(grep -oP 'cumulative BLOCKS stored=\K[0-9]+ loaded=[0-9]+' "$on" 2>/dev/null | tail -1 | sed 's/ loaded=/\//')
    sp="n/a"
    [[ -n "${os:-}" && -n "${fs:-}" ]] && sp=$(python3 -c "print(f'{${fs}/${os}:.1f}x')" 2>/dev/null || echo n/a)
    printf '%8s  %11s  %12s  %12s  %9s  %16s\n' \
        "$n" "${oc:-n/a}" "${os:-n/a}" "${fs:-skipped}" "$sp" "${bl:-n/a}"
done
echo
echo "Phase-1 recompute cold (for skipped controls): 4096=24.88s 16384=117.48s 65536=714.25s"
