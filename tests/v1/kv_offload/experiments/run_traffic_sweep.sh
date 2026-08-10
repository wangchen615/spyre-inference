#!/usr/bin/env bash
# ==============================================================================
# run_traffic_sweep.sh -- traffic-driven three-way comparison across lengths.
#
# For EACH length, one fresh engine runs 10+ junk->reload->reuse cycles, giving
# 10+ samples of each path. Then the process exits and the next length starts with
# a new engine.
#
# WHY A FRESH ENGINE PER LENGTH (not one engine for all lengths):
#   * num_gpu_blocks_override and max_model_len are fixed at LLM() construction and
#     are both length-dependent, so a shared engine would be mis-sized for every
#     length but the first;
#   * the Spyre attention kernel compiles per prompt shape, so earlier shapes'
#     compiled artifacts would bleed into later measurements;
#   * host-tier contents persist, so an earlier length's blocks would still occupy
#     the DRAM pool.
# A process boundary is the only reliable reset for all three.
#
# WHY 10+ CYCLES: at 65536 an earlier 5-run attempt was still descending at the
# last run (360 -> 148 -> 147 s), so its "median" was a transient. 10+ cycles put
# the median firmly in the plateau and make min/max meaningful.
#
# Runs are strictly SEQUENTIAL -- Spyre is contested by one process at a time;
# concurrent Spyre processes hang or corrupt the compile cache.
#
# Usage:
#   ./run_traffic_sweep.sh
#   LENGTHS="4096 16384" CYCLES=12 ./run_traffic_sweep.sh
#   HOST_GB=32 ./run_traffic_sweep.sh
# ==============================================================================
set -uo pipefail

NEW=/tmp/work-kvoffload
LENGTHS="${LENGTHS:-1024 4096 16384 32768}"
CYCLES="${CYCLES:-10}"
HOST_GB="${HOST_GB:-16}"
POOL_MULT="${POOL_MULT:-1.5}"
OUTDIR="${OUTDIR:-$NEW/results/traffic-$(date +%Y%m%d-%H%M%S)}"

export BUILD_ROOT=$NEW
# shellcheck disable=SC1091
source ~/spyre-build/repro/scripts/env.sh
export DTI_PROJECT_ROOT=$NEW
export SENLIB_DEVEL_CONFIG_FILE=/etc/ibm/spyre/senlib_config.json
export HF_HOME=/workspace/.cache/huggingface
# Pod cgroup quota is 8 CPUs while the container sees 192; pinning is worth ~13%.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
ulimit -s 524288 2>/dev/null || echo "warn: could not raise stack"

mkdir -p "$OUTDIR"
echo "traffic sweep -> $OUTDIR"
echo "lengths=$LENGTHS cycles=$CYCLES host=${HOST_GB}GB pool_mult=$POOL_MULT"
echo
# The junk is auto-sized to pool-1 blocks so each length reloads its FULL prefix.
#   * junk == prompt leaves (pool - blk) of P1 resident -> only a 50% reload;
#   * junk == pool does NOT boot: max_model_len rounds up past the pool.
# max_model_len follows the junk, not the prompt, so compiles get longer.
printf '%8s %10s %12s %10s %10s %13s %9s\n' \
    length blk/prompt device_pool junk_blk junk_len max_model_len reload
for n in $LENGTHS; do
    blk=$(( (n + 127) / 128 ))
    pool=$(python3 -c "print(int($blk*$POOL_MULT)+1)")
    jblk=$(( pool - 1 ))
    jlen=$(( jblk * 128 ))
    mml=$(python3 -c "print((( max($n,$jlen)+1+127)//128)*128)")
    frac=$(python3 -c "print(int(100*min($blk,$jblk)/$blk))")
    printf '%8s %10s %12s %10s %10s %13s %8s%%\n' \
        "$n" "$blk" "$pool" "$jblk" "$jlen" "$mml" "$frac"
done

for n in $LENGTHS; do
    log="$OUTDIR/traffic_${n}.log"
    printf '\n========== length=%s (fresh engine) ==========\n' "$n"
    t0=$(date +%s)
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
        uv run --no-sync --directory "$NEW/spyre-inference" python \
        "$NEW/traffic_three_way.py" --length "$n" --cycles "$CYCLES" \
        --host-gb "$HOST_GB" --pool-mult "$POOL_MULT" \
        >"$log" 2>&1
    rc=$?
    printf '  rc=%s in %ss -> %s\n' "$rc" "$(( $(date +%s) - t0 ))" "$log"
    if [[ $rc -ne 0 ]]; then
        grep -m4 -E "\[fatal\]|ValueError|RuntimeError" "$log" | sed 's/^/    /'
        # The harness redirects fd 1/2, so an engine-construction failure can leave
        # the visible log sparse; the traffic script restores fds in finally, but a
        # hard crash may still land in the temp file.
        printf '    (if empty, check the newest /tmp/tmp*.traffic.log)\n'
    fi
    sed -n '/^SUMMARY/,$p' "$log" | grep -E "^(recompute|reload|reuse) |recompute / reload|reload / reuse|ORDERING|ANOMALY|correctness|only [0-9]|blocks moved" \
        | sed 's/^/    /'
done

echo
echo "=================== TRAFFIC SWEEP: ALL LENGTHS ==================="
printf '%8s %11s %11s %11s %12s %12s %13s\n' \
    length recomp_s reload_s reuse_s rec/reload reload/reuse loaded_blk/req
printf '%8s %11s %11s %11s %12s %12s %13s\n' \
    -------- ----------- ----------- ----------- ------------ ------------ -------------
for n in $LENGTHS; do
    log="$OUTDIR/traffic_${n}.log"
    rc_=$(awk '$1=="recompute"&&NF>=6{print $3}' "$log" 2>/dev/null | head -1)
    rl=$(awk '$1=="reload"&&NF>=6{print $3}'    "$log" 2>/dev/null | head -1)
    ru=$(awk '$1=="reuse"&&NF>=6{print $3}'     "$log" 2>/dev/null | head -1)
    ldb=$(awk '$1=="reload"&&NF>=6{print $6}'   "$log" 2>/dev/null | head -1)
    r1=$(grep -oP 'recompute / reload = \K[0-9.]+' "$log" 2>/dev/null | head -1)
    r2=$(grep -oP 'reload / reuse     = \K[0-9.]+' "$log" 2>/dev/null | head -1)
    printf '%8s %11s %11s %11s %12s %12s %13s\n' "$n" \
        "${rc_:-n/a}" "${rl:-n/a}" "${ru:-n/a}" \
        "${r1:+${r1}x}" "${r2:+${r2}x}" "${ldb:-n/a}"
done
echo
echo "rec/reload   > 1 means offload beats recomputing under device pressure"
echo "reload/reuse > 1 is expected (a DRAM round-trip costs more than an HBM hit);"
echo "             < 1 would be the anomaly, reproduced inside a single engine."
