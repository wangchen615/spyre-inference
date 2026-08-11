#!/usr/bin/env bash
# ==============================================================================
# run_full_sweep_detached.sh -- the complete phase-2 traffic sweep, detached.
#
# Runs the traffic-driven three-way (recompute / host-reload / HBM-reuse) at every
# length from 1024 to 65536 with the CORRECTED junk geometry, so every length
# reloads its FULL prefix.
#
# WHY NOT PARALLEL: Spyre is contested by ONE process at a time. Two concurrent
# engines hang or corrupt the shared compile cache. The loop is therefore strictly
# sequential and there is no way around it -- this is a hardware/driver constraint,
# not a scripting choice.
#
# WHY DETACHED: launched under setsid + nohup so the process is reparented to init
# and survives SSH disconnection, terminal close, and exit of the CLI that started
# it. A background subagent would NOT survive -- it lives inside the CLI process.
#
# RUNTIME: ~6.8 h estimated, dominated by the junk prompts. Fitted from measured
# offload-off recompute (t ~ 1.39e-3 * L^1.18, within 10% at 1024/4096/16384/65536):
#     L=1024    5 m      L=16384   45 m
#     L=4096   14 m      L=32768   97 m
#     L=8192   28 m      L=65536  217 m
# The junk dominates because it is ~1.5x the prompt and recompute is superlinear.
#
# HOST TIER = 64 GB. At L=65536, 6 cycles, worst case is 2*512 + 6*768 = 5632
# blocks = 11.8 GB; 10 cycles would be 18.3 GB, i.e. ABOVE the 16 GB default.
# Undersizing this tier is the bug that produced the 8x-pessimistic 65536 result
# earlier (host tier thrashes, blocks never plateau). 64 GB leaves wide margin
# against 1.99 TB available.
#
# CYCLES: 10 at short lengths (cheap, tighter medians), 6 at >=16384 where each
# cycle costs minutes. 6 still leaves 5 post-compile samples, enough for a median
# in the plateau -- the earlier failure was 5 TOTAL runs still descending at the
# last one, which 6 cycles with a discarded first sample avoids.
#
# Usage:
#   setsid nohup ./run_full_sweep_detached.sh > /tmp/work-kvoffload/sweep.out 2>&1 &
# Check on it later from any session:
#   cat /tmp/work-kvoffload/results/full-sweep-*/PROGRESS
# ==============================================================================
set -uo pipefail

NEW=/tmp/work-kvoffload
HOST_GB="${HOST_GB:-64}"
POOL_MULT="${POOL_MULT:-1.5}"
OUTDIR="${OUTDIR:-$NEW/results/full-sweep-$(date +%Y%m%d-%H%M%S)}"

# length:cycles -- fewer cycles where a cycle costs minutes
PLAN="${PLAN:-1024:10 4096:10 8192:10 16384:6 32768:6 65536:6}"

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
PROG="$OUTDIR/PROGRESS"

# PROGRESS is the file to read from a future session. Written after every length so
# a crash or a kill still leaves an accurate record of what completed.
{
    echo "full phase-2 traffic sweep"
    echo "started : $(date -Is)"
    echo "pid     : $$"
    echo "outdir  : $OUTDIR"
    echo "plan    : $PLAN"
    echo "host_gb : $HOST_GB   pool_mult: $POOL_MULT"
    echo "estimate: ~6.8 h total (65536 alone ~3.6 h)"
    echo
    printf '%8s %6s %9s %9s %11s %11s %11s %12s %12s %9s\n' \
        length cycles status wall_s recompute_s reload_s reuse_s rec/reload reload/reuse loaded
    printf '%8s %6s %9s %9s %11s %11s %11s %12s %12s %9s\n' \
        -------- ------ --------- --------- ----------- ----------- ----------- ------------ ------------ ---------
} > "$PROG"

echo "sweep -> $OUTDIR   (progress: $PROG)"

for item in $PLAN; do
    n="${item%%:*}"
    cyc="${item##*:}"
    log="$OUTDIR/traffic_${n}.log"
    printf '\n========== length=%s cycles=%s (fresh engine) ==========\n' "$n" "$cyc"
    t0=$(date +%s)

    # Fresh engine per length: num_gpu_blocks_override and max_model_len are fixed
    # at construction and both length-dependent, the attention kernel compiles per
    # prompt shape, and host-tier contents persist. A process boundary is the only
    # reliable reset for all three.
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=28800 \
        uv run --no-sync --directory "$NEW/spyre-inference" python \
        "$NEW/traffic_three_way.py" --length "$n" --cycles "$cyc" \
        --host-gb "$HOST_GB" --pool-mult "$POOL_MULT" \
        >"$log" 2>&1
    rc=$?
    wall=$(( $(date +%s) - t0 ))
    printf '  rc=%s in %ss -> %s\n' "$rc" "$wall" "$log"

    rec=$(awk '$1=="recompute"&&NF>=6{print $3}' "$log" 2>/dev/null | head -1)
    rl=$(awk  '$1=="reload"   &&NF>=6{print $3}' "$log" 2>/dev/null | head -1)
    ru=$(awk  '$1=="reuse"    &&NF>=6{print $3}' "$log" 2>/dev/null | head -1)
    ldb=$(awk '$1=="reload"   &&NF>=6{print $6}' "$log" 2>/dev/null | head -1)
    r1=$(grep -oP 'recompute / reload = \K[0-9.]+' "$log" 2>/dev/null | head -1)
    r2=$(grep -oP 'reload / reuse     = \K[0-9.]+' "$log" 2>/dev/null | head -1)
    st=$([[ $rc -eq 0 ]] && echo ok || echo "rc=$rc")

    printf '%8s %6s %9s %9s %11s %11s %11s %12s %12s %9s\n' \
        "$n" "$cyc" "$st" "$wall" "${rec:-n/a}" "${rl:-n/a}" "${ru:-n/a}" \
        "${r1:+${r1}x}" "${r2:+${r2}x}" "${ldb:-n/a}" >> "$PROG"

    if [[ $rc -ne 0 ]]; then
        # Surface the reason into PROGRESS so a future session does not have to dig.
        {
            grep -m4 -E "\[fatal\]|\[warn\]|ValueError|RuntimeError|OutOfMemory" "$log" \
                | sed 's/^/           /'
            echo "           (if empty, check newest /tmp/tmp*.traffic.log)"
        } >> "$PROG"
    fi

    # Echo the per-length verdict lines to the console log too.
    sed -n '/^SUMMARY/,$p' "$log" \
        | grep -E "^(recompute|reload|reuse) |recompute / reload|reload / reuse|ORDERING|ANOMALY|correctness|FULL reload|PARTIAL reload|blocks moved|only [0-9]" \
        | sed 's/^/    /'
done

{
    echo
    echo "finished: $(date -Is)"
    echo
    echo "rec/reload   > 1 means offload beats recomputing under device pressure."
    echo "reload/reuse > 1 is expected (a DRAM round-trip costs more than an HBM hit)."
    echo "loaded should equal blocks/prompt (L/128) for a genuine FULL reload:"
    echo "  1024->8  4096->32  8192->64  16384->128  32768->256  65536->512"
} >> "$PROG"

echo
echo "=================== DONE -- see $PROG ==================="
cat "$PROG"
