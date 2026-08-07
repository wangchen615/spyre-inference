#!/usr/bin/env bash
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the spyre-inference repo for the full license text.
#
# Sweep ttft_concurrency.py over concurrency x arm, one FRESH PROCESS per point.
#
# Why a fresh process per point: the device pool is fixed at LLM() construction and
# the Spyre kernels compile per shape. Reusing one engine across points would leave
# the pool sized for the first point and let earlier shapes' compiles and resident
# blocks bleed into later measurements.
#
# Runs are strictly SEQUENTIAL. Spyre is contested by one process at a time
# (CLAUDE.md); two concurrent Spyre-backed processes hang or corrupt the compile
# cache. Do not add parallelism here.
#
# The three arms isolate each contribution -- nocache is the floor, cache is what
# prefix caching alone buys, offload is what the connector adds. cache is the
# baseline the offload claim rests on, so cache->offload is the connector alone.
#
# Usage:
#   cd /home/yuezhu/dt-inductor/spyre-inference
#   tests/v1/kv_offload/ttft_concurrency_sweep.sh
#
#   # the two smoke points, offload arm only:
#   CONCURRENCIES="8 20" ARMS="offload" ROUNDS=2 \
#       tests/v1/kv_offload/ttft_concurrency_sweep.sh
#
# Env knobs:
#   CONCURRENCIES  space-separated N values (default: 2 4 8 12 16 20)
#   ARMS           subset of "nocache cache offload" (default: all three)
#   LENGTH         prompt tokens (default: 4096)
#   OUTPUT_TOKENS  generated tokens per request (default: 128)
#   CAPACITY_REQS  pool sized for this many requests (default: 8)
#   ROUNDS         rounds per point (default: 4)
#   MAX_BATCHED    max_num_batched_tokens (default: max(N) * LENGTH, held constant)
#   OUTDIR         logs + JSON destination
#   EXTRA_ARGS     forwarded verbatim to ttft_concurrency.py
#   EXEC_TIMEOUT   VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (default: 7200)
#
# Exit status: 0 only if every point succeeded; 1 if any failed.

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="${REPO_ROOT}/tests/v1/kv_offload/ttft_concurrency.py"

CONCURRENCIES="${CONCURRENCIES:-2 4 8 12 16 20}"
ARMS="${ARMS:-nocache cache offload}"
LENGTH="${LENGTH:-4096}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
CAPACITY_REQS="${CAPACITY_REQS:-8}"
ROUNDS="${ROUNDS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-7200}"
OUTDIR="${OUTDIR:-${HOME}/workspace/test/ttft_concurrency-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "[fatal] ttft_concurrency.py not found at ${SCRIPT}" >&2
    exit 2
fi

# max_num_batched_tokens must be CONSTANT across the sweep: if it varied with N, the
# scheduler's chunking threshold would move between points and could itself create a
# knee that looks like a KV-capacity cliff.
MAX_N=0
for n in ${CONCURRENCIES}; do (( n > MAX_N )) && MAX_N=${n}; done
MAX_BATCHED="${MAX_BATCHED:-$(( MAX_N * LENGTH ))}"

arm_flags() {
    case "$1" in
        nocache) echo "--no-prefix-caching --no-kv-offload" ;;
        cache)   echo "--prefix-caching --no-kv-offload" ;;
        offload) echo "--prefix-caching --kv-offload" ;;
        *)       echo "[fatal] unknown arm: $1" >&2; exit 2 ;;
    esac
}

mkdir -p "${OUTDIR}"
SUMMARY="${OUTDIR}/sweep_summary.txt"
TABLE="${OUTDIR}/concurrency_table.txt"

{
    echo "[sweep] repo=${REPO_ROOT}"
    echo "[sweep] concurrencies=${CONCURRENCIES}"
    echo "[sweep] arms=${ARMS}"
    echo "[sweep] length=${LENGTH} output_tokens=${OUTPUT_TOKENS}"
    echo "[sweep] capacity_reqs=${CAPACITY_REQS} rounds=${ROUNDS}"
    echo "[sweep] max_num_batched_tokens=${MAX_BATCHED} (constant across points)"
    echo "[sweep] outdir=${OUTDIR}"
    echo "[sweep] started $(date -Is)"
    echo
} | tee "${SUMMARY}"

failed=()
rows=()
sweep_start=$(date +%s)

# Ascending N, and never abort the sweep on one failure: a death at the tail must
# still leave the earlier points on disk.
for n in ${CONCURRENCIES}; do
    for arm in ${ARMS}; do
        log_file="${OUTDIR}/n${n}_${arm}.log"
        json_file="${OUTDIR}/n${n}_${arm}.json"
        echo "==> N=${n} arm=${arm} -> ${log_file}" | tee -a "${SUMMARY}"
        start=$(date +%s)

        # --no-sync keeps uv from reinstalling the pinned upstream torch-spyre over a
        # hand-installed local build mid-sweep (CLAUDE.md).
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${EXEC_TIMEOUT}" \
            uv run --no-sync --directory "${REPO_ROOT}" python "${SCRIPT}" \
            --length "${LENGTH}" --output-tokens "${OUTPUT_TOKENS}" \
            --concurrency "${n}" --capacity-reqs "${CAPACITY_REQS}" \
            --rounds "${ROUNDS}" --max-batched-tokens "${MAX_BATCHED}" \
            --json "${json_file}" $(arm_flags "${arm}") ${EXTRA_ARGS} \
            >"${log_file}" 2>&1
        rc=$?

        elapsed=$(( $(date +%s) - start ))
        if [[ ${rc} -eq 0 ]]; then
            echo "    ok   rc=0 in ${elapsed}s" | tee -a "${SUMMARY}"
        else
            echo "    FAIL rc=${rc} in ${elapsed}s (see ${log_file})" | tee -a "${SUMMARY}"
            failed+=("N=${n}/${arm}")
        fi

        grep -E '^\[(summary|verdict)\]' "${log_file}" | sed 's/^/    /' >>"${SUMMARY}" 2>/dev/null
        echo >>"${SUMMARY}"

        # Scrape the script's own summary lines rather than recomputing, so the table
        # and the per-point logs cannot disagree.
        cold=$(grep -oP 'cold\(round1\)=\K[0-9.]+' "${log_file}" | tail -1)
        steady=$(grep -oP 'steady\(round3\+\) median=\K[0-9.]+' "${log_file}" | tail -1)
        cached=$(grep -oP 'mean cache_hit_frac over rounds 2\+ = \K[0-9.]+' "${log_file}" | tail -1)
        stored=$(grep -oP 'cumulative BLOCKS stored=\K[0-9]+|cumulative BLOCKS stored=\Kn/a' \
            "${log_file}" | tail -1)
        loaded=$(grep -oP 'cumulative BLOCKS stored=\S+ loaded=\K\S+' "${log_file}" | tail -1)
        # p50/p90 from the last reported round.
        last_round=$(grep -E '^\[summary\]\s+round' "${log_file}" | tail -1)
        p50=$(awk '{print $5}' <<<"${last_round}")
        p90=$(awk '{print $6}' <<<"${last_round}")
        rows+=("${n}|${arm}|${cold:-n/a}|${steady:-n/a}|${p50:-n/a}|${p90:-n/a}|${cached:-n/a}|${stored:-n/a}|${loaded:-n/a}")
    done
done

{
    echo "[sweep] finished $(date -Is) in $(( $(date +%s) - sweep_start ))s"
    if [[ ${#failed[@]} -eq 0 ]]; then
        echo "[sweep] all points ok"
    else
        echo "[sweep] FAILED points: ${failed[*]}"
    fi
    echo "[sweep] logs in ${OUTDIR}"
} | tee -a "${SUMMARY}"

{
    echo
    echo "=== KV-offload concurrency curve: L=${LENGTH}, capacity=${CAPACITY_REQS} requests ==="
    echo "rounds=${ROUNDS}  output_tokens=${OUTPUT_TOKENS}  max_num_batched_tokens=${MAX_BATCHED}"
    echo
    printf '%4s  %-8s  %10s  %11s  %8s  %8s  %8s  %8s  %8s\n' \
        N arm cold_s steady_s p50_s p90_s cached% stored loaded
    printf '%4s  %-8s  %10s  %11s  %8s  %8s  %8s  %8s  %8s\n' \
        ---- -------- ---------- ----------- -------- -------- -------- -------- --------
    for row in "${rows[@]}"; do
        IFS='|' read -r rn rarm cold steady p50 p90 cached stored loaded <<<"${row}"
        printf '%4s  %-8s  %10s  %11s  %8s  %8s  %8s  %8s  %8s\n' \
            "${rn}" "${rarm}" "${cold}" "${steady}" "${p50}" "${p90}" \
            "${cached}" "${stored}" "${loaded}"
    done
    echo
    echo "steady_s = median round-3+ wall-clock to drain all N requests"
    echo "cached%  = mean prefix-cache hit fraction over rounds 2+. Falling as N rises is"
    echo "           the lost-cache cost of displacement; the offload arm should hold it up."
    echo "nocache  = no prefix caching, no offload (floor: every prefill from scratch)"
    echo "cache    = prefix caching, no offload   (victims re-prefill -> the jump)"
    echo "offload  = prefix caching + offload     (victims reload from host)"
    echo "stored/loaded are n/a in the nocache and cache arms: no connector exists there,"
    echo "which is NOT the same as a connector that moved zero blocks."
} | tee "${TABLE}" | tee -a "${SUMMARY}"

# Pressure sanity check. vLLM's preemption counter is unreadable here (enabling stats
# crashes with this spec -- see ttft_concurrency.py), so pressure is inferred from the
# offload arm's transfers: past capacity, blocks MUST move, and at or below capacity
# they must not. A point that violates either is not measuring what the curve claims.
{
    echo
    echo "=== pressure sanity check (offload arm transfers) ==="
    echo "capacity=${CAPACITY_REQS} requests"
    for n in ${CONCURRENCIES}; do
        st=""
        for row in "${rows[@]}"; do
            IFS='|' read -r rn rarm _ _ _ _ _ rs _ <<<"${row}"
            if [[ "${rn}" == "${n}" && "${rarm}" == "offload" ]]; then st="${rs}"; fi
        done
        [[ -z "${st}" ]] && continue
        note=""
        if (( n > CAPACITY_REQS )); then
            if [[ "${st}" == "0" ]]; then
                note="   <-- WARNING: over capacity but nothing offloaded"
            fi
        elif [[ "${st}" =~ ^[0-9]+$ ]] && (( st > 0 )); then
            note="   <-- WARNING: at/below capacity but blocks moved"
        fi
        printf '  N=%-3s stored=%-8s%s\n' "${n}" "${st}" "${note}"
    done
    echo
    echo "Expected: stored=0 at N<=capacity, stored>0 above it. A WARNING means the pool"
    echo "pin or the connector is not behaving as the curve assumes -- investigate before"
    echo "drawing conclusions."
} | tee -a "${TABLE}" | tee -a "${SUMMARY}"

echo
echo "[sweep] table: ${TABLE}"

exit $(( ${#failed[@]} > 0 ))
