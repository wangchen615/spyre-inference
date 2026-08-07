#!/usr/bin/env bash
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the spyre-inference repo for the full license text.
#
# Sweep ttft_recompute.py over power-of-2 prompt lengths, one FRESH PROCESS per
# length, and report cold vs warm TTFT per length in a single table.
#
# This is the no-connector control for ttft_evict_sweep.sh. Same lengths, same pool
# sizing, same fresh-process-per-length discipline -- the only difference is that no
# KV offload connector is attached, so nothing is offloaded or reloaded.
#
# Why a new process per length: the engine sizes num_gpu_blocks_override and
# max_model_len from --length at LLM() construction time, and the Spyre kernels
# compile per prompt shape. Reusing one engine across lengths would leave the pool
# sized for the first length and let earlier shapes' compiles and resident blocks
# bleed into later measurements. Each length therefore gets its own engine and its
# own device pool.
#
# Runs are strictly SEQUENTIAL. Spyre is contested by one process at a time
# (CLAUDE.md); two concurrent Spyre-backed processes hang or corrupt the compile
# cache. Do not add parallelism here.
#
# Usage:
#   cd /home/yuezhu/dt-inductor/spyre-inference
#   tests/v1/kv_offload/ttft_recompute_sweep.sh
#
#   # subset of lengths, more runs each:
#   LENGTHS="1024 2048" RUNS=13 tests/v1/kv_offload/ttft_recompute_sweep.sh
#
#   # every run a full cold prefill (no prefix reuse at all):
#   EXTRA_ARGS="--no-prefix-caching" tests/v1/kv_offload/ttft_recompute_sweep.sh
#
# Env knobs:
#   LENGTHS     space-separated prompt lengths (default: 1024 .. 65536, powers of 2)
#   RUNS        measured runs per length, passed as --runs (default: 9)
#   OUTDIR      where per-length logs and the summary land
#               (default: ~/workspace/test/ttft_recompute_sweep-<timestamp>)
#   EXTRA_ARGS  extra flags forwarded verbatim to ttft_recompute.py
#   EXEC_TIMEOUT  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (default: 7200)
#
# Exit status: 0 only if every length succeeded; 1 if any length failed.

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="${REPO_ROOT}/tests/v1/kv_offload/ttft_recompute.py"

LENGTHS="${LENGTHS:-1024 2048 4096 8192 16384 32768 65536}"
RUNS="${RUNS:-9}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-7200}"
OUTDIR="${OUTDIR:-${HOME}/workspace/test/ttft_recompute_sweep-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "[fatal] ttft_recompute.py not found at ${SCRIPT}" >&2
    exit 2
fi

mkdir -p "${OUTDIR}"
SUMMARY="${OUTDIR}/sweep_summary.txt"
TABLE="${OUTDIR}/cold_warm_table.txt"

{
    echo "[sweep] repo=${REPO_ROOT}"
    echo "[sweep] script=ttft_recompute.py (NO KV connector -- recompute baseline)"
    echo "[sweep] lengths=${LENGTHS}"
    echo "[sweep] runs_per_length=${RUNS}"
    echo "[sweep] extra_args=${EXTRA_ARGS:-<none>}"
    echo "[sweep] outdir=${OUTDIR}"
    echo "[sweep] started $(date -Is)"
    echo
} | tee "${SUMMARY}"

failed=()
# Per-length rows accumulated for the final cold-vs-warm table.
rows=()
sweep_start=$(date +%s)

for n in ${LENGTHS}; do
    log_file="${OUTDIR}/ttft_recompute_${n}.log"
    echo "==> length=${n} runs=${RUNS} -> ${log_file}" | tee -a "${SUMMARY}"
    start=$(date +%s)

    # A fresh `uv run` per length is the process boundary: engine, device pool, and
    # compile state all die with it. --no-sync keeps uv from reinstalling the pinned
    # upstream torch-spyre over a hand-installed local build mid-sweep (CLAUDE.md).
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${EXEC_TIMEOUT}" \
        uv run --no-sync --directory "${REPO_ROOT}" python "${SCRIPT}" \
        --length "${n}" --runs "${RUNS}" ${EXTRA_ARGS} \
        >"${log_file}" 2>&1
    rc=$?

    elapsed=$(( $(date +%s) - start ))
    if [[ ${rc} -eq 0 ]]; then
        echo "    ok   rc=0 in ${elapsed}s" | tee -a "${SUMMARY}"
    else
        echo "    FAIL rc=${rc} in ${elapsed}s (see ${log_file})" | tee -a "${SUMMARY}"
        failed+=("${n}")
    fi

    # Replay this length's own [summary] lines into the combined summary, so one file
    # answers "what did the whole sweep measure" without opening seven.
    grep -E '^\[summary\]' "${log_file}" | sed 's/^/    /' >>"${SUMMARY}" 2>/dev/null
    echo >>"${SUMMARY}"

    # Scrape the three reported numbers for the cross-length table. Parsed from the
    # script's own summary lines rather than recomputed here, so the table and the
    # per-length logs cannot disagree.
    cold=$(grep -oP 'run1\(cold miss on B\)=\K[0-9.]+' "${log_file}" | tail -1)
    warm=$(grep -oP 'warm TTFT \(run2\+\): median=\K[0-9.]+' "${log_file}" | tail -1)
    wmin=$(grep -oP 'warm TTFT \(run2\+\):.*min=\K[0-9.]+' "${log_file}" | tail -1)
    wmax=$(grep -oP 'warm TTFT \(run2\+\):.*max=\K[0-9.]+' "${log_file}" | tail -1)
    ratio=$(grep -oP 'cold/warm=\K[0-9.]+' "${log_file}" | tail -1)
    rows+=("${n}|${cold:-n/a}|${warm:-n/a}|${wmin:-n/a}|${wmax:-n/a}|${ratio:-n/a}")
done

{
    echo "[sweep] finished $(date -Is) in $(( $(date +%s) - sweep_start ))s"
    if [[ ${#failed[@]} -eq 0 ]]; then
        echo "[sweep] all lengths ok"
    else
        echo "[sweep] FAILED lengths: ${failed[*]}"
    fi
    echo "[sweep] logs in ${OUTDIR}"
} | tee -a "${SUMMARY}"

# The headline artifact: cold vs warm TTFT across every length, in one place.
{
    echo
    echo "=== ttft_recompute: cold vs warm TTFT (no KV connector) ==="
    echo "runs_per_length=${RUNS}  extra_args=${EXTRA_ARGS:-<none>}"
    echo
    printf '%8s  %11s  %11s  %11s  %11s  %9s\n' \
        length cold_s warm_med_s warm_min_s warm_max_s cold/warm
    printf '%8s  %11s  %11s  %11s  %11s  %9s\n' \
        -------- ----------- ----------- ----------- ----------- ---------
    for row in "${rows[@]}"; do
        IFS='|' read -r n cold warm wmin wmax ratio <<<"${row}"
        printf '%8s  %11s  %11s  %11s  %11s  %9s\n' \
            "${n}" "${cold}" "${warm}" "${wmin}" "${wmax}" "${ratio}"
    done
    echo
    echo "cold      = run1, first send of prompt B (prompt A resident, no shared prefix)"
    echo "warm_*    = run2+, all re-sending the identical prompt B"
    echo "            (n=$(( RUNS > 1 ? RUNS - 1 : 0 )) per length)"
    echo "no connector was attached, so nothing was offloaded or reloaded"
} | tee "${TABLE}" | tee -a "${SUMMARY}"

echo
echo "[sweep] cold/warm table: ${TABLE}"

exit $(( ${#failed[@]} > 0 ))
