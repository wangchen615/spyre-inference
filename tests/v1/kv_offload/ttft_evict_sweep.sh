#!/usr/bin/env bash
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the spyre-inference repo for the full license text.
#
# Sweep ttft_evict.py over power-of-2 prompt lengths, one FRESH PROCESS per length.
#
# Why a new process per length: the engine sizes num_gpu_blocks_override, the host
# KV pool, and max_model_len from --length at LLM() construction time, and the Spyre
# kernels compile per prompt shape. Reusing one engine across lengths would leave
# the pool sized for the first length and let earlier shapes' compiles and resident
# blocks bleed into later measurements. Each length therefore gets its own engine,
# its own device pool, and its own offload counters.
#
# Runs are strictly SEQUENTIAL. Spyre is contested by one process at a time
# (CLAUDE.md); two concurrent Spyre-backed processes hang or corrupt the compile
# cache. Do not add parallelism here.
#
# Usage:
#   cd /home/yuezhu/dt-inductor/spyre-inference
#   tests/v1/kv_offload/ttft_evict_sweep.sh
#
#   # subset of lengths, more runs each:
#   LENGTHS="1024 2048" RUNS=13 tests/v1/kv_offload/ttft_evict_sweep.sh
#
#   # baseline pass with offload off, into a separate output dir:
#   OUTDIR=~/workspace/test/ttft_evict_baseline EXTRA_ARGS="--no-kv-offload" \
#       tests/v1/kv_offload/ttft_evict_sweep.sh
#
# Env knobs:
#   LENGTHS     space-separated prompt lengths (default: 1024 .. 65536, powers of 2)
#   RUNS        measured runs per length, passed as --runs (default: 9)
#   OUTDIR      where per-length logs and the summary land
#               (default: ~/workspace/test/ttft_evict_sweep-<timestamp>)
#   EXTRA_ARGS  extra flags forwarded verbatim to ttft_evict.py (e.g. --gpu-blocks 12)
#   EXEC_TIMEOUT  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (default: 7200)
#
# Exit status: 0 only if every length succeeded; 1 if any length failed.

set -u -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="${REPO_ROOT}/tests/v1/kv_offload/ttft_evict.py"

LENGTHS="${LENGTHS:-1024 2048 4096 8192 16384 32768 65536}"
RUNS="${RUNS:-9}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-7200}"
OUTDIR="${OUTDIR:-${HOME}/workspace/test/ttft_evict_sweep-$(date +%Y%m%d-%H%M%S)}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "[fatal] ttft_evict.py not found at ${SCRIPT}" >&2
    exit 2
fi

mkdir -p "${OUTDIR}"
SUMMARY="${OUTDIR}/sweep_summary.txt"

{
    echo "[sweep] repo=${REPO_ROOT}"
    echo "[sweep] lengths=${LENGTHS}"
    echo "[sweep] runs_per_length=${RUNS}"
    echo "[sweep] extra_args=${EXTRA_ARGS:-<none>}"
    echo "[sweep] outdir=${OUTDIR}"
    echo "[sweep] started $(date -Is)"
    echo
} | tee "${SUMMARY}"

failed=()
sweep_start=$(date +%s)

for n in ${LENGTHS}; do
    log_file="${OUTDIR}/ttft_evict_${n}.log"
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

    # Replay this length's own [summary]/[verdict] lines into the combined summary,
    # so one file answers "what did the whole sweep measure" without opening seven.
    grep -E '^\[(summary|verdict)\]' "${log_file}" | sed 's/^/    /' >>"${SUMMARY}" 2>/dev/null
    echo >>"${SUMMARY}"
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

exit $(( ${#failed[@]} > 0 ))
