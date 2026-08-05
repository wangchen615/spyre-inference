#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the spyre-inference repo for the full license text.

"""TTFT sweep across prompt lengths: 1K / 2K / 8K / 16K / 32K / 64K / 128K.

Design (one engine, many prompt lengths):

  * ``max_model_len`` is set ONCE to the model's full context window (131072), so
    a single engine serves every prompt length -- no restart between lengths.
  * For EACH length L we send three prompts:
        1. a warmup prompt at L      -> absorbs the one-time Spyre/Inductor
                                        compile for L's kernel shape; discarded
        2. a measured prompt at L    -> "cold"  (run1)
        3. the SAME measured prompt  -> "warm"  (run2)
  * Every prompt uses DIFFERENT text content, so nothing interferes:
      - warmup vs measured at the same L differ  -> the warmup never seeds the
        KV cache for the measured prompt
      - prompts at different L differ            -> no shared prefix across
        lengths, so 2K's run cannot be warmed by 1K's run
    This matters specifically when ``--prefix-caching`` is on: vLLM caches by
    token-prefix, so any shared leading text would leak a cache hit and corrupt
    the cold measurement.

Why the warmup is the same length as its measured prompt: ``spyre_attn.py`` pads
KV to ``KV_LENGTH_ALIGNMENT=256`` and specializes the compiled kernel per
``(num_blocks, padded_query_len)``. A warmup at a different length compiles a
different kernel and absorbs nothing.

Measured session data at 1024 tokens (see ttft_1k_warmup.log):
    warmup (compile + prefill) = 233.2s
    steady-state run           =  23.1-23.9s
So compile ~= 210s and prefill ~= 23s at 1K. Attention prefill is O(n^2), so the
128K prefill is the long pole -- hence a large default timeout. See
``--exec-timeout`` below; the sweep prints observed times as it goes so you get
real scaling data at 8K/16K/32K before 128K is attempted.

Usage:
    cd /home/yuezhu/workspace/spyre-inference
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 uv run --no-sync python \
        /home/yuezhu/workspace/test/ttft_sweep.py \
        2>&1 | tee /home/yuezhu/workspace/test/ttft_sweep.log

    # subset of lengths, prefix caching on:
    ... ttft_sweep.py --lengths 1024,2048 --prefix-caching
"""

import argparse
import os
import sys
import time


MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

# Spyre KV cache block size (tokens per block).
BLOCK_SIZE = 128

# The model's declared context window (config.json max_position_embeddings).
MAX_POSITION_EMBEDDINGS = 131072

# One token is always reserved for generation, so the largest prompt is N-1.
DEFAULT_LENGTHS = [1024, 2048, 8192, 16384, 32768, 65536, MAX_POSITION_EMBEDDINGS - 1]

# Distinct filler words give every (length, role) pair different text content.
# Index into this per length; warmup and measured use different entries.
FILLER_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]


def log(msg: str) -> None:
    """Print immediately (unbuffered) so progress streams live."""
    print(msg, flush=True)


def spyre_available() -> bool:
    try:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        import torch_spyre
        torch_spyre._autoload()
        import torch
        return torch.spyre.device_count() > 0
    except Exception as e:  # noqa: BLE001
        log(f"[spyre] availability check failed: {e!r}")
        return False


def make_prompt_exactly(tokenizer, num_tokens: int, filler: str) -> str:
    """Build a prompt that tokenizes to EXACTLY ``num_tokens`` tokens.

    ``filler`` is a single word (no trailing space) repeated to length. Distinct
    fillers yield prompts that share no leading tokens, which is what keeps the
    prefix cache from leaking between roles and between lengths.
    """
    approx_text = (filler + " ") * (num_tokens + 64)
    ids = tokenizer.encode(approx_text, add_special_tokens=False)
    assert len(ids) >= num_tokens, (
        f"filler {filler!r} tokenized to {len(ids)} tokens, need >= {num_tokens}; "
        "increase the overshoot factor"
    )
    ids = ids[:num_tokens]
    text = tokenizer.decode(ids)

    # Decode->re-encode can drift by a token or two at boundaries; converge.
    for _ in range(16):
        reids = tokenizer.encode(text, add_special_tokens=False)
        if len(reids) == num_tokens:
            return text
        if len(reids) > num_tokens:
            reids = reids[:num_tokens]
        else:
            reids = reids + ids[len(reids):num_tokens]
        text = tokenizer.decode(reids)
    raise AssertionError(
        f"could not converge to exactly {num_tokens} tokens (last was {len(reids)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths",
        type=str,
        default=",".join(str(n) for n in DEFAULT_LENGTHS),
        help=(
            "Comma-separated prompt lengths in tokens. Default: "
            f"{','.join(str(n) for n in DEFAULT_LENGTHS)} (1K..128K; the last is "
            "131071 = the 128K window minus the 1 generated token)."
        ),
    )
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable vLLM prefix caching. ON: run2 should hit the KV cache and be "
            "much faster than run1. OFF: both runs recompute the prefill "
            "(default: disabled)."
        ),
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Send a throwaway same-length prompt before each length's measured "
            "runs, so the one-time compile is not charged to run1 "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=4,
        help=(
            "Measured runs per length, all sending the SAME prompt (default: 4). "
            "run1 is the cold one; run2..runN show whether run2's number is "
            "representative or a one-off transition cost. Two runs cannot tell "
            "those apart, which is why the default is 4."
        ),
    )
    parser.add_argument(
        "--kv-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Attach the Spyre KV-offload connector (OffloadingConnector + "
            "SpyreOffloadingSpec) with cpu_bytes_to_use=--cpu-bytes. NOTE: the "
            "connector only has blocks worth fetching back when --prefix-caching "
            "is also on; with caching off this measures the connector's pure "
            "overhead on a no-reuse path (default: disabled)."
        ),
    )
    parser.add_argument(
        "--cpu-bytes",
        type=int,
        default=2_000_000_000,
        help=(
            "Host RAM budget for offloaded KV blocks, in bytes (default: 2e9 = 2GB). "
            "SpyreOffloadingSpec derives its host block count as "
            "cpu_bytes_to_use // kv_bytes_per_block, so too small a value yields "
            "num_blocks=0 and silently offloads nothing. Only used with --kv-offload."
        ),
    )
    parser.add_argument(
        "--exec-timeout",
        type=int,
        default=7200,
        help=(
            "Sets VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS if not already in the "
            "environment. This is a PER-execute_model-call deadline, so it must "
            "cover the single slowest call (the 128K prefill). vLLM's default of "
            "300 is far too low: the 1K compile alone took ~233s. Default 7200 "
            "(2h). NOTE: extrapolated from one data point -- attention prefill is "
            "O(n^2), so 128K may need more; watch the 8K/16K/32K numbers."
        ),
    )
    args = parser.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    gen_tokens = 1

    for n in lengths:
        if n + gen_tokens > MAX_POSITION_EMBEDDINGS:
            log(
                f"[fatal] length {n} + gen={gen_tokens} exceeds "
                f"max_position_embeddings={MAX_POSITION_EMBEDDINGS}"
            )
            return 2

    # The execute_model RPC deadline must be set BEFORE vllm is imported, since
    # vllm.envs snapshots it at import time.
    if "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" not in os.environ:
        os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.exec_timeout)
    log(
        "[config] VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="
        f"{os.environ['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS']}"
    )

    # One engine for every length: size the window to the model's full context.
    max_model_len = MAX_POSITION_EMBEDDINGS
    num_gpu_blocks = (max_model_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    log(
        f"[config] lengths={lengths} max_model_len={max_model_len} "
        f"block_size={BLOCK_SIZE} num_gpu_blocks_override={num_gpu_blocks} "
        f"gen_tokens={gen_tokens} warmup={'on' if args.warmup else 'off'} "
        f"prefix_caching={'on' if args.prefix_caching else 'off'}"
    )

    if not spyre_available():
        log("[fatal] Spyre device not available")
        return 2

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # KV offload is opt-in. `SpyreOffloadingSpec` is registered by
    # `spyre_inference/__init__.py` at plugin load, so the spec_name below
    # resolves without importing anything Spyre-specific here.
    kv_transfer_config = None
    if args.kv_offload:
        from vllm.config import KVTransferConfig

        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SpyreOffloadingSpec",
                "cpu_bytes_to_use": args.cpu_bytes,
            },
        )
        log(
            f"[config] kv_offload=on connector=OffloadingConnector "
            f"spec=SpyreOffloadingSpec cpu_bytes_to_use={args.cpu_bytes}"
        )
        if not args.prefix_caching:
            log(
                "[config] NOTE: kv_offload is on but prefix_caching is OFF -- there "
                "are no reusable blocks to fetch back, so this measures connector "
                "OVERHEAD, not offload benefit."
            )
    else:
        log("[config] kv_offload=off")

    log("[build] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Build every prompt up front so a tokenization failure surfaces before the
    # (slow) engine construction. Each (length, role) gets its own filler word,
    # so no two prompts anywhere in the sweep share a leading token.
    prompts: dict[int, dict[str, str]] = {}
    for idx, n in enumerate(lengths):
        warm_filler = FILLER_WORDS[(2 * idx) % len(FILLER_WORDS)]
        meas_filler = FILLER_WORDS[(2 * idx + 1) % len(FILLER_WORDS)]
        entry: dict[str, str] = {}
        if args.warmup:
            entry["warmup"] = make_prompt_exactly(tokenizer, n, warm_filler)
        entry["measured"] = make_prompt_exactly(tokenizer, n, meas_filler)
        prompts[n] = entry

        for role, text in entry.items():
            got = len(tokenizer.encode(text, add_special_tokens=False))
            assert got == n, f"{role} prompt for {n} tokenized to {got}"
        if args.warmup:
            assert entry["warmup"] != entry["measured"]
        log(
            f"[build] len={n}: warmup filler={warm_filler!r} "
            f"measured filler={meas_filler!r} (both exactly {n} tokens)"
        )

    log("[engine] constructing LLM (loads weights + warms up)...")
    t_engine = time.perf_counter()
    model = LLM(
        MODEL,
        max_model_len=max_model_len,
        max_num_seqs=1,
        num_gpu_blocks_override=num_gpu_blocks,
        enable_prefix_caching=args.prefix_caching,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
        kv_transfer_config=kv_transfer_config,
    )
    log(f"[engine] LLM ready in {time.perf_counter() - t_engine:.1f}s")

    try:
        cache_config = model.llm_engine.vllm_config.cache_config
        log(
            f"[alloc] block_size={cache_config.block_size} "
            f"num_gpu_blocks={getattr(cache_config, 'num_gpu_blocks', 'n/a')}"
        )
    except Exception as e:  # noqa: BLE001
        log(f"[alloc] (could not read cache_config: {e!r})")

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=gen_tokens, min_tokens=gen_tokens
    )

    results: list[dict] = []
    for n in lengths:
        log(f"[len] ===== {n} tokens =====")
        entry = prompts[n]

        if args.warmup:
            log(f"[warmup] len={n}: sending throwaway prompt...")
            t = time.perf_counter()
            try:
                model.generate(prompts=entry["warmup"], sampling_params=sampling_params)
            except Exception as e:  # noqa: BLE001
                log(f"[fatal] len={n}: warmup FAILED after "
                    f"{time.perf_counter() - t:.1f}s: {e!r}")
                results.append({"len": n, "error": f"warmup: {e!r}"})
                break
            log(
                f"[warmup] len={n}: done in {time.perf_counter() - t:.3f}s "
                "(NOT a measurement -- discarded)"
            )

        ttfts = []
        failed = False
        for i in range(args.runs):
            label = f"run{i + 1}-{'cold' if i == 0 else 'warm'}"
            log(f"[run] len={n} {label}: sending prompt...")
            start = time.perf_counter()
            try:
                out = model.generate(
                    prompts=entry["measured"], sampling_params=sampling_params
                )
            except Exception as e:  # noqa: BLE001
                log(f"[fatal] len={n} {label} FAILED after "
                    f"{time.perf_counter() - start:.1f}s: {e!r}")
                # Keep whatever completed before the failure -- a run that dies
                # on run3 still has valid run1/run2 numbers, and those are the
                # expensive ones to reproduce.
                partial = {"len": n, "error": f"{label}: {e!r}", "runs": list(ttfts)}
                if ttfts:
                    partial["cold"] = ttfts[0]
                results.append(partial)
                failed = True
                break
            ttft = time.perf_counter() - start
            ttfts.append(ttft)
            log(
                f"[result] len={n} {label}: "
                f"prompt_tokens={len(out[0].prompt_token_ids)} ttft={ttft:.3f}s "
                f"output={out[0].outputs[0].text!r}"
            )
            # Running tally on one grep-able line, emitted after EVERY run, so a
            # later crash never costs us the earlier numbers. Mirrors the final
            # [summary] table's columns.
            _so_far = " ".join(f"run{j + 1}={t:.3f}" for j, t in enumerate(ttfts))
            _warm = ttfts[1:]
            _wmed = ""
            if _warm:
                _o = sorted(_warm)
                _m = len(_o) // 2
                _med = _o[_m] if len(_o) % 2 else (_o[_m - 1] + _o[_m]) / 2
                _wmed = (f" warm_med={_med:.3f} warm_min={_o[0]:.3f} "
                         f"warm_max={_o[-1]:.3f} cold/warm={ttfts[0] / _med:.2f}x"
                         if _med > 0 else "")
            log(f"[ttft] len={n} {len(ttfts)}/{args.runs} cold={ttfts[0]:.3f} "
                f"{_so_far}{_wmed}")
        if failed:
            break

        # "cold" is run1; the steady state is the median of run2..runN, which is
        # robust to a single anomalous run2 in a way a 2-run mean is not.
        warm_runs = ttfts[1:]
        row = {"len": n, "cold": ttfts[0], "runs": list(ttfts)}
        if warm_runs:
            ordered = sorted(warm_runs)
            mid = len(ordered) // 2
            row["warm_median"] = (
                ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
            )
            row["warm_min"] = ordered[0]
            row["warm_max"] = ordered[-1]
            row["ratio"] = (
                ttfts[0] / row["warm_median"] if row["warm_median"] > 0 else float("nan")
            )
        results.append(row)
        log(
            f"[len] {n}: cold={row['cold']:.3f}s "
            f"warm=[{', '.join(f'{t:.3f}' for t in warm_runs)}]s "
            f"median={row.get('warm_median', float('nan')):.3f}s "
            f"cold/warm={row.get('ratio', float('nan')):.2f}x"
        )

    log("")
    log(
        f"[summary] warmup={'on' if args.warmup else 'off'} "
        f"prefix_caching={'on' if args.prefix_caching else 'off'} "
        f"kv_offload={'on' if args.kv_offload else 'off'}"
    )
    log("[summary] per-run TTFT (run1 = cold, rest = warm):")
    for row in results:
        per_run = "  ".join(f"run{i + 1}={t:.3f}" for i, t in enumerate(row.get("runs", [])))
        if "error" in row:
            # Show partial numbers alongside the error rather than dropping them.
            got = f"{per_run}  " if per_run else ""
            log(f"[summary] {row['len']:>8}  PARTIAL  {got}FAILED  {row['error']}")
        else:
            log(f"[summary] {row['len']:>8}  {per_run}")
    log("")
    log(
        f"[summary] {'tokens':>8}  {'cold(s)':>9}  {'warm_med':>9}  "
        f"{'warm_min':>9}  {'warm_max':>9}  {'cold/warm':>9}"
    )
    for row in results:
        if "cold" not in row:
            # Died before run1 completed -- genuinely nothing to report.
            if "error" in row:
                log(f"[summary] {row['len']:>8}  {'FAILED':>9}  (no completed runs)")
            continue
        # A partial row has a valid cold number and possibly warm runs; derive the
        # same stats so a crash on a later run still yields usable data.
        if "warm_median" not in row and len(row.get("runs", [])) > 1:
            _o = sorted(row["runs"][1:])
            _m = len(_o) // 2
            row["warm_median"] = _o[_m] if len(_o) % 2 else (_o[_m - 1] + _o[_m]) / 2
            row["warm_min"], row["warm_max"] = _o[0], _o[-1]
            row["ratio"] = (
                row["cold"] / row["warm_median"] if row["warm_median"] > 0 else float("nan")
            )
        mark = "  (partial)" if "error" in row else ""
        log(
            f"[summary] {row['len']:>8}  {row['cold']:>9.3f}  "
            f"{row.get('warm_median', float('nan')):>9.3f}  "
            f"{row.get('warm_min', float('nan')):>9.3f}  "
            f"{row.get('warm_max', float('nan')):>9.3f}  "
            f"{row.get('ratio', float('nan')):>8.2f}x{mark}"
        )
    # Reading the warm runs:
    #   warm_min ~= warm_max        -> steady state; run2 was representative
    #   warm_max >> warm_min        -> one anomalous run (e.g. a recompile on the
    #                                  first cache-hit shape); trust the median
    # With prefix caching OFF and warmup ON, cold/warm should be ~1.0 (nothing to
    # reuse). With caching ON, cold/warm > 1.0 is the KV-reuse benefit.
    if not any("error" in r for r in results) and len(results) > 1:
        log("[summary] cold-TTFT scaling vs the shortest length:")
        base = results[0]
        for row in results[1:]:
            log(
                f"[summary]   {row['len']:>8} = {row['len'] / base['len']:>6.1f}x tokens "
                f"-> {row['cold'] / base['cold']:>7.1f}x cold TTFT"
            )

    return 1 if any("error" in r for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
