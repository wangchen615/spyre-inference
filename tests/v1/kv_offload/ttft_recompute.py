#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the spyre-inference repo for the full license text.

"""TTFT baseline with NO KV connector: what an evicted prompt costs to recompute.

The companion script ``ttft_evict.py`` measures the *reload* path -- blocks pushed
to host by the offload connector and fetched back on the next run. This script is
its control: same model, same pool sizing, same prompt shape, but no
``kv_transfer_config`` at all. Evicted blocks are dropped and must be recomputed
from scratch, so its TTFT is the number reload has to beat.

Workload:

    warmup(alpha) -> run1(bravo) -> run2(bravo) -> run3(bravo) -> ...

The warmup on A absorbs the one-time compile for this prompt shape and leaves A
resident. run1 on B must then displace A. Every subsequent run re-sends **B**, not
an alternation -- with no connector there is nothing to reload, so alternating would
only measure two cold prefills taking turns. Holding the prompt fixed instead asks a
sharper question: once B's blocks are the only thing in the pool, does the run stay
warm? Compare against ``ttft_evict.py --length <same>``:

    this script, run2+   ~= device-resident prefix hit (nothing displaces B)
    ttft_evict.py, run3+  = host->device reload of a prompt evicted one run earlier

Both are "warm", but only the second one pays transfer cost, and only the second one
survives a second prompt sharing the pool.

**Reported buckets** are just two: ``run1`` is the cold miss on B, and ``run2+`` is
the warm plateau, reported as median / min / max. ``ttft_evict.py`` additionally
breaks out its run2, because there the prompt alternates and run2 is the first to hit
a partial-hit kernel shape that costs a one-time compile; here every measured run
sends the identical prompt, so run2 is an ordinary warm hit and belongs in the
plateau. ``--runs 9`` gives eight runs in the warm bucket.

**Pool sizing** matches ``ttft_evict.py`` exactly (``ceil(L / block_size) +
SLACK_BLOCKS``) so the two scripts' numbers are comparable at the same ``--length``.
The pool is deliberately too small for two prompts even though this variant only
sends one after warmup -- keeping it identical is what makes the comparison fair.

**Why there is no ``[offload]`` reporting.** No connector is constructed, so there
are no transfer counters to scrape and no worker log lines to parse. Anything
resembling offload activity here would be a bug; ``--kv-offload`` is not offered as
a flag precisely so this file cannot quietly become a second copy of
``ttft_evict.py``.

Usage:
    cd /home/yuezhu/dt-inductor/spyre-inference
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 uv run --no-sync python \
        tests/v1/kv_offload/ttft_recompute.py --length 1024 --runs 9

    # prefix caching off, so every run is a full cold prefill:
    ... ttft_recompute.py --length 1024 --no-prefix-caching
"""

import argparse
import os
import sys
import time

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

# Spyre KV cache block size (tokens per block). Must be a multiple of 64 -- the
# platform rounds up otherwise (TorchSpyrePlatform.check_and_update_config).
BLOCK_SIZE = 128

# Staging headroom on top of one prompt's blocks. Kept identical to ttft_evict.py so
# the two scripts size their pools the same way at the same --length.
SLACK_BLOCKS = 2

# FILLER_A is the warmup prompt; FILLER_B is every measured run. Distinct filler
# words means they share no leading token, so A cannot serve as a prefix-cache hit
# for B -- run1 is a genuine cold miss that has to displace A.
FILLER_A = "alpha"
FILLER_B = "bravo"


def log(msg: str) -> None:
    """Print immediately (unbuffered) so progress streams live."""
    print(msg, flush=True)


def spyre_available() -> bool:
    try:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        # torch must be imported BEFORE torch_spyre: torch_spyre imports torch,
        # whose backend autoload calls back into torch_spyre._autoload. Importing
        # torch_spyre first hits that half-initialized module and raises
        # "partially initialized module ... has no attribute '_autoload'".
        import torch
        import torch_spyre  # noqa: F401  (registers the `spyre` device via autoload)

        return torch.spyre.device_count() > 0
    except Exception as e:  # noqa: BLE001
        log(f"[spyre] availability check failed: {e!r}")
        return False


def make_prompt_exactly(tokenizer, num_tokens: int, filler: str) -> str:
    """Build a prompt that tokenizes to EXACTLY ``num_tokens`` tokens.

    ``filler`` is a single word (no trailing space) repeated to length. Distinct
    fillers yield prompts that share no leading tokens, which is what keeps the
    warmup prompt from serving as a prefix hit for the measured one.
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
            reids = reids + ids[len(reids) : num_tokens]
        text = tokenizer.decode(reids)
    raise AssertionError(
        f"could not converge to exactly {num_tokens} tokens (last was {len(reids)})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length",
        type=int,
        default=1024,
        help="Prompt length in tokens for both the warmup and measured prompts.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=9,
        help=(
            "Measured runs, all on prompt B, after a warmup on prompt A (default: 9). "
            "run1 is a cold miss that displaces A; later runs re-send the same B."
        ),
    )
    parser.add_argument(
        "--gpu-blocks",
        type=int,
        default=None,
        help=(
            "Device KV blocks. Default: ceil(length/block_size) + "
            f"{SLACK_BLOCKS}, matching ttft_evict.py so the two are comparable."
        ),
    )
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable vLLM prefix caching (default: ENABLED, matching ttft_evict.py). "
            "Turn it off to force every run to be a full cold prefill."
        ),
    )
    parser.add_argument(
        "--exec-timeout",
        type=int,
        default=7200,
        help=(
            "Sets VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS if not already set. Per-call "
            "deadline; vLLM's default of 300 is too low (the 1K compile alone took "
            "~233s). Default 7200 (2h)."
        ),
    )
    args = parser.parse_args()

    n = args.length
    gen_tokens = 1

    # The execute_model RPC deadline must be set BEFORE vllm is imported, since
    # vllm.envs snapshots it at import time.
    if "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" not in os.environ:
        os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.exec_timeout)

    blocks_per_prompt = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_gpu_blocks = (
        args.gpu_blocks if args.gpu_blocks is not None else blocks_per_prompt + SLACK_BLOCKS
    )

    if blocks_per_prompt < 1:
        log(f"[fatal] length {n} is shorter than one block ({BLOCK_SIZE} tokens)")
        return 2
    if num_gpu_blocks < blocks_per_prompt:
        log(
            f"[fatal] --gpu-blocks={num_gpu_blocks} cannot hold one {n}-token prompt "
            f"({blocks_per_prompt} blocks); the prompt would never fit"
        )
        return 2

    # max_model_len must cover the prompt plus the generated token.
    max_model_len = ((n + gen_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE

    log(
        f"[config] length={n} block_size={BLOCK_SIZE} "
        f"blocks_per_prompt={blocks_per_prompt} "
        f"num_gpu_blocks_override={num_gpu_blocks} "
        f"(= 1 prompt + {num_gpu_blocks - blocks_per_prompt} slack) "
        f"max_model_len={max_model_len} runs={args.runs}"
    )
    log(
        f"[config] prefix_caching={'on' if args.prefix_caching else 'off'} "
        f"kv_offload=off (NO connector -- this is the recompute baseline) "
        f"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="
        f"{os.environ['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS']}"
    )

    if not spyre_available():
        log("[fatal] Spyre device not available")
        return 2

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    log("[build] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    prompt_a = make_prompt_exactly(tokenizer, n, FILLER_A)
    prompt_b = make_prompt_exactly(tokenizer, n, FILLER_B)
    assert prompt_a != prompt_b
    for label, text in (("A", prompt_a), ("B", prompt_b)):
        got = len(tokenizer.encode(text, add_special_tokens=False))
        assert got == n, f"prompt {label} tokenized to {got}, want {n}"
    log(
        f"[build] warmup prompt A filler={FILLER_A!r} measured prompt B "
        f"filler={FILLER_B!r} (both exactly {n} tokens, no shared prefix)"
    )

    log("[engine] constructing LLM (loads weights + warms up)...")
    t_engine = time.perf_counter()
    model = LLM(
        MODEL,
        max_model_len=max_model_len,
        max_num_seqs=1,  # sequential: each run competes for the same blocks
        num_gpu_blocks_override=num_gpu_blocks,
        enable_prefix_caching=args.prefix_caching,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
        # No kv_transfer_config: that omission IS the experiment.
    )
    log(f"[engine] LLM ready in {time.perf_counter() - t_engine:.1f}s")

    try:
        cache_config = model.llm_engine.vllm_config.cache_config
        log(
            f"[alloc] block_size={cache_config.block_size} "
            f"num_gpu_blocks={getattr(cache_config, 'num_gpu_blocks', 'n/a')}"
        )
        kv_cfg = model.llm_engine.vllm_config.kv_transfer_config
        log(f"[alloc] kv_transfer_config={kv_cfg} (expected None)")
    except Exception as e:  # noqa: BLE001
        log(f"[alloc] (could not read config: {e!r})")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=gen_tokens, min_tokens=gen_tokens)

    # Warmup on A absorbs the one-time Spyre/Inductor compile for this prompt shape,
    # so run1 is not charged for it. It also leaves A resident, which is what makes
    # run1 (on B) have to displace something.
    log(f"[warmup] sending prompt A ({n} tokens) to absorb compile...")
    t = time.perf_counter()
    try:
        model.generate(prompts=prompt_a, sampling_params=sampling_params)
    except Exception as e:  # noqa: BLE001
        log(f"[fatal] warmup FAILED after {time.perf_counter() - t:.1f}s: {e!r}")
        return 1
    log(
        f"[warmup] done in {time.perf_counter() - t:.3f}s "
        "(NOT a measurement -- discarded; leaves A resident)"
    )

    rows: list[dict] = []
    for i in range(args.runs):
        tag = f"run{i + 1}-B"
        log(f"[run] {tag}: sending prompt B...")
        start = time.perf_counter()
        try:
            out = model.generate(prompts=prompt_b, sampling_params=sampling_params)
        except Exception as e:  # noqa: BLE001
            log(f"[fatal] {tag} FAILED after {time.perf_counter() - start:.1f}s: {e!r}")
            rows.append({"tag": tag, "error": repr(e)})
            break
        ttft = time.perf_counter() - start
        rows.append({"tag": tag, "ttft": ttft})
        log(
            f"[result] {tag}: prompt_tokens={len(out[0].prompt_token_ids)} "
            f"ttft={ttft:.3f}s output={out[0].outputs[0].text!r}"
        )

    log("")
    log(
        f"[summary] length={n} num_gpu_blocks={num_gpu_blocks} "
        f"blocks_per_prompt={blocks_per_prompt} "
        f"prefix_caching={'on' if args.prefix_caching else 'off'} "
        f"kv_offload=off"
    )
    log(f"[summary] {'run':>9}  {'prompt':>6}  {'ttft(s)':>9}")
    for row in rows:
        if "error" in row:
            log(f"[summary] {row['tag']:>9}  {'B':>6}  {'FAILED':>9}  {row['error']}")
            continue
        log(f"[summary] {row['tag']:>9}  {'B':>6}  {row['ttft']:>9.3f}")

    ok = [r for r in rows if "error" not in r]
    # Two buckets: run1 is the cold miss on B (prompt A is resident and shares no
    # prefix), and every run after it is warm. run2 is not broken out -- unlike
    # ttft_evict.py, where run2 switches prompts and hits a new partial-hit kernel
    # shape worth a one-time compile, every measured run here sends the identical
    # prompt, so run2 is an ordinary device-resident hit like the rest.
    if len(ok) > 1:
        cold = ok[0]["ttft"]
        warm = sorted(r["ttft"] for r in ok[1:])
        mid = len(warm) // 2
        median = warm[mid] if len(warm) % 2 else (warm[mid - 1] + warm[mid]) / 2
        log("")
        log(f"[summary] run1(cold miss on B)={cold:.3f}s")
        if median > 0:
            log(
                f"[summary] warm TTFT (run2+): median={median:.3f}s  "
                f"min={warm[0]:.3f}s  max={warm[-1]:.3f}s  "
                f"cold/warm={cold / median:.2f}x  (n={len(warm)})"
            )
        else:
            log("[summary] warm TTFT median is 0")
    log(
        "[summary] baseline: no connector, so nothing was offloaded or reloaded. "
        "Compare against ttft_evict.py at the same --length."
    )

    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
