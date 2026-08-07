#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the spyre-inference repo for the full license text.

"""KV-offload benefit as a concurrency curve: N concurrent requests on a fixed pool.

The other ttft_* scripts here all run ``max_num_seqs=1`` -- one request at a time --
so none of them can answer the question this one exists for: on a device KV pool that
does NOT grow with load, what happens as concurrent requests cross the pool's
capacity, and does offloading change it?

**Why the pool is pinned.** ``platform.py`` auto-sizes ``num_gpu_blocks_override`` to
``max_num_seqs * ceil(max_model_len / block_size)`` when left unset. That grows
capacity in lockstep with concurrency, so nothing is ever preempted and the curve
comes out flat with offloading idle -- a null result that looks like a working
experiment. Pinning the pool to ``--capacity-reqs`` requests is what creates a
capacity to cross, and it matches a real card, whose HBM does not grow with traffic.

**What crossing capacity costs.** vLLM's ``_preempt_request`` frees the victim's
blocks and resets ``num_computed_tokens`` to 0, so without a connector a preempted
request re-prefills every token. With the connector its blocks go to host RAM and
come back over PCIe. Measured single-request TTFT at 4096 tokens: ~25.1s to
re-prefill against ~1.3s to reload.

**Three arms.** Run this once per arm at each concurrency:

    --no-prefix-caching --no-kv-offload   nocache: floor, no reuse anywhere
    --prefix-caching --no-kv-offload      cache:   the honest baseline
    --prefix-caching --kv-offload         offload: the thing being sold

Three rather than two so the win cannot be attributed to prefix caching: the
cache->offload gap is the connector's contribution alone. Offload-on with
prefix-caching-off is not an arm -- with no reusable blocks nothing is offloaded.

**Rounds.** Round 1 is discarded (first-shape compile), round 2 is reported
separately (first partial-hit shape, which pays its own one-time compile), rounds 3+
are the steady state. ``ttft_evict.py`` measured that second-round effect at 28.3s
against a 4.4s steady state, which is why it is not averaged in.

**KNOWN BLOCKER (2026-08-07): round 2 dies at N>=2.**

Round 1 completes and reports correctly. Round 2 -- the first pass with *partial*
cache hits -- kills the engine with ``RecursionError: maximum recursion depth
exceeded`` after ~8s, raised under ``torch_spyre/_inductor/customops.py`` in the
``overwrite`` op, preceded by ``torch._dynamo hit
config.accumulated_recompile_limit (256)``.

Cause: ``spyre_attn.py:257-258`` writes the KV cache with one ``_overwrite`` call per
token per K/V at a distinct ``block_offsets[t]``, and that op compiles one SDSC binary
per unique offset. Round 1 writes sequentially from offset 0 and reuses binaries;
round 2 scatters across N sequences' partial hits, exhausting dynamo's accumulated
recompile budget. No existing offload test sees this because they all run
``max_num_seqs=1`` -- a single offset, no storm.

Raising the limit is NOT the fix, and ``spyre_attn.py:925-932`` already says why:
"Raising the limit unblocks short tests but compiles N binaries for a query_len=N
prefill, which doesn't scale to long contexts." The attention *output* scatter was
solved by staging on CPU; the *KV write* still uses ``overwrite``. Note also that
``accumulated_recompile_limit`` has no env-var knob and is re-defaulted in every
process that imports torch_spyre, so it cannot be raised from this script at all.

Unblocked by symbolic-offset overwrite: torch-spyre#220 / #1371-3.

Usage:
    cd /home/yuezhu/dt-inductor/spyre-inference
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 uv run --no-sync python \
        tests/v1/kv_offload/ttft_concurrency.py --length 4096 --concurrency 12
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

# Spyre KV cache block size (tokens per block). Must be a multiple of 64 -- the
# platform rounds up otherwise (TorchSpyrePlatform.check_and_update_config).
BLOCK_SIZE = 128

# Staging headroom on top of the sized capacity. Matches ttft_evict.py.
SLACK_BLOCKS = 2

# One filler word per concurrent request. Distinct words mean the prompts share no
# leading token, so no request can serve as a prefix-cache hit for another and the
# curve measures pool pressure rather than incidental prefix reuse.
FILLERS = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliett",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
    "quebec",
    "romeo",
    "sierra",
    "tango",
    "uniform",
    "victor",
    "whiskey",
    "xray",
    "yankee",
    "zulu",
]

# Emitted per transfer by SpyreOffloadingWorker. Must stay in sync with the
# logger.info call in spyre_inference/v1/kv_offload/worker.py.
TRANSFER_LINE = re.compile(
    r"SpyreOffloadingWorker (device->host|host->device):\s+job=\d+\s+num_blocks=(\d+)"
)

# Why there is no preemption counter here.
#
# The obvious pressure signal is vLLM's ``vllm:num_preemptions``, but reading it
# requires ``disable_log_stats=False``, and enabling stats crashes with this spec:
# the offloading metrics registry is built from ``spec_cls.build_metric_definitions()``
# (vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py), which
# SpyreOffloadingSpec inherits from the base as ``{}`` while its manager still reports
# metric keys at runtime -- so ``observe()`` trips ``assert key in
# self._offloading_metric_defs``. Log-scraping "Preemptions: N" is no escape: that line
# comes from LoggingStatLogger, part of the same stats machinery.
#
# So pressure is inferred from per-request ``num_cached_tokens`` instead, which needs no
# stats. This is arguably the better signal anyway: a preempted request has its blocks
# freed and ``num_computed_tokens`` reset to 0, so its prefix-cache hit collapses on
# retry. Cached tokens falling in rounds 2+ IS the lost-prefix cost that offloading
# exists to prevent -- the effect rather than the mechanism.


def log(msg: str) -> None:
    print(msg, flush=True)


def spyre_available() -> bool:
    try:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        # torch must be imported BEFORE torch_spyre: torch_spyre imports torch, whose
        # backend autoload calls back into torch_spyre._autoload. Importing
        # torch_spyre first hits that half-initialized module and raises
        # "partially initialized module ... has no attribute '_autoload'".
        import torch
        import torch_spyre  # noqa: F401

        return torch.spyre.device_count() > 0
    except Exception as e:  # noqa: BLE001
        log(f"[spyre] availability check failed: {e!r}")
        return False


def make_prompt_exactly(tokenizer, num_tokens: int, filler: str) -> str:
    """Build a prompt that tokenizes to EXACTLY ``num_tokens`` tokens."""
    approx = (filler + " ") * (num_tokens + 64)
    ids = tokenizer.encode(approx, add_special_tokens=False)
    assert len(ids) >= num_tokens, f"filler {filler!r} gave {len(ids)} tokens"
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
    raise AssertionError(f"could not converge to exactly {num_tokens} tokens")


class TransferCounter:
    """Sums SpyreOffloadingWorker transferred BLOCKS at the file-descriptor level.

    The offloading worker runs inside the VllmWorker-0 subprocess, so its counters are
    unreachable from here and its log records never enter our logging tree. What does
    reach us is the subprocess's stdout/stderr, which vLLM forwards to ours -- so
    transfers are read by redirecting fd 1/2 to a temp file and parsing the worker's
    own log lines.

    Counts BLOCKS, not lines: one job batches many blocks, so counting lines
    undercounts by orders of magnitude and reads as though nothing moved.
    """

    def __init__(self):
        # noqa SIM115: the handle must outlive __init__ -- fd 1/2 stay redirected into
        # it for the life of the run and it is closed in close().
        self._tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w+", suffix=".ttft_concurrency.log", delete=False
        )
        self._path = self._tmp.name
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)
        os.dup2(self._tmp.fileno(), 1)
        os.dup2(self._tmp.fileno(), 2)
        self._offset = 0
        self.total_stored = 0
        self.total_loaded = 0

    def delta(self) -> tuple[int, int]:
        # Flush Python buffers and the OS file first, or the newest lines are unwritten.
        sys.stdout.flush()
        sys.stderr.flush()
        with open(self._path, errors="replace") as fh:
            fh.seek(self._offset)
            text = fh.read()
            self._offset = fh.tell()

        stored = loaded = 0
        for direction, blocks in TRANSFER_LINE.findall(text):
            if direction == "device->host":
                stored += int(blocks)
            else:
                loaded += int(blocks)
        self.total_stored += stored
        self.total_loaded += loaded
        # Forward captured output to the real terminal: capturing must not swallow.
        os.write(self._saved_out, text.encode(errors="replace"))
        return stored, loaded

    def close(self) -> None:
        self.delta()
        os.dup2(self._saved_out, 1)
        os.dup2(self._saved_err, 2)
        os.close(self._saved_out)
        os.close(self._saved_err)
        self._tmp.close()
        log(f"[offload] full worker log kept at {self._path}")


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=4096, help="Prompt tokens per request.")
    p.add_argument(
        "--output-tokens",
        type=int,
        default=128,
        help=(
            "Tokens each request generates (default: 128). Not 1: with a single output "
            "token every request retires right after prefill, so requests barely "
            "overlap and there is no sustained contention to measure."
        ),
    )
    p.add_argument(
        "--concurrency",
        type=int,
        required=True,
        help="Prompts submitted at once. Also sets max_num_seqs so admission never caps it.",
    )
    p.add_argument(
        "--capacity-reqs",
        type=int,
        default=8,
        help="Pin the pool to hold this many requests (default: 8) -> knee at N+1.",
    )
    p.add_argument("--gpu-blocks", type=int, default=None, help="Raw pool override.")
    p.add_argument(
        "--rounds",
        type=int,
        default=4,
        help=(
            "Rounds of the same N prompts (default: 4). Round 1 cold, round 2 the "
            "first partial-hit shape, 3+ steady."
        ),
    )
    p.add_argument("--prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kv-offload", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu-bytes", type=int, default=2_000_000_000)
    p.add_argument(
        "--max-batched-tokens",
        type=int,
        default=None,
        help=(
            "max_num_batched_tokens. Default: concurrency * length. Hold this CONSTANT "
            "across a sweep -- if it varies, the scheduler's chunking threshold moves "
            "between points and can itself create a knee."
        ),
    )
    p.add_argument("--exec-timeout", type=int, default=7200)
    p.add_argument("--json", dest="json_path", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_req = args.concurrency

    # Must precede the vllm import: vllm.envs snapshots this at import time.
    if "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS" not in os.environ:
        os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = str(args.exec_timeout)

    if args.length < BLOCK_SIZE:
        log(
            f"[fatal] length {args.length} is shorter than one block ({BLOCK_SIZE}); "
            "only complete blocks are offloaded"
        )
        return 2
    if n_req > len(FILLERS):
        log(f"[fatal] --concurrency {n_req} exceeds the {len(FILLERS)} distinct fillers")
        return 2

    # Decode-inclusive: a request holds blocks for what it generates, not just its prompt.
    blocks_per_req = -(-(args.length + args.output_tokens) // BLOCK_SIZE)
    pool = (
        args.gpu_blocks
        if args.gpu_blocks is not None
        else args.capacity_reqs * blocks_per_req + SLACK_BLOCKS
    )
    capacity = (pool - SLACK_BLOCKS) // blocks_per_req
    if capacity < 1:
        log(f"[fatal] pool of {pool} blocks cannot hold one {args.length}-token request")
        return 2

    max_model_len = -(-(args.length + args.output_tokens) // BLOCK_SIZE) * BLOCK_SIZE
    max_batched = args.max_batched_tokens or n_req * args.length
    arm = (
        "offload"
        if args.kv_offload and args.prefix_caching
        else "cache"
        if args.prefix_caching
        else "nocache"
    )

    log(
        f"[config] arm={arm} concurrency={n_req} length={args.length} "
        f"output_tokens={args.output_tokens} blocks_per_request={blocks_per_req} "
        f"pool={pool} capacity={capacity} knee_at_N={capacity + 1} rounds={args.rounds}"
    )
    log(
        f"[config] prefix_caching={'on' if args.prefix_caching else 'off'} "
        f"kv_offload={'on' if args.kv_offload else 'off'} max_num_seqs={n_req} "
        f"max_num_batched_tokens={max_batched} max_model_len={max_model_len}"
    )
    log(
        f"[config] N={n_req} needs {n_req * blocks_per_req} blocks "
        f"({100.0 * n_req * blocks_per_req / pool:.1f}% of pool), "
        f"predicted victims={max(0, n_req - capacity)}"
    )
    if args.kv_offload and not args.prefix_caching:
        log(
            "[config] WARNING: kv_offload on with prefix_caching OFF -- no reusable "
            "blocks exist, so stored will stay 0. Not a valid arm."
        )

    if not spyre_available():
        log("[fatal] Spyre device not available")
        return 2

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    kv_transfer_config = None
    if args.kv_offload:
        from vllm.config import KVTransferConfig

        kv_transfer_config = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SpyreOffloadingSpec",
                # Import the spec module directly rather than relying on the
                # vllm.general_plugins entry point: the worker builds the connector in
                # initialize_from_config, which can run before load_general_plugins()
                # has registered the spec, and the registry lookup then fails.
                "spec_module_path": "spyre_inference.v1.kv_offload.spec",
                "cpu_bytes_to_use": args.cpu_bytes,
            },
        )

    log("[build] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompts = [make_prompt_exactly(tokenizer, args.length, FILLERS[i]) for i in range(n_req)]
    firsts = {tokenizer.encode(p, add_special_tokens=False)[0] for p in prompts}
    assert len(firsts) == n_req, "prompts share a leading token; prefix cache would leak"
    log(f"[build] {n_req} distinct prompts, each exactly {args.length} tokens")

    # fd redirection must be in place BEFORE the worker subprocess is forked, or its
    # stdout inherits the real terminal and the transfer lines never reach our file.
    counter = TransferCounter() if args.kv_offload else None

    log("[engine] constructing LLM...")
    t0 = time.perf_counter()
    try:
        llm = LLM(
            MODEL,
            max_model_len=max_model_len,
            max_num_seqs=n_req,
            max_num_batched_tokens=max_batched,
            num_gpu_blocks_override=pool,
            enable_prefix_caching=args.prefix_caching,
            # Stats stay DISABLED (the LLM default). Enabling them crashes with this
            # spec -- see the note above the percentile helper.
            attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
            kv_transfer_config=kv_transfer_config,
        )
    except Exception as e:  # noqa: BLE001
        if counter is not None:
            counter.close()
        log(f"[fatal] engine construction FAILED after {time.perf_counter() - t0:.1f}s: {e!r}")
        log(traceback.format_exc())
        return 1
    log(f"[engine] ready in {time.perf_counter() - t0:.1f}s")

    cache_config = llm.llm_engine.vllm_config.cache_config
    allocated = cache_config.num_gpu_blocks
    sched = llm.llm_engine.vllm_config.scheduler_config
    log(
        f"[alloc] num_gpu_blocks={allocated} (asked {pool}) "
        f"max_num_seqs={sched.max_num_seqs} "
        f"max_num_batched_tokens={sched.max_num_batched_tokens}"
    )
    if allocated != pool:
        if counter is not None:
            counter.close()
        log(
            f"[fatal] pool pin FAILED: asked {pool}, got {allocated}. A clamped pool "
            "moves the knee silently. Lower --capacity-reqs or --length."
        )
        return 2
    if sched.max_num_seqs < n_req:
        if counter is not None:
            counter.close()
        log(
            f"[fatal] max_num_seqs resolved to {sched.max_num_seqs} < concurrency "
            f"{n_req}; admission would cap the batch and the curve would show an "
            "admission cliff rather than a KV-capacity cliff."
        )
        return 2

    sampling = SamplingParams(
        temperature=0.0, max_tokens=args.output_tokens, min_tokens=args.output_tokens
    )

    if counter is not None:
        s, l_ = counter.delta()
        log(f"[offload] during startup: stored_blocks={s} loaded_blocks={l_}")

    rounds: list[dict] = []
    for r in range(args.rounds):
        tag = f"round{r + 1}"
        log(f"[run] {tag}: submitting {n_req} prompts...")
        start = time.perf_counter()
        try:
            outs = llm.generate(prompts, sampling)
        except Exception as e:  # noqa: BLE001
            log(f"[fatal] {tag} FAILED after {time.perf_counter() - start:.1f}s: {e!r}")
            # Print the traceback: a bare AssertionError() carries no message, so
            # without this the failure site is unrecoverable.
            log(traceback.format_exc())
            rounds.append({"tag": tag, "error": repr(e)})
            break
        e2e = time.perf_counter() - start

        # Prefix-cache hits across the batch. A preempted request loses its blocks and
        # restarts from 0 computed tokens, so this falling in rounds 2+ is the lost-cache
        # cost under pressure -- the thing offloading is supposed to prevent.
        cached = sum(o.num_cached_tokens or 0 for o in outs)
        prompt_total = sum(len(o.prompt_token_ids) for o in outs)

        per_req = []
        for o in outs:
            m = getattr(o, "metrics", None)
            arrival = getattr(m, "arrival_time", None) if m else None
            finished = getattr(m, "finished_time", None) if m else None
            if arrival and finished:
                per_req.append(finished - arrival)
        if not per_req:
            # No metrics surface: every request shares the batch wall-clock.
            per_req = [e2e] * len(outs)

        gen = sum(len(o.outputs[0].token_ids) for o in outs)
        row = {
            "tag": tag,
            "e2e_s": e2e,
            "cached_tokens": cached,
            "prompt_tokens": prompt_total,
            "cache_hit_frac": cached / prompt_total if prompt_total else 0.0,
            "gen_tokens": gen,
            "tokens_per_s": gen / e2e if e2e > 0 else 0.0,
            "requests_per_s": len(outs) / e2e if e2e > 0 else 0.0,
            "p50_s": percentile(per_req, 50),
            "p90_s": percentile(per_req, 90),
        }
        if counter is not None:
            row["stored_blocks"], row["loaded_blocks"] = counter.delta()
        rounds.append(row)

        log(
            f"[result] {tag}: e2e={e2e:.3f}s tok/s={row['tokens_per_s']:.1f} "
            f"p50={row['p50_s']:.3f}s p90={row['p90_s']:.3f}s "
            f"cached={cached}/{prompt_total} ({100 * row['cache_hit_frac']:.1f}%)"
        )
        if counter is not None:
            log(
                f"[offload] {tag}: stored_blocks={row['stored_blocks']} "
                f"loaded_blocks={row['loaded_blocks']}"
            )

    # Restore real stdout BEFORE the summary: while capture is active every log() lands
    # in the temp file and would be lost when the process exits.
    total_stored = total_loaded = None
    if counter is not None:
        total_stored, total_loaded = counter.total_stored, counter.total_loaded
        counter.close()

    ok = [r for r in rounds if "error" not in r]
    steady = [r["e2e_s"] for r in ok[2:]]
    # Rounds 2+ are where a cache hit is possible at all: round 1 populates the pool.
    reuse = ok[1:]

    log("")
    log(
        f"[summary] arm={arm} concurrency={n_req} length={args.length} pool={pool} "
        f"capacity={capacity} rounds_ok={len(ok)}"
    )
    log(
        f"[summary] {'round':>8} {'e2e(s)':>9} {'tok/s':>9} {'p50(s)':>8} "
        f"{'p90(s)':>8} {'cached%':>8}"
    )
    for r in rounds:
        if "error" in r:
            log(f"[summary] {r['tag']:>8} {'FAILED':>9}  {r['error']}")
            continue
        log(
            f"[summary] {r['tag']:>8} {r['e2e_s']:>9.3f} {r['tokens_per_s']:>9.1f} "
            f"{r['p50_s']:>8.3f} {r['p90_s']:>8.3f} "
            f"{100 * r['cache_hit_frac']:>7.1f}%"
        )

    if ok:
        log(f"[summary] cold(round1)={ok[0]['e2e_s']:.3f}s")
    if len(ok) > 1:
        log(f"[summary] first_hit(round2)={ok[1]['e2e_s']:.3f}s")
    if steady:
        log(f"[summary] steady(round3+) median={percentile(steady, 50):.3f}s (n={len(steady)})")
    reuse_frac = (
        sum(r["cache_hit_frac"] for r in reuse) / len(reuse) if reuse else 0.0
    )
    log(f"[summary] mean cache_hit_frac over rounds 2+ = {100 * reuse_frac:.1f}% (n={len(reuse)})")
    if total_stored is not None:
        log(f"[summary] cumulative BLOCKS stored={total_stored} loaded={total_loaded}")
    else:
        log("[summary] cumulative BLOCKS stored=n/a loaded=n/a (no connector in this arm)")

    # Each verdict names a way this point could have silently become meaningless.
    over_capacity = n_req > capacity
    if not ok:
        log("[verdict] NO round completed -- this point measured nothing. The counters "
            "above are startup-only and say nothing about offload behaviour; see the "
            "traceback for the failure.")
    elif not over_capacity:
        log(
            f"[verdict] N={n_req} fits capacity {capacity}, so no pressure was applied. "
            "Cache hits over rounds 2+ should be high and, in the offload arm, "
            "stored/loaded should stay 0 -- transfers here would mean the connector is "
            "active under no pressure."
        )
    elif total_stored is None:
        log(
            f"[verdict] N={n_req} over capacity {capacity} with no connector: displaced "
            "blocks were dropped and recomputed. The cache_hit_frac above is the "
            "baseline that the offload arm has to beat."
        )
    elif total_stored == 0:
        log(
            f"[verdict] WARNING: N={n_req} exceeds capacity {capacity} but NOTHING was "
            "offloaded. Only complete blocks are stored; check prefix caching, the host "
            "pool size, and that the pool pin above really bit."
        )
    elif total_loaded == 0:
        log(
            "[verdict] blocks were offloaded but never read back -- displaced blocks "
            "were recomputed instead of reloaded."
        )
    else:
        log(
            f"[verdict] offload AND reload both exercised under real pressure "
            f"(stored={total_stored} loaded={total_loaded} blocks)."
        )

    if args.json_path:
        with open(args.json_path, "w") as fh:
            json.dump(
                {
                    "arm": arm,
                    "concurrency": n_req,
                    "length": args.length,
                    "output_tokens": args.output_tokens,
                    "pool_blocks": pool,
                    "blocks_per_request": blocks_per_req,
                    "capacity_reqs": capacity,
                    "max_num_batched_tokens": sched.max_num_batched_tokens,
                    "prefix_caching": args.prefix_caching,
                    "kv_offload": args.kv_offload,
                    "rounds": rounds,
                    "steady_median_s": percentile(steady, 50) if steady else None,
                    "mean_cache_hit_frac_rounds2plus": reuse_frac,
                    "total_stored_blocks": total_stored,
                    "total_loaded_blocks": total_loaded,
                },
                fh,
                indent=2,
            )
        log(f"[summary] json written to {args.json_path}")

    return 1 if any("error" in r for r in rounds) else 0


if __name__ == "__main__":
    sys.exit(main())
