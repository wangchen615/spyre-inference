#!/usr/bin/env python3
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See the spyre-inference repo for the full license text.

"""TTFT under KV eviction pressure: alternating A/B prompts on a one-prompt pool.

``ttft_sweep.py`` re-sends the SAME prompt for every measured run, so its blocks
are never displaced -- nothing is evicted and the offload connector has nothing
to fetch back. That measures cache *hits*, not offload/reload.

This script alternates two distinct prompts on a device pool sized for roughly one
prompt:

    warmup(alpha) -> run1(bravo) -> run2(alpha) -> run3(bravo)
                  -> run4(alpha) -> run5(bravo)

Each run needs blocks the pool is already full of, so the previous prompt must be
evicted to host; the run after that must read it back. Every measured run from run2
on is therefore a *reload* of a prompt that was pushed out one run earlier.

**Pool sizing.** ``num_gpu_blocks = ceil(L / block_size) + SLACK_BLOCKS``: enough for
one prompt plus a little staging room. Zero slack also forces eviction but leaves the
scheduler no room to stage blocks while the other prompt is still resident, which
tends to surface preemption rather than a clean offload/reload. Override with
``--gpu-blocks`` to explore other ratios (e.g. 1.5x a prompt for partial reloads).

**Why prefix caching and offloading are both required.** Without
``--prefix-caching`` vLLM has no reusable blocks to track, so nothing is ever
offloaded and every run recomputes from scratch. Without ``--kv-offload`` evicted
blocks are simply dropped and recomputed. Both default ON here -- unlike
``ttft_sweep.py``, where they are opt-in -- because neither is optional for this
measurement.

**Reading the output.** ``[offload]`` reports the connector's own cumulative
counters, scraped from the worker log:

    stored=0            -> no eviction happened; the pool is too large, or the
                           prompt is shorter than one block (only COMPLETE blocks
                           are stored)
    stored>0, loaded=0   -> blocks went to host but were recomputed instead of
                           fetched; prefix caching is off, or the alternation is
                           not actually displacing them
    stored>0, loaded>0   -> offload AND reload are both exercised (the goal)

TTFT then shows what the reload costs: compare ``run2..run5`` (reload path) against
``ttft_sweep.py``'s warm runs at the same length (device-resident hit).

Usage:
    cd /home/yuezhu/dt-inductor/spyre-inference
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 uv run --no-sync python \
        /home/yuezhu/workspace/test/ttft_evict.py --length 1024 \
        2>&1 | tee /home/yuezhu/workspace/test/ttft_evict.log

    # give the pool 1.5 prompts, so reloads are partial rather than total:
    ... ttft_evict.py --length 1024 --gpu-blocks 12
"""

import argparse
import os
import re
import sys
import tempfile
import time

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"

# Spyre KV cache block size (tokens per block). Must be a multiple of 64 -- the
# platform rounds up otherwise (TorchSpyrePlatform.check_and_update_config).
BLOCK_SIZE = 128

# Staging headroom on top of one prompt's blocks. See the module docstring.
SLACK_BLOCKS = 2

# The two prompts, alternated. Distinct filler words means they share no leading
# token, so neither can serve as a prefix-cache hit for the other -- the only way
# a repeat run gets its blocks back is from the host tier.
FILLER_A = "alpha"
FILLER_B = "bravo"

# Log fragments emitted per transfer by SpyreOffloadingWorker. Must stay in sync
# with the logger.info call in spyre_inference/v1/kv_offload/worker.py.
STORE_MARKER = "SpyreOffloadingWorker device->host"
LOAD_MARKER = "SpyreOffloadingWorker host->device"


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
    prefix cache from leaking between the two alternated prompts.
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


class TransferCounter:
    """Sums SpyreOffloadingWorker transferred BLOCKS at the file-descriptor level.

    The offloading worker runs inside the ``VllmWorker-0`` subprocess, so its
    ``stored_blocks`` / ``loaded_blocks`` attributes are unreachable from this
    process and its log records never enter our logging tree. What *does* reach us
    is the subprocess's stdout/stderr, which vLLM forwards to ours -- so the
    transfers are read by redirecting fd 1/2 to a temp file and parsing the
    worker's own log lines.

    Counts BLOCKS, not log lines. One transfer job batches many blocks (a 16K
    prompt reloads 127 blocks in a single job), so counting lines undercounts by
    two orders of magnitude and reads as though almost nothing moved. Job counts
    are tracked separately since the batching itself is worth seeing.

    ``delta()`` returns per-run (stored_blocks, loaded_blocks, store_jobs,
    load_jobs) since the previous call.
    """

    # "SpyreOffloadingWorker device->host: job=0 num_blocks=64 page_copies=512 ..."
    _LINE = re.compile(
        r"SpyreOffloadingWorker (device->host|host->device):"
        r"\s+job=\d+\s+num_blocks=(\d+)"
    )

    def __init__(self):
        self._tmp = tempfile.NamedTemporaryFile(
            mode="w+", suffix=".ttft_evict.log", delete=False
        )
        self._path = self._tmp.name
        self._saved_out = os.dup(1)
        self._saved_err = os.dup(2)
        os.dup2(self._tmp.fileno(), 1)
        os.dup2(self._tmp.fileno(), 2)
        self._offset = 0
        self.total_stored = 0
        self.total_loaded = 0
        self.total_store_jobs = 0
        self.total_load_jobs = 0

    def _read_new(self) -> str:
        # Flush both the Python-level buffers and the OS file before reading, or
        # the most recent lines are still sitting unwritten.
        sys.stdout.flush()
        sys.stderr.flush()
        with open(self._path, errors="replace") as fh:
            fh.seek(self._offset)
            text = fh.read()
            self._offset = fh.tell()
        return text

    def delta(self) -> tuple[int, int, int, int]:
        text = self._read_new()
        stored = loaded = store_jobs = load_jobs = 0
        for direction, blocks in self._LINE.findall(text):
            if direction == "device->host":
                stored += int(blocks)
                store_jobs += 1
            else:
                loaded += int(blocks)
                load_jobs += 1
        self.total_stored += stored
        self.total_loaded += loaded
        self.total_store_jobs += store_jobs
        self.total_load_jobs += load_jobs
        # Forward the captured output to the real terminal so the run still streams
        # live -- capturing must not mean swallowing.
        os.write(self._saved_out, text.encode(errors="replace"))
        return stored, loaded, store_jobs, load_jobs

    def close(self) -> None:
        self.delta()
        os.dup2(self._saved_out, 1)
        os.dup2(self._saved_err, 2)
        os.close(self._saved_out)
        os.close(self._saved_err)
        self._tmp.close()
        log(f"[offload] full worker log kept at {self._path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length",
        type=int,
        default=1024,
        help="Prompt length in tokens for BOTH alternated prompts (default: 1024).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help=(
            "Measured runs, alternating B/A/B/A/... after a warmup on A (default: 5). "
            "run1 is a cold miss on B; every later run is a reload of a prompt that "
            "the run before it evicted."
        ),
    )
    parser.add_argument(
        "--gpu-blocks",
        type=int,
        default=None,
        help=(
            "Device KV blocks. Default: ceil(length/block_size) + "
            f"{SLACK_BLOCKS} = room for one prompt plus staging slack. Raise it to "
            "make eviction partial, lower it to make the pool tighter."
        ),
    )
    parser.add_argument(
        "--prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable vLLM prefix caching (default: ENABLED). Required: with it off "
            "there are no reusable blocks, so nothing is ever offloaded."
        ),
    )
    parser.add_argument(
        "--kv-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Attach the Spyre KV-offload connector (default: ENABLED). With it off, "
            "evicted blocks are dropped and recomputed -- useful as a baseline to "
            "compare reload TTFT against recompute TTFT."
        ),
    )
    parser.add_argument(
        "--cpu-bytes",
        type=int,
        default=2_000_000_000,
        help=(
            "Host RAM budget for offloaded KV blocks (default: 2e9 = 2GB). The host "
            "pool is INDEPENDENT of the device pool and should be much larger -- it "
            "exists to hold what no longer fits on device. Too small a value yields "
            "num_blocks=0 and silently offloads nothing."
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

    # A prompt shorter than one block produces no complete block, and only complete
    # blocks are offloaded -- the run would report stored=0 for a reason that has
    # nothing to do with the pool size.
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
        f"kv_offload={'on' if args.kv_offload else 'off'} "
        f"VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="
        f"{os.environ['VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS']}"
    )
    if args.kv_offload and not args.prefix_caching:
        log(
            "[config] WARNING: kv_offload is on but prefix_caching is OFF -- there "
            "are no reusable blocks to offload, so stored will stay 0."
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
                # vllm.general_plugins entry point: the worker builds the connector
                # in initialize_from_config, which can run before
                # load_general_plugins() has registered the spec, and the registry
                # lookup then fails with "Unsupported spec type". spec_module_path
                # bypasses the registry (OffloadingSpecFactory.get_spec_cls).
                "spec_module_path": "spyre_inference.v1.kv_offload.spec",
                "cpu_bytes_to_use": args.cpu_bytes,
            },
        )
        log(
            f"[config] connector=OffloadingConnector spec=SpyreOffloadingSpec "
            f"cpu_bytes_to_use={args.cpu_bytes}"
        )

    log("[build] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    prompt_a = make_prompt_exactly(tokenizer, n, FILLER_A)
    prompt_b = make_prompt_exactly(tokenizer, n, FILLER_B)
    assert prompt_a != prompt_b
    for label, text in (("A", prompt_a), ("B", prompt_b)):
        got = len(tokenizer.encode(text, add_special_tokens=False))
        assert got == n, f"prompt {label} tokenized to {got}, want {n}"
    log(
        f"[build] prompt A filler={FILLER_A!r} prompt B filler={FILLER_B!r} "
        f"(both exactly {n} tokens, no shared prefix)"
    )

    # Start capturing before the engine exists: fd redirection must be in place
    # before the worker subprocess is forked, or its stdout/stderr inherits the
    # real terminal and the transfer lines never land in our file.
    counter = TransferCounter() if args.kv_offload else None

    log("[engine] constructing LLM (loads weights + warms up)...")
    t_engine = time.perf_counter()
    model = LLM(
        MODEL,
        max_model_len=max_model_len,
        max_num_seqs=1,  # sequential: each run competes for the same blocks
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

    # Warmup on A absorbs the one-time Spyre/Inductor compile for this prompt
    # shape, so run1 is not charged for it. It also leaves A resident, which is
    # what makes run1 (on B) evict something.
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

    # Discard whatever the engine logged while starting, so each run below counts
    # only its own transfers.
    if counter is not None:
        w_stored, w_loaded, w_sj, w_lj = counter.delta()
        log(
            f"[offload] during startup+warmup: stored_blocks={w_stored} "
            f"loaded_blocks={w_loaded} (jobs: {w_sj} store, {w_lj} load)"
        )

    rows: list[dict] = []
    for i in range(args.runs):
        # run1 -> B, run2 -> A, run3 -> B, ... so every run needs the prompt the
        # previous run evicted.
        label = "B" if i % 2 == 0 else "A"
        prompt = prompt_b if label == "B" else prompt_a
        tag = f"run{i + 1}-{label}"

        log(f"[run] {tag}: sending prompt {label}...")
        start = time.perf_counter()
        try:
            out = model.generate(prompts=prompt, sampling_params=sampling_params)
        except Exception as e:  # noqa: BLE001
            log(f"[fatal] {tag} FAILED after {time.perf_counter() - start:.1f}s: {e!r}")
            rows.append({"tag": tag, "label": label, "error": repr(e)})
            break
        ttft = time.perf_counter() - start

        d_stored = d_loaded = d_sj = d_lj = None
        if counter is not None:
            d_stored, d_loaded, d_sj, d_lj = counter.delta()

        row = {
            "tag": tag,
            "label": label,
            "ttft": ttft,
            "d_stored": d_stored,
            "d_loaded": d_loaded,
            "d_store_jobs": d_sj,
            "d_load_jobs": d_lj,
        }
        rows.append(row)

        log(
            f"[result] {tag}: prompt_tokens={len(out[0].prompt_token_ids)} "
            f"ttft={ttft:.3f}s output={out[0].outputs[0].text!r}"
        )
        if counter is not None:
            log(
                f"[offload] {tag}: stored_blocks={d_stored} loaded_blocks={d_loaded} "
                f"(jobs: {d_sj} store, {d_lj} load; cumulative blocks "
                f"stored={counter.total_stored} loaded={counter.total_loaded})"
            )

    # Restore the real stdout/stderr BEFORE printing the summary: while capture is
    # active every log() lands in the temp file, so a summary written here would be
    # swallowed and lost when the process exits.
    total_stored = total_loaded = None
    if counter is not None:
        total_stored, total_loaded = counter.total_stored, counter.total_loaded
        counter.close()

    log("")
    log(
        f"[summary] length={n} num_gpu_blocks={num_gpu_blocks} "
        f"blocks_per_prompt={blocks_per_prompt} "
        f"prefix_caching={'on' if args.prefix_caching else 'off'} "
        f"kv_offload={'on' if args.kv_offload else 'off'}"
    )
    # Columns are BLOCKS moved, with job counts alongside: one job batches many
    # blocks, so a job count on its own looks like almost nothing happened.
    log(
        f"[summary] {'run':>9}  {'prompt':>6}  {'ttft(s)':>9}  "
        f"{'stored_blk':>10}  {'loaded_blk':>10}  {'jobs(s/l)':>10}"
    )
    for row in rows:
        if "error" in row:
            log(f"[summary] {row['tag']:>9}  {row['label']:>6}  {'FAILED':>9}  {row['error']}")
            continue
        ds = "n/a" if row["d_stored"] is None else str(row["d_stored"])
        dl = "n/a" if row["d_loaded"] is None else str(row["d_loaded"])
        jobs = (
            "n/a"
            if row["d_store_jobs"] is None
            else f"{row['d_store_jobs']}/{row['d_load_jobs']}"
        )
        log(
            f"[summary] {row['tag']:>9}  {row['label']:>6}  {row['ttft']:>9.3f}  "
            f"{ds:>10}  {dl:>10}  {jobs:>10}"
        )

    ok = [r for r in rows if "error" not in r]
    # run1 is the cold miss. run2 is the FIRST partial-hit shape, so it pays a
    # one-time kernel compile (spyre_attn specializes per (num_blocks,
    # padded_query_len)) and is not representative of the reload path -- at 16K it
    # was 28.3s against a 4.4s steady state. Steady state is run3 onward; run2 is
    # reported separately rather than averaged in.
    if len(ok) > 2:
        cold = ok[0]["ttft"]
        first_hit = ok[1]["ttft"]
        steady = sorted(r["ttft"] for r in ok[2:])
        mid = len(steady) // 2
        median = steady[mid] if len(steady) % 2 else (steady[mid - 1] + steady[mid]) / 2
        log("")
        log(
            f"[summary] run1(cold)={cold:.3f}s  "
            f"run2(first-hit, includes compile)={first_hit:.3f}s"
        )
        if median > 0:
            log(
                f"[summary] steady-state reload (run3+): median={median:.3f}s  "
                f"min={steady[0]:.3f}s  max={steady[-1]:.3f}s  "
                f"cold/reload={cold / median:.2f}x  (n={len(steady)})"
            )
        else:
            log("[summary] steady-state reload median is 0")

    if total_stored is not None:
        log(
            f"[summary] cumulative BLOCKS stored={total_stored} loaded={total_loaded} "
            f"(jobs: {counter.total_store_jobs} store, {counter.total_load_jobs} load)"
        )
        if total_stored == 0:
            log(
                "[verdict] NO eviction happened. The pool may be large enough to hold "
                "both prompts, or the prompt is shorter than one block (only complete "
                "blocks are stored)."
            )
        elif total_loaded == 0:
            log(
                "[verdict] blocks were offloaded but never read back -- they were "
                "recomputed instead. Check that prefix caching is on and that the "
                "alternation really displaces the other prompt."
            )
        else:
            log("[verdict] offload AND reload both exercised.")

    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
