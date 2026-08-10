#!/usr/bin/env python3
"""Recompute / host-reload / HBM-reuse in ONE engine, using request traffic.

WHY THIS EXISTS. An earlier attempt measured the three paths with three different
device pools, because the paths have contradictory pool requirements (HBM reuse
needs both prompts resident; eviction needs the pool too small for that). Changing
the pool between cases means the comparison is not truly like-for-like.

This version changes NOTHING between cases. One engine, one pool, one config. The
three paths are produced by the ORDER OF REQUESTS, which is also how a real serving
deployment produces them:

    step 1   P1 (fresh)   -> RECOMPUTE      never seen; nothing to reuse
    step 2   JUNK (large) -> (lever)        fills the pool, pushing P1 to host
    step 3   P1 again     -> HOST RELOAD    evicted, so blocks come from DRAM
    step 4   P1 again     -> HBM REUSE      just reloaded, still resident
    step 5   JUNK again   -> (lever)        evict P1 once more
    step 6   P1 again     -> HOST RELOAD    repeat, for a median
    step 7   P1 again     -> HBM REUSE      repeat, for a median
    ...
    last     P2 (fresh)   -> RECOMPUTE      confirms step 1 was not a one-off

Each measured path therefore differs only in what the engine had cached at that
moment -- not in block size, pool size, max_model_len, code, or process.

POOL AND JUNK SIZING -- this is what determines whether a "reload" is a FULL reload.

Two constraints pull in opposite directions:
  * step 4 (HBM reuse) needs P1 to fit comfortably    -> pool must exceed blk
  * step 3 (host reload) needs ALL of P1 displaced    -> junk must sweep the pool

The junk can only displace `pool - blk` blocks of P1 if it is itself only `blk`
blocks: the rest of P1 stays resident and is served from HBM, so the step is a
PARTIAL reload. Measured with junk == prompt and pool = 1.5*blk+1:

    L=1024  blk=8   pool=13  slack=5   -> loaded 4 of 8   (50%)
    L=4096  blk=32  pool=49  slack=17  -> loaded 16 of 32 (50%)

i.e. unloaded == slack == pool - blk, exactly. The fix is NOT a different block
size or a tighter pool -- it is a BIGGER JUNK:

    junk_blocks = pool     ->  the junk alone fills the whole pool
                          ->  every one of P1's blk blocks must go to host
                          ->  step 3 reloads ~blk blocks: a FULL reload

Pool stays at 1.5*blk+1 so step 4 remains a genuine device-resident hit. Making the
pool tighter instead (blk+1, the minimum that boots) would risk degrading step 4 too.

FRESH JUNK PER CYCLE. Reusing one junk prompt lets it become cache-resident: its
lever time collapsed from 24.910 s to ~1.1 s after cycle 1 and it began reporting
loaded=17, meaning it had turned into a third cached prompt competing for the pool
rather than a clean evictor. Each cycle now uses a junk prompt with a distinct
filler token, so no junk can ever hit another junk's cache.

max_model_len must cover the LONGEST request -- the junk -- not just P1.

BLOCK ACCOUNTING. Every step reports blocks stored/loaded, scraped from the
offloading worker's own log lines (it runs in a subprocess, so its counters are
otherwise unreachable). Counts BLOCKS, not log lines: one transfer job batches many
blocks. This is what proves a step took the path it claims:

    reload step : loaded ~= blocks_per_prompt   (the prefix came from DRAM)
    reuse  step : loaded == 0                   (nothing moved; it was already there)

If a "reload" step shows loaded==0, it was actually a device hit and its timing must
not be reported as reload.

Usage:
  uv run --no-sync python traffic_three_way.py --length 4096 --cycles 3
"""
import argparse
import os
import re
import statistics
import sys
import tempfile
import time

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
BLOCK_SIZE = 128
BYTES_PER_BLOCK = 2.10e6  # measured from the worker log: num_blocks=8 -> 16777216 B

_XFER = re.compile(
    r"SpyreOffloadingWorker (device->host|host->device):\s+job=\d+\s+num_blocks=(\d+)"
)


def log(m):
    print(m, flush=True)


def make_prompt_exactly(tok, n, filler):
    """Prompt that tokenizes to EXACTLY n tokens, led by a distinct filler word.

    Distinct fillers mean no two prompts share a leading token, so none can serve
    as a prefix-cache hit for another -- each 'fresh' request is genuinely cold.
    """
    ids = tok.encode((filler + " ") * (n + 64), add_special_tokens=False)[:n]
    text = tok.decode(ids)
    for _ in range(16):
        re_ids = tok.encode(text, add_special_tokens=False)
        if len(re_ids) == n:
            return text
        re_ids = re_ids[:n] if len(re_ids) > n else re_ids + ids[len(re_ids):n]
        text = tok.decode(re_ids)
    raise AssertionError("could not converge to exact token length")


class Capture:
    """Count transferred BLOCKS by capturing worker stdout at the fd level.

    Restores the descriptors in a finally block (unlike the upstream harness's
    TransferCounter), so a crash during engine construction still shows its
    traceback instead of disappearing into the temp file.
    """

    def __init__(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".traffic.log", delete=False)
        self.saved_out, self.saved_err = os.dup(1), os.dup(2)
        os.dup2(self.tmp.fileno(), 1)
        os.dup2(self.tmp.fileno(), 2)
        self.offset = 0

    def delta(self):
        sys.stdout.flush(); sys.stderr.flush()
        with open(self.tmp.name, errors="replace") as fh:
            fh.seek(self.offset)
            text = fh.read()
            self.offset = fh.tell()
        stored = loaded = 0
        for direction, blocks in _XFER.findall(text):
            if direction == "device->host":
                stored += int(blocks)
            else:
                loaded += int(blocks)
        os.write(self.saved_out, text.encode(errors="replace"))
        return stored, loaded

    def close(self):
        try:
            self.delta()
        finally:
            os.dup2(self.saved_out, 1)
            os.dup2(self.saved_err, 2)
            os.close(self.saved_out)
            os.close(self.saved_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=4096, help="length of P1/P2")
    ap.add_argument("--junk-length", type=int, default=None,
                    help="junk request length. Default: sized so the junk fills the "
                         "ENTIRE device pool, which is what makes step 3 a full "
                         "reload rather than a partial one (see module docstring). "
                         "Override only to study partial eviction deliberately.")
    ap.add_argument("--cycles", type=int, default=3,
                    help="junk->reload->reuse cycles (each yields one reload and "
                         "one reuse sample)")
    ap.add_argument("--pool-mult", type=float, default=1.5,
                    help="device pool as a multiple of one prompt (default 1.5; "
                         "2.0 fails to evict, see module docstring)")
    ap.add_argument("--host-gb", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=1)
    args = ap.parse_args()

    n = args.length
    blk = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    pool = int(blk * args.pool_mult) + 1

    # Default junk fills the pool MINUS ONE block, so P1 is displaced entirely while
    # the engine still has the block it needs for the generated token.
    #
    # pool * BLOCK_SIZE is WRONG and will not boot: max_model_len rounds up to
    # junk_len + 1 token, needing pool+1 blocks against a pool of pool. Verified:
    #   junk=1664 (13 blk), pool=13 -> ValueError, "estimated maximum model length
    #   is 1664" while max_model_len was 1792.
    # pool-1 still exceeds blk for pool_mult >= 1.5, so eviction stays total.
    jn = args.junk_length if args.junk_length else (pool - 1) * BLOCK_SIZE
    jblk = (jn + BLOCK_SIZE - 1) // BLOCK_SIZE

    # The junk displaces min(blk, jblk) blocks of P1. A full reload therefore needs
    # jblk >= blk, not jblk >= pool.
    if jblk < blk:
        log(f"[warn] junk is only {jblk} blocks against a {blk}-block prompt, so at "
            f"most {jblk}/{blk} of P1 ({100 * jblk / blk:.0f}%) can be displaced. "
            "Expect a PARTIAL reload; reload/reuse will be an underestimate.")
    # max_model_len follows the longest request, so the pool must cover it plus the
    # generated token or the engine refuses to start.
    need = (max(n, jn) + args.max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    if need > pool:
        log(f"[fatal] max_model_len needs {need} blocks but the pool is {pool}; the "
            f"engine will refuse to boot. Reduce --junk-length (default leaves one "
            f"spare block) or raise --pool-mult.")
        return 2
    if blk + jblk <= pool:
        log(f"[fatal] prompt ({blk}) + junk ({jblk}) = {blk + jblk} blocks FITS in "
            f"pool {pool}; the junk would not evict P1 at all. Lower --pool-mult or "
            f"raise --junk-length.")
        return 2

    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7200")
    for k, v in (("RANK", "0"), ("WORLD_SIZE", "1"),
                 ("LOCAL_RANK", "0"), ("LOCAL_WORLD_SIZE", "1")):
        os.environ.setdefault(k, v)

    import torch
    import torch_spyre  # noqa: F401
    if torch.spyre.device_count() < 1:
        log("[fatal] no Spyre device")
        return 2

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig, KVTransferConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    tok = AutoTokenizer.from_pretrained(MODEL)
    p1 = make_prompt_exactly(tok, n, "alpha")
    p2 = make_prompt_exactly(tok, n, "bravo")
    # One junk per cycle, each with a distinct filler, so no junk can serve as a
    # prefix-cache hit for another. A single reused junk becomes cache-resident and
    # stops evicting (observed: lever time 24.9 s -> 1.1 s after cycle 1).
    junk_fillers = ["zulu", "yankee", "xray", "whisky", "victor", "tango",
                    "sierra", "romeo", "quebec", "papa", "oscar", "november",
                    "mike", "lima", "kilo", "juliet", "india", "hotel"]
    if args.cycles > len(junk_fillers):
        log(f"[fatal] --cycles {args.cycles} exceeds {len(junk_fillers)} distinct "
            "junk fillers; add more to junk_fillers.")
        return 2
    junks = [make_prompt_exactly(tok, jn, junk_fillers[c]) for c in range(args.cycles)]

    # max_model_len must cover the LONGEST request, i.e. the junk.
    max_model_len = ((max(n, jn) + args.max_tokens + BLOCK_SIZE - 1)
                     // BLOCK_SIZE) * BLOCK_SIZE

    log("=" * 78)
    log(f"TRAFFIC-DRIVEN THREE-WAY   length={n}  junk={jn}  cycles={args.cycles}")
    log(f"  blocks/prompt={blk}  junk_blocks={jblk}  device_pool={pool} "
        f"({args.pool_mult}x prompt)")
    log(f"  prompt+junk = {blk + jblk} blocks > pool {pool}  -> junk MUST evict P1")
    # The junk displaces min(blk, jblk) of P1's blocks; the remainder stays resident
    # and is served from HBM, so this is the reload fraction we should measure.
    predicted = min(blk, jblk)
    log(f"  junk sweeps {jblk}/{pool} of the pool -> expect ~{predicted}/{blk} blocks "
        f"reloaded ({100 * predicted / blk:.0f}% of the prefix)"
        + ("  [FULL reload]" if predicted >= blk else "  [PARTIAL reload]"))
    log(f"  {args.cycles} distinct junk prompts (one per cycle, none can cache-hit "
        "another)")
    log(f"  host_pool={args.host_gb} GB   max_model_len={max_model_len}")
    log("  ONE engine, ONE pool, ONE config -- the paths come from request order")
    log("=" * 78)

    cap = Capture()
    try:
        llm = LLM(
            MODEL,
            max_model_len=max_model_len,
            max_num_seqs=1,
            num_gpu_blocks_override=pool,
            enable_prefix_caching=True,
            attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
            kv_transfer_config=KVTransferConfig(
                kv_connector="OffloadingConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "spec_name": "SpyreOffloadingSpec",
                    "spec_module_path": "spyre_inference.v1.kv_offload.spec",
                    "cpu_bytes_to_use": args.host_gb * 1_000_000_000,
                },
            ),
        )
        sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            min_tokens=args.max_tokens)

        # Absorb the one-time per-shape compile on a throwaway prompt, so step 1's
        # "recompute" is a prefill measurement and not a compile measurement.
        # Filler must not collide with any junk or prompt filler, or a later request
        # would hit this warmup's cached blocks instead of taking its intended path.
        warm = make_prompt_exactly(tok, n, "delta")
        t = time.perf_counter()
        llm.generate(prompts=warm, sampling_params=sp)
        log(f"\n[warmup] discarded, {time.perf_counter() - t:.3f}s "
            "(absorbs per-shape compile)")
        cap.delta()

        events = []

        def send(tag, path, prompt):
            t0 = time.perf_counter()
            out = llm.generate(prompts=prompt, sampling_params=sp)
            dt = time.perf_counter() - t0
            stored, loaded = cap.delta()
            events.append({"tag": tag, "path": path, "ttft": dt,
                           "stored": stored, "loaded": loaded,
                           "tokens": list(out[0].outputs[0].token_ids)})
            log(f"  {tag:<22} {path:<12} {dt:8.3f}s  stored={stored:<5} loaded={loaded}")
            return dt

        log("\n  step                   path             ttft  transfers")
        log("  " + "-" * 64)

        # RECOMPUTE: P1 has never been seen; nothing to reuse or reload.
        send("P1 (fresh)", "recompute", p1)

        for c in range(args.cycles):
            # LEVER: junk fills the pool. Not a measurement -- it is what forces
            # P1's blocks out to host.
            send(f"junk #{c + 1}", "lever", junks[c])
            # HOST RELOAD: P1 was evicted, so its prefix must come back from DRAM.
            send(f"P1 after junk #{c + 1}", "reload", p1)
            # HBM REUSE: P1 was just reloaded and is still resident -- no transfer.
            send(f"P1 repeat #{c + 1}", "reuse", p1)

        # RECOMPUTE again on a genuinely new prompt: confirms the first recompute
        # was representative and not an artifact of engine start-up.
        send("P2 (fresh)", "recompute", p2)

    finally:
        cap.close()

    # ---- aggregate --------------------------------------------------------------
    def agg(path):
        xs = [e["ttft"] for e in events if e["path"] == path]
        ld = [e["loaded"] for e in events if e["path"] == path]
        return xs, ld

    log("\n" + "=" * 78)
    log(f"SUMMARY   length={n}   ONE engine, pool={pool} blocks, host={args.host_gb} GB")
    log("=" * 78)
    log(f"{'path':<14} {'n':>3} {'median_s':>10} {'min_s':>9} {'max_s':>9} "
        f"{'loaded_blk/req':>15}")
    log("-" * 78)
    stats = {}
    for path in ("recompute", "reload", "reuse"):
        xs, ld = agg(path)
        if not xs:
            continue
        stats[path] = statistics.median(xs)
        log(f"{path:<14} {len(xs):>3} {statistics.median(xs):>10.3f} {min(xs):>9.3f} "
            f"{max(xs):>9.3f} {statistics.mean(ld):>15.1f}")

    log("")
    log("  BLOCK ACCOUNTING -- does each path match its claimed mechanism?")
    log(f"    a full prompt is {blk} blocks ({blk * BYTES_PER_BLOCK / 1e6:.0f} MB)")
    for path, expect in (("recompute", "0 (nothing cached yet)"),
                         ("reload", f"~{blk} (prefix from DRAM)"),
                         ("reuse", "0 (already resident)")):
        xs, ld = agg(path)
        if not xs:
            continue
        mean_ld = statistics.mean(ld)
        verdict = "OK"
        if path == "reload" and mean_ld < blk * 0.5:
            verdict = ("*** only %.0f%% of the prefix came from DRAM -- this was "
                       "largely a device hit, NOT a reload" % (100 * mean_ld / blk))
        if path == "reuse" and mean_ld > blk * 0.1:
            verdict = ("*** %.0f blocks moved on a supposed reuse -- the block was "
                       "not actually resident" % mean_ld)
        log(f"    {path:<10} expected {expect:<26} measured {mean_ld:>6.1f}   {verdict}")

    if {"recompute", "reload", "reuse"} <= stats.keys():
        log("")
        log(f"  recompute / reload = {stats['recompute'] / stats['reload']:.2f}x   "
            "(what offload buys under device pressure)")
        log(f"  reload / reuse     = {stats['reload'] / stats['reuse']:.2f}x   "
            "(cost of the DRAM round-trip vs an HBM hit)")
        log("")
        if stats["reuse"] <= stats["reload"] <= stats["recompute"]:
            log("  ORDERING AS EXPECTED: reuse <= reload <= recompute")
        elif stats["reload"] < stats["reuse"]:
            log("  *** ANOMALY CONFIRMED IN ONE ENGINE: DRAM reload beats HBM reuse.")
            log("      Same pool, same config, same process -- so this is not a")
            log("      configuration artifact. A round-trip is strictly more work than")
            log("      reusing resident blocks, so the reuse path must be doing")
            log("      avoidable work (e.g. re-running attention over cached blocks")
            log("      that the reload path skips).")
        else:
            log(f"  UNEXPECTED: recompute={stats['recompute']:.3f} "
                f"reload={stats['reload']:.3f} reuse={stats['reuse']:.3f}")

    # Correctness: every P1 response must be token-identical regardless of path.
    p1_toks = [e["tokens"] for e in events
               if e["tag"].startswith("P1") and e["path"] != "lever"]
    log("")
    if p1_toks and all(t == p1_toks[0] for t in p1_toks):
        log(f"  correctness: all {len(p1_toks)} P1 responses token-identical "
            "across recompute/reload/reuse")
    else:
        log("  correctness: *** P1 responses DIFFER between paths -- investigate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
