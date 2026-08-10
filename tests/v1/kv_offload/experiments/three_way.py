#!/usr/bin/env python3
"""Three-way TTFT comparison: recompute vs host reload vs HBM reuse.

THE QUESTION. A tight-pool reload from host DRAM measured 4.270 s at L=16384, while
a phase-1 prefix-cache hit at the same length measured 4.579 s. A host round-trip
being FASTER than reusing blocks already in HBM should be impossible -- it is
strictly more work. But those two numbers came from different block sizes, pool
sizes, max_model_len, code bases and access patterns, so the comparison was
meaningless. This script measures all three paths under one roof.

WHY NOT ONE ENGINE. The three paths are selected by device-pool size, and their
requirements are mutually exclusive:

    HBM reuse  needs BOTH prompts resident  -> pool >= 2*blk + 1
    recompute
    and reload needs the other prompt GONE  -> pool <= blk + 1

No single pool satisfies both. So this builds three engines IN ONE PROCESS, holding
everything else identical: same block_size, same code, same model, same prompt
objects, same thread pinning, same interpreter. Only --gpu-blocks and the presence
of the offload connector change. That is as close to a controlled comparison as the
architecture permits.

THE THREE CASES

  case 1  RECOMPUTE     pool = blk+1, offload OFF
          The other prompt is evicted and dropped. Every alternating run pays a
          full prefill. This is the baseline the other two must beat.

  case 2  HOST RELOAD   pool = blk+1, offload ON, host pool 16 GB
          The other prompt is evicted to DRAM and fetched back. Same device
          pressure as case 1 -- the ONLY difference is where the blocks go.

  case 3  HBM REUSE     pool = 2*blk+4, offload OFF
          Both prompts stay resident, so prefix caching serves each repeat from
          device memory. No eviction, no transfer. Should be the fastest.

EXPECTED ORDERING:  case 3  <  case 2  <  case 1
If case 2 < case 3, a host round-trip beats a device hit, which means the device
"hit" path is doing avoidable work -- a finding about the attention/cache
implementation, not about offload.

All cases alternate A/B so the access pattern is identical; in case 3 the
alternation simply hits cache instead of evicting.

Usage:
  uv run --no-sync python three_way.py --length 16384 --runs 7
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
FILLER_A, FILLER_B = "alpha", "bravo"

# Emitted per transfer job by SpyreOffloadingWorker; must match worker.py's
# logger.info call.
_XFER = re.compile(
    r"SpyreOffloadingWorker (device->host|host->device):\s+job=\d+\s+num_blocks=(\d+)"
)


def log(m):
    print(m, flush=True)


def make_prompt_exactly(tok, n, filler):
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
    """Capture worker stdout/stderr at the fd level to count transferred blocks.

    The offloading worker runs in a subprocess, so its counters are unreachable and
    its log records never enter our logging tree; only its forwarded stdout/stderr
    does. Counts BLOCKS (not log lines) because one job batches many blocks.

    Unlike the harness's own TransferCounter, this restores the descriptors in a
    finally block, so a crash during engine construction still surfaces its
    traceback instead of vanishing into the temp file.
    """

    def __init__(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".3way.log", delete=False)
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


def run_case(case, *, n, pool, offload, cpu_bytes, runs, pa, pb, max_tokens):
    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    kv_cfg = None
    if offload:
        from vllm.config import KVTransferConfig
        kv_cfg = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SpyreOffloadingSpec",
                "spec_module_path": "spyre_inference.v1.kv_offload.spec",
                "cpu_bytes_to_use": cpu_bytes,
            },
        )

    max_model_len = ((n + max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    log(f"\n{'=' * 70}\n[{case['id']}] {case['name']}\n{'=' * 70}")
    log(f"  device_pool={pool} blocks  offload={'ON' if offload else 'OFF'}  "
        f"max_model_len={max_model_len}")
    log(f"  {case['why']}")

    cap = Capture() if offload else None
    try:
        llm = LLM(
            MODEL,
            max_model_len=max_model_len,
            max_num_seqs=1,
            num_gpu_blocks_override=pool,
            enable_prefix_caching=True,
            attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
            kv_transfer_config=kv_cfg,
        )
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens)

        # Warmup on A: absorbs the one-time per-shape compile and leaves A resident.
        t = time.perf_counter()
        llm.generate(prompts=pa, sampling_params=sp)
        log(f"  [warmup A] {time.perf_counter() - t:.3f}s (discarded)")
        if cap:
            s, l = cap.delta()
            log(f"  [warmup A] stored={s} loaded={l}")

        rows, texts = [], {}
        for i in range(runs):
            label = "B" if i % 2 == 0 else "A"
            prompt = pb if label == "B" else pa
            t = time.perf_counter()
            out = llm.generate(prompts=prompt, sampling_params=sp)
            dt = time.perf_counter() - t
            s = l = None
            if cap:
                s, l = cap.delta()
            rows.append({"run": i + 1, "label": label, "ttft": dt, "stored": s, "loaded": l})
            texts[label] = list(out[0].outputs[0].token_ids)
            log(f"  run{i + 1}-{label}: {dt:8.3f}s"
                + (f"  stored={s:<5} loaded={l}" if cap else ""))
    finally:
        if cap:
            cap.close()

    # run1 is a cold miss; run2 pays a one-time partial-hit kernel compile (the
    # attention kernel specializes per (num_blocks, padded_query_len)). Steady state
    # is run3+.
    ok = [r for r in rows if "ttft" in r]
    steady = [r["ttft"] for r in ok[2:]]
    res = {
        "id": case["id"], "name": case["name"],
        "cold": ok[0]["ttft"] if ok else None,
        "first_hit": ok[1]["ttft"] if len(ok) > 1 else None,
        "steady_median": statistics.median(steady) if steady else None,
        "steady_min": min(steady) if steady else None,
        "steady_max": max(steady) if steady else None,
        "n_steady": len(steady),
        "stored": sum(r["stored"] or 0 for r in ok) if cap else None,
        "loaded": sum(r["loaded"] or 0 for r in ok) if cap else None,
        "tokens": texts,
    }
    log(f"  -> cold={res['cold']:.3f}s  first_hit={res['first_hit']:.3f}s  "
        f"steady_median={res['steady_median']:.3f}s (n={res['n_steady']})")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=16384)
    ap.add_argument("--runs", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--host-gb", type=int, default=16)
    args = ap.parse_args()

    n = args.length
    cpu_bytes = args.host_gb * 1_000_000_000
    blk = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    tight = blk + 1        # tightest legal: prompt + generated token
    roomy = 2 * blk + 4    # both prompts resident, with slack

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
    tok = AutoTokenizer.from_pretrained(MODEL)
    pa = make_prompt_exactly(tok, n, FILLER_A)
    pb = make_prompt_exactly(tok, n, FILLER_B)

    log("=" * 70)
    log(f"THREE-WAY at length={n}  blk/prompt={blk}  runs={args.runs}")
    log(f"  tight pool={tight} (evicts)   roomy pool={roomy} (both resident)")
    log(f"  host pool={args.host_gb} GB   two prompts need "
        f"{2 * blk * 2.10e6 / 1e9:.2f} GB")
    log("=" * 70)

    cases = [
        (dict(id="case1", name="RECOMPUTE (evict, drop, recompute)",
              why="pool holds 1 prompt; offload off -> evicted blocks are dropped"),
         dict(pool=tight, offload=False)),
        (dict(id="case2", name="HOST RELOAD (evict to DRAM, fetch back)",
              why="same device pressure as case1; only destination differs"),
         dict(pool=tight, offload=True)),
        (dict(id="case3", name="HBM REUSE (both resident, prefix-cache hit)",
              why="pool holds both prompts; no eviction, no transfer"),
         dict(pool=roomy, offload=False)),
    ]

    results = []
    for case, cfg in cases:
        results.append(run_case(case, n=n, cpu_bytes=cpu_bytes, runs=args.runs,
                                pa=pa, pb=pb, max_tokens=args.max_tokens, **cfg))

    log("\n" + "=" * 78)
    log(f"THREE-WAY SUMMARY  length={n}  (one process, same code/block_size/prompts)")
    log("=" * 78)
    log(f"{'case':<38} {'cold_s':>9} {'steady_s':>9} {'blocks s/l':>13}")
    log("-" * 78)
    for r in results:
        bl = "-" if r["stored"] is None else f"{r['stored']}/{r['loaded']}"
        log(f"{r['id'] + ' ' + r['name']:<38} {r['cold']:>9.3f} "
            f"{r['steady_median']:>9.3f} {bl:>13}")

    # --- BLOCK ACCOUNTING: prove each case took the path it claims -------------
    # Without this, "case 2 is a host reload" is an assumption. The counters come
    # from the offloading worker's own log lines, so they are its accounting, not
    # ours. Expected per steady-state run: ~blk blocks loaded in case 2, and no
    # connector at all in cases 1 and 3.
    log("")
    log("  BLOCK ACCOUNTING (proves the path, not just the timing)")
    log(f"    blocks per prompt at L={n}: {blk}   (block_size={BLOCK_SIZE})")
    for r in results:
        if r["stored"] is None:
            log(f"    {r['id']}: no connector attached -- 0 host transfers possible "
                "(so any speed here is device-only)")
        else:
            per_run = r["loaded"] / max(r["n_steady"], 1)
            log(f"    {r['id']}: stored={r['stored']} loaded={r['loaded']} blocks total; "
                f"~{per_run:.0f} loaded per steady run vs {blk} in a full prompt "
                f"({100 * per_run / blk:.0f}% of the prefix from DRAM)")
            if r["loaded"] == 0:
                log("      WARNING: connector attached but nothing loaded -- this case did "
                    "NOT exercise reload; treat its timing as a device hit.")
            mb = r["loaded"] * 2.10
            log(f"      ~{mb:.0f} MB moved host->device across {r['n_steady']} steady runs")

    by = {r["id"]: r["steady_median"] for r in results}
    log("")
    log(f"  recompute / host-reload = {by['case1'] / by['case2']:.2f}x  "
        "(offload's benefit under device pressure)")
    log(f"  host-reload / HBM-reuse = {by['case2'] / by['case3']:.2f}x  "
        "(cost of the host round-trip vs a device hit)")
    log("")
    if by["case3"] <= by["case2"] <= by["case1"]:
        log("  ORDERING AS EXPECTED: HBM reuse <= host reload <= recompute")
    elif by["case2"] < by["case3"]:
        log("  *** ANOMALY CONFIRMED: host reload is FASTER than HBM reuse. ***")
        log("  A host round-trip is strictly more work than reusing resident blocks,")
        log("  so the device-hit path must be doing avoidable work (e.g. re-running")
        log("  attention over cached blocks that the reload path skips). This is a")
        log("  finding about the attention/cache implementation, not about offload.")
    else:
        log(f"  UNEXPECTED ORDERING: case1={by['case1']:.3f} case2={by['case2']:.3f} "
            f"case3={by['case3']:.3f}")

    # Correctness: all three paths must produce identical greedy tokens. A fast path
    # that returns different KV is broken, not fast.
    log("")
    ref = results[0]["tokens"]
    same = all(r["tokens"].get(k) == v for r in results[1:] for k, v in ref.items())
    log(f"  correctness across all three paths: "
        f"{'IDENTICAL tokens' if same else 'MISMATCH -- investigate'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
