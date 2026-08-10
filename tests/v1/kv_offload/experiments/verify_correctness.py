#!/usr/bin/env python3
"""Does the KV-offload reload path return CORRECT data, or just fast data?

The eviction test measures TTFT and transfer counters, but a reload that returns
garbage KV would still look fast and still report loaded_blocks>0. This checks the
only thing that actually matters: does the model generate the SAME text whether or
not blocks made a host round-trip?

Method: generate a multi-token continuation of the same prompt in three conditions
and compare token IDs exactly.

  1. offload OFF, roomy pool  -> ground truth (no eviction, no reload)
  2. offload ON, tight pool   -> alternating A/B forces store+reload; the run we
                                 care about is a post-reload generation of A
  3. token-level comparison   -> IDs must match exactly, not just "looks similar"

Uses --max-tokens 24 rather than 1: a single greedy token is a weak test (many
prompts continue with '.'), whereas 24 tokens diverge quickly if the KV is wrong.

Usage:
  uv run --no-sync python verify_correctness.py [--length 1024] [--max-tokens 24]
"""
import argparse
import os
import sys

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
BLOCK_SIZE = 128


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
    raise AssertionError("could not converge to exact length")


def run_case(name, *, prompts_in_order, n, gpu_blocks, offload, max_tokens, cpu_bytes):
    """Build an engine, send prompts in order, return the LAST result per label."""
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
    log(f"\n[{name}] gpu_blocks={gpu_blocks} offload={offload} max_model_len={max_model_len}")
    llm = LLM(
        MODEL,
        max_model_len=max_model_len,
        max_num_seqs=1,
        num_gpu_blocks_override=gpu_blocks,
        enable_prefix_caching=True,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
        kv_transfer_config=kv_cfg,
    )
    # Greedy and deterministic: any difference in output is a difference in KV,
    # not sampling noise.
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, min_tokens=max_tokens)

    last = {}
    for label, text in prompts_in_order:
        out = llm.generate(prompts=text, sampling_params=sp)
        o = out[0].outputs[0]
        last[label] = (list(o.token_ids), o.text)
        log(f"[{name}] {label}: {len(o.token_ids)} tok  {o.text[:60]!r}")
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--cpu-bytes", type=int, default=2_000_000_000)
    args = ap.parse_args()
    n = args.length

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
    pa = make_prompt_exactly(tok, n, "alpha")
    pb = make_prompt_exactly(tok, n, "bravo")

    blocks_per_prompt = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    roomy = blocks_per_prompt * 2 + 4   # both prompts fit; nothing is evicted
    tight = blocks_per_prompt + 2       # same sizing ttft_evict.py uses

    log("=" * 72)
    log(f"length={n} blocks_per_prompt={blocks_per_prompt} "
        f"roomy_pool={roomy} tight_pool={tight} max_tokens={args.max_tokens}")
    log("=" * 72)

    # GROUND TRUTH: roomy pool, offload off. Nothing is evicted, so nothing is
    # reloaded -- whatever the model says here is the correct continuation.
    truth = run_case("truth", prompts_in_order=[("A", pa), ("B", pb)],
                     n=n, gpu_blocks=roomy, offload=False,
                     max_tokens=args.max_tokens, cpu_bytes=args.cpu_bytes)

    # RELOAD: tight pool, offload on, alternating so each prompt is stored and
    # then read back. The final A and B generations are post-reload.
    order = [("A", pa), ("B", pb), ("A", pa), ("B", pb), ("A", pa), ("B", pb)]
    reload_ = run_case("reload", prompts_in_order=order,
                       n=n, gpu_blocks=tight, offload=True,
                       max_tokens=args.max_tokens, cpu_bytes=args.cpu_bytes)

    log("\n" + "=" * 72)
    log("COMPARISON  (greedy, temperature=0 -> token IDs must match EXACTLY)")
    log("=" * 72)
    ok = True
    for label in ("A", "B"):
        t_ids, t_txt = truth[label]
        r_ids, r_txt = reload_[label]
        same = t_ids == r_ids
        ok &= same
        log(f"\nprompt {label}: {'MATCH' if same else 'MISMATCH'}")
        log(f"  truth : {t_txt[:70]!r}")
        log(f"  reload: {r_txt[:70]!r}")
        if not same:
            # Report the first divergence: an off-by-one-block KV error usually
            # shows up early and then compounds.
            for i, (a, b) in enumerate(zip(t_ids, r_ids)):
                if a != b:
                    log(f"  first divergence at token {i}: truth={a} reload={b}")
                    break
            log(f"  truth ids : {t_ids}")
            log(f"  reload ids: {r_ids}")

    log("\n" + "=" * 72)
    if ok:
        log("VERDICT: reload path returns CORRECT data "
            "(identical greedy continuations for both prompts)")
    else:
        log("VERDICT: MISMATCH -- the reload path is returning different KV than "
            "a non-offloaded run. This is a correctness bug, not a perf finding.")
    log("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
