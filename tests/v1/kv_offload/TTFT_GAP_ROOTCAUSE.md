# Root cause: why warm-prefix-hit TTFT is 2.5x slower than KV-offload reload

Measured 2026-08-09, branch `dev/0807-kv-offload-yzhu`, vLLM v0.26.0,
`ibm-ai-platform/micro-g3.3-8b-instruct-1b`, Spyre / fp16 / eager, 1024-token prompt,
`max_tokens=1` (so wall-clock of `generate()` == TTFT).

Companion to `TTFT_FINDINGS.md`, which established the 0.72 s gap and left ~0.32 s of it
unattributed. **This document closes that residual.** All ~682 ms of the ~723 ms gap comes
from one per-token loop in the KV write-back path.

---

## 1. How the profiling was done

The gap had resisted three rounds of code reading, so the goal was a *measured* time budget
rather than another hypothesis. Four obstacles had to be cleared first.

### 1.1 `SPYRE_ATTN_PROFILING=1` does not profile anything

`TTFT_FINDINGS.md` proposed this as the next measurement. It is label-only:
`spyre_attn.py:45-62` defines a `record_function` decorator gated on the env var, applied at
`spyre_attn.py:783` (`spyre_attn::forward`), `:850` (`reshape_and_cache`), and `:878`
(`online_softmax`). It starts no profiler — there is no `torch.profiler.profile` anywhere in
the repo. Running the proposed command produces no trace. The spans are useful, but only
once something is actually recording.

### 1.2 A profiler that can see the Spyre device

Requires the kineto-spyre torch wheel, which talks to `libaiupti` (Spyre's telemetry layer):

```bash
$DTI_PROJECT_ROOT/torch-spyre-docs/scripts/build-torch-spyre.sh --spyre-profiler
```

Spyre registers as PyTorch's `PrivateUse1` backend
(`torch._C._get_privateuse1_backend_name() == 'spyre'`), and
`torch.profiler.ProfilerActivity.SPYRE` is the same enum value as `PrivateUse1`.

### 1.3 vLLM asks for CUDA on hardware that isn't CUDA

`Worker.profile()` hardcodes `activities=["CPU", "CUDA"]`
(`vllm/v1/worker/gpu_worker.py:1209`) and resolves those strings through
`TorchProfilerActivityMap` (`vllm/profiler/wrapper.py:152-156`), which contains only
CPU / CUDA / XPU — no Spyre entry. Fixed by overriding `profile()` in
`spyre_inference/v1/worker/spyre_worker.py:158-169` to remap the hardcoded `"CUDA"` request
onto the device we actually have:

```python
_profiler_wrapper.TorchProfilerActivityMap["CUDA"] = (
    torch.profiler.ProfilerActivity.PrivateUse1
)
```

### 1.4 Phase spans are off by default

vLLM's `preprocess` / `forward` / `postprocess` / `sample` / `bookkeep` spans
(`gpu_model_runner.py:4149, 4388, 4402, 4532, 4660, 4710`) are gated on
`VLLM_CUSTOM_SCOPES_FOR_PROFILING`, which defaults off. **Without it a trace shows attention
ops but no phase breakdown — which is precisely why half the gap looked unexplained in the
first attempt.**

Two ordering constraints, both of which silently produce empty phase data if violated:

- `record_function_or_nullcontext` caches the resolved function in a module global on first
  call (`vllm/v1/utils.py:747-763`), so the var must be set **before `vllm` is imported**.
- EngineCore is a separate process, so the value must be in `os.environ` to be inherited.

Handled in `ttft_evict.py` next to the existing `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` block,
above all vLLM imports.

### 1.5 Zero-overhead capture

Profiler overhead was inflating the very thing being measured (a 3-iteration warm capture
read 2.17 s against an unprofiled 1.20 s). Solved with `ProfilerConfig.max_iterations`, which
auto-stops the profiler after N iterations (`vllm/profiler/wrapper.py:~112`) — so later runs
in the same process stay unprofiled and the overhead is *measured*, not assumed.

Added to `ttft_evict.py`: `--profile DIR`, `--profile-after N` (default 4, keeps
cold-miss and compile runs out of the capture), `--profile-iters N` → `max_iterations`.

```bash
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200

# warm (prefix-cache hit, connector OFF)
uv run --no-sync python tests/v1/kv_offload/ttft_evict.py \
    --length 1024 --runs 9 --gpu-blocks 18 --no-kv-offload \
    --profile /tmp/pf_warm --profile-after 4 --profile-iters 1

# offload (connector ON)
uv run --no-sync python tests/v1/kv_offload/ttft_evict.py \
    --length 1024 --runs 9 --gpu-blocks 18 \
    --profile /tmp/pf_off2 --profile-after 4 --profile-iters 3
```

`--profile-iters 3` on the offload side is required, not cosmetic: `load_kv_async` splits the
offload path into a transfer-only engine step plus a compute step, so `max_iterations=1`
captures the transfer step and misses the forward pass entirely (the trace shows only
`preprocess 5.0ms` / `execute_context_0(0)`).

**Overhead check — the captures are clean:** profiled warm run5 = 1.198 s against unprofiled
1.198 / 1.200 / 1.198 / 1.202 s. Profiled offload = 0.427 s against a 0.474 s median.

> Spyre is a single contested accelerator — **every run must be sequential.** No `pytest -n`,
> no backgrounding one run while starting another.

### 1.6 Analysis method

Two passes over the traces:

- **Phase level** — sum the `user_annotation` spans from §1.4 to get a top-level budget.
- **Self-time (non-overlapping)** — for each event, `dur` minus the sum of its direct
  children's `dur`, computed per `(pid, tid)` by sorting on `(ts, -dur)` and maintaining a
  containment stack. Self-times are additive, so they form a real budget; gross durations
  double-count nested spans and cannot be summed.

An earlier bucketing pass grouped events by *name pattern*, which could not distinguish
"copies caused by the write-back loop" from "copies caused by 4x-wider attention." The final
pass instead assigns every event to its innermost enclosing span, making the attribution
direct rather than inferred. This distinction mattered — see §5.

---

## 2. The time difference

### Wall clock

| Path | Query length | TTFT |
| ---- | -----------: | ---: |
| Warm — local prefix cache hit, connector OFF | 128 tok | **1.199 s** (n=7, spread 1.197–1.204) |
| Offload — connector reload | 1 tok | **0.476 s** (n=7, min 0.406, max 0.496) |
| | | **gap 0.723 s** |

Both at an 18-block pool with no eviction pressure, same script, same block size
(`--kv-offload` the only variable). The query-length asymmetry is upstream vLLM behavior: the
local prefix cache caps its hit at `num_tokens - 1` and must return whole blocks
(`kv_cache_manager.py:231`, `single_type_kv_cache_manager.py:710`), losing 128 tokens to
alignment; the offload connector's cap subtracts nothing (`offloading/scheduler.py:548`),
leaving 1 token. See `TTFT_FINDINGS.md` §"Why 1 token vs 128 tokens".

### Phase breakdown

| Phase | Warm | Offload | Δ |
| ----- | ---: | ------: | ---: |
| preprocess | 1.4 ms | 6.0 ms | −4.6 |
| **forward** | **1168.2 ms** | **381.0 ms** | **+787.2** |
| postprocess | 23.4 ms | 12.5 ms | +10.9 |
| sample | 0.3 ms | 0.2 ms | ~0 |
| bookkeep | 0.1 ms | 0.1 ms | 0 |

Forward accounts for 787 of the ~771 ms of phase-level delta — **102%**. Everything outside
the forward pass is a wash; the offload path's extra transfer step and the warm path's
slightly longer postprocess cancel out. No scheduler or model-runner asymmetry to chase.

---

## 3. Which code causes it

Self-time inside the forward pass, per event. Every row that differs:

| Event | Warm (calls / ms) | Offload (calls / ms) | Δ ms | Per-call |
| ----- | ----------------: | -------------------: | ---: | -------: |
| `aten::_copy_from` | 1204 / **384.7** | 180 / 107.0 | **+277.7** | 320 µs |
| `launch_jobplan:sdsc_fused_overwrite_0` | 1024 / **215.0** | 12 / 2.6 | **+212.4** | 210 µs |
| `TorchDynamo Cache Lookup` | 1666 / **139.2** | 654 / 19.5 | **+119.7** | 84 µs |
| `spyre::overwrite` | 1024 / **73.7** | 12 / 1.0 | **+72.7** | 72 µs |
| | | | **+682.5** | |

Every other event is a wash, with **identical call counts on both sides**:

| Event | Warm | Offload |
| ----- | ---: | ------: |
| `sdsc_fused_mul_0` | 90 / 23.6 ms | 90 / 23.7 ms |
| `sdsc_fused_mul_silu_slice_0` | 4 / 20.7 ms | 4 / 21.5 ms |
| `sdsc_fused_add_0` | 77 / 20.2 ms | 77 / 20.0 ms |
| `sdsc_fused_copy_from_d2d_0` | 68 / 17.0 ms | 68 / 16.8 ms |
| `sdsc_fused_bmm_0` | 64 / 16.0 ms | 64 / 16.4 ms |
| `sdsc_fused_sub_0` | 60 / 15.6 ms | 60 / 15.5 ms |
| `sdsc_fused_exp_0` | 60 / 14.4 ms | 60 / 14.5 ms |
| `sdsc_fused_mm_transpose_0` | 25 / 10.9 ms | 25 / 11.2 ms |
| `sdsc_fused_amax_0` | 32 / 8.0 ms | 32 / 7.8 ms |

**The math is identical.** The device performs the same 585 kernel launches in both runs,
totaling 15.9 ms (warm) vs 15.7 ms (offload) — a 0.2 ms difference. All 682 ms is framework
overhead around unchanged arithmetic.

The source is `specialized_reshape_and_cache_kernel`,
**`spyre_inference/v1/attention/backends/spyre_attn.py:254-258`**:

```python
for t in range(num_tokens):
    k_tok = convert(key[t].unsqueeze(1).contiguous(), target_device)
    v_tok = convert(value[t].unsqueeze(1).contiguous(), target_device)
    _overwrite(k_tok, k_pages[block_indices[t]], [1], [block_offsets[t]])
    _overwrite(v_tok, v_pages[block_indices[t]], [1], [block_offsets[t]])
```

`num_tokens` is the query length — the only quantity that differs between the two runs:

| | query len | trips | × layers | × tensors | `overwrite` calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| warm | 128 | 128 | 4 | 2 | **1024** |
| offload | 1 | 1 | 4 | 2 | 8 + 4 decode = **12** |

Trace-measured: exactly 1024 vs 12. The arithmetic matches the observation.

---

## 4. What the code does, and how it maps to the numbers

The loop writes freshly computed K and V into the paged KV cache, one token at a time. Each
trip issues six device operations, and **each one is a full Python → dispatcher →
torch-spyre → job-launch round trip.**

### `key[t].unsqueeze(1).contiguous()` — 2 per trip

`key` is `(num_tokens, num_heads, head_size)`. Slicing row `t` and forcing contiguity
materializes a fresh `(1, 1, 128)` fp16 tensor — **256 bytes**.

### `convert(...)` — 2 per trip → `aten::_copy_from`, 384.7 ms

`spyre_inference/custom_ops/utils.py:75`. It short-circuits when device *and* dtype already
match (`utils.py:99`), but `key`/`value` arrive host-side here, so the guard does not fire
and the call goes through `torch.ops.vllm.spyre_convert` — an opaque custom op, deliberately
invisible to Dynamo (`utils.py:78-80`) — i.e. a real host→device transfer per call.

**320 µs to move 256 bytes.** This is pure latency: descriptor setup, submission, completion.
Payload size is irrelevant at this scale. Note the offload path's 180 calls average *594 µs*
— **warm is not paying more per call, it is paying 6.7× as many calls.** That is the whole
mechanism, stated in one number.

### `_overwrite(...)` — 2 per trip → 215.0 ms job launch + 73.7 ms dispatch

`spyre_attn.py:110-131`. On a spyre tensor it calls
`torch.ops.spyre.overwrite(input, output, dims, offsets)` (`:112-119`).

`offsets` is `[block_offsets[t]]` — a **concrete Python int**. Each distinct `t` is therefore
a distinct op invocation that builds and submits its own `sdsc_fused_overwrite` job plan at
**210 µs each**, plus **72 µs each** of PyTorch dispatcher plumbing before the plan is even
built. A job plan is the device's unit of dispatch; its cost is fixed whether it writes
256 bytes or 256 KB.

### The Python loop itself → `TorchDynamo Cache Lookup`, 139.2 ms

1666 lookups at ~84 µs. `torch-spyre` routes ops through `dispatch_to_torch_compile`, so
every op in the loop pays a guard check before it can run. `convert()` is opaque to Dynamo
by construction, but the surrounding slice/contiguous ops are not.

### The arithmetic

**~690 µs of fixed overhead per trip × 1024 trips ≈ 682 ms.** The device moves ~128 KB of
fp16 per layer-pair — trivial for a DMA engine. The cost is not the data; it is the **1024
separate round trips used to move it**. Warm asks the device to do the same work 85× more
often.

### Why attention width contributes nothing

`QUERY_CHUNK_SIZE = 32` quantizes the attention path: it launches the same 64 `bmm`, 60
`exp`, 32 `amax` job plans whether the query is 128 tokens (4 chunks) or 1 token (1 padded
chunk). Only the tensor shapes differ, and shape barely moves the clock at these sizes —
hence the identical call counts in §3. The `KV_LENGTH_ALIGNMENT = 256` padding has the same
flattening effect on the KV side.

### Why `block_size=64` recovered only 0.199 s

`TTFT_FINDINGS.md` row 4 halved the forced re-prefill (128 → 64 tokens) and recovered only
0.199 s of 0.723 s, which is what originally looked anomalous. It is exactly what a per-call
cost predicts: halving the trip count removes roughly half of the ~682 ms overhead, but that
overhead sits on top of a ~480 ms floor that both paths pay regardless. The cost scales with
**dispatch count, not FLOPs**, so relief is partial and proportional to trips — not to work.

---

## 5. Why this loop exists (and why the fix is not in this repo)

It is a workaround for a missing torch-spyre primitive, documented at
**`spyre_attn.py:923-932`**:

> `torch.ops.spyre.overwrite` is deprecated and its compile_once wrapper compiles one SDSC
> binary per unique offset, which recurses past the dynamo cache limit once vLLM's model
> compile has filled it. Raising the limit unblocks short tests but compiles N binaries for a
> query_len=N prefill, which doesn't scale to long contexts. […] Revisit when torch-spyre
> lands symbolic-offset overwrite (torch-spyre#220 / #1371-3).

`_overwrite` requires `offsets` as concrete Python ints (`spyre_attn.py:113-119`), so
scattered per-token block offsets can only be expressed one call at a time. The
`for t in range(num_tokens)` loop exists solely to satisfy that constraint. (The docstring at
`spyre_attn.py:241` — "Dynamo unrolls the loop because `num_tokens` is a closure constant" —
explains the compile-time shape story, not the runtime cost: unrolling removes Python loop
overhead but preserves all 1024 op invocations.)

The fix is a **batched write**: one `overwrite` per layer taking a *vector* of offsets, which
collapses 1024 invocations to 4 and removes nearly all 682 ms. That is blocked on
torch-spyre#220 / #1371-3, not on `spyre_inference`.

The same constraint bites a second time in the same file: the output-scatter path at
`spyre_attn.py:934` works around it differently — staging on CPU (`output_cpu`) and
bulk-copying at the end of the per-sequence loop. That is a viable pattern here too, and is
the most promising local mitigation short of the torch-spyre fix: stage all `num_tokens`
writes host-side, then issue one transfer per layer. Not attempted or measured.

---

## 6. Corrections to `TTFT_FINDINGS.md`

That document's "Open: the ~0.32 s residual" section is now closed. Both of its candidate
explanations need amending:

1. *"The warm path writes KV back"* — directionally right, wrong mechanism. The write itself
   (`spyre::overwrite`, 73.7 ms) is only 11% of the delta. The cost is the per-token
   **dispatch fan-out around it** — transfers, job-plan construction, and Dynamo lookups
   totaling 609 ms.
2. *"Per-step overhead differs by scheduling shape"* (`load_kv_async` splitting the offload
   path) — **wrong.** The phase breakdown in §2 shows everything outside the forward pass is
   a wash: preprocess −4.6 ms, postprocess +10.9 ms, sample and bookkeep ~0.

Also superseded, from earlier in this investigation: mask tiling, `slot_mapping.detach().cpu()`,
and `KV_LENGTH_ALIGNMENT` were each suspected and are each ruled out by the trace.

One intermediate claim of mine also needs retracting: I reported at one point that rows 1–3
of the self-time table were probably a *mix* of the write-back loop and 4×-wider attention.
The containment pass (§1.6) disproved that — attention call counts are identical between the
two runs, so the loop accounts for effectively all of the delta.

---

## 7. Still open

- **Numerical correctness of reloaded KV is unverified.** At `max_tokens=1` on repetitive
  filler, every run emits `'.'` — a degenerate signal. A real check means dumping a reloaded
  block and diffing it against a fresh prefill of the same positions.
- **The CPU-staging mitigation (§5) is unmeasured.**
- **Pre-existing bug in `ttft_evict.py`:** `BLOCK_SIZE = 128` (line 71) feeds only the pool
  arithmetic and `max_model_len`; it never reaches `LLM()`. The `[config] block_size=` line is
  a claim about intent — trust `[alloc] block_size=`, which reads back `cache_config`. Every
  measurement here is still valid because vLLM's default happens to be 128.

## Artifacts

Traces (kineto JSON, gzipped): `/tmp/pf_warm/`, `/tmp/pf_off2/` — **in `/tmp`, not yet moved
somewhere durable.** Uncommitted supporting changes: the `profile()` override in
`spyre_worker.py:158-169`, and the `--profile*` flags plus the pre-import env-var block in
`ttft_evict.py`.
