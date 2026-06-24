# `copy_tensor_raw` raw-layout guard — design context

**Date:** 2026-06-24
**Scope:** Design/context note for a *next* milestone — **no code change in the current KV-offload host-pool milestone.**
**Related:** torch-spyre PR #2796 (`copy_tensor_raw`); [Copier round-trip d2h mismatch](2026-06-15-copier-round-trip-d2h-mismatch.md) (the ±1 ULP root cause this builds on).

## Summary

torch-spyre PR #2796 adds `copy_tensor_raw` — a byte-for-byte host↔device DMA
that skips the layout/dtype conversion `copy_tensor` performs. It is the fix for
the fp16 ±1 ULP loss documented in the [copier round-trip log](2026-06-15-copier-round-trip-d2h-mismatch.md).
A review comment on that PR raises a correctness gap the current KV-offload work
needs to plan for:

> The returned data of a raw-copy should not be treated as a regular tensor
> because it doesn't have the proper layout like a host-side CPU tensor. There
> should be a mechanism to check so that using that as a regular host-side tensor
> is blocked, such as copying that back to device.

## Why the raw-copy output is not a regular host tensor

Confirmed by reading `torch-spyre` (`csrc/spyre_mem.cpp::spyre_copy_from`,
`csrc/module.cpp`) and the PR #2796 diff:

- **`copy_tensor`** (`raw_copy=false`): builds a `SpyreTensorLayout` +
  `DataConversionInfo` (DCI) and passes it to `copyAsyncImpl`. On d2h it reads
  the physical allocation via `dma_sizes/dma_strides/spyre_layout`, then
  re-applies the logical view on the CPU side. Result is **logically correct,
  host row-major** — but the converting copy is **lossy for stickified fp16**
  (the −1 ULP signature on values ≥ ~1024).
- **`copy_tensor_raw`** (`raw_copy=true`): passes `nullptr` for the DCI — skips
  `generate_dci`, byte-for-byte. Result is **bit-exact** but carried in the
  device's **64-element stick layout**, *not* host row-major.

The C++ binding returns/fills a plain `at::Tensor` and the PR adds **no `.pyi`
stub or Python wrapper** — callers use raw `_C.copy_tensor_raw`. Nothing prevents
treating its d2h output as a normal CPU tensor.

### Why a symmetric round-trip hides the hazard

PR #2796's test does raw-H2D then raw-D2H **symmetrically**, so the device layout
cancels and `torch.equal(cpu_src, cpu_dst)` holds. The hazard is asymmetry:
raw-d2h output used as a regular CPU tensor (read values, CPU op, `.numpy()`,
print), or copied back with the **converting** `copy_tensor` instead of
`copy_tensor_raw`, silently corrupts — the bytes are in stick layout.

## Implication for KV offload

For offloading, raw copy is *desirable*: `raw-d2h → host staging → raw-h2d`
round-trips the opaque device-layout payload bit-exactly, so the conversion loss
never occurs. **Constraint:** a staged host page is only ever valid as an opaque
blob destined for raw-h2d back into the same layout. Nothing may interpret it as
a regular host tensor in between.

## Guard design sketch (next step)

Two layers, when we adopt `copy_tensor_raw`:

1. **torch-spyre** — add a Python wrapper over `_C.copy_tensor_raw` (the PR adds
   none). The caller already allocates `dst` (see `model_utils.py`), so the
   wrapper can mark the caller-owned output. Candidate mechanisms:
   - **Attribute tag** (e.g. `dst._spyre_raw_layout = True`): `raw_h2d` asserts
     the source carries it; converting paths (`copy_tensor`, `.to('cpu')`) assert
     it is absent. Simple, fits caller-allocates-dst; weak as a general guard
     (attribute lost across torch ops — acceptable here since pages are opaque
     and never pass through ops).
   - **Wrapper dataclass** `RawHostBlock(tensor, layout_meta)`: APIs accept/return
     it; unwrapping to a bare tensor is explicit and greppable. Strongest;
     requires the handler/adapter to hold `RawHostBlock` in
     `host_k_pages`/`host_v_pages`.
   - tensor subclass — rejected: C++ returns plain `at::Tensor`; subclassing is
     fragile and fights the `_C` path.

2. **spyre-inference copier** (`spyre_inference/v1/kv_offload/copier.py`):
   `SpyreKvDmaCopier.copy_d2h`/`copy_h2d` switch from `copy_tensor` to
   `copy_tensor_raw`, allocate raw-marked host pages in `kv_adapter._alloc_host_pages`,
   and assert the mark on `copy_h2d` so a non-raw or wrong-layout dest is
   rejected. Gated on PR #2796 being available in the pinned torch-spyre rev.

## Current milestone (for the record)

- Stays on `copy_tensor` (works for the offload round-trip, lossy but
  flow-correct).
- Verifies evict(d2h) + load(h2d) via the behavioral test
  `tests/v1/kv_offload/test_host_pool_offload_reload.py` (4 device / 8 host
  blocks, 5 stores + 1 load, flow/counter assertions, **no content checks**).
- Does **not** touch torch-spyre or swap the copy primitive.
