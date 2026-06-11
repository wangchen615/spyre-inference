# Tracking: M1 implementation of the upstream KV-connector port

Companion to [`upstream-connector-port.md`](./upstream-connector-port.md). This file
tracks the concrete implementation of **M1** (single-tier host-RAM KV offload) on the
`dev/kv-offload-m1` branch of the fork. Issues are disabled on the fork, so this
doc is the work checklist in lieu of GitHub subissues.

## Scope of this pass

- **In:** M1 only — `SpyreOffloadingSpec` + `OffloadingConnector`, committed as the
  units below.
- **Out:** M1.5 (`SpyreTieringOffloadingSpec`, `tiering/fs`, `tiering/obj`) — see
  "Deferred" below. The two upstream PRs are opened only after M1 (and later M1.5)
  test green on real Spyre hardware.

## What changed under the RFC (verified against the live install, 2026-06-11)

The RFC was written against an assumed vLLM **v0.22**. The pinned/installed
environment differs in four ways that reshape M1:

1. **Installed vLLM is 0.20.1, not v0.22.** The single-tier `OffloadingConnector` +
   `CPUOffloadingSpec` that M1 needs exist (at different paths than the RFC cites:
   `vllm/v1/kv_offload/abstract.py` not `base.py`; `worker/cpu_gpu.py` not
   `cpu/gpu_worker.py`; the worker is a package `.../offloading/worker.py`). The
   `tiering/*` framework M1.5 needs does **not** exist here.
2. **The RFC §6.6 "zero plugin change" premise is false against the current model
   runner.** `initialize_kv_cache_tensors`
   (`spyre_inference/v1/worker/spyre_model_runner.py:328`) binds each layer to a
   `SpyrePagedKVCache(k_pages, v_pages)` — two Python lists of per-block tensors
   (`[num_kv_heads, block_size, head_size]`, fp16, `device("spyre")`). vLLM's
   `OffloadingConnectorWorker.register_kv_caches` asserts each layer is a single
   `torch.Tensor` and rebuilds it via `untyped_storage().set_().view(...)`. A paged
   list is neither, so a plugin-side adaptation layer is required (Unit 2).
3. **Storage/pointer reinterpretation and on-device slicing are not viable on
   Spyre.** The working device↔host idiom here is whole-tensor `.to("cpu")` /
   `.copy_()` via `torch_spyre._C.copy_tensor` (see `custom_ops/utils.py:64`
   `convert()`; `spyre_attn.py` notes "Spyre slicing corrupts memory"). The CUDA
   handler's `.data_ptr()` + `ops.swap_blocks_batch` byte-pointer approach is not
   reusable.
4. **Neither RFC copier backend runs on this host.** `torch_spyre._C` exposes no
   `dma`/`dmpa`/`from_blob` symbols (RFC §10 Q1 — confirmed), and `senlib` is not
   importable here. `senlib` is installable from flex-runtime on a real dev image, so
   `senlib_dma` is the eventual real backend — but it can't be exercised on this
   CPU-only host.

Consequence: M1 carries one more unit than the RFC §7 estimate (the Unit 2 adapter),
and the copier ships with a CPU-testable `torch_copy` backend as the default plus
`NotImplementedError` stubs for the real hardware backends until they are available.

## Unit checklist

- [x] **Unit 0 — Tracking + branch.** `dev/kv-offload-m1` off
  `rfc/upstream-connector-port`; this doc.
- [ ] **Unit 1 — `SpyreKvDmaCopier` + env var.**
  `spyre_inference/v1/kv_offload/{__init__,copier}.py`; `SPYRE_KV_DMA_BACKEND` in
  `envs.py`. Backends: `torch_copy` (default, CPU+Spyre), `senlib_dma` (gated),
  `spyre_from_blob` (gated), `auto`.
- [ ] **Unit 2 — KV adaptation layer.** `kv_offload/kv_adapter.py`. Plugin-side only,
  no vLLM patch. Bridges `SpyrePagedKVCache` page-lists to handler-consumable views.
- [ ] **Unit 3 — `SpyreCpuOffloadingHandlers`.** `kv_offload/handlers.py`. d2h/h2d
  `OffloadingHandler`s against the 0.20.1 `worker/worker.py` contract; synchronous.
- [x] **Unit 4 — `SpyreOffloadingSpec` + registration.** `kv_offload/spec.py`
  (subclasses `OffloadingSpec` directly; reuses `CPUOffloadingManager`);
  `register_spec(...)` in `spyre_inference/__init__.py`. **Includes a minimal
  worker-side hook** (`TorchSpyreModelRunner.initialize_kv_cache`) — see deviation
  note below.
- [ ] **Unit 5 — Tests.** `tests/v1/kv_offload/`:
  `test_spec_registration.py` (CPU), `test_handler_dispatch.py` (CPU),
  `test_copier_round_trip.py` (Spyre-gated, skips on CPU-only).
- [ ] **Unit 6 — Verify.** `uv run ty`, `bash format.sh`, CPU pytest green.

## Deviation from RFC A1.3 (recorded)

RFC A1.3 set a goal of "no source changes to `TorchSpyreWorker` or
`TorchSpyrePlatform`" for M1. Implementation showed this is not achievable, because
the upstream `OffloadingConnectorWorker.register_kv_caches` canonicalizes each
layer's KV cache via `untyped_storage().set_().view(...)` *before* `get_handlers`
runs — and that crashes on Spyre's `SpyrePagedKVCache` (a list of per-block tensors,
not a single tensor whose storage can be reinterpreted; finding #2/#3 above).

**Decision (user):** add a minimal, well-scoped hook in
`TorchSpyreModelRunner.initialize_kv_cache` (the model runner, not the worker class
itself). It detects a `SpyreOffloadingSpec`-backed `OffloadingConnector` and, only
then, temporarily swaps the connector's `register_kv_caches` for a Spyre path that
primes the spec with the raw paged dict and registers the device<->host handlers
directly — bypassing the upstream canonicalization. Every other connector and the
non-connector path are untouched, and the base `initialize_kv_cache` orchestration is
otherwise reused verbatim via `super()`. `TorchSpyrePlatform` is unchanged.

Cleaner long-term alternative (filed, not done): the RFC §10 Q2 upstream one-liner
that lets `register_kv_caches` tolerate a non-tensor/paged cache would remove the need
for this hook. Tracked under "Open follow-ups".

## Acceptance gates deferred to a Spyre dev image

These cannot run on this CPU-only host and are not exercised in this pass:

- RFC **A1.1** end-to-end `vllm serve --kv-transfer-config '{... "spec_name":
  "SpyreOffloadingSpec" ...}'` two-prompt host-hit check at `temperature=0`.
- RFC **A1.2** `test_copier_round_trip.py` on a Spyre runner.
- `senlib_dma`-backed copier round-trip (requires senlib from flex-runtime).

## Open follow-ups (recorded, not done here)

- **vLLM v0.22 bump** → unblocks M1.5 (`SpyreTieringOffloadingSpec`, `tiering/*`).
- **Upstream the torch-spyre DMPA accessors** (`flim/pd-disagg` `93dc1ae`) → lets the
  copier drop to a single `spyre_from_blob` backend and removes `SPYRE_KV_DMA_BACKEND`
  (RFC §11).
- **Possible upstream one-liner** relaxing the `isinstance(..., torch.Tensor)` assert
  in `register_kv_caches` (RFC §10 Q2). If Unit 2 shows this would meaningfully
  simplify the adapter, it is filed here as an upstream change — **not** silently
  patched into site-packages.

## Deferred: M1.5

M1.5 is blocked on the vLLM v0.22 bump (finding #1). It reuses M1's
`SpyreOffloadingSpec` device↔host handler and adds only the `SecondaryTierManager`
plumbing via `SpyreTieringOffloadingSpec` (~50 LOC per RFC §4.3, §6.4). Revisit once
the pin moves.
