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
- [x] **Unit 1 — `SpyreKvDmaCopier` + env var.**
  `spyre_inference/v1/kv_offload/{__init__,copier}.py`; `SPYRE_KV_DMA_BACKEND` in
  `envs.py`. Backends: `torch_copy` (default, CPU+Spyre), `senlib_dma` (gated),
  `spyre_from_blob` (gated), `auto`.
- [x] **Unit 2 — KV adaptation layer.** `kv_offload/kv_adapter.py`. Plugin-side only,
  no vLLM patch. Bridges `SpyrePagedKVCache` page-lists to handler-consumable views.
- [x] **Unit 3 — `SpyreCpuOffloadingHandlers`.** `kv_offload/handlers.py`. d2h/h2d
  `OffloadingHandler`s against the 0.20.1 `worker/worker.py` contract; synchronous.
  Also exposes `transfer_count`/`blocks_transferred`/`bytes_transferred` counters
  and logs one line per transfer (host-hit signal; see runbook).
- [x] **Unit 4 — `SpyreOffloadingSpec` + registration.** `kv_offload/spec.py`
  (subclasses `OffloadingSpec` directly; reuses `CPUOffloadingManager`);
  `register_spec(...)` in `spyre_inference/__init__.py`. **Includes a minimal
  worker-side hook** (`TorchSpyreModelRunner.initialize_kv_cache`) — see deviation
  note below.
- [x] **Unit 5 — Tests.** `tests/v1/kv_offload/`:
  `test_spec_registration.py` (CPU), `test_handler_dispatch.py` (CPU),
  `test_copier_round_trip.py` (Spyre-gated), `test_e2e_offload.py` (Spyre-gated
  end-to-end via the `LLM` API).
- [x] **Unit 6 — Verify.** `ty check spyre_inference` clean, `ruff` lint + format
  clean. CPU host: 7 passed / 1 skipped. On a Spyre card: the e2e test boots the
  connector and matches the no-connector baseline at temperature=0 (see runbook).

### Verification note (how M1 was checked on this host)

`uv`/`uvx` could not provision the dev group here (it tried to build torch-spyre
from GitHub against a fresh `/usr/local` interpreter). Verification was instead run
against the already-provisioned runtime env at `/opt/spyre-inference` with
standalone `pytest`, `ruff==0.14.0`, and `ty==0.0.16` installed into it — the same
tool versions pinned in `.pre-commit-config.yaml`. On a properly `uv sync --group
dev`'d checkout the canonical commands (`uv run ty`, `bash format.sh`,
`uv run pytest -m "not upstream" tests/v1/kv_offload/`) should reproduce these
results.

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

## Runbook: running the end-to-end offload check on a Spyre card

This reproduces the RFC A1.1 check on your own Spyre environment. It runs in-process
through the `LLM` API (same engine path as `vllm serve`, easier to assert on).

### Prerequisites
- A host with a Spyre card and `torch-spyre` installed (the standard dev image).
- The `spyre-inference` dev env. If `uv run` can provision it for you:
  `uv sync --group dev`. On an image where `uv` cannot rebuild torch-spyre, run
  against the already-provisioned runtime interpreter instead (that is what was used
  to validate M1 here): `python -m pip install pytest` into that env, then invoke
  `python -m pytest ...` directly.

### Run the gated e2e test
```bash
# one Spyre process at a time — never run two Spyre-backed commands concurrently
uv run pytest -m "not upstream" tests/v1/kv_offload/test_e2e_offload.py -v
# or, against a hand-provisioned runtime interpreter:
SKIP_UPSTREAM_TESTS=1 python -m pytest -p no:spyre_testing_plugin -o addopts="" \
    tests/v1/kv_offload/test_e2e_offload.py -v
```
The test skips on CPU-only hosts. On a card it (1) boots an `LLM` with
`OffloadingConnector` + `SpyreOffloadingSpec`, and (2) asserts the generated tokens
are byte-identical to a no-connector baseline at `temperature=0`.

### What you should see in the worker log
- `Creating offloading spec with name: SpyreOffloadingSpec` — the factory resolved
  our lazily-registered spec.
- `SpyreKvDmaCopier using backend 'torch_copy'` — the device↔host copier initialized
  (auto-detected `torch_copy`, since the DMPA accessors / libsenlib are absent).
- If any KV block is physically moved, one line per transfer:
  `SpyreOffloadingHandler GPU->CPU: job=… blocks=N bytes=…` (store) or `CPU->GPU…`
  (host-tier load). `grep SpyreOffloadingHandler` over the run to count them.

### Compile-envelope caveat (important)
This card hits a **torch-spyre `torch.compile` RecursionError** in the Granite decode
path at roughly **≥40-token** prompts. It reproduces with **no connector at all**
(`granite.py: residual + hidden_states * residual_multiplier` →
`torch_spyre/ops/eager.py: dispatch_to_torch_compile` → Dynamo recursion), so it is a
torch-spyre issue, not M1's. A length sweep on this card: ~39 tokens OK, ~52 tokens
crash. The e2e test therefore uses ~34-token prompts. If you change the prompts, keep
them short, or you will hit this crash. (A larger model/longer context will need the
torch-spyre compile issue resolved first.)

### Manual `vllm serve` form (RFC A1.1 verbatim)
The same connector config also works under `vllm serve` (subject to the same
compile-envelope caveat for prompt length):
```bash
vllm serve ibm-ai-platform/micro-g3.3-8b-instruct-1b \
  --max-model-len 128 \
  --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"spec_name":"SpyreOffloadingSpec",
    "cpu_bytes_to_use":2000000000}}'
```

## Open question: triggering an actual device↔host transfer

The e2e check proves the connector **boots and is transparent** (identical output),
but in the small single-request workloads tested here **no KV block was physically
transferred** — the upstream scheduler's store-decision path
(`offloading/scheduler.py:_get_reqs_to_store`) did not engage, even with
`num_gpu_blocks_override=16` forcing eviction pressure. The store gate is
`num_offloadable_tokens // offloaded_block_size` minus already-stored blocks, then a
`block_id != 0` filter over the GPU block ids; understanding why it stays empty on
Spyre is upstream-vLLM / scheduler-interaction work, not M1 plugin code.

To verify an actual `GPU->CPU`/`CPU->GPU` transfer once this is understood, watch for
the `SpyreOffloadingHandler …` log lines (and/or the handler's `blocks_transferred`
counter). This is tracked as the remaining piece of A1.1; the device↔host *mechanism*
itself is unit-tested directly in `test_handler_dispatch.py` and
`test_copier_round_trip.py`.

## Acceptance status on a real Spyre card

Validated on-card (`micro-g3.3-8b-instruct-1b`, `max_model_len=128`):

- ✅ RFC **A1.1 boot + correctness** — `LLM` boots with `OffloadingConnector` +
  `SpyreOffloadingSpec`; the runner hook primes the spec and registers handlers
  without raising; generated tokens are byte-identical to the no-connector baseline
  at `temperature=0`. Covered by `tests/v1/kv_offload/test_e2e_offload.py`.
- ✅ RFC **A1.2** `test_copier_round_trip.py` — runs on the card.
- ⏳ RFC **A1.1 host-hit (actual transfer)** — NOT yet observed; the scheduler store
  path did not engage for the small workloads tested. See "Open question" above.

Still not exercised here:

- `senlib_dma`-backed copier round-trip (requires senlib from flex-runtime).
- Larger models / longer contexts (blocked by the torch-spyre compile recursion at
  ≥~40-token prompts — see the runbook's compile-envelope caveat).

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
