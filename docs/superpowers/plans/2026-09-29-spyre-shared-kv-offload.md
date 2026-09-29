# Spyre Cross-Instance Shared KV Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicitly selected Spyre offloading tier that lets two TP1 vLLM instances on one host publish and reload byte-identical KV blocks from the same Flex shared-memory directory and pool family.

**Architecture:** Keep M1's `SpyreOffloadingSpec` and private-pool behavior intact while extracting only construction and page-copy seams. M2 adds a policy-neutral `CPUOffloadingManager` subclass for shared-directory lookup/claim/pin bookkeeping and a `SpyreOffloadingWorker` subclass that routes the existing validated raw page copy through dynamically registered shared pool families. The plugin registers the M2 spec lazily, and the end-to-end run disables prefix caching so both A's self-reload and B's peer reload demonstrably come from shared host memory.

**Tech Stack:** Python 3.12, vLLM 0.28 offloading interfaces, PyTorch/torch-spyre, Flex `SharedMetadata` and `SharedHostPool`, pytest, multiprocessing, OpenAI-compatible vLLM HTTP API.

**Spec:** `docs/superpowers/specs/2026-09-29-spyre-shared-kv-offload-design.md`

## Global Constraints

- Make spyre-inference changes only in `/home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2` on branch `kvc-offload-m2`, based on M1 commit `86f56ed`.
- Make conditional Flex and torch-spyre changes only on their existing `kvc-offload-m2` branches; do not touch the dirty `kvc-offload-poc` worktree.
- M2 supports float16 KV pages, one KV-cache group, `blocks_per_chunk == 1`, `world_size == 1`, TP1, and the `uni` executor.
- `SpyreOffloadingSpec` keeps its existing engine/rank-private pools, configuration, manager type, and observable transfer behavior.
- `SpyreSharedOffloadingSpec` must remain inert until selected: importing `spyre_inference` must not import `shared_spec.py` or access `torch_spyre._C.SharedMetadata`.
- Reuse upstream admission threshold, LRU/ARC choice, ref-counting, metrics, and local victim selection; do not add a host-wide eviction policy or silently adopt peer entries into local policy.
- Publish one logical KV block only after every K/V component D2H copy has synchronized; retain each read pin until the corresponding H2D job completes.
- A missing, reserved, stale, or unpinnable directory entry is a normal `LookupResult.MISS`; configuration mismatch, `Unavailable`, transfer failure, and impossible component routing fail loudly.
- Do not use or modify vLLM `SharedOffloadRegion`; do not add disk, network, multi-host, DP, or cross-TP sharing.
- Run `uv` commands with `--no-sync` while using the local torch-spyre checkout.
- Never run two independent Spyre-backed commands concurrently. The sole exception is the deliberate two-instance acceptance topology, with one process restricted to each distinct card.
- Every commit uses `git commit -s`; before it, run `SKIP=markdownlint pre-commit run --files <changed-files>`.
- If the real KV allocations have more than one `CompositeAddress` chunk, stop spyre-inference execution at Task 7 and complete the conditional M2-F3 plan described there before continuing.

## File Structure

- Modify `spyre_inference/v1/kv_offload/spec.py`: add protected M1 construction hooks and a no-op full-vLLM-config binding seam.
- Modify `spyre_inference/v1/kv_offload/connector.py`: pass the full `VllmConfig` into the selected Spyre spec before worker construction.
- Modify `spyre_inference/v1/kv_offload/worker.py`: expose the existing single-group validation to the M2 subclass without changing M1 dispatch.
- Modify `spyre_inference/v1/worker/spyre_kv_offload.py`: extract one validated K/V page-copy helper used by both M1 and M2 routing.
- Create `spyre_inference/v1/kv_offload/shared_types.py`: runtime-neutral family, location, transfer-item, and transfer-spec value objects plus deterministic key hashing.
- Create `spyre_inference/v1/kv_offload/shared_runtime.py`: the only lazy import boundary for the required torch-spyre M2 binding symbols.
- Create `spyre_inference/v1/kv_offload/shared_manager.py`: upstream-policy integration and shared lookup/claim/pin lifecycle.
- Create `spyre_inference/v1/kv_offload/shared_worker.py`: directory/pool registration, family routing, DMA, publish, and abort handling.
- Create `spyre_inference/v1/kv_offload/shared_spec.py`: M2 configuration validation, compatibility digest construction, and manager/worker factories.
- Modify `spyre_inference/__init__.py`: lazy `OffloadingSpecFactory` registration using module and class strings.
- Extend `tests/kv_offload/test_spec.py`, `tests/kv_offload/test_spyre_kv_offload.py`, and `tests/kv_offload/test_worker_dispatch.py`: M1 preservation gates.
- Create `tests/kv_offload/test_shared_types.py`: pure key, family, and transfer-shape tests.
- Create `tests/kv_offload/test_shared_manager.py`: fake-directory policy and lifecycle tests.
- Create `tests/kv_offload/test_shared_worker_dispatch.py`: fake-runtime pool routing, ordering, and failure tests.
- Create `tests/kv_offload/test_shared_spec.py`: configuration, compatibility, factory, and lazy-import tests.
- Create `tests/kv_offload/test_connector_miss_recompute.py`: connector translation of a shared-directory miss into zero external tokens and no H2D job.
- Create `tests/kv_offload/test_shared_pool_round_trip.py`: real-Spyre shared-worker round trip and `CompositeAddress` chunk gate.
- Create `tests/kv_offload/test_cross_instance.py`: opt-in two-server functional acceptance, deterministic output comparison, and A/B reload measurements.
- Create `docs/superpowers/results/2026-09-29-spyre-shared-kv-offload.md`: exact hardware run, revisions, functional evidence, and A/B timing results.

## Review Focus

- A same-sized but incompatible model, revision, hash algorithm, hash seed, layout, or page geometry must fail attachment before DMA; Task 5 varies every encoded field.
- A partially successful multi-key claim must not leak reservations or pending upstream policy entries when a later key returns `ExistingClaim`, `NoSpace`, or `Unavailable`; Task 3 checks each rollback path and a mixed batch.
- Lookup pins must survive until `complete_load` but must not leak when lookup is abandoned before `prepare_load`; Task 3 checks both request-finalization paths.
- A peer hit must be loadable without entering the local LRU/ARC, and local eviction must target only the exact versioned entry owned by that manager; Task 3 inspects the upstream policy and eviction calls.
- A D2H or publish failure after earlier component copies must synchronize before aborting every unpublished reservation and must never leave a reported hit; Task 4 checks the complete call order and cleanup state.

---

### Task 1: Extract M1-Preserving Construction and Copy Seams

**Files:**

- Modify: `spyre_inference/v1/kv_offload/spec.py:151-224`
- Modify: `spyre_inference/v1/kv_offload/connector.py:322-350`
- Modify: `spyre_inference/v1/kv_offload/worker.py:115-172`
- Modify: `spyre_inference/v1/worker/spyre_kv_offload.py:170-250`
- Modify: `tests/kv_offload/test_spec.py`
- Modify: `tests/kv_offload/test_spyre_kv_offload.py`
- Modify: `tests/kv_offload/test_worker_dispatch.py`

**Interfaces:**

- Consumes: existing `SpyrePhysicalCaches`, `CPUOffloadingManager`, `SpyreOffloadingWorker`, and `copy_kv_page_raw` contracts.
- Produces: `SpyreOffloadingSpec.bind_vllm_config(vllm_config)`, `_create_manager()`, `_create_worker(physical)`, `SpyreOffloadingWorker._validate_gpu_spec(gpu_spec)`, and `copy_kv_page_pair(copy_fn, cache, block_id, k_pool, k_slot_id, v_pool, v_slot_id, to_device, non_blocking=False)`.

- [ ] **Step 1: Write failing tests for the construction hooks and unchanged M1 arguments**

Add a recording subclass in `test_spec.py` and assert `get_manager()` remains memoized while `get_worker()` validates canonical tensor count before calling its hook:

```python
class RecordingSpec(SpyreOffloadingSpec):
    def _create_manager(self):
        self.manager_calls = getattr(self, "manager_calls", 0) + 1
        return object()

    def _create_worker(self, physical):
        self.worker_physical = physical
        return SimpleNamespace(_bytes_per_block=KV_BYTES_PER_BLOCK)


def test_construction_hooks_preserve_memoization_and_physical_binding():
    spec = RecordingSpec(_config())
    manager = spec.get_manager()
    assert spec.get_manager() is manager
    assert spec.manager_calls == 1

    physical = _physical_for_spec()
    spec.bind_physical_caches(physical)
    worker = spec.get_worker(_canonical_for_spec(physical))
    assert worker is not None
    assert spec.worker_physical is physical
```

Add a connector test whose spec records `bind_vllm_config` and prove the call occurs before `_init_worker`. Add a page-copy test that injects a recording `copy_fn` and expects exactly K then V with M1's `2 * slot` and `2 * slot + 1` indices.

- [ ] **Step 2: Run the new tests and verify the seams are absent**

Run:

```bash
uv run --no-sync pytest tests/kv_offload/test_spec.py tests/kv_offload/test_spyre_kv_offload.py tests/kv_offload/test_worker_dispatch.py -m "not upstream" -q
```

Expected: FAIL because the protected hooks, vLLM-config binding, shared page-copy helper, and public validation seam do not exist.

- [ ] **Step 3: Refactor M1 without changing behavior**

Keep `get_manager()` and `get_worker()` as the public entry points and move their current constructors behind these hooks:

```python
def bind_vllm_config(self, vllm_config: VllmConfig) -> None:
    return

def _create_manager(self) -> OffloadingManager:
    return CPUOffloadingManager(
        num_blocks=self.num_blocks,
        cache_policy=self.eviction_policy,
        cache_policy_module_path=self.cache_policy_module_path,
        enable_events=self.kv_events_config.enable_kv_cache_events,
        store_threshold=int(self.extra_config.get("store_threshold", 0)),
        max_tracker_size=int(self.extra_config.get("max_tracker_size", 64_000)),
    )

def _create_worker(self, physical: SpyrePhysicalCaches) -> OffloadingWorker:
    return SpyreOffloadingWorker(
        physical=physical,
        num_host_blocks=self.num_blocks,
        pool_prefix=self._pool_prefix,
    )
```

Change the connector registration order to:

```python
self.spec.bind_vllm_config(self.vllm_config)
self.spec.bind_physical_caches(physical)
self._init_worker(canonical)
```

Move the single-group check into `_validate_gpu_spec()`. Extract the copy helper and make `SpyreKvPageOffloader.offload()` and `.reload()` delegate to it with the existing paired-slot mapping. The helper must issue only these calls:

```python
copy_fn(k_pages, block_id, k_pool, k_slot_id, to_device, non_blocking)
copy_fn(v_pages, block_id, v_pool, v_slot_id, to_device, non_blocking)
```

- [ ] **Step 4: Run the focused M1 tests**

Run the command from Step 2. Expected: all tests pass with the same M1 copy order, manager class, pool names, transfer byte counts, and failures.

- [ ] **Step 5: Run pre-commit and commit the seam**

```bash
SKIP=markdownlint pre-commit run --files spyre_inference/v1/kv_offload/spec.py spyre_inference/v1/kv_offload/connector.py spyre_inference/v1/kv_offload/worker.py spyre_inference/v1/worker/spyre_kv_offload.py tests/kv_offload/test_spec.py tests/kv_offload/test_spyre_kv_offload.py tests/kv_offload/test_worker_dispatch.py
git add spyre_inference/v1/kv_offload/spec.py spyre_inference/v1/kv_offload/connector.py spyre_inference/v1/kv_offload/worker.py spyre_inference/v1/worker/spyre_kv_offload.py tests/kv_offload/test_spec.py tests/kv_offload/test_spyre_kv_offload.py tests/kv_offload/test_worker_dispatch.py
git commit -s -m "Refactor Spyre KV offload construction"
```

If `ty` alone reports the repository's existing unavailable local torch-spyre imports, record that separately; do not weaken type checks or alter unrelated files.

### Task 2: Define Runtime-Neutral Shared Transfer Contracts

**Files:**

- Create: `spyre_inference/v1/kv_offload/shared_types.py`
- Create: `spyre_inference/v1/kv_offload/shared_runtime.py`
- Create: `tests/kv_offload/test_shared_types.py`

**Interfaces:**

- Consumes: `vllm.v1.kv_offload.base.OffloadKey` and `LoadStoreSpec`.
- Produces: `SharedPoolFamily`, `SharedLocation`, `SharedTransfer`, `SharedLoadStoreSpec`, `allocate_family_slots`, `shared_block_hash`, and `load_shared_runtime()`.

- [ ] **Step 1: Write failing pure-Python tests for family allocation, key hashing, and transfer validation**

Cover even and remainder capacity distribution, duplicate/empty family names, fewer slots than families, deterministic 64-bit hashing of the complete `OffloadKey`, a group-ID change altering the hash, and mismatched transfer tuple lengths. Pin the digest with a literal expected integer:

```python
def test_shared_block_hash_covers_the_complete_offload_key():
    key0 = make_offload_key(bytes.fromhex("11" * 32), 0)
    key1 = make_offload_key(bytes.fromhex("11" * 32), 1)
    assert shared_block_hash(key0) == 0x632D1E16D4E6599B
    assert shared_block_hash(key1) != shared_block_hash(key0)


def test_family_slots_distribute_remainder_in_configuration_order():
    assert allocate_family_slots(8, ("alpha", "beta", "gamma")) == (
        SharedPoolFamily("alpha", 3),
        SharedPoolFamily("beta", 3),
        SharedPoolFamily("gamma", 2),
    )
```

The algorithm is fixed to BLAKE2b, eight output bytes, personalization
`b"spyre-m2"`, interpreted unsigned big-endian.

- [ ] **Step 2: Run the pure tests and verify the module is absent**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_types.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `shared_types`.

- [ ] **Step 3: Implement immutable transfer values and strict validation**

Use these public shapes:

```python
@dataclass(frozen=True)
class SharedPoolFamily:
    name: str
    slot_count: int


@dataclass(frozen=True)
class SharedLocation:
    anchor_pool_id: int
    slot_id: int


@dataclass(frozen=True)
class SharedTransfer:
    key: OffloadKey
    location: SharedLocation
    reservation: object | None = None


class SharedLoadStoreSpec(LoadStoreSpec):
    def __init__(self, transfers: Collection[SharedTransfer]) -> None:
        self.transfers = tuple(transfers)
```

Reject empty names, duplicate names, non-positive capacities, and transfer items whose store/load mode is mixed. `shared_block_hash()` must hash `bytes(key)`, not only `get_offload_block_hash(key)`, so group IDs cannot alias.

- [ ] **Step 4: Implement the lazy torch-spyre M2 surface loader**

`shared_runtime.py` imports `torch_spyre._C` only inside `load_shared_runtime()`, verifies these names, and raises one direct `RuntimeError` listing missing names:

```python
REQUIRED_SHARED_SYMBOLS = (
    "ChunkDescriptorEntry",
    "CompatibilityDescriptor",
    "CompatibleBlockKey",
    "ExistingClaim",
    "NoSpace",
    "Reservation",
    "SharedDataPoolConfig",
    "SharedMetadata",
    "SharedMetadataCapacity",
    "SharedMetadataConfig",
    "SharedPoolKind",
    "Unavailable",
    "copy_kv_page_raw",
    "get_composite_address",
)
```

Return the extension module itself so tests can replace it with one fake object and manager/worker code can use the same concrete classes for `isinstance` checks.

- [ ] **Step 5: Run tests, pre-commit, and commit**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_types.py -q
SKIP=markdownlint pre-commit run --files spyre_inference/v1/kv_offload/shared_types.py spyre_inference/v1/kv_offload/shared_runtime.py tests/kv_offload/test_shared_types.py
git add spyre_inference/v1/kv_offload/shared_types.py spyre_inference/v1/kv_offload/shared_runtime.py tests/kv_offload/test_shared_types.py
git commit -s -m "Add shared KV offload contracts"
```

Expected: the pure tests pass without importing or initializing torch-spyre.

### Task 3: Integrate Upstream Cache Policy with the Shared Directory

**Files:**

- Create: `spyre_inference/v1/kv_offload/shared_manager.py`
- Create: `tests/kv_offload/test_shared_manager.py`

**Interfaces:**

- Consumes: `SharedPoolFamily`, `SharedLocation`, `SharedTransfer`, `SharedLoadStoreSpec`, `shared_block_hash`, and the torch-spyre objects returned by `load_shared_runtime()`.
- Produces: `SpyreSharedOffloadingManager(CPUOffloadingManager)` with the unchanged upstream `OffloadingManager` public methods.

- [ ] **Step 1: Build a deterministic fake SharedMetadata protocol and write failing lookup tests**

The fake must model `lookup`, `pin_read`, `claim`, `publish`, `abort`, `evict`, `find_pool`, slot versions, and destruction-observable read pins. Test:

```python
def test_peer_lookup_pins_without_entering_local_policy(
    manager, directory, peer_entry, ctx
):
    directory.entries[peer_entry.key.block_hash] = peer_entry
    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.HIT
    assert manager._policy.get(OFFLOAD_KEY) is None

    spec = manager.prepare_load([OFFLOAD_KEY], ctx)
    assert spec.transfers[0].location == SharedLocation(
        peer_entry.slot.pool.pool_id, peer_entry.slot.slot_id
    )
    assert peer_entry.pin.released is False

    manager.complete_load([OFFLOAD_KEY], ctx)
    assert peer_entry.pin.released is True
```

Also assert missing lookup, reserved lookup, and failed `pin_read` each return `LookupResult.MISS` and create no load spec or H2D opportunity.

- [ ] **Step 2: Run lookup tests and verify they fail**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_manager.py -q
```

Expected: FAIL because `SpyreSharedOffloadingManager` does not exist.

- [ ] **Step 3: Implement lazy attach, lookup, and pin lifetime**

Construct the superclass with exactly the M1 policy arguments. Attach lazily with an empty initial pool list and fixed capacity:

```python
SharedMetadataConfig(
    max_chunks=max_components,
    pools=[],
    capacity=SharedMetadataCapacity(
        len(families) * max_components,
        max(family.slot_count for family in families),
        1,
    ),
)
```

Resolve every `<family>.c0.k` anchor before the first operation and require all anchors to carry the same compatibility ID. Convert a vLLM key with:

```python
CompatibleBlockKey(anchor.compatibility, shared_block_hash(key))
```

Store pending and active pins in a request-local dataclass via
`ReqContext.set_state()`. `lookup()` calls `super().lookup()` first so
store-threshold accounting stays upstream-owned; local pending stores preserve
`HIT_PENDING`, while any published directory entry becomes `HIT` only after
`pin_read()` succeeds. `prepare_load()` calls `super().prepare_load()` only for
locally owned keys, moves selected pins from pending to active, releases lookup
pins that were acquired during scanning but were not selected, and returns
locations in input order. `complete_load()` releases active pins after calling
the superclass for locally owned keys. `on_request_finished()` releases
abandoned pending pins but leaves active pins for later completion callbacks.

- [ ] **Step 4: Write failing store, eviction, and rollback tests**

Cover these exact outcomes:

```python
@pytest.mark.parametrize("claim_result", ["existing_valid", "existing_reserved", "no_space"])
def test_non_reservation_claim_rolls_back_local_admission(claim_result, manager, ctx):
    manager.directory.next_claim = claim_result
    out = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert out is not None
    assert out.keys_to_store == []
    assert manager._policy.get(OFFLOAD_KEY) is None
    assert manager.directory.live_reservations == []
```

Add a mixed three-key batch in which the first key reserves, the second is an existing peer entry, and the third exhausts every family. Require only the first key in `keys_to_store`, and require the other two absent from local policy. Add a separate `Unavailable` case that aborts every reservation already acquired, rolls back every newly inserted local entry, and raises `RuntimeError`. Fill a one-slot local policy, complete its store, then admit a second key and assert the directory receives `evict()` with the first key's exact versioned `LookupEntry`.

Add a two-family claim test whose key hashes to the second family first. Make
that family return `NoSpace`, make the first family return `Reservation`, and
assert both the deterministic probe order and the location actually returned.
Also call `lookup()` before anchor registration and require a direct startup
ordering error instead of an unexplained miss.

- [ ] **Step 5: Implement policy-neutral store and eviction integration**

`prepare_store()` follows this order:

1. Call `super().prepare_store(keys, req_context)` once.
2. For each `evicted_key`, remove its owned entry and call `directory.evict(entry)`.
3. For each upstream-admitted key, try families starting at `shared_block_hash(key) % len(families)` and wrapping once.
4. Keep `Reservation` results; on `NoSpace`, try the next family.
5. On either `ExistingClaim` value or all-family `NoSpace`, call `super().complete_store([key], req_context, success=False)` and omit that key from the returned store spec.
6. On `Unavailable` or another protocol exception, abort all acquired reservations, roll back all newly admitted keys through the superclass, and raise a direct runtime error.

Return:

```python
PrepareStoreOutput(
    keys_to_store=[item.key for item in transfers],
    store_spec=SharedLoadStoreSpec(transfers),
    evicted_keys=upstream.evicted_keys,
)
```

On successful `complete_store`, re-lookup every published key, require a valid entry, let the superclass mark it ready, and record it in `_owned_entries`. On failure, remove pending reservations and call the superclass with `success=False`. `reset_cache()` evicts only `_owned_entries`, clears pending request pins/reservations after scheduler quiescence, and then calls `super().reset_cache()`. Never insert peer entries into `_policy`.

- [ ] **Step 6: Compare policy decisions with the upstream manager**

Drive an upstream `CPUOffloadingManager` and the shared manager through the same `store_threshold=2` lookup sequence, completed stores, touches, and one-capacity eviction. Assert the same admitted keys and victim key; additionally assert that only the shared manager emits the matching directory claim/eviction operations. Repeat with `eviction_policy="arc"`.

- [ ] **Step 7: Run tests, pre-commit, and commit**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_manager.py -q
SKIP=markdownlint pre-commit run --files spyre_inference/v1/kv_offload/shared_manager.py tests/kv_offload/test_shared_manager.py
git add spyre_inference/v1/kv_offload/shared_manager.py tests/kv_offload/test_shared_manager.py
git commit -s -m "Add shared KV offload manager"
```

Expected: all manager tests pass without a Spyre device or real shared memory.

### Task 4: Register Pool Families and Transfer Complete Logical Blocks

**Files:**

- Create: `spyre_inference/v1/kv_offload/shared_worker.py`
- Create: `tests/kv_offload/test_shared_worker_dispatch.py`

**Interfaces:**

- Consumes: `SpyreOffloadingWorker._run()`, `_validate_gpu_spec()`, `copy_kv_page_pair()`, `SharedLoadStoreSpec`, and the torch-spyre M2 directory/pool API.
- Produces: `SpyreSharedOffloadingWorker(SpyreOffloadingWorker)` with inherited `submit_store`, `submit_load`, `get_finished`, and `wait` behavior.

- [ ] **Step 1: Write failing initialization and routing tests with a fake runtime**

For two families with slot counts `(3, 2)` and two physical caches, assert registration in this exact order:

```text
alpha.c0.k, alpha.c0.v, alpha.c1.k, alpha.c1.v,
beta.c0.k, beta.c0.v, beta.c1.k, beta.c1.v
```

Assert each pool uses the family's logical slot count, each K/V pool uses its own physical page size, all pools use one compatibility descriptor, and anchor pool IDs map back to the correct family. A store at `SharedLocation(alpha_anchor_id, 2)` must call the extracted copy helper for every cache with slot `2` in that family's K and V pools. A load through beta must use only beta pools.

- [ ] **Step 2: Run worker tests and verify they fail**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_worker_dispatch.py -q
```

Expected: FAIL because the shared worker does not exist.

- [ ] **Step 3: Implement directory creation, component registration, and family routing**

Create or attach the directory with the same empty-pool capacity config as Task 3. For every physical cache and K/V role, build:

```python
SharedDataPoolConfig(
    f"{family.name}.c{cache_index}.{role}",
    SharedPoolKind.HOST,
    family.slot_count,
    signature.page_bytes,
    CompatibilityDescriptor(COMPATIBILITY_FORMAT_VERSION, list(digest)),
)
```

Call `register_or_attach_pool()`, then `resolve_pool()` and fail if resolution returns `None`. Index the resolved family bundle by the `pool_id` of `c0.k`. Validate that every allocation's `get_composite_address(tensor).num_chunks == 1`; retain one `ChunkDescriptorEntry(domain_id, page_size_bytes)` per K/V component, in cache-index then K/V order, for publication.

- [ ] **Step 4: Write failing ordering and cleanup tests**

Record all copy, synchronize, publish, abort, and evict operations. Require a successful store to be ordered as:

```text
pre-transfer synchronize
all K/V D2H copies for every block
post-D2H synchronize
publish each reservation
```

Require a load to perform all H2D copies and a final synchronize before its `TransferResult` becomes visible. Inject an exception on the middle component copy and assert a post-failure synchronize precedes aborting every reservation. Inject a failure on the second publish and assert the first published entry is looked up and evicted while all unpublished reservations are aborted.

- [ ] **Step 5: Implement transfer, publish, and failure cleanup**

Override `_transfer()` only. Call the inherited `_validate_gpu_spec()`, require the device-block count to equal `len(shared_spec.transfers)`, validate every anchor/slot before the first copy, and route each transfer through `copy_kv_page_pair()`. Store mode requires every item to carry a reservation; load mode requires none. Track `published_keys` and `unpublished_reservations` so the exception path can synchronize, evict already-published entries, abort the remainder, and re-raise for inherited `_run()` to report `success=False`.

Set `_bytes_per_block` to the sum of every K and V physical `page_size_bytes`; return `len(transfers) * _bytes_per_block` so upstream transfer metrics directly provide duration and throughput inputs.

- [ ] **Step 6: Run tests, pre-commit, and commit**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_worker_dispatch.py tests/kv_offload/test_worker_dispatch.py -q
SKIP=markdownlint pre-commit run --files spyre_inference/v1/kv_offload/shared_worker.py tests/kv_offload/test_shared_worker_dispatch.py spyre_inference/v1/kv_offload/worker.py spyre_inference/v1/worker/spyre_kv_offload.py
git add spyre_inference/v1/kv_offload/shared_worker.py tests/kv_offload/test_shared_worker_dispatch.py spyre_inference/v1/kv_offload/worker.py spyre_inference/v1/worker/spyre_kv_offload.py
git commit -s -m "Add shared KV offload worker"
```

### Task 5: Build and Lazily Register `SpyreSharedOffloadingSpec`

**Files:**

- Create: `spyre_inference/v1/kv_offload/shared_spec.py`
- Modify: `spyre_inference/__init__.py:92`
- Create: `tests/kv_offload/test_shared_spec.py`
- Modify: `tests/kv_offload/test_spec.py`

**Interfaces:**

- Consumes: M1 construction hooks, `SpyreSharedOffloadingManager`, `SpyreSharedOffloadingWorker`, `allocate_family_slots`, page signatures, and full `VllmConfig` from the connector.
- Produces: `SpyreSharedOffloadingSpec(SpyreOffloadingSpec)` resolvable by `spec_name="SpyreSharedOffloadingSpec"` without `spec_module_path`.

- [ ] **Step 1: Write failing configuration and factory tests**

Construct the spec with `shared_metadata_name="run-42"`, families `("pool-a", "pool-b", "pool-c")`, and eight logical blocks. Assert slot counts `(3, 3, 2)`. Reject a blank metadata name, an empty family list, duplicate family names, fewer blocks than families, `world_size != 1`, more than one KV group, and a non-float16 dtype.

Verify lazy resolution:

```python
def test_shared_spec_factory_registration_is_lazy():
    sys.modules.pop("spyre_inference.v1.kv_offload.shared_spec", None)
    import spyre_inference

    assert "spyre_inference.v1.kv_offload.shared_spec" not in sys.modules
    cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "SpyreSharedOffloadingSpec"})
    assert cls.__name__ == "SpyreSharedOffloadingSpec"
```

Run a subprocess with a stub `torch_spyre._C` that lacks `SharedMetadata`; importing `spyre_inference` and resolving the M1 spec must succeed. Selecting M2 and first invoking its runtime path must fail with a message containing `SharedMetadata` and `torch-spyre M2`.

- [ ] **Step 2: Run spec tests and verify they fail**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_spec.py tests/kv_offload/test_spec.py -q
```

Expected: FAIL because the M2 class and registration do not exist.

- [ ] **Step 3: Implement M2 validation and factory overrides**

The constructor calls `super().__init__`, then creates its immutable family allocation and directory-capacity parameters. `bind_vllm_config()` requires `distributed_executor_backend == "uni"` after vLLM resolution and stores the config for compatibility construction. `_create_manager()` lazily imports the manager module; `_create_worker()` requires the bound full config, computes the compatibility digest, and lazily imports the worker module.

Build the digest from canonical JSON with sorted keys and compact separators, then SHA-256 it to 32 bytes. Include:

```python
payload = {
    "format": COMPATIBILITY_FORMAT_VERSION,
    "model": vllm_config.model_config.model,
    "revision": vllm_config.model_config.revision,
    "hash_algorithm": vllm_config.cache_config.prefix_caching_hash_algo,
    "hash_seed": os.environ.get("PYTHONHASHSEED"),
    "tokens_per_hash": self.tokens_per_hash,
    "tokens_per_block": self.tokens_per_block,
    "dtype": self.config.model.dtype,
    "tp_size": 1,
    "components": component_signatures,
}
```

Require `PYTHONHASHSEED` to be explicitly set for M2 instead of allowing two instances to produce unrelated first-block hashes. Each component signature contains cache index, `"k"` or `"v"`, layout kind/version, block size, local KV heads, head size, and physical page bytes.

- [ ] **Step 4: Register the spec by strings only**

At package initialization, add:

```python
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

OffloadingSpecFactory.register_spec(
    "SpyreSharedOffloadingSpec",
    "spyre_inference.v1.kv_offload.shared_spec",
    "SpyreSharedOffloadingSpec",
)
```

Do not import `shared_spec`, `shared_manager`, `shared_worker`, or `torch_spyre._C` from `spyre_inference/__init__.py`.

- [ ] **Step 5: Pin compatibility failures and M1 preservation**

Tests must prove every field in the payload changes the digest, a same-configuration digest is stable across processes, mismatched pool geometry/compatibility fails at registration, the M1 factory path still builds `CPUOffloadingManager`, and M1 pool prefixes remain engine/rank private.

- [ ] **Step 6: Run tests, pre-commit, and commit**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_spec.py tests/kv_offload/test_spec.py tests/kv_offload/test_canonicalize_paged.py tests/kv_offload/test_worker_dispatch.py -q
SKIP=markdownlint pre-commit run --files spyre_inference/__init__.py spyre_inference/v1/kv_offload/shared_spec.py tests/kv_offload/test_shared_spec.py tests/kv_offload/test_spec.py
git add spyre_inference/__init__.py spyre_inference/v1/kv_offload/shared_spec.py tests/kv_offload/test_shared_spec.py tests/kv_offload/test_spec.py
git commit -s -m "Register shared Spyre offloading spec"
```

### Task 6: Prove Shared Misses Fall Back to Recompute

**Files:**

- Create: `tests/kv_offload/test_connector_miss_recompute.py`
- Modify only if the test exposes an integration defect: `spyre_inference/v1/kv_offload/shared_manager.py`

**Interfaces:**

- Consumes: `SpyreSharedOffloadingManager.lookup()` and upstream `OffloadingConnectorScheduler._maximal_prefix_lookup()` / `update_state_after_alloc()` behavior.
- Produces: a connector-level regression proving an M2 miss schedules zero externally loaded tokens and no H2D job.

- [ ] **Step 1: Write the connector-level miss test**

Use a real `SpyreSharedOffloadingManager` with a fake empty directory. Construct `OffloadingConnectorScheduler` through `object.__new__` with only the fields exercised by `_maximal_prefix_lookup`; because the first result is a miss, no event field is read:

```python
scheduler = object.__new__(OffloadingConnectorScheduler)
scheduler.manager = manager
matched = scheduler._maximal_prefix_lookup(
    [OFFLOAD_KEY], ReqContext("request-1"), MagicMock(), MagicMock(), 0
)
assert matched == 0
assert manager.directory.pin_calls == []
```

Then invoke `update_state_after_alloc(request, blocks, num_external_tokens=0)` on an object whose `manager.prepare_load` raises if called. Assert it returns without creating a load job. This targets the connector decision, not lower-level race correctness.

- [ ] **Step 2: Verify the test fails for any accidental hit translation**

Temporarily configure the fake directory lookup to return a published entry and confirm the first assertion changes from `0` to `1`; restore the empty-directory setup before proceeding.

- [ ] **Step 3: Run the final miss test**

```bash
uv run --no-sync pytest tests/kv_offload/test_connector_miss_recompute.py tests/kv_offload/test_shared_manager.py -q
```

Expected: PASS, with no worker or Spyre device needed.

- [ ] **Step 4: Run pre-commit and commit**

```bash
SKIP=markdownlint pre-commit run --files tests/kv_offload/test_connector_miss_recompute.py spyre_inference/v1/kv_offload/shared_manager.py
git add tests/kv_offload/test_connector_miss_recompute.py spyre_inference/v1/kv_offload/shared_manager.py
git commit -s -m "Test shared offload miss recomputation"
```

### Task 7: Run the Single-Chunk Gate and Shared-Pool Hardware Round Trip

**Files:**

- Create: `tests/kv_offload/test_shared_pool_round_trip.py`
- Modify if the real integration exposes a shared-worker contract defect: `spyre_inference/v1/kv_offload/shared_worker.py`
- Modify its mock-safe regression if needed: `tests/kv_offload/test_shared_worker_dispatch.py`
- Modify only if the gate proves it necessary: `/home/yzhu/dt-inductor/flex/include/flex/runtime_stream/runtime_stream.hpp`
- Modify only if the gate proves it necessary: `/home/yzhu/dt-inductor/flex/src/runtime_stream/runtime_stream.cpp`
- Modify only if the gate proves it necessary: `/home/yzhu/dt-inductor/flex/tests/runtime_stream/stream/runtime_stream_copy_raw_test.cpp`
- Modify only if the gate proves it necessary: `/home/yzhu/dt-inductor/torch-spyre/.worktrees/kvc-offload-m2/tests/distributed/test_kv_offload_distributed.py`

**Interfaces:**

- Consumes: real Spyre `get_composite_address`, shared directory bindings, shared worker, and the existing bit-exact helpers in `tests/kv_offload/hw_helpers.py`.
- Produces: a bit-exact connector-level shared-slot round trip, or hard evidence that M2-F3 must be implemented first.

- [ ] **Step 1: Add and run the real-allocation chunk gate**

Allocate both token-major and head-major KV caches through their production `allocate_pages()`. For every K and V tensor, record `total_size`, `num_chunks`, and `[(domain_id, size)]`, and require `num_chunks == 1`:

```python
address = get_composite_address(pages)
assert address.num_chunks == 1, (
    f"M2-F3 required: {layout_kind} {role} allocation has "
    f"{address.num_chunks} chunks: {address.chunks()}"
)
```

Run serially:

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_pool_round_trip.py::test_real_kv_allocations_are_single_chunk -q -s
```

Expected: PASS for every production KV allocation used by the test.

- [ ] **Step 2: If and only if Step 1 reports multiple chunks, complete M2-F3 on the lower-layer branches**

Stop spyre-inference work. In Flex `kvc-offload-m2`, change `RuntimeStream::copyRawImpl` to iterate the requested `CompositeAddress` range across chunks, issuing each DMA against the next contiguous host sub-offset, and validate `sum(selected_chunk_bytes) == Range.length` before issuing the first copy. Replace the existing `MultiChunkDeviceAddressThrows` test with D2H/H2D round trips for two unequal chunks plus the existing single-chunk regressions and host-capacity guards.

In torch-spyre `kvc-offload-m2`, preserve both public copy signatures. Add a spawned two-process test in `tests/distributed/test_kv_offload_distributed.py`: process A copies a multi-chunk tensor into one shared slot and publishes all `ChunkDescriptorEntry(domain_id, size)` records; process B looks up, pins, allocates the same shape/dtype, reloads, and asserts raw bytes. Rebuild using the repository scripts, run the focused Flex and torch-spyre tests serially, commit each repository with `-s`, reinstall the local torch-spyre wheel, and rerun Step 1. Do not continue until the gate passes.

- [ ] **Step 3: Write the shared-worker round-trip test**

Create a unique metadata name and two-family configuration. Store a known nonzero device block through `SpyreSharedOffloadingWorker`, drain a successful result, attach a second shared worker to the same names, overwrite its destination device block, lookup and pin through a second manager, reload into that block, drain completion, release the pin, and compare every K/V component with `_assert_bit_exact`. Repeat for token-major and head-major layouts. Assert the stored and loaded `TransferResult.transfer_size` equals the complete logical block size and both times are positive.

Flex's published chunk descriptor describes the claimed `c0.k` anchor slot,
not the aggregate bytes in its sibling component pools. The worker must publish
that anchor descriptor only after all K/V component copies synchronize; add a
mock-safe regression that rejects an aggregate descriptor larger than the
anchor slot.

- [ ] **Step 4: Run the round trip serially**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_pool_round_trip.py -q -s
```

Expected: PASS on real hardware; CPU-only and `FLEX_DEVICE=MOCK*` environments skip rather than claim byte fidelity.

- [ ] **Step 5: Run pre-commit and commit the spyre-inference test**

```bash
SKIP=markdownlint pre-commit run --files tests/kv_offload/test_shared_pool_round_trip.py
git add tests/kv_offload/test_shared_pool_round_trip.py
git commit -s -m "Test shared KV pool round trip"
```

### Task 8: Add the Two-Instance Functional Acceptance and A/B Timing

**Files:**

- Create: `tests/kv_offload/test_cross_instance.py`
- Create after a successful run: `docs/superpowers/results/2026-09-29-spyre-shared-kv-offload.md`

**Interfaces:**

- Consumes: `vllm serve`, `SpyreOffloadingConnector`, lazy `SpyreSharedOffloadingSpec`, two distinct Spyre cards, Prometheus offload counters, and the OpenAI-compatible completions endpoint.
- Produces: an opt-in two-process test plus recorded A self-reload and B peer-reload duration/throughput.

- [ ] **Step 1: Write the opt-in test harness and lifecycle guards**

Gate collection with `RUN_SPYRE_SHARED_KV_E2E=1`, require at least two real devices, and mark the test `uses_subprocess`. Start two server subprocesses with unique ports and log files. Use these invariant settings for both:

```python
common = {
    "model": "ibm-ai-platform/micro-g3.3-8b-instruct-1b",
    "enforce_eager": True,
    "enable_prefix_caching": False,
    "tensor_parallel_size": 1,
    "distributed_executor_backend": "uni",
}

extra = {
    "spec_name": "SpyreSharedOffloadingSpec",
    "shared_metadata_name": metadata_name,
    "shared_pool_families": [f"{metadata_name}.a", f"{metadata_name}.b"],
    "cpu_bytes_to_use": 512 * 1024 * 1024,
}
```

Pass `kv_connector="SpyreOffloadingConnector"`, `kv_role="kv_both"`, and `kv_connector_module_path="spyre_inference.v1.kv_offload.connector"`. Set `PYTHONHASHSEED=0` for both; set `SPYRE_DEVICES=0` for A and `SPYRE_DEVICES=1` for B. Poll `/health` with a bounded timeout. On every exit path, terminate both servers, wait for their child workers, then retire all named component pools and unlink the metadata directory.

The generated server command contains these explicit flags in addition to the
JSON connector configuration:

```text
--enforce-eager --no-enable-prefix-caching --tensor-parallel-size 1
--distributed-executor-backend uni
```

- [ ] **Step 2: Add metric snapshot helpers**

Parse these cumulative Prometheus counters from each server's `/metrics` endpoint:

```text
vllm:kv_offload_load_bytes_total
vllm:kv_offload_load_time_total
vllm:kv_offload_store_bytes_total
```

For each measured request, snapshot before and after, compute deltas, require positive load bytes and time, and report `bytes / seconds`. Poll A's store-byte counter after its first request so B is never released before publication is observable.

- [ ] **Step 3: Implement the functional sequence with prefix caching disabled**

Use one prompt containing at least two complete 128-token blocks and fixed greedy generation. Execute:

1. A baseline request computes and publishes; record returned token IDs and text.
2. Wait until A reports a positive store-byte delta and the original request has completed, releasing its device KV blocks.
3. Snapshot A metrics, send the same prompt to A, and require a positive host-to-device load delta; record A self-reload seconds and bytes/second.
4. Snapshot B metrics, send the same prompt as B's first request, and require a positive host-to-device load delta; record B peer-reload seconds and bytes/second.
5. Require A-self and B-peer token IDs and text to be byte-identical to the baseline.
6. Require server logs to contain shared host-tier hit/load records and no disk-tier load; prefix caching remains explicitly false in the emitted engine configuration.

The assertions are functional. Do not compare A against B, M2 against M1, or either result against a performance threshold.

- [ ] **Step 4: Verify the test catches an absent peer path**

Run once with B assigned a different metadata name and confirm B's positive load-delta assertion fails while generation still succeeds by recomputation. Restore the common name before the acceptance run.

- [ ] **Step 5: Run the two-instance acceptance**

```bash
RUN_SPYRE_SHARED_KV_E2E=1 uv run --no-sync pytest tests/kv_offload/test_cross_instance.py -q -s
```

Expected: one cross-instance test passes; A and B each report positive shared-pool load bytes/time, B loads A's block on its first request, and all deterministic outputs match.

- [ ] **Step 6: Record the actual functional evidence and timing**

Create the result document with the tested spyre-inference, torch-spyre, and Flex commit hashes; model; device identifiers; exact command; prefix-caching setting; prompt and shared block bytes; A self-reload seconds and bytes/second; B peer-reload seconds and bytes/second; and pass/fail evidence for peer hit and output identity. State explicitly that no M1 comparison or performance threshold was run.

- [ ] **Step 7: Run pre-commit and commit the acceptance artifacts**

```bash
SKIP=markdownlint pre-commit run --files tests/kv_offload/test_cross_instance.py docs/superpowers/results/2026-09-29-spyre-shared-kv-offload.md
git add tests/kv_offload/test_cross_instance.py docs/superpowers/results/2026-09-29-spyre-shared-kv-offload.md
git commit -s -m "Test cross-instance shared KV reload"
```

### Task 9: Run the Full M1/M2 Regression Gate

**Files:**

- Verify: all changed files on `kvc-offload-m2`

**Interfaces:**

- Consumes: every deliverable from Tasks 1-8.
- Produces: final evidence that M2 works and M1 remains unchanged.

- [ ] **Step 1: Run all mock-safe KV-offload tests**

```bash
uv run --no-sync pytest tests/kv_offload/test_spec.py tests/kv_offload/test_canonicalize_paged.py tests/kv_offload/test_worker_dispatch.py tests/kv_offload/test_shared_types.py tests/kv_offload/test_shared_manager.py tests/kv_offload/test_shared_worker_dispatch.py tests/kv_offload/test_shared_spec.py tests/kv_offload/test_connector_miss_recompute.py -m "not upstream" -q
```

Expected: all pass without importing the M2 runtime during ordinary plugin import.

- [ ] **Step 2: Run real-hardware tests serially**

```bash
uv run --no-sync pytest tests/kv_offload/test_spyre_kv_offload_hw.py tests/kv_offload/test_worker_hw.py tests/kv_offload/test_shared_pool_round_trip.py -q -s
```

Expected: M1 and M2 page round trips pass byte-exactly on a real card.

- [ ] **Step 3: Repeat the two-instance acceptance after the regression suite**

```bash
RUN_SPYRE_SHARED_KV_E2E=1 uv run --no-sync pytest tests/kv_offload/test_cross_instance.py -q -s
```

Expected: deterministic output identity and positive A/B shared-load measurements remain reproducible.

- [ ] **Step 4: Run formatting and repository checks**

```bash
bash format.sh
SKIP=markdownlint pre-commit run --all-files
uv run --no-sync ty
git diff --check
```

Treat the known unresolved local torch-spyre import failure as an environment blocker only if it is unchanged from baseline; all new-file diagnostics must be fixed.

- [ ] **Step 5: Audit branch isolation and lower-layer pins**

```bash
git status --short --branch
git log --oneline --decorate 86f56ed..HEAD
git -C /home/yzhu/dt-inductor/flex branch --show-current
git -C /home/yzhu/dt-inductor/torch-spyre/.worktrees/kvc-offload-m2 branch --show-current
```

Expected: spyre-inference is on `kvc-offload-m2`; any conditional lower-layer commits are on their own `kvc-offload-m2` branches; no files from the dirty `kvc-offload-poc` worktree appear in the diff.

- [ ] **Step 6: Request final code review before integration**

Use the `requesting-code-review` skill against the complete `86f56ed..HEAD` diff. Resolve findings with focused tests, rerun the affected gates, and leave the branch unpushed until the user explicitly authorizes a GitHub write.
