# Spyre One-Pool Shared KV Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace M2's per-component pool family with one shared data pool whose independently allocated fixed-size slots each hold one physical K or V page, while preserving cross-instance A -> A -> B reload behavior and the visual performance demo.

**Architecture:** `SpyreSharedOffloadingSpec` resolves one logical `pool_name`, builds a deterministic ordered component manifest, and gives the worker enough information to derive one-pool slot geometry from the physical KV allocations. The scheduler-side manager lazily attaches that registered pool, sizes its upstream logical-block policy as `pool.slot_count // component_count`, and represents every logical vLLM block as a complete set of component-qualified page entries with explicit, potentially non-contiguous slot locations. The worker copies each component page through the existing M1 `copy_kv_page_raw(cache, block_id, pool, slot_id, ...)` path; no Flex or torch-spyre production API change is required.

**Tech Stack:** Python 3.12, vLLM 0.28 offloading interfaces, PyTorch/torch-spyre, Flex `SharedMetadata` and `SharedHostPool`, pytest, OpenAI-compatible streaming completions, Prometheus metrics, POSIX shared memory.

**Spec:** `docs/superpowers/specs/2026-09-29-spyre-shared-kv-offload-design.md`

## Global Constraints

- Implement only in `/home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2` on local branch `kvc-offload-m2`; preserve all existing modified and untracked demo work.
- Follow `torch-spyre-docs` commit `114011e842014b8ce6eb2fde9e3f83fd6abb9a81`, `docs/RFCs/SharedKvPoolRFC.md`, and `docs/RFCs/RawCopyKvOffloadRFC.md` literally at the low-level pool/copy seam.
- Configure exactly one independent `SharedHostPool` data pool for this milestone; one `SharedMetadata` directory indexes every component page in that pool.
- One pool slot holds one physical page `component_tensor[block_id]`; one page never spans slots, while one logical vLLM block occupies one independently claimed slot per component.
- Set `slot_bytes` to the host-page-aligned maximum physical component-page size; set `pool_slot_count = logical_block_capacity * component_count` and fail if the budget cannot hold one complete logical block.
- Never derive component locations by adjacency. Every `(pool_id, slot_id)` returned by Flex is carried explicitly, and component slots may be fragmented in any order.
- Define a logical cache hit only when every required component page is valid and read-pinned. A missing, reserved, stale, foreign-pool, or unpinnable page releases earlier pins and returns `LookupResult.MISS`.
- Keep the existing M1 device-range contract unchanged. Do not add a pool-relative offset, packed aggregate slot, contiguous extent allocator, Flex production change, or torch-spyre production change.
- Preserve M1 restrictions: float16, one KV-cache group, `blocks_per_chunk == 1`, and supported single-chunk physical allocations. M2 additionally requires `world_size == 1`, TP1, and the `uni` executor.
- Preserve M1 selection, engine/rank-private M1 pool naming, worker behavior, metrics, lazy M2 imports, and miss-to-recompute behavior.
- Keep prefix caching disabled in the cross-instance and manual acceptance runs. Only the deliberate two-server topology may run two Spyre processes concurrently, and each server must use a distinct device.
- Run every `uv` command with `--no-sync` so the local torch-spyre installation is not replaced.
- Cleanup must refuse to unlink live objects, remove every current or legacy pool registered under the configured demo namespace after the servers exit, and leave unrelated shared memory untouched.
- Do not commit, sign, push, or rewrite branch history during Tasks 1-6. Run all validation first; create the signed commit only in Task 7 after review findings are resolved.

## File Structure

- Modify `spyre_inference/v1/kv_offload/shared_types.py`: component-manifest, page-level key, geometry, location, reservation, block-transfer, and load/store-spec contracts.
- Modify `spyre_inference/v1/kv_offload/shared_spec.py`: one-pool name resolution, ordered component manifest, compatibility digest, metadata capacity, and factories.
- Modify `spyre_inference/v1/kv_offload/shared_manager.py`: lazy one-pool attach, logical policy sizing, all-page lookup/pinning, page claims, rollback, ownership, and eviction.
- Modify `spyre_inference/v1/kv_offload/shared_worker.py`: one-pool registration and explicit per-component page DMA/publish routing.
- Preserve `spyre_inference/v1/kv_offload/shared_runtime.py`: existing lazy torch-spyre binding boundary.
- Preserve `spyre_inference/v1/kv_offload/connector.py`: existing physical-cache binding and M1/M2 connector behavior.
- Modify `tests/kv_offload/test_shared_types.py`: deterministic page keys, one-pool geometry, and transfer-shape tests.
- Modify `tests/kv_offload/test_shared_spec.py`: one-pool configuration, manifest, compatibility, lazy registration, and M1 preservation tests.
- Modify `tests/kv_offload/test_shared_manager.py`: fragmented locations, all-page hit/miss, claim rollback, pin lifetime, ownership, and policy tests.
- Modify `tests/kv_offload/test_shared_worker_dispatch.py`: one-pool registration, explicit page routing, validation, synchronization, and rollback tests.
- Modify `tests/kv_offload/test_shared_pool_round_trip.py`: bit-exact real-device round trip through one pool and fragmented slots.
- Modify `tests/kv_offload/test_cross_instance.py`: two-server one-pool acceptance and shared-memory topology assertions.
- Modify `scripts/start_shared_kv_instance_a.sh` and `scripts/start_shared_kv_instance_b.sh`: replace `shared_pool_families` with one `pool_name`.
- Modify `scripts/cleanup_shared_kv_demo.py`: clean every registered current/legacy demo-owned pool, its backing/control objects, and metadata.
- Modify `tests/kv_offload/test_cleanup_shared_kv_demo.py`: current, legacy, live-owner, unknown-pool, and idempotence cleanup tests.
- Preserve and retest `scripts/warmup_shared_kv_demo.py` and `tests/kv_offload/test_warmup_shared_kv_demo.py`: unrelated junk warmup and visible target/prompt/performance output.
- Modify `scripts/shared_kv_two_instance_demo.py` only where one-pool metrics/topology require it; preserve its visible target, identifier, prompt, TTFT, end-to-end, token-source, KV-byte, and copy-time output.
- Modify `tests/kv_offload/test_shared_kv_two_instance_demo.py`: preserve the visual A/A/B request and metric contracts.
- Modify `docs/superpowers/results/2026-09-30-spyre-shared-kv-manual-demo.md`: one-pool setup, inspection, cleanup, commands, and fresh result evidence.

## Review Focus

- A fragmented free list must yield arbitrary component slot IDs without changing bytes or component routing; Tasks 1, 3, 4, and 5 pin explicit non-contiguous locations in unit and hardware tests.
- A partial page set or a pin failure after earlier successful pins must be a normal miss with every acquired pin released; Task 3 checks missing, reserved, failed-pin, and foreign-pool cases at each component position.
- A mid-claim, D2H, or publish failure must abort unpublished reservations, evict only pages published by that attempt, preserve pre-existing peer pages, and roll back logical policy admission; Tasks 3 and 4 inject each failure.
- Same-name attachment with a different model, page-key version, component order, layout, page size, slot size, or slot count must fail before DMA; Tasks 2 and 4 vary every compatibility and geometry input.
- Cleanup must discover and remove all registered objects belonging to the current or historical demo naming schemes while refusing a live owner or an unrecognized registered pool; Task 6 tests all three outcomes.

---

### Task 1: Replace Family-Level Contracts with Page-Level Contracts

**Files:**

- Modify: `spyre_inference/v1/kv_offload/shared_types.py`
- Modify: `tests/kv_offload/test_shared_types.py`

**Interfaces:**

- Consumes: `vllm.v1.kv_offload.base.OffloadKey` and `LoadStoreSpec`.
- Produces: `SharedComponentDescriptor`, `SharedPoolGeometry`, `SharedPageLocation`, `SharedPageTransfer`, `SharedBlockTransfer`, `SharedLoadStoreSpec`, `compute_shared_pool_geometry(...)`, and `shared_page_hash(key, component_id)`.

- [ ] **Step 1: Replace family-allocation tests with deterministic page-key tests**

Use the complete `OffloadKey` plus a four-byte unsigned component ID and pin the algorithm with literal values:

```python
def test_shared_page_hash_covers_key_and_component():
    key = make_offload_key(bytes.fromhex("11" * 32), 0)
    assert shared_page_hash(key, 0) == 0xFD8C6177AECACCE2
    assert shared_page_hash(key, 1) == 0x962C86167AD9E163
    assert shared_page_hash(key, 7) == 0xC340A847F6514AA5
    assert shared_page_hash(
        make_offload_key(bytes.fromhex("11" * 32), 1), 0
    ) != shared_page_hash(key, 0)
```

Reject `component_id < 0` and `component_id > 0xFFFFFFFF` so the byte encoding cannot wrap or alias.

- [ ] **Step 2: Add pure geometry tests, including the manual 512-MiB case**

```python
def test_manual_pool_geometry_is_one_512_mib_pool():
    geometry = compute_shared_pool_geometry(
        cpu_bytes_to_use=536_870_912,
        page_bytes=(262_144,) * 8,
        alignment=4096,
    )
    assert geometry == SharedPoolGeometry(
        component_count=8,
        slot_bytes=262_144,
        logical_block_capacity=256,
        slot_count=2048,
        actual_pool_bytes=536_870_912,
    )


def test_geometry_uses_aligned_largest_page():
    geometry = compute_shared_pool_geometry(16_384, (1000, 3000), 4096)
    assert geometry.slot_bytes == 4096
    assert geometry.logical_block_capacity == 2
    assert geometry.slot_count == 4
```

Also reject an empty page list, non-positive page sizes/alignment/budget, a native `size_t` overflow (`slot_count > sys.maxsize // slot_bytes`), and a budget smaller than `component_count * slot_bytes`.

- [ ] **Step 3: Add explicit fragmented-transfer tests**

Construct one logical block with component slots `(7, 2, 11, 5)` and assert the load/store spec retains both component order and exact locations. Assert load pages all have `reservation is None`, store pages all have reservations, duplicate component IDs fail, empty per-block page sets fail, and a single spec cannot mix load and store blocks.

```python
pages = tuple(
    SharedPageTransfer(i, SharedPageLocation(pool_id=3, slot_id=slot))
    for i, slot in enumerate((7, 2, 11, 5))
)
spec = SharedLoadStoreSpec([SharedBlockTransfer(OFFLOAD_KEY, pages)])
assert [page.location.slot_id for page in spec.transfers[0].pages] == [7, 2, 11, 5]
```

- [ ] **Step 4: Run the new contract tests and observe the old API failure**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_types.py -q
```

Expected before implementation: collection or assertion failures naming `SharedPoolFamily`, `allocate_family_slots`, or missing page-level types.

- [ ] **Step 5: Implement the page-key and geometry functions**

```python
PAGE_KEY_ALGORITHM = "blake2b-64/spyre-m2-page/v1"


def shared_page_hash(key: OffloadKey, component_id: int) -> int:
    if not 0 <= component_id <= 0xFFFFFFFF:
        raise ValueError("component_id must fit in four unsigned bytes")
    digest = hashlib.blake2b(
        bytes(key) + component_id.to_bytes(4, "big"),
        digest_size=8,
        person=b"spyre-m2-page",
    ).digest()
    return int.from_bytes(digest, "big")


def compute_shared_pool_geometry(
    cpu_bytes_to_use: int,
    page_bytes: Sequence[int],
    alignment: int,
) -> SharedPoolGeometry:
    sizes = tuple(page_bytes)
    if cpu_bytes_to_use <= 0 or alignment <= 0 or not sizes or min(sizes) <= 0:
        raise ValueError("shared pool geometry requires positive inputs")
    slot_bytes = ((max(sizes) + alignment - 1) // alignment) * alignment
    component_count = len(sizes)
    logical_capacity = cpu_bytes_to_use // (component_count * slot_bytes)
    if logical_capacity == 0:
        raise ValueError("cpu_bytes_to_use cannot hold one complete KV block")
    slot_count = logical_capacity * component_count
    return SharedPoolGeometry(
        component_count,
        slot_bytes,
        logical_capacity,
        slot_count,
        slot_count * slot_bytes,
    )
```

Use frozen dataclasses. Define `SharedComponentDescriptor` with `component_id`, `cache_index`, `role`, `layout_kind`, `layout_version`, `block_size`, `local_kv_heads`, `head_size`, and `page_bytes`. `SharedBlockTransfer.pages` is a tuple of `SharedPageTransfer`; no field represents an anchor, sibling, family, base slot, or pool-relative offset.

- [ ] **Step 6: Run the focused tests**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_types.py -q
```

Expected: all page-key, geometry, explicit-location, mode, validation, and lazy-runtime tests pass without initializing a Spyre device.

### Task 2: Resolve One Pool and Build the Physical Component Manifest

**Files:**

- Modify: `spyre_inference/v1/kv_offload/shared_spec.py`
- Modify: `tests/kv_offload/test_shared_spec.py`

**Interfaces:**

- Consumes: `SpyrePhysicalCaches`, `page_signature`, `SharedComponentDescriptor`, `SharedPoolGeometry`, `compute_shared_pool_geometry`, and `PAGE_KEY_ALGORITHM`.
- Produces: `component_manifest(physical)`, `SpyreSharedOffloadingSpec.pool_name`, `component_count`, `max_pool_slots`, one-pool manager arguments, and one-pool worker arguments.

- [ ] **Step 1: Write one-pool configuration precedence tests**

Replace `shared_pool_families` fixtures with `pool_name="run-42.data"`. Assert precedence and validation exactly:

```python
def test_explicit_pool_name_wins_over_environment(monkeypatch):
    monkeypatch.setenv("SPYRE_KV_POOL_NAME", "from-env")
    spec = SpyreSharedOffloadingSpec(_config(extra={"pool_name": "explicit"}))
    assert spec.pool_name == "explicit"


def test_environment_pool_name_is_the_fallback(monkeypatch):
    monkeypatch.setenv("SPYRE_KV_POOL_NAME", "from-env")
    spec = SpyreSharedOffloadingSpec(_config_without_pool_name())
    assert spec.pool_name == "from-env"


def test_generated_pool_name_is_private_without_shared_configuration(monkeypatch):
    monkeypatch.delenv("SPYRE_KV_POOL_NAME", raising=False)
    spec = SpyreSharedOffloadingSpec(_config_without_pool_name())
    assert spec.pool_name == "spyre_kv_eng0_r0.shared"
```

An explicitly present empty/invalid `pool_name`, an empty/invalid environment value, or the removed `shared_pool_families` key must raise a migration-focused `ValueError`; it must not silently fall through.

- [ ] **Step 2: Add stable component-manifest and compatibility tests**

For two physical caches, require the manifest order `c0.k, c0.v, c1.k, c1.v` with component IDs `0, 1, 2, 3`. The compatibility payload must include `format=2`, `page_key_algorithm=PAGE_KEY_ALGORITHM`, model/revision/hash seed/hash algorithm/block sizes/dtype/TP size, and every component's ID, cache index, role, layout kind/version, block size, local KV heads, head size, and physical page bytes.

Mutate each field independently and assert `compatibility_digest()` changes. Preserve the cross-process stable-digest test with its new expected SHA-256 literal calculated from the final payload.

- [ ] **Step 3: Add factory argument and metadata-capacity tests**

With eight components and `self.num_blocks == 256`, require:

```python
assert spec.component_count == 8
assert spec.max_pool_slots == 2048
assert manager_kwargs["pool_name"] == "run-42.data"
assert manager_kwargs["component_count"] == 8
assert manager_kwargs["max_pool_slots"] == 2048
assert worker_kwargs["pool_name"] == "run-42.data"
assert worker_kwargs["geometry"].slot_count == 2048
assert worker_kwargs["manifest"] == manifest
```

Reject a physical manifest whose component count differs from the scheduler-derived `2 * len(group.layer_names)` before creating the pool.

- [ ] **Step 4: Run spec tests and observe failures from family-based configuration**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_spec.py tests/kv_offload/test_spec.py -q
```

Expected before implementation: failures reference `shared_pool_families`, `families`, and the old compatibility payload.

- [ ] **Step 5: Use the immutable component descriptor and implement one-pool naming**

Use the Task 1 descriptor:

```python
@dataclass(frozen=True)
class SharedComponentDescriptor:
    component_id: int
    cache_index: int
    role: Literal["k", "v"]
    layout_kind: str
    layout_version: int
    block_size: int
    local_kv_heads: int
    head_size: int
    page_bytes: int
```

Resolve names in `SpyreSharedOffloadingSpec.__init__` with explicit config, then `SPYRE_KV_POOL_NAME`, then `f"{self._pool_prefix}.shared"`. Validate POSIX logical names as non-empty and containing no slash beyond an optional leading slash. Set:

```python
self.component_count = 2 * len(config.groups[0].layer_names)
self.max_pool_slots = self.num_blocks * self.component_count
self.cpu_bytes_to_use = int(self.extra_config["cpu_bytes_to_use"])
```

- [ ] **Step 6: Derive the manifest and geometry before worker construction**

Walk `physical.caches` in cache order and each pair in K-then-V order. Build the descriptor from `page_signature(cache, layout_kind)`, assign monotonically increasing component IDs, and compute:

```python
geometry = compute_shared_pool_geometry(
    self.cpu_bytes_to_use,
    tuple(component.page_bytes for component in manifest),
    mmap.PAGESIZE,
)
```

Require `geometry.slot_count <= self.max_pool_slots`. Include `PAGE_KEY_ALGORITHM` and the component descriptors in the digest. Bump `COMPATIBILITY_FORMAT_VERSION` from `1` to `2` so old family/anchor objects cannot attach as the new page-key format.

- [ ] **Step 7: Construct the one-pool manager and worker**

Use these exact constructor boundaries:

```python
SpyreSharedOffloadingManager(
    metadata_name=self.metadata_name,
    pool_name=self.pool_name,
    component_count=self.component_count,
    max_pool_slots=self.max_pool_slots,
    cache_policy=self.eviction_policy,
    cache_policy_module_path=self.cache_policy_module_path,
    enable_events=self.kv_events_config.enable_kv_cache_events,
    store_threshold=int(self.extra_config.get("store_threshold", 0)),
    max_tracker_size=int(self.extra_config.get("max_tracker_size", 64_000)),
)

SpyreSharedOffloadingWorker(
    physical=physical,
    metadata_name=self.metadata_name,
    pool_name=self.pool_name,
    geometry=geometry,
    manifest=manifest,
    compatibility_digest=digest,
    max_pool_slots=self.max_pool_slots,
)
```

- [ ] **Step 8: Run spec and M1-preservation tests**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_spec.py tests/kv_offload/test_spec.py tests/kv_offload/test_canonicalize_paged.py tests/kv_offload/test_worker_dispatch.py -q
```

Expected: one-pool configuration and compatibility tests pass; M1 still creates `CPUOffloadingManager`, keeps its private pool prefix, and does not import the shared runtime unless M2 is selected.

### Task 3: Implement All-Page Lookup, Claims, Pins, and Ownership

**Files:**

- Modify: `spyre_inference/v1/kv_offload/shared_manager.py`
- Modify: `tests/kv_offload/test_shared_manager.py`

**Interfaces:**

- Consumes: one registered pool, `shared_page_hash`, page-level transfer types, and upstream `CPUOffloadingManager` policy behavior.
- Produces: `SpyreSharedOffloadingManager` with the standard `OffloadingManager` interface and explicit all-page hit/store semantics.

- [ ] **Step 1: Replace the fake directory's anchor/family model with one pool and per-page entries**

Index fake entries and claim outcomes by page hash, allow claim results to return arbitrary slots, and make pins release-observable. Instantiate the manager with `pool_name="shared.data"`, `component_count=4`, and `max_pool_slots=16`; expose a registered pool with `slot_count=8` so the local logical policy capacity must become two blocks.

- [ ] **Step 2: Write a fragmented complete-hit test**

Publish four page entries for one `OffloadKey` at slots `(7, 2, 6, 1)`. Require four lookup and pin calls, one logical `HIT`, and this exact load spec:

```python
assert [
    (page.component_id, page.location.pool_id, page.location.slot_id)
    for page in manager.prepare_load([OFFLOAD_KEY], ctx).transfers[0].pages
] == [(0, 10, 7), (1, 10, 2), (2, 10, 6), (3, 10, 1)]
```

Every pin remains live through `prepare_load` and is released only by `complete_load`.

- [ ] **Step 3: Write all partial-hit rollback tests**

Parameterize a missing page and a failed pin at component indices `0, 1, 2, 3`. Require `LookupResult.MISS`, no pending block state, and every earlier `PinRecord.released is True`. Add a page whose slot carries another `pool_id`; require a normal miss and no DMA spec. Preserve tests for `prepare_load` releasing scanned-but-unselected block pins and `on_request_finished` releasing pending but not active pins.

- [ ] **Step 4: Write per-page claim and non-contiguous reservation tests**

Return reservations at `(7, 2, 6, 1)` for the four component keys. Require one `SharedBlockTransfer` containing four store pages in component order and no arithmetic relationship among slots. Add a mixed case where components 0 and 2 are existing valid pages while 1 and 3 reserve slots; require the worker spec to contain only components 1 and 3.

- [ ] **Step 5: Write claim rollback and ownership tests**

Inject `ExistingClaim(valid=False)`, `NoSpace`, and `Unavailable` at each component after at least one prior reservation. Reserved/NoSpace outcomes must abort reservations for that logical block, remove its upstream policy admission, and return no store transfer. `Unavailable` must additionally abort all reservations retained earlier in the same batch and raise.

After successful publication of a mixed existing/new page set, require `_owned_entries[OFFLOAD_KEY]` to contain only the entries created by this manager. Local logical eviction and `reset_cache()` must evict every locally owned page, preserve peer-owned pages, release pins before eviction, and abort only still-live local reservations.

- [ ] **Step 6: Run manager tests and observe failures from anchor-only behavior**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_manager.py -q
```

Expected before implementation: failures show one directory lookup/claim and one anchor slot instead of the complete page set.

- [ ] **Step 7: Implement lazy pool attach and lazy local-policy sizing**

Make `SpyreSharedOffloadingManager` implement `OffloadingManager` and compose a `CPUOffloadingManager` after the worker has registered the pool. Attach metadata with the exact common capacity:

```python
SharedMetadataConfig(
    1,
    [],
    SharedMetadataCapacity(1, max_pool_slots, 1),
)
```

Find `pool_name`, require `registered.slot_count % component_count == 0`, require `registered.slot_count <= max_pool_slots`, and construct the local policy with `num_blocks=registered.slot_count // component_count`. Delegate `on_new_request`, `touch`, `take_events`, and `get_stats` to that local manager. This is the only scheduler capacity; do not use the older logical-byte `self.num_blocks` as the shared policy capacity.

- [ ] **Step 8: Implement all-page lookup and pin lifetime**

For `component_id in range(component_count)`, build:

```python
page_key = runtime.CompatibleBlockKey(
    registered.compatibility,
    shared_page_hash(key, component_id),
)
```

Look up, validate the returned pool ref, and pin each page. On any miss or failed validation, clear the acquired page-pin tuple before returning `MISS`. Store complete pending and active page bundles per request. Call the local manager's load lifecycle only for locally admitted logical keys.

Call `local_manager.lookup(key, req_context)` before the directory scan so upstream threshold accounting remains active. Preserve `HIT_PENDING` for an in-flight local store; for an upstream `HIT`, still scan and pin every shared page before reporting a usable hit.

- [ ] **Step 9: Implement claims, publication verification, and rollback**

Claim every page key from the same `registered.pool_ref` in component order. Carry each reservation's exact slot into `SharedPageTransfer` and reject any `ExistingClaim` that names another pool. Existing valid pages are reused and omitted from D2H; existing reserved pages and NoSpace skip that logical store after aborting its new reservations. On successful worker completion, look up all page keys, require a complete valid set, and record ownership only for page keys reserved by this attempt. If that final verification is incomplete, evict every newly published entry found for this attempt, roll back logical admission, and preserve pre-existing peer pages. On failed worker completion, drop pending ownership and call the local manager with `success=False`; the worker already quiesced DMA and rolled back reservations.

- [ ] **Step 10: Run manager, connector-miss, and policy tests**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_manager.py tests/kv_offload/test_connector_miss_recompute.py -q
```

Expected: all-page and rollback tests pass for both LRU and ARC; a partial shared set still schedules zero external tokens and normal recomputation.

### Task 4: Register One Pool and Route Every Physical Page Explicitly

**Files:**

- Modify: `spyre_inference/v1/kv_offload/shared_worker.py`
- Modify: `tests/kv_offload/test_shared_worker_dispatch.py`

**Interfaces:**

- Consumes: `SharedPoolGeometry`, ordered `SharedComponentDescriptor` values, and page-level load/store specs.
- Produces: one `SharedHostPool` registration and page-by-page D2H/H2D through `copy_kv_page_raw`.

- [ ] **Step 1: Replace fake family pools with one registered/resolved pool**

Make the fake runtime record one `SharedDataPoolConfig`. Assert:

```python
assert directory.config == (
    "shared-meta",
    FakeMetadataConfig(1, (), FakeCapacity(1, 16, 1)),
)
assert directory.configs == [
    FakeDataPoolConfig(
        "shared.data",
        "host",
        8,
        4096,
        FakeCompatibilityDescriptor(2, tuple(DIGEST)),
    )
]
```

The worker must reject a resolved pool whose returned slot count or slot bytes differs from `SharedPoolGeometry`, a mismatched compatibility descriptor, a multi-chunk component allocation, a non-divisible allocation, or a runtime page size that differs from its manifest.

- [ ] **Step 2: Write non-contiguous page routing tests**

For one device block and page slots `(7, 2, 6, 1)`, require exactly:

```python
assert copy_events == [
    ("k0", 3, "shared.data", 7, False, True),
    ("v0", 3, "shared.data", 2, False, True),
    ("k1", 3, "shared.data", 6, False, True),
    ("v1", 3, "shared.data", 1, False, True),
]
```

Repeat with `to_device=True`. A load block must contain every component exactly once; a store block may contain a non-empty subset of newly reserved components. Unknown/duplicate component IDs, foreign pool IDs, out-of-range slots, wrong block counts, and wrong reservation mode must fail before any copy.

- [ ] **Step 3: Write synchronization, descriptor, and byte-count tests**

Require one fence before copies and one fence after all copies. After the second fence, publish each reservation with that component's own single-chunk descriptor `(domain_id, page_bytes)`. A four-page load returns `sum(PAGE_BYTES)`; a two-page partial store returns only those two page sizes.

- [ ] **Step 4: Write transfer and publish rollback tests**

Fail on the third copy and require the final sequence to synchronize, abort every unpublished reservation, and leave no entry. Fail on the third publish and require the first two published entries to be evicted, the remaining reservations to be aborted, and pre-existing peer entries to remain. The inherited result queue must report `success=False` and no successful transfer size.

- [ ] **Step 5: Run worker tests and observe family-routing failures**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_worker_dispatch.py -q
```

Expected before implementation: registration creates multiple component pools and every component reuses one anchor slot.

- [ ] **Step 6: Implement physical component binding and one-pool registration**

Build a private `component_id -> (tensor, page_bytes, domain_id)` map from the ordered physical caches. Validate each complete allocation and its one chunk. Create metadata with `(max_chunks=1, max_pools=1, max_slots_per_pool=max_pool_slots, max_compatibilities=1)`, then register exactly:

```python
pool_config = runtime.SharedDataPoolConfig(
    pool_name,
    runtime.SharedPoolKind.HOST,
    geometry.slot_count,
    geometry.slot_bytes,
    runtime.CompatibilityDescriptor(
        COMPATIBILITY_FORMAT_VERSION,
        list(compatibility_digest),
    ),
)
registered = directory.register_or_attach_pool(pool_config)
pool = directory.resolve_pool(registered.pool_ref)
```

Validate `registered.slot_count`, `registered.slot_bytes`, and resolved pool before accepting jobs. Log component count, logical block capacity, slot count, slot bytes, actual bytes, budget padding loss, and pool name.

- [ ] **Step 7: Implement explicit page transfer and rollback**

For each device block and each page transfer, resolve the component record and issue:

```python
runtime.copy_kv_page_raw(
    component.tensor,
    device_block,
    pool,
    page.location.slot_id,
    to_device,
    True,
)
```

Never call `copy_kv_page_pair` from M2 because K and V may occupy unrelated slots. Synchronize before and after the copy batch. Publish only after the second synchronization. On exception, synchronize first, evict entries published by the attempt, abort the still-unpublished reservations, and re-raise for `_run()` to report failure.

- [ ] **Step 8: Run worker and M1 dispatch tests**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_worker_dispatch.py tests/kv_offload/test_worker_dispatch.py tests/kv_offload/test_spyre_kv_offload.py -q
```

Expected: the M2 worker uses one pool and arbitrary slots; M1 retains its existing per-cache K/V pairing and slot mapping.

### Task 5: Prove One-Pool Round Trip and Cross-Instance Reuse

**Files:**

- Modify: `tests/kv_offload/test_shared_pool_round_trip.py`
- Modify: `tests/kv_offload/test_cross_instance.py`

**Interfaces:**

- Consumes: completed one-pool manager/worker, real Spyre allocations, two vLLM instances, and existing offload metrics.
- Produces: byte-exact token-major/head-major page reload and the automated A -> A -> B functional acceptance.

- [ ] **Step 1: Rewrite the hardware round trip for one pool**

Use one pool name, a geometry computed from the real cache, and page-level manager/worker constructors. Before the measured store, claim and publish four dummy keys, record their returned slot IDs, then evict the maximum-ID entry followed by the minimum-ID entry. Because Flex pushes each freed slot onto the free-list head, the next two component claims must return the recorded minimum and maximum IDs, which are non-adjacent. Assert the actual store spec carries those exact locations and is not an arithmetic `base_slot + component_id` sequence.

Store a known pattern from source block 3, attach a second worker to the same pool, load into destination block 6, and compare every K/V page bit-exactly for token-major and head-major layouts. Require both `TransferResult.transfer_size` values to equal the sum of the physical component page sizes.

- [ ] **Step 2: Preserve the existing physical-allocation gate**

Keep `test_real_kv_allocations_are_single_chunk` unchanged in meaning. It must print each layout/role's `total_size`, `num_chunks`, and chunk list, then require `num_chunks == 1`. A failure is an explicit scope violation, not authorization to change Flex or torch-spyre in this milestone.

- [ ] **Step 3: Replace cross-instance families with one pool name**

Change the server command's extra configuration to:

```python
"kv_connector_extra_config": {
    "spec_name": "SpyreSharedOffloadingSpec",
    "shared_metadata_name": metadata_name,
    "pool_name": pool_name,
    "cpu_bytes_to_use": CPU_BYTES,
}
```

Use one unique `pool_name = f"{metadata_name}.data"` for A and B. The isolate-B negative control must change both B's metadata and pool name.

- [ ] **Step 4: Add one-pool topology assertions**

After A and B are ready and after A publishes, inspect the registered directory and `/dev/shm`. Require `pool_count() == 1`, `find_pool(pool_name)` to succeed, and exactly one generated backing plus its `.ctl` object for this metadata version. Require the pool's runtime geometry to equal the logged component count, slot size, slot count, and total bytes. Do not count unrelated `flex_kv_*` objects owned by other tests.

- [ ] **Step 5: Preserve the functional A -> A -> B assertions**

Keep prefix caching disabled. A's first request must have positive store bytes. A's second and B's first measured requests must have positive load bytes/time, identical token IDs, byte-identical text, and server logs showing external offloaded-token hits. Keep the isolate-B negative control: generation succeeds by recomputation but its positive peer-load assertion fails.

- [ ] **Step 6: Run mock-safe integration tests**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_types.py tests/kv_offload/test_shared_spec.py tests/kv_offload/test_shared_manager.py tests/kv_offload/test_shared_worker_dispatch.py tests/kv_offload/test_connector_miss_recompute.py -q
```

Expected: all pass without a physical card.

- [ ] **Step 7: Run the hardware page round trip serially**

```bash
uv run --no-sync pytest tests/kv_offload/test_shared_pool_round_trip.py -q -s
```

Expected on real hardware: single-chunk gate and bit-exact token-major/head-major one-pool round trips pass. A CPU/mock environment must skip the byte-fidelity cases rather than report a hardware pass.

- [ ] **Step 8: Run the two-instance test**

```bash
RUN_SPYRE_SHARED_KV_E2E=1 \
LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync pytest tests/kv_offload/test_cross_instance.py -q -s
```

Expected: A stores, A reloads, B peer-reloads, deterministic output matches, and the clean test namespace contains one metadata object and one data/control pair.

### Task 6: Update Launch, Visual Demo, Namespace Cleanup, and Documentation

**Files:**

- Modify: `scripts/start_shared_kv_instance_a.sh`
- Modify: `scripts/start_shared_kv_instance_b.sh`
- Modify: `scripts/cleanup_shared_kv_demo.py`
- Modify: `tests/kv_offload/test_cleanup_shared_kv_demo.py`
- Verify/modify: `scripts/warmup_shared_kv_demo.py`
- Verify/modify: `tests/kv_offload/test_warmup_shared_kv_demo.py`
- Verify/modify: `scripts/shared_kv_two_instance_demo.py`
- Verify/modify: `tests/kv_offload/test_shared_kv_two_instance_demo.py`
- Modify: `docs/superpowers/results/2026-09-30-spyre-shared-kv-manual-demo.md`

**Interfaces:**

- Consumes: explicit demo namespace `spyre_manual_4096`, one pool `spyre_manual_4096.data`, legacy M2 names, streaming TTFT, and Prometheus request deltas.
- Produces: repeatable three-terminal visual demo, safe complete cleanup, inspectable one-pool topology, and recorded evidence.

- [ ] **Step 1: Write cleanup tests for every demo-owned registered pool**

Cover two layouts:

```python
CURRENT_NAMES = ("demo.data",)
LEGACY_NAMES = tuple(
    f"demo.{family}.c{cache_index}.{role}"
    for family in ("a", "b")
    for cache_index in range(4)
    for role in ("k", "v")
)
```

For the current directory, assert the data backing, `.ctl`, and metadata are removed. For a legacy directory, register all 16 historical component-pool names and assert all 16 backing/control pairs and metadata are removed. Add an explicit extra demo-owned `--pool-name` and require it is removed too. Preserve tests that refuse a matching live vLLM process or mapped metadata object, refuse a directory containing an unrecognized registered pool, restore temporary mock-runtime environment variables, and return success when metadata is absent.

- [ ] **Step 2: Run cleanup tests and observe failures from the fixed 16-pool assumption**

```bash
uv run --no-sync pytest tests/kv_offload/test_cleanup_shared_kv_demo.py -q
```

Expected before implementation: the current one-pool case is not discoverable and the old CLI requires family/cache geometry.

- [ ] **Step 3: Implement current and legacy namespace discovery**

Replace `pool_families`, `cache_count`, and `total_host_blocks` as the primary interface with:

```python
def cleanup_shared_kv_demo(
    *,
    metadata_name: str = "spyre_manual_4096",
    pool_names: tuple[str, ...] | None = None,
    component_count: int = 8,
    max_pool_slots: int = 2048,
    legacy_cache_count: int = 4,
    legacy_total_host_blocks: int = 256,
    proc_root: Path = Path("/proc"),
    shm_root: Path = Path("/dev/shm"),
    runtime: Any | None = None,
) -> list[str]:
```

Default candidates are the current `f"{metadata_name}.data"` plus every historical `f"{metadata_name}.{a|b}.c{0..legacy_cache_count-1}.{k|v}"`. Repeated `--pool-name` values extend that owned candidate set. Try the exact current metadata capacity `(1, max_pool_slots, 1)` with `max_chunks=1`; if attachment reports a configuration mismatch, try the exact legacy capacity `(2 * legacy_cache_count, ceil(legacy_total_host_blocks / 2), 1)` with `max_chunks=2 * legacy_cache_count`.

Find every candidate registered in the attached directory and require `metadata.pool_count() == len(found)` before changing anything. Capture each generated backing name from its `PoolRef`, unlink every backing and `.ctl`, unlink metadata, and verify all paths are absent. Do not call `retire_pool`: retirement intentionally rejects a stale `RESERVED` slot, while this operator-only cleanup runs only after the live-owner guard and must recover interrupted runs. This removes every pool the current or historical demo registered, not merely the one expected in a clean new run, while an unknown pool causes a safe refusal.

- [ ] **Step 4: Update both launchers to one shared pool**

Use identical extra configuration in A and B:

```json
{
  "spec_name": "SpyreSharedOffloadingSpec",
  "shared_metadata_name": "spyre_manual_4096",
  "pool_name": "spyre_manual_4096.data",
  "cpu_bytes_to_use": 536870912
}
```

Preserve TP1, `uni`, `--no-enable-prefix-caching`, `--max-num-batched-tokens 512`, `--max-num-seqs 1`, `PYTHONHASHSEED=0`, distinct default devices, and worktree-first `PYTHONPATH`.

- [ ] **Step 5: Retest the visual scripts without weakening their output**

Keep the warmup prompt visibly unrelated to the measured prompt. Before every request, print instance name, host/port endpoint, identifier, prompt preview, and exact token count. After every request, print path classification, TTFT, end-to-end wall time, local-compute/external-transfer tokens, store/load bytes, store/load copy time, output token IDs, and output text.

Run:

```bash
uv run --no-sync pytest tests/kv_offload/test_cleanup_shared_kv_demo.py tests/kv_offload/test_warmup_shared_kv_demo.py tests/kv_offload/test_shared_kv_two_instance_demo.py -q
```

Expected: all script tests pass, including metric families that are absent before their first sample being treated as zero rather than as a fatal missing metric.

- [ ] **Step 6: Rewrite the manual guide's topology and inspection section**

Replace every statement about 16 component pools or two families. Document the clean live topology as:

```text
/dev/shm/spyre_manual_4096
/dev/shm/flex_kv_<metadata-version>_<pool-id>_<pool-version>
/dev/shm/flex_kv_<metadata-version>_<pool-id>_<pool-version>.ctl
```

Include this inspection command while both servers are running:

```bash
strings /dev/shm/spyre_manual_4096 \
  | grep '^/flex_kv_' \
  | sort -u \
  | while read -r pool; do
      stat -c '%n  %s bytes' "/dev/shm/${pool#/}"
      stat -c '%n  %s bytes' "/dev/shm/${pool#/}.ctl"
    done
```

State that one unique base name is expected. The metadata object is not a data pool; the `.ctl` object belongs to the same one data backing.

- [ ] **Step 7: Run the manual A -> A -> B visual sequence and save logs**

After stopping old servers:

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
uv run --no-sync python scripts/cleanup_shared_kv_demo.py
```

Start A and B with the documented launchers. In terminal C, save unrelated warmup and each measured request separately:

```bash
set -o pipefail
LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync python -u scripts/warmup_shared_kv_demo.py \
  --instance-a-host 127.0.0.1 --instance-a-port 18100 \
  --instance-b-host 127.0.0.1 --instance-b-port 18101 2>&1 \
  | tee /tmp/spyre-shared-kv-one-pool-warmup.log

LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance A --host 127.0.0.1 --port 18100 2>&1 \
  | tee /tmp/spyre-shared-kv-one-pool-a-cold.log

LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance A --host 127.0.0.1 --port 18100 2>&1 \
  | tee /tmp/spyre-shared-kv-one-pool-a-reload.log

LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance B --host 127.0.0.1 --port 18101 2>&1 \
  | tee /tmp/spyre-shared-kv-one-pool-b-reload.log
```

Use the same identifier for A, A, and B. Require 4,096 prompt tokens, 16 output tokens, one cold compute/store, two external reloads, identical output, equal expected KV bytes, and reload TTFT shorter than cold TTFT. Record performance; do not encode a timing threshold into unit tests.

- [ ] **Step 8: Stop both servers, clean every owned pool, and record post-cleanup evidence**

```bash
uv run --no-sync python scripts/cleanup_shared_kv_demo.py \
  | tee /tmp/spyre-shared-kv-one-pool-cleanup.log
test ! -e /dev/shm/spyre_manual_4096
```

Verify every backing/control name printed by the pre-cleanup inspection is absent afterward and unrelated `/dev/shm` objects remain.

- [ ] **Step 9: Update the result document with the fresh one-pool run**

Record the three repository commit hashes, model, vLLM version, device IDs, pool geometry, `/dev/shm` object list, exact commands/log paths, prompt/output token counts, cold/reload TTFT, end-to-end wall time, computed/loaded token counts, KV bytes, copy time, and output equality. Keep the earlier reference result clearly labeled as historical; do not present its 16-pool topology as current.

### Task 7: Run Final Regression Gates, Review, and Sign Only After Success

**Files:**

- Verify: every changed file in `spyre-inference-kvc-offload-m2`
- Verify only: `/home/yzhu/dt-inductor/flex` M1/M2 lower-layer tests
- Verify only: `/home/yzhu/dt-inductor/torch-spyre/.worktrees/kvc-offload-m2` M1/M2 lower-layer tests

**Interfaces:**

- Consumes: all deliverables from Tasks 1-6.
- Produces: evidence-backed final branch, one signed redesign commit, and no remote mutation.

- [ ] **Step 1: Run the complete mock-safe spyre-inference KV-offload gate**

```bash
uv run --no-sync pytest \
  tests/kv_offload/test_spec.py \
  tests/kv_offload/test_canonicalize_paged.py \
  tests/kv_offload/test_worker_dispatch.py \
  tests/kv_offload/test_spyre_kv_offload.py \
  tests/kv_offload/test_shared_types.py \
  tests/kv_offload/test_shared_spec.py \
  tests/kv_offload/test_shared_manager.py \
  tests/kv_offload/test_shared_worker_dispatch.py \
  tests/kv_offload/test_connector_miss_recompute.py \
  tests/kv_offload/test_cleanup_shared_kv_demo.py \
  tests/kv_offload/test_warmup_shared_kv_demo.py \
  tests/kv_offload/test_shared_kv_two_instance_demo.py \
  -m "not upstream" -q
```

Expected: all pass with no M2 runtime import during ordinary plugin import.

- [ ] **Step 2: Run lower-layer M1 regression tests without modifying their APIs**

Run the existing Flex shared-metadata, shared-host-pool, and raw-copy suites, including the range/bounds cases. Then run torch-spyre's shared metadata and KV page validator suites:

```bash
FLEX_DEVICE=MOCK1p0 FLEX_COMPUTE=NULL \
LD_LIBRARY_PATH=/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
/home/yzhu/dt-inductor/build/flex/tests/flex_unit_test \
  --gtest_filter='SharedHostPoolTest.*:SharedMetadata*.*:RuntimeStreamCopyRawUnittest.*'

cd /home/yzhu/dt-inductor/torch-spyre/.worktrees/kvc-offload-m2
uv run --no-sync pytest tests/test_shared_metadata.py tests/test_kv_page_validator.py -q -s
```

Expected: independent slot allocation, `copyRaw(..., Range)`, token-major bounds, and head-major bounds remain green. No lower-layer production diff is expected.

- [ ] **Step 3: Run all real Spyre tests serially**

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
uv run --no-sync pytest \
  tests/kv_offload/test_spyre_kv_offload_hw.py \
  tests/kv_offload/test_worker_hw.py \
  tests/kv_offload/test_shared_pool_round_trip.py \
  -q -s
```

Expected: M1 and one-pool M2 page transfers pass byte-exactly. Do not start either vLLM server while this command owns a card.

- [ ] **Step 4: Repeat automated and manual two-instance acceptance**

Run the Task 5 cross-instance command, then the Task 6 cleanup/warmup/A/A/B/cleanup sequence. Expected: one data pool, positive store/load metrics, identical deterministic outputs, and both reload TTFT values shorter than cold TTFT.

- [ ] **Step 5: Run formatting, static checks, and diff checks**

```bash
bash format.sh
uv run --no-sync ty
git diff --check
SKIP=markdownlint pre-commit run --all-files
```

Fix every diagnostic introduced by this branch. Record any unchanged environment-only type diagnostic separately with its baseline evidence; do not weaken checks.

- [ ] **Step 6: Audit scope and branch isolation**

```bash
git status --short --branch
git diff --stat
git diff -- spyre_inference/v1/kv_offload tests/kv_offload scripts docs/superpowers
git -C /home/yzhu/dt-inductor/flex status --short --branch
git -C /home/yzhu/dt-inductor/torch-spyre/.worktrees/kvc-offload-m2 status --short --branch
```

Expected: only the intended spyre-inference redesign/demo files changed; unrelated existing Flex and torch-spyre work remains untouched.

- [ ] **Step 7: Use the requesting-code-review skill on the complete uncommitted diff**

Review specifically for page-key compatibility, pin lifetime, reservation ownership, fragmented-slot routing, worker failure quiescence, cleanup safety, M1 behavior, and acceptance coverage. Resolve findings with focused tests, then rerun every affected gate and `git diff --check`.

- [ ] **Step 8: Create the signed commit only after all gates and review pass**

```bash
git add \
  spyre_inference/v1/kv_offload/shared_types.py \
  spyre_inference/v1/kv_offload/shared_spec.py \
  spyre_inference/v1/kv_offload/shared_manager.py \
  spyre_inference/v1/kv_offload/shared_worker.py \
  tests/kv_offload/test_shared_types.py \
  tests/kv_offload/test_shared_spec.py \
  tests/kv_offload/test_shared_manager.py \
  tests/kv_offload/test_shared_worker_dispatch.py \
  tests/kv_offload/test_shared_pool_round_trip.py \
  tests/kv_offload/test_cross_instance.py \
  tests/kv_offload/test_cleanup_shared_kv_demo.py \
  tests/kv_offload/test_warmup_shared_kv_demo.py \
  tests/kv_offload/test_shared_kv_two_instance_demo.py \
  scripts/start_shared_kv_instance_a.sh \
  scripts/start_shared_kv_instance_b.sh \
  scripts/cleanup_shared_kv_demo.py \
  scripts/warmup_shared_kv_demo.py \
  scripts/shared_kv_two_instance_demo.py \
  docs/superpowers/specs/2026-09-29-spyre-shared-kv-offload-design.md \
  docs/superpowers/plans/2026-09-29-spyre-shared-kv-offload.md \
  docs/superpowers/results/2026-09-30-spyre-shared-kv-manual-demo.md
git commit -s -m "Fix shared KV page placement"
```

If commit signing requires an unavailable interactive prompt, leave the verified changes staged and run this exact command in the user's terminal instead. Do not push, force-push, or open a pull request in this plan; obtain explicit final authorization after presenting the commit and test evidence.
