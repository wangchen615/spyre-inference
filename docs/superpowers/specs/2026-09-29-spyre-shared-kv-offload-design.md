# Spyre Cross-Instance Shared KV Offload Design

## Purpose

Add an explicitly selected `SpyreSharedOffloadingSpec` that lets compatible
vLLM instances on one host reuse Spyre KV pages through one Flex
`SharedMetadata` directory and one `SharedHostPool` data pool. Instance A can
publish the pages produced for a vLLM offload key, and instance B can locate
and reload every page required for that same key without recomputing its
prefix.

The design follows the slot geometry and copy contract in
`SharedKvPoolRFC.md` at commit
`114011e842014b8ce6eb2fde9e3f83fd6abb9a81`:

- a data pool is one DMA-reachable region divided into equal-size slots;
- one slot holds one physical KV page;
- `slot_bytes` accommodates the largest physical KV page used with the pool;
- each metadata key resolves to one `(pool_id, slot_id)`;
- one page is copied to or from the beginning of its selected slot; and
- a pool slot is claimed independently from the pool's free-slot list.

The first milestone prioritizes correctness. The acceptance run also records
time to first token (TTFT), end-to-end request time, and KV transfer metrics so
that cold computation can be compared with same-instance and peer-instance
reload.

## Starting Point and Branch Isolation

The implementation branch is `kvc-offload-m2`, rebased on
`kvc-offload-m1`, in
`/home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2`. Flex and torch-spyre
have corresponding local M2 branches rebased on their M1 branches.

M1 already provides the required page-copy mechanism:

- Flex `copyRaw` accepts an optional device `Range` on its pool-slot overload.
- torch-spyre `copy_kv_page_raw` validates the full rank-four KV allocation,
  derives the physical range for one `block_id`, and passes that range to
  Flex.
- The range works for both token-major and head-major layouts because each
  physical page is one contiguous, aligned interval in the validated device
  image.

M2 therefore does not add a pool-relative byte offset or change the raw-copy
API. There is no such pool-offset parameter in the current M1 or M2 API, so
nothing is removed: every selected device page continues to copy at byte zero
of its independently claimed host slot. Existing unrelated and visual-demo
work in the worktrees remains preserved. Commits, sign-off, and pushing are
deferred until implementation and validation are complete.

## Terminology

- **vLLM block:** the logical cache unit identified by an `OffloadKey`. With
  `block_size=128`, it represents 128 token positions for one KV-cache group.
- **KV component:** one stable K or V cache tensor participating in the
  supported KV-cache group.
- **KV page:** the physical byte interval `component_tensor[block_id]` in one
  full K or V cache tensor. For a cache shaped `[num_blocks, ...]`, each value
  of `block_id` selects one page.
- **Data-pool slot:** one fixed-size host-memory location that stores exactly
  one KV page.
- **Page key:** a deterministic key derived from the complete vLLM
  `OffloadKey` and one stable component ID. Flex maps one page key to one
  pool slot.

One vLLM block normally has several component pages. It therefore occupies
several independently claimed slots in the one data pool. A full K or V cache
tensor contains pages for many vLLM block IDs and can consequently occupy many
pool slots over time. An individual page never spans slots.

## Scope

The implementation includes:

- lazy registration of `SpyreSharedOffloadingSpec` without changing the M1
  `SpyreOffloadingSpec` selection path;
- one shared metadata directory and exactly one independently backed shared
  data pool for this PoC;
- RFC slot geometry based on one physical KV page;
- deterministic component-qualified page keys;
- scheduler-side lookup, claim, pin, completion, and local eviction across all
  component pages belonging to a vLLM offload key;
- worker-side D2H and H2D through M1's validated `copy_kv_page_raw` path;
- connector-level all-pages-present semantics for a logical cache hit;
- normal vLLM recomputation when any required page is absent or cannot be
  pinned;
- cleanup and visual-demo scripts that discover and remove every shared pool
  owned by the configured demo namespace;
- mock-safe, hardware, cross-instance, and manual-demo validation; and
- TTFT, end-to-end time, copied bytes, copy time, and output-equivalence
  reporting.

M2 retains M1's current restrictions: float16 KV pages, one KV-cache group,
`blocks_per_chunk == 1`, and the current supported single-chunk Spyre device
allocations. This milestone additionally requires `world_size == 1`, TP1, and
the `uni` executor. The two-instance test uses two independent processes and
two distinct Spyre devices.

The implementation does not add multiple active data pools, cross-pool
placement, contiguous multi-slot allocation, packed aggregate slots,
pool-relative copy offsets, disk/network tiers, multi-host sharing, DP, or a
host-wide victim policy.

## Selected Architecture

### One metadata directory and one data pool

Both instances use the same logical names:

```yaml
spec_name: SpyreSharedOffloadingSpec
shared_metadata_name: <host-unique-metadata-name>
pool_name: <host-unique-data-pool-name>
cpu_bytes_to_use: <pool-budget>
```

`shared_metadata_name` identifies the one CPU-only metadata object.
`pool_name` identifies the one `SharedHostPool`. Both instances must resolve
the same values. The RFC configuration precedence for `pool_name` remains
explicit configuration, then `SPYRE_KV_POOL_NAME`, then a generated private
name. Cross-instance tests always pass an explicit shared name.

The Flex metadata capacity is configured for one registered pool. The pool is
created or attached once by each participating worker and resolves to one
`pool_id`. All page locations used by M2 contain that same `pool_id`.

Flex remains capable of registering several independent pools for future
memory kinds or placement policies, but M2 neither configures nor exercises
that policy.

### Slot geometry

During KV-cache registration, the worker derives every component's physical
page size from its complete device allocation:

```text
page_bytes[c] = composite_address[c].total_size / num_device_blocks
```

It validates the same divisibility, contiguity, layout, and alignment
requirements as `copy_kv_page_raw`. For `C` components:

```text
slot_bytes = align_up(max(page_bytes[0:C]), required_pool_alignment)
```

Every slot in the pool has this fixed stride. A component smaller than
`slot_bytes` leaves the remainder of its slot unused. Each copy must satisfy:

```text
page_bytes[component] <= slot_bytes
```

The number of pool slots is a multiple of `C`, so scheduler-visible logical
capacity and physical page capacity remain in step:

```text
logical_block_capacity = floor(cpu_bytes_to_use / (C * slot_bytes))
pool_slot_count         = logical_block_capacity * C
actual_pool_bytes       = pool_slot_count * slot_bytes
```

If the budget cannot hold all component pages for one vLLM block, startup
fails. The effective geometry and any bytes lost to slot alignment or
fixed-slot padding are logged.

For the manual model, the expected geometry is eight page components per
vLLM block, 256-KiB page slots, 2,048 pool slots, and one 512-MiB data pool.
The implementation derives and verifies these values rather than hard-coding
them.

### No slot-contiguity contract

Flex `Claim` removes one slot from an intrusive free-slot list. Initial claims
may happen to return increasing indices, but abort and eviction return
individual slots to the head of that list. Neither the RFC nor the API offers
a contiguous extent allocation.

Accordingly, component pages for one vLLM block may reside anywhere in the
pool:

```text
component 0 -> (pool P, slot 7)
component 1 -> (pool P, slot 2)
component 2 -> (pool P, slot 11)
```

Every page location is stored and carried explicitly. No code derives a
component location with `base_slot + component_index`, and no test relies on
fresh-pool allocation order.

### Page identity

Flex stores one 64-bit `CompatibleBlockKey.block_hash` per directory entry.
spyre-inference derives it from:

```text
page_key = digest(format_version, complete OffloadKey, component_id)
```

The complete `OffloadKey` includes vLLM's block hash and KV-cache group ID.
`component_id` is a stable ordinal from the ordered component manifest shared
by compatible instances. The manifest identifies each K/V component and is
derived deterministically from the KV-cache group and physical-cache binding;
process-local object identities are never encoded.

The compatibility descriptor covers every fact required to interpret a page
key and its bytes:

- format version and page-key algorithm;
- model identity and revision;
- vLLM block-hash algorithm and fixed hash seed;
- token block size and hash granularity;
- dtype and TP size;
- ordered component IDs and K/V roles; and
- each component's layout kind, layout version, physical page bytes, block
  size, local KV heads, and head size.

Registration or attachment rejects a different manifest, slot geometry, or
model compatibility before issuing DMA.

### Logical block availability

Flex independently tracks each page entry. spyre-inference defines a vLLM
offload-key hit as the conjunction of all required page entries:

```text
logical_hit(key) = every page_key(key, component) is VALID and pinned
```

Individual page publication is visible in Flex, but a partially published or
partially evicted set is never exposed to vLLM as a logical hit. If lookup or
pinning fails for any page, pins already acquired for that key are released
and the connector reports `LookupResult.MISS`.

## Components

- `shared_spec.py` owns M2 configuration, one-pool geometry, compatibility,
  and manager/worker construction.
- `shared_types.py` defines the component manifest, component-qualified page
  key, page location, page reservation, and process-local load/store specs.
- `shared_manager.py` retains upstream admission and logical-block policy while
  translating each `OffloadKey` into its complete ordered page set.
- `shared_worker.py` registers or attaches one pool and transfers each page to
  its explicitly supplied slot.
- `shared_runtime.py` remains the lazy boundary around the torch-spyre M2
  bindings.
- `connector.py` continues to provide both canonical vLLM bookkeeping and the
  original rank-four physical cache tensors required by
  `copy_kv_page_raw`.

The component manifest is finalized during KV-cache canonicalization and must
be identical on the manager and worker sides before the first shared lookup.
The `uni`-executor restriction keeps the process-local transfer specification
and reservation objects out of serialization paths.

## Data Flow

### Lookup and reload

1. For a vLLM `OffloadKey`, the manager derives every ordered page key.
2. It looks up and read-pins each page entry independently.
3. If any lookup or pin fails, it releases all pins acquired for that logical
   key and returns `MISS`.
4. On a complete hit, `prepare_load` passes every explicit page location to
   the worker while retaining every pin.
5. For each component, the worker calls `copy_kv_page_raw` with the full
   rank-four cache tensor, the destination device block ID, the one shared
   pool, and that component's slot ID.
6. `copy_kv_page_raw` derives a device `Range`; Flex copies the selected page
   to the beginning of the host slot. No pool-relative offset is used.
7. After the H2D DMAs synchronize, completion releases all page pins.

### Claim, store, and publish

1. Upstream policy decides whether to admit the logical vLLM block and which
   locally owned logical blocks to evict.
2. The manager processes component keys in one deterministic order. This makes
   the first component claim serialize ordinary competing writers.
3. Each missing component receives an independent reservation from the same
   pool. Existing valid pages may be reused. An existing reserved page, an
   unavailable directory, or insufficient free slots causes the new store to
   stop; reservations obtained by that attempt are aborted before returning.
4. The worker copies only newly reserved component pages D2H into their exact
   slots. Existing valid pages are not overwritten.
5. After all submitted page DMAs synchronize, each new reservation is
   published with that page's own chunk descriptor.
6. The manager reports the logical store complete only after every required
   page is valid. A concurrent reader that observes only part of the page set
   reports a miss.
7. If DMA or publication fails, the worker synchronizes, evicts any entries
   published by that attempt, aborts its remaining reservations, and reports
   failure. Entries that predated the attempt are untouched.

Flex does not provide a multi-entry transaction. Connector-level all-page
lookup is therefore the visibility boundary. This is safe because a page key
is immutable for one content-derived key and a partial set is never loaded.

### Eviction and ownership

Upstream LRU or ARC policy continues to select locally admitted logical
blocks. For each selected logical block, the shared manager evicts the exact
page entries that this instance published. Eviction waits for each page's
read pin before recycling its slot.

Peer-owned entries are readable but are not silently adopted into this
instance's eviction ownership. A mixture of pre-existing and newly published
pages records ownership per page. Global victim selection and runtime cleanup
of unowned partial bundles remain outside this milestone. Controlled test and
demo teardown removes all shared objects owned by that demo namespace only
after every participant exits.

## Failure Handling

- Missing M2 runtime symbols fail only when `SpyreSharedOffloadingSpec` is
  selected.
- Metadata, compatibility, or pool-geometry mismatches fail at startup.
- A page larger than `slot_bytes`, a misaligned device range, or an invalid
  block ID fails before DMA.
- Missing, reserved, stale, or unpinnable page entries produce a logical cache
  miss rather than a partial reload.
- A mid-claim failure aborts every reservation acquired by that attempt.
- A transfer failure is synchronized and rolled back before any affected slot
  can be reused.
- `NoSpace` skips the store without changing generated output.
- Cleanup does not unlink live shared objects. After all participating servers
  stop, controlled cleanup removes every metadata object, data backing, and
  control object owned by the configured demo namespace, including stale
  objects left by interrupted or older demo runs. It never removes unrelated
  pools.

## Repository Impact

### Flex

No production API change is expected. M2 uses the existing RFC operations:

- one-pool registration and resolution;
- independent `Claim`, `Lookup`, `PinRead`, `Publish`, `Abort`, and `Evict`;
- fixed-size slot geometry and bounds checks; and
- `copyRaw(pool, slot, ..., Range)`.

Tests must confirm that page entries remain correct when the free list returns
non-contiguous slot IDs. A Flex implementation defect found by those tests is
fixed separately and narrowly.

### torch-spyre

No production API change is expected. M2 reuses M1's
`copy_kv_page_raw(cache, block_id, pool, slot_id, ...)`, including its
head-major and token-major range validation. Existing range, bounds, and
hardware round-trip tests remain regression gates.

### spyre-inference

This repository contains the substantive redesign:

- replace `shared_pool_families` and component pool names with one
  `pool_name`;
- replace anchor/sibling routing with component-qualified page entries;
- compute RFC page-slot geometry and one-pool capacity;
- carry explicit, potentially non-contiguous page locations;
- implement all-pages lookup/pinning and per-page claim/publication rollback;
- preserve M1 connector behavior and restrictions; and
- update launch, cleanup, visual demo, tests, and result documentation.

## Testing and Acceptance

### Mock-safe tests

- M1 factory selection and worker behavior remain unchanged.
- M2 imports lazily when shared runtime symbols are unavailable.
- configuration resolves exactly one data-pool name.
- slot size is the aligned maximum component page size.
- pool capacity accounts for all component slots per logical block.
- page keys are deterministic and differ by component.
- attachment rejects component-manifest or geometry mismatch.
- a deliberately fragmented free list returns non-contiguous slots and the
  complete page set still round-trips correctly.
- lookup reports a hit only after every component page is valid and pinned.
- partial lookup releases already acquired pins and reports a miss.
- claim, DMA, and publish failures abort or evict only entries created by the
  failing attempt.
- locally owned logical eviction removes every locally published component
  entry and respects outstanding page pins.

### Hardware tests

- Run existing Flex raw-copy and shared-metadata tests, including range and
  bounds coverage.
- Run existing torch-spyre `copy_kv_page_raw` tests for token-major and
  head-major caches.
- Store and reload a known-pattern logical block through one shared pool and
  compare every K/V page byte-for-byte.
- Run Spyre-backed commands serially, except for the deliberate two-instance
  topology where each process owns a distinct accelerator.

### Cross-instance test

Run two compatible TP1 vLLM instances using the same metadata and data-pool
names with prefix caching disabled. Verify that A publishes all pages, B finds
and pins their explicit locations, B performs H2D reload, and deterministic
output matches the cold baseline. The existing cross-instance functional test
must continue to pass after its configuration and shared-memory assertions are
updated from component pools to one data pool.

### Manual A -> A -> B demo

1. Stop both servers and run the cleanup script.
2. Start instance A and instance B on different Spyre devices.
3. Warm both servers with unrelated junk prompts so compilation and lazy
   initialization are excluded from measured requests.
4. Send the measured prompt to A for cold compute/store.
5. Send the same prompt to A for self-reload.
6. Send the same prompt to B for peer reload.
7. Record prompt/output token counts, cold/reload TTFT, end-to-end time,
   transferred bytes, copy time, and output equality.
8. Verify prefix caching remains disabled and shared-KV metrics prove that the
   reload requests loaded rather than recomputed the prompt blocks.
9. During the run, verify `/dev/shm` contains one metadata object and one active
   data backing/control pair. After both servers stop, run cleanup and verify
   that every shared-memory object owned by the demo namespace is gone,
   including stale pools from interrupted or older runs.

The expected functional outcome is that the self and peer reloads transfer the
same KV byte count, produce identical output, and show shorter TTFT than cold
prefill. Performance is reported rather than enforced as a unit-test
threshold. These are the same functional and measurement requirements as the
current demo; only the pool configuration, cleanup expectations, and
`/dev/shm` topology change to one data pool.

## Completion Criteria

The milestone is complete when:

- both instances attach one compatible data pool;
- each data slot holds one physical KV page and no implementation assumes slot
  adjacency;
- all page entries for one vLLM key reload correctly across instances;
- partial or stale page sets safely recompute;
- M1 Flex, torch-spyre, and spyre-inference regression tests remain green;
- the cross-instance test passes on real Spyre hardware;
- the documented A -> A -> B demo reports TTFT, end-to-end time, KV bytes,
  copy time, and identical output;
- `/dev/shm` shows only the expected one-pool objects during a clean run; and
- cleanup removes every pool and metadata object owned by the demo namespace
  after all participants exit, without touching unrelated shared memory.
