# Spyre Cross-Instance Shared KV Offload Design

## Purpose

Add an explicitly selected `SpyreSharedOffloadingSpec` that lets two vLLM
instances on one host reuse KV blocks through Flex `SharedMetadata` and
`SharedHostPool`. Instance A can publish a completed logical KV block and
instance B can recognize the same vLLM offload key, reload the complete block,
and generate the same deterministic output as a no-cache run.

The primary milestone is functional correctness. The acceptance run records
the time for instance A to reload its own published blocks and for instance B
to reload those same blocks, but an M1 versus M2 performance comparison is
deferred.

## Starting Point and Branch Isolation

The implementation branch is `kvc-offload-m2`, created from
`kvc-offload-m1` commit `86f56ed`. It lives in the isolated worktree
`/home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2`; the dirty
`kvc-offload-poc` worktree remains untouched.

Flex `kvc-offload-m2` at `e6dff26d` supplies the versioned directory, publish
gate, slot read pins, dynamic pool registration, and cross-process race tests.
torch-spyre `kvc-offload-m2` at `39b479fd` exposes that contract to Python and
already verifies metadata plus DMA across processes. Changes to those projects
are made only on their existing `kvc-offload-m2` branches and only when an
integration test demonstrates a missing runtime capability.

The mock-safe M1 baseline is 36 passing tests across `test_spec.py`,
`test_canonicalize_paged.py`, and `test_worker_dispatch.py`.

## Scope

The implementation includes:

- lazy factory registration for `SpyreSharedOffloadingSpec`;
- shared directory and shared data-pool attachment;
- deterministic vLLM offload-key conversion and cache-compatibility checks;
- scheduler-side shared lookup, read-pin lifetime, local admission, and local
  eviction integration;
- worker-side D2H and H2D transfer through M1's validated
  `copy_kv_page_raw` path;
- atomic visibility of a complete logical KV block after all of its component
  pages finish D2H;
- ordinary recomputation when lookup or read-pin validation reports a miss;
- mock-safe unit and connector tests, a Spyre-gated connector round trip, and a
  two-instance functional acceptance run;
- recorded shared-pool reload timings for instance A's self-reload and
  instance B's peer reload.

The implementation does not add a cache policy, disk or network tiers,
multi-host sharing, DP support, or a host-wide eviction algorithm. It does not
use or modify vLLM's `SharedOffloadRegion`. The existing M1
`SpyreOffloadingSpec` retains its engine/rank-private pools, configuration, and
observable behavior.

M2 retains M1's current restrictions: float16 KV pages, one KV-cache group,
and `blocks_per_chunk == 1`. This first cross-instance integration additionally
requires one worker per vLLM instance (`world_size == 1` and the `uni`
executor). The two-instance acceptance run therefore uses two independent
single-worker instances on two distinct Spyre devices. M1's existing TP support
is unaffected; shared offload across TP workers requires a later serializable
reservation and aggregate-publish protocol.

## Approaches Considered

### Worker-only shared pool substitution

This is the smallest literal interpretation of the issue: inherit M1's
manager and worker and replace only private-pool construction. It cannot
produce a peer hit. The upstream `CPUOffloadingManager` maps an offload key to
a process-local numeric block ID; instance B therefore reports a miss before
its worker is asked to load anything. M1's worker also expands one logical host
block across every physical cache, whereas one Flex directory entry names one
pool slot.

### vLLM `SharedOffloadRegion`

This preserves more upstream CPU-offload code, but it is the wrong storage
contract. Its lifetime and geometry are scoped to workers in one vLLM engine,
its DMA registration is CUDA-oriented, and it has no cross-instance block-hash
directory, versioned read pin, or publish gate.

### Policy-neutral manager integration plus a shared worker

This is the selected approach. `SpyreSharedOffloadingManager` subclasses
`CPUOffloadingManager` and calls its existing operations for admission
thresholds, LRU/ARC victim selection, reference counts, metrics, and completion
bookkeeping. It adds only the shared-directory address/protocol operations
required to turn a vLLM key into a cross-instance location. A shared worker
consumes those locations while reusing M1's page validation and raw DMA
primitive.

## Registration and Runtime Compatibility

`spyre_inference.__init__` registers
`SpyreSharedOffloadingSpec` with `OffloadingSpecFactory` using module and class
name strings. Importing `spyre_inference` does not import `shared_spec.py` or
access `torch_spyre._C.SharedMetadata`. The M2 module and runtime symbols are
loaded only when the user selects `SpyreSharedOffloadingSpec`.

Consequently:

- an M1-only torch-spyre build can still import and use the plugin;
- selecting the M2 spec on such a build fails with a direct error naming the
  missing M2 runtime surface;
- selecting `SpyreOffloadingSpec` continues to construct the M1 path.

The M1 spec gains `_create_manager()` and `_create_worker()` protected
construction hooks for its manager and worker.
Their default implementations construct the same M1 objects with the same
arguments as today. The M2 subclass overrides those hooks, avoiding a copied
version of M1's validation, metrics declaration, and physical-cache binding.

## Shared Configuration and Compatibility

Both vLLM instances use the same shared metadata name and the same ordered
pool-family configuration in `kv_connector_extra_config`:

```yaml
spec_name: SpyreSharedOffloadingSpec
shared_metadata_name: <host-unique-name>
shared_pool_families: [<family-name>, ...]
cpu_bytes_to_use: <aggregate-logical-capacity-in-bytes>
```

`shared_metadata_name` is non-empty and common to the participating instances.
`shared_pool_families` is a non-empty ordered list of unique names. M1's
existing `cpu_bytes_to_use` calculation determines the aggregate logical slot
count; slots are divided across the families in list order, with the remainder
assigned one at a time to the earliest families. A configuration with fewer
logical slots than families is rejected.

A pool family is a capacity shard with one common logical slot count. Each
physical K or V page component has a separate data pool in that family. Its
logical name is `<family>.c<cache-index>.<k-or-v>`. Slot `s` in every component
pool belongs to the same complete logical KV block.

The `c0.k` component pool is the family's anchor. `SharedMetadata.claim`
allocates only anchor slots. The returned `(anchor_pool_id, slot_id)` identifies
the family and the common logical slot; auxiliary component pools are never
independently claimed. The anchor slot's read pin protects reuse of the whole
family slot because every conforming writer obtains that slot exclusively
through the anchor claim.

The compatibility descriptor covers every fact needed to interpret the
bundle:

- format version;
- model identity and revision;
- vLLM block-hash algorithm and fixed hash seed;
- tokens per hash and block;
- dtype;
- the required TP size of one and component ordering;
- ordered physical-cache component signatures, including K/V role, layout,
  page bytes, block size, local KV heads, and head size.

Attach rejects mismatched geometry or compatibility before any transfer.
Cross-instance operation requires the same deterministic vLLM block hashes;
both acceptance servers therefore use the same explicit `PYTHONHASHSEED` and
hash algorithm.

Flex currently stores a 64-bit `BlockHash`. spyre-inference derives that value
deterministically from the complete vLLM `OffloadKey`, including its group ID,
with a fixed 64-bit digest. The compatibility descriptor isolates incompatible
models and layouts. Expanding the Flex key width is a separate runtime-format
change and is not part of this milestone.

Multiple configured pool families contribute aggregate capacity. Placement
tries compatible families in deterministic hash order and records the family
actually returned by `claim`; lookup always follows the directory entry and
therefore resolves one exact family and slot.

The worker creates or attaches the directory during KV-cache registration,
using empty initial pools and fixed capacity for every configured family and
the maximum component count implied by the KV-cache layer configuration. It
then derives the exact physical component signatures, registers or attaches
each component pool, and resolves its DMA-capable handles. The manager attaches
the same directory lazily on its first lookup or store, after worker
registration and device initialization. It resolves every family anchor by
name before lookup or claim. This ordering prevents M2 construction from
starting torch-spyre's runtime before the worker establishes its rank and
device environment.

## Components

- `shared_spec.py` owns M2 configuration, lazy runtime imports, directory
  attachment, and the M2 manager/worker construction hooks.
- `shared_manager.py` adapts `CPUOffloadingManager` decisions to versioned
  directory entries and owns pending reservations and read pins.
- `shared_worker.py` maps a family and logical slot to every physical K/V pool,
  executes M1's validated page copies, and publishes only after the complete
  logical block is durable in host memory.
- A small `SharedLoadStoreSpec` carries offload keys, family/slot locations,
  and store reservations between the manager and worker. It remains
  process-local under the required `uni` executor; no pybind protocol object is
  serialized.
- `spyre_inference.__init__` contains only the lazy factory registration and no
  M2 runtime import.

The directory is attached independently by the scheduler and worker connector
objects, but both live in the same process under the required `uni` executor.
Pool registration and resolution happen only on the worker side, where a Spyre
runtime and physical KV caches already exist. The manager's lazy attach thus
reuses an initialized process runtime and does not require a metadata-only Flex
or torch-spyre API.

## Cache Policy Boundary

`SpyreSharedOffloadingManager` retains upstream policy behavior:

- `store_threshold` still controls admission;
- the configured upstream LRU or ARC policy still selects locally owned
  victims;
- upstream reference counts protect locally owned in-flight blocks;
- upstream completion and metrics behavior remains in force.

The shared integration adds a distinction between locally owned entries and
peer entries. Locally admitted blocks participate in this instance's upstream
policy and are removed from `SharedMetadata` when that policy evicts them.
Peer hits are readable through the directory but are not silently adopted into
the local eviction policy, so one instance does not invent a policy decision
for another instance's entry.

If all compatible slots are occupied by peer-owned entries, a store is skipped
rather than evicting an arbitrary peer entry. Flex explicitly leaves
host-wide victim selection to follow-on work. This limitation does not affect
peer lookup/reload and prevents M2 from introducing an unreviewed global cache
policy.

## Data Flow

### Store and publish

1. Upstream `CPUOffloadingManager` logic decides whether the logical block is
   admitted and which locally owned entries are evicted.
2. The shared manager evicts the exact directory entries corresponding to
   those upstream-selected victims.
3. It claims an anchor slot for the new offload key. A valid existing claim
   means another instance already published the block, so no duplicate D2H is
   scheduled and the pending local admission is rolled back. A reserved
   existing claim remains invisible; this store is skipped and its pending
   local admission is also rolled back. `NoSpace` skips the store after the
   same rollback; `Unavailable` is reported as a configuration/runtime error.
4. The worker copies every K/V physical page component into the selected pool
   family's common slot using the same page-shape validation and
   `copy_kv_page_raw` operation as M1.
5. The single M2 worker synchronizes and publishes the anchor reservation only
   after every component copy completes. Until then, lookup returns a miss,
   including during a partially completed multi-component write.
6. Worker completion lets the manager finish the corresponding upstream local
   admission.
7. A failed store is synchronized before its reservation is aborted, so a slot
   is never reused while DMA can still write it.

### Lookup and reload

1. The shared manager converts the vLLM offload key and calls
   `SharedMetadata.lookup`.
2. It immediately calls `pin_read` on the returned versioned entry. A missing,
   reserved, stale, or concurrently replaced entry returns `LookupResult.MISS`.
3. A successful pin is retained by the scheduler-side manager while the load
   job carries the resolved family and slot identifier to the worker.
4. The worker reloads every K/V component from that common slot using
   `copy_kv_page_raw`, then synchronizes.
5. `complete_load` releases the pin on the scheduler thread after worker
   completion. Eviction cannot recycle the slot before that release.

### Recompute on miss

A directory lookup miss or failed `pin_read` is reported before vLLM commits to
an external load. The ordinary offloading scheduler therefore schedules the
uncached tokens for model execution. No H2D is issued and no stale slot is
consumed. This test targets spyre-inference's translation from the binding
result to vLLM's `LookupResult.MISS`; Flex and torch-spyre retain responsibility
for proving that races produce that binding result instead of torn bytes.

## Worker Reuse Boundary

The current M1 worker cannot remain byte-for-byte unchanged as an M2 worker:
it constructs private pools internally and receives only process-local host
block IDs. M2 requires a family-plus-slot location supplied by the shared
directory.

The implementation therefore preserves and reuses the correctness-sensitive
parts rather than duplicating or replacing them:

- physical rank-4 cache binding and layout signatures;
- `copy_kv_page_raw` validation and byte-exact DMA;
- one pre-store device fence per job;
- synchronous completion and transfer accounting;
- single-KV-group validation.

M1 keeps its existing private-pool worker behavior. M2 adds only shared
location routing and publish/pin protocol coordination around the reused page
copy path.

## Failure Handling

- Missing M2 runtime symbols fail only when the M2 spec is selected.
- Metadata or pool geometry mismatches fail during initialization.
- `lookup == None` and `pin_read == None` are normal cache misses.
- `ExistingClaim(valid=False)` is not readable and does not trigger a second
  writer.
- `NoSpace` skips the store without changing the serving result.
- `Unavailable`, a failed DMA, a publish failure, or an impossible component
  mapping fails the transfer loudly; it is not converted into a cache hit.
- Cleanup never unlinks a shared directory or pool while another live instance
  may use it. Forced unlink is limited to controlled test setup/teardown after
  all participants synchronize.

## Testing and Acceptance

### Mock-safe tests

- factory resolution by `spec_name` without `spec_module_path`;
- importing the plugin when `SharedMetadata` is absent;
- M1 factory resolution and worker construction unchanged;
- deterministic key and compatibility encoding;
- shared-manager admission remains upstream-controlled;
- local eviction calls `SharedMetadata.evict` for the exact versioned entry;
- reserved, stale, and missing entries return a normal miss;
- a successful lookup retains its read pin through load completion;
- publish occurs only after every component transfer succeeds;
- claim rollback covers `ExistingClaim`, `NoSpace`, `Unavailable`, and failed
  transfers.

### Spyre-gated connector test

Store a known-pattern complete KV block through the shared worker, publish it,
reload it into fresh KV pages, and compare every K/V component byte-for-byte.
A separately injected directory miss verifies that the connector schedules
normal model recomputation and produces the expected output without reading a
slot.

### Two-instance functional test

Run two isolated TP1 vLLM instances on distinct available Spyre devices with the
same model, deterministic hash settings, metadata name, and pool-family
configuration. Prefix caching is explicitly disabled on both instances. The
initial request on A completes and releases its device KV blocks before either
reload is measured, so neither result can be satisfied by a device-resident
prefix cache:

1. Instance A computes, stores, and publishes a prompt block.
2. Instance A receives the same prompt again and records a shared host-tier
   self-hit; record the reload duration and bytes per second.
3. Instance B receives the same prompt as its first request and records a
   shared host-tier hit.
4. Instrumentation confirms both reloads perform device-from-host DMA from the
   shared pool and do not use device-resident prefix KV, recompute the block, or
   read it from disk.
5. B's reloaded KV bytes are identical to A's published bytes.
6. With `temperature=0`, B's generated token IDs and text are identical to a
   no-cache baseline.
7. Record B's shared-pool reload duration and bytes per second alongside A's
   self-reload measurement. No M1 comparison or performance threshold is
   required in this milestone.

Spyre-backed commands run serially except for the deliberate two-instance
acceptance topology, which assigns one device to each process. Two commands
must never contend for the same accelerator.

The existing M1 KV-offload test suite is rerun unchanged as the regression
gate.

## Conditional M2-F3 Work

Before changing Flex, inspect `get_composite_address(...).num_chunks` for every
real KV allocation used by the connector test. If all are single-chunk, M2-F3
is not required for this milestone.

If a required KV page is multi-chunk or the connector test reaches Flex's
`MultiChunkNotSupported` error, extend `copyRaw` on the Flex
`kvc-offload-m2` branch to pack chunks contiguously into a slot and reload them
in recorded domain order. Add cross-process multi-chunk and single-chunk
regression tests in Flex and torch-spyre before resuming spyre-inference. The
public `copy_tensor_raw(dev_tensor, pool, slot_id, to_device)` shape and
hardware-runtime ownership of `CompositeAddress.total_size()` remain
unchanged.

## Completion Criteria

The milestone is complete when the shared spec resolves lazily, the M1 path is
unchanged, connector misses recompute, the connector-level shared round trip is
byte-exact, the two-instance peer hit produces baseline-identical deterministic
output with prefix caching disabled, and both A's self-reload timing and B's
peer-reload timing are recorded. Lower-layer race-correctness remains
demonstrated by the existing Flex and torch-spyre M2 tests rather than
duplicated here.
