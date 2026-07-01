# Shared KV Memory Pool Across spyre-inference Instances — Design

**Status:** Design (local, not for publication)
**Date:** 2026-06-30
**Scope:** `spyre_inference/v1/kv_offload/`; a new flex DMA method (`RuntimeStream::copyRaw`);
two custodial torch-spyre accessors; and our own flex-only shared-memory pool extension.
**Relationship to prior work:** Supersedes `docs/architecture/rfcs/shared-cpu-kv-pool-design.md`
(2026-06-26). See §2 for what changed and why.

---

## 1. Goal

Today each spyre-inference instance owns a private host KV-offload pool: a Python list
of per-block `torch.empty(..., device="cpu")` tensors allocated in
`kv_adapter.py:174`. It is anonymous heap memory, invisible to other processes. Two
instances on one host cannot reuse each other's offloaded KV.

**Goal:** replace that per-process pool with a **host-scoped shared memory pool** that
multiple instances (same model, for now) share, so KV offloaded by one instance can be
found and reloaded by another. Build it in stages, each independently valuable and
leaving the system correct.

**Preliminary goal** = a shared pool across instances of the same model (Stages A+B).
**Ultimate goal** = multiple models sharing one pool via partitions (Stage D).

The device↔host **copy** is built **in flex from the start** (a new
`RuntimeStream::copyRaw` method) — there is no separate "move the copy into flex" stage.
The **host pool** starts in our own extension (Stage A) and **moves into flex in Stage C**,
exposed to Python through torch-spyre's existing binding. So the two flex moves are
sequenced deliberately: the copy is flex-native immediately; the pool follows once the
A+B design is proven.

### Non-goals (for the preliminary goal)
- Multi-host / networked sharing (same-host only).
- Lock-free / sharded-lock metadata (single global lock for now; §8).
- Multi-model partitions (Stage D).
- ARC / exact-LRU eviction ("other cache algorithms"; §7.4).

---

## Phase summary (at a glance)

*Preliminary goal = Phases A + B. Ultimate goal = Phase D. Detail follows in §5–§13.*

### Phase 0 — Prerequisites
**Allow customized changes in flex and torch-spyre to be picked up by spyre-inference.
(Setup only — no feature code here; the new code lands in Phase A.)**
- Set up building + installing flex from the local checkout (the `copyRaw` method is added
  later, in Phase A).
- Set up rebuilding torch-spyre against that local flex (the handle accessor is added
  later, in Phase A).
- Run tests with `uv run --no-sync` so the local builds aren't overwritten by the pinned
  upstream versions.

### Phase A — Independent memory pool, with flex-based DMA KV in/out
**Build a process-local CPU memory pool that holds each KV block's pages in one contiguous
region, and copy KV in/out of it with a raw flex DMA. (Single instance; not shared yet.)**
- **Add `copyRaw` to flex** — a simple raw (no layout conversion) device↔host DMA.
  *(introduced here)*
- **Add one small torch-spyre accessor** that hands back a device tensor's flex address.
  *(introduced here)*
- Build our `SharedKvPool` extension — a **regular (process-local) CPU memory pool** +
  `copy_out`/`copy_in`, with **aggregation**: one slot holds a whole KV block's pages (all
  K+V across all layers) laid out **contiguously**, instead of scattered per-page buffers.
- Expose it to Python (pybind) — `create_or_attach`, `slot_count`, `copy_out`/`copy_in` —
  so spyre-inference can call it.
- Wire it into the offload handler: replace the old per-process tensors; per block, issue
  one DMA per device page into/out of consecutive sub-offsets of that block's slot.
- Test:
  - Offload a block, reload it into fresh tensors, confirm the bytes match.
  - End-to-end correctness: confirm generated output is identical **with vs. without**
    offload/reload in the loop.

### Phase B — Shared memory pool with cross-instance support
**Make the pool shared across vLLM instances: swap the process-local backing for shared
memory, and add a shared "which block lives in which slot" directory.**
- **Replace the regular CPU memory pool with a shared-memory pool** (`shm_open`+`mmap`,
  named segment, attach refcount) — same slot/aggregation layout as Phase A.
- Build a shared-memory hash directory (block hash → pool slot), plus a slot allocation and
  eviction scheme.
- Wrap it as a `SharedOffloadingManager` that plugs into vLLM in place of the default
  manager (no vLLM changes).
- Guard correctness: a global lock.
- Test:
  - Two instances — one stores a block, the other finds and reloads it.
  - Real-use-case scale: many blocks offloaded/reloaded (exercising eviction and slot
    reuse), with correctness checks throughout.

### Phase C — Pool moves into flex
**Move the host pool itself into flex and expose it to Python through torch-spyre, so
spyre-inference uses flex's pool directly. (Ownership move, no new capability.)**
- flex: add a `SharedHostKvPool` class (create/attach shm, map, manage slots), built on
  flex's existing shared-memory machinery.
- torch-spyre: add thin `_C` bindings — create/attach, map, query slots, copy in/out.
- spyre-inference: call the flex pool through torch-spyre; retire our Phase-A extension.
- Re-run all A+B tests unchanged (the slot interface stays the same).

### Phase D — Multi-model
**Let different models share the same pool, each in its own partition (enable both in
torch-spyre and flex).**
- Split the pool into per-model partitions, each with its own slot size and directory.
- Add a layout check so blocks from different models can't be mixed up.

---

## 2. Relationship to the prior RFC (what changed and why)

The 2026-06-26 RFC described a valid alternative. This design **converges** with it on
the copy primitive (raw `dci=null`) but **diverges** on coordination, correctness
justification, and allocation — each for a concrete reason surfaced while tracing
flex/torch-spyre:

| Topic | Prior RFC | This design | Why |
|---|---|---|---|
| Coordination | **MP-server daemon**; clients RPC prepare/commit | **Daemon-less peer**: instances map a shared index in SHM, coordinate via one robust process-shared mutex | The RFC itself lists the peer model as its deferred "next step" (§12). We target it directly; a single coarse lock on a tiny metadata region is enough for the preliminary goal. |
| Copy primitive | `copy_raw_tensor`, **`nullptr` DCI**, host stores device-tiled bytes | **same — raw `dci=null` snapshot/restore** | *Converged.* We initially considered a dci-canonical copy (to guard against the H2D destination being a different tensor than the D2H source), then **proved from source** that same-`(shape,dtype)` tensors have identical layout (§6.2), so the raw copy is correct and simpler. |
| Cross-instance correctness | runtime **layout fingerprint** → MISS on mismatch | **source-proven layout-equality invariant** (§6.2) + runtime guards (§6.4) | For one model, every page is the same `(shape,dtype)` ⇒ identical layout by the proof; no per-access fingerprint needed. The fingerprint becomes relevant only when shapes genuinely differ → **deferred to Stage D** (multi-model), not rejected. |
| Index / allocation | server-side `AddressManager` (first-fit + coalesce) | **fixed-size slabs** + **open-addressed hash directory** + scan-free eviction | KV blocks are fixed-size; a variable-size allocator (split/coalesce/free-list) is unnecessary. Fixed slabs make allocation O(1) and the directory pointer-free for SHM. |

The prior RFC's strong parts are retained in spirit: write-once-then-publish (here the
RESERVED→VALID gate), pin-across-use (refcounts), and the staged "carry the design
forward unchanged" discipline.

---

## 3. Verified facts the design rests on

Established from code; load-bearing.

- **flex DMA takes a raw host pointer; no registration needed.** `DmaParams.hmva` is a
  bare `void*`; IOMMU mapping happens at submit time over any host VA
  (`flex/.../runtime_scheduler.hpp:70`, `iommu_mapper.hpp:39`). A POSIX
  `shm_open`+`mmap` buffer is a valid DMA endpoint as-is.
- **`dci == nullptr` ⇒ straight byte copy; non-null ⇒ layout/format conversion**
  (`runtime_scheduler.hpp:62-68`). `dma_size` is always the full device allocation,
  flex-computed (`:81-85`).
- **The dci converts device-tiled bytes ⇄ a canonical contiguous host image**, derived
  from logical shape/dtype + the device tensor's layout (`generate_dci`,
  `spyre_mem.cpp:363-433`). This is what `copy_tensor` uses (`spyre_stream.cpp:168`).
- **A device `CompositeAddress` is one chunk for KV pages** (could be multi-chunk only
  under an `Interleave` allocation, which torch-spyre never uses for tensors). Allocation
  goes through `Bind`/single-domain (`spyre_allocator.cpp:120`) → exactly one chunk
  (`flex_allocator.cpp:196-203`); chunk count is set by policy, not fragmentation. So a
  raw single-span copy is correct — *proven* in §6.2, guarded in §6.4.
- **flex is per-process** (`RuntimeContext` is a one-per-device singleton,
  `runtime_context.hpp:141`); it has **no host-memory allocation/registration API**.
  So a *cross-process* shared pool cannot be a flex-owned object — sharing is external
  (POSIX SHM), owned by our extension. flex's role is the **DMA only** (it consumes a host
  pointer at submit time); it never allocates or owns the host pool.
- **The KV cache is a paged list, not one tensor.** Each layer is a
  `SpyrePagedKVCache(k_pages, v_pages)` of per-block tensors
  `[num_kv_heads, block_size, head_size]` fp16 (`spyre_model_runner.py:421-442`). K and
  V are independent allocations; a logical block fans out to `2 × num_layers` pages.
- **The offload manager runs scheduler-side; bytes move worker-side.** vLLM's
  `OffloadingConnectorScheduler` calls `spec.get_manager()` once (`scheduler.py:128`)
  and drives it through an 8-method interface (`scheduler.py:157…592`,
  `cpu/manager.py:87-201`). The byte copies happen in the worker handler
  (`handlers.py:166`). The two sides communicate only by integer `block_id`.
- **vLLM block key sizes.** `OffloadKey` = `block_hash + 4-byte group_id`
  (`abstract.py:42`). `block_hash` is **32 B** for SHA-256 or **16 B** for xxh3-128
  (`hashing.py:41,62`; `NONE_HASH = os.urandom(32)`). So the full key is **36 B or
  20 B**, config-dependent.
- **Page bytes are already a 4 KiB multiple.** `num_kv_heads × block_size × head_size ×
  2`, with `block_size=128` and `head_size` a multiple of 64, is always ≥16 KiB and
  divisible by 4096. The 4 KiB DMA-alignment requirement is satisfied with zero padding
  (assert at pool creation).

---

## 4. Prerequisites — test environment with a custom flex build

Stage A adds a method to **flex** (`RuntimeStream::copyRaw`) and two accessors to
**torch-spyre**, so the test environment must build *both* sibling C++ repos from local
source and make `spyre-inference` use them — not the pinned upstream revisions. The chain
is three layers deep; each must be rebuilt in order.

**Layout (sibling checkouts, verified paths):**
```
workspace/
  flex/             # the runtime; we add copyRaw here
  torch-spyre/      # links libflex.so; we add the accessors here
  spyre-inference/  # pins torch-spyre by git rev — must be overridden locally
```

**1. Build & install flex from the local checkout.** flex uses CMake presets
(`flex/CMakePresets.json`); build a release preset and install to a dir torch-spyre will
read via `RUNTIME_INSTALL_DIR`:
```bash
cd workspace/flex
cmake --preset release          # or ninja-release; see CMakePresets.json
cmake --build --preset release
cmake --install build/release --prefix "$RUNTIME_INSTALL_DIR"   # produces lib/libflex.so + include/
```
torch-spyre's build reads flex headers from `RUNTIME_INSTALL_DIR/include` and links
`RUNTIME_INSTALL_DIR/lib/libflex.so` (`setup.py:100-115`), plus common headers from
`SEN_COMMON_HEADERS` (defaults to `../flex`, `setup.py:24`). Export both before building
torch-spyre.

**2. Build & install torch-spyre from the local checkout** (against the flex just built).
Requires `sendnn` available (the wheel build is C++-heavy, ~50 s):
```bash
cd workspace/spyre-inference
RUNTIME_INSTALL_DIR=... SEN_COMMON_HEADERS=../flex \
  uv pip install --no-deps --force-reinstall ../torch-spyre
```
`--no-deps --force-reinstall` installs the *local source*, not the pinned git rev.

**3. Use it from spyre-inference WITHOUT re-syncing.** `spyre-inference/pyproject.toml`
pins torch-spyre to an upstream git rev (`tool.uv.sources`, `pyproject.toml:101`). A plain
`uv run …` re-syncs deps and **silently reinstalls that upstream rev over your local
build** (look for `Uninstalled 1 package … Installed 1 package …` in pytest output — the
CLAUDE.md caveat). **Always use `uv run --no-sync …`** for any iteration that depends on
the hand-built flex/torch-spyre:
```bash
uv run --no-sync pytest tests/v1/kv_offload/...
```

**4. Iteration loop.** Edits to flex → rebuild+install flex (step 1) → rebuild+install
torch-spyre (step 2, because it links `libflex.so`) → `uv run --no-sync` tests (step 3).
Editing flex alone is **not** enough — torch-spyre must be reinstalled so it links the new
`libflex.so`. Batch C++ edits before each rebuild (each torch-spyre wheel build is ~50 s).

**5. Spyre-hardware gating.** The copy/round-trip tests (§6.7) need real Spyre hardware
(`requires_spyre`) and skip silently on CPU-only hosts — "all green" on a non-Spyre box
does *not* mean the path works. CPU-only tests (pool shm lifecycle, slot math, directory
unit tests) run anywhere. **Single accelerator:** never run two Spyre-backed commands
concurrently (no `pytest -n`, no parallel `uv run`) — it hangs or corrupts device state.

**Open items (fill in with concrete values during setup):** the exact `RUNTIME_INSTALL_DIR`
path, the chosen flex preset, and whether the CI image already ships a prebuilt flex (in
which case step 1 is "rebuild only flex, reuse the rest"). These are environment-specific
and recorded in the Stage-A implementation plan, not here.

---

## 5. Staging overview

```
Stage A  Local CPU pool (our extension, per-block slots) + flex copyRaw DMA  (single-instance)
Stage B  Shared-memory backing + METADATA (hash→slot directory)              (cross-instance safe; preliminary goal met)
Stage C  Pool moves into flex, exposed via torch-spyre                       (no new capability; ownership move)
Stage D  Multi-model partitions                                             (ultimate goal)
```

The seam that holds across all stages: **the data half and the metadata half
communicate only through an integer `slot` (== vLLM `block_id`).** The metadata never
touches bytes; the data half never touches hashes. This lets Stage A ship before B
exists.

**Layer ownership (fixed across all stages — verified from code).** flex has no Python
and never sees torch tensors. The dependency arrows point **strictly downward**, and the
torch-spyre↔flex translation is isolated to the plugin handler — so `SharedKvPool` is
**flex-only** and carries no torch-spyre reference into its eventual flex home.

| Layer | Owns | Stage-A change |
|---|---|---|
| **plugin handler** (`handlers.py`, Python — the translation boundary) | extracts pure flex objects from a device tensor: `tensor → CompositeAddress*` and `device → RuntimeStream*`, then calls the pool with **flex types only**. This is the *only* place torch-spyre and flex meet. It is plugin code and never moves into flex. | route copies through the pool |
| **`SharedKvPool`** (our extension — **flex-only**) | the pool (Stage A: local CPU buffer; Stage B: shm) + `copy_out(slot, pages, stream)` / `copy_in(...)`, which call `copyRaw` per page into a block's contiguous slot. No torch-spyre header, no torch-spyre symbol. | new module |
| **flex** (C++ runtime) | the DMA — new public `RuntimeStream::copyRaw(host_ptr, composite_addr, to_device)` (`dci=null` → `launchOperationD2H/H2D`). The `CompositeAddress` and `RuntimeStream` the handler passes are flex's own objects. | add `copyRaw` |
| **torch-spyre** (the flex↔Python custodian) | exposes two **custodial** accessors used *by the handler* (not the pool): `get_composite_address(tensor)` → non-owning ptr to the device handle (in the private `SharedOwnerCtx`, `spyre_allocator.h:28`); `getDefaultStreamRuntimeHandle(device)` → a **pooled** flex stream (`spyre_stream.cpp:179,293`; streams are `GlobalRuntime`-owned, never minted by us). | add the handle accessor |

Python (vLLM/`handlers.py`) always calls **our pool**; flex is never called from Python
directly. **Why the pool is flex-only:** both objects the handler hands it
(`CompositeAddress`, `RuntimeStream`) are *flex* objects merely reachable via torch-spyre
today — they are not torch-spyre logic. The `SharedKvPool` itself includes no torch-spyre
header, so it is a clean flex-only component.

**On the torch-spyre dependency (stated precisely, not over-claimed).** The
`device tensor → CompositeAddress` lookup is **irreducible** while the device KV pages are
torch tensors (which they are, `SpyrePagedKVCache`): flex has no tensor concept, and the
tensor↔handle binding lives in torch-spyre's `SharedOwnerCtx`. So *some* torch-spyre getter
is needed in every stage — it lives in the **handler** (plugin code), never in the pool.
It would only disappear if the device KV cache itself were flex-allocated *and registered
with a flex-side handle table* — a separate, larger change **out of scope here**. The win
of the flex-only pool is therefore narrower but real: the **pool** migrates into flex
cleanly; the handler's per-copy handle extraction stays in the plugin regardless.

---

## 6. Stage A — Independent CPU pool + raw (snapshot/restore) copy

> **Stage A correctness invariant (load-bearing).**
> Two device tensors with the same `(shape, dtype)` have an **identical on-card memory
> layout** — identical tiling, identical `total_size()`, single contiguous chunk.
> Therefore a verbatim byte image of one KV page, snapshotted to host, can be restored
> into any other same-`(shape, dtype)` page and reconstitutes the original values.
> This is **proven from torch-spyre/flex source** (inline below), **guarded at runtime**
> (§6.4), and holds while KV tensors allocate under flex's `Bind` policy (re-open
> condition, §6.4). All instances run the same model, so every KV page shares one
> `(shape, dtype)` and the invariant holds by construction within a pool.

**Stage A is single-instance and not shared.** The pool is a **regular, process-local CPU
buffer** here — no POSIX shm, no cross-process attach. Shared-memory backing is introduced
in **Stage B** (§7), which swaps the backing while keeping the same slot/aggregation layout.
Stage A's job is just: get one big pooled buffer + the flex DMA path working correctly.

### 6.1 What it is — the extension (`SharedKvPool`)
Our own C++ component compiled to a pybind11 module (the shape of `torch_spyre._C`, but
ours). Two responsibilities — **own the data pool** and **issue the raw DMA** — and it is
**flex-only** (links flex's public headers; no torch-spyre header; knows nothing of hashes,
eviction, or vLLM).

```cpp
// flex-only: takes pure flex types, never an at::Tensor.
class SharedKvPool {
  SharedKvPool(num_slots, slot_bytes);         // Stage A: one process-local CPU buffer (num_slots × slot_bytes),
                                               // aligned to GetIovaAlignment(). (Stage B: shm_open+mmap+refcount.)
  ~SharedKvPool();
  size_t slot_count() const;
  // A slot holds a whole KV block = all its pages, contiguous (§6.1 aggregation).
  // `pages` are the block's device pages in fixed order; each is DMA'd to/from its sub-offset.
  void copy_out(uint64_t slot, span<DevicePage> pages, flex::RuntimeStream* stream);  // D2H (snapshot)
  void copy_in (uint64_t slot, span<DevicePage> pages, flex::RuntimeStream* stream);  // H2D (restore)
  // DevicePage = { const flex::CompositeAddress* composite; }  — one device page's handle.
};
```

- **Backing (Stage A):** one **regular CPU buffer** of `num_slots × slot_bytes`, allocated
  once and aligned to `GetIovaAlignment()` (§6.4). Slot `i` starts at `i × slot_bytes`.
  *(Stage B replaces this with a named shm segment; everything else stays.)*
- **Aggregation — one slot = one whole KV block.** A logical KV block is *not* one tensor:
  it fans out to `2 × num_layers` device pages (K and V per layer; `kv_adapter.py:64`,
  `spyre_model_runner.py:421`). A slot holds **all of them, contiguous**, in fixed
  `[layer][K,V]` order. So `slot_bytes = round_up(2 × num_layers × page_total_size, 4096)`
  (§6.3). This replaces today's scattered per-page `torch.empty` buffers with one
  contiguous region per block — simpler residency, one slot per block for the Stage-B
  directory to key on.
- **Per-block copy = N DMAs into one slot.** The device pages remain separate allocations
  (each its own `CompositeAddress`), so `copy_out`/`copy_in` issue **one `copyRaw` DMA per
  page**, each targeting that page's **sub-offset within the block's slot**
  (`slot_base + page_index × page_total_size`). Aggregation makes the *host* side one
  contiguous region; it does not fuse the device side into a single DMA.
- **Format-agnostic & data-only:** a slot is opaque bytes; the pool stores device-tiled
  images verbatim. It does **not** choose slots (Stage B), do hash lookup/eviction
  (Stage B), or touch tensors (the handler does the translation, §6.2).
- **Replaces** both `_alloc_host_pages`/`kv_adapter.py:174` (per-block host pages → one
  pool slot) and the `SpyreKvDmaCopier`/`copier.py` wrapper.

### 6.2 Copy primitive: raw `dci=null` DMA (snapshot / restore)
The copy is a **verbatim byte transfer of the device page's tiled memory** — flex's
`dci == null` "straight byte copy" mode (`runtime_scheduler.hpp:68`). **Tiling is not
converted; it is preserved by copying the tiled bytes as-is.** The host slot holds a
byte-for-byte *snapshot* of the device-tiled image; H2D *restores* that snapshot into a
device page. We never interpret the bytes — no `SpyreTensorLayout`, no `generate_dci`.

**Why recovery into a different tensor works.** The D2H source and the H2D destination
are the same `(shape, dtype)` but **not the same object** (the device page is
reused/reallocated between offload and reload). A raw restore is nonetheless correct
because of the §6 invariant: both pages tile identically, so byte *i* of the snapshot is
already the correct byte *i* of the destination's layout — no remapping, no de-tiling.
It is a disk-image copy between two identically-formatted disks: we don't parse the
filesystem (the tiling), we copy raw blocks, and identical format on both ends makes the
destination immediately valid.

**Proof the invariant holds (from source).**
1. **Tiling is a pure function of `(shape, dtype)`.** `SpyreTensorLayout::init(host_size,
   dtype)` (`spyre_tensor_impl.cpp:124`) derives `device_size`/`stride_map` from shape +
   dtype + compile-time stick tables only — no address, core count, handle, or runtime
   state. ⟹ same `(shape, dtype)` ⇒ byte-identical tiling.
2. **Single chunk, identical size.** Every device tensor allocates via
   `AllocationDirective(Bind, {0})` (`spyre_allocator.cpp:120`), and the `Bind` path
   produces exactly **one** chunk (`flex_allocator.cpp:196-203`). Chunk count is set by
   the *policy*, not by fragmentation. A `Bind` allocation either fits one chunk in one
   region or throws OOM (`flex_allocator.cpp:600`) — it never silently becomes
   multi-chunk. KV pages (KB–MB) are far below a region (`DEFAULT_1P0_MAX_REGIONS=7`,
   regions are multi-GB), so they are reliably single-chunk. ⟹ A and B have identical
   `total_size()` and chunk geometry.
3. **DMA is offset-preserving; multi-chunk is hard-rejected.** A `dci=null` DMA copies
   one contiguous span, host byte *i* ↔ device byte *i* (`pf_runtime_scheduler.cpp:360`);
   a multi-chunk address *throws* `MultiChunkDmaNotSupported` before any copy
   (`pf_runtime_scheduler.cpp:302`) — never silent corruption.

Together: raw snapshot of A → restore into same-`(shape,dtype)` B is **provably correct**
for current torch-spyre/flex.

**The `dci=null` branch is already live.** torch-spyre's program loader uses exactly this
path today — `copyProgramAsync → copyAsyncImpl(..., dci=nullptr, ...)`
(`spyre_stream.cpp:136-140,190`) → flex `launchOperationH2D/D2H`. (Cited only as evidence
the branch is real and exercised; it is H2D-only and program-specific, **not** a reuse
target for KV.)

**Where the DMA lives — `RuntimeStream::copyRaw` (NEW, in flex).** The copy body lives in
flex from the start (no later "move into flex" stage). `copyRaw` does **not** exist today;
it is a thin public wrapper we add next to `launchOperationD2H/H2D` (`runtime_stream.hpp`):

```cpp
// NEW in flex — names the (host_ptr, device_addr, direction) raw-copy pattern.
void RuntimeStream::copyRaw(void* host_addr,
                            const CompositeAddress* dev, bool to_device) {
  DmaParams p(host_addr, to_device, dev);     // dci=nullptr default; dma_size = dev->total_size()
  if (to_device) launchOperationH2D(&p);      // runtime_stream.hpp:99
  else           launchOperationD2H(&p);      // runtime_stream.hpp:115
}
```
It adds no new capability — it is the exact `DmaParams` + `launchOperation` body that
`copyAsyncImpl` runs today (`spyre_stream.cpp:194-209`), just named and public.

**The pool is flex-only; the handler does the translation.** `SharedKvPool` takes **pure
flex types** — it never includes a torch-spyre header. For Stage A (kept deliberately
simple), the plugin handler extracts the `CompositeAddress` and pooled `RuntimeStream`
**per copy** from the device tensor it already holds, and passes them in — no caching, no
handle table, no registration step. This keeps the pool's dependencies pointing down into
flex (the handler's torch-spyre lookup is irreducible while KV pages are torch tensors; §4).

```cpp
// SharedKvPool — FLEX-ONLY. No torch-spyre symbol appears here.
// A slot holds a whole block: pages packed contiguously at consecutive sub-offsets.
void SharedKvPool::copy_out(uint64_t slot, span<DevicePage> pages,
                            flex::RuntimeStream* stream) {              // D2H (snapshot)
  uint8_t* slot_base = pool_base_ + slot * slot_bytes_;
  size_t off = 0;
  for (auto& p : pages) {                                              // 2×num_layers pages
    TORCH_CHECK(p.composite->is_single_chunk());                       // guard (§6.4)
    stream->copyRaw(slot_base + off, p.composite, /*to_device=*/false);// one DMA per page → sub-offset
    off += p.composite->total_size();
  }
  stream->synchronize();                                               // whole block in slot AFTER this (§6.4)
}
void SharedKvPool::copy_in(uint64_t slot, span<DevicePage> pages,
                           flex::RuntimeStream* stream) {              // H2D (restore) — mirror
  uint8_t* slot_base = pool_base_ + slot * slot_bytes_;
  size_t off = 0;
  for (auto& p : pages) {
    TORCH_CHECK(p.composite->is_single_chunk());
    stream->copyRaw(slot_base + off, p.composite, /*to_device=*/true);
    off += p.composite->total_size();
  }
  stream->synchronize();
}
```
Pages are visited in a **fixed order** (`[layer][K,V]`) so the same block always packs
identically — the sub-offset of page *k* is the running sum of prior pages' `total_size()`.
A single `synchronize()` covers the whole block's DMAs.

```python
# handlers.py — the torch-spyre↔flex boundary (plugin code; never moves into flex).
# Per block: gather its 2×num_layers device pages (fixed [layer][K,V] order) as flex handles.
pages  = [DevicePage(torch_spyre.get_composite_address(p)) for p in block_device_pages]
stream = torch_spyre.get_dma_stream(device)                  # pooled flex stream (never minted)
pool.copy_out(slot, pages, stream)                           # pool sees ONLY flex types
```

The pool does **not** mint its own stream — streams are `GlobalRuntime`-owned and drawn
from torch-spyre's `StreamPool` (`spyre_stream.cpp:256-258`); minting an unmanaged stream
would violate the single-accelerator rule. The handler fetches a pooled one and hands it
down.

**End-to-end path, both directions (down to the bytes).** `launchOperation…` is **async**
(enqueue only); `synchronize()` is the completion barrier (`runtime_stream.cpp:193`):
```
D2H (offload):  handler → gather block's pages + stream  [torch-spyre boundary]
                        → pool.copy_out(slot, pages, stream)   [pool: flex-only]
  → for each page:  stream->copyRaw(slot_base+off, page, to_device=false)  → launchOperationD2H
       → submitDmaChunk (pf_runtime_scheduler.cpp:383):
            • assert single_chunk; if host_addr not IOVA-aligned → 4MiB shadow buffer + memcpy
            • iommu_mapper->Map(host_addr)  (lazy, no pre-pin; scheduler_config.cpp:80)
            • submitToHardware: DMA moves page total_size() bytes device → host, offset-preserving
            • completion cb (ResponseWorker): if shadowed, memcpy shadow→slot (pf_…:434); --in_flight
  → stream->synchronize() ⇒ the whole block's pages now sit contiguously in the slot
H2D (reload):   handler → pool.copy_in(slot, pages, stream)   [mirror]
  → for each page:  stream->copyRaw(slot_base+off, page, to_device=true)  → launchOperationH2D → DMA host → device
       (if shadowed, memcpy slot→shadow BEFORE the DMA, scheduler_config.cpp:71)
  → synchronize() ⇒ each device page reconstituted (same-shape ⇒ identical tiling ⇒ correct)
```

**`synchronize()` is the cross-instance ordering point.** A store must `synchronize()`
*before* its slot is published readable (Stage B: the RESERVED→VALID flip happens after
`copy_out` returns). A load must `synchronize()` before attention reads the device page.

### 6.3 Copy length and slot size — the `total_size()` rule
The copy length is the device page's **padded physical size**,
`CompositeAddress::total_size()` (`alloc_address.hpp:180`) — **not** the logical
`numel × itemsize`. `total_size()` includes two padding layers:
- **stick/tiling padding** — `get_device_size_in_bytes` rounds the stickified dim up to a
  full stick (`spyre_tensor_impl.cpp:286`); this can be substantial (e.g. a `head_size`
  that isn't a whole number of sticks rounds up); and
- **128-byte allocation alignment** — flex rounds up to `DEVICE_ALIGNMENT`
  (`flex_allocator.cpp:737`; torch-spyre notes this at `spyre_allocator.cpp:145`).

**Per-page copy length = that page's `total_size()` exactly.** Copying only the logical
size would drop the stick-padding tail and leave part of the destination's tiled layout
unwritten → corrupt reload.

**Slot size = the whole block (aggregation, §6.1).** A slot holds all `2 × num_layers`
pages of a block contiguously, so
`slot_bytes = round_up(Σ page_total_size, 4096) = round_up(2 × num_layers × page_total_size, 4096)`
(pages are uniform for one model). Page *k* sits at sub-offset `Σ_{j<k} page_total_size`.
Since each page's `total_size()` is already ≥16 KiB-class and 128-aligned, only the final
4 KiB rounding of the whole slot may add a few padding bytes.

### 6.4 Guard conditions (make a future regression loud, not silent)
The §6 invariant is proven for *current* torch-spyre/flex; these asserts turn any future
drift into a clear error instead of silent KV corruption:
1. **`composite->is_single_chunk()`** at every copy — catches a future multi-chunk
   (`Interleave`) allocation. (flex's DMA already throws on multi-chunk,
   `pf_runtime_scheduler.cpp:302`; we assert earlier for a clear message.)
2. **Copy exactly `total_size()` bytes** — never `numel × itemsize`.
3. **Uniform `(shape, dtype)`** across a pool's slots — holds by construction for one
   model.

**Performance/correctness requirements surfaced by the DMA path (§6.2):**
4. **Align the pool to `GetIovaAlignment()`** (local CPU buffer in Stage A, shm in Stage B).
   If a host slot pointer is not
   IOVA-aligned, flex silently routes the transfer through a 4 MiB aligned shadow buffer
   with an extra `memcpy` per chunk (`scheduler_config.cpp:61-74`,
   `pf_runtime_scheduler.cpp:434`) — correct but slower. Align the pool base + slot stride
   to the **queried** `GetIovaAlignment()` (page-class), not an assumed 4 KiB, to keep the
   DMA targeting our slot directly.
5. **`synchronize()` before publish/read.** D2H must complete before the slot is published
   readable (Stage B RESERVED→VALID); H2D must complete before attention reads the device
   page. The DMA is async; `synchronize()` (or a per-op callback) is the barrier.
6. **Pooled stream only.** The DMA runs on a stream drawn from torch-spyre's `StreamPool`
   (default, or a pooled dedicated one), never a stream our extension mints — streams are
   `GlobalRuntime`-owned (`spyre_stream.cpp:256-258`).

**Re-open condition:** if KV tensors ever allocate with `PlacementPolicy::Interleave`
(multi-chunk), the single-chunk invariant breaks and the copy must be revisited (a
converting/`dci` path, or per-chunk copies). No such path exists today
(`spyre_allocator.cpp:120` hard-codes `Bind`).

### 6.5 Eviction
None of ours. Stage A keeps vLLM's private `CPUOffloadingManager` (unchanged), so it
**inherits upstream LRU** (`OrderedDict`, `lru.py`). Stage A only swaps the byte
substrate; allocation/eviction policy is untouched.

### 6.6 What's in flex vs. our extension (Stage A) — and what moves later
The copy *body* is in flex from the start (`copyRaw`, §6.2). The **host pool** evolves:
**Stage A** = a regular process-local CPU buffer in our extension (slot addressing only,
no shm); **Stage B** swaps in shared memory; **Stage C** moves the pool into flex (§10),
exposed via torch-spyre (flex already has the shm/pinned-memory machinery to own it). So
"what's outside flex" is stage-dependent — only the cross-process *sharing* (POSIX shm +
coordination) is inherently external, because flex is per-process. The dedicated-DMA-stream
optimization (a pooled stream isolated from compute) is deferred — Stage A uses the shared
default stream for simplicity.

### 6.7 Gating tests (must pass before trusting the path)
1. **Cross-tensor block round-trip (the invariant in practice):** fill a block's device
   pages (tensors X) with known KV → `copy_out` the whole block to a slot → allocate a
   *different* same-`(shape,dtype)` set of pages (tensors Y) → `copy_in` slot→Y → assert
   Y == X page-by-page. Exercises §6's invariant. (Hardware-gated; `requires_spyre`.)
2. **Aggregation / slot layout:** the block's pages land at the expected contiguous
   sub-offsets; `slot_bytes == round_up(Σ page_total_size, 4096)`; reload reads them back
   from the same offsets.
3. **Single-chunk assertion:** each KV page's `CompositeAddress.is_single_chunk()` holds;
   the guard fires if violated.
4. **Generation correctness:** confirm generated output is identical **with vs. without**
   offload/reload in the loop (not just bytes — the end-to-end model path).
5. **Single-instance E2E parity:** offload/reload through the pool matches the current
   `torch.empty` path.

---

## 7. Stage B — Shared-memory pool + metadata (hash→slot directory)

Stage B turns the single-instance pool into a **shared** one, in two moves:

1. **Swap the backing to shared memory.** Replace Stage A's process-local CPU buffer with
   a POSIX shm segment (`shm_open`+`mmap`, named, attach refcount, last-out unlink) —
   **same slot size and per-block aggregation layout as Stage A** (§6.1). Only the backing
   changes; `copy_out`/`copy_in` and the slot math are unchanged. (This is the
   `bob/`-lineage shared-memory plumbing.)
2. **Add the shared directory** (below) so instances agree on which block lives in which
   slot.

Both are needed for safety: without the shared directory, each instance's private manager
hands out `block_id`s from `[0, num_slots)` independently and would overwrite the same
shared slots.

### 7.1 `SharedOffloadingManager`
A Python class (over SHM) implementing vLLM's 8-method `OffloadingManager` interface,
returned from `SpyreOffloadingSpec.get_manager()` in place of `CPUOffloadingManager`.
The upstream scheduler is untouched; it cannot tell the difference. The `block_id`s it
returns are slot indices into `SharedKvPool`.

Interface mapping (reference behavior: `cpu/manager.py`):
- `lookup(key)` → probe directory; resident iff present, VALID, `ref_cnt ≥ 0`.
- `prepare_store(keys)` → dedup already-present keys; for new keys claim slots (evicting
  if needed), insert RESERVED entries (`ref_cnt = -1`), return slots.
- `prepare_load(keys)` → probe; `ref_cnt += 1` (pin); return slots.
- `complete_load` → `ref_cnt -= 1`. `complete_store` → flip RESERVED→VALID
  (`ref_cnt = 0`) or remove+free on failure.
- `touch` → mark recently used. `take_events` / `shutdown`.

### 7.2 Control segment layout (the shared metadata)
One SHM control segment, separate from the data segment, all fixed-size records at
computed offsets (no pointers). Designed as a **partition descriptor** so Stage D adds
more without re-layout (Stage B instantiates exactly one):

```
Header / partition descriptor[0]:
  magic, version
  num_slots, slot_bytes, key_len        // key_len = 20 (xxh3) or 36 (sha256)
  recency-state (e.g. clock hand)        // see §7.4
  table_capacity                         // ≈ 1.3 × num_slots
  table_off                              // byte offset to the table (self-describing)
  lock           pthread_mutex_t          // PTHREAD_MUTEX_ROBUST, PROCESS_SHARED
  (slot warmup cursor — see §7.5)
Directory table  table[table_capacity] of DirEntry   // the only directory structure
```

### 7.3 `DirEntry` and the hash directory
The directory is an **open-addressed hash table**: entries live directly in the array;
collisions resolved by linear probing. No pointers → lives in SHM cleanly.

```
DirEntry {
  uint8  key[key_len];   // full OffloadKey, stored INLINE  (identity check)
  int64  slot;           // physical slot id  (== upstream block_id)
  int32  ref_cnt;        // -1 reserved / 0 idle / >0 pinned (== upstream BlockStatus.ref_cnt)
  <recency state>        // see §7.4
  uint32 flags;          // EMPTY / VALID / TOMBSTONE
}
```

- **Maps to upstream `BlockStatus` verbatim** (`ref_cnt`, `slot`==`block_id`) plus the
  inlined key (which `OrderedDict` held as the dict key) plus recency state (which
  `OrderedDict` encoded implicitly as insertion order). We split `OrderedDict`'s two
  jobs — hash map + recency order — into the table + recency state.
- **Position hash:** fold high-entropy bits from the **block-hash portion** of the key
  (skip the low-entropy 4-byte group_id) mod `table_capacity`. **Correctness never
  depends on the position hash:** identity is decided by a full-key comparison on probe,
  so truncation/collision can only lengthen a probe, never return a wrong block. (A
  *full-key* collision is vLLM's existing prefix-cache assumption, not ours.)
- **Why ~1.3×:** open addressing stays O(1) only below ~75% load; to index `num_slots`
  resident blocks at ≤75% the table needs ≈`1.33 × num_slots` entries. This inflates
  only the directory (tens of bytes/entry), never the data slots. (e.g. 4 M slots →
  ~5.2 M entries → ~0.3 GB directory vs. 64 GiB data.)
- **Entry-index ≠ slot-id** (forced by the 1.3× headroom: more entries than slots). The
  `slot` lives in the entry body; the two are connected only by that integer.

### 7.4 Eviction (Stage B)
The pool is finite; eviction is required. The committed core is the **hash directory**
(the source of cross-instance value); eviction is **isolated behind one
victim-selection function** and is deliberately **approximate and scan-free** — exact
global LRU is *not* attempted at DRAM scale (a full-table scan per eviction under the
lock is the bottleneck precisely when the pool is full).

**Committed properties:** approximate-LRU, O(1)/O(K), pointer-free, no global scan, no
per-eviction full sweep. **Candidate algorithms:** CLOCK / second-chance (a per-entry
referenced bit + a rotating hand; no timestamps, no RNG) or sampled-LRU (sample K
unpinned entries, evict the oldest). **The specific algorithm and its exact SHM
mechanics are settled in the Stage-B implementation plan.** Pinned entries
(`ref_cnt > 0`) are never evictable, so in-flight cross-instance reads stay safe.

Deferred ("other cache algorithms"): ARC, exact-LRU.

### 7.5 Slot allocation
Fixed-size slabs ⇒ no variable allocator. Steady state (pool full) allocation =
eviction: evict a victim, reuse its slot. A small warmup cursor covers the initial fill
before the pool has been filled once. (No `free_slots[]` array — it would duplicate
state the directory already holds.)

### 7.6 Concurrency & crash safety
- **One global `PTHREAD_MUTEX_ROBUST`, PROCESS_SHARED lock** guards all metadata
  transitions. Data copies happen **outside** the lock (a RESERVED slot is privately
  owned by the storing instance).
- **Any instance may store/load.** It takes the lock, claims/looks-up a slot against the
  shared directory, releases the lock, then DMAs the bytes itself. No allocator process,
  no IPC hand-off — the lock *is* the coordination.
- **RESERVED→VALID gate:** a store inserts RESERVED, copies bytes (unlocked), then flips
  to VALID. Readers only ever see VALID entries → never a half-written block.
- **Crash recovery:** robust mutex ⇒ on `EOWNERDEAD` the next locker runs a recovery
  pass — roll back RESERVED entries owned by the dead writer, free their slots, decrement
  pins it held. Keeps a crashed instance from wedging or leaking the pool.

---

## 8. Deferred: concurrency evolution (not the preliminary goal)
The single global lock is acceptable now but is the known scaling limit at DRAM scale.
Evolution path (documented, not built): keep data reads lock-free (write-once +
refcount), then shard the metadata lock (`key % N`) or move to a lock-free concurrent
hashmap. The §7 directory and protocol are designed to carry over unchanged.

---

## 9. Upstream connection
The entire connection is one line we already own: `SpyreOffloadingSpec.get_manager()`
(`spec.py:109`) returns `SharedOffloadingManager` instead of `CPUOffloadingManager`.
From there the upstream scheduler drives the 8-method interface (`scheduler.py`).
`prepare_store`/`complete_store` — which upstream assumes have a single owner — become
cross-instance-safe by coordinating through the SHM directory under the lock; the method
signatures and return types (`CPULoadStoreSpec` of block-ids) are unchanged. No vLLM
patching.

The block identity remains vLLM's `OffloadKey` (same prefix-cache hash on every
instance of the same model), which is exactly what makes a cross-instance lookup hit.

---

## 10. Stage C — pool moves into flex, exposed via torch-spyre
**Goal:** make the shared host KV pool a first-class **flex** object and expose it to
Python through torch-spyre's existing binding — so spyre-inference uses flex's pool
directly instead of our own extension. The *copy* (`copyRaw`) is already in flex from
Stage A; Stage C moves the *pool* down too. No new capability — an ownership relocation.

**C1 — flex-side shared host-memory pool (C++).**
- A flex class (e.g. `SharedHostKvPool`) that `shm_open`+`mmap`s a named segment, maps it,
  manages fixed 4 KiB-aligned slots (IOVA-aligned base), and refcounts attach/detach.
- Built on flex's **existing** shared-memory / pinned-memory machinery (the `HdmaShm` /
  `PinnedMemoryWrapper` lineage, `flex/.../multi_device/hdma/hdma_shm.hpp`) — reuses, not
  reinvents, flex's host-buffer handling. (This is why a flex-owned host pool is feasible
  despite §3's "no host-alloc API" note: flex already does shm/pinned host memory for
  multi-device coordination; Stage C generalizes it to a KV pool.)
- The Stage-A `copyRaw` now reads/writes slots of this flex-owned pool.

**C2 — Python binding via torch-spyre (no new binding surface anywhere).**
- flex stays pure C++ (it has no pybind today and gains none). torch-spyre's existing
  `_C` module — already the flex↔Python bridge — adds thin wrappers exposing the pool:
  `create_or_attach(name, num_slots, slot_bytes)`, `map()`, `slot_count()` /
  `available_slots()`, `copy_out(slot, device_handle)` / `copy_in(slot, device_handle)`.
- spyre-inference calls these through torch-spyre; our Stage-A `SharedKvPool` extension is
  **retired** (its pool logic → flex C1; its Python surface → torch-spyre C2).

**C3 — what moves, what stays.**
- *Moves into flex:* pool ownership, slot management, create/map/query.
- *Stays:* cross-process **sharing** still rides on POSIX shm (flex is per-process; it owns
  the pool *object* per process, instances coordinate through the shared segment). The
  Stage-B metadata directory and `SharedOffloadingManager` are unchanged — still address
  the pool by integer slot. The per-copy `device tensor → CompositeAddress` handle
  extraction (§6, handler-side) is also unchanged — KV pages are still torch tensors.

**Scope note:** Stage C is strictly downstream of A+B (both must work first). The torch-spyre
role *grows* here (it now also binds the flex pool API) — which is consistent with its job
as the flex↔Python bridge.

---

## 11. Stage D — multi-model partitions
Multiple models share one pool via non-overlapping **partitions**, each a region with
its own `slot_bytes`/format, its own directory, its own free/recency state — i.e. its
own partition descriptor (§7.2). Stage B builds one descriptor; Stage D allocates N in
the same segment. "Make the descriptor array longer," not a re-layout.

---

## 12. Execution plan

### Stage A — independent CPU pool (per-block slots) + flex `copyRaw` DMA
1. **Spike (de-risk first):** add a minimal `RuntimeStream::copyRaw` in flex (local
   checkout) + a temporary harness; run gating test §6.7(1) — snapshot a block's device
   pages, restore into a *different* same-`(shape,dtype)` set, assert equal — on hardware.
   *Gate: bit-exact recovery into different tensors.* (Confirms the §6.2 proof in practice.)
2. **flex:** finalize the public `RuntimeStream::copyRaw(host_ptr, composite_addr,
   to_device)` (§6.2). Build/install flex from the local checkout.
3. **torch-spyre (boundary accessors, used by the handler — not the pool):** add
   `get_composite_address(tensor)` (non-owning ptr to the device handle) and a
   `get_dma_stream(device)` returning a pooled `RuntimeStream*`
   (`getDefaultStreamRuntimeHandle`).
4. **Our extension (`SharedKvPool` — flex-only):** implement the pool + pybind11. **Stage A
   backing = one regular CPU buffer** (`num_slots × slot_bytes`, aligned to
   `GetIovaAlignment()` §6.4); per-block aggregation (slot = whole block, §6.1);
   `copy_out`/`copy_in(slot, pages, stream)` loop one `copyRaw` per page into sub-offsets,
   single-chunk assert, `synchronize()` before return. No torch-spyre header.
5. Wire into the plugin: replace `_alloc_host_pages`/`kv_adapter.py:174` with per-block
   pool slots; in `handlers.py:166-170`, gather the block's device pages + a pooled stream
   (the torch-spyre boundary) and call the pool with flex types. Manager unchanged
   (inherits upstream LRU).
6. Tests: §6.7 (1–5), single-instance end-to-end offload/reload parity vs. the current
   `torch.empty` path.

### Stage B — shared-memory backing + metadata
7. **Swap the pool backing to shared memory:** `shm_open`+`mmap` named segment, attach
   refcount, last-out unlink — same slot/aggregation layout as Stage A (§7).
8. Define the control-segment layout + `DirEntry` (§7.2–7.3) in C++/shared structs;
   `key_len` from vLLM hash config.
9. Implement the open-addressed directory (insert/lookup/probe; full-key compare) and
   the slot allocator (warmup cursor + evict-recycle).
10. Choose and implement the eviction algorithm (§7.4) — **this is where the
    CLOCK-vs-sampled decision and its SHM mechanics are finalized.**
11. Implement `SharedOffloadingManager` (8 methods over the directory), robust-mutex
    locking, RESERVED→VALID gate (incl. `synchronize()`-before-publish §6.4),
    `EOWNERDEAD` recovery.
12. Return it from `SpyreOffloadingSpec.get_manager()`.
13. Tests: directory unit tests (probe/collision/dedup), refcount/pin transitions,
    eviction correctness, crash-recovery (kill a holder, assert recovery), and the
    headline **two-instance produce→consume** round-trip (one instance stores, another
    reloads the same key).

### Stage C — pool moves into flex (§10)
14. **flex (C1):** implement `SharedHostKvPool` (shm create/attach, map, slots, refcount)
    on flex's existing shm/pinned-memory machinery; point `copyRaw` at its slots.
15. **torch-spyre (C2):** add `_C` bindings — `create_or_attach`, `map`, `slot_count`/
    `available_slots`, `copy_out`/`copy_in`.
16. **spyre-inference:** call the flex pool through torch-spyre; retire the Stage-A
    `SharedKvPool` extension. Re-run all A+B tests unchanged (the slot seam is identical).

### Stage D — partitions (ultimate)
17. Generalize the single partition descriptor to an array; per-partition slot
    size/format/directory; route by model.

---

## 13. Open implementation points (carried into the plans)
- **Exact shape of the torch-spyre boundary accessors** (called by the handler, not the
  pool): `get_composite_address` returning a non-owning handle (lifetime tied to the
  tensor), and `get_dma_stream` returning a pooled `RuntimeStream*`.
- **flex `copyRaw` upstreaming:** locally buildable now; if this leaves the local tree it
  needs to land in flex proper (review/release).
- Final eviction algorithm + SHM mechanics (§7.4).
- Robust-mutex recovery details: exactly which states roll back on `EOWNERDEAD` (§7.6).
- **Deferred perf:** a dedicated pooled DMA stream (isolated from compute) instead of the
  shared default stream (§6.6).

**Resolved during design (no longer open):**
- *Multi-chunk DMA / chunk geometry* — proven single-chunk for KV pages under the `Bind`
  policy (§6.2 proof, layer 2); retained only as a runtime assertion + the `Interleave`
  re-open condition (§6.4), not an open question.
- *Cross-tensor recoverability* — proven from source (§6.2): same-`(shape,dtype)` ⇒
  identical layout, so a raw snapshot restores correctly into a different tensor. No dci
  canonical image needed.

**Residual dependency (not a blocker, but recorded):** the §6.2 proof depends on
torch-spyre/flex *current* behavior — tiling purity (`spyre_tensor_impl.cpp:124`) and
single-chunk `Bind` allocation (`spyre_allocator.cpp:120`). A future upstream change to
either would re-open §6.4's condition. The guards (§6.4) make any such regression fail
loudly rather than corrupt KV.
