# Shared CPU KV Offload Pool Across vLLM Instances (PoC) — Design

**Status:** Design / PoC
**Date:** 2026-06-26
**Scope:** spyre-inference KV offload (`spyre_inference/v1/kv_offload/`), with a new
torch-spyre copy primitive (`copy_raw_tensor`).

---

## 1. Motivation & Goal

Today the Spyre KV offload host pool (`kv_adapter.py`) is **per-process**: each vLLM
instance allocates its own host staging pages with `torch.empty(..., device="cpu")`.
Those pages are anonymous heap memory visible only to that process. Two vLLM
instances on the same host cannot reuse each other's offloaded KV.

**Goal (PoC):** a **host-scoped CPU KV pool, backed by POSIX shared memory, that
multiple vLLM instances on one host can share** — so KV produced by one instance can
be reused by another. The concrete PoC target is **two instances: one produces
(stores) KV, one consumes (retrieves) it** through the shared pool.

This design also depends on a second change discussed alongside it: moving the
device↔host transfer from a typed, per-tensor copy to a **raw blob copy**
(`copy_raw_tensor`, see §5), which is what allows host pages to be slices of one
contiguous shared arena.

### Non-goals (PoC)
- Multi-host / networked sharing (we are same-host only).
- Robustness/hardening (crash recovery, eviction tuning, version skew) — see §8.
- Local-only fallback when the server is absent — see §8. The PoC assumes the
  server is present.

---

## 2. Background: what the code guarantees

Three facts established from the code make this design viable. They are load-bearing
and are restated here so the design is self-contained.

### 2.1 A torch CPU tensor's storage is one contiguous buffer
`torch.empty(N, dtype, device="cpu")` allocates a single contiguous storage of
`N * itemsize` bytes. A slice `arena[a:b].view(shape)` is a **view** sharing that
storage — no copy — with a known byte address `arena.data_ptr() + storage_offset *
itemsize`. This is what lets us expose arena slots as both indexable tensors *and*
byte-addressable blobs.

### 2.2 Spyre `copy_tensor` accepts offset views; offset is honored
`torch_spyre._C.copy_tensor` → `spyre_copy_from` (`spyre_mem.cpp:594`) →
`SpyreStream::copyAsync` (`spyre_stream.cpp:143`) takes the CPU **storage base**
(`storage().data()`) plus `storage_offset()` separately, threading the offset through
`generate_dci` → `get_device_stride_infos`. So a CPU tensor that is an offset-view
into a larger arena copies to/from the correct slot. Caveat: the **D2H** path falls
back to a temp-buffer copy if the *device* source is non-dense
(`spyre_mem.cpp:607-648`); per-block device pages are dense, so the fast path
applies (verify on hardware).

### 2.3 The device KV layout is a deterministic pure function of (shape, dtype)
`SpyreTensorLayout::init(host_size, dtype)` (`spyre_tensor_impl.cpp:124`) computes the
tiled `device_size`/`stride_map` from fixed stick tables
(`get_generic_stick_layout`, `elems_per_stick`) with **no** dependency on core count
(`SENCORES`), device handle, allocation address, or any runtime state. Two instances
with the same `(shape, dtype)` produce **byte-identical** device layouts. This is what
makes raw-blob cross-instance reuse possible, and `device_size`/`stride_map` are
exposed to Python via `get_spyre_tensor_layout` for the fingerprint (§6).

### 2.4 Reference: LMCache engine-driven SHM transport
LMCache solves the same problem for non-CUDA devices with an **MP-server** owning a
SHM pool, clients exchanging `prepare/commit` messages and copying KV through SHM
slots (`lmcache/v1/multiprocess/transfer_context/shm.py`,
`docs/design/v1/multiprocess/engine_driven_transfer_design.md`). This design adapts
that proven shape to Spyre.

---

## 3. Deployment model

- **Same host, multiple processes.** All instances run on one machine and share
  physical CPU RAM via POSIX shared memory (`/dev/shm`).
- **A host may have multiple Spyre cards.** Workers are pinned per card via
  `torch.spyre.set_device(local_rank)` (`spyre_worker.py:92`). The CPU pool is
  **host-scoped** — one server per host serves all cards. Blocks remain
  **shard/layout-scoped**: a block from one card's KV shard is not byte-interchangeable
  with another's, which the layout fingerprint (§6) enforces on read.
- **Sharing wins:** (a) separate same-config instances reusing each other's prefixes;
  (b) within a TP instance, same-shard workers share; different shards partition
  naturally by fingerprint.

---

## 4. Architecture

A dedicated **MP-server** process owns the shared pool and its index. vLLM Spyre
instances are **clients**: they map the same SHM arena and copy KV bytes directly
into/out of server-assigned slots, exchanging only small index messages over IPC.

```
  ┌─ vLLM inst 1 (produce) ─┐   ┌─ vLLM inst 2 (consume) ─┐
  │   SpyreKvPoolClient     │   │   SpyreKvPoolClient     │
  └──────────┬──────────────┘   └──────────┬──────────────┘
             │ IPC: prepare/commit + slot descriptors (tiny msgs)
             ▼                             ▼
  ┌──────────────── SpyreKvPoolServer (1 per host) ───────────────┐
  │  owns: SHM arena + AddressManager + index {key → SlotEntry}    │
  │  serializes all index ops internally (in-process lock)         │
  └────────────────────────────────────────────────────────────────┘
             ▲  both instances mmap the same SHM arena  ▲
             └──── KV bytes copied directly via slots ───┘   (never over IPC)
```

### 4.1 Components

| Component | Process | Responsibility |
|---|---|---|
| `SpyreKvPoolServer` | server (1/host) | Owns SHM arena, `AddressManager` (first-fit + coalesce — same allocator shape as LMCache's `memory_management.py:AddressManager`), index `key → SlotEntry`. Serializes index ops on an in-process lock. Handles the RPCs (§7.2). |
| `SpyreKvPoolClient` | each instance | Maps the SHM arena by name; sends prepare/commit RPCs; performs `copy_raw_tensor` device↔slot; computes content key + layout fingerprint. |
| `pool_server_launcher` | client-side | Lockfile election + detached server spawn + readiness poll (§9). Separated from data-path logic so it is testable in isolation. |
| SHM arena | shared pages | `/dev/shm/<shm_name>`, one flat `uint8` buffer via `multiprocessing.shared_memory.SharedMemory`. Pages are `torch.frombuffer(...).view(page_shape)` offset-views. Unpinned (no Spyre `cudaHostRegister`). |
| `SlotEntry` | server index | One per block (a block's K and V pages together). `(k_offset, v_offset, length, layout_fingerprint, refcount, state)`, where `state ∈ {reserved, published}`. "A slot" throughout means this K+V pair, not a single region. |
| `SlotDescriptor` | wire type | `(k_offset, v_offset, length)` byte offsets into the arena; the wire-facing subset of a `SlotEntry`. Serializable (`to_dict`/`from_dict`). |
| `copy_raw_tensor` | torch-spyre | New `_C` primitive, `nullptr` DCI, tiled-blob contract (§5). |

### 4.2 Integration with existing code
- Client replaces host-page ownership in `kv_adapter.py` (`_alloc_host_pages` →
  map-from-SHM offset views).
- Transfer calls in `handlers.py` / `copier.py` become `prepare → copy_raw_tensor →
  commit`.
- `SpyreOffloadingSpec` gains `shm_name` and server-endpoint config knobs alongside
  `cpu_bytes_to_use`.
- The offload addressing moves from **positional `block_id`** (current
  `CPUOffloadingManager`) to **content key** (§6). This is the substantive change.

---

## 5. `copy_raw_tensor` — raw blob copy primitive

A new torch-spyre primitive: `copy_raw_tensor(src, dst, non_blocking=False)`, same
tensor-in/tensor-out signature as `copy_tensor` but issuing the DMA with a **`nullptr`
DCI** — i.e. no tiling/layout conversion, a verbatim byte copy.

**Contract (the "tiled-blob" contract):** the host side stores **device-native tiled
bytes**, opaque, and replays them only to an identical device layout. This is correct
for:
- **host↔host** arena copies (both sides plain CPU, no tiling) — unambiguously a
  `memcpy`;
- **device↔host** offload/reload **when the host blob is replayed to the same device
  layout** — which the fingerprint (§6) guarantees.

It is **not** safe to interpret a `nullptr`-DCI host blob as a logical
`[heads, block, head_size]` tensor, nor to replay it to a *different* device layout.
The fingerprint check turns any such mismatch into a clean MISS.

K and V are copied as **two separate blobs** (device K and V pages are independent
allocations, not contiguous) — fusing them into one DMA is out of scope.

**Still pass tensors, not pointers.** The C++ path derives `storage().data() +
storage_offset()`, so arena offset-views land at the correct slot; there is no
pointer-based entry point.

**Hardware verifications (gating, not assumptions):**
1. A `nullptr`-DCI D2H→H2D round-trip to the same device tensor is bit-exact.
2. Per-block device pages take the dense D2H fast path, not the temp-buffer fallback
   (the §2.2 caveat).

These are the two unknowns the implementation must confirm on real Spyre before the
raw-blob path can be trusted.

---

## 6. Layout fingerprint — the cross-instance correctness guard

Raw-blob copy moves device-native tiled bytes verbatim; those bytes are interchangeable
between two instances only if both agree on the exact device layout. The fingerprint
lets the server **verify that agreement before serving a block**, turning any
disagreement into a clean MISS instead of wrong bytes on the device.

**Definition** — a tuple computed by the client, attached to every stored block and
presented on every retrieve:

| Field | Source | Why included |
|---|---|---|
| `model_name` | vLLM config | Different model ⇒ different KV semantics |
| `tp_degree` | `parallel_config.world_size` | TP splits `num_kv_heads` across cards |
| `shard_id` / rank | worker `local_rank` | Each card holds a different KV shard |
| `dtype` | KV dtype (fp16) | Governs stick size / tiling |
| `num_kv_heads`, `head_size`, `block_size` | model + vLLM block config | Logical page shape |
| `device_size`, `stride_map` | `get_spyre_tensor_layout(page)` | The decisive actual tiled byte layout (§2.3) |

**Content key** (the index key, distinct from the fingerprint):
`(model_name, tp_degree, shard_id, dtype, chunk_hash)`, where `chunk_hash` is a
prefix-chained hash of the chunk's token ids (so the same prompt prefix on the same
model/shard yields an identical key on every instance — this is what makes
cross-instance lookup *meaningful*). Key → location is a plain index lookup; the offset
is chosen by the allocator, never derived from the hash.

**Usage (wired into the protocol):**
- `PREPARE_STORE` carries the fingerprint → stored in the `SlotEntry`.
- `PREPARE_RETRIEVE` carries it → server returns a slot **only if** present, published,
  **and** stored fingerprint == client fingerprint; else MISS.

The full tuple is carried from the PoC onward (cheap; a few ints + two short vectors per
block) so the design is correct-by-construction the moment instances diverge
(multi-card, multi-model, TP>1).

---

## 7. Protocol & data flow

### 7.1 The two conversations (explainer)

The whole protocol is: **client asks the server "where do I put it / where is it?",
copies bytes through shared memory itself, then tells the server "done."** The socket
carries tiny messages; KV bytes never touch it. (One `REGISTER` exchange happens once
at startup, before either conversation below; it is omitted from the diagrams.)

```
Store:
 client                         server                     SHM arena
   │ ── PREPARE_STORE(key,fp) ─► │ allocate slot, pin       │
   │ ◄─ SlotDescriptor ──────────│                          │
   │ copy_raw_tensor(device ──────────────────────────────► [slot]
   │ ── COMMIT_STORE(key) ─────► │ slot reserved→published, unpin
   │ ◄─ ok ──────────────────────│

Retrieve:
 client                         server                     SHM arena
   │ ── PREPARE_RETRIEVE(key,fp)►│ found+published+fp match? pin
   │ ◄─ SlotDescriptor / MISS ───│                          │
   │ copy_raw_tensor([slot] ◄──────────────────────────────  bytes
   │ ── COMMIT_RETRIEVE(key) ──► │ unpin                     │
   │ ◄─ ok ──────────────────────│
```

### 7.2 Message types (reference)

| Request | Payload | Response | Server action |
|---|---|---|---|
| `REGISTER` | `client_id, shm_name, model/tp/shard identity` | `shm_name, pool_size` | Record client; confirm it maps the right arena. |
| `PREPARE_STORE` | `client_id, [content_key], layout_fingerprint` | `[SlotDescriptor \| ALREADY_PRESENT]` per key | Per key: if already **published** with the same fingerprint → return `ALREADY_PRESENT` (skip, dedup). Otherwise allocate a slot, mark **reserved**, pin, return a `SlotDescriptor`. (A published key whose fingerprint *differs* is treated as not-present for this producer and gets a new slot — the store-side fingerprint is authoritative for what this producer writes. PoC assumes matching layouts, so this branch is not exercised.) |
| `COMMIT_STORE` | `client_id, [content_key]` | `ok` | Flip reserved → **published**; unpin. Block becomes a hit for all. |
| `PREPARE_RETRIEVE` | `client_id, [content_key], layout_fingerprint` | `[SlotDescriptor \| MISS]` per key | Return slot iff published **and** stored fp == client fp; pin (refcount++). |
| `COMMIT_RETRIEVE` | `client_id, [content_key]` | `ok` | Unpin (refcount--). |

- **Transport:** request/reply over UNIX-domain socket or ZMQ `REQ`/`REP`; endpoint
  published in the lockfile. Payloads msgpack-serializable.
- **`SlotDescriptor`** = `(k_offset, v_offset, length)`; client builds arena offset-views
  via `torch.frombuffer(shm.buf, ...).view(page_shape)`.
- **Batching:** keys are lists — one round-trip covers a whole offload/load group, so IPC
  cost is per-operation, not per-block.
- **`client_id`** assigned at `REGISTER`; scopes pin ownership.

### 7.3 Why two-step (prepare + commit)

The split is what buys the happy-path invariants:
- **Write-once-then-publish** — the server flips a slot to *published* (and only then
  serves it to readers) at `COMMIT_STORE`, after the client's `copy_raw_tensor` has
  finished. Readers never see a half-written slot. Implemented by the server's
  reserved/published state, **not** a shared flag clients read.
- **Pin-across-use** — refcount held over the prepare→commit window prevents
  eviction reclaiming a slot mid-copy.
- **Verify-on-read** — fingerprint compared at `PREPARE_RETRIEVE` (§6).

Index ops serialize in the server; data copies run concurrently across clients
(different slots, direct SHM access). Eviction (when full): server evicts LRU
**unpinned + published** slots, recycling the offset via `AddressManager`. The index is
always the source of truth for what a slot currently holds.

---

## 8. PoC scope & assumptions

**Handled in PoC (correctness-critical only):**

| # | Concern | Minimal handling |
|---|---|---|
| C1 | One server, started before clients | Lockfile election: one instance spawns the server, the other connects. Mechanism in §9. |
| C2 | Server outlives the spawner | Server spawned detached so the producer exiting doesn't kill the pool before the consumer reads. Mechanism in §9. |
| C3 | Layout-fingerprint mismatch → MISS | Because `copy_raw_tensor` moves device-native bytes verbatim (§5), serving a block whose device layout differs from the consumer's (different shard/TP/model/dtype) would push **garbage to the device** — not a recoverable error. The server compares the stored vs. consumer layout fingerprint on retrieve and returns **MISS** on any difference, so a mismatch degrades to a recompute, never corruption. Fingerprint mechanism and fields: §6. |

**PoC assumptions (explicit):**
- Two instances, same build, same model/layout — producer stores, consumer retrieves.
- Pool sized not to fill during the run (eviction may be a stub).
- Cooperative clients, no crash recovery; a leaked pin is acceptable for a PoC run.

**Deferred to future work (documented, not built):**
- **Local-only fallback** when the server is unreachable (→ existing `torch.empty`
  path).
- Stale-lockfile reclaim; version/protocol handshake.
- Pool-full backpressure: a `STORE_SKIPPED` response when eviction cannot free a slot
  (not a PoC response type — the PoC assumes the pool does not fill), and eviction tuning.
- Client/server crash-recovery, reservation timeouts, heartbeats.
- `/dev/shm` exhaustion handling.
- **Lock-free / sharded-lock peer model** removing the daemon (§11).

---

## 9. Server lifecycle — first-instance-wins auto-spawn

Race-free election via atomic filesystem rendezvous:

1. Each starting instance attempts `open(/tmp/spyre_kv_pool_<name>.lock,
   O_CREAT|O_EXCL)`. The OS guarantees one winner.
2. **Winner** spawns the server **detached** (`setsid`/double-fork — not in the
   spawner's process group, so it outlives the instance), waits until the server socket
   accepts, writes the endpoint (and server PID) into the lockfile, then proceeds as a
   normal client.
3. **Losers** see `EEXIST`, poll the lockfile until the endpoint appears and the socket
   is reachable, then connect as clients.

**Server lifetime — chosen: refcounted-with-grace.** The server exits when the last
client on the host disconnects (after a grace period). *Option, not chosen:*
**lives-until-reboot** — drawback is a **data-residency / leakage risk**: stale KV
bytes (possibly from another user's prompts) linger in `/dev/shm` after the workload
exits. Refcounted-with-grace reclaims the segment when the last client leaves, which is
why it is preferred.

(Stale-lockfile reclaim and version handshake are future work, §8.)

---

## 10. Testing strategy (PoC-scaled)

- **Headline E2E test:** two-process produce→consume round-trip. Process A stores a known
  KV block under a content key; process B retrieves it and asserts the bytes match what A
  wrote (bit-exact). This is the PoC's definition of done.
- **`copy_raw_tensor` round-trip:** device→arena→device bit-exact, on hardware (gated by
  `requires_spyre`), plus the dense-D2H fast-path check (§2.2).
- **Arena offset-view unit test:** `arena[a:b].view(shape)` slices map to the expected
  byte offsets and share storage (CPU-only, no Spyre).
- **Fingerprint guard:** a retrieve with a mismatched fingerprint returns MISS (CPU-only,
  server logic).
- **Protocol unit tests:** prepare/commit state transitions (reserved→published, pin/unpin
  refcounts), dedup on `PREPARE_STORE`, MISS on absent/unpublished key.
- **Launcher:** lockfile election picks one winner among concurrent starts; loser connects
  to the winner's endpoint.

Spyre device tests skip silently on CPU-only hosts (`requires_spyre`); CPU-only tests
(arena math, protocol, fingerprint, launcher) run everywhere.

---

## 11. Drawbacks & next steps

**MP-server drawbacks (acknowledged):**
- An extra daemon to spawn and supervise (mitigated by auto-spawn, §9).
- One IPC round-trip per transfer *group* for index ops (data copy is direct SHM, so the
  hop never touches the hot byte path).
- The server is a single point of failure for the *pool* (not for the instances; instance
  resilience via local-only fallback is future work).

**Next step — lock-free peer model.** Remove the daemon: instances map the arena directly
and coordinate through a **shared index in SHM**, starting with a coarse cross-process lock
on the index only (data reads stay lock-free via write-once + refcount), migrating to
**sharded locks** (`key % N`) or a lock-free concurrent hashmap. Deferred because the hard
part — cross-process robust locking (`pthread_mutexattr_setrobust` / `EOWNERDEAD` recovery)
and crash recovery — is not worth it for a PoC, and the MP-server gives an
effectively-lock-free cross-process story for free (one owner serializes a tiny index).
The §7 protocol and the §6 fingerprint are designed to carry over to the peer model
unchanged.
