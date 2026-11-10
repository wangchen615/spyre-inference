# Shared pool: evict on NoSpace (fix A)

Status: **implemented and verified on hardware** (2026-11-10). Branch `kvc-offload-m4-marvell`.

## Problem

A shared KV pool smaller than its working set fills and then never takes another block.

`SpyreSharedOffloadingManager` (`spyre_inference/v1/kv_offload/shared_manager.py`) keeps one
`CPUOffloadingManager` per replica as its cache policy. That local LRU:

- tracks only the blocks **this replica** stored (`_owned_entries`, `_local_keys`);
- is sized to the **whole** pool (`num_blocks=policy_capacity`, the plan's
  `logical_block_capacity`).

The local LRU evicts only when it reaches its own capacity. With two replicas sharing the pool,
neither reaches it, so neither evicts. When the pool's slots are all in use, flex's
`SharedMetadata::Claim` returns `NoSpace` (`shared_metadata.cpp`: `free_head == kNoSlot`), and
`prepare_store` drops the block (lines 423-437). Every later store is dropped too.

Measured (LongBench reuse list, shared 120 GiB, two replicas): stores stopped at exactly 120.0 GiB
(A 50.5 + B 69.5) during warm-up; 71% of measured prompt tokens cached against 98% predicted; all
13 mostly-computed requests asked one of the 10 documents warmed after the pool filled.

Not affected: a shared pool larger than its working set (never full); private pools (one local LRU
sized to the pool, `spec.py`); host pools through the delegator (no shared manager).

## Fix A: on NoSpace, evict this replica's oldest idle block and claim again

The cache policy stays per replica (the RFC leaves the eviction policy to spyre-inference, §3.1 and
the slot-lifecycle section). What changes: when the shared pool has no free slot, the replica makes
room from the blocks **it** owns, oldest first, instead of giving up.

```
prepare_store(keys):
    upstream = local_manager.prepare_store(keys)        # unchanged: admission, local eviction
    for key in upstream.keys_to_store:
        for each component of key:
            result = directory.claim(component)
            while result is NoSpace and attempts < MAX:
                victim = oldest idle block this replica owns, not in `keys`
                if no victim: break
                evict every page of victim from the shared pool
                drop victim from the local LRU and from _owned_entries/_local_keys
                result = directory.claim(component)      # retry the same component
            ...existing handling of Reservation / ExistingClaim / NoSpace / Unavailable
```

### Picking the victim

The local LRU already orders this replica's blocks and knows which are idle: a block a request is
loading has `ref_cnt > 0` and is not in the policy's evictable list. `CachePolicy.evict(n,
protected)` returns the oldest idle blocks, or `None` if it cannot (atomic: no change on `None`).
So the victim is `local_manager._policy.evict(1, protected=...)`, where `protected` is the keys of
this call **plus every key pinned by any request** (`pending_pins` and `active_pins` across
`_request_states`).

The pinned keys must be protected explicitly. `lookup` takes the flex read lock at once
(`directory.pin_read`), but the local LRU only marks the block busy later, in `prepare_load`
(`CPUOffloadingManager.lookup` does not touch `ref_cnt`). Between a request's lookup and its
`prepare_load`, a pinned block still looks idle to the policy. Evicting it in that window would make
`Evict` wait on this replica's own read lock forever.

`CPUOffloadingManager` has no public "evict one" method; eviction happens only inside its
`prepare_store`. The fix wraps the private access in one helper on the shared manager,
`_evict_one_idle(protected) -> OffloadKey | None`, which does exactly what
`CPUOffloadingManager.prepare_store` does for an eviction:

```
evicted = policy.evict(1, protected)              # None -> nothing idle
key, block = evicted[0]
local_manager._num_evictable_cache_blocks -= 1
local_manager._free_block(block)
```

This ties the helper to vLLM 0.28's `CPUOffloadingManager` internals; a test pins that contract
(below) so a vLLM bump that changes them fails loudly.

### Freeing the victim's slots

Same as the existing local-eviction path (lines 388-391):

```
for entry in _owned_entries.pop(victim): directory.evict(entry)
_local_keys.discard(victim)
```

`evict` returns `False` if the entry is already gone (another replica or a reset freed it); that is
not an error, the claim simply retries.

### Why this cannot deadlock or corrupt a read

- flex's `Evict` takes the slot's **write** lock (`pthread_rwlock_wrlock`); `PinRead` holds the
  **read** lock until the pin is released (`SlotReadPin`).
- **Own pins:** a page this replica has pinned is in some request's `pending_pins` (after `lookup`)
  or `active_pins` (after `prepare_load`, when its `ref_cnt` is also raised). Both sets are passed as
  `protected`, so the policy never offers a pinned block, including in the lookup-to-load window
  where `ref_cnt` is still 0. So a replica never waits on its own read lock.
- **The other replica's pins:** if the other replica is loading the victim's page, `Evict` waits for
  that load (milliseconds) and then frees the slot. The other replica's reader then holds a stale
  `LookupEntry`; flex's `slot_version` check makes its next `PinRead` of that slot fail cleanly
  (a miss), as for any eviction today.
- **Blocks being stored in this call:** `protected=set(keys)` keeps the blocks of the current
  `prepare_store` from being chosen.

### A block is stored whole or not at all

A block has one page per component (granite TP1: 80). On `NoSpace` mid-block, the fix evicts one
whole victim block (freeing all its components' slots) and retries the **same** component. If no
victim is left, the existing path aborts the components already reserved for this block. So a block
is never half stored.

### Bounding the retries

At most `MAX_EVICT_ATTEMPTS = 4` victims per component claim. Each victim frees one slot per
component it owns, so one victim is normally enough; the bound only stops a loop if another replica
keeps taking the freed slots.

## What does not change

- flex, torch-spyre, the delegator: no change, no rebuild.
- Private pools and host pools: not on this path.
- Shared pools larger than their working set: `NoSpace` never happens, so behaviour is identical.
- `ExistingClaim`, `Unavailable`, abort and rollback paths: unchanged.

## Limitation (vs a pool-wide LRU, fix B)

Each replica evicts only blocks it stored, ordered by its own use. A block replica A stored and
replica B uses heavily looks idle to A and can be evicted by A. And if one replica owns no blocks
(the other filled the pool), it cannot make room. In the LongBench runs both replicas store during
warm-up and mostly reuse the same documents, so the effect should be small; the results will say so.
A pool-wide LRU needs a last-use field in flex's shared metadata (layout, torch-spyre bindings,
delegator compatibility) and is out of scope here.

## Tests (`tests/kv_offload/test_shared_manager.py`, fake directory)

The fake directory gets a free-slot count so `claim` can return `NoSpace` and `evict` frees a slot.

1. **Full pool, evicts the oldest idle block and stores:** pool holds 2 blocks; store A, B; store C
   -> A is evicted from the directory and the local LRU, C is stored.
2. **Touch order is respected:** store A, B; touch A; store C -> B is evicted, not A.
3. **Never evicts a block being loaded:** A pinned and loading (`prepare_load`), B idle; store C ->
   B evicted. A pinned, B pinned; store C -> nothing evicted, C dropped, pins untouched.
3a. **Never evicts a block pinned by lookup only:** A looked up by another request (pinned, no
   `prepare_load` yet, `ref_cnt` 0), B idle; store C -> B evicted, A kept, A's pin still held.
4. **Never evicts a block of the same call:** `prepare_store([A, B])` into a pool that fits one ->
   A is stored, B dropped; A is not evicted to make room for B.
5. **Only own blocks:** a block another replica published (in the directory, not in
   `_owned_entries`) is never evicted.
6. **Mid-block NoSpace:** a 4-component block hits `NoSpace` at component 2 -> one victim evicted,
   the claim resumes at component 2, the block completes; with no victim, components 0-1 are
   aborted.
7. **`evict` returns False:** victim already gone -> no error, claim retried, bookkeeping clean.
8. **Bounded retries:** a directory that always returns `NoSpace` -> at most `MAX_EVICT_ATTEMPTS`
   victims, then the block is dropped.
9. **Accounting matches upstream:** after a fix-A eviction the local manager's evictable count and
   free list equal those after an equivalent upstream `CPUOffloadingManager.prepare_store`
   eviction (pins the private-field contract).
10. All existing shared-manager tests pass unchanged.

## Hardware check (worker-26, two replicas, after the unit tests)

Two replicas on c8 and ec, one shared Marvell pool sized below the documents' KV (e.g. 40 GiB for a
short document list), card prefix caching off so every reuse goes to the pool. Pass if: stores keep
growing after the pool is full; the second pass over the documents loads from Marvell (not 0);
0 errors, no hang (a hang would point at a lock), no RAS/BusFence; pool destroyed by generation;
`/dev/shm` clean.

## Order of work

1. Fake-directory support and the tests above (fail first).
2. The change in `shared_manager.py`.
3. `uv run --no-sync pytest tests/kv_offload/` on CPU.
4. Hardware check.
5. Diff for review; staged with `Signed-off-by: Yue Zhu <Yue.Zhu@ibm.com>` for your signature.

## Verification (2026-11-10, worker-26)

- CPU: `tests/kv_offload/test_shared_manager.py` 64 passed (the 11 tests above + 53 unchanged);
  `tests/kv_offload` 378 passed (its `_hw` files need a card and are skipped by error without one).
- Hardware, replica A on Spyre c8 and B on ec, one 20 GiB shared Marvell pool, 12 qmsum documents
  (26.9 GiB) warmed on both replicas, card prefix caching off:
  - asked newest-first, warm documents #4-#11 (the newest 20 GiB) came back 98-100% from the pool,
    including #8-#11, which a pool that never evicts does not store at all; #0-#2 were evicted;
  - each replica served documents the other stored, so both evicted to make room;
  - 0 errors, no hang (the self-lock case), no RAS or BusFence, pool destroyed by generation.
  - Asked in warm-up order instead (oldest first), each recomputed document's store evicts the next
    one asked: the cyclic pattern is LRU's worst case, and only 15% loaded. Expected, not a fault.

