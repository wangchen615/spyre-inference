# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy and lifecycle tests for cross-instance shared KV metadata."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from spyre_inference.v1.kv_offload.shared_manager import (
    SpyreSharedOffloadingManager,
)
from spyre_inference.v1.kv_offload.shared_types import (
    SharedLocation,
    SharedPoolFamily,
    shared_block_hash,
)

OFFLOAD_KEY = make_offload_key(bytes.fromhex("11" * 32), 0)
OTHER_KEY = make_offload_key(bytes.fromhex("22" * 32), 0)


@dataclass(frozen=True)
class FakeCompatibility:
    metadata_version: int
    compatibility_id: int


@dataclass(frozen=True)
class FakePoolRef:
    metadata_version: int
    pool_id: int
    pool_version: int


@dataclass(frozen=True)
class FakeRegisteredPool:
    name: str
    pool_ref: FakePoolRef
    compatibility: FakeCompatibility
    slot_count: int = 2
    slot_bytes: int = 4096


@dataclass(frozen=True)
class FakeCompatibleBlockKey:
    compatibility: FakeCompatibility
    block_hash: int


@dataclass(frozen=True)
class FakeSlotRef:
    pool: FakePoolRef
    slot_id: int
    slot_version: int


@dataclass
class FakeLookupEntry:
    key: FakeCompatibleBlockKey
    slot: FakeSlotRef
    chunks: tuple = ()
    pin: object | None = None


@dataclass
class FakeReservation:
    key: FakeCompatibleBlockKey
    slot: FakeSlotRef


@dataclass(frozen=True)
class FakeExistingClaim:
    slot: FakeSlotRef
    valid: bool


class FakeNoSpace:
    pass


class FakeUnavailable:
    pass


@dataclass
class PinRecord:
    released: bool = False


class FakePin:
    def __init__(self, record):
        self.record = record

    def __del__(self):
        self.record.released = True


class FakeDirectory:
    def __init__(self, anchors):
        self.anchors = {anchor.name: anchor for anchor in anchors}
        self.entries = {}
        self.reserved = set()
        self.pin_failures = set()
        self.pin_calls = []
        self.claim_results = {}
        self.claim_calls = []
        self.live_reservations = []
        self.aborted = []
        self.evicted = []

    def find_pool(self, name):
        return self.anchors.get(name)

    def lookup(self, key):
        if key.block_hash in self.reserved:
            return None
        return self.entries.get(key.block_hash)

    def pin_read(self, entry):
        self.pin_calls.append(entry)
        if entry.key.block_hash in self.pin_failures:
            return None
        record = PinRecord()
        entry.pin = record
        return FakePin(record)

    def claim(self, pool_ref, key):
        self.claim_calls.append((pool_ref, key))
        outcome = self.claim_results.get((key.block_hash, pool_ref.pool_id))
        slot = FakeSlotRef(pool_ref, slot_id=0, slot_version=1)
        if outcome == "existing_valid":
            return FakeExistingClaim(slot, True)
        if outcome == "existing_reserved":
            return FakeExistingClaim(slot, False)
        if outcome == "no_space":
            return FakeNoSpace()
        if outcome == "unavailable":
            return FakeUnavailable()
        reservation = FakeReservation(key, slot)
        self.live_reservations.append(reservation)
        return reservation

    def publish(self, reservation, chunks=()):
        self.live_reservations.remove(reservation)
        entry = FakeLookupEntry(reservation.key, reservation.slot, tuple(chunks))
        self.entries[reservation.key.block_hash] = entry
        return entry

    def abort(self, reservation):
        self.live_reservations.remove(reservation)
        self.aborted.append(reservation)

    def evict(self, entry):
        self.evicted.append(entry)
        current = self.entries.get(entry.key.block_hash)
        if current != entry:
            return False
        del self.entries[entry.key.block_hash]
        return True


def _runtime(directory):
    class FakeSharedMetadata:
        @staticmethod
        def create_or_attach(name, config):
            directory.attach_calls.append((name, config))
            return directory

    return SimpleNamespace(
        SharedMetadata=FakeSharedMetadata,
        SharedMetadataConfig=lambda max_chunks, pools, capacity: SimpleNamespace(
            max_chunks=max_chunks, pools=pools, capacity=capacity
        ),
        SharedMetadataCapacity=lambda max_pools, max_slots, max_compat: SimpleNamespace(
            max_pools=max_pools,
            max_slots_per_pool=max_slots,
            max_compatibilities=max_compat,
        ),
        CompatibleBlockKey=FakeCompatibleBlockKey,
        Reservation=FakeReservation,
        ExistingClaim=FakeExistingClaim,
        NoSpace=FakeNoSpace,
        Unavailable=FakeUnavailable,
    )


@pytest.fixture
def manager_directory_runtime():
    return _manager_directory_runtime()


def _manager_directory_runtime(*, num_blocks=4, cache_policy="lru", store_threshold=0):
    compatibility = FakeCompatibility(1, 7)
    anchors = [
        FakeRegisteredPool(
            f"{name}.c0.k",
            FakePoolRef(1, pool_id, 1),
            compatibility,
        )
        for pool_id, name in enumerate(("alpha", "beta"), start=10)
    ]
    directory = FakeDirectory(anchors)
    directory.attach_calls = []
    runtime = _runtime(directory)
    manager = SpyreSharedOffloadingManager(
        metadata_name="shared-meta",
        families=(SharedPoolFamily("alpha", 2), SharedPoolFamily("beta", 2)),
        max_components=4,
        num_blocks=num_blocks,
        cache_policy=cache_policy,
        cache_policy_module_path=None,
        enable_events=False,
        store_threshold=store_threshold,
        max_tracker_size=64_000,
        runtime_loader=lambda: runtime,
    )
    return manager, directory, runtime


def _peer_entry(directory, runtime, key=OFFLOAD_KEY, family="alpha", slot_id=1):
    anchor = directory.anchors[f"{family}.c0.k"]
    compatible_key = runtime.CompatibleBlockKey(anchor.compatibility, shared_block_hash(key))
    return FakeLookupEntry(
        compatible_key,
        FakeSlotRef(anchor.pool_ref, slot_id=slot_id, slot_version=3),
    )


def test_peer_lookup_pins_without_entering_local_policy(manager_directory_runtime):
    manager, directory, runtime = manager_directory_runtime
    peer_entry = _peer_entry(directory, runtime)
    directory.entries[peer_entry.key.block_hash] = peer_entry
    ctx = ReqContext("request-1")

    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.HIT
    assert manager._policy.get(OFFLOAD_KEY) is None

    spec = manager.prepare_load([OFFLOAD_KEY], ctx)
    assert spec.transfers[0].location == SharedLocation(
        peer_entry.slot.pool.pool_id, peer_entry.slot.slot_id
    )
    assert peer_entry.pin.released is False

    manager.complete_load([OFFLOAD_KEY], ctx)
    assert peer_entry.pin.released is True


@pytest.mark.parametrize("mode", ["missing", "reserved"])
def test_missing_or_reserved_lookup_is_a_normal_miss(mode, manager_directory_runtime):
    manager, directory, _ = manager_directory_runtime
    if mode == "reserved":
        directory.reserved.add(shared_block_hash(OFFLOAD_KEY))
    ctx = ReqContext("request-1")

    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.MISS
    assert directory.pin_calls == []


def test_failed_pin_is_a_normal_miss(manager_directory_runtime):
    manager, directory, runtime = manager_directory_runtime
    peer_entry = _peer_entry(directory, runtime)
    directory.entries[peer_entry.key.block_hash] = peer_entry
    directory.pin_failures.add(peer_entry.key.block_hash)
    ctx = ReqContext("request-1")

    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.MISS
    assert len(directory.pin_calls) == 1
    assert manager._policy.get(OFFLOAD_KEY) is None


def test_prepare_load_releases_scanned_but_unselected_pins(manager_directory_runtime):
    manager, directory, runtime = manager_directory_runtime
    selected = _peer_entry(directory, runtime, OFFLOAD_KEY, slot_id=0)
    unselected = _peer_entry(directory, runtime, OTHER_KEY, slot_id=1)
    directory.entries[selected.key.block_hash] = selected
    directory.entries[unselected.key.block_hash] = unselected
    ctx = ReqContext("request-1")
    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.HIT
    assert manager.lookup(OTHER_KEY, ctx) is LookupResult.HIT

    manager.prepare_load([OFFLOAD_KEY], ctx)

    assert selected.pin.released is False
    assert unselected.pin.released is True


def test_request_finish_releases_pending_but_not_active_pins(
    manager_directory_runtime,
):
    manager, directory, runtime = manager_directory_runtime
    active = _peer_entry(directory, runtime, OFFLOAD_KEY, slot_id=0)
    pending = _peer_entry(directory, runtime, OTHER_KEY, slot_id=1)
    directory.entries[active.key.block_hash] = active
    directory.entries[pending.key.block_hash] = pending
    ctx = ReqContext("request-1")
    manager.lookup(OFFLOAD_KEY, ctx)
    manager.lookup(OTHER_KEY, ctx)
    manager.prepare_load([OFFLOAD_KEY], ctx)

    manager.on_request_finished(ctx)

    assert pending.pin.released is True
    assert active.pin.released is False
    manager.complete_load([OFFLOAD_KEY], ctx)
    assert active.pin.released is True


def test_request_state_cleanup_uses_identity_not_dataclass_equality(
    manager_directory_runtime,
):
    manager, _, _ = manager_directory_runtime
    first_ctx = ReqContext("request-1")
    second_ctx = ReqContext("request-2")
    assert manager.lookup(OFFLOAD_KEY, first_ctx) is LookupResult.MISS
    assert manager.lookup(OFFLOAD_KEY, second_ctx) is LookupResult.MISS
    first_state = first_ctx.get_state(type(manager._request_states[0]))

    manager.on_request_finished(second_ctx)

    assert len(manager._request_states) == 1
    assert manager._request_states[0] is first_state


def _set_claim_result(directory, key, result, *, family=None):
    pool_ids = (
        [directory.anchors[f"{family}.c0.k"].pool_ref.pool_id]
        if family is not None
        else [anchor.pool_ref.pool_id for anchor in directory.anchors.values()]
    )
    for pool_id in pool_ids:
        directory.claim_results[(shared_block_hash(key), pool_id)] = result


def _publish_store(directory, output):
    for transfer in output.store_spec.transfers:
        directory.publish(transfer.reservation)


@pytest.mark.parametrize("claim_result", ["existing_valid", "existing_reserved", "no_space"])
def test_non_reservation_claim_rolls_back_local_admission(claim_result, manager_directory_runtime):
    manager, directory, _ = manager_directory_runtime
    _set_claim_result(directory, OFFLOAD_KEY, claim_result)
    ctx = ReqContext("request-1")

    out = manager.prepare_store([OFFLOAD_KEY], ctx)

    assert out is not None
    assert out.keys_to_store == []
    assert manager._policy.get(OFFLOAD_KEY) is None
    assert directory.live_reservations == []


def test_mixed_claim_batch_retains_only_successful_reservations(
    manager_directory_runtime,
):
    manager, directory, _ = manager_directory_runtime
    third_key = make_offload_key(bytes.fromhex("33" * 32), 0)
    _set_claim_result(directory, OTHER_KEY, "existing_valid")
    _set_claim_result(directory, third_key, "no_space")
    ctx = ReqContext("request-1")

    out = manager.prepare_store([OFFLOAD_KEY, OTHER_KEY, third_key], ctx)

    assert out is not None
    assert out.keys_to_store == [OFFLOAD_KEY]
    assert [item.key for item in out.store_spec.transfers] == [OFFLOAD_KEY]
    assert manager._policy.get(OFFLOAD_KEY) is not None
    assert manager._policy.get(OTHER_KEY) is None
    assert manager._policy.get(third_key) is None
    assert directory.live_reservations == [out.store_spec.transfers[0].reservation]


def test_unavailable_rolls_back_batch_and_aborts_prior_reservations(
    manager_directory_runtime,
):
    manager, directory, _ = manager_directory_runtime
    _set_claim_result(directory, OTHER_KEY, "unavailable")
    ctx = ReqContext("request-1")

    with pytest.raises(RuntimeError, match="unavailable"):
        manager.prepare_store([OFFLOAD_KEY, OTHER_KEY], ctx)

    assert directory.live_reservations == []
    assert len(directory.aborted) == 1
    assert manager._policy.get(OFFLOAD_KEY) is None
    assert manager._policy.get(OTHER_KEY) is None


def test_local_eviction_uses_the_exact_published_directory_entry():
    manager, directory, _ = _manager_directory_runtime(num_blocks=1)
    ctx = ReqContext("request-1")
    first = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert first is not None
    _publish_store(directory, first)
    published_entry = directory.entries[shared_block_hash(OFFLOAD_KEY)]
    manager.complete_store(first.keys_to_store, ctx)

    second = manager.prepare_store([OTHER_KEY], ctx)

    assert second is not None
    assert second.evicted_keys == [OFFLOAD_KEY]
    assert directory.evicted == [published_entry]


def test_claim_probes_all_families_from_the_hash_selected_start():
    assert shared_block_hash(OFFLOAD_KEY) % 2 == 1
    manager, directory, _ = _manager_directory_runtime()
    _set_claim_result(directory, OFFLOAD_KEY, "no_space", family="beta")
    ctx = ReqContext("request-1")

    out = manager.prepare_store([OFFLOAD_KEY], ctx)

    assert out is not None
    assert out.keys_to_store == [OFFLOAD_KEY]
    assert [call[0].pool_id for call in directory.claim_calls] == [11, 10]
    assert out.store_spec.transfers[0].location == SharedLocation(10, 0)


def test_lookup_before_worker_anchor_registration_fails_directly():
    manager, directory, _ = _manager_directory_runtime()
    directory.anchors.clear()

    with pytest.raises(RuntimeError, match="worker KV-cache registration"):
        manager.lookup(OFFLOAD_KEY, ReqContext("request-1"))


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_shared_manager_reuses_upstream_admission_and_victim_policy(cache_policy):
    manager, directory, _ = _manager_directory_runtime(
        num_blocks=2, cache_policy=cache_policy, store_threshold=2
    )
    upstream = CPUOffloadingManager(
        num_blocks=2,
        cache_policy=cache_policy,
        store_threshold=2,
    )
    shared_ctx = ReqContext("shared")
    upstream_ctx = ReqContext("upstream")
    third_key = make_offload_key(bytes.fromhex("33" * 32), 0)

    for key in (OFFLOAD_KEY, OTHER_KEY, third_key):
        for _ in range(2):
            assert manager.lookup(key, shared_ctx) is LookupResult.MISS
            assert upstream.lookup(key, upstream_ctx) is LookupResult.MISS

    shared_first = manager.prepare_store([OFFLOAD_KEY, OTHER_KEY], shared_ctx)
    upstream_first = upstream.prepare_store([OFFLOAD_KEY, OTHER_KEY], upstream_ctx)
    assert shared_first is not None and upstream_first is not None
    assert shared_first.keys_to_store == upstream_first.keys_to_store
    _publish_store(directory, shared_first)
    manager.complete_store(shared_first.keys_to_store, shared_ctx)
    upstream.complete_store(upstream_first.keys_to_store, upstream_ctx)
    manager.touch([OFFLOAD_KEY], shared_ctx)
    upstream.touch([OFFLOAD_KEY], upstream_ctx)

    shared_second = manager.prepare_store([third_key], shared_ctx)
    upstream_second = upstream.prepare_store([third_key], upstream_ctx)

    assert shared_second is not None and upstream_second is not None
    assert shared_second.keys_to_store == upstream_second.keys_to_store
    assert shared_second.evicted_keys == upstream_second.evicted_keys == [OTHER_KEY]
    assert len(directory.claim_calls) == 3
    assert len(directory.evicted) == 1


def test_failed_store_drops_pending_reservation_and_local_admission(
    manager_directory_runtime,
):
    manager, directory, _ = manager_directory_runtime
    ctx = ReqContext("request-1")
    out = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert out is not None
    directory.abort(out.store_spec.transfers[0].reservation)

    manager.complete_store(out.keys_to_store, ctx, success=False)

    assert manager._policy.get(OFFLOAD_KEY) is None
    assert manager._pending_reservations == {}


def test_reset_releases_only_state_owned_by_this_manager():
    manager, directory, runtime = _manager_directory_runtime()
    ctx = ReqContext("request-1")
    first = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert first is not None
    _publish_store(directory, first)
    owned_entry = directory.entries[shared_block_hash(OFFLOAD_KEY)]
    manager.complete_store(first.keys_to_store, ctx)

    peer_entry = _peer_entry(directory, runtime, OTHER_KEY)
    directory.entries[peer_entry.key.block_hash] = peer_entry
    assert manager.lookup(OTHER_KEY, ctx) is LookupResult.HIT

    pending_key = make_offload_key(bytes.fromhex("33" * 32), 0)
    pending = manager.prepare_store([pending_key], ctx)
    assert pending is not None
    pending_reservation = pending.store_spec.transfers[0].reservation

    manager.reset_cache()

    assert directory.evicted == [owned_entry]
    assert pending_reservation in directory.aborted
    assert directory.entries[peer_entry.key.block_hash] is peer_entry
    assert peer_entry.pin.released is True
    assert manager._policy.get(OFFLOAD_KEY) is None


def test_reset_releases_active_pin_before_evicting_owned_entry():
    manager, directory, _ = _manager_directory_runtime()
    ctx = ReqContext("request-1")
    store = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert store is not None
    _publish_store(directory, store)
    owned_entry = directory.entries[shared_block_hash(OFFLOAD_KEY)]
    manager.complete_store(store.keys_to_store, ctx)
    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.HIT
    manager.prepare_load([OFFLOAD_KEY], ctx)
    assert owned_entry.pin.released is False

    evict = directory.evict

    def evict_after_pin_release(entry):
        assert entry.pin.released is True
        return evict(entry)

    directory.evict = evict_after_pin_release

    manager.reset_cache()

    assert manager._request_states == []
