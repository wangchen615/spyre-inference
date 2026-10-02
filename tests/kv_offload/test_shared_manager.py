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

"""Policy and lifecycle tests for one-pool shared KV metadata."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from spyre_inference.v1.kv_offload.shared_manager import SpyreSharedOffloadingManager
from spyre_inference.v1.kv_offload.shared_types import shared_page_hash

OFFLOAD_KEY = make_offload_key(bytes.fromhex("11" * 32), 0)
OTHER_KEY = make_offload_key(bytes.fromhex("22" * 32), 0)
THIRD_KEY = make_offload_key(bytes.fromhex("33" * 32), 0)
COMPONENT_COUNT = 4


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
    slot_count: int = 8
    slot_bytes: int = 4096


@dataclass(frozen=True)
class FakeCompatibleBlockKey:
    compatibility: FakeCompatibility
    block_hash: int


@dataclass(frozen=True)
class FakeSlotRef:
    pool: FakePoolRef
    slot_id: int
    slot_version: int = 1


@dataclass
class PinRecord:
    released: bool = False


class FakePin:
    def __init__(self, record: PinRecord):
        self.record = record

    def __del__(self):
        self.record.released = True


@dataclass
class FakeLookupEntry:
    key: FakeCompatibleBlockKey
    slot: FakeSlotRef
    chunks: tuple = ()
    pin_records: list[PinRecord] = field(default_factory=list)


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


@dataclass(frozen=True)
class FakeCapacity:
    max_pools: int
    max_slots_per_pool: int
    max_compatibilities: int


@dataclass(frozen=True)
class FakeMetadataConfig:
    max_chunks: int
    pools: tuple
    capacity: FakeCapacity


class FakeDirectory:
    def __init__(self, registered: FakeRegisteredPool):
        self.registered = registered
        self.attach_calls = []
        self.lookup_calls = []
        self.entries: dict[int, FakeLookupEntry] = {}
        self.pin_failures: set[int] = set()
        self.pin_calls = []
        self.claim_results: dict[int, object] = {}
        self.claim_calls = []
        self.live_reservations: list[FakeReservation] = []
        self.aborted: list[FakeReservation] = []
        self.evicted: list[FakeLookupEntry] = []

    def find_pool(self, name):
        return self.registered if name == self.registered.name else None

    def lookup(self, key):
        self.lookup_calls.append(key)
        return self.entries.get(key.block_hash)

    def pin_read(self, entry):
        self.pin_calls.append(entry)
        if entry.key.block_hash in self.pin_failures:
            return None
        record = PinRecord()
        entry.pin_records.append(record)
        return FakePin(record)

    def claim(self, pool_ref, key):
        self.claim_calls.append((pool_ref, key))
        outcome = self.claim_results.get(key.block_hash, 0)
        if isinstance(outcome, int):
            reservation = FakeReservation(key, FakeSlotRef(pool_ref, outcome))
            self.live_reservations.append(reservation)
            return reservation
        return outcome

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
        if self.entries.get(entry.key.block_hash) is not entry:
            return False
        del self.entries[entry.key.block_hash]
        return True


def _runtime(directory: FakeDirectory):
    class FakeSharedMetadata:
        @staticmethod
        def create_or_attach(name, config):
            directory.attach_calls.append((name, config))
            return directory

    return SimpleNamespace(
        SharedMetadata=FakeSharedMetadata,
        SharedMetadataConfig=lambda max_chunks, pools, capacity: FakeMetadataConfig(
            max_chunks, tuple(pools), capacity
        ),
        SharedMetadataCapacity=FakeCapacity,
        CompatibleBlockKey=FakeCompatibleBlockKey,
        Reservation=FakeReservation,
        ExistingClaim=FakeExistingClaim,
        NoSpace=FakeNoSpace,
        Unavailable=FakeUnavailable,
    )


def _manager_directory_runtime(
    *,
    slot_count: int = 8,
    cache_policy: str = "lru",
    store_threshold: int = 0,
):
    registered = FakeRegisteredPool(
        "shared.data",
        FakePoolRef(1, 10, 1),
        FakeCompatibility(2, 7),
        slot_count=slot_count,
    )
    directory = FakeDirectory(registered)
    runtime = _runtime(directory)
    manager = SpyreSharedOffloadingManager(
        metadata_name="shared-meta",
        pool_name="shared.data",
        component_count=COMPONENT_COUNT,
        max_pool_slots=16,
        cache_policy=cache_policy,
        cache_policy_module_path=None,
        enable_events=False,
        store_threshold=store_threshold,
        max_tracker_size=64_000,
        runtime_loader=lambda: runtime,
    )
    return manager, directory, runtime


@pytest.fixture
def manager_directory_runtime():
    return _manager_directory_runtime()


def _compatible_key(directory, runtime, key, component_id):
    return runtime.CompatibleBlockKey(
        directory.registered.compatibility,
        shared_page_hash(key, component_id),
    )


def _entry(directory, runtime, key, component_id, slot_id, *, pool_ref=None):
    compatible_key = _compatible_key(directory, runtime, key, component_id)
    return FakeLookupEntry(
        compatible_key,
        FakeSlotRef(pool_ref or directory.registered.pool_ref, slot_id),
    )


def _publish_complete(directory, runtime, key=OFFLOAD_KEY, slots=(7, 2, 6, 1)):
    entries = []
    for component_id, slot_id in enumerate(slots):
        entry = _entry(directory, runtime, key, component_id, slot_id)
        directory.entries[entry.key.block_hash] = entry
        entries.append(entry)
    return tuple(entries)


def _set_reservation_slots(directory, key, slots=(7, 2, 6, 1)):
    for component_id, slot_id in enumerate(slots):
        directory.claim_results[shared_page_hash(key, component_id)] = slot_id


def _publish_store(directory, output):
    for transfer in output.store_spec.transfers:
        for page in transfer.pages:
            directory.publish(page.reservation)


def test_directory_attach_lazily_sizes_policy_from_registered_pool(
    manager_directory_runtime,
):
    manager, directory, _ = manager_directory_runtime
    assert manager._local_manager is None

    assert manager.lookup(OFFLOAD_KEY, ReqContext("request-1")) is LookupResult.MISS

    assert directory.attach_calls == [
        (
            "shared-meta",
            FakeMetadataConfig(1, (), FakeCapacity(1, 16, 1)),
        )
    ]
    assert manager._local_manager is not None
    assert manager._local_manager._num_blocks == 2


@pytest.mark.parametrize("slot_count", [0, 6, 20])
def test_registered_pool_geometry_must_match_scheduler_limits(slot_count):
    manager, _, _ = _manager_directory_runtime(slot_count=slot_count)
    with pytest.raises(RuntimeError, match="slot_count"):
        manager.lookup(OFFLOAD_KEY, ReqContext("request-1"))


def test_fragmented_complete_hit_pins_every_page_until_complete_load(
    manager_directory_runtime,
):
    manager, directory, runtime = manager_directory_runtime
    entries = _publish_complete(directory, runtime)
    ctx = ReqContext("request-1")

    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.HIT
    assert len(directory.lookup_calls) == COMPONENT_COUNT
    assert len(directory.pin_calls) == COMPONENT_COUNT

    spec = manager.prepare_load([OFFLOAD_KEY], ctx)
    assert [
        (page.component_id, page.location.pool_id, page.location.slot_id)
        for page in spec.transfers[0].pages
    ] == [(0, 10, 7), (1, 10, 2), (2, 10, 6), (3, 10, 1)]
    assert all(not entry.pin_records[-1].released for entry in entries)

    manager.complete_load([OFFLOAD_KEY], ctx)
    gc.collect()
    assert all(entry.pin_records[-1].released for entry in entries)


@pytest.mark.parametrize("missing_component", range(COMPONENT_COUNT))
def test_partial_page_set_is_a_miss_and_releases_earlier_pins(
    missing_component, manager_directory_runtime
):
    manager, directory, runtime = manager_directory_runtime
    entries = _publish_complete(directory, runtime)
    del directory.entries[entries[missing_component].key.block_hash]
    ctx = ReqContext("request-1")

    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.MISS
    gc.collect()
    assert all(
        entry.pin_records[-1].released for entry in entries[:missing_component] if entry.pin_records
    )
    state = ctx.get_state(type(manager._request_states[0]))
    assert state.pending_pins == {}


@pytest.mark.parametrize("failed_component", range(COMPONENT_COUNT))
def test_failed_page_pin_is_a_miss_and_releases_earlier_pins(
    failed_component, manager_directory_runtime
):
    manager, directory, runtime = manager_directory_runtime
    entries = _publish_complete(directory, runtime)
    directory.pin_failures.add(entries[failed_component].key.block_hash)
    ctx = ReqContext("request-1")

    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.MISS
    gc.collect()
    assert all(
        entry.pin_records[-1].released for entry in entries[:failed_component] if entry.pin_records
    )


def test_foreign_pool_page_is_a_normal_miss(manager_directory_runtime):
    manager, directory, runtime = manager_directory_runtime
    entries = list(_publish_complete(directory, runtime))
    foreign = FakePoolRef(1, 99, 1)
    entries[2] = _entry(directory, runtime, OFFLOAD_KEY, 2, 6, pool_ref=foreign)
    directory.entries[entries[2].key.block_hash] = entries[2]
    ctx = ReqContext("request-1")

    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.MISS
    with pytest.raises(RuntimeError, match="not pinned"):
        manager.prepare_load([OFFLOAD_KEY], ctx)


def test_prepare_load_releases_scanned_but_unselected_page_bundles(
    manager_directory_runtime,
):
    manager, directory, runtime = manager_directory_runtime
    selected = _publish_complete(directory, runtime, OFFLOAD_KEY, (0, 1, 2, 3))
    unselected = _publish_complete(directory, runtime, OTHER_KEY, (4, 5, 6, 7))
    ctx = ReqContext("request-1")
    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.HIT
    assert manager.lookup(OTHER_KEY, ctx) is LookupResult.HIT

    manager.prepare_load([OFFLOAD_KEY], ctx)
    gc.collect()
    assert all(not entry.pin_records[-1].released for entry in selected)
    assert all(entry.pin_records[-1].released for entry in unselected)


def test_request_finish_releases_pending_but_not_active_page_bundles(
    manager_directory_runtime,
):
    manager, directory, runtime = manager_directory_runtime
    active = _publish_complete(directory, runtime, OFFLOAD_KEY, (0, 1, 2, 3))
    pending = _publish_complete(directory, runtime, OTHER_KEY, (4, 5, 6, 7))
    ctx = ReqContext("request-1")
    manager.lookup(OFFLOAD_KEY, ctx)
    manager.lookup(OTHER_KEY, ctx)
    manager.prepare_load([OFFLOAD_KEY], ctx)

    manager.on_request_finished(ctx)
    gc.collect()
    assert all(entry.pin_records[-1].released for entry in pending)
    assert all(not entry.pin_records[-1].released for entry in active)
    manager.complete_load([OFFLOAD_KEY], ctx)
    gc.collect()
    assert all(entry.pin_records[-1].released for entry in active)


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


def test_store_claims_fragmented_component_slots_in_order(manager_directory_runtime):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY)
    ctx = ReqContext("request-1")

    output = manager.prepare_store([OFFLOAD_KEY], ctx)

    assert output is not None
    assert output.keys_to_store == [OFFLOAD_KEY]
    assert [
        (page.component_id, page.location.pool_id, page.location.slot_id)
        for page in output.store_spec.transfers[0].pages
    ] == [(0, 10, 7), (1, 10, 2), (2, 10, 6), (3, 10, 1)]


def test_store_omits_preexisting_valid_pages_from_worker_spec(
    manager_directory_runtime,
):
    manager, directory, runtime = manager_directory_runtime
    for component_id, slot_id in ((0, 7), (2, 6)):
        entry = _entry(directory, runtime, OFFLOAD_KEY, component_id, slot_id)
        directory.entries[entry.key.block_hash] = entry
        directory.claim_results[entry.key.block_hash] = FakeExistingClaim(entry.slot, True)
    for component_id, slot_id in ((1, 2), (3, 1)):
        directory.claim_results[shared_page_hash(OFFLOAD_KEY, component_id)] = slot_id

    output = manager.prepare_store([OFFLOAD_KEY], ReqContext("request-1"))

    assert output is not None
    assert [page.component_id for page in output.store_spec.transfers[0].pages] == [
        1,
        3,
    ]


@pytest.mark.parametrize("failed_component", [1, 2, 3])
@pytest.mark.parametrize("outcome", ["existing_reserved", "no_space"])
def test_partial_claim_failure_aborts_new_pages_and_rolls_back_logical_admission(
    failed_component, outcome, manager_directory_runtime
):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY)
    page_hash = shared_page_hash(OFFLOAD_KEY, failed_component)
    if outcome == "existing_reserved":
        directory.claim_results[page_hash] = FakeExistingClaim(
            FakeSlotRef(directory.registered.pool_ref, 7), False
        )
    else:
        directory.claim_results[page_hash] = FakeNoSpace()

    output = manager.prepare_store([OFFLOAD_KEY], ReqContext("request-1"))

    assert output is not None
    assert output.keys_to_store == []
    assert len(directory.aborted) == failed_component
    assert directory.live_reservations == []
    assert manager._local_manager._policy.get(OFFLOAD_KEY) is None


def test_unavailable_aborts_current_and_prior_batch_reservations(
    manager_directory_runtime,
):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY)
    _set_reservation_slots(directory, OTHER_KEY, (3, 4, 5, 6))
    directory.claim_results[shared_page_hash(OTHER_KEY, 2)] = FakeUnavailable()
    ctx = ReqContext("request-1")

    with pytest.raises(RuntimeError, match="unavailable"):
        manager.prepare_store([OFFLOAD_KEY, OTHER_KEY], ctx)

    assert directory.live_reservations == []
    assert len(directory.aborted) == 6
    assert manager._local_manager._policy.get(OFFLOAD_KEY) is None
    assert manager._local_manager._policy.get(OTHER_KEY) is None


def test_foreign_pool_existing_claim_aborts_and_raises(manager_directory_runtime):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY, (0, 1, 2, 3))
    component_id = 2
    directory.claim_results[shared_page_hash(OFFLOAD_KEY, component_id)] = FakeExistingClaim(
        FakeSlotRef(FakePoolRef(1, 99, 1), 3), True
    )

    with pytest.raises(RuntimeError, match="foreign pool"):
        manager.prepare_store([OFFLOAD_KEY], ReqContext("request-1"))
    assert len(directory.aborted) == component_id


def test_successful_mixed_publication_owns_only_new_pages(manager_directory_runtime):
    manager, directory, runtime = manager_directory_runtime
    peer_entries = []
    for component_id, slot_id in ((0, 7), (2, 6)):
        entry = _entry(directory, runtime, OFFLOAD_KEY, component_id, slot_id)
        directory.entries[entry.key.block_hash] = entry
        directory.claim_results[entry.key.block_hash] = FakeExistingClaim(entry.slot, True)
        peer_entries.append(entry)
    for component_id, slot_id in ((1, 2), (3, 1)):
        directory.claim_results[shared_page_hash(OFFLOAD_KEY, component_id)] = slot_id
    ctx = ReqContext("request-1")
    output = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert output is not None
    _publish_store(directory, output)

    manager.complete_store(output.keys_to_store, ctx)

    owned = manager._owned_entries[OFFLOAD_KEY]
    assert {entry.key.block_hash for entry in owned} == {
        shared_page_hash(OFFLOAD_KEY, 1),
        shared_page_hash(OFFLOAD_KEY, 3),
    }
    assert all(entry not in owned for entry in peer_entries)


def test_incomplete_publication_evicts_only_attempt_pages_and_rolls_back(
    manager_directory_runtime,
):
    manager, directory, runtime = manager_directory_runtime
    peer = _entry(directory, runtime, OFFLOAD_KEY, 0, 7)
    directory.entries[peer.key.block_hash] = peer
    directory.claim_results[peer.key.block_hash] = FakeExistingClaim(peer.slot, True)
    for component_id, slot_id in ((1, 2), (2, 6), (3, 1)):
        directory.claim_results[shared_page_hash(OFFLOAD_KEY, component_id)] = slot_id
    ctx = ReqContext("request-1")
    output = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert output is not None
    directory.publish(output.store_spec.transfers[0].pages[0].reservation)

    with pytest.raises(RuntimeError, match="complete page set"):
        manager.complete_store(output.keys_to_store, ctx)

    assert directory.entries[peer.key.block_hash] is peer
    assert directory.evicted and peer not in directory.evicted
    assert directory.live_reservations == []
    assert manager._local_manager._policy.get(OFFLOAD_KEY) is None


def test_complete_store_preserves_replacement_with_new_slot_tenancy(
    manager_directory_runtime,
):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY)
    ctx = ReqContext("request-1")
    output = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert output is not None
    _publish_store(directory, output)
    replaced_page = output.store_spec.transfers[0].pages[0]
    replacement = FakeLookupEntry(
        replaced_page.reservation.key,
        FakeSlotRef(
            replaced_page.reservation.slot.pool,
            replaced_page.reservation.slot.slot_id,
            replaced_page.reservation.slot.slot_version + 1,
        ),
    )
    directory.entries[replacement.key.block_hash] = replacement

    with pytest.raises(RuntimeError, match="publication changed"):
        manager.complete_store(output.keys_to_store, ctx)

    assert directory.entries[replacement.key.block_hash] is replacement
    assert replacement not in directory.evicted
    assert manager._local_manager._policy.get(OFFLOAD_KEY) is None


def test_failed_store_drops_pending_ownership_and_local_admission(
    manager_directory_runtime,
):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY)
    ctx = ReqContext("request-1")
    output = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert output is not None
    for page in output.store_spec.transfers[0].pages:
        directory.abort(page.reservation)

    manager.complete_store(output.keys_to_store, ctx, success=False)

    assert manager._local_manager._policy.get(OFFLOAD_KEY) is None
    assert manager._pending_stores == {}


def test_local_eviction_removes_all_owned_pages():
    manager, directory, _ = _manager_directory_runtime(slot_count=4)
    _set_reservation_slots(directory, OFFLOAD_KEY, (0, 1, 2, 3))
    ctx = ReqContext("request-1")
    first = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert first is not None
    _publish_store(directory, first)
    manager.complete_store(first.keys_to_store, ctx)
    owned = manager._owned_entries[OFFLOAD_KEY]

    _set_reservation_slots(directory, OTHER_KEY, (0, 1, 2, 3))
    second = manager.prepare_store([OTHER_KEY], ctx)

    assert second is not None
    assert second.evicted_keys == [OFFLOAD_KEY]
    assert directory.evicted == list(owned)


@pytest.mark.parametrize("cache_policy", ["lru", "arc"])
def test_shared_manager_reuses_upstream_admission_and_victim_policy(cache_policy):
    manager, directory, _ = _manager_directory_runtime(cache_policy=cache_policy, store_threshold=2)
    upstream = CPUOffloadingManager(
        num_blocks=2,
        cache_policy=cache_policy,
        store_threshold=2,
    )
    shared_ctx = ReqContext("shared")
    upstream_ctx = ReqContext("upstream")

    for key in (OFFLOAD_KEY, OTHER_KEY, THIRD_KEY):
        _set_reservation_slots(directory, key)
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

    shared_second = manager.prepare_store([THIRD_KEY], shared_ctx)
    upstream_second = upstream.prepare_store([THIRD_KEY], upstream_ctx)
    assert shared_second is not None and upstream_second is not None
    assert shared_second.keys_to_store == upstream_second.keys_to_store
    assert shared_second.evicted_keys == upstream_second.evicted_keys == [OTHER_KEY]
    assert len(directory.evicted) == COMPONENT_COUNT


def test_hit_pending_does_not_scan_partial_directory(manager_directory_runtime):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY)
    output = manager.prepare_store([OFFLOAD_KEY], ReqContext("store"))
    assert output is not None
    directory.lookup_calls.clear()

    assert manager.lookup(OFFLOAD_KEY, ReqContext("load")) is LookupResult.HIT_PENDING
    assert directory.lookup_calls == []


def test_reset_releases_pins_then_evicts_owned_pages_and_aborts_pending(
    manager_directory_runtime,
):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY)
    ctx = ReqContext("request-1")
    stored = manager.prepare_store([OFFLOAD_KEY], ctx)
    assert stored is not None
    _publish_store(directory, stored)
    manager.complete_store(stored.keys_to_store, ctx)
    owned = manager._owned_entries[OFFLOAD_KEY]
    assert manager.lookup(OFFLOAD_KEY, ctx) is LookupResult.HIT
    manager.prepare_load([OFFLOAD_KEY], ctx)

    _set_reservation_slots(directory, OTHER_KEY, (4, 5, 6, 7))
    pending = manager.prepare_store([OTHER_KEY], ctx)
    assert pending is not None
    pending_reservations = [page.reservation for page in pending.store_spec.transfers[0].pages]
    evict = directory.evict

    def evict_after_pin_release(entry):
        assert entry.pin_records[-1].released is True
        return evict(entry)

    directory.evict = evict_after_pin_release
    manager.reset_cache()

    assert directory.evicted == list(owned)
    assert directory.aborted[-COMPONENT_COUNT:] == pending_reservations
    assert manager._request_states == []
    assert manager._local_manager._policy.get(OFFLOAD_KEY) is None


def test_reset_reclaims_partially_published_pending_store(manager_directory_runtime):
    manager, directory, _ = manager_directory_runtime
    _set_reservation_slots(directory, OFFLOAD_KEY)
    pending = manager.prepare_store([OFFLOAD_KEY], ReqContext("store"))
    assert pending is not None
    pages = pending.store_spec.transfers[0].pages
    published = [directory.publish(page.reservation) for page in pages[:2]]

    manager.reset_cache()

    assert directory.evicted == published
    assert directory.aborted == [page.reservation for page in pages[2:]]
    assert directory.entries == {}
    assert directory.live_reservations == []
