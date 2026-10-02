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

"""One-pool shared worker registration and routing without a Spyre device."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from vllm.v1.kv_offload.base import GPULoadStoreSpec, make_offload_key

from spyre_inference.v1.kv_offload.connector import TOKEN_MAJOR, SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.shared_types import (
    SharedBlockTransfer,
    SharedComponentDescriptor,
    SharedLoadStoreSpec,
    SharedPageLocation,
    SharedPageTransfer,
    SharedPoolGeometry,
)
from spyre_inference.v1.kv_offload.shared_worker import SpyreSharedOffloadingWorker

PAGE_BYTES = (100, 120, 200, 240)
DIGEST = bytes(range(32))
OFFLOAD_KEY = make_offload_key(bytes.fromhex("11" * 32), 0)


@dataclass(frozen=True)
class FakeTensor:
    name: str


@dataclass(frozen=True)
class FakeChunk:
    domain_id: int
    size: int


class FakeAddress:
    def __init__(self, domain_id, total_size, num_chunks=1):
        self.total_size = total_size
        self.num_chunks = num_chunks
        self._chunks = [FakeChunk(domain_id, total_size)] * num_chunks

    def chunks(self):
        return self._chunks


@dataclass(frozen=True)
class FakeCompatibilityDescriptor:
    format_version: int
    data: tuple[int, ...]

    def __init__(self, format_version, data):
        object.__setattr__(self, "format_version", format_version)
        object.__setattr__(self, "data", tuple(data))


@dataclass(frozen=True)
class FakeDataPoolConfig:
    name: str
    kind: object
    num_slots: int
    slot_bytes: int
    compatibility: FakeCompatibilityDescriptor


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

    def __init__(self, max_chunks, pools, capacity):
        object.__setattr__(self, "max_chunks", max_chunks)
        object.__setattr__(self, "pools", tuple(pools))
        object.__setattr__(self, "capacity", capacity)


@dataclass(frozen=True)
class FakeCompatibilityRef:
    metadata_version: int
    compatibility_id: int


@dataclass(frozen=True)
class FakePoolRef:
    metadata_version: int
    pool_id: int
    pool_version: int


@dataclass(frozen=True)
class FakeRegisteredPool:
    pool_ref: FakePoolRef
    compatibility: FakeCompatibilityRef
    name: str
    slot_count: int
    slot_bytes: int


@dataclass(frozen=True)
class FakePool:
    name: str
    count: int
    size: int

    def slot_count(self):
        return self.count

    def slot_bytes(self):
        return self.size


@dataclass(frozen=True)
class FakeCompatibleKey:
    compatibility: object
    block_hash: int


@dataclass(frozen=True)
class FakeSlotRef:
    pool: FakePoolRef
    slot_id: int
    slot_version: int = 1


@dataclass(frozen=True)
class FakeReservation:
    key: FakeCompatibleKey
    slot: FakeSlotRef


@dataclass(frozen=True)
class FakeChunkDescriptorEntry:
    domain_id: int
    size: int


@dataclass(frozen=True)
class FakeLookupEntry:
    key: FakeCompatibleKey
    slot: FakeSlotRef
    chunks: tuple


class RecordingResults(list):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def append(self, result):
        self.events.append(("result", result.success))
        super().append(result)


class FakeDirectory:
    def __init__(self, events):
        self.events = events
        self.config = None
        self.configs = []
        self.config_by_name = {}
        self.registered = {}
        self.resolved = {}
        self.entries = {}
        self.fail_resolve = False
        self.registered_slot_count = None
        self.registered_slot_bytes = None
        self.resolved_slot_count = None
        self.resolved_slot_bytes = None

    def register_or_attach_pool(self, config):
        self.configs.append(config)
        if config.name in self.registered:
            if config != self.config_by_name[config.name]:
                raise ValueError(f"shared pool {config.name!r} configuration mismatch")
            return self.registered[config.name]
        registered = FakeRegisteredPool(
            FakePoolRef(1, 10, 1),
            FakeCompatibilityRef(1, 1),
            config.name,
            self.registered_slot_count or config.num_slots,
            self.registered_slot_bytes or config.slot_bytes,
        )
        self.config_by_name[config.name] = config
        self.registered[config.name] = registered
        self.resolved[registered.pool_ref] = FakePool(
            config.name,
            self.resolved_slot_count or config.num_slots,
            self.resolved_slot_bytes or config.slot_bytes,
        )
        return registered

    def resolve_pool(self, pool_ref):
        if self.fail_resolve:
            return None
        return self.resolved.get(pool_ref)

    def publish(self, reservation, chunks):
        self.events.append(("publish", reservation.key.block_hash))
        entry = FakeLookupEntry(reservation.key, reservation.slot, tuple(chunks))
        self.entries[reservation.key.block_hash] = entry
        return entry

    def lookup(self, key):
        return self.entries.get(key.block_hash)

    def abort(self, reservation):
        self.events.append(("abort", reservation.key.block_hash))

    def evict(self, entry):
        self.events.append(("evict", entry.key.block_hash))
        self.entries.pop(entry.key.block_hash, None)
        return True


def _runtime(directory, addresses, events):
    class FakeSharedMetadata:
        @staticmethod
        def create_or_attach(name, config):
            directory.config = (name, config)
            return directory

    def copy_kv_page_raw(tensor, block_id, pool, slot_id, to_device, non_blocking):
        events.append(
            (
                "copy",
                tensor.name,
                block_id,
                pool.name,
                slot_id,
                to_device,
                non_blocking,
            )
        )

    return SimpleNamespace(
        SharedMetadata=FakeSharedMetadata,
        SharedMetadataConfig=FakeMetadataConfig,
        SharedMetadataCapacity=FakeCapacity,
        SharedDataPoolConfig=FakeDataPoolConfig,
        SharedPoolKind=SimpleNamespace(HOST="host"),
        CompatibilityDescriptor=FakeCompatibilityDescriptor,
        ChunkDescriptorEntry=FakeChunkDescriptorEntry,
        copy_kv_page_raw=copy_kv_page_raw,
        get_composite_address=lambda tensor: addresses[tensor.name],
    )


def _manifest(page_bytes=PAGE_BYTES):
    roles = ("k", "v", "k", "v")
    return tuple(
        SharedComponentDescriptor(
            component_id=index,
            cache_index=index // 2,
            role=roles[index],
            layout_kind=TOKEN_MAJOR,
            layout_version=1,
            block_size=128,
            local_kv_heads=8,
            head_size=128,
            page_bytes=size,
        )
        for index, size in enumerate(page_bytes)
    )


GEOMETRY = SharedPoolGeometry(4, 4096, 2, 8, 32_768)


def _make_worker(
    monkeypatch,
    *,
    addresses=None,
    directory=None,
    compatibility_digest=DIGEST,
    manifest=None,
    geometry=GEOMETRY,
):
    events = []
    tensors = tuple(FakeTensor(name) for name in ("k0", "v0", "k1", "v1"))
    if addresses is None:
        addresses = {
            tensor.name: FakeAddress(index + 1, PAGE_BYTES[index] * 4)
            for index, tensor in enumerate(tensors)
        }
    if directory is None:
        directory = FakeDirectory(events)
    else:
        directory.events = events
    runtime = _runtime(directory, addresses, events)
    from spyre_inference.v1.kv_offload import shared_worker as worker_mod

    monkeypatch.setattr(
        worker_mod.torch,
        "spyre",
        SimpleNamespace(synchronize=lambda: events.append(("sync",))),
        raising=False,
    )
    physical = SpyrePhysicalCaches(
        caches=cast(
            tuple[tuple[torch.Tensor, torch.Tensor], ...],
            ((tensors[0], tensors[1]), (tensors[2], tensors[3])),
        ),
        layout_kinds=(TOKEN_MAJOR, TOKEN_MAJOR),
        tensor_idx_to_cache={0: 0, 1: 0, 2: 1, 3: 1},
        num_blocks=4,
    )
    worker = SpyreSharedOffloadingWorker(
        physical=physical,
        metadata_name="shared-meta",
        pool_name="shared.data",
        geometry=geometry,
        manifest=manifest or _manifest(),
        compatibility_digest=compatibility_digest,
        max_pool_slots=16,
        runtime_loader=lambda: runtime,
    )
    return worker, directory, events, tensors


@pytest.fixture
def worker_directory_events(monkeypatch):
    return _make_worker(monkeypatch)


def _gpu_spec(block_ids):
    return GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)], block_indices=[0])


def _page(directory, component_id, slot_id, *, reservation=True, key_hash=None):
    pool_ref = directory.registered["shared.data"].pool_ref
    value = None
    if reservation:
        value = FakeReservation(
            FakeCompatibleKey(SimpleNamespace(), key_hash or 100 + component_id),
            FakeSlotRef(pool_ref, slot_id),
        )
    return SharedPageTransfer(
        component_id,
        SharedPageLocation(pool_ref.pool_id, slot_id),
        value,
    )


def _block(directory, slots=(7, 2, 6, 1), *, reservation=True):
    return SharedBlockTransfer(
        OFFLOAD_KEY,
        tuple(
            _page(directory, component_id, slot_id, reservation=reservation)
            for component_id, slot_id in enumerate(slots)
        ),
    )


def test_registers_exactly_one_pool_with_rfc_geometry(worker_directory_events):
    worker, directory, _, _ = worker_directory_events
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
    assert worker._bytes_per_block == sum(PAGE_BYTES)


def test_rejects_multi_chunk_allocations_before_registering_pool(monkeypatch):
    addresses = {
        name: FakeAddress(index + 1, PAGE_BYTES[index] * 4)
        for index, name in enumerate(("k0", "v0", "k1", "v1"))
    }
    addresses["v0"] = FakeAddress(2, PAGE_BYTES[1] * 4, num_chunks=2)
    directory = FakeDirectory([])
    with pytest.raises(NotImplementedError, match="single chunk"):
        _make_worker(monkeypatch, addresses=addresses, directory=directory)
    assert directory.configs == []


def test_rejects_non_divisible_component_allocation(monkeypatch):
    addresses = {
        name: FakeAddress(index + 1, PAGE_BYTES[index] * 4)
        for index, name in enumerate(("k0", "v0", "k1", "v1"))
    }
    addresses["v0"] = FakeAddress(2, PAGE_BYTES[1] * 4 + 1)
    with pytest.raises(ValueError, match="does not divide"):
        _make_worker(monkeypatch, addresses=addresses)


def test_rejects_runtime_page_size_that_differs_from_manifest(monkeypatch):
    addresses = {
        name: FakeAddress(index + 1, PAGE_BYTES[index] * 4)
        for index, name in enumerate(("k0", "v0", "k1", "v1"))
    }
    addresses["v0"] = FakeAddress(2, (PAGE_BYTES[1] + 1) * 4)
    with pytest.raises(ValueError, match="manifest"):
        _make_worker(monkeypatch, addresses=addresses)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("registered_slot_count", 7), ("registered_slot_bytes", 8192)],
)
def test_rejects_registered_pool_geometry_mismatch(monkeypatch, attribute, value):
    directory = FakeDirectory([])
    setattr(directory, attribute, value)
    with pytest.raises(RuntimeError, match="registered pool geometry"):
        _make_worker(monkeypatch, directory=directory)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [("resolved_slot_count", 7), ("resolved_slot_bytes", 8192)],
)
def test_rejects_resolved_pool_geometry_mismatch(monkeypatch, attribute, value):
    directory = FakeDirectory([])
    setattr(directory, attribute, value)
    with pytest.raises(RuntimeError, match="resolved pool geometry"):
        _make_worker(monkeypatch, directory=directory)


def test_rejects_registered_pool_that_cannot_be_resolved(monkeypatch):
    directory = FakeDirectory([])
    directory.fail_resolve = True
    with pytest.raises(RuntimeError, match="could not be resolved"):
        _make_worker(monkeypatch, directory=directory)


def test_registration_rejects_mismatched_compatibility(monkeypatch):
    _, directory, _, _ = _make_worker(monkeypatch)
    with pytest.raises(ValueError, match="configuration mismatch"):
        _make_worker(
            monkeypatch,
            directory=directory,
            compatibility_digest=bytes(reversed(DIGEST)),
        )


@pytest.mark.parametrize("to_device", [False, True])
def test_routes_non_contiguous_component_pages_through_one_pool(worker_directory_events, to_device):
    worker, directory, events, _ = worker_directory_events
    block = _block(directory, reservation=not to_device)
    spec = SharedLoadStoreSpec([block])
    events.clear()

    if to_device:
        assert worker.submit_load(1, spec, _gpu_spec([3])) is True
    else:
        assert worker.submit_store(1, _gpu_spec([3]), spec) is True

    assert [event[1:] for event in events if event[0] == "copy"] == [
        ("k0", 3, "shared.data", 7, to_device, True),
        ("v0", 3, "shared.data", 2, to_device, True),
        ("k1", 3, "shared.data", 6, to_device, True),
        ("v1", 3, "shared.data", 1, to_device, True),
    ]


def test_partial_store_copies_and_counts_only_new_pages(worker_directory_events):
    worker, directory, events, _ = worker_directory_events
    block = SharedBlockTransfer(
        OFFLOAD_KEY,
        (_page(directory, 1, 2), _page(directory, 3, 1)),
    )
    events.clear()

    worker.submit_store(2, _gpu_spec([3]), SharedLoadStoreSpec([block]))

    assert [event[1] for event in events if event[0] == "copy"] == ["v0", "v1"]
    [result] = worker.get_finished()
    assert result.success is True
    assert result.transfer_size == PAGE_BYTES[1] + PAGE_BYTES[3]


def test_load_requires_every_component(worker_directory_events):
    worker, directory, events, _ = worker_directory_events
    block = SharedBlockTransfer(
        OFFLOAD_KEY,
        tuple(_page(directory, index, index, reservation=False) for index in range(3)),
    )
    events.clear()
    worker.submit_load(3, SharedLoadStoreSpec([block]), _gpu_spec([3]))
    assert not [event for event in events if event[0] == "copy"]
    [result] = worker.get_finished()
    assert result.success is False


@pytest.mark.parametrize("invalid", ["unknown", "duplicate", "foreign", "slot"])
def test_invalid_page_routes_fail_before_any_copy(worker_directory_events, invalid):
    worker, directory, events, _ = worker_directory_events
    block = _block(directory)
    pages = list(block.pages)
    if invalid == "unknown":
        pages[2] = _page(directory, 9, 6)
    elif invalid == "duplicate":
        pages[2] = _page(directory, 1, 6)
    elif invalid == "foreign":
        pages[2] = SharedPageTransfer(
            2,
            SharedPageLocation(99, 6),
            pages[2].reservation,
        )
    else:
        pages[2] = _page(directory, 2, 8)
    object.__setattr__(block, "pages", tuple(pages))
    events.clear()

    worker.submit_store(4, _gpu_spec([3]), SharedLoadStoreSpec([block]))

    assert not [event for event in events if event[0] == "copy"]
    [result] = worker.get_finished()
    assert result.success is False


def test_wrong_block_count_and_direction_modes_fail_before_copy(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    load = SharedLoadStoreSpec([_block(directory, reservation=False)])
    store = SharedLoadStoreSpec([_block(directory)])
    worker.submit_load(5, load, _gpu_spec([1, 2]))
    worker.submit_load(6, store, _gpu_spec([3]))
    worker.submit_store(7, _gpu_spec([3]), load)
    assert not [event for event in events if event[0] == "copy"]
    assert [result.success for result in worker.get_finished()] == [False, False, False]


def test_store_shape_validation_aborts_every_reservation(worker_directory_events):
    worker, directory, events, _ = worker_directory_events
    block = _block(directory)
    events.clear()

    worker.submit_store(
        8,
        _gpu_spec([1, 2]),
        SharedLoadStoreSpec([block]),
    )

    assert not [event for event in events if event[0] == "copy"]
    assert [event for event in events if event[0] == "abort"] == [
        ("abort", 100),
        ("abort", 101),
        ("abort", 102),
        ("abort", 103),
    ]
    [result] = worker.get_finished()
    assert result.success is False


def test_store_fences_then_publishes_each_component_descriptor(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    block = _block(directory)
    events.clear()
    worker._finished_jobs = RecordingResults(events)

    worker.submit_store(8, _gpu_spec([3]), SharedLoadStoreSpec([block]))

    assert events == [
        ("sync",),
        ("copy", "k0", 3, "shared.data", 7, False, True),
        ("copy", "v0", 3, "shared.data", 2, False, True),
        ("copy", "k1", 3, "shared.data", 6, False, True),
        ("copy", "v1", 3, "shared.data", 1, False, True),
        ("sync",),
        ("publish", 100),
        ("publish", 101),
        ("publish", 102),
        ("publish", 103),
        ("result", True),
    ]
    assert directory.entries[100].chunks == (FakeChunkDescriptorEntry(1, 100),)
    assert directory.entries[101].chunks == (FakeChunkDescriptorEntry(2, 120),)
    assert directory.entries[102].chunks == (FakeChunkDescriptorEntry(3, 200),)
    assert directory.entries[103].chunks == (FakeChunkDescriptorEntry(4, 240),)
    [result] = worker.get_finished()
    assert result.transfer_size == sum(PAGE_BYTES)


def test_load_fences_before_success_result(worker_directory_events):
    worker, directory, events, _ = worker_directory_events
    spec = SharedLoadStoreSpec([_block(directory, reservation=False)])
    events.clear()
    worker._finished_jobs = RecordingResults(events)
    worker.submit_load(9, spec, _gpu_spec([3]))
    assert events[-2:] == [("sync",), ("result", True)]
    [result] = worker.get_finished()
    assert result.transfer_size == sum(PAGE_BYTES)


def test_copy_failure_quiesces_then_aborts_every_unpublished_page(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    block = _block(directory)
    copy = worker._runtime.copy_kv_page_raw
    copy_count = 0

    def fail_on_third_copy(*args):
        nonlocal copy_count
        copy_count += 1
        copy(*args)
        if copy_count == 3:
            raise RuntimeError("injected copy failure")

    worker._runtime.copy_kv_page_raw = fail_on_third_copy
    events.clear()
    worker.submit_store(10, _gpu_spec([3]), SharedLoadStoreSpec([block]))

    assert events[-5:] == [
        ("sync",),
        ("abort", 100),
        ("abort", 101),
        ("abort", 102),
        ("abort", 103),
    ]
    assert directory.entries == {}
    [result] = worker.get_finished()
    assert result.success is False
    assert result.transfer_size is None


def test_third_publish_failure_evicts_attempt_pages_and_preserves_peer(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    block = _block(directory)
    peer = FakeLookupEntry(
        FakeCompatibleKey(SimpleNamespace(), 999),
        FakeSlotRef(directory.registered["shared.data"].pool_ref, 0),
        (),
    )
    directory.entries[999] = peer
    publish = directory.publish
    publish_count = 0

    def fail_on_third_publish(reservation, chunks):
        nonlocal publish_count
        publish_count += 1
        if publish_count == 3:
            events.append(("publish", reservation.key.block_hash))
            raise RuntimeError("injected publish failure")
        publish(reservation, chunks)

    directory.publish = fail_on_third_publish
    events.clear()
    worker._finished_jobs = RecordingResults(events)
    worker.submit_store(11, _gpu_spec([3]), SharedLoadStoreSpec([block]))

    assert events[-7:] == [
        ("publish", 102),
        ("sync",),
        ("evict", 100),
        ("evict", 101),
        ("abort", 102),
        ("abort", 103),
        ("result", False),
    ]
    assert directory.entries == {999: peer}
    [result] = worker.get_finished()
    assert result.success is False


def test_publish_failure_preserves_replacement_with_new_slot_tenancy(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    block = _block(directory)
    publish = directory.publish
    replacement = None
    publish_count = 0

    def replace_on_third_publish(reservation, chunks):
        nonlocal publish_count, replacement
        publish_count += 1
        if publish_count == 3:
            replacement = FakeLookupEntry(
                reservation.key,
                FakeSlotRef(
                    reservation.slot.pool,
                    reservation.slot.slot_id,
                    reservation.slot.slot_version + 1,
                ),
                (),
            )
            directory.entries[reservation.key.block_hash] = replacement
            raise RuntimeError("reservation was replaced")
        publish(reservation, chunks)

    directory.publish = replace_on_third_publish
    events.clear()
    worker.submit_store(12, _gpu_spec([3]), SharedLoadStoreSpec([block]))

    assert replacement is not None
    assert directory.entries[replacement.key.block_hash] is replacement
    assert ("evict", replacement.key.block_hash) not in events
    [result] = worker.get_finished()
    assert result.success is False
