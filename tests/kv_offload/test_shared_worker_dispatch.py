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

"""Shared worker registration and transfer routing without a Spyre device."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from vllm.v1.kv_offload.base import GPULoadStoreSpec, make_offload_key

from spyre_inference.v1.kv_offload.connector import (
    TOKEN_MAJOR,
    SpyrePhysicalCaches,
)
from spyre_inference.v1.kv_offload.shared_types import (
    SharedLoadStoreSpec,
    SharedLocation,
    SharedPoolFamily,
    SharedTransfer,
)
from spyre_inference.v1.kv_offload.shared_worker import (
    COMPATIBILITY_FORMAT_VERSION,
    SpyreSharedOffloadingWorker,
)

PAGE_BYTES = (100, 120, 200, 240)
DIGEST = bytes(range(32))
OFFLOAD_KEY = make_offload_key(bytes.fromhex("11" * 32), 0)
OFFLOAD_KEY_2 = make_offload_key(bytes.fromhex("22" * 32), 0)


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
class FakePoolRef:
    metadata_version: int
    pool_id: int
    pool_version: int


@dataclass(frozen=True)
class FakeRegisteredPool:
    pool_ref: FakePoolRef
    compatibility: object
    name: str


@dataclass(frozen=True)
class FakePool:
    name: str


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
        self.configs_by_name = {}
        self.registered = {}
        self.resolved = {}
        self.entries = {}
        self.fail_resolve = False

    def register_or_attach_pool(self, config):
        self.configs.append(config)
        if config.name in self.registered:
            if config != self.configs_by_name[config.name]:
                raise ValueError(f"shared pool {config.name!r} configuration mismatch")
            return self.registered[config.name]
        registered = FakeRegisteredPool(
            FakePoolRef(1, len(self.configs), 1),
            SimpleNamespace(metadata_version=1, compatibility_id=1),
            config.name,
        )
        self.configs_by_name[config.name] = config
        self.registered[config.name] = registered
        self.resolved[registered.pool_ref] = FakePool(config.name)
        return registered

    def resolve_pool(self, pool_ref):
        if self.fail_resolve:
            return None
        return self.resolved.get(pool_ref)

    def publish(self, reservation, chunks):
        self.events.append(("publish", reservation.key.block_hash))
        registered = next(
            pool for pool in self.registered.values() if pool.pool_ref == reservation.slot.pool
        )
        slot_bytes = self.configs_by_name[registered.name].slot_bytes
        if sum(chunk.size for chunk in chunks) > slot_bytes:
            raise ValueError("chunk descriptor exceeds the claimed pool slot")
        entry = FakeLookupEntry(reservation.key, reservation.slot, tuple(chunks))
        self.entries[reservation.key.block_hash] = entry

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


def _make_worker(monkeypatch, *, addresses=None, directory=None, compatibility_digest=DIGEST):
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
        families=(SharedPoolFamily("alpha", 3), SharedPoolFamily("beta", 2)),
        compatibility_digest=compatibility_digest,
        max_components=4,
        runtime_loader=lambda: runtime,
    )
    return worker, directory, events, tensors


@pytest.fixture
def worker_directory_events(monkeypatch):
    return _make_worker(monkeypatch)


def _anchor(directory, family):
    return directory.registered[f"{family}.c0.k"].pool_ref


def _gpu_spec(block_ids):
    return GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)], block_indices=[0])


def test_registers_every_component_in_family_then_cache_order(
    worker_directory_events,
):
    worker, directory, _, _ = worker_directory_events
    assert [config.name for config in directory.configs] == [
        "alpha.c0.k",
        "alpha.c0.v",
        "alpha.c1.k",
        "alpha.c1.v",
        "beta.c0.k",
        "beta.c0.v",
        "beta.c1.k",
        "beta.c1.v",
    ]
    assert [config.num_slots for config in directory.configs] == [3] * 4 + [2] * 4
    assert [config.slot_bytes for config in directory.configs] == list(PAGE_BYTES) * 2
    assert {
        (config.compatibility.format_version, config.compatibility.data)
        for config in directory.configs
    } == {(COMPATIBILITY_FORMAT_VERSION, tuple(DIGEST))}
    assert worker._bytes_per_block == sum(PAGE_BYTES)


def test_rejects_multi_chunk_allocations_before_registering_pools(monkeypatch):
    addresses = {
        name: FakeAddress(index + 1, PAGE_BYTES[index] * 4)
        for index, name in enumerate(("k0", "v0", "k1", "v1"))
    }
    addresses["v0"] = FakeAddress(2, PAGE_BYTES[1] * 4, num_chunks=2)
    directory = FakeDirectory([])

    with pytest.raises(NotImplementedError, match="M2-F3 required"):
        _make_worker(monkeypatch, addresses=addresses, directory=directory)

    assert directory.configs == []


def test_rejects_a_registered_pool_that_cannot_be_resolved(monkeypatch):
    directory = FakeDirectory([])
    directory.fail_resolve = True

    with pytest.raises(RuntimeError, match="could not be resolved"):
        _make_worker(monkeypatch, directory=directory)


def test_pool_registration_rejects_mismatched_page_geometry(monkeypatch):
    _, directory, _, _ = _make_worker(monkeypatch)
    addresses = {
        name: FakeAddress(index + 1, PAGE_BYTES[index] * 4)
        for index, name in enumerate(("k0", "v0", "k1", "v1"))
    }
    addresses["k0"] = FakeAddress(1, (PAGE_BYTES[0] + 1) * 4)

    with pytest.raises(ValueError, match="configuration mismatch"):
        _make_worker(monkeypatch, addresses=addresses, directory=directory)


def test_pool_registration_rejects_mismatched_compatibility(monkeypatch):
    _, directory, _, _ = _make_worker(monkeypatch)

    with pytest.raises(ValueError, match="configuration mismatch"):
        _make_worker(
            monkeypatch,
            directory=directory,
            compatibility_digest=bytes(reversed(DIGEST)),
        )


def test_store_routes_every_component_through_the_selected_family(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    anchor = _anchor(directory, "alpha")
    reservation = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 101), FakeSlotRef(anchor, 2))
    spec = SharedLoadStoreSpec(
        [SharedTransfer(OFFLOAD_KEY, SharedLocation(anchor.pool_id, 2), reservation)]
    )

    assert worker.submit_store(1, _gpu_spec([7]), spec) is True

    assert [event[1:6] for event in events if event[0] == "copy"] == [
        ("k0", 7, "alpha.c0.k", 2, False),
        ("v0", 7, "alpha.c0.v", 2, False),
        ("k1", 7, "alpha.c1.k", 2, False),
        ("v1", 7, "alpha.c1.v", 2, False),
    ]


def test_load_routes_only_through_the_selected_family(worker_directory_events):
    worker, directory, events, _ = worker_directory_events
    anchor = _anchor(directory, "beta")
    spec = SharedLoadStoreSpec([SharedTransfer(OFFLOAD_KEY, SharedLocation(anchor.pool_id, 1))])

    assert worker.submit_load(2, spec, _gpu_spec([6])) is True

    assert [event[1:6] for event in events if event[0] == "copy"] == [
        ("k0", 6, "beta.c0.k", 1, True),
        ("v0", 6, "beta.c0.v", 1, True),
        ("k1", 6, "beta.c1.k", 1, True),
        ("v1", 6, "beta.c1.v", 1, True),
    ]


def test_store_synchronizes_complete_bundle_before_publishing_anchor_descriptor(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    anchor = _anchor(directory, "alpha")
    first = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 101), FakeSlotRef(anchor, 2))
    second = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 102), FakeSlotRef(anchor, 1))
    spec = SharedLoadStoreSpec(
        [
            SharedTransfer(OFFLOAD_KEY, SharedLocation(anchor.pool_id, 2), first),
            SharedTransfer(OFFLOAD_KEY_2, SharedLocation(anchor.pool_id, 1), second),
        ]
    )
    events.clear()

    assert worker.submit_store(3, _gpu_spec([7, 8]), spec) is True

    assert events == [
        ("sync",),
        ("copy", "k0", 7, "alpha.c0.k", 2, False, True),
        ("copy", "v0", 7, "alpha.c0.v", 2, False, True),
        ("copy", "k1", 7, "alpha.c1.k", 2, False, True),
        ("copy", "v1", 7, "alpha.c1.v", 2, False, True),
        ("copy", "k0", 8, "alpha.c0.k", 1, False, True),
        ("copy", "v0", 8, "alpha.c0.v", 1, False, True),
        ("copy", "k1", 8, "alpha.c1.k", 1, False, True),
        ("copy", "v1", 8, "alpha.c1.v", 1, False, True),
        ("sync",),
        ("publish", 101),
        ("publish", 102),
    ]
    assert directory.entries[101].chunks == (FakeChunkDescriptorEntry(1, 100),)
    [result] = worker.get_finished()
    assert result.success is True
    assert result.transfer_size == 2 * sum(PAGE_BYTES)


def test_load_synchronizes_before_making_completion_visible(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    anchor = _anchor(directory, "beta")
    spec = SharedLoadStoreSpec([SharedTransfer(OFFLOAD_KEY, SharedLocation(anchor.pool_id, 1))])
    events.clear()
    worker._finished_jobs = RecordingResults(events)

    assert worker.submit_load(4, spec, _gpu_spec([6])) is True

    assert events == [
        ("sync",),
        ("copy", "k0", 6, "beta.c0.k", 1, True, True),
        ("copy", "v0", 6, "beta.c0.v", 1, True, True),
        ("copy", "k1", 6, "beta.c1.k", 1, True, True),
        ("copy", "v1", 6, "beta.c1.v", 1, True, True),
        ("sync",),
        ("result", True),
    ]


@pytest.mark.parametrize("invalid_location", ["anchor", "slot"])
def test_validates_every_location_before_copying_any_block(
    worker_directory_events, invalid_location
):
    worker, directory, events, _ = worker_directory_events
    anchor = _anchor(directory, "alpha")
    first = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 101), FakeSlotRef(anchor, 0))
    second = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 102), FakeSlotRef(anchor, 1))
    bad_location = (
        SharedLocation(999, 0)
        if invalid_location == "anchor"
        else SharedLocation(anchor.pool_id, 3)
    )
    spec = SharedLoadStoreSpec(
        [
            SharedTransfer(OFFLOAD_KEY, SharedLocation(anchor.pool_id, 0), first),
            SharedTransfer(OFFLOAD_KEY_2, bad_location, second),
        ]
    )
    events.clear()

    assert worker.submit_store(5, _gpu_spec([7, 8]), spec) is True

    assert [event for event in events if event[0] == "copy"] == []
    assert [event for event in events if event[0] == "abort"] == [
        ("abort", 101),
        ("abort", 102),
    ]
    [result] = worker.get_finished()
    assert result.success is False


def test_copy_failure_synchronizes_before_aborting_every_reservation(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    anchor = _anchor(directory, "alpha")
    first = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 101), FakeSlotRef(anchor, 2))
    second = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 102), FakeSlotRef(anchor, 1))
    spec = SharedLoadStoreSpec(
        [
            SharedTransfer(OFFLOAD_KEY, SharedLocation(anchor.pool_id, 2), first),
            SharedTransfer(OFFLOAD_KEY_2, SharedLocation(anchor.pool_id, 1), second),
        ]
    )
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

    assert worker.submit_store(6, _gpu_spec([7, 8]), spec) is True

    assert events[-3:] == [("sync",), ("abort", 101), ("abort", 102)]
    assert directory.entries == {}
    [result] = worker.get_finished()
    assert result.success is False


def test_second_publish_failure_evicts_published_and_aborts_unpublished(
    worker_directory_events,
):
    worker, directory, events, _ = worker_directory_events
    anchor = _anchor(directory, "alpha")
    first = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 101), FakeSlotRef(anchor, 2))
    second = FakeReservation(FakeCompatibleKey(SimpleNamespace(), 102), FakeSlotRef(anchor, 1))
    spec = SharedLoadStoreSpec(
        [
            SharedTransfer(OFFLOAD_KEY, SharedLocation(anchor.pool_id, 2), first),
            SharedTransfer(OFFLOAD_KEY_2, SharedLocation(anchor.pool_id, 1), second),
        ]
    )
    publish = directory.publish
    publish_count = 0

    def fail_on_second_publish(reservation, chunks):
        nonlocal publish_count
        publish_count += 1
        if publish_count == 2:
            events.append(("publish", reservation.key.block_hash))
            raise RuntimeError("injected publish failure")
        publish(reservation, chunks)

    directory.publish = fail_on_second_publish
    events.clear()

    assert worker.submit_store(7, _gpu_spec([7, 8]), spec) is True

    assert events[-5:] == [
        ("publish", 101),
        ("publish", 102),
        ("sync",),
        ("evict", 101),
        ("abort", 102),
    ]
    assert directory.entries == {}
    [result] = worker.get_finished()
    assert result.success is False
