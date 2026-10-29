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

"""Rank-local routing through a globally planned shared host arena."""

from __future__ import annotations

import base64
import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from vllm.v1.kv_offload.base import GPULoadStoreSpec, make_offload_key

from spyre_inference.v1.kv_offload.connector import TOKEN_MAJOR, SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.pool_plan import parse_planned_pool
from spyre_inference.v1.kv_offload.shared_types import (
    SharedBlockTransfer,
    SharedComponentDescriptor,
    SharedLoadStoreSpec,
    SharedPageLocation,
    SharedPageTransfer,
)
from spyre_inference.v1.kv_offload.shared_worker import SpyreSharedOffloadingWorker
from tests.kv_offload.test_pool_plan import valid_planned_pool_document
from tests.kv_offload.test_shared_worker_dispatch import (
    FakeAddress,
    FakeCompatibilityRef,
    FakeDirectory,
    FakePool,
    FakePoolRef,
    FakeRegisteredPool,
    FakeReservation,
    FakeSlotRef,
    FakeTensor,
    _runtime,
)

GENERATION = bytes.fromhex("22" * 16)
OFFLOAD_KEY = make_offload_key(bytes.fromhex("33" * 32), 0)


def _tp4_document() -> dict:
    cache_counts = (1, 2, 2, 1)
    page_bytes = {
        (rank, cache): 3000 + 2000 * cache
        for rank in range(4)
        for cache in range(cache_counts[rank])
    }
    rank_caches = []
    partitions = []
    components = []
    offset = 0
    for rank, cache_count in enumerate(cache_counts):
        for cache in range(cache_count):
            page = page_bytes[rank, cache]
            stride = 4096 if page <= 4096 else 8192
            partition_id = len(partitions)
            name = f"host-r{rank}-c{cache}"
            rank_caches.append(
                {
                    "tpRank": rank,
                    "cacheIndex": cache,
                    "physicalPageBytes": str(page),
                    "layoutKind": TOKEN_MAJOR,
                    "layoutVersion": 1,
                    "localKvHeads": 4 + cache,
                    "headSize": 128,
                }
            )
            partitions.append(
                {
                    "partitionId": str(partition_id),
                    "arenaName": "host-0",
                    "partitionName": name,
                    "tpRank": rank,
                    "cacheIndex": cache,
                    "offsetBytes": str(offset),
                    "numSlots": "4",
                    "slotBytes": str(stride),
                    "physicalPageBytes": str(page),
                }
            )
            for role in ("COMPONENT_ROLE_K", "COMPONENT_ROLE_V"):
                components.append(
                    {
                        "globalComponentId": str(len(components)),
                        "tpRank": rank,
                        "cacheIndex": cache,
                        "role": role,
                        "arenaName": "host-0",
                        "partitionName": name,
                        "partitionId": str(partition_id),
                        "partitionOffsetBytes": str(offset),
                        "partitionSlotBytes": str(stride),
                        "physicalPageBytes": str(page),
                    }
                )
            offset += 4 * stride
    canonical_layout = "".join(
        f"{row['tpRank']}:{row['cacheIndex']}:{row['physicalPageBytes']}:"
        f"{row['layoutKind']}:{row['layoutVersion']}:{row['localKvHeads']}:"
        f"{row['headSize']};"
        for row in rank_caches
    )
    layout_signature = hashlib.sha1(canonical_layout.encode()).hexdigest()
    arena = {
        "name": "host-0",
        "backend": "HOST_DRAM",
        "compatibility": {
            "model": "model-a@revision-1",
            "kvDtype": "float16",
            "blockSize": "128",
            "tensorParallelSize": 4,
            "layoutSignature": layout_signature,
        },
        "partitions": [
            {
                "name": row["partitionName"],
                "tpRank": row["tpRank"],
                "cacheIndex": row["cacheIndex"],
                "offsetBytes": row["offsetBytes"],
                "numSlots": row["numSlots"],
                "slotBytes": row["slotBytes"],
            }
            for row in partitions
        ],
    }
    bytes_per_block = 2 * sum(int(row["slotBytes"]) for row in partitions)
    config = {
        "name": "/model-a-tp4",
        "metadata": {"maxChunks": 1},
        "dataPools": [arena],
    }
    effective_config = copy.deepcopy(config)
    effective_config["dataPools"][0]["numSlots"] = str(offset // 4096)
    effective_config["dataPools"][0]["slotBytes"] = "4096"
    return {
        "code": "OK",
        "poolDescriptor": {
            "name": config["name"],
            "generation": base64.b64encode(GENERATION).decode(),
            "state": "READY",
            "effectiveConfig": effective_config,
        },
        "plan": {
            "normalizedSpec": {
                "planningVersion": 1,
                "name": config["name"],
                "model": "model-a",
                "modelRevision": "revision-1",
                "kvDtype": "float16",
                "blockSize": "128",
                "tensorParallelSize": 4,
                "cpuBytesToUse": str(offset + 123),
                "rankCaches": rank_caches,
            },
            "bytesPerCompleteBlock": str(bytes_per_block),
            "logicalBlockCapacity": "2",
            "allocatedBytes": str(offset),
            "unusedBytes": "123",
            "config": config,
            "partitions": partitions,
            "components": components,
        },
    }


def _manifest(
    *,
    page_bytes: tuple[int, ...] = (3000, 3000, 5000, 5000),
    layouts: tuple[str, ...] = (TOKEN_MAJOR, TOKEN_MAJOR),
    versions: tuple[int, ...] = (1, 1),
    heads: tuple[int, ...] = (4, 5),
    head_sizes: tuple[int, ...] = (128, 128),
) -> tuple[SharedComponentDescriptor, ...]:
    result = []
    for cache_index in range(2):
        for role_index, role in enumerate(("k", "v")):
            result.append(
                SharedComponentDescriptor(
                    component_id=len(result),
                    cache_index=cache_index,
                    role=role,
                    layout_kind=layouts[cache_index],
                    layout_version=versions[cache_index],
                    block_size=128,
                    local_kv_heads=heads[cache_index],
                    head_size=head_sizes[cache_index],
                    page_bytes=page_bytes[2 * cache_index + role_index],
                )
            )
    return tuple(result)


def _make_worker(monkeypatch, *, manifest=None, directory=None):
    plan = parse_planned_pool(_tp4_document())
    events = []
    tensors = tuple(FakeTensor(name) for name in ("k0", "v0", "k1", "v1"))
    addresses = {
        tensor.name: FakeAddress(index + 1, (3000 if index < 2 else 5000) * 4)
        for index, tensor in enumerate(tensors)
    }
    if directory is None:
        directory = FakeDirectory(events)
    directory.events = events
    registrations = {}
    resolved = {}
    for partition in plan.partitions:
        pool_ref = FakePoolRef(1, 100 + partition.partition_id, 1)
        registered = FakeRegisteredPool(
            pool_ref,
            FakeCompatibilityRef(1, partition.partition_id),
            partition.partition_name,
            partition.num_slots,
            partition.slot_bytes,
        )
        registrations[partition.partition_name] = registered
        resolved[pool_ref] = FakePool(
            partition.partition_name, partition.num_slots, partition.slot_bytes
        )
    directory.registered = registrations
    directory.resolved = resolved
    directory.configure_geometry = lambda: None
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
        plan=plan,
        tp_rank=2,
        manifest=manifest or _manifest(),
        runtime_loader=lambda: runtime,
    )
    return worker, plan, directory, events


def _gpu_spec(block_ids):
    return GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)], block_indices=[0])


def _global_block(plan, directory, *, reservation: bool):
    pages = []
    for component in plan.components:
        registered = directory.registered[component.partition_name]
        token = None
        if reservation:
            token = FakeReservation(
                SimpleNamespace(block_hash=1000 + component.global_component_id),
                FakeSlotRef(registered.pool_ref, component.global_component_id % 4),
            )
        pages.append(
            SharedPageTransfer(
                component.global_component_id,
                SharedPageLocation(registered.pool_ref.pool_id, component.global_component_id % 4),
                token,
            )
        )
    return SharedBlockTransfer(OFFLOAD_KEY, tuple(pages))


def test_parsed_plan_retains_rank_cache_layout_contract():
    """Dropping normalized layout facts would let incompatible pages attach."""

    plan = parse_planned_pool(valid_planned_pool_document())

    [cache] = plan.caches_for_rank(1)
    assert (
        cache.tp_rank,
        cache.cache_index,
        cache.physical_page_bytes,
        cache.layout_kind,
        cache.layout_version,
        cache.local_kv_heads,
        cache.head_size,
    ) == (1, 0, 3000, "token-major", 1, 4, 128)


def test_tp4_rank_two_resolves_and_copies_only_its_local_partitions(monkeypatch):
    """Resolving a foreign partition would pin another rank's arena range."""

    worker, plan, directory, events = _make_worker(monkeypatch)
    local = plan.components_for_rank(2)

    assert directory.resolve_calls == [
        directory.registered["host-r2-c0"].pool_ref,
        directory.registered["host-r2-c1"].pool_ref,
    ]

    events.clear()
    block = _global_block(plan, directory, reservation=False)
    worker.submit_load(1, SharedLoadStoreSpec([block]), _gpu_spec([3]))

    assert [event[1:] for event in events if event[0] == "copy"] == [
        ("k0", 3, "host-r2-c0", local[0].global_component_id % 4, True, True),
        ("v0", 3, "host-r2-c0", local[1].global_component_id % 4, True, True),
        ("k1", 3, "host-r2-c1", local[2].global_component_id % 4, True, True),
        ("v1", 3, "host-r2-c1", local[3].global_component_id % 4, True, True),
    ]
    [result] = worker.get_finished()
    assert result.success is True
    assert result.transfer_size == 2 * 3000 + 2 * 5000


def test_k_and_v_use_one_partition_with_independently_reserved_slots(monkeypatch):
    worker, plan, directory, events = _make_worker(monkeypatch)
    local_ids = {item.global_component_id for item in plan.components_for_rank(2)}
    block = _global_block(plan, directory, reservation=True)
    local_pages = [page for page in block.pages if page.component_id in local_ids]
    assert local_pages[0].location.pool_id == local_pages[1].location.pool_id
    assert local_pages[0].location.slot_id != local_pages[1].location.slot_id

    events.clear()
    worker.submit_store(2, _gpu_spec([1]), SharedLoadStoreSpec([block]))

    copies = [event for event in events if event[0] == "copy"]
    assert len(copies) == 4
    assert all("host-r2-" in event[3] for event in copies)
    last_copy = max(index for index, event in enumerate(events) if event[0] == "copy")
    first_publish = min(index for index, event in enumerate(events) if event[0] == "publish")
    assert any(event == ("sync",) for event in events[last_copy + 1 : first_publish])
    [result] = worker.get_finished()
    assert result.success is True


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest(page_bytes=(5000, 5000, 3000, 3000)), "page size"),
        (_manifest(layouts=("head-major", TOKEN_MAJOR)), "layout"),
        (_manifest(versions=(2, 1)), "layout version"),
        (_manifest(heads=(9, 5)), "KV heads"),
        (_manifest(head_sizes=(64, 128)), "head size"),
        (
            tuple(replace(item, cache_index=1 - item.cache_index) for item in _manifest()),
            "reordered",
        ),
        (_manifest()[:2], "cache count"),
        (
            _manifest()
            + tuple(replace(item, component_id=item.component_id + 4) for item in _manifest()[:2]),
            "cache count",
        ),
    ],
)
def test_physical_mismatch_fails_before_attachment(monkeypatch, manifest, message):
    directory = FakeDirectory([])

    with pytest.raises(ValueError, match=message):
        _make_worker(monkeypatch, manifest=manifest, directory=directory)

    assert directory.attach_calls == []


@pytest.mark.parametrize("failure", ["registered", "resolved", "resolved_geometry"])
def test_post_attach_constructor_failure_detaches_once(monkeypatch, failure):
    directory = FakeDirectory([])
    original_find = directory.find_pool
    original_resolve = directory.resolve_pool
    if failure == "registered":
        directory.find_pool = lambda name: (
            replace(original_find(name), slot_bytes=1234)
            if name == "host-r2-c0"
            else original_find(name)
        )
    elif failure == "resolved":
        directory.resolve_pool = lambda pool_ref: (
            None if getattr(pool_ref, "pool_id", -1) == 103 else original_resolve(pool_ref)
        )
    else:
        directory.resolve_pool = lambda pool_ref: (
            replace(original_resolve(pool_ref), size=1234)
            if getattr(pool_ref, "pool_id", -1) == 103
            else original_resolve(pool_ref)
        )

    with pytest.raises(RuntimeError):
        _make_worker(monkeypatch, directory=directory)

    assert directory.detach_calls == [directory]


@pytest.mark.parametrize("invalid", ["foreign_pool", "malformed_token"])
def test_invalid_local_page_fails_without_copy(monkeypatch, invalid):
    worker, plan, directory, events = _make_worker(monkeypatch)
    block = _global_block(plan, directory, reservation=True)
    pages = list(block.pages)
    local_index = next(index for index, page in enumerate(pages) if page.component_id == 6)
    page = pages[local_index]
    if invalid == "foreign_pool":
        pages[local_index] = SharedPageTransfer(
            page.component_id,
            SharedPageLocation(999, page.location.slot_id),
            page.reservation,
        )
    else:
        pages[local_index] = SharedPageTransfer(page.component_id, page.location, SimpleNamespace())
    object.__setattr__(block, "pages", tuple(pages))
    events.clear()

    worker.submit_store(3, _gpu_spec([1]), SharedLoadStoreSpec([block]))

    assert not [event for event in events if event[0] == "copy"]
    [result] = worker.get_finished()
    assert result.success is False


def test_store_failure_aborts_only_valid_rank_local_tokens(monkeypatch):
    worker, plan, directory, events = _make_worker(monkeypatch)
    block = _global_block(plan, directory, reservation=True)
    original_copy = worker._runtime.copy_kv_page_raw
    calls = 0

    def fail_first_local_copy(*args):
        nonlocal calls
        calls += 1
        original_copy(*args)
        if calls == 1:
            raise RuntimeError("injected local copy failure")

    worker._runtime.copy_kv_page_raw = fail_first_local_copy
    events.clear()
    worker.submit_store(4, _gpu_spec([1]), SharedLoadStoreSpec([block]))

    local_hashes = {1000 + item.global_component_id for item in plan.components_for_rank(2)}
    assert {event[1] for event in events if event[0] == "abort"} == local_hashes
    [result] = worker.get_finished()
    assert result.success is False
