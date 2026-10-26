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

"""Real-device one-pool shared KV transfer tests."""

from __future__ import annotations

import gc
import mmap
import os
import uuid

import pytest
import torch
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LookupResult, ReqContext, make_offload_key

from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.shared_manager import SpyreSharedOffloadingManager
from spyre_inference.v1.kv_offload.shared_spec import component_manifest
from spyre_inference.v1.kv_offload.shared_types import compute_shared_pool_geometry
from spyre_inference.v1.kv_offload.shared_worker import SpyreSharedOffloadingWorker
from tests.kv_offload.hw_helpers import (  # noqa: F401 - fixture used by name
    IMPL_IDS,
    IMPLS,
    LAYOUT_KIND,
    NUM_BLOCKS,
    _assert_bit_exact,
    _bits,
    _cache,
    _fill_all,
    _init_device,
    requires_hardware,
)

OFFLOAD_KEY = make_offload_key(bytes.fromhex("53" * 32), 0)


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_real_kv_allocations_are_single_chunk(impl_cls):
    from torch_spyre._C import get_composite_address

    cache = _cache(impl_cls)
    for role, pages in (("k", cache.k_pages), ("v", cache.v_pages)):
        address = get_composite_address(pages)
        chunks = [(chunk.domain_id, chunk.size) for chunk in address.chunks()]
        print(
            f"{LAYOUT_KIND[impl_cls]} {role}: total_size={address.total_size} "
            f"num_chunks={address.num_chunks} chunks={chunks}"
        )
        assert address.num_chunks == 1, (
            f"M2-F3 required: {LAYOUT_KIND[impl_cls]} {role} allocation has "
            f"{address.num_chunks} chunks: {chunks}"
        )


def _physical(cache, impl_cls) -> SpyrePhysicalCaches:
    return SpyrePhysicalCaches(
        caches=((cache.k_pages, cache.v_pages),),
        layout_kinds=(LAYOUT_KIND[impl_cls],),
        tensor_idx_to_cache={0: 0, 1: 0},
        num_blocks=NUM_BLOCKS,
    )


def _manager(metadata_name, pool_name, max_pool_slots):
    return SpyreSharedOffloadingManager(
        metadata_name=metadata_name,
        pool_name=pool_name,
        component_count=2,
        max_pool_slots=max_pool_slots,
        cache_policy="lru",
        cache_policy_module_path=None,
        enable_events=False,
        store_threshold=0,
        max_tracker_size=64_000,
    )


def _drain(worker):
    [result] = worker.get_finished()
    return result


def _assert_transfer_ok(result, expected_bytes):
    assert result.success is True
    assert result.transfer_size == expected_bytes
    assert result.transfer_time is not None and result.transfer_time > 0


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_shared_worker_round_trip_is_bit_exact_through_fragmented_one_pool(impl_cls):
    from torch_spyre._C import (
        ChunkDescriptorEntry,
        CompatibleBlockKey,
        SharedMetadata,
        get_composite_address,
    )

    metadata_name = f"kv_shared_{os.getpid()}_{uuid.uuid4().hex}"
    pool_name = f"{metadata_name}.data"
    SharedMetadata.unlink_by_name(metadata_name)
    manager_a = manager_b = worker_a = worker_b = directory = registered = None
    dummy_entries = []
    try:
        source = _cache(impl_cls)
        destination = _cache(impl_cls)
        expected_k, expected_v = _fill_all(source, {3: 17})[3]
        destination_before = _fill_all(destination, {6: 91})[6]
        assert not torch.equal(_bits(expected_k), _bits(destination_before[0]))

        source_physical = _physical(source, impl_cls)
        manifest = component_manifest(source_physical)
        slot_bytes = (
            (max(item.page_bytes for item in manifest) + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        ) * mmap.PAGESIZE
        geometry = compute_shared_pool_geometry(
            slot_bytes * 16,
            tuple(item.page_bytes for item in manifest),
            mmap.PAGESIZE,
        )
        expected_bytes = sum(item.page_bytes for item in manifest)
        worker_a = SpyreSharedOffloadingWorker(
            physical=source_physical,
            metadata_name=metadata_name,
            pool_name=pool_name,
            geometry=geometry,
            manifest=manifest,
            compatibility_digest=bytes(range(32)),
            max_pool_slots=geometry.slot_count,
        )
        directory = worker_a._directory
        registered = worker_a._registered

        domain_id = get_composite_address(source.k_pages).chunks()[0].domain_id
        for block_hash in range(0xD00, 0xD04):
            reservation = directory.claim(
                registered.pool_ref,
                CompatibleBlockKey(registered.compatibility, block_hash),
            )
            directory.publish(
                reservation,
                [ChunkDescriptorEntry(domain_id, manifest[0].page_bytes)],
            )
            dummy_entries.append(directory.lookup(reservation.key))
        dummy_slots = [entry.slot.slot_id for entry in dummy_entries]
        max_entry = dummy_entries[dummy_slots.index(max(dummy_slots))]
        min_entry = dummy_entries[dummy_slots.index(min(dummy_slots))]
        directory.evict(max_entry)
        directory.evict(min_entry)
        dummy_entries = [entry for entry in dummy_entries if entry not in (max_entry, min_entry)]

        manager_a = _manager(metadata_name, pool_name, geometry.slot_count)
        store_context = ReqContext("store")
        store = manager_a.prepare_store([OFFLOAD_KEY], store_context)
        assert store is not None and store.keys_to_store == [OFFLOAD_KEY]
        store_slots = [page.location.slot_id for page in store.store_spec.transfers[0].pages]
        assert store_slots == [min(dummy_slots), max(dummy_slots)]
        assert store_slots != list(range(store_slots[0], store_slots[0] + 2))

        assert worker_a.submit_store(
            1,
            GPULoadStoreSpec([3], group_sizes=[1], block_indices=[0]),
            store.store_spec,
        )
        store_result = _drain(worker_a)
        manager_a.complete_store(store.keys_to_store, store_context, success=store_result.success)
        _assert_transfer_ok(store_result, expected_bytes)

        destination_physical = _physical(destination, impl_cls)
        worker_b = SpyreSharedOffloadingWorker(
            physical=destination_physical,
            metadata_name=metadata_name,
            pool_name=pool_name,
            geometry=geometry,
            manifest=component_manifest(destination_physical),
            compatibility_digest=bytes(range(32)),
            max_pool_slots=geometry.slot_count,
        )
        manager_b = _manager(metadata_name, pool_name, geometry.slot_count)
        load_context = ReqContext("load")
        assert manager_b.lookup(OFFLOAD_KEY, load_context) is LookupResult.HIT
        load = manager_b.prepare_load([OFFLOAD_KEY], load_context)
        assert [page.location.slot_id for page in load.transfers[0].pages] == store_slots

        assert worker_b.submit_load(
            2,
            load,
            GPULoadStoreSpec([6], group_sizes=[1], block_indices=[0]),
        )
        load_result = _drain(worker_b)
        manager_b.complete_load([OFFLOAD_KEY], load_context)
        _assert_transfer_ok(load_result, expected_bytes)

        _assert_bit_exact(destination.k_pages[6], expected_k, "shared K reload")
        _assert_bit_exact(destination.v_pages[6], expected_v, "shared V reload")
    finally:
        if manager_b is not None:
            manager_b.reset_cache()
        if manager_a is not None:
            manager_a.reset_cache()
        if directory is not None:
            for entry in dummy_entries:
                directory.evict(entry)
        torch.spyre.synchronize()
        if directory is not None and registered is not None:
            directory.retire_pool(registered.pool_ref)
        del manager_a, manager_b, worker_a, worker_b, directory
        gc.collect()
        SharedMetadata.unlink_by_name(metadata_name)
