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

"""Real-device shared-pool KV transfer tests."""

from __future__ import annotations

import gc
import os
import uuid

import pytest
import torch
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LookupResult, ReqContext, make_offload_key

from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.shared_manager import (
    SpyreSharedOffloadingManager,
)
from spyre_inference.v1.kv_offload.shared_types import SharedPoolFamily
from spyre_inference.v1.kv_offload.shared_worker import (
    SpyreSharedOffloadingWorker,
)
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


def _manager(metadata_name, families):
    return SpyreSharedOffloadingManager(
        metadata_name=metadata_name,
        families=families,
        max_components=2,
        num_blocks=sum(family.slot_count for family in families),
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
def test_shared_worker_round_trip_is_bit_exact_across_attachments(impl_cls):
    from torch_spyre._C import SharedMetadata

    metadata_name = f"kv_shared_{os.getpid()}_{uuid.uuid4().hex}"
    families = (
        SharedPoolFamily(f"{metadata_name}.a", 1),
        SharedPoolFamily(f"{metadata_name}.b", 1),
    )
    component_names = [f"{family.name}.c0.{role}" for family in families for role in ("k", "v")]
    SharedMetadata.unlink_by_name(metadata_name)

    manager_a = manager_b = worker_a = worker_b = directory = None
    try:
        source = _cache(impl_cls)
        destination = _cache(impl_cls)
        expected_k, expected_v = _fill_all(source, {3: 17})[3]
        destination_before = _fill_all(destination, {6: 91})[6]
        assert not torch.equal(_bits(expected_k), _bits(destination_before[0]))

        worker_a = SpyreSharedOffloadingWorker(
            physical=_physical(source, impl_cls),
            metadata_name=metadata_name,
            families=families,
            compatibility_digest=bytes(range(32)),
            max_components=2,
        )
        manager_a = _manager(metadata_name, families)
        store_context = ReqContext("store")
        store = manager_a.prepare_store([OFFLOAD_KEY], store_context)
        assert store is not None and store.keys_to_store == [OFFLOAD_KEY]

        assert worker_a.submit_store(
            1,
            GPULoadStoreSpec([3], group_sizes=[1], block_indices=[0]),
            store.store_spec,
        )
        store_result = _drain(worker_a)
        manager_a.complete_store(store.keys_to_store, store_context, success=store_result.success)
        _assert_transfer_ok(store_result, worker_a._bytes_per_block)

        worker_b = SpyreSharedOffloadingWorker(
            physical=_physical(destination, impl_cls),
            metadata_name=metadata_name,
            families=families,
            compatibility_digest=bytes(range(32)),
            max_components=2,
        )
        manager_b = _manager(metadata_name, families)
        load_context = ReqContext("load")
        assert manager_b.lookup(OFFLOAD_KEY, load_context) is LookupResult.HIT
        load = manager_b.prepare_load([OFFLOAD_KEY], load_context)

        assert worker_b.submit_load(
            2,
            load,
            GPULoadStoreSpec([6], group_sizes=[1], block_indices=[0]),
        )
        load_result = _drain(worker_b)
        manager_b.complete_load([OFFLOAD_KEY], load_context)
        _assert_transfer_ok(load_result, worker_b._bytes_per_block)

        _assert_bit_exact(destination.k_pages[6], expected_k, "shared K reload")
        _assert_bit_exact(destination.v_pages[6], expected_v, "shared V reload")
        directory = worker_a._directory
    finally:
        if manager_b is not None:
            manager_b.reset_cache()
        if manager_a is not None:
            manager_a.reset_cache()
        torch.spyre.synchronize()
        if directory is not None:
            for name in component_names:
                registered = directory.find_pool(name)
                if registered is not None:
                    directory.retire_pool(registered.pool_ref)
        del manager_a, manager_b, worker_a, worker_b, directory
        gc.collect()
        SharedMetadata.unlink_by_name(metadata_name)
