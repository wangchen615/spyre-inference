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

"""Tests for SpyreKvPageOffloader.

Caches are built through the impls' real `allocate_pages` so they carry the
production device layouts; a hand-built layout here could drift from the one
the kernels actually use.

These tests check routing, slot pairing and the validator's rejections. They do
not assert byte fidelity after a round trip -- that needs a card and lives with
the hardware cases in the M1 plan.
"""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl
from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
    SpyreHeadMajorAttentionImpl,
)
from spyre_inference.v1.worker.spyre_kv_offload import (
    SpyreKvPageOffloader,
    k_slot,
    page_bytes,
    v_slot,
)

DTYPE = torch.float16
NUM_BLOCKS = 8
BLOCK_SIZE = 128
NUM_KV_HEADS = 8
HEAD_SIZE = 128

IMPLS = [SpyreAttentionImpl, SpyreHeadMajorAttentionImpl]
IMPL_IDS = ["token-major", "head-major"]


def _spec(head_size=HEAD_SIZE, num_kv_heads=NUM_KV_HEADS, block_size=BLOCK_SIZE):
    return AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=DTYPE,
    )


def _cache(impl_cls, num_blocks=NUM_BLOCKS, **spec_kwargs):
    device = torch.device("spyre")
    return impl_cls.allocate_pages(num_blocks, _spec(**spec_kwargs), device)


# --- pairing logic: no device needed ---------------------------------------


def test_k_and_v_slots_are_disjoint():
    """K and V are separate allocations, so one logical page needs two slots."""
    assert [k_slot(s) for s in range(4)] == [0, 2, 4, 6]
    assert [v_slot(s) for s in range(4)] == [1, 3, 5, 7]
    used = [k_slot(s) for s in range(32)] + [v_slot(s) for s in range(32)]
    assert len(set(used)) == 64, "K/V host slots collide"


# --- everything below drives real device allocations -----------------------


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_page_bytes_matches_one_page(impl_cls):
    cache = _cache(impl_cls)
    expected = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * DTYPE.itemsize
    assert page_bytes(cache) == expected


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_pool_has_two_slots_per_logical_page(impl_cls, request):
    cache = _cache(impl_cls)
    off = SpyreKvPageOffloader(cache, request.node.name, num_slots=4)
    assert off.pool.slot_count() == 8
    assert off.pool.slot_bytes() == off.page_size_bytes


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_offload_and_reload_every_block(impl_cls, request):
    cache = _cache(impl_cls)
    off = SpyreKvPageOffloader(cache, request.node.name, num_slots=2)
    for block_id in range(NUM_BLOCKS):
        off.offload(block_id=block_id, slot=block_id % 2)
        off.reload(slot=block_id % 2, block_id=block_id)


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_reload_may_relocate_to_another_block(impl_cls, request):
    """A page is position-independent, so a restore may target a new block."""
    cache = _cache(impl_cls)
    off = SpyreKvPageOffloader(cache, request.node.name, num_slots=1)
    off.offload(block_id=0, slot=0)
    off.reload(slot=0, block_id=NUM_BLOCKS - 1)


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_batch_helpers(impl_cls, request):
    cache = _cache(impl_cls)
    off = SpyreKvPageOffloader(cache, request.node.name, num_slots=4)
    off.offload_many([(0, 0), (1, 1), (2, 2)])
    off.reload_many([(0, 0), (1, 1), (2, 2)])


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_rejects_out_of_range_slot(impl_cls, request):
    cache = _cache(impl_cls)
    off = SpyreKvPageOffloader(cache, request.node.name, num_slots=2)
    with pytest.raises(IndexError, match="out of range"):
        off.offload(block_id=0, slot=2)
    with pytest.raises(IndexError, match="out of range"):
        off.reload(slot=-1, block_id=0)


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_rejects_out_of_range_block(impl_cls, request):
    """Rejected by the torch-spyre validator before any DMA is enqueued."""
    cache = _cache(impl_cls)
    off = SpyreKvPageOffloader(cache, request.node.name, num_slots=1)
    with pytest.raises(RuntimeError, match="out of range"):
        off.offload(block_id=NUM_BLOCKS, slot=0)


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
def test_rejects_page_view_instead_of_cache():
    """The validator requires the whole cache plus a block_id.

    A zero-offset prefix view is the dangerous case: storage_offset() is 0, so
    only the exact num_blocks*inner == device_size[0] identity catches it.
    """
    from torch_spyre._C import copy_kv_page_raw  # ty: ignore[unresolved-import]

    cache = _cache(SpyreAttentionImpl)
    off = SpyreKvPageOffloader(cache, "reject_view", num_slots=1)
    for lo, hi in ((0, 4), (2, 6)):
        with pytest.raises(RuntimeError, match="is not num_blocks"):
            copy_kv_page_raw(cache.k_pages[lo:hi], 0, off.pool, 0, False, False)


@pytest.mark.skipif(not spyre_available(), reason="requires a Spyre device")
def test_rejects_generic_layout():
    """A plain .to('spyre') gets the generic tiled layout, which is not a cache."""
    from torch_spyre._C import copy_kv_page_raw  # ty: ignore[unresolved-import]

    cache = _cache(SpyreAttentionImpl)
    off = SpyreKvPageOffloader(cache, "reject_generic", num_slots=1)
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    generic = torch.zeros(shape, dtype=DTYPE).to("spyre")
    with pytest.raises(RuntimeError, match="rank-4 device layout"):
        copy_kv_page_raw(generic, 0, off.pool, 0, False, False)
