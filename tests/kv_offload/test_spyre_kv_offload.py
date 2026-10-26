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

# ruff: noqa: F811 - imported pytest fixtures are intentionally used by name

import dataclasses

import pytest
import torch
from vllm.v1.kv_cache_interface import AttentionSpec

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl
from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
    SpyreHeadMajorAttentionImpl,
)
from spyre_inference.v1.worker.spyre_kv_offload import (
    PageSignature,
    SpyreKvPageOffloader,
    page_bytes,
    page_signature,
)
from tests.kv_offload.hw_helpers import (  # noqa: F401 - fixtures used by name
    _init_device,
    offloader,
    persistent_pool,
    requires_hardware,
)

DTYPE = torch.float16
NUM_BLOCKS = 8
BLOCK_SIZE = 128
NUM_KV_HEADS = 8
HEAD_SIZE = 128

IMPLS = [SpyreAttentionImpl, SpyreHeadMajorAttentionImpl]
IMPL_IDS = ["token-major", "head-major"]
# The layout an impl produces is not recoverable from the tensor (both fold
# logical dim 1 into device dim 0), so the offloader takes it explicitly.
LAYOUT_KIND = {
    SpyreAttentionImpl: "token-major",
    SpyreHeadMajorAttentionImpl: "head-major",
}


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


def test_fixed_child_routes_k_and_v_to_disjoint_component_slots():
    """Each logical block owns a consecutive run spanning every K/V component."""
    calls = []
    cache = (object(), object())
    pool = object()
    signature = PageSignature(
        layout_kind="token-major",
        device_dtype="torch.float16",
        block_size=128,
        local_kv_heads=8,
        head_size=128,
        page_bytes=4096,
    )

    def copy_fn(*args):
        calls.append(args)

    offloader = SpyreKvPageOffloader(
        cache,
        pool=pool,
        num_slots=4,
        layout_kind="token-major",
        component_count=6,
        component_offset=2,
        signature=signature,
        copy_fn=copy_fn,
    )
    offloader.offload(block_id=7, slot=3, non_blocking=True)

    assert calls == [
        (cache[0], 7, pool, 20, False, True),
        (cache[1], 7, pool, 21, False, True),
    ]


def test_close_releases_the_resolved_fixed_child_handle():
    pool = object()
    offloader = SpyreKvPageOffloader(
        (object(), object()),
        pool=pool,
        num_slots=1,
        layout_kind="token-major",
        component_count=2,
        component_offset=0,
        signature=PageSignature(
            layout_kind="token-major",
            device_dtype="torch.float16",
            block_size=128,
            local_kv_heads=8,
            head_size=128,
            page_bytes=4096,
        ),
        copy_fn=lambda *args: None,
    )

    offloader.close()
    offloader.close()

    assert offloader.pool is None


# --- everything below drives real device allocations -----------------------


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_page_bytes_matches_one_page(impl_cls):
    cache = _cache(impl_cls)
    expected = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * DTYPE.itemsize
    assert page_bytes(cache) == expected


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_pool_has_two_slots_per_logical_page(impl_cls, offloader):
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls, num_slots=4)
    assert off.pool.slot_count() == 8
    assert off.pool.slot_bytes() == off.page_size_bytes


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_offload_and_reload_every_block(impl_cls, offloader):
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls, num_slots=2)
    for block_id in range(NUM_BLOCKS):
        off.offload(block_id=block_id, slot=block_id % 2)
        off.reload(slot=block_id % 2, block_id=block_id)


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_reload_may_relocate_to_another_block(impl_cls, offloader):
    """A page is position-independent, so a restore may target a new block."""
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls, num_slots=1)
    off.offload(block_id=0, slot=0)
    off.reload(slot=0, block_id=NUM_BLOCKS - 1)


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_batch_helpers(impl_cls, offloader):
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls, num_slots=4)
    off.offload_many([(0, 0), (1, 1), (2, 2)])
    off.reload_many([(0, 0), (1, 1), (2, 2)])


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_rejects_out_of_range_slot(impl_cls, offloader):
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls, num_slots=2)
    with pytest.raises(IndexError, match="out of range"):
        off.offload(block_id=0, slot=2)
    with pytest.raises(IndexError, match="out of range"):
        off.reload(slot=-1, block_id=0)


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_rejects_out_of_range_block(impl_cls, offloader):
    """Rejected by the torch-spyre validator before any DMA is enqueued."""
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls, num_slots=1)
    with pytest.raises(RuntimeError, match="out of range"):
        off.offload(block_id=NUM_BLOCKS, slot=0)


@requires_hardware
def test_rejects_page_view_instead_of_cache(offloader):
    """The validator requires the whole cache plus a block_id.

    A zero-offset prefix view is the dangerous case: storage_offset() is 0, so
    only the exact num_blocks*inner == device_size[0] identity catches it.
    """
    from torch_spyre._C import copy_kv_page_raw  # ty: ignore[unresolved-import]

    cache = _cache(SpyreAttentionImpl)
    off = offloader(cache, SpyreAttentionImpl, num_slots=1)
    for lo, hi in ((0, 4), (2, 6)):
        with pytest.raises(RuntimeError, match="full cache"):
            copy_kv_page_raw(cache.k_pages[lo:hi], 0, off.pool, 0, False, False)


@requires_hardware
def test_rejects_generic_layout(offloader):
    """A plain .to('spyre') gets the generic tiled layout, which is not a cache."""
    from torch_spyre._C import copy_kv_page_raw  # ty: ignore[unresolved-import]

    cache = _cache(SpyreAttentionImpl)
    off = offloader(cache, SpyreAttentionImpl, num_slots=1)
    shape = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    generic = torch.zeros(shape, dtype=DTYPE).to("spyre")
    with pytest.raises(RuntimeError, match="rank-4 device layout"):
        copy_kv_page_raw(generic, 0, off.pool, 0, False, False)


# --- page signature (contract check 8) --------------------------------------


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_signature_reports_true_geometry(impl_cls):
    """block_size and local_kv_heads must not be swapped by layout.

    Both layouts describe the same 128-token, 8-head page; only the axis order
    differs. An earlier version inferred the layout from device_size[0] and
    silently transposed these two fields for head-major caches.
    """
    sig = page_signature(_cache(impl_cls), LAYOUT_KIND[impl_cls])
    assert sig.block_size == BLOCK_SIZE
    assert sig.local_kv_heads == NUM_KV_HEADS
    assert sig.head_size == HEAD_SIZE
    assert sig.layout_kind == LAYOUT_KIND[impl_cls]


@requires_hardware
def test_layouts_differ_despite_identical_page_bytes():
    """The case byte length alone cannot catch."""
    tm = page_signature(_cache(SpyreAttentionImpl), "token-major")
    hm = page_signature(_cache(SpyreHeadMajorAttentionImpl), "head-major")
    assert tm.page_bytes == hm.page_bytes, "expected the same page size"
    assert tm != hm, "token-major and head-major pages must not compare equal"


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_matching_signature_accepted(impl_cls, offloader):
    cache = _cache(impl_cls)
    first = offloader(cache, impl_cls, num_slots=2)
    second = offloader(cache, impl_cls, num_slots=2, expect_signature=first.signature)
    assert second.signature == first.signature


@requires_hardware
def test_mismatched_layout_rejected(offloader):
    """Attaching a head-major cache to token-major slots must fail."""
    tm = offloader(_cache(SpyreAttentionImpl), SpyreAttentionImpl, num_slots=2)
    with pytest.raises(ValueError, match="signature mismatch"):
        offloader(
            _cache(SpyreHeadMajorAttentionImpl),
            SpyreHeadMajorAttentionImpl,
            num_slots=2,
            expect_signature=tm.signature,
        )


@requires_hardware
def test_mismatched_head_size_rejected(offloader):
    """Same layout kind, different head_size: also a mismatch."""
    tm = offloader(_cache(SpyreAttentionImpl), SpyreAttentionImpl, num_slots=2)
    with pytest.raises(ValueError, match="signature mismatch"):
        offloader(
            _cache(SpyreAttentionImpl, head_size=64),
            SpyreAttentionImpl,
            num_slots=2,
            expect_signature=tm.signature,
        )


@requires_hardware
def test_bad_layout_kind_rejected():
    with pytest.raises(ValueError, match="layout_kind"):
        page_signature(_cache(SpyreAttentionImpl), "row-major")


def test_signature_is_hashable_and_frozen():
    """Frozen so it can key a slot table, and cannot drift after the check."""
    sig = PageSignature(
        layout_kind="token-major",
        device_dtype="torch.float16",
        block_size=128,
        local_kv_heads=8,
        head_size=128,
        page_bytes=262144,
    )
    assert len({sig, dataclasses.replace(sig)}) == 1
    assert sig != dataclasses.replace(sig, layout_kind="head-major")
    with pytest.raises(dataclasses.FrozenInstanceError):
        sig.block_size = 64  # type: ignore[misc]
