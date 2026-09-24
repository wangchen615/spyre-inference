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

"""End-to-end KV page offload and reuse on real Spyre hardware.

What these prove that the mock tests cannot: that a page offloaded to shared
host memory and reloaded is **bit-exact**, and that reusing it does not disturb
anything else. The mock accepts a DMA without moving bytes through an IOMMU
mapping, so only a card can close this.

Every assertion is bitwise on the raw fp16 bit patterns (`view(torch.int16)`),
not `allclose`. A tolerance-based check would mask exactly the corruption modes
worth catching -- a page reloaded at the wrong offset, a K page landing in V's
slot, or a stale slot being read back.

Gating: these require a real device and are skipped otherwise. `spyre_available()`
is deliberately NOT used -- it returns true under FLEX_DEVICE=MOCK*, which would
let these "pass" without moving a byte. See `_real_device()`.

Sizing is deliberately conservative: an 8-page cache of 256 KiB pages, copied
synchronously with a synchronize() between phases. An earlier unbounded
2048x128B DMA burst wedged the card; nothing here resembles that pattern.
"""

from __future__ import annotations

import pytest
import torch

# Fixtures and byte-fidelity helpers live in hw_helpers so the connector/worker
# hardware tests share them verbatim. `offloader` and `_init_device` are
# fixtures and must be imported into this module's namespace for pytest to
# collect them.
from tests.kv_offload.hw_helpers import (  # noqa: F401 - fixtures used by name
    DTYPE,
    IMPL_IDS,
    IMPLS,
    LAYOUT_KIND,
    NUM_BLOCKS,
    _assert_bit_exact,
    _bits,
    _cache,
    _fill_all,
    _init_device,
    _pattern,
    _spec,
    _zero_pages,
    offloader,
    requires_hardware,
)


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_round_trip_is_bit_exact(impl_cls, offloader):
    """The core claim: offload, destroy the device copy, reload, recover exactly.

    Zeroing the page between offload and reload is what makes this a real test.
    Without it a no-op reload would pass.
    """
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls)
    k, v = _fill_all(cache, {3: 1})[3]

    off.offload(block_id=3, slot=0)
    torch.spyre.synchronize()

    _zero_pages(cache, [3])
    assert _bits(cache.k_pages[3]).any() == False, "zeroing the device page failed"

    off.reload(slot=0, block_id=3)
    torch.spyre.synchronize()

    _assert_bit_exact(cache.k_pages[3], k, "K page after round trip")
    _assert_bit_exact(cache.v_pages[3], v, "V page after round trip")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_reload_relocates_to_another_block(impl_cls, offloader):
    """A page is position-independent: it may come back into a different block."""
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls)
    # One write covers both: block 6 gets different content from block 1, so a
    # reload that landed in the wrong block would be caught.
    expected = _fill_all(cache, {1: 2, 6: 3})
    k, v = expected[1]

    off.offload(block_id=1, slot=0)
    torch.spyre.synchronize()
    off.reload(slot=0, block_id=6)
    torch.spyre.synchronize()

    _assert_bit_exact(cache.k_pages[6], k, "K relocated 1 -> 6")
    _assert_bit_exact(cache.v_pages[6], v, "V relocated 1 -> 6")
    _assert_bit_exact(cache.k_pages[1], k, "source K must be left intact")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_siblings_untouched_by_offload_and_reload(impl_cls, offloader):
    """A wrong page_bytes or offset would spill into a neighbour.

    This is the check that catches an off-by-one page stride, which a
    single-page round trip cannot see.
    """
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls)
    expected = _fill_all(cache, {b: 10 + b for b in range(NUM_BLOCKS)})

    target = 4
    expected_target = expected[target]
    off.offload(block_id=target, slot=0)
    torch.spyre.synchronize()

    # Zeroing rewrites the whole allocation, so the siblings' expected values
    # are re-read afterwards. The TARGET keeps its pre-offload expectation --
    # that it comes back from the pool is exactly what is being proved.
    after_zero = _zero_pages(cache, [target])
    expected = {b: after_zero[b] for b in after_zero if b != target}
    expected[target] = expected_target

    off.reload(slot=0, block_id=target)
    torch.spyre.synchronize()

    for b, (k, v) in expected.items():
        _assert_bit_exact(cache.k_pages[b], k, f"K page {b} (target={target})")
        _assert_bit_exact(cache.v_pages[b], v, f"V page {b} (target={target})")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_k_and_v_do_not_cross_over(impl_cls, offloader):
    """K and V are separate allocations sharing one pool; prove no crosstalk.

    Reloading must restore K from the K slot and V from the V slot. If the
    pairing were reversed, both pages would still be "valid" page-sized blobs,
    so only comparing against distinct expected content catches it.
    """
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls)
    k, v = _fill_all(cache, {2: 7})[2]
    assert not torch.equal(_bits(k), _bits(v)), "test needs distinct K and V"

    off.offload(block_id=2, slot=0)
    torch.spyre.synchronize()
    _zero_pages(cache, [2])
    off.reload(slot=0, block_id=2)
    torch.spyre.synchronize()

    _assert_bit_exact(cache.k_pages[2], k, "K must come from the K slot")
    _assert_bit_exact(cache.v_pages[2], v, "V must come from the V slot")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_multi_page_eviction_and_restore(impl_cls, offloader):
    """The realistic shape: evict several pages, reuse the device space, restore.

    Between eviction and restore the device pages are overwritten with other
    content, so a reload that silently did nothing would fail here.
    """
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls, num_slots=3)
    blocks = [0, 3, 7]
    expected = _fill_all(cache, {b: 20 + b for b in blocks})

    off.offload_many([(b, i) for i, b in enumerate(blocks)])
    torch.spyre.synchronize()

    # Device space reused by other data. Written as a whole-cache copy (per-page
    # writes are rejected), with different seeds so nothing matches what was
    # evicted -- a reload that silently did nothing would fail below.
    _fill_all(cache, {b: 99 + b for b in blocks})

    off.reload_many([(i, b) for i, b in enumerate(blocks)])
    torch.spyre.synchronize()

    for b in blocks:
        k, v = expected[b]
        _assert_bit_exact(cache.k_pages[b], k, f"K page {b} restored")
        _assert_bit_exact(cache.v_pages[b], v, f"V page {b} restored")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_non_blocking_round_trip(impl_cls, offloader):
    """Asynchronous copies must be correct once synchronize() returns."""
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls)
    k, v = _fill_all(cache, {5: 4})[5]

    off.offload(block_id=5, slot=0, non_blocking=True)
    off.synchronize()
    _zero_pages(cache, [5])
    off.reload(slot=0, block_id=5, non_blocking=True)
    off.synchronize()

    _assert_bit_exact(cache.k_pages[5], k, "K after non-blocking round trip")
    _assert_bit_exact(cache.v_pages[5], v, "V after non-blocking round trip")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_slot_reuse_across_different_pages(impl_cls, offloader):
    """One host slot, used for several pages in turn, must not leak stale bytes."""
    cache = _cache(impl_cls)
    off = offloader(cache, impl_cls, num_slots=1)
    expected = _fill_all(cache, {b: 30 + b for b in (0, 1, 2)})

    for b in (0, 1, 2):
        off.offload(block_id=b, slot=0)
        torch.spyre.synchronize()
        _zero_pages(cache, [b])
        off.reload(slot=0, block_id=b)
        torch.spyre.synchronize()
        k, v = expected[b]
        _assert_bit_exact(cache.k_pages[b], k, f"K page {b} via reused slot")
        _assert_bit_exact(cache.v_pages[b], v, f"V page {b} via reused slot")
