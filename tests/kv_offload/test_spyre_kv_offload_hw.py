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

import os

import pytest
import torch

from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl
from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
    SpyreHeadMajorAttentionImpl,
)
from spyre_inference.v1.worker.spyre_kv_offload import SpyreKvPageOffloader

DTYPE = torch.float16
NUM_BLOCKS = 8
BLOCK_SIZE = 128
NUM_KV_HEADS = 8
HEAD_SIZE = 128

IMPLS = [SpyreAttentionImpl, SpyreHeadMajorAttentionImpl]
IMPL_IDS = ["token-major", "head-major"]
LAYOUT_KIND = {
    SpyreAttentionImpl: "token-major",
    SpyreHeadMajorAttentionImpl: "head-major",
}


def _real_device() -> bool:
    """True only on an actual card.

    FLEX_DEVICE=MOCK* satisfies spyre_available() but performs no real DMA, so
    a byte-fidelity test must not run there: it would pass vacuously.
    """
    return not os.environ.get("FLEX_DEVICE", "").upper().startswith("MOCK")


requires_hardware = pytest.mark.skipif(
    not _real_device(),
    reason="byte fidelity requires a real device (FLEX_DEVICE must not be MOCK*)",
)


@pytest.fixture(autouse=True)
def _init_device():
    """Create the RuntimeContext before any device allocation.

    Without this the first `.to('spyre')` raises
    RAS::RUNTIMECONTEXT::ContextNotCreated (0x8c0d). The mock tests never hit it
    because the mock path initializes lazily elsewhere; on a real card the
    context must exist first.
    """
    torch.spyre._impl._lazy_init()


def _spec(head_size=HEAD_SIZE, num_kv_heads=NUM_KV_HEADS, block_size=BLOCK_SIZE):
    from vllm.v1.kv_cache_interface import AttentionSpec

    return AttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=DTYPE,
    )


def _cache(impl_cls, num_blocks=NUM_BLOCKS, **spec_kwargs):
    return impl_cls.allocate_pages(num_blocks, _spec(**spec_kwargs), torch.device("spyre"))


def _pattern(cache, block_id: int, seed: int) -> torch.Tensor:
    """A distinct, non-repeating pattern for one page.

    Distinct per (block, seed) so a page reloaded from the wrong slot or offset
    cannot coincidentally match, and non-constant so an offset error inside the
    page is visible rather than cancelling out.

    Subnormal fp16 values are deliberately excluded. The device write path does
    not preserve them: a subnormal written to the device comes back halved, and
    writing that image back halves it again, so such a value never reaches a
    fixed point. Measured on hardware -- rewriting a device image unchanged left
    53 of 1048576 values moving, every one a subnormal (magnitude <= ~3e-5),
    each reduced by one binade. Normal fp16 values are stable under rewrite.

    That flush-to-zero behaviour is a property of the write path, not of
    offload, so keeping it out of the fixtures stops it from masking or faking a
    real offload defect. randn() alone produces ~50 subnormals per page.
    """
    shape = tuple(cache.k_pages.shape[1:])
    g = torch.Generator().manual_seed(seed * 1000 + block_id)
    vals = torch.randn(shape, generator=g, dtype=torch.float32)
    # fp16 subnormals are those with magnitude < 2^-14; push them well clear.
    tiny = vals.abs() < 2.0**-13
    vals = torch.where(tiny, vals.sign() * 0.5 + (vals.sign() == 0) * 0.5, vals)
    return vals.to(DTYPE)


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Raw bit pattern on the host, for exact comparison."""
    return t.to("cpu").contiguous().view(torch.int16)


def _assert_bit_exact(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    a, e = _bits(actual), _bits(expected)
    if torch.equal(a, e):
        return
    bad = (a != e).sum().item()
    idx = (a != e).nonzero()[0].tolist()
    raise AssertionError(
        f"{what}: {bad}/{a.numel()} fp16 values differ; first at {idx} "
        f"(got bits {a[tuple(idx)].item()}, want {e[tuple(idx)].item()})"
    )


def _fill_all(cache, seeds: dict[int, int]) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Write every page in ONE whole-tensor copy; return the device's own image.

    Per-page writes (`cache.k_pages[i].copy_(x)`) are rejected by the eager DMA
    path -- a page slice's shape does not match the allocation's device sizes
    ("Invalid dma sizes ... Expected: 128 8 128, got: 8 128 8 128"). Reading one
    page back and zeroing one page in place both work; only writing a slice does
    not. So the host builds the full image and it goes down in a single copy.

    `seeds` maps block_id -> seed; blocks not listed are filled with a default
    pattern so no page is left zeroed (a zeroed neighbour would make a spill
    into it invisible).
    """
    num_blocks = int(cache.k_pages.shape[0])
    k_host = torch.empty(tuple(cache.k_pages.shape), dtype=DTYPE)
    v_host = torch.empty(tuple(cache.v_pages.shape), dtype=DTYPE)
    for b in range(num_blocks):
        seed = seeds.get(b, 900 + b)
        k = _pattern(cache, b, seed)
        v = _pattern(cache, b, seed + 500)
        k_host[b] = k
        v_host[b] = v
    cache.k_pages.copy_(k_host)
    cache.v_pages.copy_(v_host)
    torch.spyre.synchronize()

    # Read the expected values back FROM THE DEVICE, not from k_host/v_host.
    #
    # The host->device write is not bit-preserving: writing a host fp16 tensor
    # and reading it straight back (no offload involved) differs in ~50% of
    # values, almost always by exactly +1 in the fp16 bit pattern -- the device
    # write path rounds differently than torch's fp32->fp16 cast. Measured on
    # hardware: 524275 of 524330 differing values had delta exactly +1.
    #
    # That rounding is not what these tests are about. The offload contract is
    # that a page reloads to exactly what was on the device before it was
    # offloaded, so the device's own post-write image is the correct reference.
    # Comparing against the host source instead would fail every test for a
    # reason that has nothing to do with offload.
    k_dev = cache.k_pages.to("cpu")
    v_dev = cache.v_pages.to("cpu")
    return {b: (k_dev[b].clone(), v_dev[b].clone()) for b in range(num_blocks)}


def _zero_pages(cache, blocks) -> dict:
    """Zero only `blocks`, leaving every other page intact.

    `cache.k_pages[b].zero_()` CANNOT be used: an in-place op on a page slice
    wipes the whole allocation. Verified on hardware -- after
    `k_pages[4].zero_()` all 8 pages read back as zero, not just page 4. A page
    slice carries the whole allocation's device layout (views propagate
    spyre_layout verbatim), so the op applies to everything.

    Instead read the current image to the host, zero the rows there, and write
    the whole tensor back -- the only write shape the device accepts.

    Returns refreshed expectations for every page, read back from the device
    after the write, exactly as _fill_all does. The surviving pages are rewritten
    by this call, so their pre-call expectations cannot simply be carried over.
    """
    k = cache.k_pages.to("cpu").clone()
    v = cache.v_pages.to("cpu").clone()
    for b in blocks:
        k[b].zero_()
        v[b].zero_()
    cache.k_pages.copy_(k)
    cache.v_pages.copy_(v)
    torch.spyre.synchronize()

    k_dev = cache.k_pages.to("cpu")
    v_dev = cache.v_pages.to("cpu")
    return {b: (k_dev[b].clone(), v_dev[b].clone()) for b in range(int(cache.k_pages.shape[0]))}


@pytest.fixture
def offloader(request):
    """An offloader on a uniquely-named pool, unlinked before and after.

    Stale POSIX SHM segments survive a crashed run, so the name is unlinked on
    the way in as well as out; otherwise a previous failure's segment would be
    attached and its geometry check would mask the real error.
    """
    from torch_spyre._C import SharedHostPool

    made = []

    def _make(cache, impl_cls, num_slots=2):
        name = f"kv_e2e_{os.getpid()}_{abs(hash(request.node.name)) % 10**6}_{len(made)}"
        SharedHostPool.unlink_by_name(name)
        off = SpyreKvPageOffloader(
            cache, name, num_slots=num_slots, layout_kind=LAYOUT_KIND[impl_cls]
        )
        made.append(name)
        return off

    yield _make

    for name in made:
        try:
            SharedHostPool.unlink_by_name(name)
        except Exception:  # noqa: BLE001 - teardown must not mask a test failure
            pass


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
