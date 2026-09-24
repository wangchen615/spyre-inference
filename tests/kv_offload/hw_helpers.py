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

"""Shared fixtures and byte-fidelity helpers for hardware KV-offload tests.

Extracted from `test_spyre_kv_offload_hw.py` so the connector/worker hardware
tests can reuse them unchanged. Every device quirk these encode was measured on
a real card, and each one silently invalidates a test if worked around
differently -- see the individual docstrings.
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
