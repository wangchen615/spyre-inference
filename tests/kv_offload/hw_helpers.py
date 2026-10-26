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

import gc
import os
import subprocess
import time
import uuid
from pathlib import Path

import pytest
import torch

from scripts.cleanup_shared_kv_demo import cleanup_pool
from scripts.shared_kv_two_instance_demo import PreparedDemoPool, prepare_demo_pool
from scripts.shared_pool_delegator import SharedPoolCtl
from spyre_inference.v1.attention.backends.spyre_attn import SpyreAttentionImpl
from spyre_inference.v1.attention.backends.spyre_head_major_attn import (
    SpyreHeadMajorAttentionImpl,
)
from spyre_inference.v1.kv_offload.shared_attachment import SharedPoolAttachment
from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
from spyre_inference.v1.worker.spyre_kv_offload import SpyreKvPageOffloader, page_signature

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


def _delegator_artifacts() -> tuple[Path, Path]:
    build = Path(os.environ.get("DELEGATOR_BUILD_DIR", ""))
    return build / "shared-pool-delegator", build / "shared-poolctl"


def _wait_for_delegator(ctl: Path, socket: Path, process: subprocess.Popen) -> None:
    for _ in range(200):
        if process.poll() is not None:
            raise RuntimeError("shared-pool-delegator exited before becoming ready")
        result = subprocess.run(
            [str(ctl), "--socket", str(socket), "health"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        time.sleep(0.05)
    raise RuntimeError("shared-pool-delegator did not become ready")


@pytest.fixture
def persistent_pool(tmp_path):
    """Create complete READY generations and destroy them after consumers detach."""

    daemon, ctl_binary = _delegator_artifacts()
    if not daemon.is_file() or not ctl_binary.is_file():
        pytest.skip("DELEGATOR_BUILD_DIR does not contain delegator test artifacts")

    socket = tmp_path / "delegator.sock"
    process = subprocess.Popen(
        [str(daemon), "--socket-path", str(socket)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    prepared_pools: list[PreparedDemoPool] = []
    try:
        _wait_for_delegator(ctl_binary, socket, process)
        ctl = SharedPoolCtl(binary=ctl_binary, socket_path=socket)

        def _create(
            *,
            num_slots: int,
            slot_bytes: int,
            block_size: int = BLOCK_SIZE,
        ) -> SharedPoolIdentity:
            prepared = prepare_demo_pool(
                ctl=ctl,
                pool_name=f"/spyre-kv-hw-{os.getpid()}-{uuid.uuid4().hex}",
                data_pool_name="host-0",
                model="spyre-kv-hardware-test",
                cpu_bytes_to_use=num_slots * slot_bytes,
                slot_bytes=slot_bytes,
                block_size=block_size,
            )
            prepared_pools.append(prepared)
            return SharedPoolIdentity(
                prepared.pool_name,
                bytes.fromhex(prepared.generation),
                prepared.data_pool_name,
            )

        yield _create
        gc.collect()
        for prepared in reversed(prepared_pools):
            cleanup_pool(prepared.pool_name, prepared.generation, ctl=ctl)
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


@pytest.fixture(autouse=True)
def _init_device():
    """Create the RuntimeContext before any device allocation.

    Without this the first `.to('spyre')` raises
    RAS::RUNTIMECONTEXT::ContextNotCreated (0x8c0d). The mock tests never hit it
    because the mock path initializes lazily elsewhere; on a real card the
    context must exist first.
    """
    if not _real_device():
        return
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
def offloader(persistent_pool):
    """Attach an offloader to a delegator-created persistent generation."""

    attachments: list[SharedPoolAttachment] = []
    offloaders: list[SpyreKvPageOffloader] = []

    def _make(cache, impl_cls, num_slots=2, expect_signature=None):
        signature = page_signature(cache, LAYOUT_KIND[impl_cls])
        identity = persistent_pool(
            num_slots=num_slots * 2,
            slot_bytes=signature.page_bytes,
            block_size=signature.block_size,
        )
        attachment = SharedPoolAttachment.attach(identity)
        attachments.append(attachment)
        offloader = SpyreKvPageOffloader(
            cache=cache,
            pool=attachment.data_pool,
            num_slots=num_slots,
            layout_kind=LAYOUT_KIND[impl_cls],
            component_count=2,
            component_offset=0,
            signature=signature,
            expect_signature=expect_signature,
        )
        offloaders.append(offloader)
        return offloader

    yield _make

    for offloader in reversed(offloaders):
        offloader.close()
    for attachment in reversed(attachments):
        attachment.close()
    gc.collect()
