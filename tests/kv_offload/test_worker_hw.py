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

"""`SpyreOffloadingWorker` byte fidelity on real Spyre hardware.

These assert **page-level containment**: a transfer of N blocks moves exactly
those N page pairs and nothing adjacent, in either direction. That is the property
the worker's correctness rests on, and it is not implied by the offloader-level
tests in `test_spyre_kv_offload_hw.py`, which exercise one page at a time and so
cannot see a transfer that moves more than it was asked to.

A note on what these do *not* prove. They were written expecting to discriminate
against the draft worker's rangeless `copy_tensor_raw(cache.tensor[block_id],
pool, slot)`, on the belief that a block view does not narrow the device address.
That belief is out of date: the installed torch-spyre derives a `Range` from the
view (`spyre_mem.cpp:1090-1107`) and flex bounds-checks the host side, so the
draft's path is range-correct too and these tests pass under it. They were run
against it to check exactly that. What still distinguishes `copy_kv_page_raw` is
validation, not containment -- it rejects an out-of-range block id and a
non-rank-4 tensor that `copy_tensor_raw` accepts silently (see `worker.py`).

So read these as containment regression tests against a future change to the
transfer path -- batching, striding, a layout change -- not as evidence about the
draft. Reading neighbouring host slots and neighbouring device blocks is what
makes a spill visible at all: comparing whole device tensors after a round trip
re-reads the same bytes and cancels the error out.

Assertions are bitwise on raw fp16 patterns via `_bits`. Expectations are always
re-read from the device after a write, never taken from the host source: H2D is
not bit-preserving (see `hw_helpers._fill_all`).

Sizing stays at the proven conservative shape -- a few 256 KiB pages, moved
synchronously with a synchronize() between phases.
"""

from __future__ import annotations

import os

import pytest
import torch
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.worker import SpyreOffloadingWorker
from tests.kv_offload.hw_helpers import (  # noqa: F401 - fixtures used by name
    IMPL_IDS,
    IMPLS,
    LAYOUT_KIND,
    _assert_bit_exact,
    _bits,
    _cache,
    _fill_all,
    _init_device,
    _zero_pages,
    requires_hardware,
)

NUM_HOST_BLOCKS = 4


def _gpu_spec(block_ids: list[int]) -> GPULoadStoreSpec:
    """A single-KV-group GPU spec.

    `group_sizes`/`block_indices` are per KV cache group; M1 is full attention,
    so there is exactly one group covering every block.
    """
    return GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)], block_indices=[0])


def _physical(caches, impl_cls) -> SpyrePhysicalCaches:
    """Build the DMA-side descriptor directly, bypassing canonicalization.

    The connector's `spyre_paged_to_canonical` needs a KVCacheConfig and a
    populated static_forward_context; neither is available without a model, and
    neither is what these tests are about.
    """
    return SpyrePhysicalCaches(
        caches=tuple((c.k_pages, c.v_pages) for c in caches),
        layout_kinds=tuple(LAYOUT_KIND[impl_cls] for _ in caches),
        tensor_idx_to_cache={i: i // 2 for i in range(2 * len(caches))},
        num_blocks=int(caches[0].k_pages.shape[0]),
    )


@pytest.fixture
def worker(request):
    """A worker whose pools are unlinked on both ends.

    Stale POSIX SHM segments outlive a crashed run, and `create_or_attach` will
    happily attach one whose geometry matches -- its bytes then read back as a
    "valid" page. Unlinking on the way in as well as out keeps a previous
    failure from masking the next one.
    """
    from torch_spyre._C import SharedHostPool

    prefixes: list[tuple[str, int]] = []

    def _make(caches, impl_cls, num_host_blocks=NUM_HOST_BLOCKS):
        prefix = f"kv_wrk_{os.getpid()}_{abs(hash(request.node.name)) % 10**6}"
        for c_idx in range(len(caches)):
            SharedHostPool.unlink_by_name(f"{prefix}_c{c_idx}")
        prefixes.append((prefix, len(caches)))
        return SpyreOffloadingWorker(
            physical=_physical(caches, impl_cls),
            num_host_blocks=num_host_blocks,
            pool_prefix=prefix,
        )

    yield _make

    for prefix, count in prefixes:
        for c_idx in range(count):
            try:
                SharedHostPool.unlink_by_name(f"{prefix}_c{c_idx}")
            except Exception:  # noqa: BLE001 - teardown must not mask a failure
                pass


def _drain_ok(wrk: SpyreOffloadingWorker, expect_bytes: int | None = None) -> None:
    """Assert exactly one successful result, with instrumentation populated."""
    results = wrk.get_finished()
    assert len(results) == 1, f"expected 1 result, got {len(results)}"
    result = results[0]
    assert result.success, "transfer reported failure; see the logged traceback"
    if expect_bytes is not None:
        assert result.transfer_size == expect_bytes
    assert result.transfer_time is not None and result.transfer_time >= 0.0


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_single_block_store_does_not_touch_neighbouring_host_slots(impl_cls, worker):
    """Storing one block must write one page pair, not the whole allocation.

    Host block 1 is stored; host blocks 0 and 2 are then reloaded into device
    blocks and must read back as zero, because nothing was ever stored to them
    and a fresh pool is zero-filled (POSIX shared memory is zeroed on creation,
    and the fixture unlinks the name first so this is always the creator path).

    Reading the *untouched host slots* is the point. A transfer that moved the
    whole 8-page allocation instead of one page would land bytes in the slots
    around slot 2, and they would show up here as non-zero. The equivalent check
    on the device side cannot see it: a round trip re-reads whatever was written,
    so an overflow cancels itself out.
    """
    cache = _cache(impl_cls)
    wrk = worker([cache], impl_cls)
    expected = _fill_all(cache, {3: 11})

    wrk.submit_store(1, _gpu_spec([3]), CPULoadStoreSpec([1]))
    torch.spyre.synchronize()
    _drain_ok(wrk, expect_bytes=wrk._bytes_per_block)

    # Reload the two host blocks that were never written into scratch device
    # blocks. Block 3 (the one actually stored) is left alone so the reload
    # cannot be satisfied from it.
    wrk.submit_load(2, CPULoadStoreSpec([0, 2]), _gpu_spec([0, 1]))
    torch.spyre.synchronize()
    _drain_ok(wrk, expect_bytes=2 * wrk._bytes_per_block)

    k_dev = cache.k_pages.to("cpu")
    v_dev = cache.v_pages.to("cpu")
    for dev_blk, host_blk in ((0, 0), (1, 2)):
        for name, got in (("k", k_dev[dev_blk]), ("v", v_dev[dev_blk])):
            nonzero = int(_bits(got).ne(0).sum().item())
            assert nonzero == 0, (
                f"{name} page reloaded from untouched host slot {host_blk} has "
                f"{nonzero} non-zero fp16 values: the store to host block 1 "
                f"spilled past its own slot"
            )

    # The stored block itself must be unharmed by the reloads above.
    _assert_bit_exact(cache.k_pages.to("cpu")[3], expected[3][0], "k page 3 after reloads")
    _assert_bit_exact(cache.v_pages.to("cpu")[3], expected[3][1], "v page 3 after reloads")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_single_block_load_does_not_touch_neighbouring_device_blocks(impl_cls, worker):
    """Loading one block must overwrite one page pair, not the whole allocation.

    The mirror of the store test, on the H2D side: store block 2, refill the
    entire cache with different content, load block 2 back, and require every
    *other* device block to still hold its post-refill bytes bit-exactly. An H2D
    that rewrote the whole allocation from one slot would clobber them all.
    """
    cache = _cache(impl_cls)
    wrk = worker([cache], impl_cls)
    before = _fill_all(cache, {2: 21})

    wrk.submit_store(1, _gpu_spec([2]), CPULoadStoreSpec([1]))
    torch.spyre.synchronize()
    _drain_ok(wrk)

    # Overwrite every page, so a correct load restores block 2 only and the
    # other blocks keep these new values.
    after_refill = _fill_all(cache, {b: 700 + b for b in range(int(cache.k_pages.shape[0]))})
    _assert_bit_exact(
        cache.k_pages.to("cpu")[2], after_refill[2][0], "k page 2 was refilled"
    )

    wrk.submit_load(2, CPULoadStoreSpec([1]), _gpu_spec([2]))
    torch.spyre.synchronize()
    _drain_ok(wrk)

    k_dev = cache.k_pages.to("cpu")
    v_dev = cache.v_pages.to("cpu")
    _assert_bit_exact(k_dev[2], before[2][0], "k page 2 restored")
    _assert_bit_exact(v_dev[2], before[2][1], "v page 2 restored")
    for b in range(int(cache.k_pages.shape[0])):
        if b == 2:
            continue
        _assert_bit_exact(k_dev[b], after_refill[b][0], f"k page {b} untouched by load")
        _assert_bit_exact(v_dev[b], after_refill[b][1], f"v page {b} untouched by load")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_multi_layer_pages_do_not_cross_pools(impl_cls, worker):
    """Each cache must get its own pool, keyed by cache index.

    Three caches store to the *same* host block. If the `_c{c}` suffix were
    dropped -- one shared pool -- they would all write slots 0/1 of one segment
    and the last writer would win, so reloading would hand every cache the third
    cache's pages. Zeroing all three and loading them back catches that: each
    must recover its own content.
    """
    caches = [_cache(impl_cls, num_blocks=2) for _ in range(3)]
    wrk = worker(caches, impl_cls)

    expected = [_fill_all(c, {0: 31 + 7 * i}) for i, c in enumerate(caches)]

    wrk.submit_store(1, _gpu_spec([0]), CPULoadStoreSpec([0]))
    torch.spyre.synchronize()
    _drain_ok(wrk)

    # Destroy the device copies so a no-op load cannot pass.
    for c in caches:
        _zero_pages(c, [0])
    for c in caches:
        assert int(_bits(c.k_pages.to("cpu")[0]).ne(0).sum().item()) == 0

    wrk.submit_load(2, CPULoadStoreSpec([0]), _gpu_spec([0]))
    torch.spyre.synchronize()
    _drain_ok(wrk)

    for i, c in enumerate(caches):
        _assert_bit_exact(c.k_pages.to("cpu")[0], expected[i][0][0], f"cache {i} k page 0")
        _assert_bit_exact(c.v_pages.to("cpu")[0], expected[i][0][1], f"cache {i} v page 0")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS, ids=IMPL_IDS)
def test_worker_round_trip_via_load_store_specs(impl_cls, worker):
    """Several blocks through the real spec objects, with block ids permuted.

    Device blocks [5, 2, 6] pair with host blocks [2, 0, 3] positionally, not by
    sorted order, so a worker that sorted or mismatched the pairing would return
    the wrong page. The reload targets *different* device blocks to prove the
    host slot, and not a leftover device page, is the source.
    """
    cache = _cache(impl_cls)
    wrk = worker([cache], impl_cls)

    device_blocks = [5, 2, 6]
    host_blocks = [2, 0, 3]
    expected = _fill_all(cache, {b: 41 + b for b in device_blocks})

    wrk.submit_store(7, _gpu_spec(device_blocks), CPULoadStoreSpec(host_blocks))
    torch.spyre.synchronize()
    _drain_ok(wrk, expect_bytes=3 * wrk._bytes_per_block)

    # Relocate: the same host slots land on three previously-unrelated blocks.
    targets = [0, 1, 4]
    wrk.submit_load(8, CPULoadStoreSpec(host_blocks), _gpu_spec(targets))
    torch.spyre.synchronize()
    _drain_ok(wrk, expect_bytes=3 * wrk._bytes_per_block)

    k_dev = cache.k_pages.to("cpu")
    v_dev = cache.v_pages.to("cpu")
    for src, dst in zip(device_blocks, targets, strict=True):
        _assert_bit_exact(k_dev[dst], expected[src][0], f"k block {src} -> {dst}")
        _assert_bit_exact(v_dev[dst], expected[src][1], f"v block {src} -> {dst}")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS[:1], ids=IMPL_IDS[:1])
def test_out_of_range_host_block_fails_before_any_dma(impl_cls, worker):
    """An out-of-range host block is rejected, and nothing on the device moves.

    The range check runs over every host block before the first copy, so a bad
    id in the middle of a batch cannot leave a half-applied transfer behind.
    """
    cache = _cache(impl_cls, num_blocks=2)
    wrk = worker([cache], impl_cls, num_host_blocks=2)
    expected = _fill_all(cache, {0: 51, 1: 52})

    assert wrk.submit_store(1, _gpu_spec([0, 1]), CPULoadStoreSpec([0, 9])) is True
    results = wrk.get_finished()
    assert len(results) == 1
    assert results[0].success is False

    k_dev = cache.k_pages.to("cpu")
    for b in (0, 1):
        _assert_bit_exact(k_dev[b], expected[b][0], f"k page {b} after rejected store")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS[:1], ids=IMPL_IDS[:1])
def test_out_of_range_device_block_is_rejected_not_silently_moved(impl_cls, worker):
    """A device block id past the end of the cache must fail, not move a page.

    Device block ids come from the scheduler's block table, so an off-by-one has
    to be loud and has to leave the cache alone. The job is marked failed rather
    than raising out of `submit_store`, which is how upstream surfaces it
    (`offloading/worker.py` asserts `result.success`).

    This does *not* discriminate between `copy_kv_page_raw` and the draft's
    `copy_tensor_raw`: at this call shape `k_pages[5]` on a 2-block cache raises
    IndexError in PyTorch before either reaches the DMA layer. `copy_kv_page_raw`
    does check the block id itself (`block_id 99 out of range [0, 8)`) where
    `copy_tensor_raw` accepts a masked index silently, but that difference only
    shows through an index a view can still resolve, which this worker never
    constructs. Kept as a contract test on the failure path, not as evidence
    about the copy primitive.
    """
    cache = _cache(impl_cls, num_blocks=2)
    wrk = worker([cache], impl_cls, num_host_blocks=2)
    expected = _fill_all(cache, {0: 61, 1: 62})

    # 5 is past the end of a 2-block cache.
    assert wrk.submit_store(1, _gpu_spec([5]), CPULoadStoreSpec([0])) is True
    results = wrk.get_finished()
    assert len(results) == 1
    assert results[0].success is False, (
        "an out-of-range device block was accepted; copy_kv_page_raw should have "
        "rejected it before issuing any DMA"
    )

    k_dev = cache.k_pages.to("cpu")
    v_dev = cache.v_pages.to("cpu")
    for b in (0, 1):
        _assert_bit_exact(k_dev[b], expected[b][0], f"k page {b} after rejected store")
        _assert_bit_exact(v_dev[b], expected[b][1], f"v page {b} after rejected store")


@requires_hardware
@pytest.mark.parametrize("impl_cls", IMPLS[:1], ids=IMPL_IDS[:1])
def test_multiple_kv_groups_are_rejected(impl_cls, worker):
    """A multi-group GPU spec must be refused rather than paired flat.

    `GPULoadStoreSpec.block_ids` is a concatenation ordered by KV group, and each
    group's slice applies only to that group's layers. This worker pairs every
    block against every offloader, which is correct for one group only, so a
    hybrid model has to be rejected instead of quietly offloading the wrong
    pages.
    """
    cache = _cache(impl_cls, num_blocks=2)
    wrk = worker([cache], impl_cls, num_host_blocks=2)
    expected = _fill_all(cache, {0: 71, 1: 72})

    two_groups = GPULoadStoreSpec([0, 1], group_sizes=[1, 1], block_indices=[0, 0])
    assert wrk.submit_store(1, two_groups, CPULoadStoreSpec([0, 1])) is True
    results = wrk.get_finished()
    assert len(results) == 1
    assert results[0].success is False

    k_dev = cache.k_pages.to("cpu")
    for b in (0, 1):
        _assert_bit_exact(k_dev[b], expected[b][0], f"k page {b} after rejected store")
