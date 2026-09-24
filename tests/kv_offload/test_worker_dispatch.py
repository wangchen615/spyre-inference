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

"""`SpyreOffloadingWorker` dispatch: which copies it issues, and how it reports.

`SpyreKvPageOffloader` is replaced with a recorder, so this runs with no device
and no shared memory. It checks the *shape* of the dispatch -- how many copies,
against which pool, in which direction -- which is exactly what bit-exactness
tests on hardware cannot see: a worker that offloads every page twice still
round-trips correctly.

The central assertion is the call count. Iterating canonical tensors instead of
offloaders doubles every transfer, because each offloader already owns its own
K/V pair. That is invisible in a round trip and costly in a benchmark.
"""

from __future__ import annotations

import pytest
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from spyre_inference.v1.kv_offload import worker as worker_mod
from spyre_inference.v1.kv_offload.connector import TOKEN_MAJOR, SpyrePhysicalCaches

PAGE_BYTES = 4096
NUM_CACHES = 4
NUM_HOST_BLOCKS = 8
PREFIX = "test_pool"


class _RecordingOffloader:
    """Stands in for SpyreKvPageOffloader, recording calls on a shared list."""

    def __init__(self, calls, *, cache, pool_name, num_slots, layout_kind):
        self._calls = calls
        self.pool_name = pool_name
        self.num_slots = num_slots
        self.layout_kind = layout_kind
        self.page_size_bytes = PAGE_BYTES

    def offload(self, block_id, slot, non_blocking=False):
        self._calls.append(("offload", self.pool_name, block_id, slot))

    def reload(self, slot, block_id, non_blocking=False):
        self._calls.append(("reload", self.pool_name, slot, block_id))


@pytest.fixture
def calls(monkeypatch):
    """Patch the offloader class; yield the list every instance records into."""
    recorded: list[tuple] = []

    def factory(*, cache, pool_name, num_slots, layout_kind):
        return _RecordingOffloader(
            recorded,
            cache=cache,
            pool_name=pool_name,
            num_slots=num_slots,
            layout_kind=layout_kind,
        )

    monkeypatch.setattr(worker_mod, "SpyreKvPageOffloader", factory)
    return recorded


def _physical(num_caches: int = NUM_CACHES) -> SpyrePhysicalCaches:
    return SpyrePhysicalCaches(
        caches=tuple((object(), object()) for _ in range(num_caches)),
        layout_kinds=tuple(TOKEN_MAJOR for _ in range(num_caches)),
        tensor_idx_to_cache={i: i // 2 for i in range(2 * num_caches)},
        num_blocks=16,
    )


def _worker(num_caches: int = NUM_CACHES) -> worker_mod.SpyreOffloadingWorker:
    return worker_mod.SpyreOffloadingWorker(
        physical=_physical(num_caches),
        num_host_blocks=NUM_HOST_BLOCKS,
        pool_prefix=PREFIX,
    )


def _gpu_spec(block_ids: list[int]) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(
        block_ids, group_sizes=[len(block_ids)], block_indices=[0]
    )


def test_one_offloader_per_cache_named_by_cache_index(calls):
    worker = _worker()
    assert len(worker._offloaders) == NUM_CACHES
    assert [o.pool_name for o in worker._offloaders] == [
        f"{PREFIX}_c{c}" for c in range(NUM_CACHES)
    ]
    # Each pool holds one slot per host block; the offloader doubles it for K/V.
    assert all(o.num_slots == NUM_HOST_BLOCKS for o in worker._offloaders)


def test_store_issues_one_copy_per_cache_per_block_not_per_tensor(calls):
    """Two blocks across four caches is eight copies, not sixteen.

    Sixteen is what iterating canonical tensors gives, since each cache appears
    twice there. It round-trips correctly and doubles the DMA -- so only a call
    count catches it.
    """
    worker = _worker()
    assert worker.submit_store(1, _gpu_spec([1, 2]), CPULoadStoreSpec([0, 3])) is True

    assert len(calls) == 8
    assert {(name, blk, slot) for _, name, blk, slot in calls} == {
        (f"{PREFIX}_c{c}", blk, slot)
        for c in range(NUM_CACHES)
        for blk, slot in ((1, 0), (2, 3))
    }
    assert {kind for kind, *_ in calls} == {"offload"}


def test_load_inverts_the_direction_and_the_argument_order(calls):
    """`offload(block, slot)` but `reload(slot, block)` -- reversed, by design."""
    worker = _worker(num_caches=1)
    assert worker.submit_load(2, CPULoadStoreSpec([5]), _gpu_spec([7])) is True

    assert calls == [("reload", f"{PREFIX}_c0", 5, 7)]


def test_block_count_mismatch_is_reported_as_a_failed_job(calls):
    """A spec disagreement must not partially transfer.

    `_transfer` validates before issuing anything, so a mismatch leaves zero
    copies behind rather than a prefix of them.
    """
    worker = _worker()
    assert worker.submit_store(3, _gpu_spec([1, 2]), CPULoadStoreSpec([0])) is True

    assert calls == []
    (result,) = worker.get_finished()
    assert result.job_id == 3
    assert result.success is False


def test_out_of_range_host_block_fails_before_issuing_any_copy(calls):
    worker = _worker()
    assert (
        worker.submit_store(4, _gpu_spec([0]), CPULoadStoreSpec([NUM_HOST_BLOCKS]))
        is True
    )

    assert calls == []
    (result,) = worker.get_finished()
    assert result.success is False


def test_offloader_exception_yields_failure_not_a_raised_submit(calls):
    """Upstream asserts `success`, so a failure must be reported, not thrown.

    `submit_*` returning False would mean "rejected for resource constraints",
    which is a different thing from "this transfer went wrong".
    """
    worker = _worker(num_caches=1)

    def boom(*args, **kwargs):
        raise RuntimeError("dma refused")

    worker._offloaders[0].offload = boom
    assert worker.submit_store(5, _gpu_spec([0]), CPULoadStoreSpec([0])) is True

    (result,) = worker.get_finished()
    assert result.job_id == 5
    assert result.success is False


def test_successful_result_carries_size_and_time_for_benchmarking(calls):
    """Upstream reads both fields; populating them is free instrumentation."""
    worker = _worker()
    worker.submit_store(6, _gpu_spec([1, 2]), CPULoadStoreSpec([0, 3]))

    (result,) = worker.get_finished()
    assert result.success is True
    # 2 blocks x 4 caches x (K + V) pages.
    assert result.transfer_size == 2 * NUM_CACHES * 2 * PAGE_BYTES
    assert result.transfer_time > 0


def test_get_finished_drains(calls):
    worker = _worker()
    worker.submit_store(7, _gpu_spec([0]), CPULoadStoreSpec([0]))
    worker.submit_load(8, CPULoadStoreSpec([0]), _gpu_spec([0]))

    assert [r.job_id for r in worker.get_finished()] == [7, 8]
    assert worker.get_finished() == []


def test_wait_is_a_noop_because_transfers_are_synchronous(calls):
    worker = _worker()
    worker.submit_store(9, _gpu_spec([0]), CPULoadStoreSpec([0]))
    # Completion already happened during submit; wait() must not block or raise
    # on a job id it has never seen.
    worker.wait({9, 999})


def test_multiple_kv_groups_are_rejected(calls):
    """Group-partitioned block ids cannot be paired flat.

    Each group's slice applies only to its own layers, so a flat loop would
    offload one group's pages against another group's blocks -- silently.
    """
    worker = _worker()
    spec = GPULoadStoreSpec([1, 2], group_sizes=[1, 1], block_indices=[0, 1])
    worker.submit_store(10, spec, CPULoadStoreSpec([0, 1]))

    assert calls == []
    (result,) = worker.get_finished()
    assert result.success is False


def test_zero_host_blocks_is_rejected_at_construction(calls):
    with pytest.raises(ValueError, match="num_host_blocks"):
        worker_mod.SpyreOffloadingWorker(
            physical=_physical(), num_host_blocks=0, pool_prefix=PREFIX
        )
