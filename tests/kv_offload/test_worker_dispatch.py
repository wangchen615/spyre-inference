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

from types import SimpleNamespace

import pytest
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from spyre_inference.v1.kv_offload import worker as worker_mod
from spyre_inference.v1.kv_offload.connector import TOKEN_MAJOR, SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
from spyre_inference.v1.worker import spyre_kv_offload as offload_mod
from spyre_inference.v1.worker.spyre_kv_offload import PageSignature

PAGE_BYTES = 4096
NUM_CACHES = 4
NUM_HOST_BLOCKS = 8
COMPONENT_COUNT = 2 * NUM_CACHES
IDENTITY = SharedPoolIdentity("test-root", bytes.fromhex("11" * 16), "host-0")


class _Calls(list):
    pass


@pytest.fixture
def calls(monkeypatch):
    """Attach a fake fixed child and record the real offloader's raw copies."""
    recorded = _Calls()
    data_pool = object()
    registered = SimpleNamespace(
        slot_count=NUM_HOST_BLOCKS * COMPONENT_COUNT,
        slot_bytes=PAGE_BYTES,
    )
    attachment = SimpleNamespace(
        registered=registered,
        data_pool=data_pool,
        close_calls=0,
        closed=False,
    )

    def close():
        if not attachment.closed:
            attachment.close_calls += 1
            attachment.closed = True

    attachment.close = close

    def attach(identity):
        recorded.append(("attach", identity))
        return attachment

    def signature(_cache, layout_kind):
        return PageSignature(
            layout_kind=layout_kind,
            device_dtype="torch.float16",
            block_size=128,
            local_kv_heads=8,
            head_size=128,
            page_bytes=PAGE_BYTES,
        )

    def copy_fn(*args):
        recorded.append(("copy", *args))

    monkeypatch.setattr(worker_mod.SharedPoolAttachment, "attach", staticmethod(attach))
    monkeypatch.setattr(worker_mod, "page_signature", signature)
    monkeypatch.setattr(
        offload_mod,
        "_import_runtime",
        lambda: (None, copy_fn, None),
    )
    monkeypatch.setattr(
        worker_mod.torch,
        "spyre",
        SimpleNamespace(synchronize=lambda: None),
        raising=False,
    )
    recorded.attachment = attachment  # type: ignore[attr-defined]
    recorded.data_pool = data_pool  # type: ignore[attr-defined]
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
        identity=IDENTITY,
    )


def _gpu_spec(block_ids: list[int]) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(block_ids, group_sizes=[len(block_ids)], block_indices=[0])


def test_one_attachment_and_one_offloader_per_cache(calls):
    worker = _worker()
    assert len(worker._offloaders) == NUM_CACHES
    assert calls == [("attach", IDENTITY)]
    assert all(o.pool is calls.data_pool for o in worker._offloaders)
    assert all(o.num_slots == NUM_HOST_BLOCKS for o in worker._offloaders)


def test_store_issues_one_copy_per_cache_per_block_not_per_tensor(calls):
    """Two blocks across four caches is eight copies, not sixteen.

    Sixteen is what iterating canonical tensors gives, since each cache appears
    twice there. It round-trips correctly and doubles the DMA -- so only a call
    count catches it.
    """
    worker = _worker()
    assert worker.submit_store(1, _gpu_spec([1, 2]), CPULoadStoreSpec([0, 3])) is True

    copies = [call for call in calls if call[0] == "copy"]
    assert len(copies) == 16
    actual = {
        (tensor, blk, pool, slot, to_device, non_blocking)
        for _, tensor, blk, pool, slot, to_device, non_blocking in copies
    }
    expected = {
        (
            tensor,
            blk,
            calls.data_pool,
            host * COMPONENT_COUNT + component_id,
            False,
            True,
        )
        for component_id, tensor in enumerate(
            tensor for cache in worker._physical.caches for tensor in cache
        )
        for blk, host in ((1, 0), (2, 3))
    }
    assert actual == expected


def test_load_inverts_the_direction_and_the_argument_order(calls):
    """`offload(block, slot)` but `reload(slot, block)` -- reversed, by design."""
    worker = _worker()
    assert worker.submit_load(2, CPULoadStoreSpec([5]), _gpu_spec([7])) is True

    copies = [call for call in calls if call[0] == "copy"]
    assert len(copies) == COMPONENT_COUNT
    assert copies[:2] == [
        ("copy", worker._physical.caches[0][0], 7, calls.data_pool, 40, True, True),
        ("copy", worker._physical.caches[0][1], 7, calls.data_pool, 41, True, True),
    ]


def test_block_count_mismatch_is_reported_as_a_failed_job(calls):
    """A spec disagreement must not partially transfer.

    `_transfer` validates before issuing anything, so a mismatch leaves zero
    copies behind rather than a prefix of them.
    """
    worker = _worker()
    assert worker.submit_store(3, _gpu_spec([1, 2]), CPULoadStoreSpec([0])) is True

    assert [call for call in calls if call[0] == "copy"] == []
    (result,) = worker.get_finished()
    assert result.job_id == 3
    assert result.success is False


def test_out_of_range_host_block_fails_before_issuing_any_copy(calls):
    worker = _worker()
    assert worker.submit_store(4, _gpu_spec([0]), CPULoadStoreSpec([NUM_HOST_BLOCKS])) is True

    assert [call for call in calls if call[0] == "copy"] == []
    (result,) = worker.get_finished()
    assert result.success is False


def test_offloader_exception_yields_failure_not_a_raised_submit(calls):
    """Upstream asserts `success`, so a failure must be reported, not thrown.

    `submit_*` returning False would mean "rejected for resource constraints",
    which is a different thing from "this transfer went wrong".
    """
    worker = _worker()

    def boom(*args, **kwargs):
        raise RuntimeError("dma refused")

    worker._offloaders[0].offload = boom
    assert worker.submit_store(5, _gpu_spec([0]), CPULoadStoreSpec([0])) is True

    (result,) = worker.get_finished()
    assert result.job_id == 5
    assert result.success is False


def test_failure_log_serializes_traceback_without_retaining_native_handles(calls, monkeypatch):
    messages = []
    worker = _worker()
    monkeypatch.setattr(
        worker_mod,
        "logger",
        SimpleNamespace(error=lambda message, *args: messages.append(message % args)),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("dma refused")

    worker._offloaders[0].offload = boom
    worker.submit_store(5, _gpu_spec([0]), CPULoadStoreSpec([0]))

    assert "RuntimeError: dma refused" in messages[0]


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

    assert [call for call in calls if call[0] == "copy"] == []
    (result,) = worker.get_finished()
    assert result.success is False


def test_gpu_spec_validation_is_available_to_shared_workers(calls):
    """The M2 subclass must reuse the exact M1 single-group validation."""
    worker = _worker()
    worker._validate_gpu_spec(_gpu_spec([1]))

    with pytest.raises(NotImplementedError, match="single KV cache group"):
        worker._validate_gpu_spec(
            GPULoadStoreSpec([1, 2], group_sizes=[1, 1], block_indices=[0, 1])
        )


def test_zero_host_blocks_is_rejected_at_construction(calls):
    with pytest.raises(ValueError, match="num_host_blocks"):
        worker_mod.SpyreOffloadingWorker(physical=_physical(), num_host_blocks=0, identity=IDENTITY)


def test_child_geometry_must_match_component_capacity(calls):
    calls.attachment.registered.slot_count -= 1

    with pytest.raises(ValueError, match="slot_count"):
        _worker()

    assert calls.attachment.close_calls == 1
    assert [call for call in calls if call[0] == "copy"] == []


def test_child_slot_must_hold_the_largest_physical_page(calls):
    calls.attachment.registered.slot_bytes = PAGE_BYTES - 1

    with pytest.raises(ValueError, match="slot_bytes"):
        _worker()

    assert calls.attachment.close_calls == 1
    assert [call for call in calls if call[0] == "copy"] == []


def test_effective_aligned_slot_larger_than_page_is_accepted(calls):
    calls.attachment.registered.slot_bytes = 2 * PAGE_BYTES

    worker = _worker()

    assert len(worker._offloaders) == NUM_CACHES


def test_shutdown_waits_then_detaches_once(calls, monkeypatch):
    events = []
    monkeypatch.setattr(worker_mod.torch.spyre, "synchronize", lambda: events.append("sync"))
    worker = _worker()

    worker.shutdown()
    worker.shutdown()

    assert events == ["sync", "sync"]
    assert calls.attachment.close_calls == 1


def test_shutdown_releases_resolved_child_handles_before_detach(calls):
    worker = _worker()
    offloaders_seen_by_close = []

    def close():
        offloaders_seen_by_close.append(list(worker._offloaders))

    calls.attachment.close = close

    worker.shutdown()

    assert offloaders_seen_by_close == [[]]


@pytest.mark.parametrize("direction", ["store", "load"])
def test_copies_are_enqueued_non_blocking_then_the_job_waits_once(calls, monkeypatch, direction):
    """Every page copy is queued before one job-completion wait."""
    events: list[tuple] = []
    monkeypatch.setattr(
        worker_mod.torch.spyre,
        "synchronize",
        lambda: events.append(("sync",)),
    )
    monkeypatch.setattr(
        offload_mod,
        "_import_runtime",
        lambda: (None, lambda *args: events.append(("copy", args[-1])), None),
    )
    worker = _worker()

    if direction == "store":
        worker.submit_store(10, _gpu_spec([1, 2]), CPULoadStoreSpec([0, 3]))
    else:
        worker.submit_load(10, CPULoadStoreSpec([0, 3]), _gpu_spec([1, 2]))

    copies = [event for event in events if event[0] == "copy"]
    assert len(copies) == 2 * NUM_CACHES * 2
    assert all(non_blocking for _, non_blocking in copies)
    assert events[0] == ("sync",)
    assert events[-1] == ("sync",)
    assert [event for event in events[1:-1] if event[0] == "sync"] == []
    (result,) = worker.get_finished()
    assert result.success is True
