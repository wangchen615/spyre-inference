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

"""SpyreOffloadingWorker block routing, accounting, and lifecycle.

Runs on CPU with a stub copier: the point is that the right *pairs* of pages are
handed to the copier in the right direction, which is independent of how the bytes
actually move. ``test_copier_round_trip`` covers the real device path.
"""

import pytest
import torch

from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from spyre_inference.v1.kv_offload.kv_adapter import build_layer_views
from spyre_inference.v1.kv_offload.worker import SpyreOffloadingWorker

NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE = 2, 4, 64
PAGE_BYTES = NUM_KV_HEADS * BLOCK_SIZE * HEAD_SIZE * 2  # float16
NUM_DEVICE_BLOCKS, NUM_HOST_BLOCKS = 4, 8


class _RecordingCopier:
    """Stands in for SpyreKvDmaCopier, doing a host-side copy and logging direction."""

    def __init__(self):
        self.d2h: list[tuple[int, int]] = []
        self.h2d: list[tuple[int, int]] = []

    def copy_d2h(self, src, dst):
        self.d2h.append((src.data_ptr(), dst.data_ptr()))
        dst.copy_(src)

    def copy_h2d(self, src, dst):
        self.h2d.append((src.data_ptr(), dst.data_ptr()))
        dst.copy_(src)


def _page(fill: float) -> torch.Tensor:
    return torch.full((NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE), fill, dtype=torch.float16)


@pytest.fixture
def views():
    """Two layers, each block pre-filled with a distinct recognizable value."""
    from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache

    kv_caches = {}
    for layer in range(2):
        k_pages = [_page(100 * layer + b) for b in range(NUM_DEVICE_BLOCKS)]
        v_pages = [_page(100 * layer + b + 50) for b in range(NUM_DEVICE_BLOCKS)]
        kv_caches[f"layer{layer}"] = SpyrePagedKVCache(k_pages=k_pages, v_pages=v_pages)
    return build_layer_views(kv_caches, num_cpu_blocks=NUM_HOST_BLOCKS)


@pytest.fixture
def worker(views):
    return SpyreOffloadingWorker(views=views, blocks_per_chunk=1, copier=_RecordingCopier())


def _gpu_spec(block_ids: list[int]) -> GPULoadStoreSpec:
    return GPULoadStoreSpec(block_ids=block_ids, group_sizes=[len(block_ids)], block_indices=[0])


@pytest.mark.spyre
def test_store_copies_device_blocks_into_the_requested_host_slots(worker, views):
    assert worker.submit_store(1, _gpu_spec([1, 3]), CPULoadStoreSpec([5, 6]))

    for view in views:
        # device block 1 -> host slot 5, device block 3 -> host slot 6
        assert torch.equal(view.host_k_pages[5], view.device_k_pages[1])
        assert torch.equal(view.host_v_pages[5], view.device_v_pages[1])
        assert torch.equal(view.host_k_pages[6], view.device_k_pages[3])
        assert torch.equal(view.host_v_pages[6], view.device_v_pages[3])


@pytest.mark.spyre
def test_load_restores_device_blocks_from_host(worker, views):
    worker.submit_store(1, _gpu_spec([1]), CPULoadStoreSpec([5]))
    expected = [view.device_k_pages[1].clone() for view in views]

    for view in views:
        view.device_k_pages[1].zero_()
        view.device_v_pages[1].zero_()

    assert worker.submit_load(2, CPULoadStoreSpec([5]), _gpu_spec([1]))

    for view, want in zip(views, expected):
        assert torch.equal(view.device_k_pages[1], want)


@pytest.mark.spyre
def test_store_and_load_use_opposite_copier_directions(worker):
    worker.submit_store(1, _gpu_spec([0]), CPULoadStoreSpec([0]))
    copier = worker._copier
    # one layer x (K + V) x 2 layers = 4 page copies, all device->host
    assert len(copier.d2h) == 4
    assert copier.h2d == []

    worker.submit_load(2, CPULoadStoreSpec([0]), _gpu_spec([0]))
    assert len(copier.h2d) == 4


@pytest.mark.spyre
def test_byte_and_block_counters_track_direction(worker):
    worker.submit_store(1, _gpu_spec([0, 1]), CPULoadStoreSpec([0, 1]))
    # 2 blocks x 2 layers x (K + V) pages
    assert worker.stored_blocks == 2
    assert worker.loaded_blocks == 0
    assert worker.bytes_transferred == 2 * 2 * 2 * PAGE_BYTES

    worker.submit_load(2, CPULoadStoreSpec([0]), _gpu_spec([0]))
    assert worker.stored_blocks == 2
    assert worker.loaded_blocks == 1


@pytest.mark.spyre
def test_transfers_finish_synchronously(worker):
    worker.submit_store(7, _gpu_spec([0]), CPULoadStoreSpec([0]))

    results = worker.get_finished()
    assert [r.job_id for r in results] == [7]
    assert results[0].success
    assert results[0].transfer_size == 2 * 2 * PAGE_BYTES

    # get_finished drains: a second call reports nothing.
    assert worker.get_finished() == []


@pytest.mark.spyre
def test_wait_is_a_noop_because_nothing_is_in_flight(worker):
    worker.submit_store(1, _gpu_spec([0]), CPULoadStoreSpec([0]))
    worker.wait({1})


@pytest.mark.spyre
def test_mismatched_block_counts_are_rejected(worker):
    with pytest.raises(AssertionError, match="1:1"):
        worker.submit_store(1, _gpu_spec([0, 1]), CPULoadStoreSpec([0]))


@pytest.mark.spyre
def test_multiple_kv_groups_are_rejected(worker):
    """Only a single KV cache group is supported; block_indices is otherwise load-bearing."""
    two_groups = GPULoadStoreSpec(block_ids=[0, 1], group_sizes=[1, 1], block_indices=[0, 0])
    with pytest.raises(AssertionError, match="single KV cache group"):
        worker.submit_store(1, two_groups, CPULoadStoreSpec([0, 1]))


@pytest.mark.spyre
def test_multi_block_chunks_are_rejected(views):
    with pytest.raises(AssertionError, match="one offloaded block per device block"):
        SpyreOffloadingWorker(views=views, blocks_per_chunk=2, copier=_RecordingCopier())


@pytest.mark.spyre
def test_worker_requires_at_least_one_view():
    with pytest.raises(AssertionError, match="no KV cache layer views"):
        SpyreOffloadingWorker(views=[], blocks_per_chunk=1, copier=_RecordingCopier())


@pytest.mark.spyre
def test_shutdown_clears_state(worker):
    worker.submit_store(1, _gpu_spec([0]), CPULoadStoreSpec([0]))
    worker.shutdown()
    assert worker.get_finished() == []


@pytest.mark.spyre
def test_host_pool_is_sized_independently_of_device_blocks(views):
    """The host tier exists to hold blocks evicted from device, so it is expected
    to be larger than the device block count."""
    for view in views:
        assert view.num_blocks == NUM_DEVICE_BLOCKS
        assert len(view.host_k_pages) == NUM_HOST_BLOCKS
        assert len(view.host_v_pages) == NUM_HOST_BLOCKS
        assert view.page_size_bytes() == PAGE_BYTES
