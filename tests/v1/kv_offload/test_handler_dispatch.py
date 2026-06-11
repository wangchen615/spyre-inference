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

"""Pure-CPU tests for the Spyre offloading handler dispatch.

Drives ``device_to_host_handler`` / ``host_to_device_handler`` with the
``torch_copy`` backend over CPU tensors (standing in for Spyre device tensors)
and asserts content lands and ``get_finished`` reports success. No Spyre device
required.
"""

from typing import NamedTuple

import pytest
import torch

from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier
from spyre_inference.v1.kv_offload.handlers import SpyreCpuOffloadingHandlers
from spyre_inference.v1.kv_offload.kv_adapter import build_layer_views


class _FakePagedKVCache(NamedTuple):
    """Stand-in for SpyrePagedKVCache (k_pages, v_pages) on CPU."""

    k_pages: list[torch.Tensor]
    v_pages: list[torch.Tensor]


def _make_cache(num_blocks: int, seed: int = 0):
    torch.manual_seed(seed)

    def pages():
        return [
            torch.randn(2, 16, 64, dtype=torch.float16) for _ in range(num_blocks)
        ]

    return _FakePagedKVCache(pages(), pages())


def _make_handlers(num_blocks: int, num_cpu_blocks: int | None = None):
    cache = _make_cache(num_blocks)
    kv_caches = {"layer.0": cache}
    copier = SpyreKvDmaCopier(backend="torch_copy")
    views = build_layer_views(kv_caches, num_cpu_blocks or num_blocks)
    handlers = SpyreCpuOffloadingHandlers(
        views=views, block_size_factor=1, copier=copier
    )
    return cache, views, handlers


def _gpu_spec(block_ids):
    # Single KV group, logical offset 0.
    return GPULoadStoreSpec(
        list(block_ids), group_sizes=[len(block_ids)], block_indices=[0]
    )


@pytest.mark.spyre
def test_device_to_host_then_host_to_device_round_trips():
    num_blocks = 4
    cache, views, handlers = _make_handlers(num_blocks)
    view = views[0]
    orig_k = [t.clone() for t in cache.k_pages]
    orig_v = [t.clone() for t in cache.v_pages]

    block_ids = list(range(num_blocks))
    gpu_spec = _gpu_spec(block_ids)
    cpu_spec = CPULoadStoreSpec(block_ids)

    # device -> host
    assert handlers.device_to_host_handler.transfer_async(1, (gpu_spec, cpu_spec))
    finished = handlers.device_to_host_handler.get_finished()
    assert len(finished) == 1
    assert finished[0].job_id == 1
    assert finished[0].success
    assert finished[0].transfer_type == ("GPU", "CPU")
    # 4 blocks * 2 pages (k+v) * (2*16*64 elems * 2 bytes) = 32768 bytes
    assert finished[0].transfer_size == num_blocks * 2 * (2 * 16 * 64 * 2)
    for i in range(num_blocks):
        assert torch.equal(view.host_k_pages[i], orig_k[i])
        assert torch.equal(view.host_v_pages[i], orig_v[i])

    # get_finished drains: a second poll returns nothing.
    assert handlers.device_to_host_handler.get_finished() == []

    # corrupt the device cache, then host -> device restores it
    for t in cache.k_pages:
        t.zero_()
    for t in cache.v_pages:
        t.zero_()

    assert handlers.host_to_device_handler.transfer_async(2, (cpu_spec, gpu_spec))
    finished = handlers.host_to_device_handler.get_finished()
    assert len(finished) == 1
    assert finished[0].job_id == 2
    assert finished[0].transfer_type == ("CPU", "GPU")
    for i in range(num_blocks):
        assert torch.equal(cache.k_pages[i], orig_k[i])
        assert torch.equal(cache.v_pages[i], orig_v[i])


@pytest.mark.spyre
def test_partial_block_subset_transfers_only_selected_blocks():
    num_blocks = 6
    cache, views, handlers = _make_handlers(num_blocks)
    view = views[0]
    orig_k = [t.clone() for t in cache.k_pages]

    # Offload only device blocks {1, 3} into host slots {0, 1}.
    device_ids = [1, 3]
    host_ids = [0, 1]
    gpu_spec = _gpu_spec(device_ids)
    cpu_spec = CPULoadStoreSpec(host_ids)

    assert handlers.device_to_host_handler.transfer_async(7, (gpu_spec, cpu_spec))
    finished = handlers.device_to_host_handler.get_finished()
    assert finished[0].success
    assert torch.equal(view.host_k_pages[0], orig_k[1])
    assert torch.equal(view.host_k_pages[1], orig_k[3])


@pytest.mark.spyre
def test_wait_and_shutdown_are_safe():
    _, _, handlers = _make_handlers(2)
    # Nothing in flight (synchronous), so wait is a no-op and must not raise.
    handlers.device_to_host_handler.wait({1, 2, 3})
    handlers.device_to_host_handler.shutdown()
    handlers.host_to_device_handler.shutdown()


@pytest.mark.spyre
def test_block_size_factor_must_be_one():
    cache = _make_cache(2)
    copier = SpyreKvDmaCopier(backend="torch_copy")
    views = build_layer_views({"layer.0": cache}, 2)
    with pytest.raises(AssertionError):
        SpyreCpuOffloadingHandlers(views=views, block_size_factor=2, copier=copier)
