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

"""Behavioral offload→load round-trip for a host pool LARGER than device.

Drives the *real* scheduler-side ``CPUOffloadingManager`` together with the
worker-side ``SpyreCpuOffloadingHandlers`` + ``SpyreKvDmaCopier`` against real
Spyre device pages. The point is to prove the host offload pool can hold more
blocks than exist on device (4 device blocks, 8 host blocks) and that both the
device->host (evict) and host->device (load) paths run across several requests
without the ``IndexError`` that the old device-clamped host allocation produced.

This is the regression test for the host-pool-sizing fix in ``kv_adapter`` (the
dropped ``min(num_cpu_blocks, device_blocks)`` clamp): the manager hands out
host ``block_id``s up to 8, and host slot ids >= 4 (beyond device capacity) must
be writable/readable.

**No data-content assertions.** The current copier uses ``copy_tensor``, whose
d2h path is a layout/dtype-*converting* copy and is lossy for stickified fp16
(the -1 ULP signature documented in
``.claude/skills/debug-spyre/logs/copier-d2h-ulp/``). A bit-exact round-trip
needs ``copy_tensor_raw`` (torch-spyre PR #2796) + a raw-layout guard, which is
the next milestone — see
``docs/debug_logs/2026-06-24-copy-tensor-raw-layout-guard.md``. So
this test asserts the *flow* (transfers succeed, counters increment, slots >= 4
used) and leaves content verification as a TODO below.

Hardware-gated: skips on CPU-only hosts via ``_spyre_available()``.
"""

import pytest
import torch

from vllm.v1.kv_offload.abstract import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier
from spyre_inference.v1.kv_offload.handlers import SpyreCpuOffloadingHandlers
from spyre_inference.v1.kv_offload.kv_adapter import build_layer_views

DEVICE_BLOCKS = 4
HOST_BLOCKS = 8
NUM_REQUESTS = 5  # > DEVICE_BLOCKS, so host slots >= DEVICE_BLOCKS get used.

# Page shape mirrors TorchSpyreModelRunner.initialize_kv_cache_tensors:
# [num_kv_heads, block_size, head_size], fp16.
NUM_KV_HEADS = 2
BLOCK_SIZE = 16
HEAD_SIZE = 64


def _spyre_available() -> bool:
    """Probe for a usable Spyre device, replicating the worker bring-up.

    The ``spyre`` device only registers after ``torch_spyre._autoload()`` runs,
    which ``spyre_inference`` defers until a worker sets the rank env vars. Copied
    from ``test_e2e_offload.py``; here we additionally need the device live so we
    can allocate ``device="spyre"`` pages below.
    """
    try:
        import os

        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
        import torch_spyre

        torch_spyre._autoload()
        return torch.spyre.device_count() > 0
    except Exception:
        return False


def _make_device_cache(num_blocks: int):
    """Fabricate a SpyrePagedKVCache-shaped (k_pages, v_pages) on the Spyre device.

    Returns a plain 2-tuple of page lists; ``build_layer_views`` duck-types it via
    ``_is_paged_kv_cache`` (tuple of two equal-length tensor lists), so we do not
    need to import the attention backend here.
    """
    device = torch.device("spyre")

    def pages():
        return [
            torch.zeros(NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE, dtype=torch.float16, device=device)
            for _ in range(num_blocks)
        ]

    return (pages(), pages())


def _gpu_spec(block_ids):
    # Single KV group, logical offset 0.
    return GPULoadStoreSpec(list(block_ids), group_sizes=[len(block_ids)], block_indices=[0])


def _key(i: int):
    # 32-byte fake block hash + group 0. Distinct per request so each is a fresh
    # store (prepare_store only allocates keys not already present).
    return make_offload_key(i.to_bytes(32, "big"), 0)


@pytest.mark.spyre
def test_host_pool_exceeds_device_offload_and_reload():
    if not _spyre_available():
        pytest.skip("Spyre device not available")

    # --- worker side: 4 device blocks, 8 host blocks ---
    cache = _make_device_cache(DEVICE_BLOCKS)
    views = build_layer_views({"layer.0": cache}, HOST_BLOCKS)
    view = views[0]
    # The fix: host pages sized to the pool (8), not the device count (4).
    assert len(view.host_k_pages) == HOST_BLOCKS
    assert len(view.host_v_pages) == HOST_BLOCKS
    assert view.num_blocks == DEVICE_BLOCKS  # device count unchanged

    handlers = SpyreCpuOffloadingHandlers(
        views=views, block_size_factor=1, copier=SpyreKvDmaCopier()
    )

    # --- scheduler side: one manager, host pool of 8 blocks ---
    manager = CPUOffloadingManager(num_blocks=HOST_BLOCKS, cache_policy="lru")
    req = ReqContext()

    # --- 5 store (evict) requests: device -> host ---
    used_host_ids: list[int] = []
    for i in range(NUM_REQUESTS):
        key = _key(i)
        out = manager.prepare_store([key], req)
        assert out is not None, "store preparation should not fail (pool not full)"
        assert out.keys_to_store == [key]
        host_ids = list(out.store_spec.block_ids)
        assert len(host_ids) == 1
        host_id = int(host_ids[0])
        used_host_ids.append(host_id)

        # Offload device block (cycled within the 4 device blocks) into the host
        # slot the manager chose.
        device_id = i % DEVICE_BLOCKS
        gpu_spec = _gpu_spec([device_id])
        cpu_spec = CPULoadStoreSpec([host_id])
        assert handlers.device_to_host_handler.transfer_async(i, (gpu_spec, cpu_spec))

        finished = handlers.device_to_host_handler.get_finished()
        assert len(finished) == 1
        assert finished[0].job_id == i
        assert finished[0].success
        assert finished[0].transfer_type == ("GPU", "CPU")

        manager.complete_store([key])

    # The whole point: with 4 device blocks the manager filled fresh host slots
    # 0..4, so at least one slot >= DEVICE_BLOCKS was written without IndexError.
    assert max(used_host_ids) >= DEVICE_BLOCKS, (
        f"host pool did not exceed device capacity: used slots {used_host_ids}"
    )
    assert handlers.device_to_host_handler.transfer_count == NUM_REQUESTS
    assert handlers.device_to_host_handler.blocks_transferred > 0
    assert handlers.device_to_host_handler.bytes_transferred > 0

    # --- 1 load request: host -> device (round-trip the load path) ---
    # Pick a stored key whose host slot is >= DEVICE_BLOCKS to exercise the
    # beyond-device index on the load side too.
    load_idx = next(i for i, h in enumerate(used_host_ids) if h >= DEVICE_BLOCKS)
    load_key = _key(load_idx)
    load_spec = manager.prepare_load([load_key], req)
    load_host_id = int(list(load_spec.block_ids)[0])
    assert load_host_id >= DEVICE_BLOCKS

    device_id = load_idx % DEVICE_BLOCKS
    gpu_spec = _gpu_spec([device_id])
    cpu_spec = CPULoadStoreSpec([load_host_id])
    job_id = NUM_REQUESTS
    # host -> device: src is the CPU spec, dst is the GPU spec.
    assert handlers.host_to_device_handler.transfer_async(job_id, (cpu_spec, gpu_spec))
    finished = handlers.host_to_device_handler.get_finished()
    assert len(finished) == 1
    assert finished[0].success
    assert finished[0].transfer_type == ("CPU", "GPU")
    manager.complete_load([load_key])

    assert handlers.host_to_device_handler.transfer_count == 1
    assert handlers.host_to_device_handler.blocks_transferred > 0

    # TODO(copy_tensor_raw): once the copier adopts torch-spyre PR #2796's
    # copy_tensor_raw (bit-exact, with the raw-layout guard), assert the reloaded
    # device block equals the original bytes. The current copy_tensor path is
    # lossy for stickified fp16, so content is not checked here. See
    # docs/debug_logs/2026-06-24-copy-tensor-raw-layout-guard.md.
    #
    # TODO(eviction): a separate test should store > HOST_BLOCKS keys to force
    # LRU eviction and assert eviction via manager.take_events(). With 8 host
    # slots and 5 stores the free-list never runs dry, so no eviction happens
    # here by design.
