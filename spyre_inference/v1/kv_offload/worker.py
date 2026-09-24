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

"""OffloadingWorker that moves Spyre KV pages to and from shared host memory.

The job/result skeleton (`submit_store`/`submit_load`/`_run`/`get_finished`/
`wait`, synchronous completion, per-job exception capture) follows the draft in
torch-spyre#3470. The transfer path is deliberately *not* that draft's.

## Why `copy_kv_page_raw` and not `copy_tensor_raw`

The draft (torch-spyre#3470 with the copier in #704) routed bytes through
`copy_tensor_raw(cache.tensor[block_id], pool, slot)`. On the torch-spyre build
installed here that call is **range-aware** and correct: `spyre_mem.cpp:1090-1107`
derives `Range(storage_offset * element_size, numel * element_size)` from the view
and hands it to `copyRaw`, and flex's `copyRawImpl` bounds-checks the host side
(`deviceSizeFitsHostCapacity`) so an oversized transfer throws before any DMA
rather than spilling into the next slot. Verified on hardware: storing each of 8
pages through the block-view call and reading them back returns each page in its
own slot, byte-identical to `copy_kv_page_raw`, on both token-major and
head-major layouts.

So the draft's path is not corrupting. It is, however, unvalidated. Measured on
hardware, with a rank-4 cache and a 4-slot pool:

| input | `copy_tensor_raw` | `copy_kv_page_raw` |
| --- | --- | --- |
| block id past the end | **accepted silently** | `block_id 99 out of range [0, 8)` |
| flattened `(N,-1)` view | **accepted silently** | `expected a rank-4 KV cache [N,X,Y,D], got rank 2` |
| slot past the end | rejected | rejected |

Both silent acceptances matter here. The connector hands vLLM *flattened*
canonical tensors (`connector.py`), so a wiring mistake that reaches the DMA path
with a canonical tensor instead of the physical one is exactly the rank-2 case --
caught by `copy_kv_page_raw`, silently mis-transferred by `copy_tensor_raw`. And
device block ids arrive from the scheduler's block table, so an off-by-one there
should fail loudly rather than move an arbitrary page.

`copy_kv_page_raw` also derives the physical page range from the device layout
itself instead of trusting a view's `storage_offset`, which keeps this worker
correct if a future layout makes logical page order differ from physical.

## Slot indexing

One `SpyreKvPageOffloader` -- hence one `SharedHostPool` -- per physical cache:

    (cache c, host block h) -> offloaders[c].offload(block_id=dev_blk, slot=h)
                                -> pool "{prefix}_c{c}", slot 2h   (K)
                                -> pool "{prefix}_c{c}", slot 2h+1 (V)

This is the identity mapping because upstream's host block ids are
per-canonical-tensor, not global: `CPUOffloadingWorker` allocates one
`(num_cpu_blocks, page_size)` buffer per canonical tensor, so host block `h`
means "slot `h` of every tensor". Per-cache pools also keep one `PageSignature`
per pool, which a single shared pool would collapse -- wrong the moment two
layers differ in `head_size`.

## Single KV cache group only

`GPULoadStoreSpec.block_ids` is a *concatenation* ordered by KV group, with
`group_sizes[g]` blocks belonging to group `g` and applying only to that group's
layers (`cpu/gpu_worker.py:579-600` slices `block_ids` per group before touching
`layer_refs_per_group[g]`). This worker instead applies every device block to
every offloader, which is the same thing only when there is exactly one group.
M1 is scoped to full attention, where there is; `_transfer` rejects anything else
rather than silently offloading a hybrid model's layers against the wrong blocks.
"""

import time

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)

from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches
from spyre_inference.v1.worker.spyre_kv_offload import SpyreKvPageOffloader

logger = init_logger(__name__)


class SpyreOffloadingWorker(OffloadingWorker):
    """Synchronous KV page transfers between Spyre device memory and host slots.

    Transfers complete before `submit_*` returns, so `wait()` is a no-op and
    `get_finished()` drains results recorded during submission.
    """

    def __init__(
        self,
        physical: SpyrePhysicalCaches,
        num_host_blocks: int,
        pool_prefix: str,
    ) -> None:
        if num_host_blocks <= 0:
            raise ValueError(f"num_host_blocks must be positive, got {num_host_blocks}")

        self._finished_jobs: list[TransferResult] = []
        self._physical = physical
        self._num_host_blocks = num_host_blocks

        # One offloader per physical cache. Constructed eagerly so a geometry or
        # shared-memory problem surfaces at registration, not mid-serve.
        self._offloaders: list[SpyreKvPageOffloader] = [
            SpyreKvPageOffloader(
                cache=cache,
                pool_name=f"{pool_prefix}_c{cache_idx}",
                num_slots=num_host_blocks,
                layout_kind=layout_kind,
            )
            for cache_idx, (cache, layout_kind) in enumerate(
                zip(physical.caches, physical.layout_kinds, strict=True)
            )
        ]
        # K + V per logical page.
        self._bytes_per_block = 2 * sum(
            offloader.page_size_bytes for offloader in self._offloaders
        )
        logger.info(
            "Spyre offloading worker: %d cache(s), %d host block(s), "
            "%d bytes per block, pools %s_c0..%s_c%d",
            len(self._offloaders),
            num_host_blocks,
            self._bytes_per_block,
            pool_prefix,
            pool_prefix,
            len(self._offloaders) - 1,
        )

    def _transfer(
        self, host_spec: LoadStoreSpec, gpu_spec: GPULoadStoreSpec, to_device: bool
    ) -> int:
        """Move one page pair per (device block, host block) per cache.

        Returns the number of bytes moved. Iterates *offloaders*, not canonical
        tensors: each offloader already owns its own K/V pairing, so iterating
        canonical tensors would move every page twice.
        """
        # One group means block_ids is a flat run of device blocks that all of
        # this model's layers share, which is what the loop below assumes. With
        # several groups each group's slice applies to only its own layers, and
        # pairing them flat would offload the wrong pages -- silently.
        group_sizes = getattr(gpu_spec, "group_sizes", None)
        if group_sizes is not None and len(group_sizes) != 1:
            raise NotImplementedError(
                f"Spyre KV offloading supports a single KV cache group; got "
                f"{len(group_sizes)} (hybrid/HMA models are out of scope for M1)"
            )

        device_blocks = list(gpu_spec.block_ids)
        host_blocks = list(host_spec.block_ids)
        if len(device_blocks) != len(host_blocks):
            raise ValueError(
                f"block count mismatch: {len(device_blocks)} device blocks vs "
                f"{len(host_blocks)} host blocks"
            )

        for host_block in host_blocks:
            if not 0 <= host_block < self._num_host_blocks:
                raise IndexError(
                    f"host block {host_block} out of range "
                    f"[0, {self._num_host_blocks})"
                )

        for offloader in self._offloaders:
            for device_block, host_block in zip(device_blocks, host_blocks, strict=True):
                if to_device:
                    offloader.reload(host_block, device_block)
                else:
                    offloader.offload(device_block, host_block)

        return len(device_blocks) * self._bytes_per_block

    def _run(
        self,
        job_id: int,
        host_spec: LoadStoreSpec,
        gpu_spec: GPULoadStoreSpec,
        to_device: bool,
    ) -> bool:
        """Run the transfer job and record its result."""
        started = time.perf_counter()
        try:
            transfer_size = self._transfer(host_spec, gpu_spec, to_device=to_device)
        except Exception:
            # Upstream asserts `transfer_result.success` in its get_finished(),
            # so a failure here surfaces as an assertion there rather than a
            # silently dropped page. Log with the traceback so the real cause is
            # in the serve log.
            logger.exception("Spyre KV offload job %d failed", job_id)
            self._finished_jobs.append(TransferResult(job_id=job_id, success=False))
        else:
            self._finished_jobs.append(
                TransferResult(
                    job_id=job_id,
                    success=True,
                    transfer_size=transfer_size,
                    transfer_time=time.perf_counter() - started,
                )
            )

        # Report the job as accepted. Returning False would mean rejected for
        # resource constraints, which this implementation does not have.
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Device -> host."""
        return self._run(job_id, dst_spec, src_spec, to_device=False)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Host -> device."""
        return self._run(job_id, src_spec, dst_spec, to_device=True)

    def get_finished(self) -> list[TransferResult]:
        """Drain results for jobs completed since the last call."""
        finished_jobs = self._finished_jobs
        self._finished_jobs = []
        return finished_jobs

    def wait(self, job_ids: set[int]) -> None:
        """No-op: transfers complete synchronously before `submit_*` returns."""
