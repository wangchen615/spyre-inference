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
| flattened `(N,-1)` view | **accepted silently** | `expected rank-4 KV cache, got rank 2` |
| slot past the end | rejected | rejected |

Both silent acceptances matter here. The connector hands vLLM *flattened*
canonical tensors (`connector.py`), so a wiring mistake that reaches the DMA path
with a canonical tensor instead of the physical one is exactly the rank-2 case --
caught by `copy_kv_page_raw`, silently mistransferred by `copy_tensor_raw`. And
device block ids arrive from the scheduler's block table, so an off-by-one there
should fail loudly rather than move an arbitrary page.

`copy_kv_page_raw` also derives the physical page range from the device layout
itself instead of trusting a view's `storage_offset`, which keeps this worker
correct if a future layout makes logical page order differ from physical.

## Slot indexing

The delegator creates one fixed child for every physical K/V component. If a
model has `C` components, logical host block `h` owns the consecutive run
`[h*C, (h+1)*C)`. Cache `c` uses component ids `2*c` for K and `2*c+1` for V.
This preserves upstream's meaning of one host block across every canonical
tensor while ensuring that layers and K/V allocations never share a slot.

## Why `_transfer` fences first

Every transfer is issued from `start_kv_transfers`, called by the connector's
`start_load_kv` before the forward pass. Loads are therefore safe by construction:
they write pages before anything reads them.

Stores are not, because they are deferred one step -- `get_finished` calls
`prepare_store_kv`, which queues them for the *next* step's `start_load_kv`. The
pages being read were written by the previous forward, and `copy_kv_page_raw`
waits at most on its own DMA. So `_transfer` calls `torch.spyre.synchronize()`
once per job before issuing anything.

Within a job, every page copy is enqueued non-blocking and the job waits once,
at the end. A blocking copy synchronizes the stream after every K and every V
page, which made a reload one host round trip per page (about 64 us each,
against about 7 us of transfer for a 256 KiB page). The end-of-job synchronize
keeps the contract `wait()` and `get_finished()` rely on: the job is complete
when `submit_*` returns.

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
import traceback

import torch
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)

from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.shared_attachment import SharedPoolAttachment
from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
from spyre_inference.v1.worker.spyre_kv_offload import (
    SpyreKvPageOffloader,
    page_signature,
)

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
        identity: SharedPoolIdentity,
    ) -> None:
        if num_host_blocks <= 0:
            raise ValueError(f"num_host_blocks must be positive, got {num_host_blocks}")
        if not physical.caches:
            raise ValueError("at least one physical KV cache is required")

        self._finished_jobs: list[TransferResult] = []
        self._physical = physical
        self._num_host_blocks = num_host_blocks

        cache_layouts = tuple(zip(physical.caches, physical.layout_kinds, strict=True))
        signatures = tuple(page_signature(cache, layout) for cache, layout in cache_layouts)
        component_count = 2 * len(cache_layouts)
        expected_slots = num_host_blocks * component_count
        largest_page = max(signature.page_bytes for signature in signatures)

        attachment = SharedPoolAttachment.attach(identity)
        self._attachment = attachment
        try:
            if attachment.registered.slot_count != expected_slots:
                raise ValueError(
                    f"data pool {identity.data_pool_name!r} has slot_count="
                    f"{attachment.registered.slot_count}, expected {expected_slots} "
                    f"({num_host_blocks} blocks x {component_count} components)"
                )
            if attachment.registered.slot_bytes < largest_page:
                raise ValueError(
                    f"data pool {identity.data_pool_name!r} has slot_bytes="
                    f"{attachment.registered.slot_bytes}, smaller than the largest "
                    f"physical KV page ({largest_page})"
                )

            self._offloaders = [
                SpyreKvPageOffloader(
                    cache=cache,
                    pool=attachment.data_pool,
                    num_slots=num_host_blocks,
                    layout_kind=layout_kind,
                    component_count=component_count,
                    component_offset=2 * cache_idx,
                    signature=signatures[cache_idx],
                )
                for cache_idx, (cache, layout_kind) in enumerate(cache_layouts)
            ]
        except BaseException:
            attachment.close()
            raise

        self._component_count = component_count
        self._bytes_per_block = 2 * sum(signature.page_bytes for signature in signatures)
        logger.info(
            "Spyre offloading worker: %d cache(s), %d host block(s), %d components, "
            "%d bytes per block, shared child %s/%s",
            len(self._offloaders),
            num_host_blocks,
            component_count,
            self._bytes_per_block,
            identity.pool_name,
            identity.data_pool_name,
        )

    def _validate_gpu_spec(self, gpu_spec: GPULoadStoreSpec) -> None:
        group_sizes = getattr(gpu_spec, "group_sizes", None)
        if group_sizes is not None and len(group_sizes) != 1:
            raise NotImplementedError(
                f"Spyre KV offloading supports a single KV cache group; got "
                f"{len(group_sizes)} (hybrid/HMA models are out of scope for M1)"
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
        self._validate_gpu_spec(gpu_spec)

        # Fence before reading device pages. Stores are deferred a step:
        # get_finished() -> prepare_store_kv() queues them, and they are issued
        # from the *next* step's start_load_kv, so the bytes being offloaded were
        # written by the previous forward with nothing in between. A blocking
        # copy_kv_page_raw waits on its own DMA, not on prior compute, so without
        # this a store can read pages that still have async work pending. The
        # symptom would be stale bytes offloaded -- wrong tokens on a later cache
        # hit, no error raised, and unit tests unaffected because they never have
        # pending compute. One synchronize per job, not per page.
        torch.spyre.synchronize()

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
                    f"host block {host_block} out of range [0, {self._num_host_blocks})"
                )

        # Enqueue all pages, then wait once for the complete job. The worker's
        # synchronous contract is preserved without one host wait per page.
        for offloader in self._offloaders:
            for device_block, host_block in zip(device_blocks, host_blocks, strict=True):
                if to_device:
                    offloader.reload(host_block, device_block, non_blocking=True)
                else:
                    offloader.offload(device_block, host_block, non_blocking=True)
        torch.spyre.synchronize()

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
            # silently dropped page. Serialize the traceback before logging so
            # handlers cannot retain native pool handles through its frames.
            failure = traceback.format_exc()
            logger.error("Spyre KV offload job %d failed:\n%s", job_id, failure)
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

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec) -> bool:
        """Host -> device."""
        return self._run(job_id, src_spec, dst_spec, to_device=True)

    def get_finished(self) -> list[TransferResult]:
        """Drain results for jobs completed since the last call."""
        finished_jobs = self._finished_jobs
        self._finished_jobs = []
        return finished_jobs

    def wait(self, job_ids: set[int]) -> None:
        """No-op: transfers complete synchronously before `submit_*` returns."""

    def shutdown(self) -> None:
        """Fence outstanding DMA and release this worker's root lock lease."""
        try:
            torch.spyre.synchronize()
        finally:
            self._finished_jobs.clear()
            for offloader in self._offloaders:
                offloader.close()
            self._offloaders.clear()
            self._attachment.close()
