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

"""Synchronous Spyre device<->host offloading worker.

Implements the upstream ``OffloadingWorker`` contract (``vllm/v1/kv_offload/base.py``)
for Spyre, in place of the CUDA ``CPUOffloadingWorker``. The CUDA version is not
reusable: it moves bytes over raw ``data_ptr()`` via Triton swap kernels and assumes a
canonicalized ``(num_blocks, page_size_bytes)`` int8 view of each layer, none of which
is available on Spyre's paged-list cache.

Instead, both directions walk the ``(device_block, host_block)`` pairs and call
``SpyreKvDmaCopier.copy_{d2h,h2d}`` once per block, for the K and V page of every layer.
Transfers complete *synchronously* (Spyre has no async/stream API today), so a job is
finished the moment ``submit_*`` returns and ``wait`` is a no-op.

Scope: one offloaded block per device block. The sub-block pointer math the CUDA worker
does for larger offloaded blocks is out of scope and asserted against.
"""

from collections import deque

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier
from spyre_inference.v1.kv_offload.kv_adapter import KVCacheLayerView

logger = init_logger(__name__)


class SpyreOffloadingWorker(OffloadingWorker):
    """Moves KV blocks between Spyre device memory and host RAM, synchronously.

    Direction is explicit in the upstream API (``submit_store`` is device->host,
    ``submit_load`` is host->device), so both funnel into one copy loop.
    """

    def __init__(
        self,
        views: list[KVCacheLayerView],
        blocks_per_chunk: int,
        copier: SpyreKvDmaCopier,
    ):
        assert blocks_per_chunk == 1, (
            "SpyreOffloadingWorker only supports one offloaded block per device "
            "block. Setting 'blocks_per_chunk' in kv_connector_extra_config (got "
            f"{blocks_per_chunk}) is not supported."
        )
        assert len(views) > 0, "no KV cache layer views to offload"

        self._views = views
        self._copier = copier

        # Synchronous transfers: results are produced immediately in submit_* and
        # queued here for get_finished to drain.
        self._finished: deque[TransferResult] = deque()

        # Cumulative counters across the worker's lifetime. These give an in-process
        # host-hit signal (e.g. loaded_blocks > 0 means the host tier was read back)
        # without depending on vLLM's stat-logging, which LLM(...) disables by
        # default. Counts logical blocks, not the per-layer page copies -- see
        # _run_transfer for the distinction.
        self.stored_blocks: int = 0
        self.loaded_blocks: int = 0
        self.bytes_transferred: int = 0

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._transfer(job_id, src_spec, dst_spec, device_to_host=True)

    def submit_load(self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec) -> bool:
        return self._transfer(job_id, dst_spec, src_spec, device_to_host=False)

    def _transfer(
        self,
        job_id: int,
        device_spec: GPULoadStoreSpec,
        host_spec: LoadStoreSpec,
        device_to_host: bool,
    ) -> bool:
        assert isinstance(device_spec, BlockIDsLoadStoreSpec)
        assert isinstance(host_spec, BlockIDsLoadStoreSpec)

        device_blocks = device_spec.block_ids
        host_blocks = host_spec.block_ids
        assert device_blocks.ndim == 1
        assert host_blocks.ndim == 1
        assert len(device_blocks) == len(host_blocks), (
            "device and host block lists must be 1:1 for one offloaded block per "
            f"device block; got {len(device_blocks)} device vs {len(host_blocks)} host"
        )

        # block_indices exists to let a worker skip part of a leading offloaded block
        # when offloaded blocks span several device blocks. With a 1:1 mapping every
        # device block is aligned, so there is nothing to skip -- but assert the
        # single-group shape rather than silently ignoring the field.
        assert len(device_spec.group_sizes) == 1, (
            "SpyreOffloadingWorker supports a single KV cache group; got "
            f"{len(device_spec.group_sizes)} groups"
        )

        num_bytes = self._run_transfer(device_blocks, host_blocks, device_to_host)

        num_blocks = len(device_blocks)
        if device_to_host:
            self.stored_blocks += num_blocks
        else:
            self.loaded_blocks += num_blocks
        self.bytes_transferred += num_bytes

        # One line per transfer so host hits are observable in the worker log even
        # when vLLM stat-logging is disabled (the default under LLM(...)). A logical
        # block is one KV slot; offloading it copies that slot out of every layer (K
        # and V), so page_copies is num_blocks * num_layers * 2.
        logger.info(
            "SpyreOffloadingWorker %s: job=%d num_blocks=%d page_copies=%d "
            "(=%d layers) bytes=%d (cumulative: stored=%d loaded=%d bytes=%d)",
            "device->host" if device_to_host else "host->device",
            job_id,
            num_blocks,
            num_blocks * len(self._views) * 2,
            len(self._views),
            num_bytes,
            self.stored_blocks,
            self.loaded_blocks,
            self.bytes_transferred,
        )

        self._finished.append(TransferResult(job_id=job_id, success=True, transfer_size=num_bytes))
        return True

    def _run_transfer(
        self,
        device_blocks: np.ndarray,
        host_blocks: np.ndarray,
        device_to_host: bool,
    ) -> int:
        """Copy every (device_block, host_block) pair, K and V, across all views.

        Returns the total number of bytes moved.
        """
        num_bytes = 0
        for view in self._views:
            page_bytes = view.page_size_bytes()
            for device_id, host_id in zip(device_blocks, host_blocks):
                d_idx = int(device_id)
                h_idx = int(host_id)
                if device_to_host:
                    self._copier.copy_d2h(view.device_k_pages[d_idx], view.host_k_pages[h_idx])
                    self._copier.copy_d2h(view.device_v_pages[d_idx], view.host_v_pages[h_idx])
                else:
                    self._copier.copy_h2d(view.host_k_pages[h_idx], view.device_k_pages[d_idx])
                    self._copier.copy_h2d(view.host_v_pages[h_idx], view.device_v_pages[d_idx])
                # K and V each move one page.
                num_bytes += 2 * page_bytes
        return num_bytes

    def get_finished(self) -> list[TransferResult]:
        results = list(self._finished)
        self._finished.clear()
        return results

    def wait(self, job_ids: set[int]) -> None:
        # All transfers are synchronous, so nothing is ever in flight.
        return

    def shutdown(self) -> None:
        self._finished.clear()
        self._views = []
