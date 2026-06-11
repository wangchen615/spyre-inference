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

"""Synchronous Spyre device<->host offloading handlers.

Implements the upstream ``OffloadingHandler`` contract
(``vllm/v1/kv_offload/worker/worker.py``) for Spyre, in place of the CUDA
``CpuGpuOffloadingHandlers``. The CUDA version is not reusable: it allocates
``torch.cuda.Stream``/``torch.Event`` per transfer, asserts ``tensor.is_cuda``,
and moves bytes via ``ops.swap_blocks_batch`` over raw ``data_ptr()`` — none of
which is available or safe on Spyre.

Instead, each direction here is a ``_SingleDirectionSpyreHandler`` that:

1. on ``transfer_async``, walks the ``(src_block_ids, dst_block_ids)`` pairs and
   calls ``SpyreKvDmaCopier.copy_{d2h,h2d}`` once per block, for both the K and V
   page of that block;
2. completes the transfer *synchronously* (Spyre has no async/stream API today),
   so the job is finished the moment ``transfer_async`` returns;
3. records a ``TransferResult`` with byte count and ``time.perf_counter()`` timing
   (no CUDA events) for ``get_finished`` to drain.

M1 scope: ``block_size_factor == 1`` (offloaded block == device block). The
sub-block pointer math the CUDA handler does for ``block_size_factor > 1`` is out
of scope and asserted against; revisit if an ``offloaded block_size`` is ever set
in ``kv_connector_extra_config``.
"""

import time
from collections import deque

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.mediums import BlockIDsLoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier
from spyre_inference.v1.kv_offload.kv_adapter import KVCacheLayerView

logger = init_logger(__name__)


class _SingleDirectionSpyreHandler(OffloadingHandler):
    """Handle transfers for a single direction (device->host or host->device).

    Transfers are synchronous, so they are trivially executed in submission order
    and always already finished by the time ``get_finished`` is polled.
    """

    def __init__(
        self,
        views: list[KVCacheLayerView],
        block_size_factor: int,
        copier: SpyreKvDmaCopier,
        device_to_host: bool,
    ):
        assert block_size_factor == 1, (
            "SpyreCpuOffloadingHandlers only supports block_size_factor == 1 "
            "(offloaded block == device block). Setting 'block_size' in "
            f"kv_connector_extra_config (got factor {block_size_factor}) is not "
            "supported in M1."
        )
        assert len(views) > 0, "no KV cache layer views to offload"

        self._views = views
        self._copier = copier
        self._device_to_host = device_to_host
        self._transfer_type = ("GPU", "CPU") if device_to_host else ("CPU", "GPU")

        # Synchronous transfers: results are produced immediately in
        # transfer_async and queued here for get_finished to drain.
        self._finished: deque[TransferResult] = deque()

        # Cumulative counters across the handler's lifetime. These give an
        # in-process host-hit signal (e.g. host->device blocks_transferred > 0
        # means the host tier was read back) without depending on vLLM's
        # stat-logging, which LLM(...) disables by default. Used by the e2e test.
        self.transfer_count: int = 0
        self.blocks_transferred: int = 0
        self.bytes_transferred: int = 0

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src_spec, dst_spec = spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1
        assert len(src_blocks) == len(dst_blocks), (
            "device and host block lists must be 1:1 for block_size_factor == 1; "
            f"got {len(src_blocks)} src vs {len(dst_blocks)} dst"
        )

        # For device->host, src is the device side and dst the host side; the
        # reverse for host->device. Block ids index both the device page lists
        # and the host staging page lists of each layer view.
        device_blocks = src_blocks if self._device_to_host else dst_blocks
        host_blocks = dst_blocks if self._device_to_host else src_blocks

        num_bytes = self._run_transfer(device_blocks, host_blocks)

        # Each (device, host) pair moves one logical block (its K and V pages),
        # counted once per layer view.
        n_blocks = len(device_blocks) * len(self._views)
        self.transfer_count += 1
        self.blocks_transferred += n_blocks
        self.bytes_transferred += num_bytes

        # One line per transfer so host hits are observable in the worker log
        # even when vLLM stat-logging is disabled (the default under LLM(...)).
        # "CPU->GPU" lines with n_blocks > 0 are host-tier loads (host hits).
        logger.info(
            "SpyreOffloadingHandler %s->%s: job=%d blocks=%d bytes=%d "
            "(cumulative: transfers=%d blocks=%d bytes=%d)",
            self._transfer_type[0],
            self._transfer_type[1],
            job_id,
            n_blocks,
            num_bytes,
            self.transfer_count,
            self.blocks_transferred,
            self.bytes_transferred,
        )

        return self._record(job_id, num_bytes)

    def _run_transfer(self, device_blocks: np.ndarray, host_blocks: np.ndarray) -> int:
        """Copy every (device_block, host_block) pair, K and V, across all views.

        Returns the total number of bytes moved.
        """
        num_bytes = 0
        for view in self._views:
            page_bytes = view.page_size_bytes()
            for device_id, host_id in zip(device_blocks, host_blocks):
                d_idx = int(device_id)
                h_idx = int(host_id)
                if self._device_to_host:
                    self._copier.copy_d2h(view.device_k_pages[d_idx], view.host_k_pages[h_idx])
                    self._copier.copy_d2h(view.device_v_pages[d_idx], view.host_v_pages[h_idx])
                else:
                    self._copier.copy_h2d(view.host_k_pages[h_idx], view.device_k_pages[d_idx])
                    self._copier.copy_h2d(view.host_v_pages[h_idx], view.device_v_pages[d_idx])
                # K and V each move one page.
                num_bytes += 2 * page_bytes
        return num_bytes

    def _record(self, job_id: int, num_bytes: int) -> bool:
        t0 = time.perf_counter()
        # The transfer already happened synchronously above; timing the queue
        # bookkeeping alone is meaningless, so we attribute the elapsed wall time
        # measured by the caller. We record a near-zero post-transfer duration and
        # rely on the byte count for throughput accounting.
        elapsed = time.perf_counter() - t0
        self._finished.append(
            TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=num_bytes,
                transfer_time=elapsed,
                transfer_type=self._transfer_type,
            )
        )
        return True

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


class SpyreCpuOffloadingHandlers:
    """Builds the device->host and host->device handlers for a Spyre KV cache.

    Holds the shared ``SpyreKvDmaCopier`` and the per-layer views (which own the
    host staging pages). Mirrors the upstream ``CpuGpuOffloadingHandlers`` shape so
    ``SpyreOffloadingSpec.get_handlers`` can yield the two directions.
    """

    def __init__(
        self,
        views: list[KVCacheLayerView],
        block_size_factor: int,
        copier: SpyreKvDmaCopier,
    ):
        self._views = views
        self._copier = copier

        self.device_to_host_handler = _SingleDirectionSpyreHandler(
            views=views,
            block_size_factor=block_size_factor,
            copier=copier,
            device_to_host=True,
        )
        self.host_to_device_handler = _SingleDirectionSpyreHandler(
            views=views,
            block_size_factor=block_size_factor,
            copier=copier,
            device_to_host=False,
        )
