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

"""OffloadingSpec for Spyre KV offloading to shared host memory.

Subclasses `OffloadingSpec` directly rather than `CPUOffloadingSpec`. The CPU
spec's worker construction is built around pinned host tensors and
`torch.accelerator` device indexing, neither of which applies here: Spyre's host
side is a POSIX shared-memory pool pinned through the IOMMU, and its DMA is
issued by `copy_kv_page_raw` against the original rank-4 allocation.

The scheduler side needs no Spyre code at all, so `get_manager` reuses upstream's
`CPUOffloadingManager` verbatim -- it is a plain block-id bookkeeper with no
`current_platform` or `torch.cuda` dependency.

The worker side cannot be built from the canonical KV caches alone (those are
flattened views, which `copy_kv_page_raw` rejects), so the connector hands the
original caches over via `bind_physical_caches` before `get_worker` is called.
"""

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

if TYPE_CHECKING:
    from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches

logger = init_logger(__name__)


class SpyreOffloadingSpec(OffloadingSpec):
    """Spyre host-memory offloading: upstream manager, Spyre worker."""

    def __init__(self, config: OffloadingConfig) -> None:
        super().__init__(config)

        # One host block must map to exactly one device block. With chunking,
        # upstream's host page is gpu_page_size * blocks_per_chunk and a host
        # block spans several device blocks, but copy_kv_page_raw moves exactly
        # one page into one slot -- there is no representation for that here.
        # Assert on the *derived* value: upstream computes it from either the
        # `blocks_per_chunk` or the `block_size` key.
        if self.blocks_per_chunk != 1:
            raise ValueError(
                f"Spyre KV offloading requires blocks_per_chunk == 1, got "
                f"{self.blocks_per_chunk}. Remove 'blocks_per_chunk' and "
                f"'block_size' from kv_connector_extra_config."
            )

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        # Mirrors CPUOffloadingSpec's sizing so the scheduler's host block ids
        # stay in step with upstream's accounting. Note this is *logical* KV
        # bytes; the pools are sized from the physical page size, which may be
        # larger because of stick padding (see get_worker).
        world_size = config.parallel.world_size
        kv_bytes_per_block = config.worker_kv_bytes_per_block * world_size
        if kv_bytes_per_block <= 0:
            raise ValueError(
                f"cannot size host pool: worker_kv_bytes_per_block="
                f"{config.worker_kv_bytes_per_block}, world_size={world_size}"
            )
        self.num_blocks = int(cpu_bytes_to_use) // kv_bytes_per_block
        if self.num_blocks <= 0:
            raise ValueError(
                f"cpu_bytes_to_use={int(cpu_bytes_to_use)} is too small for even "
                f"one block of {kv_bytes_per_block} bytes. Every lookup would "
                f"miss and nothing would be offloaded."
            )

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

        # Pool names are per engine and per rank: two ranks on one host must not
        # collide on a POSIX shared-memory name.
        self._pool_prefix = (
            f"spyre_kv_{config.engine_id}_r{config.parallel.rank}"
        )

        # scheduler-side
        self._manager: OffloadingManager | None = None
        # worker-side, bound by the connector before get_worker()
        self._physical: "SpyrePhysicalCaches | None" = None

        logger.info(
            "SpyreOffloadingSpec: %d host block(s) of %d logical KV bytes, "
            "pool prefix %s",
            self.num_blocks,
            kv_bytes_per_block,
            self._pool_prefix,
        )

    def bind_physical_caches(self, physical: "SpyrePhysicalCaches") -> None:
        """Hand over the original rank-4 caches for the DMA path.

        Called by `SpyreOffloadingConnectorWorker.register_kv_caches` before
        `_init_worker`, because the canonical caches upstream passes to
        `get_worker` are flattened views that `copy_kv_page_raw` rejects.
        """
        self._physical = physical

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for offloading. Values < 2 disable filtering.
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        """Build the Spyre worker from the *physical* caches.

        `kv_caches` is accepted to satisfy the upstream signature and is used
        only to cross-check that canonicalization and the physical binding agree.
        """
        # Imported here: this module is resolved by OffloadingSpecFactory on the
        # scheduler side too, where the connector module need not be loaded.
        from spyre_inference.v1.kv_offload.worker import SpyreOffloadingWorker

        if self._physical is None:
            raise RuntimeError(
                "bind_physical_caches() must be called before get_worker(); the "
                "canonical KV caches are flattened views and cannot be used for "
                "DMA. Is SpyreOffloadingConnector in use?"
            )

        expected_tensors = len(self._physical.tensor_idx_to_cache)
        if len(kv_caches.tensors) != expected_tensors:
            raise ValueError(
                f"canonical tensor count {len(kv_caches.tensors)} disagrees with "
                f"the {expected_tensors} recorded during canonicalization"
            )

        worker = SpyreOffloadingWorker(
            physical=self._physical,
            num_host_blocks=self.num_blocks,
            pool_prefix=self._pool_prefix,
        )

        # The manager hands out host block ids from logical-byte math, while the
        # pools are sized from the physical page size (which includes stick
        # padding). The indices still agree, so this is a host-RAM overrun and
        # not corruption -- warn rather than "correct" it, since correcting would
        # desync from the scheduler, which owns host block ids.
        logical_per_block = sum(
            ref.page_size_bytes for refs in kv_caches.group_data_refs for ref in refs
        )
        physical_per_block = worker._bytes_per_block  # noqa: SLF001
        if physical_per_block > logical_per_block > 0:
            logger.warning(
                "Spyre host pools need %d bytes per block but the scheduler "
                "budgeted %d (stick padding); host memory use is %.1f%% above "
                "cpu_bytes_to_use.",
                physical_per_block,
                logical_per_block,
                100.0 * (physical_per_block / logical_per_block - 1.0),
            )
        return worker
