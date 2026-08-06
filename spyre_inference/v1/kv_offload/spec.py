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

"""SpyreOffloadingSpec: single-tier host-RAM KV offload for Spyre.

Implements the upstream ``OffloadingSpec`` contract so the upstream
``OffloadingConnector`` works on Spyre. Subclasses ``OffloadingSpec`` directly rather
than ``CPUOffloadingSpec``: the CPU spec's ``get_worker`` is hard-gated behind
``current_platform.is_cuda_alike() or is_xpu()``, so the only thing we would inherit is
the small ``num_blocks`` math -- which we duplicate here instead of inheriting a class
that refuses to run on this platform.

- ``get_manager`` reuses the upstream ``CPUOffloadingManager`` verbatim (pure
  bookkeeping, keyed by ``LoadStoreSpec`` types, nothing CUDA-specific).
- ``get_worker`` yields the Spyre device<->host worker.

**KV ingestion seam.** Upstream ``OffloadingConnectorWorker.register_kv_caches``
canonicalizes the KV cache (``untyped_storage()`` + ``torch.as_strided``) before
building the worker. That canonicalization cannot run on Spyre's paged-list cache --
each layer is a ``SpyrePagedKVCache`` (two lists of per-block tensors), not one tensor,
and storage reinterpretation is not viable on Spyre device memory (see ``kv_adapter``).
So instead of receiving the cache through that path, this spec is *primed* with the raw
``{layer_name: SpyrePagedKVCache}`` dict by ``TorchSpyreModelRunner``. ``get_worker``
then builds from the primed dict and ignores its ``CanonicalKVCaches`` argument.
"""

from vllm.logger import init_logger
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier
from spyre_inference.v1.kv_offload.kv_adapter import build_layer_views
from spyre_inference.v1.kv_offload.worker import SpyreOffloadingWorker

logger = init_logger(__name__)


class SpyreOffloadingSpec(OffloadingSpec):
    """Single-tier (Spyre device <-> host RAM) offloading spec."""

    BLOCK_SIZE_ALIGNMENT = 1

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config "
                "for SpyreOffloadingSpec"
            )

        if self.blocks_per_chunk != 1:
            raise ValueError(
                "SpyreOffloadingSpec supports one offloaded block per device block "
                f"only; 'block_size' in kv_connector_extra_config yielded "
                f"blocks_per_chunk={self.blocks_per_chunk}."
            )

        # num_blocks math, duplicated from CPUOffloadingSpec.__init__ (which we do not
        # subclass -- see module docstring). worker_kv_bytes_per_block is the per-worker
        # share, so scale by world_size to get the bytes one offloaded block occupies.
        world_size = config.parallel.world_size
        self.num_blocks = 0
        if config.worker_kv_bytes_per_block > 0 and world_size > 0:
            kv_bytes_per_chunk = (
                config.worker_kv_bytes_per_block * world_size * self.blocks_per_chunk
            )
            self.num_blocks = int(cpu_bytes_to_use) // round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

        # scheduler-side
        self._manager: OffloadingManager | None = None
        # worker-side
        self._copier = SpyreKvDmaCopier()
        self._worker: SpyreOffloadingWorker | None = None
        # raw {layer_name: SpyrePagedKVCache}, primed by the model runner.
        self._raw_kv_caches: dict[str, object] | None = None

    def prime_kv_caches(self, kv_caches: dict[str, object]) -> None:
        """Hand the raw bound paged KV caches to the spec.

        Called on the worker by ``TorchSpyreModelRunner`` before ``get_worker``. See
        the module docstring for why this bypasses upstream canonicalization.
        """
        self._raw_kv_caches = kv_caches

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # ty: ignore[invalid-argument-type]
                enable_events=self.kv_events_config.enable_kv_cache_events,
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        # The CanonicalKVCaches argument is ignored: on Spyre we build from the raw
        # paged dict primed via prime_kv_caches (see module docstring).
        if not self._worker:
            if self._raw_kv_caches is None:
                raise RuntimeError(
                    "SpyreOffloadingSpec.get_worker called before prime_kv_caches; "
                    "the TorchSpyreModelRunner hook must run first."
                )
            views = build_layer_views(self._raw_kv_caches, self.num_blocks)
            logger.info(
                "SpyreOffloadingSpec: %d host blocks across %d layer view(s)",
                self.num_blocks,
                len(views),
            )
            self._worker = SpyreOffloadingWorker(
                views=views,
                blocks_per_chunk=self.blocks_per_chunk,
                copier=self._copier,
            )

        return self._worker
