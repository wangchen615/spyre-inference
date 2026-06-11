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
``OffloadingConnector`` works on Spyre (RFC §6.3). Subclasses ``OffloadingSpec``
directly rather than ``CPUOffloadingSpec`` (RFC §10 Q5): the CPU spec's
``get_handlers`` is hard-gated behind ``current_platform.is_cuda_alike()``, so the
only thing we would inherit is the small ``num_blocks`` math — which we duplicate
here instead of inheriting a class that documents itself as CUDA-only.

- ``get_manager`` reuses the upstream ``CPUOffloadingManager`` verbatim (pure
  bookkeeping, keyed by ``LoadStoreSpec`` types, nothing CUDA-specific).
- ``get_handlers`` yields the Spyre device<->host handler pair.

**KV ingestion seam.** Upstream ``OffloadingConnectorWorker.register_kv_caches``
canonicalizes the KV cache (``untyped_storage().set_().view(...)``) *before* calling
``spec.get_handlers``. That canonicalization cannot run on Spyre's paged-list cache
(see ``kv_adapter`` and the RFC tracking doc). So instead of receiving the cache
through that path, this spec is *primed* with the raw
``{layer_name: SpyrePagedKVCache}`` dict by ``TorchSpyreModelRunner`` (a minimal,
documented worker-side hook — a deliberate deviation from RFC A1.3's "no worker
change" goal, recorded in the tracking doc). ``get_handlers`` then builds handlers
from the primed dict and ignores its ``CanonicalKVCaches`` argument.
"""

from collections.abc import Iterator

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier
from spyre_inference.v1.kv_offload.handlers import SpyreCpuOffloadingHandlers
from spyre_inference.v1.kv_offload.kv_adapter import build_layer_views

logger = init_logger(__name__)


class SpyreOffloadingSpec(OffloadingSpec):
    """Single-tier (Spyre device <-> host RAM) offloading spec."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise ValueError(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config "
                "for SpyreOffloadingSpec"
            )

        # num_blocks math, duplicated from CPUOffloadingSpec.__init__ (which we do
        # not subclass — see module docstring). One offloaded block per device
        # block (block_size_factor == 1 in M1).
        assert kv_cache_config is not None
        if kv_cache_config.num_blocks > 0:
            total_gpu_kv_bytes = sum(
                t.size for t in kv_cache_config.kv_cache_tensors
            )
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * vllm_config.parallel_config.world_size
        else:
            kv_bytes_per_block = 0

        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        self.num_blocks = (
            int(cpu_bytes_to_use) // kv_bytes_per_offloaded_block
            if kv_bytes_per_offloaded_block > 0
            else 0
        )

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

        # scheduler-side
        self._manager: OffloadingManager | None = None
        # worker-side
        self._copier = SpyreKvDmaCopier()
        self._handlers: SpyreCpuOffloadingHandlers | None = None
        # raw {layer_name: SpyrePagedKVCache}, primed by the model runner.
        self._raw_kv_caches: dict[str, object] | None = None

    # --- worker-side priming hook (called by TorchSpyreModelRunner) ---

    def prime_kv_caches(self, kv_caches: dict[str, object]) -> None:
        """Hand the raw bound paged KV caches to the spec.

        Called once on the worker before ``get_handlers``. See the module
        docstring for why this bypasses upstream canonicalization.
        """
        self._raw_kv_caches = kv_caches

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None
                and kv_events_config.enable_kv_cache_events
            )
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
            )
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        # The CanonicalKVCaches argument is ignored: on Spyre we build handlers
        # from the raw paged dict primed via prime_kv_caches (see module docstring).
        if not self._handlers:
            if self._raw_kv_caches is None:
                raise RuntimeError(
                    "SpyreOffloadingSpec.get_handlers called before prime_kv_caches; "
                    "the TorchSpyreModelRunner hook must run first."
                )
            views = build_layer_views(self._raw_kv_caches, self.num_blocks)
            self._handlers = SpyreCpuOffloadingHandlers(
                views=views,
                block_size_factor=self.block_size_factor,
                copier=self._copier,
            )

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.device_to_host_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.host_to_device_handler
