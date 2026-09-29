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

"""Configuration and factories for cross-instance shared KV offloading."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from typing_extensions import override
from vllm.v1.kv_offload.base import OffloadingManager, OffloadingWorker
from vllm.v1.kv_offload.config import OffloadingConfig

from spyre_inference.v1.kv_offload.shared_types import (
    COMPATIBILITY_FORMAT_VERSION,
    SharedPoolFamily,
    allocate_family_slots,
)
from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec
from spyre_inference.v1.worker.spyre_kv_offload import page_signature

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches


def compatibility_digest(payload: Mapping[str, object]) -> bytes:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).digest()


class SpyreSharedOffloadingSpec(SpyreOffloadingSpec):
    """Spyre offloading backed by a cross-instance shared pool family."""

    def __init__(self, config: OffloadingConfig) -> None:
        super().__init__(config)

        if config.parallel.world_size != 1 or config.parallel.tp_size != 1:
            raise ValueError("Spyre shared KV offloading requires world_size == 1 and TP1")
        if len(config.groups) != 1:
            raise ValueError("Spyre shared KV offloading requires one KV cache group")
        if config.model.dtype != "float16":
            raise ValueError("Spyre shared KV offloading requires float16 KV pages")

        metadata_name = self.extra_config.get("shared_metadata_name")
        if not isinstance(metadata_name, str) or not metadata_name.strip():
            raise ValueError("shared_metadata_name must be a non-empty string")
        family_names = self.extra_config.get("shared_pool_families")
        if (
            not isinstance(family_names, Sequence)
            or isinstance(family_names, (str, bytes))
            or not family_names
        ):
            raise ValueError("at least one shared pool family is required")

        self.metadata_name = metadata_name
        self.families: tuple[SharedPoolFamily, ...] = allocate_family_slots(
            self.num_blocks, family_names
        )
        self.max_components = 2 * len(config.groups[0].layer_names)
        if self.max_components == 0:
            raise ValueError("the shared KV cache group must contain at least one layer")
        self._vllm_config: VllmConfig | None = None

    @override
    def bind_vllm_config(self, vllm_config: VllmConfig) -> None:
        backend = vllm_config.parallel_config.distributed_executor_backend
        if backend != "uni":
            raise ValueError(
                "Spyre shared KV offloading requires distributed_executor_backend='uni'"
            )
        self._vllm_config = vllm_config

    def compatibility_payload(self, physical: SpyrePhysicalCaches) -> dict[str, Any]:
        if self._vllm_config is None:
            raise RuntimeError("bind_vllm_config() must be called before worker creation")
        hash_seed = os.environ.get("PYTHONHASHSEED")
        if not hash_seed or hash_seed.lower() == "random":
            raise ValueError("PYTHONHASHSEED must be set to a fixed value for shared KV offloading")

        components = []
        for cache_index, (cache, layout_kind) in enumerate(
            zip(physical.caches, physical.layout_kinds, strict=True)
        ):
            signature = page_signature(cache, layout_kind)
            for role in ("k", "v"):
                components.append(
                    {
                        "cache_index": cache_index,
                        "role": role,
                        "layout_kind": signature.layout_kind,
                        "layout_version": signature.layout_version,
                        "block_size": signature.block_size,
                        "local_kv_heads": signature.local_kv_heads,
                        "head_size": signature.head_size,
                        "page_bytes": signature.page_bytes,
                    }
                )

        return {
            "format": COMPATIBILITY_FORMAT_VERSION,
            "model": self._vllm_config.model_config.model,
            "revision": self._vllm_config.model_config.revision,
            "hash_algorithm": (self._vllm_config.cache_config.prefix_caching_hash_algo),
            "hash_seed": hash_seed,
            "tokens_per_hash": self.tokens_per_hash,
            "tokens_per_block": self.tokens_per_block,
            "dtype": self.config.model.dtype,
            "tp_size": 1,
            "components": tuple(components),
        }

    @override
    def _create_manager(self) -> OffloadingManager:
        from spyre_inference.v1.kv_offload.shared_manager import (
            SpyreSharedOffloadingManager,
        )

        return SpyreSharedOffloadingManager(
            metadata_name=self.metadata_name,
            families=self.families,
            max_components=self.max_components,
            num_blocks=self.num_blocks,
            cache_policy=self.eviction_policy,
            cache_policy_module_path=self.cache_policy_module_path,
            enable_events=self.kv_events_config.enable_kv_cache_events,
            store_threshold=int(self.extra_config.get("store_threshold", 0)),
            max_tracker_size=int(self.extra_config.get("max_tracker_size", 64_000)),
        )

    @override
    def _create_worker(self, physical: SpyrePhysicalCaches) -> OffloadingWorker:
        from spyre_inference.v1.kv_offload.shared_worker import (
            SpyreSharedOffloadingWorker,
        )

        digest = compatibility_digest(self.compatibility_payload(physical))
        return SpyreSharedOffloadingWorker(
            physical=physical,
            metadata_name=self.metadata_name,
            families=self.families,
            compatibility_digest=digest,
            max_components=self.max_components,
        )
