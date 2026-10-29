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

"""Configuration and factories for persistent shared KV offloading."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from typing_extensions import override
from vllm.v1.kv_offload.base import OffloadingManager, OffloadingWorker
from vllm.v1.kv_offload.config import OffloadingConfig

from spyre_inference.v1.kv_offload.shared_types import (
    COMPATIBILITY_FORMAT_VERSION,
    PAGE_KEY_ALGORITHM,
    SharedComponentDescriptor,
    compute_shared_pool_geometry,
)
from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec
from spyre_inference.v1.worker.spyre_kv_offload import page_signature

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches


def compatibility_digest(payload: Mapping[str, object]) -> bytes:
    """Return a stable digest for a complete shared-page interpretation."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).digest()


def component_manifest(
    physical: SpyrePhysicalCaches,
) -> tuple[SharedComponentDescriptor, ...]:
    """Describe physical caches in stable cache, then K/V order."""

    components: list[SharedComponentDescriptor] = []
    for cache_index, (cache, layout_kind) in enumerate(
        zip(physical.caches, physical.layout_kinds, strict=True)
    ):
        signature = page_signature(cache, layout_kind)
        for role in ("k", "v"):
            components.append(
                SharedComponentDescriptor(
                    component_id=len(components),
                    cache_index=cache_index,
                    role=role,
                    layout_kind=signature.layout_kind,
                    layout_version=signature.layout_version,
                    block_size=signature.block_size,
                    local_kv_heads=signature.local_kv_heads,
                    head_size=signature.head_size,
                    page_bytes=signature.page_bytes,
                )
            )
    return tuple(components)


class SpyreSharedOffloadingSpec(SpyreOffloadingSpec):
    """Spyre offloading through a delegator-created composite pool generation."""

    supports_partitioned_pool_plan = True

    def __init__(self, config: OffloadingConfig) -> None:
        super().__init__(config)

        if len(config.groups) != 1:
            raise ValueError("Spyre shared KV offloading requires one KV cache group")
        if config.model.dtype != "float16":
            raise ValueError("Spyre shared KV offloading requires float16 KV pages")
        if "shared_pool_families" in self.extra_config:
            raise ValueError("shared_pool_families was removed; configure one pool_name instead")

        if config.parallel.tp_size > 1 and self.shared_pool_plan is None:
            raise ValueError("world_size/tp_size greater than one requires a complete pool_plan")
        if self.shared_pool_plan is not None:
            if self.shared_pool_plan.tensor_parallel_size != config.parallel.tp_size:
                raise ValueError(
                    "pool_plan tensor parallel size does not match the vLLM configuration"
                )
            if self.shared_pool_plan.model != config.model.name:
                raise ValueError("pool_plan model does not match the vLLM configuration")
            if self.shared_pool_plan.kv_dtype != config.model.dtype:
                raise ValueError("pool_plan KV dtype does not match the vLLM configuration")
            if self.shared_pool_plan.block_size != config.groups[0].tokens_per_block:
                raise ValueError("pool_plan block size does not match the KV cache group")

        self.component_count = (
            len(self.shared_pool_plan.components)
            if self.shared_pool_plan is not None
            else 2 * len(config.groups[0].layer_names)
        )
        if self.component_count == 0:
            raise ValueError("the shared KV cache group must contain at least one layer")
        self.max_pool_slots = (
            sum(partition.num_slots for partition in self.shared_pool_plan.partitions)
            if self.shared_pool_plan is not None
            else self.num_blocks * self.component_count
        )
        self.cpu_bytes_to_use = (
            self.shared_pool_plan.cpu_bytes_to_use
            if self.shared_pool_plan is not None
            else int(self.extra_config["cpu_bytes_to_use"])
        )
        self._vllm_config: VllmConfig | None = None

    @override
    def bind_vllm_config(self, vllm_config: VllmConfig) -> None:
        backend = vllm_config.parallel_config.distributed_executor_backend
        tp_size = self.config.parallel.tp_size
        required_backend = "uni" if tp_size == 1 else "external_launcher"
        if backend != required_backend:
            raise ValueError(
                "Spyre shared KV offloading with "
                f"TP{tp_size} requires distributed_executor_backend={required_backend!r}"
            )
        if self.shared_pool_plan is not None and (
            vllm_config.model_config.model != self.shared_pool_plan.model
            or (vllm_config.model_config.revision or "") != self.shared_pool_plan.model_revision
        ):
            raise ValueError("resolved model and revision do not match the immutable pool_plan")
        self._vllm_config = vllm_config

    def compatibility_payload(
        self, manifest: Sequence[SharedComponentDescriptor]
    ) -> dict[str, Any]:
        """Describe every value that changes shared KV byte interpretation."""

        if self._vllm_config is None:
            raise RuntimeError("bind_vllm_config() must be called before worker creation")
        hash_seed = os.environ.get("PYTHONHASHSEED")
        if not hash_seed or hash_seed.lower() == "random":
            raise ValueError("PYTHONHASHSEED must be set to a fixed value for shared KV offloading")

        return {
            "format": COMPATIBILITY_FORMAT_VERSION,
            "page_key_algorithm": PAGE_KEY_ALGORITHM,
            "model": self._vllm_config.model_config.model,
            "revision": self._vllm_config.model_config.revision,
            "hash_algorithm": self._vllm_config.cache_config.prefix_caching_hash_algo,
            "hash_seed": hash_seed,
            "tokens_per_hash": self.tokens_per_hash,
            "tokens_per_block": self.tokens_per_block,
            "dtype": self.config.model.dtype,
            "tp_size": self.config.parallel.tp_size,
            "components": tuple(asdict(component) for component in manifest),
        }

    @override
    def _create_manager(self) -> OffloadingManager:
        from spyre_inference.v1.kv_offload.shared_manager import (
            SpyreSharedOffloadingManager,
        )

        if self.shared_pool_plan is not None:
            return SpyreSharedOffloadingManager(
                plan=self.shared_pool_plan,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=int(self.extra_config.get("store_threshold", 0)),
                max_tracker_size=int(self.extra_config.get("max_tracker_size", 64_000)),
            )
        return SpyreSharedOffloadingManager(
            cache_policy=self.eviction_policy,
            cache_policy_module_path=self.cache_policy_module_path,
            enable_events=self.kv_events_config.enable_kv_cache_events,
            store_threshold=int(self.extra_config.get("store_threshold", 0)),
            max_tracker_size=int(self.extra_config.get("max_tracker_size", 64_000)),
            identity=self.shared_pool_identity,
            component_count=self.component_count,
            max_pool_slots=self.max_pool_slots,
        )

    @override
    def _create_worker(self, physical: SpyrePhysicalCaches) -> OffloadingWorker:
        from spyre_inference.v1.kv_offload.shared_worker import (
            SpyreSharedOffloadingWorker,
        )

        manifest = component_manifest(physical)
        if self.shared_pool_plan is not None:
            tp_rank = self.config.parallel.rank % self.config.parallel.tp_size
            local_component_count = len(self.shared_pool_plan.components_for_rank(tp_rank))
            if len(manifest) != local_component_count:
                raise ValueError(
                    f"physical component count {len(manifest)} disagrees with "
                    f"planned rank {tp_rank} component count {local_component_count}"
                )
            compatibility_digest(self.compatibility_payload(manifest))
            return SpyreSharedOffloadingWorker(
                physical=physical,
                manifest=manifest,
                plan=self.shared_pool_plan,
                tp_rank=tp_rank,
            )

        if len(manifest) != self.component_count:
            raise ValueError(
                f"physical component count {len(manifest)} disagrees with "
                f"scheduler component count {self.component_count}"
            )
        geometry = compute_shared_pool_geometry(
            self.cpu_bytes_to_use,
            tuple(component.page_bytes for component in manifest),
            mmap.PAGESIZE,
        )
        if geometry.slot_count > self.max_pool_slots:
            raise ValueError(
                f"physical geometry requires {geometry.slot_count} slots, exceeding "
                f"the scheduler limit {self.max_pool_slots}"
            )

        # Validate process-wide interpretation inputs before attaching. The
        # generation itself identifies the delegator's immutable compatibility.
        compatibility_digest(self.compatibility_payload(manifest))
        return SpyreSharedOffloadingWorker(
            physical=physical,
            identity=self.shared_pool_identity,
            geometry=geometry,
            manifest=manifest,
            max_pool_slots=self.max_pool_slots,
        )
