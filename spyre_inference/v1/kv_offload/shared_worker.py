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

"""Worker-side registration and DMA routing for shared KV pool families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import override
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LoadStoreSpec

from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.shared_runtime import load_shared_runtime
from spyre_inference.v1.kv_offload.shared_types import (
    COMPATIBILITY_FORMAT_VERSION,
    SharedLoadStoreSpec,
    SharedPoolFamily,
)
from spyre_inference.v1.kv_offload.worker import SpyreOffloadingWorker
from spyre_inference.v1.worker.spyre_kv_offload import copy_kv_page_pair


@dataclass(frozen=True)
class _FamilyPools:
    slot_count: int
    component_pools: tuple[tuple[object, object], ...]


class SpyreSharedOffloadingWorker(SpyreOffloadingWorker):
    """Route complete logical blocks through versioned shared pool slots."""

    def __init__(
        self,
        *,
        physical: SpyrePhysicalCaches,
        metadata_name: str,
        families: tuple[SharedPoolFamily, ...],
        compatibility_digest: bytes,
        max_components: int,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> None:
        self._finished_jobs = []
        self._physical = physical
        self._runtime = runtime_loader()

        components = []
        for cache in physical.caches:
            cache_components = []
            for tensor in cache:
                address = self._runtime.get_composite_address(tensor)
                if address.num_chunks != 1:
                    raise NotImplementedError(
                        "M2-F3 required: shared KV allocation has "
                        f"{address.num_chunks} CompositeAddress chunks"
                    )
                if address.total_size % physical.num_blocks:
                    raise ValueError(
                        f"KV allocation of {address.total_size} bytes does not divide "
                        f"into {physical.num_blocks} pages"
                    )
                page_bytes = address.total_size // physical.num_blocks
                chunk = address.chunks()[0]
                cache_components.append((tensor, page_bytes, chunk.domain_id))
            components.append(tuple(cache_components))

        capacity = self._runtime.SharedMetadataCapacity(
            len(families) * max_components,
            max(family.slot_count for family in families),
            1,
        )
        config = self._runtime.SharedMetadataConfig(max_components, [], capacity)
        self._directory = self._runtime.SharedMetadata.create_or_attach(metadata_name, config)
        compatibility = self._runtime.CompatibilityDescriptor(
            COMPATIBILITY_FORMAT_VERSION, list(compatibility_digest)
        )

        self._families_by_anchor_id: dict[int, _FamilyPools] = {}
        for family in families:
            pools = []
            anchor_pool_id = None
            for cache_index, cache_components in enumerate(components):
                cache_pools = []
                for role, (_, page_bytes, _) in zip(("k", "v"), cache_components, strict=True):
                    pool_config = self._runtime.SharedDataPoolConfig(
                        f"{family.name}.c{cache_index}.{role}",
                        self._runtime.SharedPoolKind.HOST,
                        family.slot_count,
                        page_bytes,
                        compatibility,
                    )
                    registered = self._directory.register_or_attach_pool(pool_config)
                    pool = self._directory.resolve_pool(registered.pool_ref)
                    if pool is None:
                        raise RuntimeError(
                            f"registered shared pool {pool_config.name!r} could not be resolved"
                        )
                    if anchor_pool_id is None:
                        anchor_pool_id = registered.pool_ref.pool_id
                    cache_pools.append(pool)
                pools.append(tuple(cache_pools))
            assert anchor_pool_id is not None
            self._families_by_anchor_id[anchor_pool_id] = _FamilyPools(
                family.slot_count, tuple(pools)
            )

        # Flex's descriptor describes the claimed c0.k slot. Publishing it
        # after the bundle-wide fence makes every sibling component visible.
        _, anchor_page_bytes, anchor_domain_id = components[0][0]
        self._anchor_chunk_descriptor = (
            self._runtime.ChunkDescriptorEntry(anchor_domain_id, anchor_page_bytes),
        )
        self._bytes_per_block = sum(
            page_bytes for cache_components in components for _, page_bytes, _ in cache_components
        )

    @override
    def _transfer(
        self, host_spec: LoadStoreSpec, gpu_spec: GPULoadStoreSpec, to_device: bool
    ) -> int:
        if not isinstance(host_spec, SharedLoadStoreSpec):
            raise TypeError(
                f"shared worker requires SharedLoadStoreSpec, got {type(host_spec).__name__}"
            )
        self._validate_gpu_spec(gpu_spec)
        transfers = host_spec.transfers
        device_blocks = list(gpu_spec.block_ids)
        if len(device_blocks) != len(transfers):
            raise ValueError(
                f"block count mismatch: {len(device_blocks)} device blocks vs "
                f"{len(transfers)} shared transfers"
            )
        if to_device and any(item.reservation is not None for item in transfers):
            raise ValueError("shared load transfers must not carry reservations")
        if not to_device and any(item.reservation is None for item in transfers):
            raise ValueError("shared store transfers require reservations")

        unpublished = [
            transfer.reservation for transfer in transfers if transfer.reservation is not None
        ]
        published = []
        try:
            routes = []
            for transfer in transfers:
                family = self._families_by_anchor_id.get(transfer.location.anchor_pool_id)
                if family is None:
                    raise ValueError(
                        f"unknown shared anchor pool {transfer.location.anchor_pool_id}"
                    )
                if not 0 <= transfer.location.slot_id < family.slot_count:
                    raise IndexError(
                        f"shared slot {transfer.location.slot_id} out of range "
                        f"[0, {family.slot_count})"
                    )
                routes.append(family)

            torch.spyre.synchronize()
            for device_block, transfer, family in zip(
                device_blocks, transfers, routes, strict=True
            ):
                for cache, (k_pool, v_pool) in zip(
                    self._physical.caches, family.component_pools, strict=True
                ):
                    copy_kv_page_pair(
                        self._runtime.copy_kv_page_raw,
                        cache,
                        device_block,
                        k_pool,
                        transfer.location.slot_id,
                        v_pool,
                        transfer.location.slot_id,
                        to_device,
                        True,
                    )
            torch.spyre.synchronize()

            if not to_device:
                for transfer in transfers:
                    reservation = transfer.reservation
                    self._directory.publish(reservation, self._anchor_chunk_descriptor)
                    unpublished.remove(reservation)
                    published.append(reservation)
        except Exception:
            torch.spyre.synchronize()
            for reservation in published:
                entry = self._directory.lookup(reservation.key)
                if entry is not None:
                    self._directory.evict(entry)
            for reservation in unpublished:
                self._directory.abort(reservation)
            raise

        return len(transfers) * self._bytes_per_block
