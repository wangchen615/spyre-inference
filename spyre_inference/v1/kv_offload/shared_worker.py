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

"""Worker-side DMA routing through one persistent shared data child."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from typing_extensions import override
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LoadStoreSpec

from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.pool_plan import PlannedPool
from spyre_inference.v1.kv_offload.shared_attachment import (
    SharedMetadataAttachment,
    SharedPoolAttachment,
)
from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
from spyre_inference.v1.kv_offload.shared_runtime import load_shared_runtime
from spyre_inference.v1.kv_offload.shared_types import (
    SharedComponentDescriptor,
    SharedLoadStoreSpec,
    SharedPageTransfer,
    SharedPoolGeometry,
)
from spyre_inference.v1.kv_offload.worker import SpyreOffloadingWorker

logger = init_logger(__name__)


@dataclass(frozen=True)
class _ComponentBinding:
    tensor: object
    page_bytes: int
    domain_id: int
    registered: object | None = None
    data_pool: object | None = None


class SpyreSharedOffloadingWorker(SpyreOffloadingWorker):
    """Copy explicitly located pages through a delegator-created data child."""

    def __init__(
        self,
        *,
        physical: SpyrePhysicalCaches,
        manifest: Sequence[SharedComponentDescriptor],
        identity: SharedPoolIdentity | None = None,
        geometry: SharedPoolGeometry | None = None,
        max_pool_slots: int | None = None,
        plan: PlannedPool | None = None,
        tp_rank: int | None = None,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> None:
        self._finished_jobs = []
        self._physical = physical
        self._runtime = runtime_loader()
        self._attachment: SharedMetadataAttachment | SharedPoolAttachment | None = None
        self._directory = None
        self._registered = None
        self._pool = None
        self._is_planned = plan is not None

        manifest = tuple(manifest)
        if tuple(item.component_id for item in manifest) != tuple(range(len(manifest))):
            raise ValueError("manifest component IDs must be ordered and contiguous")
        if plan is None:
            if identity is None or geometry is None or max_pool_slots is None:
                raise ValueError("legacy shared worker requires identity and pool geometry")
            if len(manifest) != geometry.component_count:
                raise ValueError("manifest component count disagrees with pool geometry")
            if geometry.slot_count > max_pool_slots:
                raise ValueError("pool geometry exceeds the scheduler slot limit")
        elif identity is not None or geometry is not None or max_pool_slots is not None:
            raise ValueError("planned shared worker cannot mix legacy pool geometry")
        elif tp_rank is None:
            raise ValueError("planned shared worker requires a tensor-parallel rank")

        if plan is not None:
            assert tp_rank is not None
            self._validate_planned_manifest(plan, tp_rank, physical, manifest)

        # Resolve device addresses first. A malformed local cache must never
        # acquire a root attachment that can make destroy report PoolBusy.
        components: dict[int, _ComponentBinding] = {}
        for descriptor in manifest:
            if not 0 <= descriptor.cache_index < len(physical.caches):
                raise ValueError("manifest cache index is outside the physical caches")
            if descriptor.role not in ("k", "v"):
                raise ValueError(f"unknown manifest component role {descriptor.role!r}")
            role_index = 0 if descriptor.role == "k" else 1
            tensor = physical.caches[descriptor.cache_index][role_index]
            address = self._runtime.get_composite_address(tensor)
            if address.num_chunks != 1:
                raise NotImplementedError(
                    "shared KV allocations must contain exactly one single chunk"
                )
            if address.total_size % physical.num_blocks:
                raise ValueError(
                    f"KV allocation of {address.total_size} bytes does not divide "
                    f"into {physical.num_blocks} pages"
                )
            page_bytes = address.total_size // physical.num_blocks
            if page_bytes != descriptor.page_bytes:
                raise ValueError(
                    f"component {descriptor.component_id} runtime page size "
                    f"{page_bytes} disagrees with manifest {descriptor.page_bytes}"
                )
            if geometry is not None:
                slot_bytes = geometry.slot_bytes
            else:
                assert plan is not None and tp_rank is not None
                slot_bytes = plan.partitions_for_rank(tp_rank)[descriptor.cache_index].slot_bytes
            if page_bytes > slot_bytes:
                raise ValueError(
                    f"component {descriptor.component_id} page is larger than a pool slot"
                )
            chunk = address.chunks()[0]
            components[descriptor.component_id] = _ComponentBinding(
                tensor, page_bytes, chunk.domain_id
            )

        if plan is None:
            assert identity is not None and geometry is not None
            attachment = SharedPoolAttachment.attach(identity, lambda: self._runtime)
            try:
                registered = attachment.registered
                if (
                    int(registered.slot_count) != geometry.slot_count
                    or int(registered.slot_bytes) != geometry.slot_bytes
                ):
                    raise RuntimeError("registered pool geometry disagrees with requested geometry")
                pool = attachment.data_pool
                if (
                    int(pool.slot_count()) != geometry.slot_count
                    or int(pool.slot_bytes()) != geometry.slot_bytes
                ):
                    raise RuntimeError("resolved pool geometry disagrees with requested geometry")
            except BaseException:
                attachment.close()
                raise
            self._registered = registered
            self._pool = pool
            self._geometry = geometry
            self._known_component_ids = set(components)
            log_name = identity.data_pool_name
            log_pool_name = identity.pool_name
            log_capacity = geometry.logical_block_capacity
            log_slots = geometry.slot_count
            log_slot_bytes = geometry.slot_bytes
            log_actual_bytes = geometry.actual_pool_bytes
            padding_bytes = geometry.logical_block_capacity * sum(
                geometry.slot_bytes - item.page_bytes for item in components.values()
            )
        else:
            assert tp_rank is not None
            attachment = SharedMetadataAttachment.attach(plan, lambda: self._runtime)
            try:
                resolved = attachment.resolve_rank(tp_rank)
                resolved_by_cache = {item.partition.cache_index: item for item in resolved}
                planned_components = {
                    (item.cache_index, item.role): item
                    for item in plan.components_for_rank(tp_rank)
                }
                local_components = {}
                for descriptor in manifest:
                    planned = planned_components[(descriptor.cache_index, descriptor.role)]
                    partition = resolved_by_cache[descriptor.cache_index]
                    pool = partition.data_pool
                    if (
                        int(pool.slot_count()) != partition.partition.num_slots
                        or int(pool.slot_bytes()) != partition.partition.slot_bytes
                    ):
                        raise RuntimeError(
                            f"resolved partition {partition.partition.partition_name!r} "
                            "geometry disagrees with the immutable pool plan"
                        )
                    base = components[descriptor.component_id]
                    local_components[planned.global_component_id] = _ComponentBinding(
                        base.tensor,
                        base.page_bytes,
                        base.domain_id,
                        partition.registered,
                        pool,
                    )
            except BaseException:
                attachment.close()
                raise
            components = local_components
            self._known_component_ids = {item.global_component_id for item in plan.components}
            log_name = f"{plan.arena.name}/rank-{tp_rank}"
            log_pool_name = plan.identity.pool_name
            log_capacity = plan.logical_block_capacity
            log_slots = sum(item.partition.num_slots for item in resolved)
            log_slot_bytes = 0
            log_actual_bytes = sum(
                item.partition.num_slots * item.partition.slot_bytes for item in resolved
            )
            padding_bytes = plan.logical_block_capacity * sum(
                2 * cache.physical_page_bytes for cache in plan.caches_for_rank(tp_rank)
            )
            padding_bytes = log_actual_bytes - padding_bytes

        self._attachment = attachment
        self._directory = attachment.metadata
        self._components = components
        self._bytes_per_block = sum(item.page_bytes for item in components.values())
        logger.info(
            "Spyre shared KV pool %s/%s: %d components, %d logical blocks, "
            "%d slots, %d bytes/slot, %d bytes actual, %d bytes fixed-slot padding",
            log_pool_name,
            log_name,
            len(components),
            log_capacity,
            log_slots,
            log_slot_bytes,
            log_actual_bytes,
            padding_bytes,
        )

    @staticmethod
    def _validate_planned_manifest(
        plan: PlannedPool,
        tp_rank: int,
        physical: SpyrePhysicalCaches,
        manifest: tuple[SharedComponentDescriptor, ...],
    ) -> None:
        caches = plan.caches_for_rank(tp_rank)
        if len(physical.caches) != len(caches) or len(manifest) != 2 * len(caches):
            raise ValueError("physical cache count disagrees with the immutable pool plan")
        for cache in caches:
            pair = manifest[2 * cache.cache_index : 2 * cache.cache_index + 2]
            if tuple((item.cache_index, item.role) for item in pair) != (
                (cache.cache_index, "k"),
                (cache.cache_index, "v"),
            ):
                raise ValueError("physical caches are reordered relative to the pool plan")
            for item in pair:
                if item.page_bytes != cache.physical_page_bytes:
                    raise ValueError("physical page size disagrees with the immutable pool plan")
                if item.layout_kind != cache.layout_kind:
                    raise ValueError("physical cache layout disagrees with the immutable pool plan")
                if item.layout_version != cache.layout_version:
                    raise ValueError(
                        "physical cache layout version disagrees with the immutable pool plan"
                    )
                if item.block_size != plan.block_size:
                    raise ValueError("physical cache block size disagrees with the pool plan")
                if item.local_kv_heads != cache.local_kv_heads:
                    raise ValueError("physical cache KV heads disagree with the pool plan")
                if item.head_size != cache.head_size:
                    raise ValueError("physical cache head size disagrees with the pool plan")

    @staticmethod
    def _same_pool(left: object, right: object) -> bool:
        return all(
            getattr(left, field_name, None) == getattr(right, field_name, None)
            for field_name in ("metadata_version", "pool_id", "pool_version")
        )

    @classmethod
    def _same_slot(cls, left: object, right: object) -> bool:
        return (
            cls._same_pool(left.pool, right.pool)
            and left.slot_id == right.slot_id
            and left.slot_version == right.slot_version
        )

    def _validate_page(self, page: SharedPageTransfer, *, to_device: bool) -> _ComponentBinding:
        component = self._components.get(page.component_id)
        if component is None:
            raise ValueError(f"unknown shared component {page.component_id}")
        registered = component.registered or self._registered
        if registered is None:
            raise RuntimeError("shared component has no registered partition")
        if page.location.pool_id != registered.pool_ref.pool_id:
            raise ValueError(f"foreign shared pool ID {page.location.pool_id}")
        slot_count = int(registered.slot_count)
        if not 0 <= page.location.slot_id < slot_count:
            raise IndexError(f"shared slot {page.location.slot_id} out of range [0, {slot_count})")
        if to_device:
            if page.reservation is not None:
                raise ValueError("shared load pages must not carry reservations")
        else:
            if page.reservation is None:
                raise ValueError("shared store pages require reservations")
            try:
                reservation_slot = page.reservation.slot
                matches = self._same_pool(reservation_slot.pool, registered.pool_ref)
                matches = matches and reservation_slot.slot_id == page.location.slot_id
            except (AttributeError, TypeError):
                matches = False
            if not matches:
                raise ValueError("store reservation does not match its page location")
        return component

    @override
    def _transfer(
        self, host_spec: LoadStoreSpec, gpu_spec: GPULoadStoreSpec, to_device: bool
    ) -> int:
        if not isinstance(host_spec, SharedLoadStoreSpec):
            raise TypeError(
                f"shared worker requires SharedLoadStoreSpec, got {type(host_spec).__name__}"
            )
        transfers = host_spec.transfers
        reservations: list[Any] = (
            []
            if self._is_planned
            else [
                page.reservation
                for transfer in transfers
                for page in transfer.pages
                if page.reservation is not None
            ]
        )
        try:
            self._validate_gpu_spec(gpu_spec)
            device_blocks = list(gpu_spec.block_ids)
            if len(device_blocks) != len(transfers):
                raise ValueError(
                    f"block count mismatch: {len(device_blocks)} device blocks vs "
                    f"{len(transfers)} shared transfers"
                )

            local_pages_by_transfer: list[tuple[SharedPageTransfer, ...]] = []
            routes: list[tuple[_ComponentBinding, ...]] = []
            for transfer in transfers:
                unknown = [
                    page.component_id
                    for page in transfer.pages
                    if page.component_id not in self._known_component_ids
                ]
                if unknown:
                    raise ValueError(f"unknown shared component {unknown[0]}")
                local_pages = tuple(
                    page for page in transfer.pages if page.component_id in self._components
                )
                component_ids = tuple(page.component_id for page in local_pages)
                if len(set(component_ids)) != len(component_ids):
                    raise ValueError("shared block contains duplicate component IDs")
                if to_device and set(component_ids) != set(self._components):
                    raise ValueError("shared load requires every component exactly once")
                route = []
                for page in local_pages:
                    route.append(self._validate_page(page, to_device=to_device))
                    if self._is_planned and page.reservation is not None:
                        reservations.append(page.reservation)
                local_pages_by_transfer.append(local_pages)
                routes.append(tuple(route))

            torch.spyre.synchronize()
            transfer_size = 0
            for device_block, pages, components in zip(
                device_blocks, local_pages_by_transfer, routes, strict=True
            ):
                for page, component in zip(pages, components, strict=True):
                    self._runtime.copy_kv_page_raw(
                        component.tensor,
                        device_block,
                        component.data_pool or self._pool,
                        page.location.slot_id,
                        to_device,
                        True,
                    )
                    transfer_size += component.page_bytes
            torch.spyre.synchronize()

            if not to_device:
                for pages, components in zip(local_pages_by_transfer, routes, strict=True):
                    for page, component in zip(pages, components, strict=True):
                        descriptor = (
                            self._runtime.ChunkDescriptorEntry(
                                component.domain_id, component.page_bytes
                            ),
                        )
                        self._directory.publish(page.reservation, descriptor)
        except Exception:
            torch.spyre.synchronize()
            for reservation in reservations:
                entry = self._directory.lookup(reservation.key)
                if entry is None:
                    self._directory.abort(reservation)
                elif self._same_slot(entry.slot, reservation.slot):
                    self._directory.evict(entry)
            raise

        return transfer_size

    @override
    def shutdown(self) -> None:
        """Fence DMA, drop all child owners, then release the root lock lease."""

        attachment = self._attachment
        if attachment is None:
            return
        try:
            torch.spyre.synchronize()
        finally:
            self._finished_jobs.clear()
            self._components.clear()
            self._pool = None
            self._registered = None
            self._directory = None
            self._physical = None
            self._attachment = None
            attachment.close()
