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
from spyre_inference.v1.kv_offload.shared_attachment import SharedPoolAttachment
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


class SpyreSharedOffloadingWorker(SpyreOffloadingWorker):
    """Copy explicitly located pages through a delegator-created data child."""

    def __init__(
        self,
        *,
        physical: SpyrePhysicalCaches,
        identity: SharedPoolIdentity,
        geometry: SharedPoolGeometry,
        manifest: Sequence[SharedComponentDescriptor],
        max_pool_slots: int,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> None:
        self._finished_jobs = []
        self._physical = physical
        self._runtime = runtime_loader()
        self._attachment: SharedPoolAttachment | None = None
        self._directory = None
        self._registered = None
        self._pool = None

        manifest = tuple(manifest)
        if len(manifest) != geometry.component_count:
            raise ValueError("manifest component count disagrees with pool geometry")
        if tuple(item.component_id for item in manifest) != tuple(range(len(manifest))):
            raise ValueError("manifest component IDs must be ordered and contiguous")
        if geometry.slot_count > max_pool_slots:
            raise ValueError("pool geometry exceeds the scheduler slot limit")

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
            if page_bytes > geometry.slot_bytes:
                raise ValueError(
                    f"component {descriptor.component_id} page is larger than a pool slot"
                )
            chunk = address.chunks()[0]
            components[descriptor.component_id] = _ComponentBinding(
                tensor, page_bytes, chunk.domain_id
            )

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

        self._attachment = attachment
        self._directory = attachment.metadata
        self._registered = registered
        self._pool = pool
        self._geometry = geometry
        self._components = components
        self._bytes_per_block = sum(item.page_bytes for item in components.values())
        padding_bytes = geometry.logical_block_capacity * sum(
            geometry.slot_bytes - item.page_bytes for item in components.values()
        )
        logger.info(
            "Spyre shared KV pool %s/%s: %d components, %d logical blocks, "
            "%d slots, %d bytes/slot, %d bytes actual, %d bytes fixed-slot padding",
            identity.pool_name,
            identity.data_pool_name,
            geometry.component_count,
            geometry.logical_block_capacity,
            geometry.slot_count,
            geometry.slot_bytes,
            geometry.actual_pool_bytes,
            padding_bytes,
        )

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
        if page.location.pool_id != self._registered.pool_ref.pool_id:
            raise ValueError(f"foreign shared pool ID {page.location.pool_id}")
        if not 0 <= page.location.slot_id < self._geometry.slot_count:
            raise IndexError(
                f"shared slot {page.location.slot_id} out of range [0, {self._geometry.slot_count})"
            )
        if to_device:
            if page.reservation is not None:
                raise ValueError("shared load pages must not carry reservations")
        else:
            if page.reservation is None:
                raise ValueError("shared store pages require reservations")
            if (
                not self._same_pool(page.reservation.slot.pool, self._registered.pool_ref)
                or page.reservation.slot.slot_id != page.location.slot_id
            ):
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
        reservations: list[Any] = [
            page.reservation
            for transfer in transfers
            for page in transfer.pages
            if page.reservation is not None
        ]
        try:
            self._validate_gpu_spec(gpu_spec)
            device_blocks = list(gpu_spec.block_ids)
            if len(device_blocks) != len(transfers):
                raise ValueError(
                    f"block count mismatch: {len(device_blocks)} device blocks vs "
                    f"{len(transfers)} shared transfers"
                )

            routes: list[tuple[_ComponentBinding, ...]] = []
            for transfer in transfers:
                component_ids = tuple(page.component_id for page in transfer.pages)
                if len(set(component_ids)) != len(component_ids):
                    raise ValueError("shared block contains duplicate component IDs")
                if to_device and set(component_ids) != set(self._components):
                    raise ValueError("shared load requires every component exactly once")
                routes.append(
                    tuple(self._validate_page(page, to_device=to_device) for page in transfer.pages)
                )

            torch.spyre.synchronize()
            transfer_size = 0
            for device_block, transfer, components in zip(
                device_blocks, transfers, routes, strict=True
            ):
                for page, component in zip(transfer.pages, components, strict=True):
                    self._runtime.copy_kv_page_raw(
                        component.tensor,
                        device_block,
                        self._pool,
                        page.location.slot_id,
                        to_device,
                        True,
                    )
                    transfer_size += component.page_bytes
            torch.spyre.synchronize()

            if not to_device:
                for transfer, components in zip(transfers, routes, strict=True):
                    for page, component in zip(transfer.pages, components, strict=True):
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
