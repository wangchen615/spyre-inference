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

"""Runtime-neutral values carried by shared KV offload jobs."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from vllm.v1.kv_offload.base import LoadStoreSpec, OffloadKey

COMPATIBILITY_FORMAT_VERSION = 2
PAGE_KEY_ALGORITHM = "blake2b-64/spyre-m2-page/v1"


@dataclass(frozen=True)
class SharedComponentDescriptor:
    component_id: int
    cache_index: int
    role: str
    layout_kind: str
    layout_version: int
    block_size: int
    local_kv_heads: int
    head_size: int
    page_bytes: int

    def __post_init__(self) -> None:
        if not 0 <= self.component_id <= 0xFFFFFFFF:
            raise ValueError("component_id must fit in four unsigned bytes")
        if self.cache_index < 0:
            raise ValueError("cache_index must be non-negative")
        if self.role not in {"k", "v"}:
            raise ValueError("component role must be 'k' or 'v'")
        if not self.layout_kind:
            raise ValueError("layout_kind must be non-empty")
        if (
            min(
                self.layout_version,
                self.block_size,
                self.local_kv_heads,
                self.head_size,
                self.page_bytes,
            )
            <= 0
        ):
            raise ValueError("component dimensions and versions must be positive")


@dataclass(frozen=True)
class SharedPoolGeometry:
    component_count: int
    slot_bytes: int
    logical_block_capacity: int
    slot_count: int
    actual_pool_bytes: int

    def __post_init__(self) -> None:
        if (
            min(
                self.component_count,
                self.slot_bytes,
                self.logical_block_capacity,
                self.slot_count,
                self.actual_pool_bytes,
            )
            <= 0
        ):
            raise ValueError("shared pool geometry values must be positive")
        if self.slot_count != self.component_count * self.logical_block_capacity:
            raise ValueError("slot_count must cover every component of each logical block")
        if self.actual_pool_bytes != self.slot_count * self.slot_bytes:
            raise ValueError("actual_pool_bytes must equal slot_count * slot_bytes")


@dataclass(frozen=True)
class SharedPageLocation:
    pool_id: int
    slot_id: int

    def __post_init__(self) -> None:
        if self.pool_id < 0 or self.slot_id < 0:
            raise ValueError("shared page pool and slot IDs must be non-negative")


@dataclass(frozen=True)
class SharedPageTransfer:
    component_id: int
    location: SharedPageLocation
    reservation: object | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.component_id <= 0xFFFFFFFF:
            raise ValueError("component_id must fit in four unsigned bytes")
        if not isinstance(self.location, SharedPageLocation):
            raise TypeError("location must be a SharedPageLocation")


@dataclass(frozen=True)
class SharedBlockTransfer:
    key: OffloadKey
    pages: tuple[SharedPageTransfer, ...]

    def __post_init__(self) -> None:
        pages = tuple(self.pages)
        object.__setattr__(self, "pages", pages)
        if not pages:
            raise ValueError("a shared block transfer requires at least one page")
        if any(not isinstance(page, SharedPageTransfer) for page in pages):
            raise TypeError("every page must be a SharedPageTransfer")
        component_ids = tuple(page.component_id for page in pages)
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("component IDs must be unique within a block transfer")
        if len({page.reservation is not None for page in pages}) > 1:
            raise ValueError("a shared block transfer cannot mix load and store pages")

    @property
    def is_store(self) -> bool:
        return self.pages[0].reservation is not None


class SharedLoadStoreSpec(LoadStoreSpec):
    transfers: tuple[SharedBlockTransfer, ...]

    def __init__(self, transfers: Collection[SharedBlockTransfer]) -> None:
        items = tuple(transfers)
        if any(not isinstance(item, SharedBlockTransfer) for item in items):
            raise TypeError("every transfer must be a SharedBlockTransfer")
        if len({item.is_store for item in items}) > 1:
            raise ValueError("a shared transfer spec cannot mix load and store blocks")
        self.transfers = items


def compute_shared_pool_geometry(
    cpu_bytes_to_use: int,
    page_bytes: Sequence[int],
    alignment: int,
) -> SharedPoolGeometry:
    sizes = tuple(page_bytes)
    if cpu_bytes_to_use <= 0 or alignment <= 0 or not sizes or min(sizes) <= 0:
        raise ValueError("shared pool geometry requires positive inputs")

    slot_bytes = ((max(sizes) + alignment - 1) // alignment) * alignment
    component_count = len(sizes)
    logical_capacity = cpu_bytes_to_use // (component_count * slot_bytes)
    if logical_capacity == 0:
        raise ValueError("cpu_bytes_to_use cannot hold one complete KV block")
    slot_count = logical_capacity * component_count
    if slot_count > sys.maxsize // slot_bytes:
        raise OverflowError("shared pool byte size exceeds native size_t")
    return SharedPoolGeometry(
        component_count=component_count,
        slot_bytes=slot_bytes,
        logical_block_capacity=logical_capacity,
        slot_count=slot_count,
        actual_pool_bytes=slot_count * slot_bytes,
    )


def shared_page_hash(key: OffloadKey, component_id: int) -> int:
    if not 0 <= component_id <= 0xFFFFFFFF:
        raise ValueError("component_id must fit in four unsigned bytes")
    digest = hashlib.blake2b(
        bytes(key) + component_id.to_bytes(4, "big"),
        digest_size=8,
        person=b"spyre-m2-page",
    ).digest()
    return int.from_bytes(digest, "big")
