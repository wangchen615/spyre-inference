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
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from vllm.v1.kv_offload.base import LoadStoreSpec, OffloadKey

COMPATIBILITY_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SharedPoolFamily:
    name: str
    slot_count: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("shared pool family name must be non-empty")
        if self.slot_count <= 0:
            raise ValueError("shared pool family slot_count must be positive")


@dataclass(frozen=True)
class SharedLocation:
    anchor_pool_id: int
    slot_id: int


@dataclass(frozen=True)
class SharedTransfer:
    key: OffloadKey
    location: SharedLocation
    reservation: object | None = None


class SharedLoadStoreSpec(LoadStoreSpec):
    transfers: tuple[SharedTransfer, ...]

    def __init__(self, transfers: Collection[SharedTransfer]) -> None:
        items = tuple(transfers)
        if any(not isinstance(item, SharedTransfer) for item in items):
            raise TypeError("every transfer must be a SharedTransfer")
        if len({item.reservation is not None for item in items}) > 1:
            raise ValueError("a shared transfer spec cannot mix load and store items")
        self.transfers = items


def allocate_family_slots(
    total_slots: int, family_names: Sequence[str]
) -> tuple[SharedPoolFamily, ...]:
    names = tuple(family_names)
    if not names:
        raise ValueError("at least one shared pool family is required")
    if total_slots <= 0:
        raise ValueError("total shared pool slots must be positive")
    if any(not name.strip() for name in names):
        raise ValueError("shared pool family names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError("shared pool family names must be unique")
    if total_slots < len(names):
        raise ValueError("fewer slots than shared pool families")

    base, remainder = divmod(total_slots, len(names))
    return tuple(
        SharedPoolFamily(name, base + (index < remainder)) for index, name in enumerate(names)
    )


def shared_block_hash(key: OffloadKey) -> int:
    digest = hashlib.blake2b(bytes(key), digest_size=8, person=b"spyre-m2").digest()
    return int.from_bytes(digest, "big", signed=False)
