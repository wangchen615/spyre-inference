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

"""Spyre-specific KV page offload between device pages and a shared slot pool.

The pool is either host memory (`SharedHostPool`, POSIX shared memory pinned
through the IOMMU) or a window of a Marvell card's BAR2 (`SharedMarvellPool`,
peer-to-peer DMA that never touches host DRAM). Both are `flex::SharedPool`s and
`copy_kv_page_raw` takes either, so nothing below the pool factory knows which
tier it is driving.

Why this is Spyre-specific rather than the generic vLLM offload path:

- A Spyre layer's cache is a `SpyrePagedKVCache`, i.e. two separate device
  allocations (`k_pages`, `v_pages`). The generic path assumes one tensor per
  layer and may hand out an int8 `set_(untyped_storage())` view of it. That
  view has neither the rank-4 shape nor the `spyre_layout` that
  `copy_kv_page_raw` requires, so the copy must be issued against the original
  allocations.
- Because K and V are distinct allocations, one logical page needs *two* host
  slots. This module pairs them as `2 * slot` and `2 * slot + 1`.

`copy_kv_page_raw` derives the device byte range for a page itself and
validates the cache's layout before enqueueing any DMA, so this module only
has to route the right tensor, block and slot. Nothing here computes offsets.

What `copy_kv_page_raw` cannot check is whether the *host slot* holds a page of
the same shape as the cache being restored into: a slot is an untyped byte
window, so a token-major page and a head-major page of identical size are
indistinguishable to it. Two offloaders sharing a pool name therefore agree on
a `PageSignature` (see below), which is compared on attach.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache

logger = logging.getLogger(__name__)


def _import_runtime():
    """Import the torch-spyre entry points lazily.

    Keeps this module importable (and unit-testable for its pairing logic) on
    hosts without a built torch_spyre extension.
    """
    from torch_spyre._C import (  # ty: ignore[unresolved-import]
        copy_kv_page_raw,
        get_composite_address,
    )

    return copy_kv_page_raw, get_composite_address


def _synchronize() -> None:
    torch.spyre.synchronize()


def _pool_class(name: str):
    """`torch_spyre._C.<name>` (SharedHostPool / SharedMarvellPool), lazily."""
    import torch_spyre._C as _C  # ty: ignore[unresolved-import]

    return getattr(_C, name)


# flex's DEVICE_ALIGNMENT (device_memory_addressing.hpp). SharedMarvellPool
# rounds each slot up to it and requires bar_offset to be a multiple of it.
DEVICE_ALIGNMENT = 128


# (name, num_slots, slot_bytes) -> a flex SharedPool handle. `num_slots` is the
# *physical* slot count (K and V each take one); the offloader doubles it.
PoolFactory = Callable[[str, int, int], Any]


class HostPoolFactory:
    """Creates (or attaches to) a `SharedHostPool` -- the M1 host-DRAM tier."""

    backend = "host"

    def __call__(self, name: str, num_slots: int, slot_bytes: int):
        return _pool_class("SharedHostPool").create_or_attach(
            name, num_slots=num_slots, slot_bytes=slot_bytes
        )

    @staticmethod
    def unlink_by_name(name: str) -> None:
        _pool_class("SharedHostPool").unlink_by_name(name)

    def __repr__(self) -> str:
        return "HostPoolFactory()"


@dataclasses.dataclass(frozen=True)
class MarvellPoolFactory:
    """Creates (or attaches to) a `SharedMarvellPool` over one BAR2 window.

    flex does not detect overlapping windows of two differently named pools --
    they silently share bytes -- so the window is assigned by the caller
    (`SpyreOffloadingSpec`) and checked here once more against what flex
    actually allocated: if flex's slot rounding ever diverges from the spec's
    `marvell_window_bytes`, the pool would spill into the next cache's window,
    and that is refused rather than served.
    """

    pci_bdf: str
    bar_offset: int
    # Bytes reserved for this pool starting at bar_offset; 0 means unchecked.
    window_bytes: int = 0

    backend = "marvell"

    def __post_init__(self) -> None:
        if not self.pci_bdf:
            raise ValueError("MarvellPoolFactory needs a PCI BDF")
        if self.bar_offset < 0 or self.bar_offset % DEVICE_ALIGNMENT:
            raise ValueError(
                f"bar_offset {self.bar_offset} must be a non-negative multiple of "
                f"{DEVICE_ALIGNMENT}"
            )

    def __call__(self, name: str, num_slots: int, slot_bytes: int):
        pool = _pool_class("SharedMarvellPool").create_or_attach(
            name, self.pci_bdf, num_slots, slot_bytes, self.bar_offset
        )
        total = int(pool.total_bytes())
        if self.window_bytes and total > self.window_bytes:
            del pool
            raise ValueError(
                f"SharedMarvellPool {name!r} occupies {total} B but its BAR2 window "
                f"at offset {self.bar_offset} is only {self.window_bytes} B; it "
                f"would overlap the next window"
            )
        return pool

    @staticmethod
    def unlink_by_name(name: str) -> None:
        _pool_class("SharedMarvellPool").unlink_by_name(name)


def k_slot(slot: int) -> int:
    """Host slot holding the K half of logical page slot `slot`."""
    return 2 * slot


def v_slot(slot: int) -> int:
    """Host slot holding the V half of logical page slot `slot`."""
    return 2 * slot + 1


def page_bytes(cache: SpyrePagedKVCache) -> int:
    """Bytes in one K (or V) page of `cache`, including device stick padding.

    Read from the device allocation rather than computed from the logical
    shape: the two differ whenever `head_size` is not a whole number of sticks.
    """
    _, get_composite_address = _import_runtime()
    k_pages = cache[0]
    num_blocks = k_pages.shape[0]
    total = get_composite_address(k_pages).total_size
    if total % num_blocks:
        raise ValueError(
            f"KV cache of {total} B does not divide into {num_blocks} equal pages; "
            "the allocation has inter-page padding and cannot be offloaded per page"
        )
    return total // num_blocks


@dataclasses.dataclass(frozen=True)
class PageSignature:
    """Identity of a physical KV page layout.

    Two pages may be copied into each other's slots only if their signatures
    are equal. Byte length alone is not sufficient: a token-major [B,S,H,D]
    page and a head-major [B,H,S,D] page with the same block size and head
    count occupy exactly the same number of bytes but order their elements
    differently, so restoring one as the other is silent corruption.

    `layout_version` is bumped by hand if the meaning of any field changes.
    Fields that would distinguish two pages on the same card are omitted
    deliberately: device generation and TP rank cannot differ between two
    processes sharing one host pool for one card's pages, so they would be
    constant. Add them if the pool is ever shared across cards.
    """

    layout_kind: str
    device_dtype: str
    block_size: int
    local_kv_heads: int
    head_size: int
    page_bytes: int
    layout_version: int = 1

    def describe(self) -> str:
        return (
            f"{self.layout_kind} v{self.layout_version} {self.device_dtype} "
            f"block={self.block_size} kv_heads={self.local_kv_heads} "
            f"head={self.head_size} page={self.page_bytes}B"
        )


def page_signature(cache: SpyrePagedKVCache, layout_kind: str) -> PageSignature:
    """Derive the signature of `cache`, given which layout its backend produced.

    `layout_kind` must be "token-major" ([B,S,H,D], SpyreAttentionImpl) or
    "head-major" ([B,H,S,D], SpyreHeadMajorAttentionImpl).

    It is a parameter rather than something inferred from the tensor because the
    layouts are genuinely indistinguishable from shape and device layout alone.
    Both fold logical dim 1 into device dim 0, so device_size[0] // num_blocks
    equals size(1) for either one:

        token-major  logical (8,128,8,128) -> device [1024,8,2,64]  inner=128=S
        head-major   logical (8,8,128,128) -> device [  64,128,2,64] inner=8=H

    Only the allocating backend knows whether dim 1 counts tokens or heads.
    Guessing it would mislabel one layout as the other and, worse, swap the
    block_size and local_kv_heads fields, so a mismatch would still be caught
    but reported as nonsense.
    """
    if layout_kind not in ("token-major", "head-major"):
        raise ValueError(f"layout_kind must be 'token-major' or 'head-major', got {layout_kind!r}")

    k_pages = cache[0]
    num_blocks, dim1, dim2, head_size = (int(d) for d in k_pages.shape)
    if layout_kind == "token-major":
        block_size, kv_heads = dim1, dim2
    else:
        kv_heads, block_size = dim1, dim2

    return PageSignature(
        layout_kind=layout_kind,
        device_dtype=str(k_pages.dtype),
        block_size=block_size,
        local_kv_heads=kv_heads,
        head_size=head_size,
        page_bytes=page_bytes(cache),
    )


class SpyreKvPageOffloader:
    """Moves whole K/V page pairs between Spyre pages and one shared slot pool.

    One offloader serves one layer's cache. `num_slots` is the number of
    *logical* pages the pool holds; the pool itself is created with twice that
    many slots so K and V each get their own.

    The pool comes from `pool_factory` (default: host memory, as in M1), which
    is how the Marvell tier is selected without this class knowing about it.
    """

    def __init__(
        self,
        cache: SpyrePagedKVCache,
        pool_name: str,
        num_slots: int,
        layout_kind: str,
        expect_signature: PageSignature | None = None,
        pool_factory: PoolFactory | None = None,
    ) -> None:
        copy_kv_page_raw, _ = _import_runtime()
        self._copy = copy_kv_page_raw
        self._cache = cache
        self._signature = page_signature(cache, layout_kind)
        # Check 8 of the contract: a caller that shares a pool with another
        # process passes the signature it expects the slots to hold. Mismatches
        # are rejected here, before any page is written or read, because the
        # slot bytes themselves carry no layout information.
        if expect_signature is not None and expect_signature != self._signature:
            raise ValueError(
                "KV page signature mismatch for pool "
                f"{pool_name!r}: slots hold {expect_signature.describe()}, "
                f"this cache is {self._signature.describe()}"
            )
        self._page_bytes = self._signature.page_bytes
        self._num_slots = num_slots
        self._pool_name = pool_name
        self._pool_factory = pool_factory or HostPoolFactory()
        # Two host slots per logical page: one for K, one for V.
        self._pool = self._pool_factory(pool_name, 2 * num_slots, self._page_bytes)
        # A pool attached by name keeps the geometry its creator chose; flex
        # rejects a mismatch, but pin it here too so a factory that ignored its
        # arguments cannot hand back a pool whose slots are too small.
        if int(self._pool.slot_count()) != 2 * num_slots or (
            int(self._pool.slot_bytes()) < self._page_bytes
        ):
            got = (int(self._pool.slot_count()), int(self._pool.slot_bytes()))
            self._pool = None
            raise ValueError(
                f"pool {pool_name!r} has {got[0]} slots of {got[1]} B; this cache "
                f"needs {2 * num_slots} slots of >= {self._page_bytes} B"
            )
        logger.debug(
            "SpyreKvPageOffloader(%s, %r): %d logical slots, %s",
            pool_name,
            self._pool_factory,
            num_slots,
            self._signature.describe(),
        )

    @property
    def pool(self):
        if self._pool is None:
            raise RuntimeError(f"pool {self._pool_name!r} has been released")
        return self._pool

    @property
    def pool_name(self) -> str:
        return self._pool_name

    @property
    def backend(self) -> str:
        return getattr(self._pool_factory, "backend", "custom")

    def release(self) -> None:
        """Drop this process's attach on the pool.

        Destroying the last flex handle for a name unlinks it (the whole host
        segment for SharedHostPool, the `.ctl` control segment for
        SharedMarvellPool). Doing it explicitly on shutdown, rather than relying
        on interpreter teardown to run the pybind destructor, is what makes an
        orderly exit leave nothing behind in /dev/shm. This releases the attach
        refcount rather than calling `unlink_by_name`, which would ignore other
        processes still attached. Idempotent.
        """
        if self._pool is None:
            return
        # Nothing may still be in flight against the slots being released.
        _synchronize()
        self._pool = None

    @property
    def page_size_bytes(self) -> int:
        return self._page_bytes

    @property
    def signature(self) -> PageSignature:
        """Signature of the pages this offloader reads and writes.

        Pass it to another offloader's `expect_signature` to assert that both
        agree on the layout before they share a pool.
        """
        return self._signature

    def _check_slot(self, slot: int) -> None:
        if not 0 <= slot < self._num_slots:
            raise IndexError(f"host slot {slot} out of range [0, {self._num_slots})")

    def offload(self, block_id: int, slot: int, non_blocking: bool = False) -> None:
        """Copy device page `block_id` (K and V) out to host slot `slot`."""
        self._check_slot(slot)
        pool = self.pool
        k_pages, v_pages = self._cache
        self._copy(k_pages, block_id, pool, k_slot(slot), False, non_blocking)
        self._copy(v_pages, block_id, pool, v_slot(slot), False, non_blocking)

    def reload(self, slot: int, block_id: int, non_blocking: bool = False) -> None:
        """Copy host slot `slot` back into device page `block_id`.

        `block_id` need not be the block the page was offloaded from: a page is
        position-independent, so a restore may relocate it.
        """
        self._check_slot(slot)
        pool = self.pool
        k_pages, v_pages = self._cache
        self._copy(k_pages, block_id, pool, k_slot(slot), True, non_blocking)
        self._copy(v_pages, block_id, pool, v_slot(slot), True, non_blocking)

    def offload_many(self, pairs: list[tuple[int, int]], non_blocking: bool = False) -> None:
        """Offload several `(block_id, slot)` pairs.

        Issued as independent single-page copies. Batching them into one
        descriptor list is deferred work (see the plan's gather/scatter
        section); the loop is correct meanwhile and keeps the device-side
        contract identical.
        """
        for block_id, slot in pairs:
            self.offload(block_id, slot, non_blocking=non_blocking)

    def reload_many(self, pairs: list[tuple[int, int]], non_blocking: bool = False) -> None:
        """Reload several `(slot, block_id)` pairs."""
        for slot, block_id in pairs:
            self.reload(slot, block_id, non_blocking=non_blocking)

    def synchronize(self) -> None:
        """Wait for any `non_blocking=True` copies issued above."""
        torch.spyre.synchronize()
