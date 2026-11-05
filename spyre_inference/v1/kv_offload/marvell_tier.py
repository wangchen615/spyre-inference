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

"""m3's Marvell BAR2 tier, parked: not wired into the KV offload path.

`SpyreOffloadingSpec` now attaches a delegator-created pool and accepts only
`pool_backend="host"`. These helpers are m3's create-or-attach Marvell tier
(`SharedMarvellPool.create_or_attach` over spec-planned BAR2 windows), kept
unchanged until Marvell moves onto delegator-planned pools.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Callable, Sequence
from typing import Any

# flex's DEVICE_ALIGNMENT. SharedMarvellPool rounds each slot up to it and
# requires bar_offset to be a multiple of it.
DEVICE_ALIGNMENT = 128

# sysfs root for PCI devices; a parameter so tests can point it at a fake tree.
PCI_SYSFS_ROOT = "/sys/bus/pci/devices"
# IORESOURCE_MEM in <linux/ioport.h>.
_IORESOURCE_MEM = 0x200


def _align_up(n: int, alignment: int = DEVICE_ALIGNMENT) -> int:
    return -(-n // alignment) * alignment


def _align_down(n: int, alignment: int = DEVICE_ALIGNMENT) -> int:
    return (n // alignment) * alignment


def read_bar_size(pci_bdf: str, bar_index: int = 2, sysfs_root: str | None = None) -> int:
    """Size in bytes of BAR `bar_index` of `pci_bdf`, from sysfs `resource`.

    Each line of `/sys/bus/pci/devices/<bdf>/resource` is `start end flags` in
    hex, one line per BAR; this is the same file (and line) flex's
    SharedMarvellPool reads its bus address from, and it is world-readable.
    """
    if not pci_bdf or "/" in pci_bdf:
        raise ValueError(f"invalid PCI BDF {pci_bdf!r}")
    path = os.path.join(sysfs_root or PCI_SYSFS_ROOT, pci_bdf, "resource")
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError as e:
        raise ValueError(
            f"cannot read {path} ({e.strerror}); is marvell_pci_bdf={pci_bdf!r} "
            f"correct and /sys visible?"
        ) from e
    if len(lines) <= bar_index:
        raise ValueError(f"{path} has no BAR{bar_index}")
    try:
        start, end, flags = (int(x, 16) for x in lines[bar_index].split()[:3])
    except ValueError as e:
        raise ValueError(f"{path}: malformed BAR{bar_index} line {lines[bar_index]!r}") from e
    if end <= start or not flags & _IORESOURCE_MEM:
        raise ValueError(
            f"BAR{bar_index} of {pci_bdf} is not a memory BAR "
            f"(start={start:#x} end={end:#x} flags={flags:#x})"
        )
    return end - start + 1


@dataclasses.dataclass(frozen=True)
class MarvellWindow:
    """One per-cache pool's slice of BAR2: `[offset, offset + size)`."""

    offset: int
    size: int

    @property
    def end(self) -> int:
        return self.offset + self.size


def marvell_window_bytes(page_bytes: int, num_slots: int) -> int:
    """BAR2 bytes one per-cache pool occupies.

    `2 * num_slots` slots (K and V) of `page_bytes` each, every slot rounded up
    to DEVICE_ALIGNMENT exactly as SharedMarvellPool rounds it.
    """
    if page_bytes <= 0 or num_slots <= 0:
        raise ValueError(f"page_bytes={page_bytes}, num_slots={num_slots} must be > 0")
    return _align_up(2 * num_slots * _align_up(page_bytes))


def validate_marvell_windows(
    windows: Sequence[MarvellWindow],
    *,
    region_offset: int,
    region_bytes: int,
    bar_size: int,
) -> None:
    """Refuse any window layout that could make two pools share bytes.

    Checks alignment, pairwise disjointness, and that every window lies inside
    `[region_offset, region_offset + region_bytes)` and inside BAR2. flex checks
    only that a single window fits the BAR, so overlap between pools -- the one
    failure that corrupts KV silently -- is caught here or nowhere.
    """
    region_end = region_offset + region_bytes
    ordered = sorted(windows, key=lambda w: w.offset)
    for w in ordered:
        if w.size <= 0 or w.offset < 0 or w.offset % DEVICE_ALIGNMENT:
            raise ValueError(
                f"Marvell window {w} must have a positive size and a "
                f"{DEVICE_ALIGNMENT}-aligned offset"
            )
        if w.offset < region_offset or w.end > region_end:
            raise ValueError(
                f"Marvell window [{w.offset:#x}, {w.end:#x}) lies outside the "
                f"configured region [{region_offset:#x}, {region_end:#x}) "
                f"(marvell_bar_offset + marvell_bytes_to_use); raise "
                f"marvell_bytes_to_use to at least {w.end - region_offset}"
            )
        if w.end > bar_size:
            raise ValueError(
                f"Marvell window [{w.offset:#x}, {w.end:#x}) ends past BAR2 ({bar_size:#x} bytes)"
            )
    for a, b in zip(ordered, ordered[1:]):
        if b.offset < a.end:
            raise ValueError(
                f"Marvell windows overlap: [{a.offset:#x}, {a.end:#x}) and "
                f"[{b.offset:#x}, {b.end:#x}); two pools would share KV bytes"
            )


def plan_marvell_windows(
    page_bytes_per_cache: Sequence[int],
    num_slots: int,
    *,
    region_offset: int,
    region_bytes: int,
    bar_size: int,
) -> list[MarvellWindow]:
    """Assign cache `c` the window right after cache `c - 1`'s, and validate.

    With a uniform page size this is the plan's
    `offset_c = region_offset + c * align_up(2 * num_slots * page_bytes)`; with
    differing page sizes (layers of different head_size) the windows are packed
    cumulatively instead, which reduces to the same formula.
    """
    windows = []
    offset = region_offset
    for pb in page_bytes_per_cache:
        size = marvell_window_bytes(pb, num_slots)
        windows.append(MarvellWindow(offset=offset, size=size))
        offset += size
    validate_marvell_windows(
        windows, region_offset=region_offset, region_bytes=region_bytes, bar_size=bar_size
    )
    return windows


def _pool_class(name: str):
    """`torch_spyre._C.<name>` (SharedHostPool / SharedMarvellPool), lazily."""
    import torch_spyre._C as _C  # ty: ignore[unresolved-import]

    return getattr(_C, name)


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
