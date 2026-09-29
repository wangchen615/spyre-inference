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

"""OffloadingSpec for Spyre KV offloading to shared host memory or a Marvell card.

## Pool backends

`kv_connector_extra_config["pool_backend"]` selects the tier:

- `"host"` (default, M1 behaviour): one `SharedHostPool` (POSIX shared memory)
  per physical cache, sized by `cpu_bytes_to_use`.
- `"marvell"`: one `SharedMarvellPool` per physical cache, each a window of the
  Marvell card's BAR2. Keys: `marvell_pci_bdf` (required),
  `marvell_bytes_to_use` (required; same sizing arithmetic as
  `cpu_bytes_to_use`), `marvell_bar_offset` (default 0, DEVICE_ALIGNMENT
  aligned), so two engines or a DMA test can use disjoint regions of one card.

Host pools are independent shm objects. Marvell pools are windows of one BAR,
and flex does **not** detect two windows overlapping -- they would silently
share KV bytes. So this spec assigns every cache an explicit, disjoint window
(`plan_marvell_windows`) and refuses to start if the last one does not end
inside both BAR2 and `marvell_bytes_to_use`.

Subclasses `OffloadingSpec` directly rather than `CPUOffloadingSpec`. The CPU
spec's worker construction is built around pinned host tensors and
`torch.accelerator` device indexing, neither of which applies here: Spyre's host
side is a POSIX shared-memory pool pinned through the IOMMU, and its DMA is
issued by `copy_kv_page_raw` against the original rank-4 allocation.

The scheduler side needs no Spyre code at all, so `get_manager` reuses upstream's
`CPUOffloadingManager` verbatim -- it is a plain block-id bookkeeper with no
`current_platform` or `torch.cuda` dependency.

The worker side cannot be built from the canonical KV caches alone (those are
flattened views, which `copy_kv_page_raw` rejects), so the connector hands the
original caches over via `bind_physical_caches` before `get_worker` is called.
"""

import dataclasses
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadingMetricMetadata,
    )

    from spyre_inference.v1.kv_offload.connector import SpyrePhysicalCaches

logger = init_logger(__name__)

POOL_BACKENDS = ("host", "marvell")

# flex's DEVICE_ALIGNMENT; kept in step with spyre_kv_offload.DEVICE_ALIGNMENT
# (not imported, so this module stays importable without the worker stack).
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


@dataclasses.dataclass(frozen=True)
class MarvellTierConfig:
    """Parsed `pool_backend="marvell"` settings, for one rank."""

    pci_bdf: str
    # The whole tier, across ranks, as configured.
    bytes_to_use: int
    bar_offset: int
    bar_size: int
    # This rank's share: [region_offset, region_offset + region_bytes).
    region_offset: int
    region_bytes: int


class SpyreOffloadingSpec(OffloadingSpec):
    """Spyre offloading to host memory or Marvell: upstream manager, Spyre worker."""

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, "OffloadingMetricMetadata"]:
        """Declare the metrics `CPUOffloadingManager` emits.

        Prometheus builds its metric set from the *spec* class
        (`offloading/metrics.py:327`) but the values are reported by the
        *manager*. Since `get_manager` reuses `CPUOffloadingManager` verbatim, the
        metrics it emits are the CPU ones, and the base class's empty default made
        upstream's `observe()` assert on the first request:

            File "offloading/metrics.py", line 489, in observe
                assert key in self._offloading_metric_defs
            AssertionError

        Delegating to `CPUOffloadingSpec`'s classmethod rather than copying its
        four definitions keeps the declaration tied to the manager actually in
        use: if upstream adds a metric to the manager it adds it here too, where a
        copy would silently fall behind and assert again. The classmethod is pure
        -- it reads only `extra_config`, which is what it is handed -- so this
        borrows no CPU-spec instance behaviour. `CPUOffloadingSpec` is still not a
        base class: its `__init__` sizes pinned host tensors via
        `SharedOffloadRegion`, which does not apply here.
        """
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        return CPUOffloadingSpec.build_metric_definitions(extra_config)

    def __init__(self, config: OffloadingConfig) -> None:
        super().__init__(config)

        # One host block must map to exactly one device block. With chunking,
        # upstream's host page is gpu_page_size * blocks_per_chunk and a host
        # block spans several device blocks, but copy_kv_page_raw moves exactly
        # one page into one slot -- there is no representation for that here.
        # Assert on the *derived* value: upstream computes it from either the
        # `blocks_per_chunk` or the `block_size` key.
        if self.blocks_per_chunk != 1:
            raise ValueError(
                f"Spyre KV offloading requires blocks_per_chunk == 1, got "
                f"{self.blocks_per_chunk}. Remove 'blocks_per_chunk' and "
                f"'block_size' from kv_connector_extra_config."
            )

        self.pool_backend: str = str(self.extra_config.get("pool_backend", "host"))
        if self.pool_backend not in POOL_BACKENDS:
            raise ValueError(
                f"pool_backend must be one of {POOL_BACKENDS}, got {self.pool_backend!r}"
            )
        # The key that sizes the tier: upstream's name for host memory (so
        # nothing changes there), its own name for the Marvell card.
        size_key = "cpu_bytes_to_use" if self.pool_backend == "host" else "marvell_bytes_to_use"
        bytes_to_use = self.extra_config.get(size_key)
        if not bytes_to_use:
            raise ValueError(
                f"{size_key} must be specified in kv_connector_extra_config "
                f"for pool_backend={self.pool_backend!r}"
            )
        bytes_to_use = int(bytes_to_use)
        if self.pool_backend == "marvell" and self.extra_config.get("cpu_bytes_to_use"):
            logger.warning(
                "pool_backend='marvell': ignoring cpu_bytes_to_use; the tier is "
                "sized by marvell_bytes_to_use=%d",
                bytes_to_use,
            )

        # Mirrors CPUOffloadingSpec's sizing so the scheduler's host block ids
        # stay in step with upstream's accounting. Note this is *logical* KV
        # bytes; the pools are sized from the physical page size, which may be
        # larger because of stick padding (see get_worker).
        world_size = config.parallel.world_size
        kv_bytes_per_block = config.worker_kv_bytes_per_block * world_size
        if kv_bytes_per_block <= 0:
            raise ValueError(
                f"cannot size host pool: worker_kv_bytes_per_block="
                f"{config.worker_kv_bytes_per_block}, world_size={world_size}"
            )
        self.num_blocks = bytes_to_use // kv_bytes_per_block
        if self.num_blocks <= 0:
            raise ValueError(
                f"{size_key}={bytes_to_use} is too small for even "
                f"one block of {kv_bytes_per_block} bytes. Every lookup would "
                f"miss and nothing would be offloaded."
            )

        self.marvell: MarvellTierConfig | None = None
        if self.pool_backend == "marvell":
            self.marvell = self._parse_marvell(bytes_to_use, config)

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")
        self.cache_policy_module_path: str | None = self.extra_config.get(
            "cache_policy_module_path"
        )

        # Pool names are per engine and per rank: two ranks on one host must not
        # collide on a POSIX shared-memory name.
        self._pool_prefix = f"spyre_kv_{config.engine_id}_r{config.parallel.rank}"

        # scheduler-side
        self._manager: OffloadingManager | None = None
        # worker-side, bound by the connector before get_worker()
        self._physical: SpyrePhysicalCaches | None = None

        logger.info(
            "SpyreOffloadingSpec: backend=%s, %d offload block(s) of %d logical "
            "KV bytes, pool prefix %s%s",
            self.pool_backend,
            self.num_blocks,
            kv_bytes_per_block,
            self._pool_prefix,
            (
                f", Marvell {self.marvell.pci_bdf} BAR2 region "
                f"[{self.marvell.region_offset:#x}, "
                f"{self.marvell.region_offset + self.marvell.region_bytes:#x}) "
                f"of {self.marvell.bar_size:#x}"
                if self.marvell
                else ""
            ),
        )

    def _parse_marvell(self, bytes_to_use: int, config: OffloadingConfig) -> MarvellTierConfig:
        """Validate the Marvell keys and carve out this rank's BAR2 region.

        Checked here, on both the scheduler and the worker side, so a region
        that does not fit BAR2 fails at startup before any cache is allocated.
        The exact per-cache windows depend on the physical page size and are
        validated again in `get_worker`.
        """
        pci_bdf = self.extra_config.get("marvell_pci_bdf")
        if not pci_bdf:
            raise ValueError(
                "marvell_pci_bdf must be specified in kv_connector_extra_config "
                "for pool_backend='marvell'"
            )
        pci_bdf = str(pci_bdf)
        bar_offset = int(self.extra_config.get("marvell_bar_offset", 0))
        if bar_offset < 0 or bar_offset % DEVICE_ALIGNMENT:
            raise ValueError(
                f"marvell_bar_offset={bar_offset} must be a non-negative multiple "
                f"of {DEVICE_ALIGNMENT}"
            )
        bar_size = read_bar_size(pci_bdf, 2)
        if bar_offset + bytes_to_use > bar_size:
            raise ValueError(
                f"marvell_bar_offset={bar_offset} + marvell_bytes_to_use="
                f"{bytes_to_use} ends past BAR2 of {pci_bdf} ({bar_size} bytes)"
            )
        # Like cpu_bytes_to_use, the budget is for all ranks together. Ranks on
        # one card each get a disjoint, aligned share of the region.
        world_size = config.parallel.world_size
        rank = config.parallel.rank
        region_bytes = _align_down(bytes_to_use // world_size)
        if region_bytes <= 0:
            raise ValueError(
                f"marvell_bytes_to_use={bytes_to_use} leaves no aligned region "
                f"per rank at world_size={world_size}"
            )
        return MarvellTierConfig(
            pci_bdf=pci_bdf,
            bytes_to_use=bytes_to_use,
            bar_offset=bar_offset,
            bar_size=bar_size,
            region_offset=bar_offset + rank * region_bytes,
            region_bytes=region_bytes,
        )

    def pool_factories(self, page_bytes_per_cache: Sequence[int]) -> list:
        """One pool factory per physical cache, in cache order.

        Host: the same `HostPoolFactory` for every cache (independent shm
        segments by name). Marvell: one `MarvellPoolFactory` per cache, each over
        its own validated, disjoint BAR2 window.
        """
        from spyre_inference.v1.worker.spyre_kv_offload import (
            HostPoolFactory,
            MarvellPoolFactory,
        )

        if self.marvell is None:
            return [HostPoolFactory() for _ in page_bytes_per_cache]
        windows = plan_marvell_windows(
            page_bytes_per_cache,
            self.num_blocks,
            region_offset=self.marvell.region_offset,
            region_bytes=self.marvell.region_bytes,
            bar_size=self.marvell.bar_size,
        )
        for c, w in enumerate(windows):
            logger.info(
                "Spyre KV offload: cache %d -> Marvell %s BAR2 [%#x, %#x) (%d B)",
                c,
                self.marvell.pci_bdf,
                w.offset,
                w.end,
                w.size,
            )
        return [
            MarvellPoolFactory(
                pci_bdf=self.marvell.pci_bdf, bar_offset=w.offset, window_bytes=w.size
            )
            for w in windows
        ]

    def bind_physical_caches(self, physical: "SpyrePhysicalCaches") -> None:
        """Hand over the original rank-4 caches for the DMA path.

        Called by `SpyreOffloadingConnectorWorker.register_kv_caches` before
        `_init_worker`, because the canonical caches upstream passes to
        `get_worker` are flattened views that `copy_kv_page_raw` rejects.
        """
        self._physical = physical

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for offloading. Values < 2 disable filtering.
            store_threshold = int(self.extra_config.get("store_threshold", 0))
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))
            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,
                cache_policy_module_path=self.cache_policy_module_path,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        """Build the Spyre worker from the *physical* caches.

        `kv_caches` is accepted to satisfy the upstream signature and is used
        only to cross-check that canonicalization and the physical binding agree.
        """
        # Imported here: this module is resolved by OffloadingSpecFactory on the
        # scheduler side too, where the connector module need not be loaded.
        from spyre_inference.v1.kv_offload.worker import SpyreOffloadingWorker
        from spyre_inference.v1.worker.spyre_kv_offload import page_bytes

        if self._physical is None:
            raise RuntimeError(
                "bind_physical_caches() must be called before get_worker(); the "
                "canonical KV caches are flattened views and cannot be used for "
                "DMA. Is SpyreOffloadingConnector in use?"
            )

        expected_tensors = len(self._physical.tensor_idx_to_cache)
        if len(kv_caches.tensors) != expected_tensors:
            raise ValueError(
                f"canonical tensor count {len(kv_caches.tensors)} disagrees with "
                f"the {expected_tensors} recorded during canonicalization"
            )

        # Window assignment needs the *physical* page size (stick padding
        # included), which only the device allocation knows; validate the whole
        # layout before creating a single pool.
        factories = self.pool_factories([page_bytes(cache) for cache in self._physical.caches])
        worker = SpyreOffloadingWorker(
            physical=self._physical,
            num_host_blocks=self.num_blocks,
            pool_prefix=self._pool_prefix,
            pool_factories=factories,
        )

        # The manager hands out host block ids from logical-byte math, while the
        # pools are sized from the physical page size (which includes stick
        # padding). The indices still agree, so this is a host-RAM overrun and
        # not corruption -- warn rather than "correct" it, since correcting would
        # desync from the scheduler, which owns host block ids.
        logical_per_block = sum(
            ref.page_size_bytes for refs in kv_caches.group_data_refs for ref in refs
        )
        physical_per_block = worker._bytes_per_block  # noqa: SLF001
        if physical_per_block > logical_per_block > 0:
            # For Marvell the windows were validated against
            # marvell_bytes_to_use with the physical size already, so this is
            # informational there; for host it is an unchecked RAM overrun.
            logger.warning(
                "Spyre %s pools need %d bytes per block but the scheduler "
                "budgeted %d (stick padding); tier use is %.1f%% above "
                "the configured bytes_to_use.",
                self.pool_backend,
                physical_per_block,
                logical_per_block,
                100.0 * (physical_per_block / logical_per_block - 1.0),
            )
        return worker
