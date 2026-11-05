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

"""m3's parked Marvell BAR2 helpers and the spec's pool_backend guard, without a card.

flex does not detect two m3 windows overlapping: they would silently share KV
bytes. The parked window arithmetic and its validation stay pinned here until
the Marvell tier moves onto delegator-planned pools.

BAR2 is read from a fake sysfs tree, and the factories' `torch_spyre._C`
lookups are replaced by a recorder, so nothing here needs a device.
"""

from __future__ import annotations

import pytest

from spyre_inference.v1.kv_offload import marvell_tier
from spyre_inference.v1.kv_offload.marvell_tier import (
    DEVICE_ALIGNMENT,
    HostPoolFactory,
    MarvellPoolFactory,
    MarvellWindow,
    marvell_window_bytes,
    plan_marvell_windows,
    read_bar_size,
    validate_marvell_windows,
)
from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec
from tests.kv_offload.test_spec import _config

BDF = "0000:83:00.0"
GiB = 1 << 30
# worker-36's Marvell BAR2 as sysfs reports it: 512 GiB at 0xb8000000000.
BAR2_START = 0xB8000000000
BAR2_SIZE = 512 * GiB
PAGE = 262144  # one K page at the benchmarked geometry


def _resource(bar2_start=BAR2_START, bar2_size=BAR2_SIZE, flags=0x14220C) -> str:
    zero = "0x0000000000000000 0x0000000000000000 0x0000000000000000"
    bar0 = "0x00000c0000000000 0x00000c00003fffff 0x000000000014220c"
    bar2 = f"{bar2_start:#018x} {bar2_start + bar2_size - 1:#018x} {flags:#018x}"
    return "\n".join([bar0, zero, bar2, zero, zero, zero]) + "\n"


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    """A fake /sys/bus/pci/devices with the Marvell card's resource file."""

    def _write(bdf=BDF, text=None):
        d = tmp_path / bdf
        d.mkdir(parents=True, exist_ok=True)
        (d / "resource").write_text(_resource() if text is None else text)

    _write()
    monkeypatch.setattr(marvell_tier, "PCI_SYSFS_ROOT", str(tmp_path))
    return _write


# --- sysfs BAR2 lookup -------------------------------------------------------


def test_read_bar_size_parses_line_two(sysfs):
    assert read_bar_size(BDF, 2) == BAR2_SIZE


def test_read_bar_size_rejects_missing_device(sysfs):
    with pytest.raises(ValueError, match="cannot read"):
        read_bar_size("0000:ff:1f.7", 2)


@pytest.mark.parametrize("bdf", ["", "../0000:83:00.0"])
def test_read_bar_size_rejects_bad_bdf(sysfs, bdf):
    with pytest.raises(ValueError, match="invalid PCI BDF"):
        read_bar_size(bdf, 2)


def test_read_bar_size_rejects_unassigned_bar(sysfs):
    sysfs(text=_resource(bar2_start=0, bar2_size=1, flags=0))
    with pytest.raises(ValueError, match="not a memory BAR"):
        read_bar_size(BDF, 2)


def test_read_bar_size_rejects_short_file(sysfs):
    sysfs(text="0x0 0x0 0x0\n")
    with pytest.raises(ValueError, match="no BAR2"):
        read_bar_size(BDF, 2)


# --- spec guard ---------------------------------------------------------------


def test_default_backend_is_host():
    SpyreOffloadingSpec(_config())
    SpyreOffloadingSpec(_config(extra={"pool_backend": "host"}))


@pytest.mark.parametrize("backend", ["marvell", "nvme"])
def test_non_host_backend_is_rejected_not_silently_host(backend):
    """m4's spec would otherwise ignore the key and serve from a host pool."""
    with pytest.raises(ValueError, match="pool_backend"):
        SpyreOffloadingSpec(_config(extra={"pool_backend": backend}))


# --- window arithmetic -------------------------------------------------------


def test_window_bytes_is_k_and_v_slots_aligned():
    assert marvell_window_bytes(PAGE, 32) == 2 * 32 * PAGE
    # Unaligned pages are rounded per slot, as SharedMarvellPool rounds them.
    assert marvell_window_bytes(100, 3) == 6 * DEVICE_ALIGNMENT


@pytest.mark.parametrize("page_bytes,num_slots", [(0, 1), (1, 0), (-1, 4)])
def test_window_bytes_rejects_degenerate(page_bytes, num_slots):
    with pytest.raises(ValueError):
        marvell_window_bytes(page_bytes, num_slots)


def test_plan_matches_the_plan_formula():
    """cache c at bar_offset + c * align_up(2 * num_slots * page_bytes)."""
    base, n, caches = 1 << 20, 32, 4
    stride = marvell_window_bytes(PAGE, n)
    windows = plan_marvell_windows(
        [PAGE] * caches,
        n,
        region_offset=base,
        region_bytes=caches * stride,
        bar_size=BAR2_SIZE,
    )
    assert [w.offset for w in windows] == [base + c * stride for c in range(caches)]
    assert all(w.size == stride for w in windows)
    # Adjacent, never overlapping.
    assert all(a.end == b.offset for a, b in zip(windows, windows[1:]))


def test_plan_packs_mixed_page_sizes_cumulatively():
    windows = plan_marvell_windows(
        [PAGE, 2 * PAGE, PAGE], 4, region_offset=0, region_bytes=1 << 30, bar_size=BAR2_SIZE
    )
    assert [w.size for w in windows] == [8 * PAGE, 16 * PAGE, 8 * PAGE]
    assert [w.offset for w in windows] == [0, 8 * PAGE, 24 * PAGE]


def test_plan_rejects_overflow_of_bytes_to_use():
    """The physical page may be larger than the logical one (stick padding)."""
    n, caches = 32, 4
    stride = marvell_window_bytes(PAGE, n)
    with pytest.raises(ValueError, match="marvell_bytes_to_use"):
        plan_marvell_windows(
            [PAGE] * caches,
            n,
            region_offset=0,
            region_bytes=caches * stride - 1,
            bar_size=BAR2_SIZE,
        )


def test_plan_rejects_overflow_of_bar2():
    stride = marvell_window_bytes(PAGE, 32)
    with pytest.raises(ValueError, match="BAR2"):
        plan_marvell_windows(
            [PAGE] * 2,
            32,
            region_offset=0,
            region_bytes=4 * stride,
            bar_size=2 * stride - DEVICE_ALIGNMENT,
        )


def test_plan_accepts_last_window_ending_exactly_at_bar2_end():
    stride = marvell_window_bytes(PAGE, 32)
    windows = plan_marvell_windows(
        [PAGE] * 2, 32, region_offset=0, region_bytes=2 * stride, bar_size=2 * stride
    )
    assert windows[-1].end == 2 * stride


def test_validate_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_marvell_windows(
            [MarvellWindow(0, 4096), MarvellWindow(4096 - DEVICE_ALIGNMENT, 4096)],
            region_offset=0,
            region_bytes=1 << 20,
            bar_size=BAR2_SIZE,
        )


def test_validate_rejects_overlap_regardless_of_order():
    with pytest.raises(ValueError, match="overlap"):
        validate_marvell_windows(
            [MarvellWindow(8192, 4096), MarvellWindow(0, 8192 + DEVICE_ALIGNMENT)],
            region_offset=0,
            region_bytes=1 << 20,
            bar_size=BAR2_SIZE,
        )


def test_validate_rejects_window_before_region():
    with pytest.raises(ValueError, match="outside the configured region"):
        validate_marvell_windows(
            [MarvellWindow(0, 4096)],
            region_offset=4096,
            region_bytes=1 << 20,
            bar_size=BAR2_SIZE,
        )


def test_validate_rejects_misaligned_offset():
    with pytest.raises(ValueError, match="aligned"):
        validate_marvell_windows(
            [MarvellWindow(64, 4096)], region_offset=0, region_bytes=1 << 20, bar_size=BAR2_SIZE
        )


def test_validate_accepts_adjacent_windows():
    validate_marvell_windows(
        [MarvellWindow(0, 4096), MarvellWindow(4096, 4096)],
        region_offset=0,
        region_bytes=8192,
        bar_size=BAR2_SIZE,
    )


# --- factories ----------------------------------------------------------------


class _FakePool:
    def __init__(self, kind, name, num_slots, slot_bytes, bar_offset=None, bdf=None):
        self.kind, self._name = kind, name
        self._num_slots, self._bdf, self.bar_offset = num_slots, bdf, bar_offset
        self._slot_bytes = -(-slot_bytes // DEVICE_ALIGNMENT) * DEVICE_ALIGNMENT
        self.bus_address = BAR2_START + (bar_offset or 0)

    def slot_count(self):
        return self._num_slots

    def slot_bytes(self):
        return self._slot_bytes

    def total_bytes(self):
        return self._num_slots * self._slot_bytes

    def name(self):
        return self._name


@pytest.fixture
def fake_c(monkeypatch):
    """Replace the factories' torch_spyre._C lookup with a recorder.

    Only `marvell_tier._pool_class` is patched; `torch_spyre._C` itself is left
    alone, since swapping it in sys.modules would leak into any torch_spyre
    submodule first imported during the test.
    """
    log: list[tuple] = []

    class SharedHostPool:
        @staticmethod
        def create_or_attach(name, num_slots, slot_bytes):
            log.append(("host", name, num_slots, slot_bytes))
            return _FakePool("host", name, num_slots, slot_bytes)

        @staticmethod
        def unlink_by_name(name):
            log.append(("host-unlink", name))

    class SharedMarvellPool:
        @staticmethod
        def create_or_attach(name, pci_bdf, num_slots, slot_bytes, bar_offset=0):
            log.append(("marvell", name, pci_bdf, num_slots, slot_bytes, bar_offset))
            return _FakePool("marvell", name, num_slots, slot_bytes, bar_offset, pci_bdf)

        @staticmethod
        def unlink_by_name(name):
            log.append(("marvell-unlink", name))

    classes = {"SharedHostPool": SharedHostPool, "SharedMarvellPool": SharedMarvellPool}

    monkeypatch.setattr(marvell_tier, "_pool_class", classes.__getitem__)
    return log


def test_host_factory_creates_host_pool(fake_c):
    pool = HostPoolFactory()("p", 8, PAGE)
    assert pool.kind == "host"
    assert fake_c == [("host", "p", 8, PAGE)]


def test_marvell_factory_passes_bdf_and_offset(fake_c):
    pool = MarvellPoolFactory(BDF, bar_offset=1 << 20, window_bytes=8 * PAGE)("p", 8, PAGE)
    assert pool.kind == "marvell"
    assert fake_c == [("marvell", "p", BDF, 8, PAGE, 1 << 20)]


def test_marvell_factory_refuses_pool_larger_than_window(fake_c):
    """If flex's slot rounding ever outgrew ours, the next window would be hit."""
    with pytest.raises(ValueError, match="overlap"):
        MarvellPoolFactory(BDF, bar_offset=0, window_bytes=8 * PAGE - 1)("p", 8, PAGE)


@pytest.mark.parametrize("offset", [-DEVICE_ALIGNMENT, 1])
def test_marvell_factory_rejects_bad_offset(offset):
    with pytest.raises(ValueError, match="bar_offset"):
        MarvellPoolFactory(BDF, bar_offset=offset)


def test_marvell_factory_rejects_empty_bdf():
    with pytest.raises(ValueError, match="BDF"):
        MarvellPoolFactory("", bar_offset=0)


def test_connector_shutdown_reaches_the_worker():
    """Upstream's chain OffloadingConnector -> connector worker -> worker.shutdown.

    Pinned so an upstream rename of the hook shows up here rather than as a
    leaked .ctl segment after every orderly exit.
    """
    import inspect

    from spyre_inference.v1.kv_offload.upstream_compat import OffloadingConnectorWorker

    assert "self.worker.shutdown()" in inspect.getsource(OffloadingConnectorWorker.shutdown)
