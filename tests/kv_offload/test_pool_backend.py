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

"""Pool-backend selection and Marvell BAR2 partitioning, without a card.

The Marvell tier is one BAR carved into per-cache windows, and flex does not
detect two windows overlapping: they would silently share KV bytes. So the
window arithmetic and its validation are the only guard, and they are pinned
here exhaustively -- parsing, dispatch to the right pool type, the offsets
themselves, and every rejection (overlap, overflow of BAR2, overflow of
`marvell_bytes_to_use`, misalignment).

BAR2 is read from a fake sysfs tree, and the offloader's `torch_spyre._C`
lookups are replaced by a recorder, so nothing here needs the new torch-spyre
binding or a device.
"""

from __future__ import annotations

import types

import pytest

from spyre_inference.v1.kv_offload import spec as spec_mod
from spyre_inference.v1.kv_offload import worker as worker_mod
from spyre_inference.v1.kv_offload.connector import TOKEN_MAJOR, SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.spec import (
    DEVICE_ALIGNMENT,
    MarvellWindow,
    SpyreOffloadingSpec,
    marvell_window_bytes,
    plan_marvell_windows,
    read_bar_size,
    validate_marvell_windows,
)
from spyre_inference.v1.worker import spyre_kv_offload as off_mod
from spyre_inference.v1.worker.spyre_kv_offload import (
    HostPoolFactory,
    MarvellPoolFactory,
    SpyreKvPageOffloader,
)
from tests.kv_offload.test_spec import KV_BYTES_PER_BLOCK, _config

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
    monkeypatch.setattr(spec_mod, "PCI_SYSFS_ROOT", str(tmp_path))
    return _write


def _marvell(**extra):
    base = {
        "pool_backend": "marvell",
        "marvell_pci_bdf": BDF,
        "marvell_bytes_to_use": 32 * KV_BYTES_PER_BLOCK,
        "cpu_bytes_to_use": None,
    }
    base.update(extra)
    return base


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


# --- spec parsing ------------------------------------------------------------


def test_default_backend_is_host_and_unchanged():
    spec = SpyreOffloadingSpec(_config())
    assert spec.pool_backend == "host"
    assert spec.marvell is None
    assert spec.num_blocks == 32


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="pool_backend"):
        SpyreOffloadingSpec(_config(extra={"pool_backend": "nvme"}))


def test_marvell_sizes_from_marvell_bytes_to_use(sysfs):
    spec = SpyreOffloadingSpec(_config(extra=_marvell()))
    assert spec.pool_backend == "marvell"
    assert spec.num_blocks == 32
    assert spec.marvell.pci_bdf == BDF
    assert spec.marvell.bar_size == BAR2_SIZE
    assert spec.marvell.region_offset == 0
    assert spec.marvell.region_bytes == 32 * KV_BYTES_PER_BLOCK


def test_marvell_ignores_cpu_bytes_to_use(sysfs):
    """The Marvell tier is sized by its own key, never by the host one."""
    spec = SpyreOffloadingSpec(_config(extra=_marvell(cpu_bytes_to_use=1000 * KV_BYTES_PER_BLOCK)))
    assert spec.num_blocks == 32


def test_marvell_requires_its_size_key(sysfs):
    """cpu_bytes_to_use alone must not silently size a Marvell tier."""
    with pytest.raises(ValueError, match="marvell_bytes_to_use"):
        SpyreOffloadingSpec(
            _config(
                extra=_marvell(marvell_bytes_to_use=None, cpu_bytes_to_use=32 * KV_BYTES_PER_BLOCK)
            )
        )


def test_marvell_requires_bdf(sysfs):
    with pytest.raises(ValueError, match="marvell_pci_bdf"):
        SpyreOffloadingSpec(_config(extra=_marvell(marvell_pci_bdf=None)))


def test_marvell_too_small_raises(sysfs):
    with pytest.raises(ValueError, match="too small"):
        SpyreOffloadingSpec(_config(extra=_marvell(marvell_bytes_to_use=KV_BYTES_PER_BLOCK - 1)))


@pytest.mark.parametrize("offset", [1, 64, DEVICE_ALIGNMENT + 1, -DEVICE_ALIGNMENT])
def test_marvell_bar_offset_must_be_aligned(sysfs, offset):
    with pytest.raises(ValueError, match="marvell_bar_offset"):
        SpyreOffloadingSpec(_config(extra=_marvell(marvell_bar_offset=offset)))


def test_marvell_region_past_bar2_fails_at_startup(sysfs):
    """Checked in __init__, before any cache or pool exists."""
    with pytest.raises(ValueError, match="past BAR2"):
        SpyreOffloadingSpec(
            _config(
                extra=_marvell(
                    marvell_bar_offset=BAR2_SIZE - DEVICE_ALIGNMENT,
                    marvell_bytes_to_use=KV_BYTES_PER_BLOCK,
                )
            )
        )


def test_marvell_region_exactly_filling_bar2_is_accepted(sysfs):
    offset = BAR2_SIZE - 32 * KV_BYTES_PER_BLOCK
    spec = SpyreOffloadingSpec(_config(extra=_marvell(marvell_bar_offset=offset)))
    assert spec.marvell.region_offset == offset


def test_marvell_missing_card_fails_at_startup(sysfs):
    with pytest.raises(ValueError, match="cannot read"):
        SpyreOffloadingSpec(_config(extra=_marvell(marvell_pci_bdf="0000:ff:1f.7")))


def test_marvell_ranks_get_disjoint_regions(sysfs):
    """Two ranks on one card must not be handed the same BAR2 bytes."""
    specs = [
        SpyreOffloadingSpec(_config(extra=_marvell(marvell_bar_offset=4096), rank=r, world_size=2))
        for r in (0, 1)
    ]
    a, b = (s.marvell for s in specs)
    assert a.region_offset == 4096
    assert b.region_offset == a.region_offset + a.region_bytes
    assert a.region_bytes == b.region_bytes == 16 * KV_BYTES_PER_BLOCK
    assert all(s.num_blocks == 16 for s in specs)


def test_marvell_through_the_upstream_spec_factory(sysfs):
    """`vllm serve` builds the spec from the same extra_config dict."""
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    config = _config(
        extra=_marvell(
            spec_name="SpyreOffloadingSpec",
            spec_module_path="spyre_inference.v1.kv_offload.spec",
        )
    )
    spec = OffloadingSpecFactory.create_spec(config)
    assert isinstance(spec, SpyreOffloadingSpec)
    assert spec.pool_backend == "marvell"


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


# --- dispatch: spec -> factories -> pools -------------------------------------


def test_host_spec_yields_host_factories():
    factories = SpyreOffloadingSpec(_config()).pool_factories([PAGE] * 3)
    assert len(factories) == 3
    assert all(isinstance(f, HostPoolFactory) for f in factories)


def test_marvell_spec_yields_disjoint_marvell_factories(sysfs):
    spec = SpyreOffloadingSpec(
        _config(extra=_marvell(marvell_bar_offset=1 << 20, marvell_bytes_to_use=4 * GiB))
    )
    page = KV_BYTES_PER_BLOCK // 2 // 4  # four caches share one logical block
    factories = spec.pool_factories([page] * 4)
    assert all(isinstance(f, MarvellPoolFactory) for f in factories)
    assert {f.pci_bdf for f in factories} == {BDF}
    stride = marvell_window_bytes(page, spec.num_blocks)
    assert [f.bar_offset for f in factories] == [(1 << 20) + c * stride for c in range(4)]
    assert all(f.window_bytes == stride for f in factories)


def test_marvell_spec_rejects_padded_pages_that_overflow(sysfs):
    """Logical sizing fits exactly; physical padding must not spill past it."""
    spec = SpyreOffloadingSpec(_config(extra=_marvell()))
    logical_page = KV_BYTES_PER_BLOCK // 2
    spec.pool_factories([logical_page])  # exactly fits
    with pytest.raises(ValueError, match="marvell_bytes_to_use"):
        spec.pool_factories([logical_page + DEVICE_ALIGNMENT])


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
    """Replace the offloader's torch_spyre._C seams with a recorder.

    Only `spyre_kv_offload`'s own lookups are patched; `torch_spyre._C` itself
    is left alone, since swapping it in sys.modules would leak into any
    torch_spyre submodule first imported during the test.
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

    def copy_kv_page_raw(*a):
        log.append(("copy",) + a[1:])

    monkeypatch.setattr(off_mod, "_pool_class", classes.__getitem__)
    # page_signature is stubbed by _fake_cache, so get_composite_address is unused.
    monkeypatch.setattr(off_mod, "_import_runtime", lambda: (copy_kv_page_raw, None))
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


def _fake_cache(monkeypatch):
    """A 'cache' plus a stubbed page_signature, so no device is touched."""
    sig = off_mod.PageSignature("token-major", "torch.float16", 128, 8, 128, PAGE)
    monkeypatch.setattr(off_mod, "page_signature", lambda cache, kind: sig)
    return (object(), object())


def test_offloader_defaults_to_host_pool(fake_c, monkeypatch):
    off = SpyreKvPageOffloader(_fake_cache(monkeypatch), "p", 4, "token-major")
    assert off.backend == "host"
    assert fake_c == [("host", "p", 8, PAGE)]


def test_offloader_uses_given_factory_and_keeps_kv_pairing(fake_c, monkeypatch):
    cache = _fake_cache(monkeypatch)
    off = SpyreKvPageOffloader(
        cache, "p", 4, "token-major", pool_factory=MarvellPoolFactory(BDF, 0)
    )
    assert off.backend == "marvell"
    off.offload(block_id=3, slot=1)
    off.reload(slot=2, block_id=5)
    copies = [c for c in fake_c if c[0] == "copy"]
    pool = off.pool
    # (block, pool, slot, to_device, non_blocking) -- K at 2s, V at 2s+1.
    assert copies == [
        ("copy", 3, pool, 2, False, False),
        ("copy", 3, pool, 3, False, False),
        ("copy", 5, pool, 4, True, False),
        ("copy", 5, pool, 5, True, False),
    ]


def test_offloader_rejects_pool_with_wrong_geometry(fake_c, monkeypatch):
    def bad_factory(name, num_slots, slot_bytes):
        return _FakePool("host", name, num_slots // 2, slot_bytes)

    with pytest.raises(ValueError, match="slots"):
        SpyreKvPageOffloader(
            _fake_cache(monkeypatch), "p", 4, "token-major", pool_factory=bad_factory
        )


def test_offloader_release_is_idempotent_and_blocks_further_use(fake_c, monkeypatch):
    monkeypatch.setattr(off_mod, "_synchronize", lambda: None)
    off = SpyreKvPageOffloader(_fake_cache(monkeypatch), "p", 4, "token-major")
    off.release()
    off.release()
    with pytest.raises(RuntimeError, match="released"):
        off.offload(block_id=0, slot=0)
    # Releasing drops the attach; it never force-unlinks by name, which would
    # yank the segment from under another attached process.
    assert not any(c[0].endswith("unlink") for c in fake_c)


# --- worker wiring ------------------------------------------------------------


def _physical(n):
    return SpyrePhysicalCaches(
        caches=tuple((object(), object()) for _ in range(n)),
        layout_kinds=tuple(TOKEN_MAJOR for _ in range(n)),
        tensor_idx_to_cache={i: i // 2 for i in range(2 * n)},
        num_blocks=16,
    )


class _Recorder:
    made: list = []

    def __init__(self, *, cache, pool_name, num_slots, layout_kind, pool_factory):
        if getattr(pool_factory, "fail", False):
            raise RuntimeError("pool creation failed")
        self.pool_name, self.pool_factory = pool_name, pool_factory
        self.page_size_bytes = PAGE
        self.backend = getattr(pool_factory, "backend", "host")
        self.released = False
        _Recorder.made.append(self)

    def release(self):
        self.released = True


@pytest.fixture
def recorder(monkeypatch):
    _Recorder.made = []
    monkeypatch.setattr(worker_mod, "SpyreKvPageOffloader", _Recorder)
    return _Recorder.made


def test_worker_hands_each_cache_its_own_factory(recorder):
    factories = [MarvellPoolFactory(BDF, c * 4096) for c in range(3)]
    worker_mod.SpyreOffloadingWorker(
        physical=_physical(3), num_host_blocks=4, pool_prefix="x", pool_factories=factories
    )
    assert [o.pool_factory for o in recorder] == factories
    assert [o.pool_name for o in recorder] == ["x_c0", "x_c1", "x_c2"]


def test_worker_rejects_factory_count_mismatch(recorder):
    with pytest.raises(ValueError, match="pool factories"):
        worker_mod.SpyreOffloadingWorker(
            physical=_physical(3), num_host_blocks=4, pool_prefix="x", pool_factories=[None]
        )


def test_worker_shutdown_releases_every_pool(recorder):
    wrk = worker_mod.SpyreOffloadingWorker(
        physical=_physical(3), num_host_blocks=4, pool_prefix="x"
    )
    wrk.shutdown()
    wrk.shutdown()
    assert all(o.released for o in recorder)


def test_worker_releases_created_pools_when_a_later_one_fails(recorder):
    """A failed start must not leave the earlier caches' segments attached."""
    bad = types.SimpleNamespace(fail=True)
    with pytest.raises(RuntimeError, match="pool creation failed"):
        worker_mod.SpyreOffloadingWorker(
            physical=_physical(3),
            num_host_blocks=4,
            pool_prefix="x",
            pool_factories=[None, None, bad],
        )
    assert len(recorder) == 2
    assert all(o.released for o in recorder)


def test_connector_shutdown_reaches_the_worker():
    """Upstream's chain OffloadingConnector -> connector worker -> worker.shutdown.

    Pinned so an upstream rename of the hook shows up here rather than as a
    leaked .ctl segment after every orderly exit.
    """
    import inspect

    from spyre_inference.v1.kv_offload.upstream_compat import OffloadingConnectorWorker

    assert "self.worker.shutdown()" in inspect.getsource(OffloadingConnectorWorker.shutdown)
