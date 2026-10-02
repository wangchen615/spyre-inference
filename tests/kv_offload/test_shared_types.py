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

"""Pure-Python contracts for cross-instance shared KV transfers."""

from __future__ import annotations

import builtins
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
from vllm.v1.kv_offload.base import make_offload_key

from spyre_inference.v1.kv_offload.shared_types import (
    SharedBlockTransfer,
    SharedComponentDescriptor,
    SharedLoadStoreSpec,
    SharedPageLocation,
    SharedPageTransfer,
    SharedPoolGeometry,
    compute_shared_pool_geometry,
    shared_page_hash,
)

OFFLOAD_KEY = make_offload_key(bytes.fromhex("22" * 32), 0)


def test_shared_page_hash_covers_key_and_component():
    key = make_offload_key(bytes.fromhex("11" * 32), 0)
    assert shared_page_hash(key, 0) == 0xFD8C6177AECACCE2
    assert shared_page_hash(key, 1) == 0x962C86167AD9E163
    assert shared_page_hash(key, 7) == 0xC340A847F6514AA5
    assert shared_page_hash(make_offload_key(bytes.fromhex("11" * 32), 1), 0) != shared_page_hash(
        key, 0
    )


@pytest.mark.parametrize("component_id", [-1, 0x1_0000_0000])
def test_shared_page_hash_rejects_component_id_outside_uint32(component_id):
    with pytest.raises(ValueError, match="four unsigned bytes"):
        shared_page_hash(OFFLOAD_KEY, component_id)


def test_manual_pool_geometry_is_one_512_mib_pool():
    geometry = compute_shared_pool_geometry(
        cpu_bytes_to_use=536_870_912,
        page_bytes=(262_144,) * 8,
        alignment=4096,
    )
    assert geometry == SharedPoolGeometry(
        component_count=8,
        slot_bytes=262_144,
        logical_block_capacity=256,
        slot_count=2048,
        actual_pool_bytes=536_870_912,
    )


def test_geometry_uses_aligned_largest_page():
    geometry = compute_shared_pool_geometry(16_384, (1000, 3000), 4096)
    assert geometry.slot_bytes == 4096
    assert geometry.logical_block_capacity == 2
    assert geometry.slot_count == 4
    assert geometry.actual_pool_bytes == 16_384


@pytest.mark.parametrize(
    ("budget", "page_bytes", "alignment", "message"),
    [
        (0, (1,), 1, "positive"),
        (-1, (1,), 1, "positive"),
        (16_384, (), 4096, "positive"),
        (16_384, (0,), 4096, "positive"),
        (16_384, (-1,), 4096, "positive"),
        (16_384, (1,), 0, "positive"),
        (4095, (4096, 1), 4096, "complete KV block"),
    ],
)
def test_geometry_rejects_invalid_inputs(budget, page_bytes, alignment, message):
    with pytest.raises(ValueError, match=message):
        compute_shared_pool_geometry(budget, page_bytes, alignment)


def test_geometry_rejects_native_size_t_overflow():
    with pytest.raises(OverflowError, match="size_t"):
        compute_shared_pool_geometry(sys.maxsize + 1, (1,), 1)


def test_component_descriptor_retains_physical_page_contract():
    descriptor = SharedComponentDescriptor(
        component_id=3,
        cache_index=1,
        role="v",
        layout_kind="head_major",
        layout_version=1,
        block_size=128,
        local_kv_heads=4,
        head_size=128,
        page_bytes=131_072,
    )
    assert descriptor.component_id == 3
    assert descriptor.page_bytes == 131_072


def _page(component_id: int, slot_id: int, *, reservation=None):
    return SharedPageTransfer(
        component_id,
        SharedPageLocation(pool_id=3, slot_id=slot_id),
        reservation,
    )


def test_load_spec_retains_fragmented_component_locations_in_order():
    pages = tuple(_page(i, slot) for i, slot in enumerate((7, 2, 11, 5)))
    spec = SharedLoadStoreSpec([SharedBlockTransfer(OFFLOAD_KEY, pages)])
    assert [page.component_id for page in spec.transfers[0].pages] == [0, 1, 2, 3]
    assert [page.location.slot_id for page in spec.transfers[0].pages] == [7, 2, 11, 5]
    assert all(page.reservation is None for page in spec.transfers[0].pages)


def test_store_spec_retains_each_page_reservation():
    reservations = tuple(object() for _ in range(4))
    pages = tuple(
        _page(i, slot, reservation=reservation)
        for i, (slot, reservation) in enumerate(zip((7, 2, 11, 5), reservations))
    )
    spec = SharedLoadStoreSpec([SharedBlockTransfer(OFFLOAD_KEY, pages)])
    assert tuple(page.reservation for page in spec.transfers[0].pages) == reservations


def test_block_transfer_rejects_empty_or_duplicate_component_pages():
    with pytest.raises(ValueError, match="at least one page"):
        SharedBlockTransfer(OFFLOAD_KEY, ())
    with pytest.raises(ValueError, match="component IDs must be unique"):
        SharedBlockTransfer(OFFLOAD_KEY, (_page(0, 7), _page(0, 2)))


def test_block_transfer_rejects_mixed_page_modes():
    with pytest.raises(ValueError, match="mix load and store"):
        SharedBlockTransfer(
            OFFLOAD_KEY,
            (_page(0, 7), _page(1, 2, reservation=object())),
        )


def test_load_store_spec_rejects_mixed_block_modes():
    load = SharedBlockTransfer(OFFLOAD_KEY, (_page(0, 7),))
    store = SharedBlockTransfer(
        make_offload_key(bytes.fromhex("33" * 32), 0),
        (_page(0, 2, reservation=object()),),
    )
    with pytest.raises(ValueError, match="mix load and store"):
        SharedLoadStoreSpec([load, store])


def test_load_store_spec_rejects_malformed_transfers():
    with pytest.raises(TypeError, match="SharedBlockTransfer"):
        SharedLoadStoreSpec([object()])


def test_shared_runtime_module_import_is_inert(monkeypatch):
    module_name = "spyre_inference.v1.kv_offload.shared_runtime"
    sys.modules.pop(module_name, None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("torch_spyre"):
            raise AssertionError("shared_runtime imported torch_spyre eagerly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module(module_name)


def test_load_shared_runtime_returns_complete_extension(monkeypatch):
    from spyre_inference.v1.kv_offload.shared_runtime import (
        REQUIRED_SHARED_SYMBOLS,
        load_shared_runtime,
    )

    extension = SimpleNamespace(**{name: object() for name in REQUIRED_SHARED_SYMBOLS})
    package = ModuleType("torch_spyre")
    package._C = extension
    monkeypatch.setitem(sys.modules, "torch_spyre", package)

    assert load_shared_runtime() is extension


def test_load_shared_runtime_lists_every_missing_symbol(monkeypatch):
    from spyre_inference.v1.kv_offload.shared_runtime import load_shared_runtime

    package = ModuleType("torch_spyre")
    package._C = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "torch_spyre", package)

    with pytest.raises(RuntimeError, match="torch-spyre M2") as exc_info:
        load_shared_runtime()

    assert "SharedMetadata" in str(exc_info.value)
    assert "copy_kv_page_raw" in str(exc_info.value)
