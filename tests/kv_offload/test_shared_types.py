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
    SharedLoadStoreSpec,
    SharedLocation,
    SharedPoolFamily,
    SharedTransfer,
    allocate_family_slots,
    shared_block_hash,
)


def test_family_slots_distribute_remainder_in_configuration_order():
    assert allocate_family_slots(8, ("alpha", "beta", "gamma")) == (
        SharedPoolFamily("alpha", 3),
        SharedPoolFamily("beta", 3),
        SharedPoolFamily("gamma", 2),
    )


def test_family_slots_distribute_even_capacity():
    assert allocate_family_slots(6, ("alpha", "beta", "gamma")) == (
        SharedPoolFamily("alpha", 2),
        SharedPoolFamily("beta", 2),
        SharedPoolFamily("gamma", 2),
    )


@pytest.mark.parametrize(
    ("slots", "names", "message"),
    [
        (8, (), "at least one"),
        (8, ("alpha", ""), "non-empty"),
        (8, ("alpha", "   "), "non-empty"),
        (8, ("alpha", "alpha"), "unique"),
        (2, ("alpha", "beta", "gamma"), "fewer slots"),
        (0, ("alpha",), "positive"),
    ],
)
def test_family_slot_allocation_rejects_invalid_configuration(slots, names, message):
    with pytest.raises(ValueError, match=message):
        allocate_family_slots(slots, names)


def test_pool_family_rejects_invalid_direct_construction():
    with pytest.raises(ValueError, match="non-empty"):
        SharedPoolFamily("", 1)
    with pytest.raises(ValueError, match="positive"):
        SharedPoolFamily("alpha", 0)


def test_shared_block_hash_covers_the_complete_offload_key():
    key0 = make_offload_key(bytes.fromhex("11" * 32), 0)
    key1 = make_offload_key(bytes.fromhex("11" * 32), 1)
    assert shared_block_hash(key0) == 0x632D1E16D4E6599B
    assert shared_block_hash(key1) != shared_block_hash(key0)


def test_shared_block_hash_is_deterministic():
    key = make_offload_key(bytes.fromhex("a5" * 32), 7)
    assert shared_block_hash(key) == shared_block_hash(key)


def _transfer(*, reservation=None):
    return SharedTransfer(
        key=make_offload_key(bytes.fromhex("22" * 32), 0),
        location=SharedLocation(anchor_pool_id=3, slot_id=5),
        reservation=reservation,
    )


def test_load_store_spec_accepts_uniform_load_and_store_modes():
    load = SharedLoadStoreSpec([_transfer(), _transfer()])
    store = SharedLoadStoreSpec([_transfer(reservation=object()), _transfer(reservation=object())])
    assert len(load.transfers) == 2
    assert len(store.transfers) == 2


def test_load_store_spec_rejects_mixed_transfer_modes():
    with pytest.raises(ValueError, match="mix load and store"):
        SharedLoadStoreSpec([_transfer(), _transfer(reservation=object())])


@pytest.mark.parametrize("bad_transfer", [(object(), object()), (1, 2, 3, 4)])
def test_load_store_spec_rejects_malformed_transfer_tuples(bad_transfer):
    with pytest.raises(TypeError, match="SharedTransfer"):
        SharedLoadStoreSpec([bad_transfer])


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
