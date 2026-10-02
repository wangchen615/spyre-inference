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

"""Configuration, compatibility, and lazy loading for shared KV offload."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from spyre_inference.v1.kv_offload.connector import TOKEN_MAJOR, SpyrePhysicalCaches
from spyre_inference.v1.kv_offload.shared_types import PAGE_KEY_ALGORITHM
from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec
from spyre_inference.v1.worker.spyre_kv_offload import PageSignature

PAGE_BYTES = 262_144


def _shared_spec_module():
    from spyre_inference.v1.kv_offload import shared_spec

    return shared_spec


def _config(
    *,
    extra: dict | None = None,
    num_blocks: int = 8,
    world_size: int = 1,
    group_count: int = 1,
    layer_count: int = 1,
    dtype: str = "float16",
) -> OffloadingConfig:
    worker_kv_bytes_per_block = 2 * max(layer_count, 1) * PAGE_BYTES
    extra_config = {
        "cpu_bytes_to_use": num_blocks * worker_kv_bytes_per_block * world_size,
        "shared_metadata_name": "run-42",
        "pool_name": "run-42.data",
    }
    if extra is not None:
        extra_config.update(extra)
    groups = tuple(
        OffloadingGroupConfig(
            tokens_per_block=128,
            layer_names=tuple(f"layer.{group}.{index}" for index in range(layer_count)),
        )
        for group in range(group_count)
    )
    return OffloadingConfig(
        groups=groups,
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config=extra_config,
        engine_id="eng0",
        model=OffloadingModelConfig(name="micro-g3.3-8b", dtype=dtype),
        cache=OffloadingCacheConfig(tokens_per_hash=128, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )


def _config_without_pool_name() -> OffloadingConfig:
    config = _config()
    config.extra_config.pop("pool_name")
    return config


def _vllm_config(
    *,
    backend: str = "uni",
    model: str = "model-a",
    revision: str | None = "rev-a",
    hash_algorithm: str = "sha256",
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(distributed_executor_backend=backend),
        model_config=SimpleNamespace(model=model, revision=revision),
        cache_config=SimpleNamespace(prefix_caching_hash_algo=hash_algorithm),
    )


def _physical(cache_count: int = 1, layout_kind: str = TOKEN_MAJOR) -> SpyrePhysicalCaches:
    caches = tuple((object(), object()) for _ in range(cache_count))
    return SpyrePhysicalCaches(
        caches=cast(tuple[tuple[torch.Tensor, torch.Tensor], ...], caches),
        layout_kinds=(layout_kind,) * cache_count,
        tensor_idx_to_cache={
            tensor_index: tensor_index // 2 for tensor_index in range(2 * cache_count)
        },
        num_blocks=4,
    )


def _signature(
    *,
    layout_kind: str = TOKEN_MAJOR,
    device_dtype: str = "torch.float16",
    block_size: int = 128,
    local_kv_heads: int = 8,
    head_size: int = 128,
    page_bytes: int = PAGE_BYTES,
    layout_version: int = 1,
) -> PageSignature:
    return PageSignature(
        layout_kind=layout_kind,
        device_dtype=device_dtype,
        block_size=block_size,
        local_kv_heads=local_kv_heads,
        head_size=head_size,
        page_bytes=page_bytes,
        layout_version=layout_version,
    )


def test_explicit_pool_name_wins_over_environment(monkeypatch):
    monkeypatch.setenv("SPYRE_KV_POOL_NAME", "from-env")
    spec = _shared_spec_module().SpyreSharedOffloadingSpec(_config(extra={"pool_name": "explicit"}))
    assert spec.pool_name == "explicit"


def test_environment_pool_name_is_the_fallback(monkeypatch):
    monkeypatch.setenv("SPYRE_KV_POOL_NAME", "from-env")
    spec = _shared_spec_module().SpyreSharedOffloadingSpec(_config_without_pool_name())
    assert spec.pool_name == "from-env"


def test_generated_pool_name_is_private_without_shared_configuration(monkeypatch):
    monkeypatch.delenv("SPYRE_KV_POOL_NAME", raising=False)
    spec = _shared_spec_module().SpyreSharedOffloadingSpec(_config_without_pool_name())
    assert spec.pool_name == "spyre_kv_eng0_r0.shared"


@pytest.mark.parametrize("pool_name", ["", " ", "/", "nested/name", "/nested/name"])
def test_explicit_invalid_pool_name_does_not_fall_through(monkeypatch, pool_name):
    monkeypatch.setenv("SPYRE_KV_POOL_NAME", "valid-env")
    with pytest.raises(ValueError, match="pool_name"):
        _shared_spec_module().SpyreSharedOffloadingSpec(_config(extra={"pool_name": pool_name}))


def test_invalid_environment_pool_name_does_not_generate_fallback(monkeypatch):
    monkeypatch.setenv("SPYRE_KV_POOL_NAME", " ")
    with pytest.raises(ValueError, match="SPYRE_KV_POOL_NAME"):
        _shared_spec_module().SpyreSharedOffloadingSpec(_config_without_pool_name())


def test_removed_family_configuration_has_migration_error():
    with pytest.raises(ValueError, match="shared_pool_families.*pool_name"):
        _shared_spec_module().SpyreSharedOffloadingSpec(
            _config(extra={"shared_pool_families": ["a", "b"]})
        )


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"shared_metadata_name": " "}, "shared_metadata_name"),
        ({"shared_metadata_name": "nested/name"}, "shared_metadata_name"),
    ],
)
def test_rejects_invalid_shared_metadata_configuration(extra, match):
    with pytest.raises(ValueError, match=match):
        _shared_spec_module().SpyreSharedOffloadingSpec(_config(extra=extra))


@pytest.mark.parametrize(
    ("config", "match"),
    [
        (_config(world_size=2), "world_size"),
        (_config(group_count=2), "one KV cache group"),
        (_config(layer_count=0), "at least one layer"),
        (_config(dtype="bfloat16"), "float16"),
    ],
)
def test_rejects_unsupported_shared_execution_shapes(config, match):
    with pytest.raises(ValueError, match=match):
        _shared_spec_module().SpyreSharedOffloadingSpec(config)


def test_rejects_non_uni_executor_after_vllm_resolution():
    spec = _shared_spec_module().SpyreSharedOffloadingSpec(_config())
    with pytest.raises(ValueError, match="uni"):
        spec.bind_vllm_config(_vllm_config(backend="mp"))


def test_component_manifest_is_stable_cache_then_k_v(monkeypatch):
    module = _shared_spec_module()
    monkeypatch.setattr(
        module,
        "page_signature",
        lambda _cache, layout: _signature(layout_kind=layout),
    )
    manifest = module.component_manifest(_physical(cache_count=2))
    assert [(item.component_id, item.cache_index, item.role) for item in manifest] == [
        (0, 0, "k"),
        (1, 0, "v"),
        (2, 1, "k"),
        (3, 1, "v"),
    ]


def test_compatibility_payload_covers_every_interpretation_field(monkeypatch):
    module = _shared_spec_module()
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(module, "page_signature", lambda *_: _signature())
    spec = module.SpyreSharedOffloadingSpec(_config())
    spec.bind_vllm_config(_vllm_config())

    assert spec.compatibility_payload(module.component_manifest(_physical())) == {
        "format": 2,
        "page_key_algorithm": PAGE_KEY_ALGORITHM,
        "model": "model-a",
        "revision": "rev-a",
        "hash_algorithm": "sha256",
        "hash_seed": "0",
        "tokens_per_hash": 128,
        "tokens_per_block": (128,),
        "dtype": "float16",
        "tp_size": 1,
        "components": (
            {
                "component_id": 0,
                "cache_index": 0,
                "role": "k",
                "layout_kind": "token-major",
                "layout_version": 1,
                "block_size": 128,
                "local_kv_heads": 8,
                "head_size": 128,
                "page_bytes": PAGE_BYTES,
            },
            {
                "component_id": 1,
                "cache_index": 0,
                "role": "v",
                "layout_kind": "token-major",
                "layout_version": 1,
                "block_size": 128,
                "local_kv_heads": 8,
                "head_size": 128,
                "page_bytes": PAGE_BYTES,
            },
        ),
    }


def test_every_compatibility_field_changes_the_digest():
    module = _shared_spec_module()
    base: dict[str, Any] = {
        "format": 2,
        "page_key_algorithm": PAGE_KEY_ALGORITHM,
        "model": "model-a",
        "revision": "rev-a",
        "hash_algorithm": "sha256",
        "hash_seed": "0",
        "tokens_per_hash": 128,
        "tokens_per_block": (128,),
        "dtype": "float16",
        "tp_size": 1,
        "components": (
            {
                "component_id": 0,
                "cache_index": 0,
                "role": "k",
                "layout_kind": "token-major",
                "layout_version": 1,
                "block_size": 128,
                "local_kv_heads": 8,
                "head_size": 128,
                "page_bytes": PAGE_BYTES,
            },
        ),
    }
    expected = module.compatibility_digest(base)
    mutations = {
        "format": 3,
        "page_key_algorithm": "another",
        "model": "model-b",
        "revision": "rev-b",
        "hash_algorithm": "sha256_cbor_64bit",
        "hash_seed": "1",
        "tokens_per_hash": 64,
        "tokens_per_block": (64,),
        "dtype": "bfloat16",
        "tp_size": 2,
    }
    for field, value in mutations.items():
        changed = copy.deepcopy(base)
        changed[field] = value
        assert module.compatibility_digest(changed) != expected, field

    for field, value in {
        "component_id": 1,
        "cache_index": 1,
        "role": "v",
        "layout_kind": "head-major",
        "layout_version": 2,
        "block_size": 64,
        "local_kv_heads": 4,
        "head_size": 64,
        "page_bytes": 131_072,
    }.items():
        changed = copy.deepcopy(base)
        changed["components"][0][field] = value
        assert module.compatibility_digest(changed) != expected, field


def test_compatibility_digest_is_stable_across_processes():
    module = _shared_spec_module()
    payload = {
        "format": 2,
        "page_key_algorithm": PAGE_KEY_ALGORITHM,
        "model": "model-a",
        "revision": "rev-a",
        "hash_algorithm": "sha256",
        "hash_seed": "0",
        "tokens_per_hash": 128,
        "tokens_per_block": (128,),
        "dtype": "float16",
        "tp_size": 1,
        "components": (),
    }
    expected = module.compatibility_digest(payload).hex()
    assert expected == "d199b281b070184311bd69db73e5995fe6bd8a93ded59b899437277e9c3df567"
    code = (
        "from spyre_inference.v1.kv_offload.shared_spec import "
        "compatibility_digest; "
        f"print(compatibility_digest({payload!r}).hex())"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )
    assert completed.stdout.strip().splitlines()[-1] == expected


def test_compatibility_requires_an_explicit_python_hash_seed(monkeypatch):
    module = _shared_spec_module()
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setattr(module, "page_signature", lambda *_: _signature())
    spec = module.SpyreSharedOffloadingSpec(_config())
    spec.bind_vllm_config(_vllm_config())
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        spec.compatibility_payload(module.component_manifest(_physical()))


def test_shared_factories_receive_one_pool_manifest_and_geometry(monkeypatch):
    module = _shared_spec_module()
    from spyre_inference.v1.kv_offload import shared_manager, shared_worker

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(module, "page_signature", lambda *_: _signature())
    manager_kwargs: dict[str, Any] = {}
    worker_kwargs: dict[str, Any] = {}

    class RecordingManager:
        def __init__(self, **kwargs):
            manager_kwargs.update(kwargs)

    class RecordingWorker:
        _bytes_per_block = 8 * PAGE_BYTES

        def __init__(self, **kwargs):
            worker_kwargs.update(kwargs)

    monkeypatch.setattr(shared_manager, "SpyreSharedOffloadingManager", RecordingManager)
    monkeypatch.setattr(shared_worker, "SpyreSharedOffloadingWorker", RecordingWorker)
    spec = module.SpyreSharedOffloadingSpec(_config(num_blocks=256, layer_count=4))
    spec.bind_vllm_config(_vllm_config())
    physical = _physical(cache_count=4)
    spec.bind_physical_caches(physical)

    assert spec.component_count == 8
    assert spec.max_pool_slots == 2048
    assert spec.get_manager().__class__ is RecordingManager
    spec._create_worker(physical)

    manifest = module.component_manifest(physical)
    assert manager_kwargs["metadata_name"] == "run-42"
    assert manager_kwargs["pool_name"] == "run-42.data"
    assert manager_kwargs["component_count"] == 8
    assert manager_kwargs["max_pool_slots"] == 2048
    assert worker_kwargs["physical"] is physical
    assert worker_kwargs["pool_name"] == "run-42.data"
    assert worker_kwargs["geometry"].slot_count == 2048
    assert worker_kwargs["manifest"] == manifest
    assert worker_kwargs["max_pool_slots"] == 2048
    assert len(worker_kwargs["compatibility_digest"]) == 32


def test_worker_factory_rejects_physical_component_count_mismatch(monkeypatch):
    module = _shared_spec_module()
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(module, "page_signature", lambda *_: _signature())
    spec = module.SpyreSharedOffloadingSpec(_config(layer_count=2))
    spec.bind_vllm_config(_vllm_config())
    with pytest.raises(ValueError, match="component count"):
        spec._create_worker(_physical(cache_count=1))


def test_shared_spec_factory_registration_is_lazy():
    code = """
import sys
import spyre_inference
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

module = "spyre_inference.v1.kv_offload.shared_spec"
assert module not in sys.modules
cls = OffloadingSpecFactory.get_spec_cls({"spec_name": "SpyreSharedOffloadingSpec"})
assert cls.__name__ == "SpyreSharedOffloadingSpec"
assert module in sys.modules
print("lazy-ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )
    assert completed.stdout.strip().splitlines()[-1] == "lazy-ok"


def test_missing_m2_runtime_does_not_break_plugin_or_m1():
    code = """
import sys
import types

torch_spyre = types.ModuleType("torch_spyre")
torch_spyre.__path__ = []
torch_spyre._C = types.ModuleType("torch_spyre._C")
sys.modules["torch_spyre"] = torch_spyre
sys.modules["torch_spyre._C"] = torch_spyre._C

import spyre_inference
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)

m1 = OffloadingSpecFactory.get_spec_cls({
    "spec_name": "SpyreOffloadingSpec",
    "spec_module_path": "spyre_inference.v1.kv_offload.spec",
})
assert m1.__name__ == "SpyreOffloadingSpec"
m2 = OffloadingSpecFactory.get_spec_cls({"spec_name": "SpyreSharedOffloadingSpec"})
config = OffloadingConfig(
    groups=(OffloadingGroupConfig(tokens_per_block=128, layer_names=("layer.0",)),),
    worker_kv_bytes_per_block=524288,
    enable_kv_cache_events=False,
    extra_config={
        "cpu_bytes_to_use": 4194304,
        "shared_metadata_name": "run-42",
        "pool_name": "run-42.data",
    },
    engine_id="eng0",
    model=OffloadingModelConfig(name="model-a", dtype="float16"),
    cache=OffloadingCacheConfig(tokens_per_hash=128, blocks_per_chunk=1),
    parallel=OffloadingParallelConfig(
        rank=0, world_size=1, tp_size=1, pp_size=1, pcp_size=1, dcp_size=1,
        data_parallel_index=0, data_parallel_size=1,
        data_parallel_rank_local=None, is_parallelism_agnostic=False,
    ),
)
try:
    m2(config).get_manager().directory
except RuntimeError as error:
    message = str(error)
    assert "torch-spyre M2" in message
    assert "SharedMetadata" in message
else:
    raise AssertionError("missing M2 runtime unexpectedly succeeded")
print("missing-runtime-ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONHASHSEED": "0",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
        },
    )
    assert completed.stdout.strip().splitlines()[-1] == "missing-runtime-ok"


def test_m1_factory_still_builds_the_upstream_manager():
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    config = _config(
        extra={
            "spec_name": "SpyreOffloadingSpec",
            "spec_module_path": "spyre_inference.v1.kv_offload.spec",
        }
    )
    spec = OffloadingSpecFactory.create_spec(config)
    assert type(spec) is SpyreOffloadingSpec
    assert type(spec.get_manager()) is CPUOffloadingManager
