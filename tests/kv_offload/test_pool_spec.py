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

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from spyre_inference.v1.attention.ops.layout import (
    head_major_kv_layout,
    slot_major_kv_layout,
)
from spyre_inference.v1.kv_offload.pool_spec import (
    build_model_pool_spec,
    planned_page_bytes,
)


class _FakeSpyreTensorLayout:
    def __init__(self, *args, **kwargs):
        if args:
            self.device_size = list(args[0])
            self.device_dtype = args[2]
        else:
            self.device_size = list(kwargs["device_size"])
            self.device_dtype = kwargs["device_dtype"]


def _install_fake_torch_spyre(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("torch_spyre._C")
    fake.SpyreTensorLayout = _FakeSpyreTensorLayout
    fake.get_device_dtype = lambda dtype: dtype
    fake.get_elem_in_stick = lambda dtype: 128 // torch.empty((), dtype=dtype).element_size()
    fake.ElementArrangement = SimpleNamespace(QFP8CH="qfp8ch", STANDARD="standard")
    monkeypatch.setitem(sys.modules, "torch_spyre._C", fake)


@pytest.mark.parametrize(
    ("layout_kind", "dtype", "head_size", "expected"),
    [
        ("token-major", torch.float16, 63, 4096),
        ("token-major", torch.float16, 64, 4096),
        ("token-major", torch.float16, 65, 8192),
        ("head-major", torch.float16, 65, 8192),
        ("head-major", torch.float8_e4m3fn, 127, 4096),
        ("head-major", torch.float8_e4m3fn, 128, 4096),
        ("head-major", torch.float8_e4m3fn, 129, 8192),
    ],
)
def test_planned_page_bytes_matches_layout_device_extent(
    monkeypatch, layout_kind, dtype, head_size, expected
):
    """Changing stick rounding in either planner or runtime must fail this test."""
    _install_fake_torch_spyre(monkeypatch)
    num_blocks, block_size, local_kv_heads = 3, 16, 2
    if layout_kind == "token-major":
        layout = slot_major_kv_layout(num_blocks * block_size, local_kv_heads, head_size, dtype)
    else:
        layout = head_major_kv_layout(num_blocks * local_kv_heads, block_size, head_size, dtype)
    runtime_page_bytes = (
        math.prod(layout.device_size) * torch.empty((), dtype=dtype).element_size() // num_blocks
    )

    assert (
        planned_page_bytes(
            dtype,
            block_size,
            local_kv_heads,
            head_size,
            layout_kind=layout_kind,
        )
        == runtime_page_bytes
        == expected
    )


def test_planned_page_bytes_matches_recorded_m3_geometry():
    assert planned_page_bytes(torch.float16, 128, 8, 128) == 262_144


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dtype": torch.float32}, "dtype"),
        ({"layout_kind": "sideways"}, "layout_kind"),
        ({"layout_version": 2}, "layout_version"),
        ({"block_size": 0}, "positive"),
    ],
)
def test_planned_page_bytes_rejects_unsupported_geometry(kwargs, match):
    values = {
        "dtype": torch.float16,
        "block_size": 128,
        "local_kv_heads": 8,
        "head_size": 128,
        "layout_kind": "token-major",
        "layout_version": 1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        planned_page_bytes(**values)


class _StaticLayout:
    def __init__(self, kind: str, version: int = 1):
        self.kv_layout_kind = kind
        self.kv_layout_version = version


def _vllm_config(
    layer_names: list[str],
    *,
    tp_size: int = 1,
    aliases: dict[str, str] | None = None,
    missing_layout: set[str] | None = None,
    pipeline_size: int = 1,
    data_parallel_size: int = 1,
    decode_context_size: int = 1,
):
    aliases = aliases or {}
    missing_layout = missing_layout or set()
    static = {}
    for index, name in enumerate(layer_names):
        static[name] = SimpleNamespace(
            impl=None
            if name in missing_layout
            else _StaticLayout("head-major" if index % 2 else "token-major"),
            kv_sharing_target_layer_name=aliases.get(name),
        )
    return SimpleNamespace(
        model_config=SimpleNamespace(model="model-a", revision="revision-1"),
        cache_config=SimpleNamespace(block_size=128),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pipeline_size,
            data_parallel_size=data_parallel_size,
            decode_context_parallel_size=decode_context_size,
            world_size=tp_size * pipeline_size * data_parallel_size,
        ),
        compilation_config=SimpleNamespace(static_forward_context=static),
    )


def _kv_cache_config(layer_names: list[str], *, groups: int = 1, spec=None):
    cache_spec = spec or FullAttentionSpec(
        block_size=128,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float16,
    )
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=cache_spec)
            for _ in range(groups)
        ],
    )


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_model_pool_spec_covers_every_tp_rank_and_physical_cache(tp_size):
    layers = ["layer.0", "layer.1"]
    result = build_model_pool_spec(
        _vllm_config(layers, tp_size=tp_size),
        _kv_cache_config(layers),
        "/deployment-a",
        1 << 30,
    )

    assert result["planningVersion"] == 1
    assert result["name"] == "/deployment-a"
    assert result["model"] == "model-a"
    assert result["modelRevision"] == "revision-1"
    assert result["kvDtype"] == "float16"
    assert result["blockSize"] == "128"
    assert result["tensorParallelSize"] == tp_size
    assert result["cpuBytesToUse"] == str(1 << 30)
    assert [(row["tpRank"], row["cacheIndex"]) for row in result["rankCaches"]] == [
        (rank, cache) for rank in range(tp_size) for cache in range(2)
    ]
    assert all(isinstance(row["physicalPageBytes"], str) for row in result["rankCaches"])


def test_shared_layer_alias_produces_one_physical_cache_row():
    layers = ["layer.0", "layer.1", "layer.2"]
    result = build_model_pool_spec(
        _vllm_config(layers, tp_size=2, aliases={"layer.1": "layer.0"}),
        _kv_cache_config(layers),
        "/shared-layers",
        1 << 30,
    )

    assert [(row["tpRank"], row["cacheIndex"]) for row in result["rankCaches"]] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]


@pytest.mark.parametrize(
    ("vllm_config", "kv_cache_config", "match"),
    [
        (
            _vllm_config(["layer.0"], missing_layout={"layer.0"}),
            _kv_cache_config(["layer.0"]),
            "layout",
        ),
        (
            _vllm_config(["layer.0"], pipeline_size=2),
            _kv_cache_config(["layer.0"]),
            "pipeline",
        ),
        (
            _vllm_config(["layer.0"], data_parallel_size=2),
            _kv_cache_config(["layer.0"]),
            "data parallel",
        ),
        (
            _vllm_config(["layer.0"], decode_context_size=2),
            _kv_cache_config(["layer.0"]),
            "context parallel",
        ),
        (
            _vllm_config(["layer.0"]),
            _kv_cache_config(["layer.0"], groups=2),
            "one KV cache group",
        ),
        (
            _vllm_config(["layer.0"]),
            _kv_cache_config(["layer.0"], spec=object()),
            "attention spec",
        ),
    ],
)
def test_model_pool_spec_rejects_unsupported_inputs(vllm_config, kv_cache_config, match):
    with pytest.raises(ValueError, match=match):
        build_model_pool_spec(
            vllm_config,
            kv_cache_config,
            "/unsupported",
            1 << 30,
        )
