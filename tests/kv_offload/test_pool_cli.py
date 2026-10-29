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

import json
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from spyre_inference.v1.kv_offload import pool_cli
from spyre_inference.v1.kv_offload.pool_cli import main


def _resolved_inputs(arguments):
    assert arguments.model == "model-a"
    assert arguments.tensor_parallel_size == 2
    layer = SimpleNamespace(
        impl=SimpleNamespace(kv_layout_kind="token-major", kv_layout_version=1),
        kv_sharing_target_layer_name=None,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(model=arguments.model, revision=arguments.revision),
        cache_config=SimpleNamespace(block_size=arguments.block_size),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=arguments.tensor_parallel_size,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            world_size=arguments.tensor_parallel_size,
        ),
        compilation_config=SimpleNamespace(static_forward_context={"layer.0": layer}),
    )
    cache = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer.0"],
                kv_cache_spec=FullAttentionSpec(
                    block_size=arguments.block_size,
                    num_kv_heads=4,
                    head_size=128,
                    dtype=torch.float16,
                ),
            )
        ],
    )
    return config, cache


def _arguments(output=None):
    result = [
        "--model",
        "model-a",
        "--revision",
        "revision-1",
        "--tensor-parallel-size",
        "2",
        "--block-size",
        "128",
        "--pool-name",
        "/deployment-a",
        "--cpu-bytes-to-use",
        str(1 << 30),
    ]
    if output is not None:
        result.extend(("--output", str(output)))
    return result


def test_cli_writes_deterministic_delegator_json_to_stdout(capsys):
    assert main(_arguments(), resolve_inputs=_resolved_inputs) == 0
    first = capsys.readouterr().out
    assert main(_arguments(), resolve_inputs=_resolved_inputs) == 0
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert payload["planningVersion"] == 1
    assert payload["blockSize"] == "128"
    assert payload["cpuBytesToUse"] == str(1 << 30)
    assert [(row["tpRank"], row["cacheIndex"]) for row in payload["rankCaches"]] == [
        (0, 0),
        (1, 0),
    ]


def test_cli_writes_output_file_without_stdout(tmp_path, capsys):
    output = tmp_path / "model-pool.json"

    assert main(_arguments(output), resolve_inputs=_resolved_inputs) == 0

    assert capsys.readouterr().out == ""
    payload = json.loads(output.read_text())
    assert payload["name"] == "/deployment-a"


def test_default_resolver_uses_effective_block_size_and_marks_shared_layers(
    monkeypatch,
):
    text_config = SimpleNamespace(
        num_kv_shared_layers=2,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
    )
    model = SimpleNamespace(
        dtype=torch.float16,
        hf_text_config=text_config,
        model_arch_config=SimpleNamespace(head_size=128),
        get_num_layers=lambda _parallel: 4,
        get_num_kv_heads=lambda _parallel, _architecture: 4,
    )
    config = SimpleNamespace(
        model_config=model,
        cache_config=SimpleNamespace(block_size=64),
        parallel_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )

    class FakeEngineArgs:
        def __init__(self, **_kwargs):
            pass

        def create_engine_config(self, *, headless):
            assert headless
            return config

    monkeypatch.setattr("vllm.engine.arg_utils.EngineArgs", FakeEngineArgs)
    arguments = SimpleNamespace(
        model="model-a",
        revision=None,
        tensor_parallel_size=1,
        dtype="auto",
        kv_cache_dtype="auto",
        block_size=128,
        trust_remote_code=False,
    )

    _, kv_cache_config = pool_cli._default_resolve_inputs(arguments)

    group = kv_cache_config.kv_cache_groups[0]
    assert group.kv_cache_spec.block_size == 64
    assert (
        config.compilation_config.static_forward_context["layers.2"].kv_sharing_target_layer_name
        == "layers.0"
    )
    assert (
        config.compilation_config.static_forward_context["layers.3"].kv_sharing_target_layer_name
        == "layers.1"
    )


def test_cli_redirects_resolver_output_away_from_machine_json(capsys):
    def noisy_resolver(arguments):
        print("resolver diagnostic")
        return _resolved_inputs(arguments)

    assert main(_arguments(), resolve_inputs=noisy_resolver) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out)["name"] == "/deployment-a"
    assert "resolver diagnostic" in captured.err
