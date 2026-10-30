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

"""Generate delegator input from model configuration without starting workers."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

from spyre_inference import envs
from spyre_inference.v1.kv_offload.pool_spec import build_model_pool_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a persistent Spyre shared-pool model specification"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--pool-name", required=True)
    parser.add_argument("--cpu-bytes-to-use", required=True, type=int)
    parser.add_argument("--output", type=Path)
    return parser


def _cache_dtype(value: str, model_dtype: torch.dtype | str) -> torch.dtype:
    if value == "auto":
        if isinstance(model_dtype, torch.dtype):
            return model_dtype
        value = model_dtype
    if value in ("fp8", "fp8_e4m3"):
        return torch.uint8
    result = getattr(torch, value, None)
    if not isinstance(result, torch.dtype):
        raise ValueError(f"unsupported KV-cache dtype {value!r}")
    return result


def _arch_for_layer(model_config, layer_index: int):
    architecture = model_config.model_arch_config
    if isinstance(architecture, (list, tuple)):
        return architecture[layer_index]
    try:
        return architecture[layer_index]
    except (TypeError, KeyError):
        return architecture


def _shared_layer_targets(model_config, num_layers: int) -> dict[int, int]:
    text_config = model_config.hf_text_config
    shared_count = int(getattr(text_config, "num_kv_shared_layers", 0))
    if shared_count == 0:
        return {}
    first_shared = num_layers - shared_count
    layer_types = tuple(getattr(text_config, "layer_types", ()))
    if first_shared <= 0 or len(layer_types) != num_layers:
        raise ValueError("cannot resolve shared KV layers from the model configuration")
    targets = {}
    for layer_index in range(first_shared, num_layers):
        layer_type = layer_types[layer_index]
        candidates = [index for index in range(first_shared) if layer_types[index] == layer_type]
        if not candidates:
            raise ValueError(f"shared KV layer {layer_index} has no physical {layer_type!r} target")
        targets[layer_index] = candidates[-1]
    return targets


def _default_resolve_inputs(arguments):
    from vllm.engine.arg_utils import EngineArgs

    engine_args = EngineArgs(
        model=arguments.model,
        revision=arguments.revision,
        tensor_parallel_size=arguments.tensor_parallel_size,
        dtype=arguments.dtype,
        kv_cache_dtype=arguments.kv_cache_dtype,
        block_size=arguments.block_size,
        trust_remote_code=arguments.trust_remote_code,
    )
    config = engine_args.create_engine_config(headless=True)
    model = config.model_config
    parallel = config.parallel_config
    num_layers = model.get_num_layers(parallel)
    block_size = int(config.cache_config.block_size)
    dtype = _cache_dtype(arguments.kv_cache_dtype, model.dtype)
    layout_kind = {
        "token_major": "token-major",
        "head_major": "head-major",
    }.get(envs.SPYRE_ATTN_KV_LAYOUT)
    if layout_kind is None:
        raise ValueError(f"unsupported SPYRE_ATTN_KV_LAYOUT={envs.SPYRE_ATTN_KV_LAYOUT!r}")

    specs: dict[str, KVCacheSpec] = {}
    static_context = {}
    shared_targets = _shared_layer_targets(model, num_layers)
    for layer_index in range(num_layers):
        layer_name = f"layers.{layer_index}"
        architecture = _arch_for_layer(model, layer_index)
        head_size = int(architecture.head_size)
        specs[layer_name] = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=model.get_num_kv_heads(parallel, architecture),
            head_size=head_size,
            dtype=dtype,
        )
        static_context[layer_name] = SimpleNamespace(
            impl=SimpleNamespace(
                kv_layout_kind=layout_kind,
                kv_layout_version=1,
            ),
            kv_sharing_target_layer_name=(
                f"layers.{shared_targets[layer_index]}" if layer_index in shared_targets else None
            ),
        )

    layer_names = list(specs)
    group_spec = UniformTypeKVCacheSpecs(
        block_size=block_size,
        kv_cache_specs=specs,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=group_spec)],
    )
    config.compilation_config.static_forward_context = static_context
    return config, kv_cache_config


def resolve_model_pool_spec(
    *,
    model: str,
    revision: str | None,
    tensor_parallel_size: int,
    dtype: str,
    kv_cache_dtype: str,
    block_size: int,
    trust_remote_code: bool,
    pool_name: str,
    cpu_bytes_to_use: int,
    resolve_inputs: Callable | None = None,
) -> dict:
    arguments = SimpleNamespace(
        model=model,
        revision=revision,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        trust_remote_code=trust_remote_code,
    )
    resolver = resolve_inputs or _default_resolve_inputs
    with redirect_stdout(sys.stderr):
        vllm_config, kv_cache_config = resolver(arguments)
    return build_model_pool_spec(
        vllm_config,
        kv_cache_config,
        pool_name,
        cpu_bytes_to_use,
    )


def main(
    argv: list[str] | None = None,
    *,
    resolve_inputs: Callable | None = None,
) -> int:
    arguments = _parser().parse_args(argv)
    spec = resolve_model_pool_spec(
        model=arguments.model,
        revision=arguments.revision,
        tensor_parallel_size=arguments.tensor_parallel_size,
        dtype=arguments.dtype,
        kv_cache_dtype=arguments.kv_cache_dtype,
        block_size=arguments.block_size,
        trust_remote_code=arguments.trust_remote_code,
        pool_name=arguments.pool_name,
        cpu_bytes_to_use=arguments.cpu_bytes_to_use,
        resolve_inputs=resolve_inputs,
    )
    rendered = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
