#!/usr/bin/env bash
# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export LOCAL_RANK=0
export SPYRE_DEVICES="${SPYRE_DEVICE_A:-0}"
export PYTHONHASHSEED=0
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_PLUGINS=spyre_inference

model="${MODEL:-ibm-ai-platform/micro-g3.3-8b-instruct-1b}"
kv_transfer_config='{"kv_connector":"SpyreOffloadingConnector","kv_role":"kv_both","kv_connector_module_path":"spyre_inference.v1.kv_offload.connector","kv_connector_extra_config":{"spec_name":"SpyreSharedOffloadingSpec","shared_metadata_name":"spyre_manual_4096","shared_pool_families":["spyre_manual_4096.a","spyre_manual_4096.b"],"cpu_bytes_to_use":536870912}}'

command=(
    uv run --no-sync vllm serve "$model"
    --host 127.0.0.1
    --port 18100
    --enforce-eager
    --no-enable-prefix-caching
    --tensor-parallel-size 1
    --distributed-executor-backend uni
    --max-model-len 4352
    --max-num-seqs 1
    --max-num-batched-tokens 512
    --kv-transfer-config "$kv_transfer_config"
)

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'LOCAL_RANK=%q SPYRE_DEVICES=%q PYTHONHASHSEED=%q PYTHONPATH=%q VLLM_ENABLE_V1_MULTIPROCESSING=%q VLLM_PLUGINS=%q' \
        "$LOCAL_RANK" "$SPYRE_DEVICES" "$PYTHONHASHSEED" "$PYTHONPATH" \
        "$VLLM_ENABLE_V1_MULTIPROCESSING" "$VLLM_PLUGINS"
    printf ' %q' "${command[@]}"
    printf '\n'
    exit 0
fi

exec "${command[@]}"
