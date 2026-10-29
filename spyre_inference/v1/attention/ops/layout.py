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

"""Device layout for the paged KV cache, and the stick geometry its index rows follow."""

import torch

# Elements per stick for int32 (128-byte stick / 4 bytes). Page-index rows are
# padded to this width so each row starts on a stick boundary; see
# SpyreAttentionMetadata.page_index_tables_cpu.
INT32_ELEMS_PER_STICK = 32


def kv_stick_geometry(dtype: torch.dtype, head_size: int) -> tuple[int, int]:
    """Return ``(sticks, elements_per_stick)`` for one KV-cache row.

    Spyre stores every innermost row in whole 128-byte sticks.  Keep this
    arithmetic independent of torch-spyre so the control-plane planner can
    compute the same physical extent without loading a device runtime.
    """
    element_sizes = {
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float8_e4m3fn: 1,
        # vLLM represents an fp8 KV-cache specification as byte storage.
        torch.uint8: 1,
    }
    if dtype not in element_sizes:
        raise ValueError(f"unsupported KV-cache dtype {dtype}")
    if head_size <= 0:
        raise ValueError(f"head_size must be positive, got {head_size}")
    elements_per_stick = 128 // element_sizes[dtype]
    sticks = (head_size + elements_per_stick - 1) // elements_per_stick
    return sticks, elements_per_stick


def slot_major_kv_layout(num_slots: int, num_kv_heads: int, head_size: int, dtype: torch.dtype):
    """Slot-axis-outermost layout. The default tiled layout spreads the slot index
    across two device dims, making the indirect store write to the wrong rows
    (torch-spyre#3705)."""
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    sticks, eps = kv_stick_geometry(dtype, head_size)
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def temporary_chunk_major_page_index_layout(num_chunks: int, entries_per_chunk: int):
    """TEMPORARY until torch-spyre#4603 is validated: one page ID per 128-byte stick.

    Logical ``[C, E, 1]`` maps to device ``[C, 1, E, 32]``: each int32 ID uses
    one of a stick's 32 positions. Only index metadata expands, not the KV cache.
    Remove this layout and its upload/kernel branches once the native 2-D index
    compiles, returns correct values and splits across cores in emitted code.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    return SpyreTensorLayout(
        [num_chunks, 1, entries_per_chunk, INT32_ELEMS_PER_STICK],
        [entries_per_chunk, -1, 1, -1],
        get_device_dtype(torch.int32),
    )


def head_major_kv_layout(num_pages: int, block_size: int, head_size: int, dtype: torch.dtype):
    """Head-major cache with the (page, kv_head) row the kernel gathers at device dim 0.

    ``num_pages`` counts folded rows, num_blocks * num_kv_heads. The indexed axis has to
    stay whole at device position 0, or the gather costs the whole tensor rather than one
    page (torch-spyre#3705).

    A float8 cache is tagged QFP8CH, the in-stick order ``spyre.qfp8ch`` writes and the
    fp8 -> fp16 read expects.
    """
    from torch_spyre._C import (
        ElementArrangement,
        SpyreTensorLayout,
        get_device_dtype,
    )

    sticks, eps = kv_stick_geometry(dtype, head_size)
    return SpyreTensorLayout(
        device_size=[num_pages, block_size, sticks, eps],
        stride_map=[block_size * head_size, head_size, eps, 1],
        device_dtype=get_device_dtype(dtype),
        element_arrangement=(
            ElementArrangement.QFP8CH
            if dtype == torch.float8_e4m3fn
            else ElementArrangement.STANDARD
        ),
    )
