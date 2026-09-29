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

"""Lazy access to the torch-spyre shared KV runtime surface."""

REQUIRED_SHARED_SYMBOLS = (
    "ChunkDescriptorEntry",
    "CompatibilityDescriptor",
    "CompatibleBlockKey",
    "ExistingClaim",
    "NoSpace",
    "Reservation",
    "SharedDataPoolConfig",
    "SharedMetadata",
    "SharedMetadataCapacity",
    "SharedMetadataConfig",
    "SharedPoolKind",
    "Unavailable",
    "copy_kv_page_raw",
    "get_composite_address",
)


def load_shared_runtime():
    try:
        from torch_spyre import _C as extension  # ty: ignore[unresolved-import]
    except ImportError as exc:
        missing = ", ".join(REQUIRED_SHARED_SYMBOLS)
        raise RuntimeError(
            f"torch-spyre M2 runtime surface is unavailable; missing: {missing}"
        ) from exc

    missing = [name for name in REQUIRED_SHARED_SYMBOLS if not hasattr(extension, name)]
    if missing:
        raise RuntimeError(
            "torch-spyre M2 runtime surface is incomplete; missing: " + ", ".join(missing)
        )
    return extension
