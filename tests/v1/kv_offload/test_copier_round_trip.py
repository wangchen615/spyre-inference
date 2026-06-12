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

"""Spyre-gated round-trip test for SpyreKvDmaCopier.

Allocates a Spyre device tensor with a known fp16 pattern, copies it to host,
mutates the host copy, copies it back, and asserts the device tensor matches.
Skips cleanly on CPU-only hosts (mirrors the gate in tests/test_spyre_attn.py).
"""

import pytest
import torch

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier


def _spyre_available() -> bool:
    try:
        torch.randn(1, device=torch.device("spyre"))
        return True
    except Exception:
        return False


@pytest.mark.spyre
def test_copier_round_trip_spyre():
    """Device->host->mutate->device round-trip on a real Spyre tensor."""
    if not _spyre_available():
        pytest.skip("Spyre device not available")

    device = torch.device("spyre")
    copier = SpyreKvDmaCopier(backend="torch_copy")

    # Known pattern on device.
    pattern = torch.arange(2 * 16 * 64, dtype=torch.float16).reshape(2, 16, 64)
    src_spyre = pattern.to(device)

    # device -> host
    host = torch.empty(2, 16, 64, dtype=torch.float16, device="cpu")
    copier.copy_d2h(src_spyre, host)
    assert torch.equal(host, pattern)

    # mutate host, then host -> device into a fresh device tensor
    host.add_(2.0)
    dst_spyre = torch.zeros(2, 16, 64, dtype=torch.float16, device=device)
    copier.copy_h2d(host, dst_spyre)

    # bring it back to verify
    check = torch.empty(2, 16, 64, dtype=torch.float16, device="cpu")
    copier.copy_d2h(dst_spyre, check)
    assert torch.equal(check, pattern + 2.0)
