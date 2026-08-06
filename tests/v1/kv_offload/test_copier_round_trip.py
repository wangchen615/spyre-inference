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

"""Round-trip test for SpyreKvDmaCopier against real Spyre device memory.

Exercises the one place the offloading stack moves bytes between host and device.
Equality is exact: ``copy_tensor`` is a DMA, not a compute kernel, so a KV block
must survive device->host->device unchanged bit for bit. Any tolerance here would
hide exactly the corruption this test exists to catch.
"""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier

PAGE_SHAPE = (2, 16, 64)


def _pattern() -> torch.Tensor:
    """Distinct fp16-exact values per element, so a reordered copy shows up."""
    numel = PAGE_SHAPE[0] * PAGE_SHAPE[1] * PAGE_SHAPE[2]
    return torch.arange(numel, dtype=torch.float16).reshape(PAGE_SHAPE)


@pytest.mark.spyre
def test_copier_round_trip_preserves_bytes():
    if not spyre_available():
        pytest.skip("Spyre device not available")

    device = torch.device("spyre")
    copier = SpyreKvDmaCopier()
    pattern = _pattern()

    host = torch.empty(PAGE_SHAPE, dtype=torch.float16, device="cpu")
    copier.copy_d2h(pattern.to(device), host)
    assert torch.equal(host, pattern)

    # Mutate on the host, push back to a fresh device page, and read it again.
    host.add_(2.0)
    dst = torch.zeros(PAGE_SHAPE, dtype=torch.float16, device=device)
    copier.copy_h2d(host, dst)

    check = torch.empty(PAGE_SHAPE, dtype=torch.float16, device="cpu")
    copier.copy_d2h(dst, check)
    assert torch.equal(check, pattern + 2.0)


@pytest.mark.spyre
def test_copier_keeps_pages_distinct():
    """Copying several pages must not cross-contaminate them."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    device = torch.device("spyre")
    copier = SpyreKvDmaCopier()

    device_pages = [
        torch.full(PAGE_SHAPE, float(i), dtype=torch.float16).to(device) for i in range(4)
    ]
    host_pages = [torch.empty(PAGE_SHAPE, dtype=torch.float16, device="cpu") for _ in range(4)]

    for src, dst in zip(device_pages, host_pages):
        copier.copy_d2h(src, dst)

    for i, page in enumerate(host_pages):
        assert torch.equal(page, torch.full(PAGE_SHAPE, float(i), dtype=torch.float16))
