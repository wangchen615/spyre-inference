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

**What is asserted, and what is deliberately not.** The invariant offloading depends
on is that evicting a KV block to host and reloading it yields the original block:
``device -> host -> device`` is the identity. The host staging tensor in between is
opaque -- nothing ever reads or computes on it -- so its element order is *not*
asserted. ``copy_tensor`` converts between host logical layout and the device's
stickified layout, so a d2h-only comparison against a logical pattern compares
representations rather than data and fails for reasons that do not affect offloading.
Equality on the round trip is exact: no tolerance, since a real transfer bug would
show up as changed values.
"""

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available

from spyre_inference.v1.kv_offload.copier import SpyreKvDmaCopier

# [num_kv_heads, block_size, head_size]. block_size is a multiple of 64 because the
# platform forces it (see TorchSpyrePlatform.check_and_update_config), and head_size
# must be a multiple of 64 for stick alignment -- so this is the geometry offloading
# actually sees, not an arbitrary small shape.
PAGE_SHAPE = (2, 64, 64)


def _pattern(shape=PAGE_SHAPE) -> torch.Tensor:
    """Distinct fp16-exact values per element, so a reordered copy shows up.

    Kept under 2048 (`% 1000`): every integer below 2048 is exactly representable in
    fp16, so a mismatch means the data changed, never that the dtype rounded.
    """
    numel = 1
    for dim in shape:
        numel *= dim
    return (torch.arange(numel, dtype=torch.float16) % 1000).reshape(shape)


@pytest.mark.spyre
@pytest.mark.parametrize("num_kv_heads", [2, 4, 8])
def test_offload_and_reload_is_the_identity(num_kv_heads):
    """A KV page must come back unchanged after a host round trip."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    shape = (num_kv_heads, PAGE_SHAPE[1], PAGE_SHAPE[2])
    device = torch.device("spyre")
    copier = SpyreKvDmaCopier()
    pattern = _pattern(shape)

    # Evict: device -> host staging.
    host = torch.empty(shape, dtype=torch.float16, device="cpu")
    copier.copy_d2h(pattern.to(device), host)

    # Reload: host staging -> a different device page.
    reloaded = torch.zeros(shape, dtype=torch.float16, device=device)
    copier.copy_h2d(host, reloaded)

    check = torch.empty(shape, dtype=torch.float16, device="cpu")
    copier.copy_d2h(reloaded, check)
    assert torch.equal(check, pattern), (
        f"{(check != pattern).sum().item()} of {pattern.numel()} elements changed "
        "across a device->host->device round trip"
    )


@pytest.mark.spyre
def test_round_trip_pages_stay_distinct():
    """Round-tripping several pages must not cross-contaminate them."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    device = torch.device("spyre")
    copier = SpyreKvDmaCopier()
    num_pages = 4

    originals = [torch.full(PAGE_SHAPE, float(i), dtype=torch.float16) for i in range(num_pages)]
    host_pages = [torch.empty(PAGE_SHAPE, dtype=torch.float16) for _ in range(num_pages)]

    for original, host in zip(originals, host_pages):
        copier.copy_d2h(original.to(device), host)

    for i, (host, original) in enumerate(zip(host_pages, originals)):
        reloaded = torch.zeros(PAGE_SHAPE, dtype=torch.float16, device=device)
        copier.copy_h2d(host, reloaded)
        check = torch.empty(PAGE_SHAPE, dtype=torch.float16)
        copier.copy_d2h(reloaded, check)
        assert torch.equal(check, original), f"page {i} came back with another page's data"
