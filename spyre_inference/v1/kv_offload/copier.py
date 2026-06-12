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

"""Single-purpose owner of every host<->Spyre KV byte transfer.

``SpyreKvDmaCopier`` is the one place the offloading stack touches the path
between Spyre device memory and host RAM. The ``OffloadingHandler`` pair in
``handlers.py`` calls ``copy_d2h`` / ``copy_h2d`` per block and knows nothing
about how the bytes actually move.

Uses ``torch_spyre._C.copy_tensor(src, dst, non_blocking=False)`` as the
underlying primitive. Direction is auto-detected from device types; with
``non_blocking=False``, the call is synchronous. This is the proven path
used throughout the torch-spyre ecosystem.

See ``docs/architecture/rfcs/upstream-connector-port.md`` (§6.1) for
the design rationale and verification trail.
"""

import torch
import torch_spyre._C as _C

from vllm.logger import init_logger

logger = init_logger(__name__)


class SpyreKvDmaCopier:
    """Owns every host<->Spyre KV byte transfer for the offloading handlers.

    Both methods are synchronous (Spyre has no public async/stream API today;
    ``TorchSpyreModelRunner._sync_device`` is a no-op for the same reason) and
    neither allocates — the handler owns allocation and passes in both tensors.

    The wrapper exists as a seam for test injection and future async swapping.
    """

    def copy_d2h(self, src_spyre: torch.Tensor, dst_host: torch.Tensor) -> None:
        """Copy one KV block from Spyre device memory into a host tensor.

        Args:
            src_spyre: source block tensor on the Spyre device.
            dst_host: destination block tensor on the host. Must already be the
                right shape/dtype to receive ``src_spyre``'s contents.
        """
        _C.copy_tensor(src_spyre, dst_host, non_blocking=False)

    def copy_h2d(self, src_host: torch.Tensor, dst_spyre: torch.Tensor) -> None:
        """Copy one KV block from a host tensor into Spyre device memory.

        Args:
            src_host: source block tensor on the host.
            dst_spyre: destination block tensor on the Spyre device. Must
                already be the right shape/dtype.
        """
        _C.copy_tensor(src_host, dst_spyre, non_blocking=False)
