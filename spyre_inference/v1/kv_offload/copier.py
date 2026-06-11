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

Backends (selected once at construction from ``SPYRE_KV_DMA_BACKEND``):

- ``torch_copy``      whole-tensor ``.copy_()`` / ``.to("cpu")`` via torch-spyre's
                      ``copy_tensor`` path. This mirrors the proven device<->host
                      idiom used elsewhere in the plugin (see
                      ``spyre_inference/custom_ops/utils.py:convert``). It is the
                      only backend that runs on a CPU-only host and on a Spyre dev
                      image that does not have libsenlib, so it is the M1 default.
- ``senlib_dma``      libsenlib ``DmaiQPush`` / ``DmaoQPush`` over ``flit_offset``
                      addresses, matching the prior PD-disagg prototype. Requires
                      libsenlib (installable from flex-runtime). Not yet wired here;
                      selecting it raises ``NotImplementedError`` with guidance.
- ``spyre_from_blob`` ``torch_spyre._C.{get_dma_address, spyre_from_blob}``.
                      Forward-looking: those accessors are not in the pinned
                      torch-spyre, so selecting it raises ``NotImplementedError``.

See ``docs/architecture/rfcs/upstream-connector-port.md`` (§6.1) and the companion
``upstream-connector-port-TRACKING.md`` for the verification trail behind the
backend availability story.
"""

import importlib.util

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Backend identifiers accepted by SPYRE_KV_DMA_BACKEND (besides "auto").
TORCH_COPY = "torch_copy"
SENLIB_DMA = "senlib_dma"
SPYRE_FROM_BLOB = "spyre_from_blob"
AUTO = "auto"

_VALID_BACKENDS = (AUTO, TORCH_COPY, SENLIB_DMA, SPYRE_FROM_BLOB)


def _dmpa_accessors_available() -> bool:
    """True if torch_spyre._C exposes the DMPA accessors the from_blob path needs.

    The pinned torch-spyre does not export these; the accessors live on the
    ``flim/pd-disagg`` branch and have not landed upstream (RFC §10 Q1).
    """
    try:
        import torch_spyre._C as _C  # noqa: PLC0415
    except Exception:
        return False
    return hasattr(_C, "get_dma_address") and hasattr(_C, "spyre_from_blob")


def _senlib_available() -> bool:
    """True if a libsenlib Python binding is importable in this environment."""
    return (
        importlib.util.find_spec("senlib") is not None
        or importlib.util.find_spec("pysenlib") is not None
    )


def _resolve_backend(requested: str) -> str:
    """Resolve "auto" to a concrete backend; validate an explicit choice."""
    if requested not in _VALID_BACKENDS:
        raise ValueError(
            f"SPYRE_KV_DMA_BACKEND={requested!r} is not one of {_VALID_BACKENDS}"
        )
    if requested != AUTO:
        return requested

    # auto-detect, preferring the most direct path that is actually present.
    if _dmpa_accessors_available():
        return SPYRE_FROM_BLOB
    if _senlib_available():
        return SENLIB_DMA
    return TORCH_COPY


class SpyreKvDmaCopier:
    """Owns every host<->Spyre KV byte transfer for the offloading handlers.

    Both methods are synchronous (Spyre has no public async/stream API today;
    ``TorchSpyreModelRunner._sync_device`` is a no-op for the same reason) and
    neither allocates — the handler owns allocation and passes in both tensors.

    A single instance is shared across both directions; the backend is read from
    ``SPYRE_KV_DMA_BACKEND`` once at construction and locked in.
    """

    def __init__(self, backend: str | None = None):
        if backend is None:
            from spyre_inference import envs  # noqa: PLC0415

            backend = envs.SPYRE_KV_DMA_BACKEND

        self.backend = _resolve_backend(backend)
        logger.info("SpyreKvDmaCopier using backend %r", self.backend)

        if self.backend == SENLIB_DMA:
            raise NotImplementedError(
                "SPYRE_KV_DMA_BACKEND='senlib_dma' is selected but the libsenlib "
                "DMA backend is not wired up yet. Install senlib from flex-runtime "
                "and implement the DmaiQPush/DmaoQPush path, or use 'torch_copy'."
            )
        if self.backend == SPYRE_FROM_BLOB:
            raise NotImplementedError(
                "SPYRE_KV_DMA_BACKEND='spyre_from_blob' is selected but "
                "torch_spyre._C does not expose the DMPA accessors "
                "(get_dma_address / spyre_from_blob) in this build. Use "
                "'torch_copy' until those accessors land upstream."
            )

    def copy_d2h(self, src_spyre: torch.Tensor, dst_host: torch.Tensor) -> None:
        """Copy one KV block from Spyre device memory into a host tensor.

        Args:
            src_spyre: source block tensor (on the Spyre device, or CPU in tests).
            dst_host: destination block tensor on the host. Must already be the
                right shape/dtype to receive ``src_spyre``'s contents.
        """
        # torch_copy: pull to host first, then write into the caller's buffer.
        # Going through .to("cpu") keeps us on torch-spyre's supported
        # copy_tensor path and avoids any on-device slicing (which corrupts
        # memory on Spyre).
        dst_host.copy_(src_spyre.to(dst_host.device))

    def copy_h2d(self, src_host: torch.Tensor, dst_spyre: torch.Tensor) -> None:
        """Copy one KV block from a host tensor into Spyre device memory.

        Args:
            src_host: source block tensor on the host.
            dst_spyre: destination block tensor (on the Spyre device, or CPU in
                tests). Must already be the right shape/dtype.
        """
        dst_spyre.copy_(src_host.to(dst_spyre.device))
