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

"""A Torch Spyre worker class."""

import os
import sys
from contextlib import AbstractContextManager, nullcontext

import torch

# `import torch_spyre` is intentionally deferred to inside `init_device`.
# Importing it loads `libspyre_comms.so`, which captures
# `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `LOCAL_WORLD_SIZE` via
# `std::getenv` at dlopen time and caches them. Those env vars are only
# known per-worker, so they must be populated before the C library
# loads. `spyre_inference/__init__.py` sets
# `TORCH_DEVICE_BACKEND_AUTOLOAD=0` so torch's `[torch.backends]`
# autoload doesn't trigger the load at `import torch` time.

from vllm.logger import init_logger
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.worker_base import CompilationTimes

from spyre_inference.custom_ops import register_all
from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

logger = init_logger(__name__)


class TorchSpyreWorker(Worker):
    """A worker class that executes the model on IBM's Spyre device.

    Inherits from Worker (gpu_worker) directly — Spyre is not a CPU device
    and does not need any of the CPU-specific init (NUMA binding,
    torch.ops._C.init_cpu_memory_env, host-RAM profiling) that CPUWorker
    provides. The distributed init, random seed, and model runner
    construction are handled here.
    """

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # Worker.load_model wraps weight loading in a memory pool context
        # that calls get_mem_allocator_instance(). That only short-circuits
        # to nullcontext() when current_platform.is_cpu() returns True; our
        # platform reports OOT, so the upstream check falls through and raises.
        # Spyre weights live on-device, not in a host-side cumem allocator.
        return nullcontext()

    def init_device(self) -> None:
        # torch_spyre's eager op-dispatch re-enters torch.compile per aten op
        # (ops/eager.py: dispatch_to_torch_compile), so a large prefill recurses
        # deeper than Python 3.12's default recursion limit (1000) AND the
        # separate C-level Dynamo recursion limit. Raise both here -- this is the
        # worker process where the RecursionError is actually thrown, out of reach
        # of any setrecursionlimit in the driver.
        #
        # The required depth scales with prefill length: limit 4096 clears 8192
        # tokens but 16384 still overflowed (~2k+ eager frames). We set the limit
        # high enough for the full 131072 window; ttft_sweep.sh raises `ulimit -s`
        # (the worker main-thread stack) to back it, since a large recursion limit
        # without a matching stack segfaults instead of running (torch docs warn
        # of exactly this).
        _recursion_limit = int(os.environ.get("SPYRE_WORKER_RECURSION_LIMIT", "100000"))
        sys.setrecursionlimit(_recursion_limit)
        torch._dynamo.set_recursion_limit(_recursion_limit)

        # The CUSTOM (list-based) attention backend passes page_indices as a
        # Python list, and Dynamo recompiles the eager op wrapper on EACH unique
        # block-index value (see tests/test_spyre_attn.py:68-70, which raises this
        # same limit for exactly this reason). A prefill of L tokens touches
        # ceil(L/block_size) blocks, so past ~1024 tokens the default
        # accumulated_recompile_limit (256) is exhausted and Dynamo falls back
        # into the deep-recursing dispatch -> RecursionError. This is the actual
        # ceiling with prefix caching on (1024 passes, 2048 hit the 256 cap even
        # at recursion limit 50000). Raise it high enough for the full 131072
        # window (1024-block granularity -> a few thousand recompiles).
        _recompile_limit = int(os.environ.get("SPYRE_WORKER_RECOMPILE_LIMIT", "100000"))
        torch._dynamo.config.accumulated_recompile_limit = _recompile_limit
        # Also lift the per-code-object cache_size_limit so a single frame with
        # many block-index specializations isn't independently capped.
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, _recompile_limit
        )

        # Populate the env vars that `libspyre_comms.so` reads at dlopen
        # time. `setdefault` leaves torchrun-supplied values intact.
        # DP>1 is rejected in TorchSpyrePlatform.check_and_update_config,
        # so parallel_config.world_size is the global rank count and
        # LOCAL_WORLD_SIZE == WORLD_SIZE on a single node. Revisit once
        # multi-node TP is supported.
        world_size = self.vllm_config.parallel_config.world_size
        os.environ.setdefault("RANK", str(self.rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        os.environ.setdefault("LOCAL_RANK", str(self.local_rank))
        os.environ.setdefault("LOCAL_WORLD_SIZE", str(world_size))

        # Trigger torch_spyre's autoload manually now that the env vars
        # are set. Autoload registers the `spyre` device and the
        # `spyreccl` distributed backend, and imports
        # `torch_spyre._C` (which loads `libspyre_comms.so`).
        import torch_spyre

        torch_spyre._autoload()

        # Pin this worker to its assigned card before the spyreccl
        # backend is constructed in `init_process_group`.
        torch.spyre.set_device(self.local_rank)

        # Register all the custom ops here when a worker is created.
        # This has to happen before the model is loaded, so that all the
        # layers will be swapped out with the custom implementations for spyre.
        register_all()

        # Initialize the distributed environment.
        from vllm.platforms import current_platform

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner directly — no monkey-patching needed.
        self.model_runner = TorchSpyreModelRunner(
            self.vllm_config,
            torch.device("spyre"),
        )

    def determine_available_memory(self) -> int:
        # Spyre's KV cache lives on-device with a fixed budget set by
        # TorchSpyrePlatform.check_and_update_config (via VLLM_CPU_KVCACHE_SPACE).
        # num_gpu_blocks_override is also set, so this value is only used as
        # an upper bound sanity check by the engine.
        assert self.cache_config.kv_cache_memory_bytes is not None
        return self.cache_config.kv_cache_memory_bytes

    def compile_or_warm_up_model(self) -> CompilationTimes:
        # FIXME: Work around for https://github.com/torch-spyre/torch-spyre/issues/1420
        # Ensure registration of Spyre decompositions before FX Graph tracing
        import time

        import torch._inductor.decomposition
        from torch_spyre._inductor.decompositions import spyre_decompositions

        for op, impl in spyre_decompositions.items():
            if "addm" in op.name():
                logger.warning(
                    "FIXME: Adding %s decomposition to work-around torch-spyre crash", op.name()
                )
                torch._inductor.decomposition.decompositions[op] = impl

        warmup_start_time = time.perf_counter()
        self.model_runner.warming_up_model()
        self.compilation_config.compilation_time = time.perf_counter() - warmup_start_time
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def sleep(self, level: int = 1) -> None:
        pass

    def wake_up(self, tags: list[str] | None = None) -> None:
        pass
