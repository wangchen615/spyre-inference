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

"""Worker-side ownership of one persistent shared-pool attachment."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

from spyre_inference.v1.kv_offload.shared_identity import SharedPoolIdentity
from spyre_inference.v1.kv_offload.shared_runtime import load_shared_runtime


def _code_name(code: object) -> str:
    return str(getattr(code, "name", code))


def _attach_error(pool_name: str, result: object) -> str:
    code = _code_name(getattr(result, "code", "UNKNOWN"))
    detail = str(getattr(result, "detail", "") or "")
    message = f"cannot attach shared pool {pool_name!r}: {code}"
    if detail:
        message += f": {detail}"
    return message


class SharedPoolAttachment:
    """Own metadata, a fixed data child, and the root's shared lock lease."""

    def __init__(self, runtime: Any, metadata: Any, registered: Any, data_pool: Any) -> None:
        self._runtime = runtime
        self.metadata = metadata
        self.registered = registered
        self.data_pool = data_pool

    @classmethod
    def attach(
        cls,
        identity: SharedPoolIdentity,
        runtime_loader: Callable[[], Any] = load_shared_runtime,
    ) -> SharedPoolAttachment:
        runtime = runtime_loader()
        result = runtime.SharedPool.attach_pool(identity.pool_name, identity.generation)
        if result.code is not runtime.PoolResultCode.OK or result.metadata is None:
            message = _attach_error(identity.pool_name, result)
            if result.code is runtime.PoolResultCode.OK and result.metadata is None:
                message += ": no metadata returned"
            raise RuntimeError(message)

        metadata = result.metadata
        try:
            registered = metadata.find_pool(identity.data_pool_name)
            if registered is None:
                raise RuntimeError(
                    f"data pool {identity.data_pool_name!r} is not configured "
                    f"in shared pool {identity.pool_name!r}"
                )
            data_pool = metadata.resolve_pool(registered.pool_ref)
            if data_pool is None:
                raise RuntimeError(
                    f"data pool {identity.data_pool_name!r} cannot be resolved "
                    f"in shared pool {identity.pool_name!r}"
                )
            return cls(runtime, metadata, registered, data_pool)
        except BaseException:
            runtime.SharedPool.detach_pool(metadata)
            raise

    def close(self) -> None:
        """Release child handles, then detach the root exactly once."""

        metadata = getattr(self, "metadata", None)
        if metadata is None:
            return
        runtime = self._runtime
        self.data_pool = None
        self.registered = None
        self.metadata = None
        self._runtime = None
        runtime.SharedPool.detach_pool(metadata)

    def __del__(self) -> None:
        with suppress(BaseException):
            self.close()
