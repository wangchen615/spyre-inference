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

"""Lazy access to the torch-spyre persistent shared-pool API."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


def load_shared_runtime() -> ModuleType:
    """Load the M4 symbols only in processes that attach shared pools."""

    try:
        runtime = import_module("torch_spyre._C")
    except ImportError as error:
        raise RuntimeError(
            "torch-spyre with persistent SharedPool lifecycle support is required"
        ) from error

    missing = [name for name in ("SharedPool", "PoolResultCode") if not hasattr(runtime, name)]
    if missing:
        raise RuntimeError(
            "torch-spyre is missing persistent SharedPool lifecycle symbols: " + ", ".join(missing)
        )
    return runtime
