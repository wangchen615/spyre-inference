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

"""Strict configuration for a delegator-created shared pool generation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

_GENERATION_PATTERN = re.compile(r"[0-9a-fA-F]{32}\Z")


@dataclass(frozen=True)
class SharedPoolIdentity:
    """The exact persistent composite generation and fixed data child to attach."""

    pool_name: str
    generation: bytes
    data_pool_name: str


def _required_setting(
    extra_config: Mapping[str, object],
    environ: Mapping[str, str],
    key: str,
    environment_key: str,
) -> object:
    if key in extra_config:
        return extra_config[key]
    if environment_key in environ:
        return environ[environment_key]
    raise ValueError(f"{key} must be specified in kv_connector_extra_config or {environment_key}")


def _optional_setting(
    extra_config: Mapping[str, object],
    environ: Mapping[str, str],
    key: str,
    environment_key: str,
    default: str,
) -> object:
    if key in extra_config:
        return extra_config[key]
    return environ.get(environment_key, default)


def _logical_name(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    component = value[1:] if value.startswith("/") else value
    if not component or "/" in component or "\0" in component:
        raise ValueError(f"{field} must contain one non-empty path component")
    return value


def _generation_bytes(value: object) -> bytes:
    if not isinstance(value, str):
        raise ValueError("pool_generation must be a string containing 32 hexadecimal characters")
    if _GENERATION_PATTERN.fullmatch(value) is None:
        raise ValueError("pool_generation must contain exactly 32 hexadecimal characters")
    return bytes.fromhex(value)


def load_shared_pool_identity(
    extra_config: Mapping[str, object], environ: Mapping[str, str] = os.environ
) -> SharedPoolIdentity:
    """Load an exact pool identity without silently replacing invalid explicit values."""

    if "shared_metadata_name" in extra_config:
        raise ValueError("shared_metadata_name was removed; use pool_name and pool_generation")

    pool_name = _required_setting(extra_config, environ, "pool_name", "SPYRE_KV_POOL_NAME")
    generation = _required_setting(
        extra_config,
        environ,
        "pool_generation",
        "SPYRE_KV_POOL_GENERATION",
    )
    data_pool_name = _optional_setting(
        extra_config,
        environ,
        "data_pool_name",
        "SPYRE_KV_DATA_POOL_NAME",
        "host-0",
    )
    return SharedPoolIdentity(
        pool_name=_logical_name(pool_name, "pool_name"),
        generation=_generation_bytes(generation),
        data_pool_name=_logical_name(data_pool_name, "data_pool_name"),
    )
