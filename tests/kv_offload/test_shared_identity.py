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

from dataclasses import FrozenInstanceError

import pytest

from spyre_inference.v1.kv_offload.shared_identity import (
    load_shared_pool_configuration,
    load_shared_pool_identity,
)
from tests.kv_offload.test_pool_plan import valid_planned_pool_document

GENERATION_HEX = "11" * 16


def test_explicit_identity_wins_over_environment(monkeypatch):
    monkeypatch.setenv("SPYRE_KV_POOL_NAME", "env-root")
    monkeypatch.setenv("SPYRE_KV_POOL_GENERATION", "22" * 16)
    monkeypatch.setenv("SPYRE_KV_DATA_POOL_NAME", "env-child")

    identity = load_shared_pool_identity(
        {
            "pool_name": "explicit-root",
            "pool_generation": GENERATION_HEX,
            "data_pool_name": "explicit-child",
        }
    )

    assert identity.pool_name == "explicit-root"
    assert identity.generation == bytes.fromhex(GENERATION_HEX)
    assert identity.data_pool_name == "explicit-child"


def test_default_data_pool_name_is_host_zero():
    identity = load_shared_pool_identity(
        {"pool_name": "root", "pool_generation": GENERATION_HEX}, environ={}
    )

    assert identity.data_pool_name == "host-0"


def test_all_identity_fields_can_come_from_environment():
    identity = load_shared_pool_identity(
        {},
        environ={
            "SPYRE_KV_POOL_NAME": "env-root",
            "SPYRE_KV_POOL_GENERATION": "AA" * 16,
            "SPYRE_KV_DATA_POOL_NAME": "env-child",
        },
    )

    assert identity.pool_name == "env-root"
    assert identity.generation == bytes.fromhex("AA" * 16)
    assert identity.data_pool_name == "env-child"


def test_invalid_explicit_generation_does_not_fall_back(monkeypatch):
    monkeypatch.setenv("SPYRE_KV_POOL_GENERATION", "22" * 16)

    with pytest.raises(ValueError, match="32 hexadecimal"):
        load_shared_pool_identity({"pool_name": "root", "pool_generation": "bad"})


@pytest.mark.parametrize("key", ["pool_name", "data_pool_name"])
def test_logical_names_cannot_contain_slashes(key):
    config = {
        "pool_name": "root",
        "pool_generation": GENERATION_HEX,
        "data_pool_name": "host-0",
    }
    config[key] = "invalid/name"

    with pytest.raises(ValueError, match=key):
        load_shared_pool_identity(config, environ={})


@pytest.mark.parametrize("missing", ["pool_name", "pool_generation"])
def test_required_identity_fields_cannot_be_missing(missing):
    config = {"pool_name": "root", "pool_generation": GENERATION_HEX}
    del config[missing]

    with pytest.raises(ValueError, match=missing):
        load_shared_pool_identity(config, environ={})


def test_obsolete_shared_metadata_name_has_migration_error():
    with pytest.raises(ValueError, match="shared_metadata_name was removed"):
        load_shared_pool_identity(
            {
                "pool_name": "root",
                "pool_generation": GENERATION_HEX,
                "shared_metadata_name": "old-root",
            },
            environ={},
        )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pool_name", 4),
        ("pool_generation", b"11" * 16),
        ("data_pool_name", ["host-0"]),
    ],
)
def test_identity_fields_must_be_strings(key, value):
    config = {
        "pool_name": "root",
        "pool_generation": GENERATION_HEX,
        "data_pool_name": "host-0",
    }
    config[key] = value

    with pytest.raises(ValueError, match=key):
        load_shared_pool_identity(config, environ={})


def test_identity_is_immutable():
    identity = load_shared_pool_identity(
        {"pool_name": "root", "pool_generation": GENERATION_HEX}, environ={}
    )

    with pytest.raises(FrozenInstanceError):
        identity.pool_name = "replacement"


def test_complete_plan_takes_precedence_over_legacy_identity():
    configuration = load_shared_pool_configuration(
        {
            "pool_plan": valid_planned_pool_document(),
            "pool_name": "legacy-root",
            "pool_generation": GENERATION_HEX,
        },
        environ={},
    )

    assert configuration.identity.pool_name == "/model-a"
