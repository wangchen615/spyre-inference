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

from __future__ import annotations

import base64
import copy
import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from spyre_inference.v1.kv_offload.pool_plan import (
    PlannedPool,
    load_planned_pool,
    parse_planned_pool,
)

GENERATION = bytes.fromhex("11" * 16)


def valid_planned_pool_document() -> dict:
    rank_caches = [
        {
            "tpRank": rank,
            "cacheIndex": 0,
            "physicalPageBytes": "3000",
            "layoutKind": "token-major",
            "layoutVersion": 1,
            "localKvHeads": 4,
            "headSize": 128,
        }
        for rank in range(2)
    ]
    layout_signature = hashlib.sha1(
        "".join(
            f"{row['tpRank']}:{row['cacheIndex']}:{row['physicalPageBytes']}:"
            f"{row['layoutKind']}:{row['layoutVersion']}:{row['localKvHeads']}:"
            f"{row['headSize']};"
            for row in rank_caches
        ).encode()
    ).hexdigest()
    partitions = [
        {
            "partitionId": str(rank),
            "arenaName": "host-0",
            "partitionName": f"host-r{rank}-c0",
            "tpRank": rank,
            "cacheIndex": 0,
            "offsetBytes": str(rank * 16_384),
            "numSlots": "4",
            "slotBytes": "4096",
            "physicalPageBytes": "3000",
        }
        for rank in range(2)
    ]
    components = []
    for partition in partitions:
        for role in ("COMPONENT_ROLE_K", "COMPONENT_ROLE_V"):
            components.append(
                {
                    "globalComponentId": str(len(components)),
                    "tpRank": partition["tpRank"],
                    "cacheIndex": 0,
                    "role": role,
                    "arenaName": "host-0",
                    "partitionName": partition["partitionName"],
                    "partitionId": partition["partitionId"],
                    "partitionOffsetBytes": partition["offsetBytes"],
                    "partitionSlotBytes": "4096",
                    "physicalPageBytes": "3000",
                }
            )
    model_spec = {
        "planningVersion": 1,
        "name": "/model-a",
        "model": "model-a",
        "modelRevision": "revision-1",
        "kvDtype": "float16",
        "blockSize": "16",
        "tensorParallelSize": 2,
        "cpuBytesToUse": "40000",
        "rankCaches": rank_caches,
    }
    arena = {
        "name": "host-0",
        "backend": "HOST_DRAM",
        "compatibility": {
            "model": "model-a@revision-1",
            "kvDtype": "float16",
            "blockSize": "16",
            "tensorParallelSize": 2,
            "layoutSignature": layout_signature,
        },
        "partitions": [
            {
                "name": partition["partitionName"],
                "tpRank": partition["tpRank"],
                "cacheIndex": 0,
                "numSlots": "4",
                "slotBytes": "4096",
                "offsetBytes": partition["offsetBytes"],
            }
            for partition in partitions
        ],
    }
    config = {
        "name": "/model-a",
        "metadata": {"maxChunks": 1},
        "dataPools": [arena],
    }
    effective_config = copy.deepcopy(config)
    effective_config["dataPools"][0]["numSlots"] = "8"
    effective_config["dataPools"][0]["slotBytes"] = "4096"
    return {
        "code": "OK",
        "poolDescriptor": {
            "name": "/model-a",
            "generation": base64.b64encode(GENERATION).decode(),
            "state": "READY",
            "effectiveConfig": effective_config,
        },
        "plan": {
            "normalizedSpec": model_spec,
            "bytesPerCompleteBlock": "16384",
            "logicalBlockCapacity": "2",
            "allocatedBytes": "32768",
            "unusedBytes": "7232",
            "config": config,
            "partitions": partitions,
            "components": components,
        },
    }


def test_parser_returns_immutable_complete_plan():
    plan = parse_planned_pool(valid_planned_pool_document())

    assert isinstance(plan, PlannedPool)
    assert plan.identity.pool_name == "/model-a"
    assert plan.identity.generation == GENERATION
    assert plan.tensor_parallel_size == 2
    assert plan.logical_block_capacity == 2
    assert plan.arena.name == "host-0"
    assert [(p.tp_rank, p.cache_index) for p in plan.partitions] == [(0, 0), (1, 0)]
    assert [component.role for component in plan.components] == ["k", "v", "k", "v"]
    with pytest.raises(FrozenInstanceError):
        plan.logical_block_capacity = 3


def test_parser_accepts_compatible_already_existing_generation():
    document = valid_planned_pool_document()
    document["code"] = "ALREADY_EXISTS"

    plan = parse_planned_pool(document)

    assert plan.identity.generation == GENERATION


def test_inline_plan_wins_over_path(tmp_path):
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not json")

    plan = load_planned_pool(
        {
            "pool_plan": valid_planned_pool_document(),
            "pool_plan_path": str(invalid_path),
        },
        environ={},
    )

    assert plan.identity.generation == GENERATION


def test_environment_inline_plan_is_used_when_config_is_absent():
    plan = load_planned_pool(
        {},
        environ={"SPYRE_KV_POOL_PLAN": json.dumps(valid_planned_pool_document())},
    )

    assert plan.identity.pool_name == "/model-a"


def test_environment_path_is_used_when_inline_is_absent(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(valid_planned_pool_document()))

    plan = load_planned_pool({}, environ={"SPYRE_KV_POOL_PLAN_PATH": str(path)})

    assert plan.arena.name == "host-0"


def test_explicit_path_wins_over_environment_inline(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(valid_planned_pool_document()))

    plan = load_planned_pool(
        {"pool_plan_path": str(path)},
        environ={"SPYRE_KV_POOL_PLAN": "not json"},
    )

    assert plan.identity.pool_name == "/model-a"


def _mutate(path, value):
    document = valid_planned_pool_document()
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return document


@pytest.mark.parametrize(
    ("document", "match"),
    [
        (
            _mutate(("poolDescriptor", "generation"), "not-base64"),
            "generation",
        ),
        (
            _mutate(("plan", "normalizedSpec", "planningVersion"), 2),
            "planning version",
        ),
        (
            _mutate(("poolDescriptor", "name"), "/other"),
            "name",
        ),
        (
            _mutate(("poolDescriptor", "state"), "INITIALIZING"),
            "READY",
        ),
        (
            _mutate(("plan", "bytesPerCompleteBlock"), "1"),
            "bytesPerCompleteBlock",
        ),
        (
            _mutate(("plan", "logicalBlockCapacity"), "3"),
            "logicalBlockCapacity",
        ),
        (
            _mutate(("plan", "allocatedBytes"), "32767"),
            "allocatedBytes",
        ),
        (
            _mutate(("plan", "unusedBytes"), "7231"),
            "unusedBytes",
        ),
        (
            _mutate(
                ("plan", "config", "dataPools", 0, "partitions", 0, "slotBytes"),
                "8192",
            ),
            "partition geometry",
        ),
        (
            _mutate(
                ("plan", "config", "dataPools", 0, "compatibility", "tensorParallelSize"),
                4,
            ),
            "tensor parallel",
        ),
    ],
)
def test_parser_recalculates_scalar_and_geometry_invariants(document, match):
    if match in {"partition geometry", "tensor parallel"}:
        document["poolDescriptor"]["effectiveConfig"] = copy.deepcopy(document["plan"]["config"])
    with pytest.raises(ValueError, match=match):
        parse_planned_pool(document)


def test_parser_requires_exactly_one_arena():
    for arenas in ([], [valid_planned_pool_document()["plan"]["config"]["dataPools"][0]] * 2):
        document = valid_planned_pool_document()
        document["plan"]["config"]["dataPools"] = arenas
        document["poolDescriptor"]["effectiveConfig"] = copy.deepcopy(document["plan"]["config"])
        with pytest.raises(ValueError, match="one.*arena"):
            parse_planned_pool(document)


@pytest.mark.parametrize("collection", ["partitions", "components"])
def test_parser_rejects_duplicate_ids(collection):
    document = valid_planned_pool_document()
    document["plan"][collection].append(copy.deepcopy(document["plan"][collection][0]))

    with pytest.raises(ValueError, match="duplicate"):
        parse_planned_pool(document)


def test_parser_rejects_foreign_component_partition():
    document = valid_planned_pool_document()
    document["plan"]["components"][0]["partitionName"] = "foreign"

    with pytest.raises(ValueError, match="partition"):
        parse_planned_pool(document)


@pytest.mark.parametrize("offset", ["1", "16385"])
def test_parser_rejects_gapped_or_overlapping_partition_offsets(offset):
    document = valid_planned_pool_document()
    document["plan"]["partitions"][1]["offsetBytes"] = offset
    document["plan"]["config"]["dataPools"][0]["partitions"][1]["offsetBytes"] = offset
    document["poolDescriptor"]["effectiveConfig"] = copy.deepcopy(document["plan"]["config"])

    with pytest.raises(ValueError, match="offset"):
        parse_planned_pool(document)


def test_parser_requires_complete_k_v_pair():
    document = valid_planned_pool_document()
    document["plan"]["components"].pop()

    with pytest.raises(ValueError, match="K/V"):
        parse_planned_pool(document)


def test_parser_requires_every_tensor_parallel_rank():
    document = valid_planned_pool_document()
    document["plan"]["normalizedSpec"]["rankCaches"].pop()
    document["plan"]["partitions"].pop()
    document["plan"]["components"] = document["plan"]["components"][:2]
    document["plan"]["config"]["dataPools"][0]["partitions"].pop()
    document["poolDescriptor"]["effectiveConfig"] = copy.deepcopy(document["plan"]["config"])

    with pytest.raises(ValueError, match="rank"):
        parse_planned_pool(document)


def test_descriptor_effective_geometry_must_equal_plan():
    document = valid_planned_pool_document()
    document["poolDescriptor"]["effectiveConfig"]["dataPools"][0]["partitions"][0]["numSlots"] = "8"

    with pytest.raises(ValueError, match="effective.*config"):
        parse_planned_pool(document)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("plan", "normalizedSpec", "cpuBytesToUse"), str(1 << 64), "uint64"),
        (("plan", "normalizedSpec", "rankCaches", 0, "layoutVersion"), 2, "layout"),
        (("plan", "normalizedSpec", "rankCaches", 0, "layoutKind"), "other", "layout"),
    ],
)
def test_parser_rejects_out_of_range_and_unsupported_layout_facts(path, value, match):
    with pytest.raises(ValueError, match=match):
        parse_planned_pool(_mutate(path, value))


def test_parser_requires_page_aligned_maximal_geometry():
    unaligned = valid_planned_pool_document()
    unaligned["plan"]["partitions"][0]["slotBytes"] = "3000"
    unaligned["plan"]["config"]["dataPools"][0]["partitions"][0]["slotBytes"] = "3000"
    unaligned["poolDescriptor"]["effectiveConfig"]["dataPools"][0]["partitions"][0]["slotBytes"] = (
        "3000"
    )
    with pytest.raises(ValueError, match="aligned"):
        parse_planned_pool(unaligned)

    nonmaximal = valid_planned_pool_document()
    nonmaximal["plan"]["normalizedSpec"]["cpuBytesToUse"] = "49152"
    nonmaximal["plan"]["unusedBytes"] = "16384"
    with pytest.raises(ValueError, match="maximal"):
        parse_planned_pool(nonmaximal)


def test_parser_requires_deterministic_partition_names_and_component_order():
    renamed = valid_planned_pool_document()
    for location in (
        renamed["plan"]["partitions"],
        renamed["plan"]["config"]["dataPools"][0]["partitions"],
        renamed["poolDescriptor"]["effectiveConfig"]["dataPools"][0]["partitions"],
    ):
        location[0]["partitionName" if "partitionName" in location[0] else "name"] = "other"
    for component in renamed["plan"]["components"][:2]:
        component["partitionName"] = "other"
    with pytest.raises(ValueError, match="name"):
        parse_planned_pool(renamed)

    reordered = valid_planned_pool_document()
    reordered["plan"]["components"][0]["role"] = "COMPONENT_ROLE_V"
    reordered["plan"]["components"][1]["role"] = "COMPONENT_ROLE_K"
    with pytest.raises(ValueError, match="order"):
        parse_planned_pool(reordered)


def test_parser_requires_layout_signature():
    document = valid_planned_pool_document()
    del document["plan"]["config"]["dataPools"][0]["compatibility"]["layoutSignature"]
    del document["poolDescriptor"]["effectiveConfig"]["dataPools"][0]["compatibility"][
        "layoutSignature"
    ]

    with pytest.raises(ValueError, match="layout signature"):
        parse_planned_pool(document)
