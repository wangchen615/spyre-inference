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
            "arenaName": f"host-r{rank}",
            "partitionName": f"host-r{rank}-c0",
            "tpRank": rank,
            "cacheIndex": 0,
            "offsetBytes": "0",
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
                    "arenaName": partition["arenaName"],
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
    arenas = [
        {
            "name": f"host-r{rank}",
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
                    "name": partitions[rank]["partitionName"],
                    "tpRank": rank,
                    "cacheIndex": 0,
                    "numSlots": "4",
                    "slotBytes": "4096",
                    "offsetBytes": "0",
                }
            ],
        }
        for rank in range(2)
    ]
    config = {
        "name": "/model-a",
        "metadata": {"maxChunks": 1},
        "dataPools": arenas,
    }
    effective_config = copy.deepcopy(config)
    for arena in effective_config["dataPools"]:
        arena["numSlots"] = "4"
        arena["slotBytes"] = "4096"
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
    assert [arena.name for arena in plan.arenas] == ["host-r0", "host-r1"]
    assert plan.arena_for_rank(1).name == "host-r1"
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

    assert plan.arena_for_rank(0).name == "host-r0"


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


def test_parser_requires_exactly_one_arena_per_tensor_parallel_rank():
    valid = valid_planned_pool_document()["plan"]["config"]["dataPools"]
    for arenas in ([], valid[:1]):
        document = valid_planned_pool_document()
        document["plan"]["config"]["dataPools"] = arenas
        document["poolDescriptor"]["effectiveConfig"] = copy.deepcopy(document["plan"]["config"])
        with pytest.raises(ValueError, match="arena.*rank|rank.*arena"):
            parse_planned_pool(document)

    duplicate = valid_planned_pool_document()
    duplicate["plan"]["config"]["dataPools"][1] = copy.deepcopy(
        duplicate["plan"]["config"]["dataPools"][0]
    )
    duplicate["poolDescriptor"]["effectiveConfig"]["dataPools"][1] = copy.deepcopy(
        duplicate["poolDescriptor"]["effectiveConfig"]["dataPools"][0]
    )
    with pytest.raises(ValueError, match="arena.*rank|rank.*arena"):
        parse_planned_pool(duplicate)


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


@pytest.mark.parametrize("offset", ["1", "4096"])
def test_parser_rejects_gapped_or_overlapping_partition_offsets(offset):
    document = valid_planned_pool_document()
    document["plan"]["partitions"][0]["offsetBytes"] = offset
    document["plan"]["config"]["dataPools"][0]["partitions"][0]["offsetBytes"] = offset
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
    document["plan"]["config"]["dataPools"].pop()
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


MARVELL_BDF = "0000:ed:00.0"


def _rename_tier(value, old: str, new: str):
    if isinstance(value, dict):
        return {key: _rename_tier(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_tier(item, old, new) for item in value]
    if isinstance(value, str) and value.startswith(f"{old}-r"):
        return new + value[len(old) :]
    return value


def marvell_planned_pool_document(bar_offsets=(0, 1 << 20)) -> dict:
    """The host document as the delegator plans it for one Marvell card."""
    document = _rename_tier(valid_planned_pool_document(), "host", "marvell")
    spec = document["plan"]["normalizedSpec"]
    spec["marvell"] = {"pciBdf": MARVELL_BDF, "marvellBytesToUse": spec["cpuBytesToUse"]}
    spec["cpuBytesToUse"] = "0"
    for arena in document["plan"]["config"]["dataPools"]:
        arena["backend"] = "MARVELL"
        arena["marvell"] = {"pciBdf": MARVELL_BDF}
    for arena, offset in zip(
        document["poolDescriptor"]["effectiveConfig"]["dataPools"], bar_offsets, strict=True
    ):
        arena["backend"] = "MARVELL"
        arena["marvell"] = {"pciBdf": MARVELL_BDF, "barOffset": str(offset)}
    return document


def test_marvell_plan_uses_marvell_tier_names_and_flex_bar_offsets():
    plan = parse_planned_pool(marvell_planned_pool_document())

    assert plan.backend == "MARVELL"
    assert plan.cpu_bytes_to_use == 40000  # the plan's marvellBytesToUse.
    assert [arena.name for arena in plan.arenas] == ["marvell-r0", "marvell-r1"]
    assert {arena.backend for arena in plan.arenas} == {"MARVELL"}
    assert {arena.pci_bdf for arena in plan.arenas} == {MARVELL_BDF}
    assert [arena.bar_offset for arena in plan.arenas] == [0, 1 << 20]  # placed by flex.
    assert [item.partition_name for item in plan.partitions] == [
        "marvell-r0-c0",
        "marvell-r1-c0",
    ]
    assert plan.identity.data_pool_name == "marvell-r0"


def test_host_plan_carries_no_marvell_placement():
    plan = parse_planned_pool(valid_planned_pool_document())

    assert plan.backend == "HOST_DRAM"
    assert {(arena.pci_bdf, arena.bar_offset) for arena in plan.arenas} == {("", None)}


def test_mixed_host_and_marvell_arenas_are_rejected():
    """D3: one backend per pool."""
    document = marvell_planned_pool_document()
    for config in (
        document["plan"]["config"],
        document["poolDescriptor"]["effectiveConfig"],
    ):
        config["dataPools"][1]["backend"] = "HOST_DRAM"
    with pytest.raises(ValueError, match="does not match the plan's 'MARVELL'"):
        parse_planned_pool(document)


def test_host_names_are_rejected_in_a_marvell_plan():
    document = marvell_planned_pool_document()
    document["plan"]["config"]["dataPools"][0]["name"] = "host-r0"
    with pytest.raises(ValueError, match="'marvell-r0'"):
        parse_planned_pool(document)


def test_host_plan_rejects_a_marvell_placement():
    document = valid_planned_pool_document()
    document["poolDescriptor"]["effectiveConfig"]["dataPools"][0]["marvell"] = {
        "pciBdf": MARVELL_BDF,
        "barOffset": "0",
    }
    with pytest.raises(ValueError, match="host arena must not carry a Marvell placement"):
        parse_planned_pool(document)


@pytest.mark.parametrize(
    ("location", "message"),
    [
        (("plan", "config", "dataPools", 0, "marvell"), "differs from the normalized spec"),
        (
            ("poolDescriptor", "effectiveConfig", "dataPools", 1, "marvell"),
            "effective Marvell card",
        ),
    ],
)
def test_marvell_card_must_match_the_spec_everywhere(location, message):
    document = marvell_planned_pool_document()
    target = document
    for key in location:
        target = target[key]
    target["pciBdf"] = "0000:ee:00.0"
    with pytest.raises(ValueError, match=message):
        parse_planned_pool(document)


def test_marvell_effective_bar_offset_is_required_and_page_aligned():
    missing = marvell_planned_pool_document()
    del missing["poolDescriptor"]["effectiveConfig"]["dataPools"][0]["marvell"]["barOffset"]
    with pytest.raises(ValueError, match="barOffset"):
        parse_planned_pool(missing)

    unaligned = marvell_planned_pool_document(bar_offsets=(0, 128))
    with pytest.raises(ValueError, match="page aligned"):
        parse_planned_pool(unaligned)


def test_marvell_plan_requires_zero_cpu_bytes_to_use():
    document = marvell_planned_pool_document()
    document["plan"]["normalizedSpec"]["cpuBytesToUse"] = "40000"
    with pytest.raises(ValueError, match="cpuBytesToUse 0"):
        parse_planned_pool(document)
