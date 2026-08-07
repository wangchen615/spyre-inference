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

"""KV blocks must physically move to host under memory pressure, and come back.

This is the test that proves offloading *works*, as opposed to
``test_e2e_offload``, which only proves the connector is transparent. Transparency
passes vacuously if no block is ever transferred, so this test forces the transfer
and asserts it happened.

**How pressure is created.** The device pool is pinned small
(``num_gpu_blocks_override``) while the host pool stays large: the two are
independent by design -- the host tier exists precisely to hold what no longer fits
on device. Several distinct long prompts are then run sequentially, so each needs
fresh device blocks and the scheduler must evict earlier ones to host. Re-issuing an
early prompt afterwards must then read it back from host.

**Why the prompts are long.** The offloading scheduler stores only *complete*
blocks, so a prompt shorter than one block never produces anything offloadable.
Spyre forces ``block_size`` to a multiple of 64 (see
``TorchSpyrePlatform.check_and_update_config``), so each prompt below is comfortably
longer than one block.

**How transfers are observed.** ``SpyreOffloadingWorker`` logs one line per transfer
and keeps cumulative counters, but it lives in the worker subprocess -- ``caplog``
only sees the test process, so the log lines are captured at fd level with
``capfd``. NOTE: ``capfd`` is inactive under ``pytest -s``; run without ``-s`` or the
assertions see nothing.

**Why the hardware gate is env-based.** The card admits exactly one process at a
time, so anything that opens it in the *pytest* process makes the vLLM worker
subprocess fail with VFIO ``EBUSY``. That rules out ``spyre_available()``, which
gates by allocating a device tensor. ``spyre_device_count()`` reads
``AIU_WORLD_SIZE`` instead and never touches the runtime -- the required pattern for
every ``uses_subprocess`` test.
"""

import pytest
from spyre_testing_plugin.pytest_plugin import spyre_device_count

MODEL = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
MAX_LEN = 512
DEVICE_BLOCKS = 4
CPU_BYTES_TO_USE = 2_000_000_000

# Log fragments emitted by SpyreOffloadingWorker for each direction. These must stay
# in sync with the logger.info call in spyre_inference/v1/kv_offload/worker.py.
STORE_MARKER = "SpyreOffloadingWorker device->host"
LOAD_MARKER = "SpyreOffloadingWorker host->device"

# Six distinct ~150-token prompts. Each occupies fresh blocks, so DEVICE_BLOCKS
# cannot hold them all and the scheduler must offload to host.
PROMPTS = {
    "A": (
        "Paris is the capital and most populous city of France, situated on the banks of "
        "the river Seine in the north of the country. For centuries it has been a global "
        "center of art, fashion, gastronomy, and culture. Its nineteenth-century cityscape "
        "is crossed by wide boulevards and the river Seine. Beyond landmarks such as the "
        "Eiffel Tower and the Gothic Notre-Dame cathedral, the city is renowned for its cafe "
        "culture, its many museums including the Louvre and the Musee d'Orsay, and its "
        "reputation as a destination for romance, learning, and the arts throughout modern "
        "European history."
    ),
    "B": (
        "Tokyo is the capital and largest metropolitan area of Japan, located at the head of "
        "Tokyo Bay on the eastern coast of the main island of Honshu. Once a small fishing "
        "village known as Edo, it grew into one of the most populous and economically "
        "powerful urban regions in the world. The city blends ancient temples and quiet "
        "gardens with dense districts of neon, commerce, and technology. It serves as the "
        "political, financial, and cultural heart of the nation, hosting government "
        "institutions, global corporations, world-class transit, and a cuisine celebrated "
        "across the entire planet for its precision."
    ),
    "C": (
        "London is the capital and largest city of England and the United Kingdom, standing "
        "on the river Thames in the south-east of the island of Great Britain. With a history "
        "spanning nearly two millennia since its founding by the Romans as Londinium, it has "
        "grown into a leading global city for finance, commerce, law, education, and the arts. "
        "Famous landmarks include Big Ben, the Tower of London, Tower Bridge, Westminster "
        "Abbey, and Buckingham Palace. Its many museums, theatres, parks, and universities "
        "draw millions of visitors and students every year from every corner of the wider "
        "connected world."
    ),
    "D": (
        "Berlin is the capital and largest city of Germany, lying on the banks of the river "
        "Spree in the north-eastern part of the country. Once divided by a wall that came to "
        "symbolize the wider Cold War, it is today a unified, vibrant metropolis celebrated "
        "for its layered history, its progressive culture, and its thriving arts scene. The "
        "city is dotted with monuments, memorials, and museums that document centuries of "
        "triumph and tragedy, from the Brandenburg Gate to Museum Island. It has also become "
        "a magnet for artists, musicians, startups, and students drawn by its openness and "
        "its restless energy."
    ),
    "E": (
        "Rome is the capital city of Italy and a special municipality lying along the banks "
        "of the river Tiber in the central western portion of the Italian peninsula. As the "
        "former heart of the vast Roman Empire, it is often called the Eternal City and "
        "contains an extraordinary concentration of ancient ruins, monuments, and works of "
        "art. Visitors come to see the Colosseum, the Roman Forum, the Pantheon, and the many "
        "fountains and piazzas that fill the historic center. Surrounding the independent "
        "enclave of Vatican City, Rome remains a profound center of religion, history, "
        "architecture, and Renaissance art."
    ),
    "F": (
        "Madrid is the capital and most populous city of Spain, set on the elevated plains of "
        "the Iberian peninsula near the geographic center of the country. It grew from a modest "
        "Moorish fortress into the seat of the Spanish court and today serves as the political, "
        "economic, and cultural hub of the nation. The city is celebrated for its grand "
        "boulevards, its royal palace, and its world-class art museums such as the Prado and the "
        "Reina Sofia. Lively plazas, late-night dining, and football passion give Madrid a "
        "distinctive energy that draws travellers and residents alike throughout every season "
        "of the year."
    ),
}

# A (warm-up) then B..F to overflow the device pool and evict B, then B again --
# which must come back from host rather than being recomputed.
STEP_ORDER = ("A", "B", "C", "D", "E", "F", "B")


@pytest.mark.spyre
@pytest.mark.uses_subprocess
def test_blocks_offload_to_host_and_reload_under_pressure(capfd):
    if spyre_device_count() < 1:
        pytest.skip("Spyre device not available")

    from vllm import LLM, SamplingParams
    from vllm.config import AttentionConfig, KVTransferConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    llm = LLM(
        MODEL,
        max_model_len=MAX_LEN,
        max_num_seqs=1,  # sequential, so each prompt competes for the same blocks
        num_gpu_blocks_override=DEVICE_BLOCKS,
        enable_prefix_caching=True,
        attention_config=AttentionConfig(backend=AttentionBackendEnum["CUSTOM"]),
        kv_transfer_config=KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "spec_name": "SpyreOffloadingSpec",
                # Import the spec module directly instead of relying on the
                # `vllm.general_plugins` entry point. The worker process builds the
                # connector in `initialize_from_config`, which can run before
                # `load_general_plugins()` has registered our spec -- the registry
                # lookup then fails with "Unsupported spec type". `spec_module_path`
                # bypasses the registry (see OffloadingSpecFactory.get_spec_cls).
                "spec_module_path": "spyre_inference.v1.kv_offload.spec",
                "cpu_bytes_to_use": CPU_BYTES_TO_USE,
            },
        ),
    )
    params = SamplingParams(temperature=0.0, max_tokens=3, min_tokens=1)

    # Drop anything logged while the engine was starting, so each step below only
    # sees transfer lines from its own generate() call.
    capfd.readouterr()

    stores, loads = 0, 0
    for label in STEP_ORDER:
        output = llm.generate(prompts=PROMPTS[label], sampling_params=params)
        assert output[0].outputs[0].text is not None, f"prompt {label} produced no output"

        captured = capfd.readouterr()
        text = captured.out + captured.err
        stores += text.count(STORE_MARKER)
        loads += text.count(LOAD_MARKER)

    assert stores > 0, (
        f"no device->host transfer was logged across {len(STEP_ORDER)} prompts. Either "
        f"{DEVICE_BLOCKS} device blocks did not create enough pressure, the prompts are "
        "shorter than one block (only complete blocks are stored), or the connector is "
        "not active."
    )
    assert loads > 0, (
        "blocks were offloaded to host but never read back. Re-issuing prompt B after "
        "five other prompts should have reloaded it from the host tier; it may have "
        "still been resident on device (a device prefix-cache hit instead), or it was "
        "recomputed rather than fetched."
    )
