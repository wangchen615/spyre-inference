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

"""Single-prompt greedy generation, eager or compiled, against a known reference.

Drives the same model, prompt and sampling params as
`tests/e2e/test_compile.py::test_basic_llm_inference` but with the compile mode under
env control, so the eager and compiled paths can be compared outside pytest.

    EAGER=1 python scripts/probes/basic_llm_inference.py   # eager
    EAGER=0 python scripts/probes/basic_llm_inference.py   # STOCK_TORCH_COMPILE

Exits non-zero when the generated text does not match REF_OUTPUT.

Env: EAGER, MODEL, REF_OUTPUT, PROMPT, MAX_TOKENS, MAX_MODEL_LEN, MAX_NUM_SEQS,
MAX_NUM_BATCHED_TOKENS
"""

import os
import sys

MODEL = os.environ.get("MODEL") or "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
REF_OUTPUT = os.environ.get(
    "REF_OUTPUT",
    "\n\nIBMs main businesses are the companies that provide the services of the",
)
PROMPT = os.environ.get("PROMPT") or "What are IBMs main businesses?"
EAGER = os.environ.get("EAGER", "1") not in ("0", "false", "False", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS") or "16")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN") or "128")
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS") or "2")
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS") or "8")


def main() -> int:
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    os.environ.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "36000")

    kwargs: dict = dict(
        model=MODEL,
        enforce_eager=EAGER,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
    )
    # Mirrors the compiled e2e test, which pins the two buckets it warms.
    if not EAGER:
        kwargs["compilation_config"] = CompilationConfig(compile_sizes=[1, 8])

    engine = LLM(**kwargs)
    out = engine.generate(
        PROMPT, SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS), use_tqdm=False
    )
    text = out[0].outputs[0].text

    print(f"mode: {'eager' if EAGER else 'compiled'}")
    print(f"prompt_ok: {PROMPT == out[0].prompt}")
    print(f"text: {text!r}")
    print(f"matches_ref: {text == REF_OUTPUT}")
    return 0 if text == REF_OUTPUT else 1


if __name__ == "__main__":
    sys.exit(main())
