# Spyre Shared KV Offload Functional Result

## Tested revisions

- spyre-inference: `kvc-offload-m2` at `8294ec7524dbf5bfaa7930c2a10ede8c176305b1`,
  plus the acceptance test and this result in the current commit
- torch-spyre: `39b479fdfdc2bf6291336b9f717df888fd05ab5d`
- Flex: `e6dff26d800ab9b56b62e54c90347f34ac9e3eaa`
- vLLM: `0.28.0`

The model was `ibm-ai-platform/micro-g3.3-8b-instruct-1b` at cached snapshot
`6e9c6465a9d7e5e9fa35004a29f0c90befa7d23f`. Instance A used Spyre device 0
(`0000:1a:00.0`); instance B used device 1 (`0000:1b:00.0`).

## Run

```bash
RUN_SPYRE_SHARED_KV_E2E=1 uv run --no-sync pytest \
  tests/kv_offload/test_cross_instance.py -q -s
```

Both servers used `--enforce-eager`, `--no-enable-prefix-caching`, TP1, the
`uni` executor, `PYTHONHASHSEED=0`, and the same metadata and two-family pool
configuration. `VLLM_ENABLE_V1_MULTIPROCESSING=0` prevented the inherited CPU
platform setup from rewriting the requested `uni` executor to `mp`.

The prompt contained 320 explicit token IDs, where token `i` was
`100 + (i * 37) % 1000`. This provided two complete 128-token shared blocks.
Each complete logical KV block was 2,097,152 bytes, so each measured reload
moved 4,194,304 bytes.

## Functional evidence and timing

| Path | Loaded bytes | Load time | Reload throughput |
| --- | ---: | ---: | ---: |
| Instance A self-reload | 4,194,304 | 0.207080 s | 20,254,501 B/s |
| Instance B first-request peer reload | 4,194,304 | 0.208612 s | 20,105,733 B/s |

These are deltas from vLLM's `kv_offload_load_bytes` and
`kv_offload_load_time` counters, not end-to-end request latency. Instance A's
initial request stored 4,194,304 bytes before either reload was released.

Both reload logs reported `hit 256 offloaded tokens after 0 GPU hit tokens`.
The engine configuration logged `enable_prefix_caching=False` for both
instances, so no on-device prefix-cache hit staged the KV data. B's first
generation returned token IDs and text byte-identical to A's no-hit baseline;
A's self-reload did the same. No disk tier was configured or used.

The positive test passed: `1 passed in 602.62s`.

As a mutation check, B was first assigned a different metadata directory. Its
generation still completed by recomputation, while its load-byte and load-time
deltas remained exactly zero and the peer-hit assertion failed. This proves the
positive result depends on the shared directory rather than output equality
alone.

No M1 comparison or performance threshold was run. The recorded timing is
functional evidence for instance A and instance B reloading from the same pool;
M2-versus-M1 performance comparison is deferred.
