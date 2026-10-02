# Spyre Shared KV Two-Instance Manual Demo

This procedure runs two independent vLLM instances on separate Spyre devices
and demonstrates that A, then A again, then B use one shared host-memory KV
pool for the same 4,096-token prompt. Prefix caching is disabled, so a measured
reload cannot be satisfied by device-resident prefix KV.

## Verified configuration

The 2026-10-02 run used:

- Flex `c22ae7663f625180081ed259ad4ac1916f03e9a3`;
- torch-spyre `0548715a0d3a2b338f71d144e60147fb72025f3b`;
- spyre-inference base `428321d38dbb3fec54987167eac8a42dec17136f`
  plus the one-pool redesign contained by the eventual commit that includes
  this result;
- vLLM `0.28.0`;
- model `ibm-ai-platform/micro-g3.3-8b-instruct-1b`;
- device 0 at `0000:ac:00.0` for A and device 1 at <code>0000:b&#x61;:00.0</code> for B; and
- `PYTHONHASHSEED=0`, TP1, the `uni` executor, and eager execution.

Both launchers pass the same connector configuration:

```yaml
enable_prefix_caching: false
max_num_batched_tokens: 512
max_num_seqs: 1
kv_connector: SpyreOffloadingConnector
kv_role: kv_both
shared_metadata_name: spyre_manual_4096
pool_name: spyre_manual_4096.data
cpu_bytes_to_use: 536870912
```

`enable_prefix_caching: false` prevents vLLM's device-local prefix cache from
masking the shared-KV path. `max_num_batched_tokens: 512` bounds each scheduler
step, and `max_num_seqs: 1` keeps one request active per instance.
`SpyreOffloadingConnector` can both store and load because `kv_role` is
`kv_both`. The metadata name identifies the shared directory; the pool name
identifies its one data pool. The 512-MiB budget produces eight physical page
components per logical block, 256-KiB slots, 2,048 slots, and capacity for 256
logical blocks.

## Clean stale demo objects

Stop both demo servers before cleanup. The command refuses to unlink the
namespace while a vLLM process references it.

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
uv run --no-sync python scripts/cleanup_shared_kv_demo.py
```

The cleanup script discovers and removes the current one-pool layout and the
historical 16-pool layout registered under `spyre_manual_4096`. Repeated
`--pool-name NAME` arguments add other known demo-owned pools. It refuses a
directory containing any unrecognized pool and never wildcard-deletes
unrelated `/dev/shm/flex_kv_*` objects. Its temporary runtime uses the mock
device and null compute backend, so it does not reserve a physical card.

## Start the servers

In terminal A:

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
bash scripts/start_shared_kv_instance_a.sh
```

In terminal B:

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
bash scripts/start_shared_kv_instance_b.sh
```

Wait for both terminals to report `Application startup complete`. To use other
devices, set `SPYRE_DEVICE_A` and `SPYRE_DEVICE_B` independently.

## Inspect the one-pool topology

While both servers are running:

```bash
stat -c '%n  %s bytes' /dev/shm/spyre_manual_4096
strings /dev/shm/spyre_manual_4096 \
  | grep '^/flex_kv_' \
  | sort -u \
  | while read -r pool; do
      stat -c '%n  %s bytes' "/dev/shm/${pool#/}"
      stat -c '%n  %s bytes' "/dev/shm/${pool#/}.ctl"
    done
```

Expect one unique `flex_kv_...` base name:

```text
/dev/shm/spyre_manual_4096  328984 bytes
/dev/shm/flex_kv_d5a9901a07cded00_0_0000000000000001  536870912 bytes
/dev/shm/flex_kv_d5a9901a07cded00_0_0000000000000001.ctl  4096 bytes
```

The first object is metadata, not another data pool. The final two paths are
the backing and control objects for the same single data pool.

## Warm both instances with unrelated data

The warmup prompt is visibly different from the measured incident prompt. It
first computes one unique prompt on each server, then exercises A's self-load
and B's peer-load paths.

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
set -o pipefail
LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync python -u scripts/warmup_shared_kv_demo.py \
  --instance-a-host 127.0.0.1 --instance-a-port 18100 \
  --instance-b-host 127.0.0.1 --instance-b-port 18101 2>&1 \
  | tee /tmp/spyre-shared-kv-one-pool-warmup.log
```

## Send the measured request to A, A, and B

Each invocation prints its instance and endpoint, stable identifier, prompt
preview, exact token count, path classification, TTFT, E2E time, prompt-source
tokens, KV bytes/copy time, output token IDs, and output text.

First send the cold request to A:

```bash
LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance A --host 127.0.0.1 --port 18100 2>&1 \
  | tee /tmp/spyre-shared-kv-one-pool-a-cold.log
```

Send the identical request to A again:

```bash
LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance A --host 127.0.0.1 --port 18100 2>&1 \
  | tee /tmp/spyre-shared-kv-one-pool-a-reload.log
```

Finally, send the same request to B:

```bash
LD_LIBRARY_PATH=/opt/ibm/spyre/spyre-comms/lib:/home/yzhu/dt-inductor/sentient/runtime/lib:/opt/ibm/spyre/runtime/lib:$LD_LIBRARY_PATH \
uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance B --host 127.0.0.1 --port 18101 2>&1 \
  | tee /tmp/spyre-shared-kv-one-pool-b-reload.log
```

Keep the default identifier `spyre-shared-kv-visual-demo-v1`, or pass the same
explicit `--identifier` to all three invocations.

## Fresh one-pool result

The 2026-10-02 run produced:

| Request | TTFT | E2E wall time | Prompt source | KV copy time | KV bytes |
| --- | ---: | ---: | --- | ---: | ---: |
| A cold compute/store | 3.386 s | 12.142 s | 4,096 local-compute tokens | 0.010332 s | 67,108,864 stored |
| A self reload | 0.575 s | 9.244 s | 4,096 external-transfer tokens | 0.006399 s | 67,108,864 loaded |
| B peer reload | 0.574 s | 9.302 s | 4,096 external-transfer tokens | 0.005588 s | 67,108,864 loaded |

Self-reload reduced TTFT by 83.0%, and peer reload reduced TTFT by 83.0%.
Their E2E reductions were 23.9% and 23.4%. All requests generated 16 output
tokens. The measured token IDs were identical:

```text
[203, 203, 1397, 44, 10720, 3303, 518, 322, 15872, 2582, 11385, 39171, 461, 7624, 8019, 26322]
```

They decoded byte-identically to:

```text
"\n\nResponse: Summarize the recurring symptoms and give three concrete"
```

The warmup also proved both paths before measurement:

| Warmup request | TTFT | E2E wall time | Prompt source | KV bytes |
| --- | ---: | ---: | --- | ---: |
| A junk compute/store | 82.518 s | 131.801 s | 4,096 local-compute tokens | 67,108,864 stored |
| B distinct junk compute/store | 81.636 s | 132.506 s | 4,096 local-compute tokens | 67,108,864 stored |
| A junk self reload | 8.449 s | 17.292 s | 4,096 external-transfer tokens | 67,108,864 loaded |
| B junk peer reload | 8.414 s | 17.303 s | 4,096 external-transfer tokens | 67,108,864 loaded |

## Stop and clean up

Stop B and then A with Ctrl-C. Run:

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
uv run --no-sync python scripts/cleanup_shared_kv_demo.py \
  | tee /tmp/spyre-shared-kv-one-pool-cleanup.log
test ! -e /dev/shm/spyre_manual_4096
```

The verified result was:

```text
Removed 1 shared data pools and metadata 'spyre_manual_4096'.
```

The recorded backing and `.ctl` objects were also absent afterward, while
unrelated `flex_kv_*` objects remained.

## Artifacts

- `/tmp/spyre-shared-kv-one-pool-a-server.log`
- `/tmp/spyre-shared-kv-one-pool-b-server.log`
- `/tmp/spyre-shared-kv-one-pool-topology.log`
- `/tmp/spyre-shared-kv-one-pool-topology-after.log`
- `/tmp/spyre-shared-kv-one-pool-warmup.log`
- `/tmp/spyre-shared-kv-one-pool-a-cold.log`
- `/tmp/spyre-shared-kv-one-pool-a-reload.log`
- `/tmp/spyre-shared-kv-one-pool-b-reload.log`
- `/tmp/spyre-shared-kv-one-pool-cleanup.log`

## Historical pre-redesign reference

The 2026-10-01 family-layout run used 16 component pools. It reported 3.243 s
cold TTFT, 0.563 s A reload TTFT, and 0.573 s B reload TTFT. Those performance
numbers remain useful as a historical comparison, but its 16-pool topology is
not the current architecture or expected output.
