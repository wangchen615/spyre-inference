# Spyre Shared KV M4 Two-Instance Manual Demo

This runbook starts two independent vLLM instances on separate Spyre cards and
demonstrates that instance A can compute and publish KV pages that are later
loaded by both A and B from one persistent, node-local Flex M4 pool.

M4 changes the ownership model from the earlier demo:

- `shared-pool-delegator` creates and destroys the complete metadata-plus-data
  pool generation.
- Spyre-inference workers only attach to the exact generation and detach.
- The pool survives with zero worker attachments and also survives a delegator
  restart.
- The last detach never unlinks the pool. Cleanup requires an explicit,
  generation-checked destroy.

Do not directly unlink anything under `/dev/shm`.

## What this demo proves

The request sequence is:

1. Send an unrelated warmup request to A and another unrelated request to B.
2. Send the measured 4,096-token request to A: cold compute and publish.
3. Send the identical request to A: self-reload from the shared pool.
4. Send the identical request to B: peer reload from the same pool generation.
5. Verify destroy returns `POOL_BUSY` while workers remain attached.
6. Stop both workers, then destroy the exact generation successfully.

Prefix caching is disabled, so a measured reload cannot be satisfied by
vLLM's device-local prefix cache.

## Prerequisites

The following must already be built:

- Flex branch `kvc-offload-m4`;
- torch-spyre branch `kvc-offload-m4`;
- spyre-inference branch `kvc-offload-m4`;
- shared-pool-delegator branch `kvc-offload-m4`; and
- the spyre-comms revision compatible with the Flex M4 launch ABI.

The delegator and both workers must run as the same numeric host UID, see the
same `/dev/shm` namespace, and use compatible native libraries. Two real Spyre
cards must be idle.

The commands below use the paths verified on the development machine. Adjust
only the path variables if the repositories were built elsewhere.

```bash
export DTI_PROJECT_ROOT=/home/yzhu/dt-inductor
export FLEX_M4="$DTI_PROJECT_ROOT/flex-worktrees/persistent-shared-kv-pool-lifecycle"
export TORCH_SPYRE_M4="$DTI_PROJECT_ROOT/torch-spyre/.worktrees/kvc-offload-m4"
export SPYRE_INFERENCE_M4="$DTI_PROJECT_ROOT/spyre-inference-kvc-offload-m4"
export DELEGATOR_M4="$DTI_PROJECT_ROOT/shared-pool-delegator/.worktrees/kvc-offload-m4"
export SPYRE_COMMS_M4="$DTI_PROJECT_ROOT/sentient/spyre_comms-m4-compat"
export DELEGATOR_SOCKET="/tmp/shared-pool-delegator-${UID}.sock"

export PYTHONPATH="$TORCH_SPYRE_M4:${PYTHONPATH-}"
export LD_LIBRARY_PATH="$FLEX_M4/build:$SPYRE_COMMS_M4/lib:$DTI_PROJECT_ROOT/sentient/libaiupti/lib:$DTI_PROJECT_ROOT/sentient/runtime/lib:$DTI_PROJECT_ROOT/sentient/deeptools/lib:${LD_LIBRARY_PATH-}"

cd "$SPYRE_INFERENCE_M4"
```

Do not replace the complete existing `LD_LIBRARY_PATH` with only the Flex and
spyre-comms directories. DeepTools utilities also need `libdcc.so` from
`$DTI_PROJECT_ROOT/sentient/deeptools/lib`.

Confirm the required binaries exist:

```bash
test -x "$DELEGATOR_M4/build/shared-pool-delegator"
test -x "$DELEGATOR_M4/build/shared-poolctl"
test -d "$FLEX_M4/build"
test -d "$TORCH_SPYRE_M4/torch_spyre"
```

## 1. Start the delegator

Run this in the control terminal:

```bash
"$DELEGATOR_M4/build/shared-pool-delegator" \
  --socket-path "$DELEGATOR_SOCKET" \
  --socket-mode 0600 \
  >/tmp/spyre-shared-kv-m4-delegator.log 2>&1 &
export DELEGATOR_PID=$!

until "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" health >/dev/null 2>&1; do
  kill -0 "$DELEGATOR_PID"
  sleep 0.1
done
```

The delegator is a lifecycle controller, not the owner of the pool's lifetime.
If it exits after creation, a READY pool remains in shared memory.

## 2. Create the complete persistent pool

The helper constructs one composite generation containing the metadata
directory and the fixed `host-0` data child before publishing it as READY:

```bash
POOL_JSON=$(uv run --active --no-sync python \
  scripts/shared_kv_two_instance_demo.py prepare-pool \
  --ctl "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" \
  --pool-name /micro-g3-m4-demo \
  --data-pool-name host-0 \
  --model ibm-ai-platform/micro-g3.3-8b-instruct-1b \
  --cpu-bytes-to-use 536870912 \
  --slot-bytes 262144)

printf '%s\n' "$POOL_JSON" | jq .

export SHARED_POOL_NAME=$(jq -r .poolName <<<"$POOL_JSON")
export SHARED_POOL_GENERATION=$(jq -r .poolGeneration <<<"$POOL_JSON")
export SHARED_POOL_DATA_NAME=$(jq -r .dataPoolName <<<"$POOL_JSON")
```

Expected descriptor properties:

- pool name `/micro-g3-m4-demo`;
- a 32-hexadecimal-character opaque generation;
- state `READY`;
- data child `host-0`;
- 2,048 fixed slots of 262,144 bytes each; and
- 512 MiB of host data capacity.

Save the generation. A reused pool name receives a different generation, and
the old descriptor cannot attach to or destroy that replacement.

### Optional: prove delegator restart persistence

This check is safe before workers attach:

```bash
kill -TERM "$DELEGATOR_PID"
wait "$DELEGATOR_PID"

"$DELEGATOR_M4/build/shared-pool-delegator" \
  --socket-path "$DELEGATOR_SOCKET" \
  --socket-mode 0600 \
  >/tmp/spyre-shared-kv-m4-delegator.log 2>&1 &
export DELEGATOR_PID=$!

until "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" health >/dev/null 2>&1; do
  kill -0 "$DELEGATOR_PID"
  sleep 0.1
done

"$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" open \
  --name "$SHARED_POOL_NAME" | jq .
```

The opened descriptor must still be READY and must report the same generation.
Opening only inspects the lifecycle descriptor; it does not attach a worker.

## 3. Start both Spyre-inference instances

The launch scripts read the three exported pool identity values. Start both
servers from the same control terminal so they inherit them:

```bash
SPYRE_DEVICE_A=0 scripts/start_shared_kv_instance_a.sh \
  >/tmp/spyre-shared-kv-m4-instance-a.log 2>&1 &
export INSTANCE_A_PID=$!

SPYRE_DEVICE_B=1 scripts/start_shared_kv_instance_b.sh \
  >/tmp/spyre-shared-kv-m4-instance-b.log 2>&1 &
export INSTANCE_B_PID=$!
```

The launchers configure:

```json
{
  "spec_name": "SpyreSharedOffloadingSpec",
  "pool_name": "/micro-g3-m4-demo",
  "pool_generation": "<generation returned by create>",
  "data_pool_name": "host-0",
  "cpu_bytes_to_use": 536870912
}
```

They use TP1, one request at a time, disabled prefix caching, and the `uni`
executor. Wait until both health endpoints respond:

```bash
until curl -fsS http://127.0.0.1:18100/health >/dev/null; do
  kill -0 "$INSTANCE_A_PID"
  sleep 1
done

until curl -fsS http://127.0.0.1:18101/health >/dev/null; do
  kill -0 "$INSTANCE_B_PID"
  sleep 1
done
```

Server startup can take several minutes while model and attention graphs are
compiled. Inspect logs with:

```bash
tail -f /tmp/spyre-shared-kv-m4-instance-a.log
tail -f /tmp/spyre-shared-kv-m4-instance-b.log
```

## 4. Verify that attached workers block destroy

While either server is running, an exact-generation destroy must fail with
`POOL_BUSY`:

```bash
if "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" destroy \
  --name "$SHARED_POOL_NAME" \
  --generation "$SHARED_POOL_GENERATION"; then
  echo "ERROR: destroy unexpectedly succeeded while workers were attached"
  exit 1
else
  echo "Expected POOL_BUSY while workers are attached"
fi
```

The workers hold shared locks on the root. The destroy path cannot acquire its
exclusive lock until every worker attachment and resolved child handle is
released.

## 5. Warm both instances with unrelated prompts

Warmup A and B with different content so the measured prompt is not already in
the pool:

```bash
set -o pipefail
uv run --active --no-sync python -u scripts/warmup_shared_kv_demo.py \
  --instance-a-host 127.0.0.1 --instance-a-port 18100 \
  --instance-b-host 127.0.0.1 --instance-b-port 18101 \
  --show-json 2>&1 \
  | tee /tmp/spyre-shared-kv-m4-warmup.log
```

Each warmup request should report local computation and a store. It should not
reload the measured prompt.

## 6. Run A cold, A reload, then B peer reload

Keep the default identifier unchanged for all three commands.

Cold compute and publish on A:

```bash
uv run --active --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance A --host 127.0.0.1 --port 18100 \
  --show-json 2>&1 \
  | tee /tmp/spyre-shared-kv-m4-a-cold.log
```

Reload the identical prompt on A:

```bash
uv run --active --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance A --host 127.0.0.1 --port 18100 \
  --show-json 2>&1 \
  | tee /tmp/spyre-shared-kv-m4-a-reload.log
```

Reload the same prompt from B:

```bash
uv run --active --no-sync python -u scripts/shared_kv_two_instance_demo.py \
  --instance B --host 127.0.0.1 --port 18101 \
  --show-json 2>&1 \
  | tee /tmp/spyre-shared-kv-m4-b-reload.log
```

Expected evidence:

| Request | Expected prompt source | Expected KV activity |
| --- | --- | --- |
| A cold | local compute | store bytes increase |
| A reload | external KV transfer | load bytes and load time increase |
| B reload | external KV transfer | load bytes and load time increase |

The three responses must have identical output token IDs and byte-identical
decoded text. The A and B reloads should each load the complete shared prefix.

## Example M4 performance

### Verified two-card M4 acceptance result

The coordinated M4 acceptance run on 2026-10-27 used the same persistent
delegator-created generation across two cards. Its shorter deterministic test
prompt contained 320 tokens and two complete 128-token blocks. It produced:

| Operation | KV bytes | Flex KV-copy time | Effective copy rate |
| --- | ---: | ---: | ---: |
| A publish | 4,194,304 | not recorded by this test | not calculated |
| A self reload | 4,194,304 | 0.000965 s | 4.35 GB/s |
| B peer reload | 4,194,304 | 0.001694 s | 2.48 GB/s |

The test passed in 274.73 seconds including two server startups, model graph
compilation, requests, graceful detach, and generation-checked destroy. The KV
copy times above are transfer metrics only; they do not include compilation,
scheduling, token generation, or HTTP overhead.

### Historical 4,096-token reference

The earlier M2 one-pool manual run used the same model and a 4,096-token
measured prompt. It is useful as a scale reference but is **not** an M4
measurement:

| Request | TTFT | E2E wall time | KV copy time | KV bytes |
| --- | ---: | ---: | ---: | ---: |
| A cold compute/store | 0.710 s | 1.125 s | 0.008910 s | 67,108,864 stored |
| A self reload | 0.044 s | 0.467 s | 0.005367 s | 67,108,864 loaded |
| B peer reload | 0.045 s | 0.474 s | 0.005124 s | 67,108,864 loaded |

Use the commands in this runbook to collect comparable full-prompt M4 numbers.
Native stack versions, compilation caches, card load, and model revisions can
materially change TTFT and end-to-end latency.

## 7. Stop workers and destroy the exact generation

Stop B and then A so their managers release pins and both workers release their
root-lock ownership:

```bash
kill -TERM "$INSTANCE_B_PID"
wait "$INSTANCE_B_PID"

kill -TERM "$INSTANCE_A_PID"
wait "$INSTANCE_A_PID"
```

Destroy exactly the generation created at the beginning:

```bash
uv run --active --no-sync python scripts/cleanup_shared_kv_demo.py \
  --ctl "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" \
  --pool-name "$SHARED_POOL_NAME" \
  --pool-generation "$SHARED_POOL_GENERATION" \
  | tee /tmp/spyre-shared-kv-m4-cleanup.log
```

Expected result:

```text
destroyed /micro-g3-m4-demo generation <generation>: {"detail": "", "result": "OK"}
```

Finally stop the delegator:

```bash
kill -TERM "$DELEGATOR_PID"
wait "$DELEGATOR_PID"
```

## Troubleshooting

### `POOL_BUSY` during final cleanup

At least one root attachment or resolved child handle is still alive. Confirm
both server process groups have exited, then retry the exact same generation.
Do not unlink shared-memory objects manually.

### `NOT_FOUND`

The named pool does not exist in the shared-memory namespace visible to the
worker. Check the pool name, UID, container IPC namespace, and delegator host.
The worker does not create a missing pool.

### `STALE_GENERATION`

The name was destroyed and reused. Obtain the replacement descriptor from the
lifecycle controller. Never retry an old descriptor against the current
same-name generation.

### `ALREADY_EXISTS` or configuration mismatch during create

The name already identifies a READY generation. Reuse it only when the
effective configuration exactly matches. Otherwise stop its workers, destroy
that exact generation, or choose a different deployment-specific pool name.

### `dbo-opt` cannot load `libdcc.so`

The DeepTools library directory was dropped from `LD_LIBRARY_PATH`. Preserve
the existing native path and prepend the M4 Flex and compatible spyre-comms
directories as shown in the prerequisites.

### Worker attach fails during `INITIALIZING` or `DESTROYING`

Only READY roots are attachable. Let the delegator finish the management
operation or recover the abandoned generation, then use the descriptor it
returns. Workers must not attempt lifecycle recovery themselves.

## Captured artifacts

This procedure writes:

- `/tmp/spyre-shared-kv-m4-delegator.log`;
- `/tmp/spyre-shared-kv-m4-instance-a.log`;
- `/tmp/spyre-shared-kv-m4-instance-b.log`;
- `/tmp/spyre-shared-kv-m4-warmup.log`;
- `/tmp/spyre-shared-kv-m4-a-cold.log`;
- `/tmp/spyre-shared-kv-m4-a-reload.log`;
- `/tmp/spyre-shared-kv-m4-b-reload.log`; and
- `/tmp/spyre-shared-kv-m4-cleanup.log`.

Keep these logs with any reported performance numbers so the model, prompt
size, lifecycle generation, and native-stack errors can be audited later.
