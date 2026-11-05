# Persistent KV offload pools

Spyre KV offload uses a node-local Flex composite pool containing shared
metadata and one fixed host-memory data child. A `shared-pool-delegator`
process owns pool management. vLLM workers only attach to an exact generation,
resolve its configured child, and detach during shutdown.

The pool remains available with zero attachments. Stopping the delegator or the
last worker does not remove it. Only an explicit, generation-checked destroy
removes the pool.

All delegator and worker processes must run under the same numeric host UID,
share the same `/dev/shm` namespace, and use compatible Flex and torch-spyre M4
builds.

## Start the delegator

Set the paths for the three independently built projects:

```bash
export DTI_PROJECT_ROOT=/home/yzhu/dt-inductor
export FLEX_M4="$DTI_PROJECT_ROOT/flex-worktrees/persistent-shared-kv-pool-lifecycle"
export TORCH_SPYRE_M4="$DTI_PROJECT_ROOT/torch-spyre/.worktrees/kvc-offload-m4"
export DELEGATOR_M4="$DTI_PROJECT_ROOT/shared-pool-delegator/.worktrees/kvc-offload-m4"
export DELEGATOR_SOCKET=/tmp/shared-pool-delegator.sock
export LD_LIBRARY_PATH="$FLEX_M4/build:${LD_LIBRARY_PATH-}"
export PYTHONPATH="$TORCH_SPYRE_M4:${PYTHONPATH-}"
```

Start the long-running node-local service and wait until it is ready:

```bash
"$DELEGATOR_M4/build/shared-pool-delegator" \
  --socket-path "$DELEGATOR_SOCKET" --socket-mode 0600 &
export DELEGATOR_PID=$!

until "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" health >/dev/null 2>&1; do
  sleep 0.1
done
```

Kubernetes can run the same binary as a node-local daemon. systemd integration
is not required by this initial scope.

## Create the whole pool before workers start

The demo helper builds the complete metadata-plus-data configuration and calls
`shared-poolctl create` once:

```bash
POOL_JSON=$(uv run --active --no-sync python \
  scripts/shared_kv_two_instance_demo.py prepare-pool \
  --ctl "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" \
  --pool-name /micro-g3-demo \
  --data-pool-name host-0 \
  --model ibm-ai-platform/micro-g3.3-8b-instruct-1b \
  --cpu-bytes-to-use 536870912 \
  --slot-bytes 262144)

export SHARED_POOL_NAME=$(jq -r .poolName <<<"$POOL_JSON")
export SHARED_POOL_GENERATION=$(jq -r .poolGeneration <<<"$POOL_JSON")
export SHARED_POOL_DATA_NAME=$(jq -r .dataPoolName <<<"$POOL_JSON")
```

The opaque generation is 16 bytes represented as 32 hexadecimal characters.
Keep it with the deployment state. If create returns an existing name, the
helper accepts it only when its effective configuration exactly matches the
requested model, KV dtype, block size, tensor-parallel size, slot count, and
slot size.

For another model deployment on the same node, use another root name. Pools for
different models must not share metadata or data children.

## Worker connector settings

Both replicas receive the same three identity fields in
`kv_connector_extra_config`:

```json
{
  "spec_name": "SpyreSharedOffloadingSpec",
  "pool_name": "/micro-g3-demo",
  "pool_generation": "0123456789abcdef0123456789abcdef",
  "data_pool_name": "host-0",
  "cpu_bytes_to_use": 536870912
}
```

`pool_name`, `pool_generation`, and `data_pool_name` identify a single
delegator-created resource. A missing or stale identity fails attachment; a
worker never creates a replacement.

For launchers that cannot populate `kv_connector_extra_config`, the equivalent
environment fallbacks are `SPYRE_KV_POOL_NAME`, `SPYRE_KV_POOL_GENERATION`, and
`SPYRE_KV_DATA_POOL_NAME`. An explicitly configured field always takes
precedence. If that explicit value is malformed, startup fails instead of
silently using the environment value.

For the demo, start one vLLM process per card after exporting the identity:

```bash
SPYRE_DEVICE_A=0 scripts/start_shared_kv_instance_a.sh &
export INSTANCE_A_PID=$!

SPYRE_DEVICE_B=1 scripts/start_shared_kv_instance_b.sh &
export INSTANCE_B_PID=$!
```

`SpyreOffloadingSpec` is the M1 local-policy path. Give each deployment its own
private pool generation so its pages are never interpreted by another model or
deployment. `SpyreSharedOffloadingSpec` is the M2 shared-policy path: replicas
of one compatible model deliberately receive the same generation, allowing a
request computed by replica A to be reloaded by replica B. Scheduling new
requests between replicas is outside pool lifecycle management; active requests
are not migrated.

## Verify busy destroy, detach, and exact destroy

Destroy while either worker is attached must fail with `POOL_BUSY`; this proves
the root's shared lock lease is protecting the generation:

```bash
"$DELEGATOR_M4/build/shared-poolctl" --socket "$DELEGATOR_SOCKET" destroy \
  --name "$SHARED_POOL_NAME" --generation "$SHARED_POOL_GENERATION"
test $? -ne 0
```

Stop both workers gracefully so their offload managers and workers release all
metadata pins, child handles, and root attachments:

```bash
kill -TERM "$INSTANCE_A_PID" "$INSTANCE_B_PID"
wait "$INSTANCE_A_PID" "$INSTANCE_B_PID"
```

Then destroy the exact generation:

```bash
uv run --active --no-sync python scripts/cleanup_shared_kv_demo.py \
  --ctl "$DELEGATOR_M4/build/shared-poolctl" \
  --socket "$DELEGATOR_SOCKET" \
  --pool-name "$SHARED_POOL_NAME" \
  --pool-generation "$SHARED_POOL_GENERATION"
```

Never unlink `/dev/shm` objects directly. `POOL_BUSY` is not cleanup success,
and a stale generation must not be retried against a replacement pool.
