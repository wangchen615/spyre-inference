# Spyre Shared KV Two-Instance Manual Demo

This procedure starts two independent vLLM instances on separate Spyre devices
and demonstrates that they reload the same 4,096-token prompt from one shared
host-memory KV pool. Prefix caching is disabled, so the measured reloads cannot
be satisfied by device-resident prefix KV.

## Prerequisites

- Use the `kvc-offload-m2` worktree with its local M2 torch-spyre installation.
- Reserve two Spyre devices. The launchers default to device 0 for instance A
  and device 1 for instance B.
- Keep both servers dedicated to this run.

The launchers use the same metadata directory and pool families, TP1, the
`uni` executor, `PYTHONHASHSEED=0`, and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`. They also prepend this worktree to
`PYTHONPATH`, ensuring vLLM loads the M2 plugin rather than another checkout.

## Start the servers

In terminal A:

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
bash scripts/start_shared_kv_instance_a.sh
```

In terminal B:

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
bash scripts/start_shared_kv_instance_b.sh
```

Wait for both terminals to report `Application startup complete`. To select
different physical devices, set the instance-specific variables instead of
the shared `SPYRE_DEVICES` variable:

```bash
SPYRE_DEVICE_A=2 bash scripts/start_shared_kv_instance_a.sh
SPYRE_DEVICE_B=3 bash scripts/start_shared_kv_instance_b.sh
```

## Run the demo

In a third terminal:

```bash
cd /home/yzhu/dt-inductor/spyre-inference-kvc-offload-m2
uv run --no-sync python -u scripts/shared_kv_two_instance_demo.py
```

The demo performs this sequence:

1. A computes a distinct 4,096-token warmup prompt and generates 16 tokens.
2. B computes another distinct 4,096-token warmup prompt and generates 16
   tokens.
3. A computes the fresh measured prompt and publishes its KV blocks.
4. A reloads its own blocks from the shared host pool.
5. B reloads A's blocks from the same pool.

For each request, the script checks vLLM's prompt-source and KV-transfer
counters. The warmups and first measured A request must report local compute;
both reloads must report 4,096 externally transferred tokens and zero locally
computed tokens. Store and reload byte counts must match. The generated token
IDs and UTF-8 text bytes must also be identical across A's initial generation,
A's self-reload, and B's peer reload.

## Recorded result

The run used model `ibm-ai-platform/micro-g3.3-8b-instruct-1b` with vLLM
`0.28.0`. Instance A used device 0 at `0000:1d:00.0`; instance B used device 1
at `0000:1e:00.0`.

| Request | Wall time | KV copy time | KV bytes |
| --- | ---: | ---: | ---: |
| A warmup | 136.527 s | — | — |
| B warmup | 119.868 s | — | — |
| A measured compute/store | 30.994 s | 3.164556 s | 67,108,864 stored |
| A self shared-pool reload | 21.969 s | 3.177898 s | 67,108,864 loaded |
| B peer shared-pool reload | 21.953 s | 3.144324 s | 67,108,864 loaded |

The two measured reload wall times differed by 0.016 seconds. All three
measured generations returned these 16 token IDs:

```text
[203, 203, 1397, 44, 10720, 3303, 518, 322, 15872, 2582, 11385, 39171, 461, 7624, 8019, 26322]
```

They decoded byte-identically to:

```text
"\n\nResponse: Summarize the recurring symptoms and give three concrete"
```

No M1 comparison or performance threshold was used. These timings demonstrate
that the warmed instances reload from the same shared pool with comparable
wall time; they are not an M1-versus-M2 benchmark.
