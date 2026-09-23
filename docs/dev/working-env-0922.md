# Working Spyre environment — 2026-09-22

A known-good stack where **both the eager and compiled e2e paths pass**:

```
tests/e2e/test_compile.py::test_basic_llm_inference[model_ref_output0] PASSED
```

Recorded so the environment can be rebuilt after the container is gone. Branch
`spyre-inference-base-0918` across every source repo that has one.

## What made the compiled path work

The compiled path produced **all-NaN logits** until `deeptools` was moved forward 129
commits, from `85f9432623` (2026-08-19) to `22617c9191` (2026-08-28), and the stack was
rebuilt against it. Eager was unaffected the whole time, because `deeptools` owns the
compiler/jobplan path that only the compiled route exercises.

The image's `ibm-deeptools` RPM (`2457.d868b05`) is **older** than both this source rev and
`spyre-rpms.lock`'s `2470.fb2446d` — so the source checkout, not the RPM, is what matters
here. `spyre-comms` was moved forward at the same time (`71d161e` → `217f6cb`).

## Container image

```
image-registry.openshift-image-registry.svc:5000/a6-quantization/torch-spyre-sshd:2026-09-19-kvc-c81130bd
```

RHEL 10.2, Python 3.12.13, torch 2.13.0+cpu. RPMs installed 2026-09-16.

The image tag is pinned, so pulling it again gives this same RPM set. A pod *restart*
reuses the running image; to pick up a different tag the pod must be recreated.

`/opt/ibm/spyre/components.txt` (the image's own manifest):

```
ibm-deeptools:2.0.0-0.main.1+2457.d868b05_0.el10
ibm-senlib-core:2.0.0-0.main.1+280.3869e2e_0.el10
ibm-senlib-dd2:2.0.0-0.main.1+280.3869e2e_0.el10
ibm-flex:2.0.0-0.main.1+576.402f9b2_0.el10
ibm-aiu-toolbox-e2e:2.0.0-0.main.1+29.0826a00_0.el10
```

Full installed set (`rpm -qa | grep ^ibm-`):

```
ibm-aiu-toolbox-e2e-2.0.0-0.main.1+29.0826a00_0.el10
ibm-deeptools-2.0.0-0.main.1+2457.d868b05_0.el10
ibm-deeptools-devel-2.0.0-0.main.1+2457.d868b05_0.el10
ibm-flex-2.0.0-0.main.1+576.402f9b2_0.el10
ibm-flex-devel-2.0.0-0.main.1+576.402f9b2_0.el10
ibm-libaiupti-2.0.0-0.main.1+30.ef4e622_0.el10
ibm-senlib-core-2.0.0-0.main.1+280.3869e2e_0.el10
ibm-senlib-dd2-2.0.0-0.main.1+280.3869e2e_0.el10
ibm-senlib-headers-2.0.0-0.main.1+280.3869e2e_0.el10
ibm-spyre-comms-1.0.0-0.main.1+156.906a4e6_0.el10
ibm-spyre-comms-devel-1.0.0-0.main.1+156.906a4e6_0.el10
ibm-spyre-comms-test-1.0.0-0.main.1+156.906a4e6_0.el10
```

Note these do **not** all match `spyre-rpms.lock`: `ibm-deeptools` is older
(`2457.d868b05` vs `2470.fb2446d`), `ibm-flex` newer (`576.402f9b2` vs `570.c11584a`),
`ibm-spyre-comms` newer (`156.906a4e6` vs `154.ce55e90`). The stack works anyway because
flex, deeptools, libaiupti and spyre-comms are built from source and take precedence via
`SENTIENT_BASE_INSTALL_DIR`; only senlib is consumed from `/opt/ibm/spyre`.

## Source revisions (`$DTI_PROJECT_ROOT` = `~/dt-inductor`)

All commits below are reachable from their upstream remote by SHA (verified by fetching
each into a fresh clone), so none of this depends on a local branch or this pod.

| Repo | Commit | Date | Remote |
|---|---|---|---|
| `deeptools` | `22617c9191374219ad7c8773b514c42211290bd3` | 2026-08-28 | `git@github.ibm.com:ai-chip-toolchain/deeptools.git` |
| `flex` | `eeafb3d8a1bd6f15f40691b2df14a2734d746f2f` | 2026-09-18 | `git@github.ibm.com:ai-chip-toolchain/flex.git` |
| `libaiupti` | `ef4e6226d0642f16e6a3687d0bf5b81c3dd79e4d` | 2026-09-14 | `git@github.ibm.com:ai-chip-toolchain/libaiupti.git` |
| `spyre-comms` | `217f6cb0a0cdaaca3ae3a6f2831e0738b0bf7243` | 2026-08-31 | `git@github.ibm.com:ai-chip-toolchain/spyre-comms.git` |
| `torch-spyre` | `27081f241dbd5cc5292b9cf4e4f834bdc6b19ff1` | 2026-09-18 | `https://github.com/torch-spyre/torch-spyre` |
| `torch-spyre-docs` | `3095d566dd9cf84b4ca2a4b35e517c4091008f1c` | 2026-08-25 | `git@github.ibm.com:ai-foundation/torch-spyre-docs.git` |
| `llvm-project` | `e9846648fd6183ee6d8cbdb4502213fcf902a211` (`llvmorg-22.1.3`) | 2026-04-06 | `https://github.com/llvm/llvm-project` |
| `pytorch` | `911aa98c48b7a80a708dc168b1c418c7ae0bb9de` (`release/2.10`) | 2026-02-10 | `git@github.com:pytorch/pytorch.git` |
| `sendnn` | `8cc4fe436f161f72e9bb4b76b8252d9bea981da6` | 2026-06-29 | `git@github.ibm.com:ai-chip-toolchain/sendnn.git` |
| `senbfcc` | `93c60dff46d933378c7833a6132690f4af3b94dc` | 2026-02-11 | `git@github.ibm.com:ai-chip-toolchain/senbfcc.git` |

On this pod every repo except `llvm-project` (detached), `pytorch`, `sendnn`, `senbfcc` and
`torch-spyre-docs` sits on a local branch named `spyre-inference-base-0918`. Those branches
are labels only — they carry no commits of their own, so checking out the SHA is equivalent.
`torch-spyre`'s local `origin` is a personal fork; the upstream URL above serves the same
commit.

`deeptools` submodules (set by `git submodule update --init --recursive`):

| Submodule path | Commit | URL |
|---|---|---|
| `dataflow-scheduler` | `9a9d29095cad7cca4970628686d189144b34a812` | `https://github.com/torch-spyre/dataflow-scheduler.git` |
| `dataflow-scheduler/external/dataflow-scheduler-dialects` | `4d66ebb3253de0e112a206e00e6e46e413cf5b33` | `https://github.com/torch-spyre/dataflow-scheduler-mlir-dialects.git` |
| `dataflow-scheduler/external/ktir-mlir-frontend` | `fc21b6c2fa380172d43bc38fa296fcb9bcdaa83a` | `https://github.com/torch-spyre/ktir-mlir-frontend.git` |

`libaiupti` submodule:

| Submodule path | Commit | URL |
|---|---|---|
| `src/aiupti/common` | `9962b3b9ba195e77c168b24249dcd44de3d820da` | `git@github.ibm.com:ai-chip-toolchain/spyre_common_headers` |

Note the dialects submodule's path and repo name differ (`dataflow-scheduler-dialects`
vs `dataflow-scheduler-mlir-dialects`). `git submodule update --init --recursive` reads
all of these from `.gitmodules`, so the URLs are only needed to fetch a pin by hand.

`spyre-inference`: branch `spyre-inference-base-0918`, vLLM pinned `v0.28.0`,
`torch-spyre` sourced from `../dt-inductor/torch-spyre` (editable).

## Rebuild procedure

```bash
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh
```

Build order matters — each links against the previous:

```bash
bash $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/build-deeptools.sh    # ~25 min
bash $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/build-flex.sh         # ~5 min
bash $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/build-libaiupti.sh    # ~1 min
bash $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/build-torch-spyre.sh  # ~2 min
```

`build-deeptools.sh` expects a prebuilt LLVM at `$LLVM_PROJ_BUILD`
(`build-llvm-deeptools.sh`, several hours) and passes `MANAGE_LLVM=0`.

`build-flex.sh` and `build-libaiupti.sh` start with `rm -rf` on their install prefix
(`sentient/runtime`, 513 MB). Back it up first if the current one is working.

Then, in the `spyre-inference` checkout:

```bash
uv sync --group dev
bash $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/build-torch-spyre.sh   # re-run, see below
```

## Gotchas

**`uv sync` silently downgrades torch-spyre.** It rebuilds without
`--no-build-isolation`, shrinking `_C.so` from ~83 MB to ~77 MB and dropping the local
flex/libaiupti headers. Always re-run `build-torch-spyre.sh` afterwards, or sync with
`--no-install-package torch-spyre --inexact`.

**The venv has no `pytest`/`pip` console script.** `uv run pytest` can fall through to
`~/.local/bin/pytest` under a different interpreter and fail on `spyre_testing_plugin`.
`python -m pytest` after sourcing `dev-env.sh` always works.

**`UV_PROJECT_ENVIRONMENT`** is set by `dev-env.sh` to `$DTI_PROJECT_ROOT/.venv`, so
`uv sync` in `spyre-inference` installs into the dt-inductor venv, not a local `.venv`.

**One process at a time on the device.** Never run two Spyre-backed commands concurrently.

## Verification

```bash
source $DTI_PROJECT_ROOT/torch-spyre-docs/scripts/dev-env.sh

EAGER=1 python scripts/probes/basic_llm_inference.py   # eager
EAGER=0 python scripts/probes/basic_llm_inference.py   # compiled

uv run --no-sync pytest \
  'tests/e2e/test_compile.py::test_basic_llm_inference[model_ref_output0]' \
  -m "not upstream" -v
```

Expected on this stack: both probes print `matches_ref: True`, and the test passes.
Device smoke numbers: fp32 add 1.2e-07, fp16 matmul 0.0625, 1 device.

Last verified 2026-09-23 on the image above: `1 passed in 97.96s`. The test passes with
either the 83 MB (`build-torch-spyre.sh`) or 77 MB (`uv sync`) `_C.so`, so the build-isolation
difference is an inconsistency to be aware of rather than a correctness problem.
