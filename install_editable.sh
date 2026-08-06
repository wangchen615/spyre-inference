#!/usr/bin/env bash
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
#
# ---------------------------------------------------------------------------
# Install spyre-inference as an EDITABLE package into an existing hand-built
# Spyre stack venv (default: /home/yuezhu/dt-inductor/.venv).
#
# WHY NOT `uv sync --frozen` (the documented install in
# docs/getting_started/installation.md)?
#
#   1. `uv sync` is project-managed: it creates/owns `spyre-inference/.venv`.
#      A hand-built stack venv living *outside* the repo is not that venv, and
#      `uv sync` will not adopt it.
#   2. `uv sync` resolves the whole dependency graph from pyproject.toml, which
#      pins `torch-spyre` to an upstream git rev. Syncing therefore *reinstalls*
#      torch-spyre from that rev, clobbering the local editable build in
#      ../torch-spyre that the hand-built stack depends on.
#   3. Same hazard for plain `uv run`: it re-syncs before every invocation and
#      silently reverts hand-installed local deps (see CLAUDE.md). Any command
#      run against this venv must use `uv run --no-sync ...` or the venv's
#      python directly.
#
# So we drive `uv pip install` against VIRTUAL_ENV instead, with `--no-deps` on
# everything that would otherwise re-resolve torch/torch-spyre, and then install
# the genuinely-missing leaf dependencies explicitly.
#
# This script never claims the Spyre accelerator: it runs no pytest, no
# inference, nothing that opens the device. Spyre admits exactly one process at
# a time, so it prints suggested next commands rather than running them.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_VENV="/home/yuezhu/dt-inductor/.venv"

# Fallback pins, used only if they cannot be parsed out of pyproject.toml.
FALLBACK_VLLM_REV="v0.26.0"
FALLBACK_HF_ADAPTERS_REV="a1e0b01e01b77d7b995c4ecd11ecaf30a19d7ba4"

VENV="${DEFAULT_VENV}"
REPO="${DEFAULT_REPO}"
SKIP_VLLM=0
FORCE_VLLM=0
INSTALL_TEST_DEPS=1

RED=""; GREEN=""; YELLOW=""; BOLD=""; RESET=""
if [[ -t 1 ]]; then
    RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'
    BOLD=$'\033[1m'; RESET=$'\033[0m'
fi

info()  { printf '%s==>%s %s\n' "${BOLD}" "${RESET}" "$*"; }
ok()    { printf '%s  ok%s  %s\n' "${GREEN}" "${RESET}" "$*"; }
warn()  { printf '%swarn%s  %s\n' "${YELLOW}" "${RESET}" "$*" >&2; }
die()   { printf '%sfail%s  %s\n' "${RED}" "${RESET}" "$*" >&2; exit 1; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Installs spyre-inference (and its test plugin) as EDITABLE packages into an
existing hand-built Spyre stack venv, without disturbing the local editable
torch-spyre / torch install already present there.

Options:
  --venv PATH        Target virtualenv (default: ${DEFAULT_VENV})
  --repo PATH        spyre-inference checkout (default: ${DEFAULT_REPO})
  --skip-vllm        Never build/install vLLM (fast path when it is present)
  --force-vllm       Rebuild vLLM from source even if already installed
  --no-test-deps     Skip the pytest / test-only dependency group
  -h, --help         Show this help

Idempotent: safe to re-run. vLLM and hf-adapters-spyre are skipped when already
installed; the editable installs are cheap and always refreshed.

Runs no Spyre-hardware workload (Spyre admits one process at a time).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)  [[ $# -ge 2 ]] || die "--venv requires a PATH"; VENV="$2"; shift 2 ;;
        --repo)  [[ $# -ge 2 ]] || die "--repo requires a PATH"; REPO="$2"; shift 2 ;;
        --skip-vllm)     SKIP_VLLM=1; shift ;;
        --force-vllm)    FORCE_VLLM=1; shift ;;
        --no-test-deps)  INSTALL_TEST_DEPS=0; shift ;;
        -h|--help)       usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

if (( SKIP_VLLM && FORCE_VLLM )); then
    die "--skip-vllm and --force-vllm are mutually exclusive"
fi

# --------------------------------------------------------------------------
# Pre-flight
# --------------------------------------------------------------------------
info "Pre-flight checks"

[[ -d "${VENV}" ]] || die "venv not found: ${VENV} (pass --venv PATH)"
PY="${VENV}/bin/python"
[[ -x "${PY}" ]] || die "no python interpreter at ${PY} — is ${VENV} really a virtualenv?"
VENV="$(cd -- "${VENV}" && pwd)"
PY="${VENV}/bin/python"

command -v uv >/dev/null 2>&1 || die "\`uv\` not found on PATH; install it (https://docs.astral.sh/uv/)"

[[ -d "${REPO}" ]] || die "repo not found: ${REPO} (pass --repo PATH)"
REPO="$(cd -- "${REPO}" && pwd)"
PYPROJECT="${REPO}/pyproject.toml"
[[ -f "${PYPROJECT}" ]] || die "no pyproject.toml in ${REPO} — not a spyre-inference checkout?"

export VIRTUAL_ENV="${VENV}"

PY_VERSION="$("${PY}" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
ok "venv ${VENV} (python ${PY_VERSION})"
ok "repo ${REPO}"
ok "uv  $(uv --version 2>/dev/null | head -1)"

# torch must already be there; capture the version so post-install can prove it
# was not swapped out from under us.
TORCH_VERSION_BEFORE="$("${PY}" -c 'import torch; print(torch.__version__)' 2>/dev/null)" \
    || die "torch is not importable in ${VENV}; this script expects a pre-built stack venv"
ok "torch ${TORCH_VERSION_BEFORE}"

# torch-spyre must be an *editable* install pointing at the local checkout.
# `pip`/`uv` record that in <dist-info>/direct_url.json as
# {"url": "file:///...", "dir_info": {"editable": true}}.
read -r TS_EDITABLE TS_URL TS_FILE <<<"$("${PY}" - <<'PYEOF'
import importlib.metadata as md, json, pathlib
editable, url, mod = "no", "-", "-"
try:
    dist = md.distribution("torch-spyre")
    raw = dist.read_text("direct_url.json")
    if raw:
        info = json.loads(raw)
        url = info.get("url", "-")
        editable = "yes" if info.get("dir_info", {}).get("editable") else "no"
except md.PackageNotFoundError:
    editable = "missing"
try:
    # torch first: importing torch_spyre bare trips a circular import through
    # torch's backend autoload (see verification step (b) below).
    import torch  # noqa: F401
    import torch_spyre
    mod = str(pathlib.Path(torch_spyre.__file__).resolve())
except Exception:
    pass
print(editable, url, mod)
PYEOF
)"

case "${TS_EDITABLE}" in
    yes)
        ok "torch-spyre is editable from ${TS_URL}"
        ;;
    missing)
        warn "torch-spyre is NOT INSTALLED in ${VENV}."
        warn "Expected a hand-built editable install (e.g. \`uv pip install -e ../torch-spyre\`)."
        warn "Continuing — but nothing here will install it for you, and \`import"
        warn "spyre_inference\` will fail without it."
        ;;
    *)
        warn "torch-spyre is installed but NOT EDITABLE (direct_url: ${TS_URL})."
        warn "That usually means it came from the pinned upstream git rev rather than"
        warn "the local ./torch-spyre checkout — i.e. a previous \`uv sync\`/\`uv run\`"
        warn "clobbered the local build. Reinstall with:"
        warn "    VIRTUAL_ENV=${VENV} uv pip install --no-deps -e <path-to>/torch-spyre"
        warn "Continuing anyway (this may be a deliberate fresh setup)."
        ;;
esac
TS_FILE_BEFORE="${TS_FILE}"

# --------------------------------------------------------------------------
# Pinned revs, parsed from pyproject.toml so this script does not rot when the
# pins bump. Falls back to the literals verified at authoring time.
# --------------------------------------------------------------------------
parse_rev() {
    # $1 = uv source key (e.g. "hf-adapters-spyre"), $2 = fallback
    local key="$1" fallback="$2" rev
    rev="$("${PY}" - "${PYPROJECT}" "${key}" <<'PYEOF'
import re, sys, tomllib
path, key = sys.argv[1], sys.argv[2]
with open(path, "rb") as fh:
    data = tomllib.load(fh)
src = data.get("tool", {}).get("uv", {}).get("sources", {}).get(key)
entries = src if isinstance(src, list) else [src] if src else []
for entry in entries:
    if isinstance(entry, dict) and entry.get("git"):
        rev = entry.get("rev") or entry.get("tag") or entry.get("branch")
        if rev and re.fullmatch(r"[A-Za-z0-9._/-]+", rev):
            print(rev)
            break
PYEOF
)" || rev=""
    printf '%s' "${rev:-${fallback}}"
}

VLLM_REV="$(parse_rev vllm "${FALLBACK_VLLM_REV}")"
HF_ADAPTERS_REV="$(parse_rev hf-adapters-spyre "${FALLBACK_HF_ADAPTERS_REV}")"
ok "pins from pyproject.toml: vllm=${VLLM_REV} hf-adapters=${HF_ADAPTERS_REV}"

has_dist() { "${PY}" -c "import importlib.metadata as m; m.distribution('$1')" >/dev/null 2>&1; }
dist_version() { "${PY}" -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || echo "-"; }

# --------------------------------------------------------------------------
# 1. vLLM — the slow step (built from source with the empty backend).
# --------------------------------------------------------------------------
info "Step 1/5: vLLM"
if (( SKIP_VLLM )); then
    if has_dist vllm; then
        ok "--skip-vllm: keeping installed vllm $(dist_version vllm)"
    else
        warn "--skip-vllm requested but vllm is NOT installed; later verification will fail"
    fi
elif has_dist vllm && (( ! FORCE_VLLM )); then
    ok "vllm $(dist_version vllm) already installed (use --force-vllm to rebuild)"
else
    info "building vllm @ ${VLLM_REV} from source (VLLM_TARGET_DEVICE=empty) — this takes minutes"
    # Build isolation is kept ON, with the build deps supplied explicitly via
    # UV_EXTRA_BUILD_DEPENDENCIES_VLLM. `--no-build-isolation-package vllm`
    # instead fails with ModuleNotFoundError: No module named 'setuptools_rust',
    # and would also pollute the runtime venv with build tooling.
    VLLM_TARGET_DEVICE=empty \
    CMAKE_ARGS="--fresh" \
    UV_EXTRA_BUILD_DEPENDENCIES_VLLM='["torch=='"${TORCH_VERSION_BEFORE%%+*}"'","setuptools_rust","setuptools>=82","setuptools_scm>=8","wheel","cmake","ninja","packaging","jinja2","regex"]' \
    uv pip install "vllm @ git+https://github.com/vllm-project/vllm@${VLLM_REV}" \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match
    ok "vllm $(dist_version vllm) installed"
fi

# --------------------------------------------------------------------------
# 2. hf-adapters-spyre — `--no-deps` so it cannot drag in a conflicting torch.
#    Its real dependencies are installed in step 5.
# --------------------------------------------------------------------------
info "Step 2/5: hf-adapters-spyre"
if has_dist hf-adapters-spyre; then
    ok "hf-adapters-spyre $(dist_version hf-adapters-spyre) already installed"
else
    uv pip install --no-deps \
        "hf-adapters-spyre @ git+https://github.com/torch-spyre/hf-adapters.git@${HF_ADAPTERS_REV}"
    ok "hf-adapters-spyre $(dist_version hf-adapters-spyre) installed"
fi

# --------------------------------------------------------------------------
# 3. spyre-inference, EDITABLE. The whole point of this script.
#    `--no-deps` keeps uv from re-resolving torch / torch-spyre / vllm.
# --------------------------------------------------------------------------
info "Step 3/5: spyre-inference (editable)"
uv pip install --no-deps -e "${REPO}"
ok "spyre-inference $(dist_version spyre-inference) installed editable from ${REPO}"

# --------------------------------------------------------------------------
# 4. spyre-testing-plugin, EDITABLE — pytest needs it for collection/markers.
# --------------------------------------------------------------------------
info "Step 4/5: spyre-testing-plugin (editable)"
PLUGIN_DIR="${REPO}/tests/plugin"
if [[ -f "${PLUGIN_DIR}/pyproject.toml" ]]; then
    uv pip install --no-deps -e "${PLUGIN_DIR}"
    ok "spyre-testing-plugin $(dist_version spyre-testing-plugin) installed editable from ${PLUGIN_DIR}"
else
    warn "no pyproject.toml at ${PLUGIN_DIR}; skipping test plugin"
fi

# --------------------------------------------------------------------------
# 5. Fill in the deps that `--no-deps` skipped above. None of these touch torch.
# --------------------------------------------------------------------------
info "Step 5/5: leaf dependencies"
uv pip install accelerate sentence-transformers
ok "runtime deps (accelerate, sentence-transformers)"

if (( INSTALL_TEST_DEPS )); then
    uv pip install \
        "buildkite-test-collector==0.1.9" \
        "datasets>=3.3.0,<=3.6.0" \
        pytest-asyncio pytest-cov pytest-forked pytest-rerunfailures \
        pytest-shard pytest-timeout tblib
    ok "test deps"
else
    ok "--no-test-deps: skipped test dependency group"
fi

# --------------------------------------------------------------------------
# Post-install verification
# --------------------------------------------------------------------------
info "Verification"

# (a) torch must be byte-identical in version to what we captured pre-flight.
#     A change here means something re-resolved the graph and clobbered the
#     hand-built stack — the exact hazard this script exists to avoid.
TORCH_VERSION_AFTER="$("${PY}" -c 'import torch; print(torch.__version__)' 2>/dev/null)" \
    || die "torch is no longer importable — the install clobbered the stack venv"
if [[ "${TORCH_VERSION_AFTER}" != "${TORCH_VERSION_BEFORE}" ]]; then
    die "torch version changed: ${TORCH_VERSION_BEFORE} -> ${TORCH_VERSION_AFTER} (stack clobbered)"
fi
ok "(a) torch unchanged at ${TORCH_VERSION_AFTER}"

# (b) torch_spyre must still resolve to the local checkout, not site-packages.
#     `import torch` MUST come first: torch_spyre imports torch, whose backend
#     autoload then calls back into the still-initializing torch_spyre module.
#     Importing torch_spyre bare raises "partially initialized module
#     'torch_spyre' has no attribute '_autoload'" — a circular-import artifact
#     that looks exactly like a broken install but is purely import ordering.
if [[ "${TS_EDITABLE}" == "yes" ]]; then
    TS_FILE_AFTER="$("${PY}" -c 'import torch, pathlib, torch_spyre; print(pathlib.Path(torch_spyre.__file__).resolve())' 2>/dev/null)" \
        || die "torch_spyre is not importable (with torch imported first) — check the local editable build at ${TS_URL}"
    case "${TS_FILE_AFTER}" in
        */site-packages/*)
            die "torch_spyre now resolves inside site-packages (${TS_FILE_AFTER}); the local editable build was clobbered" ;;
    esac
    if [[ -n "${TS_FILE_BEFORE}" && "${TS_FILE_BEFORE}" != "-" && "${TS_FILE_AFTER}" != "${TS_FILE_BEFORE}" ]]; then
        die "torch_spyre moved: ${TS_FILE_BEFORE} -> ${TS_FILE_AFTER}"
    fi
    ok "(b) torch_spyre still local: ${TS_FILE_AFTER}"
else
    warn "(b) skipped: torch-spyre was not editable at pre-flight"
fi

# (c) spyre_inference must resolve INSIDE the repo working tree.
#     Run from /tmp: with cwd == repo root, `import spyre_inference` would pick
#     up the source tree via sys.path[0] even for a NON-editable install, which
#     would make a broken install look fine. Leaving the repo removes that mask.
# `tail -1`: importing spyre_inference triggers vLLM's plugin-registration
# banner on STDOUT, which would otherwise be captured as part of the path.
SI_FILE="$(cd /tmp && "${PY}" -c 'import pathlib, spyre_inference; print(pathlib.Path(spyre_inference.__file__).resolve())' 2>/dev/null | tail -1)" \
    || die "cannot import spyre_inference from outside the repo — editable install did not take"
case "${SI_FILE}" in
    "${REPO}"/*) ok "(c) spyre_inference editable: ${SI_FILE}" ;;
    *) die "spyre_inference resolves outside the repo (${SI_FILE}); expected under ${REPO} — not an editable install" ;;
esac

# (d) The vLLM platform plugin must resolve to our platform class.
# `tail -1` for the same reason as (c): vLLM logs plugin registration to stdout.
PLATFORM="$(cd /tmp && "${PY}" -c '
from vllm.platforms import current_platform
cls = type(current_platform)
print(f"{cls.__module__}.{cls.__qualname__}")
' 2>/dev/null | tail -1)" || die "failed to resolve vllm.platforms.current_platform"
EXPECTED_PLATFORM="spyre_inference.platform.TorchSpyrePlatform"
if [[ "${PLATFORM}" != "${EXPECTED_PLATFORM}" ]]; then
    die "current_platform is ${PLATFORM}, expected ${EXPECTED_PLATFORM} (plugin entry point not registered?)"
fi
ok "(d) vllm current_platform = ${PLATFORM}"

# (e) `uv pip check`. Exactly three incompatibilities are expected and benign in
#     this tree; anything else is a real regression.
#       - setuptools>=82 vs vllm's <81.0.0 cap: overridden on purpose for CVEs
#         (see override-dependencies in pyproject.toml).
#       - torch-sendnn requires torch<=2.10: stale upstream pin, tree is on 2.13.
#       - missing triton: overridden out via `triton; sys_platform == 'never'`.
# `uv pip check` styles its output with ANSI escapes even when piped, so the
# lines do NOT literally start with "The package" — strip the escapes first or
# every grep below silently matches nothing and the check passes vacuously.
CHECK_OUT="$(cd /tmp && uv pip check 2>&1 || true)"
CHECK_PLAIN="$(printf '%s\n' "${CHECK_OUT}" | sed -E 's/\x1b\[[0-9;]*[A-Za-z]//g')"
FOUND_LINES="$(printf '%s\n' "${CHECK_PLAIN}" | grep -E '^The package ' || true)"
UNEXPECTED="$(printf '%s\n' "${FOUND_LINES}" \
    | grep -vE 'requires `setuptools>=[0-9.]+,<81\.0\.0' \
    | grep -vE '`torch-sendnn` requires `torch' \
    | grep -vE 'requires `triton ' || true)"
FOUND_HITS="$(printf '%s\n' "${FOUND_LINES}" | grep -cE '^The package ' || true)"
# Cross-check against uv's own "Found N incompatibilities" tally: if the two
# disagree, our parsing is broken and a real regression could slip through.
REPORTED_HITS="$(printf '%s\n' "${CHECK_PLAIN}" \
    | sed -nE 's/^Found ([0-9]+) incompatibilit.*/\1/p' | tail -1)"
REPORTED_HITS="${REPORTED_HITS:-0}"
if [[ -n "${UNEXPECTED}" ]]; then
    printf '%s\n' "${CHECK_PLAIN}" >&2
    die "unexpected dependency incompatibilities:"$'\n'"${UNEXPECTED}"
fi
if [[ "${FOUND_HITS}" != "${REPORTED_HITS}" ]]; then
    printf '%s\n' "${CHECK_PLAIN}" >&2
    die "parsed ${FOUND_HITS} incompatibility line(s) but uv reported ${REPORTED_HITS} — parser is out of date"
fi
ok "(e) uv pip check: ${FOUND_HITS} incompatibility/ies, all known-benign"
printf '%s\n' "${FOUND_LINES}" | sed 's/^/       benign: /' || true

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
# Read direct_url.json out of the installed dist-info explicitly. Going through
# importlib.metadata.distribution() is unreliable here: for a package whose
# import name also exists as a directory in cwd (spyre_inference in the repo
# root), it can resolve to the source tree, which has no direct_url.json — so an
# editable install reports as non-editable. Run from /tmp for the same reason.
editable_src() {
    (cd /tmp && "${PY}" - "$1" <<'PYEOF'
import json, pathlib, sys, sysconfig
name = sys.argv[1].replace("-", "_")
site = pathlib.Path(sysconfig.get_paths()["purelib"])
for dist in sorted(site.glob(f"{name}-*.dist-info")):
    durl = dist / "direct_url.json"
    if not durl.is_file():
        continue
    try:
        info = json.loads(durl.read_text())
    except Exception:
        continue
    if info.get("dir_info", {}).get("editable"):
        print(info.get("url", "").removeprefix("file://"))
        break
else:
    print("-")
PYEOF
)
}

printf '\n%sInstalled%s\n' "${BOLD}" "${RESET}"
for pkg in torch torch-spyre torch-sendnn vllm hf-adapters-spyre spyre-inference spyre-testing-plugin; do
    if has_dist "${pkg}"; then
        src="$(editable_src "${pkg}")"
        if [[ -n "${src}" && "${src}" != "-" ]]; then
            printf '  %-22s %-18s %seditable%s <- %s\n' "${pkg}" "$(dist_version "${pkg}")" \
                "${GREEN}" "${RESET}" "${src}"
        else
            printf '  %-22s %s\n' "${pkg}" "$(dist_version "${pkg}")"
        fi
    else
        printf '  %-22s %s(not installed)%s\n' "${pkg}" "${YELLOW}" "${RESET}"
    fi
done

cat <<EOF

${BOLD}${YELLOW}WARNING — never use a bare \`uv run\` / \`uv sync\` against this tree.${RESET}
Both re-resolve dependencies from ${REPO}/pyproject.toml, which pins torch-spyre
to an upstream git rev. That REVERTS the local editable torch-spyre build (look
for "Uninstalled 1 package ... Installed 1 package" in the output). Use one of:

  VIRTUAL_ENV=${VENV} uv run --no-sync pytest -m "not upstream" tests/test_rms_norm.py
  ${VENV}/bin/python -m pytest -m "not upstream" tests/test_rms_norm.py

Suggested next steps (${BOLD}not run here${RESET} — Spyre admits exactly one process at a
time, so never start two of these concurrently):

  cd ${REPO}
  ${VENV}/bin/python -m pytest -m "not upstream" tests/test_rms_norm.py
  ${VENV}/bin/python examples/offline_inference/torch_spyre_inference.py

EOF

info "Done."
