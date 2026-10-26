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

import subprocess
import sys
from argparse import Namespace

import pytest

from scripts import cleanup_shared_kv_demo as cleanup

GENERATION = "ab" * 16


class FakeCtl:
    def __init__(self, result=None):
        self.result = result or {"result": "OK"}
        self.calls = []

    def destroy(self, name, generation):
        self.calls.append((name, generation))
        return self.result


def test_cleanup_calls_only_generation_checked_destroy():
    ctl = FakeCtl()

    cleanup.cleanup_pool("deployment-root", GENERATION, ctl=ctl)

    assert ctl.calls == [("deployment-root", GENERATION)]


def test_cleanup_validates_generation_before_calling_delegator():
    ctl = FakeCtl()

    with pytest.raises(ValueError, match="32 hexadecimal"):
        cleanup.cleanup_pool("deployment-root", "bad", ctl=ctl)

    assert ctl.calls == []


def test_cleanup_reports_pool_busy_instead_of_claiming_success():
    ctl = FakeCtl({"result": "POOL_BUSY", "detail": "one attachment remains"})

    with pytest.raises(RuntimeError, match="POOL_BUSY.*one attachment remains"):
        cleanup.cleanup_pool("deployment-root", GENERATION, ctl=ctl)


def test_cleanup_main_uses_explicit_identity(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        cleanup,
        "_parse_args",
        lambda: Namespace(
            pool_name="deployment-root",
            pool_generation=GENERATION,
            ctl="shared-poolctl",
            socket="/tmp/delegator.sock",
        ),
    )
    monkeypatch.setattr(
        cleanup,
        "cleanup_pool",
        lambda name, generation, **kwargs: calls.append((name, generation, kwargs))
        or {"result": "OK"},
    )

    cleanup.main()

    assert calls[0][:2] == ("deployment-root", GENERATION)
    assert "destroyed deployment-root generation" in capsys.readouterr().out


def test_cleanup_script_can_run_directly_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/cleanup_shared_kv_demo.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--pool-generation" in result.stdout
    assert "--socket" in result.stdout
