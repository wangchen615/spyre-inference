#!/usr/bin/env python3
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

"""Small JSON client for the shared-pool delegator CLI."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


class SharedPoolCtl:
    """Invoke one shared-poolctl binary against one node-local daemon."""

    def __init__(
        self,
        *,
        binary: str | Path = "shared-poolctl",
        socket_path: str | Path | None = None,
    ) -> None:
        self.binary = str(binary)
        self.socket_path = str(
            socket_path
            or os.environ.get(
                "SHARED_POOL_DELEGATOR_SOCKET",
                "/run/shared-pool-delegator/shared-pool.sock",
            )
        )

    def _run(self, *args: str) -> dict[str, Any]:
        completed = subprocess.run(
            [self.binary, "--socket", self.socket_path, *args],
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout.strip() or completed.stderr.strip()
        try:
            result = json.loads(output)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"shared-poolctl {' '.join(args)} failed with exit "
                f"{completed.returncode}: {output or 'no output'}"
            ) from error
        if not isinstance(result, dict):
            raise RuntimeError("shared-poolctl returned a non-object JSON response")
        return result

    def create(self, config: dict[str, Any]) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as file:
            json.dump(config, file)
            file.flush()
            return self._run("create", "--config", file.name)

    def open(self, name: str) -> dict[str, Any]:
        return self._run("open", "--name", name)

    def destroy(self, name: str, generation: str) -> dict[str, Any]:
        return self._run(
            "destroy",
            "--name",
            name,
            "--generation",
            generation,
        )
