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

"""Explicitly destroy one demo pool generation through the delegator."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from typing import Any

if __package__:
    from scripts.shared_pool_delegator import SharedPoolCtl
else:
    from shared_pool_delegator import SharedPoolCtl

_GENERATION = re.compile(r"[0-9a-fA-F]{32}\Z")


def cleanup_pool(
    pool_name: str,
    pool_generation: str,
    *,
    ctl: Any,
) -> dict[str, Any]:
    """Destroy exactly one generation; never unlink by name or attach."""

    if not pool_name:
        raise ValueError("pool_name must be non-empty")
    if _GENERATION.fullmatch(pool_generation) is None:
        raise ValueError("pool_generation must contain exactly 32 hexadecimal characters")
    result = ctl.destroy(pool_name, pool_generation)
    code = str(result.get("result", "UNKNOWN"))
    if code != "OK":
        detail = str(result.get("detail", "") or "")
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"shared pool destroy returned {code}{suffix}")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-name", required=True)
    parser.add_argument("--pool-generation", required=True)
    parser.add_argument("--ctl", default="shared-poolctl")
    parser.add_argument("--socket", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    ctl = SharedPoolCtl(binary=args.ctl, socket_path=args.socket)
    result = cleanup_pool(
        args.pool_name,
        args.pool_generation,
        ctl=ctl,
    )
    print(
        f"destroyed {args.pool_name} generation {args.pool_generation}: "
        f"{json.dumps(result, sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
