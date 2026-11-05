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

"""m3's pool_backend key is refused rather than silently served from a host pool.

The tier is chosen by the delegator plan (one backend per pool); see test_pool_plan.
"""

from __future__ import annotations

import pytest

from spyre_inference.v1.kv_offload.spec import SpyreOffloadingSpec
from tests.kv_offload.test_spec import _config


def test_default_backend_is_host():
    SpyreOffloadingSpec(_config())
    SpyreOffloadingSpec(_config(extra={"pool_backend": "host"}))


@pytest.mark.parametrize("backend", ["marvell", "nvme"])
def test_non_host_backend_is_rejected_not_silently_host(backend):
    """m4's spec would otherwise ignore the key and serve from a host pool."""
    with pytest.raises(ValueError, match="pool_backend.*--marvell-bdf"):
        SpyreOffloadingSpec(_config(extra={"pool_backend": backend}))


def test_connector_shutdown_reaches_the_worker():
    """Upstream's chain OffloadingConnector -> connector worker -> worker.shutdown.

    Pinned so an upstream rename of the hook shows up here rather than as a
    leaked .ctl segment after every orderly exit.
    """
    import inspect

    from spyre_inference.v1.kv_offload.upstream_compat import OffloadingConnectorWorker

    assert "self.worker.shutdown()" in inspect.getsource(OffloadingConnectorWorker.shutdown)
