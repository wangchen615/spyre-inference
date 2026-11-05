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

from types import SimpleNamespace

import pytest

from spyre_inference.platform import _keep_uni_for_shared_spec


def _config(world_size=1, backend="mp", spec_name="SpyreSharedOffloadingSpec"):
    kv_transfer_config = (
        None
        if spec_name is None
        else SimpleNamespace(kv_connector_extra_config={"spec_name": spec_name})
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=world_size, distributed_executor_backend=backend
        ),
        kv_transfer_config=kv_transfer_config,
    )


def test_shared_spec_at_tp1_runs_on_uni():
    config = _config()
    _keep_uni_for_shared_spec(config)
    assert config.parallel_config.distributed_executor_backend == "uni"


@pytest.mark.parametrize(
    "config",
    [
        _config(spec_name=None),  # no KV connector.
        _config(spec_name="SpyreOffloadingSpec"),  # local spec keeps m3's layout.
        _config(world_size=2, backend="external_launcher"),  # TP>1.
        _config(backend="ray"),  # an explicit non-mp backend.
    ],
)
def test_everything_else_keeps_its_executor(config):
    before = config.parallel_config.distributed_executor_backend
    _keep_uni_for_shared_spec(config)
    assert config.parallel_config.distributed_executor_backend == before
