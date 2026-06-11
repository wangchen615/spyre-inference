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

"""Pure-CPU test: SpyreOffloadingSpec registers and resolves via the factory.

No Spyre device required — exercises only the lazy registration wired in
``spyre_inference.__init__.register_offloading_specs``.
"""

import pytest


@pytest.mark.spyre
def test_spec_is_registered_and_resolves():
    import spyre_inference  # noqa: F401  (import triggers registration)
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory
    from vllm.v1.kv_offload.spec import OffloadingSpec

    assert "SpyreOffloadingSpec" in OffloadingSpecFactory._registry

    # The factory stores a lazy loader; resolving it must import our module and
    # return the SpyreOffloadingSpec class without dragging in Spyre-only deps.
    loader = OffloadingSpecFactory._registry["SpyreOffloadingSpec"]
    spec_cls = loader()
    assert spec_cls.__name__ == "SpyreOffloadingSpec"
    assert spec_cls.__module__ == "spyre_inference.v1.kv_offload.spec"
    assert issubclass(spec_cls, OffloadingSpec)


@pytest.mark.spyre
def test_registration_is_idempotent():
    import spyre_inference

    # register_offloading_specs guards against the factory's duplicate-name
    # ValueError, so calling it again is a no-op.
    spyre_inference.register_offloading_specs()
    spyre_inference.register_offloading_specs()
