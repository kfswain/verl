# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ray-free stand-ins shared by the DAS CPU tests."""

from __future__ import annotations

from verl.workers.rollout.replica import TokenOutput


class Ref:
    """Awaitable holding a ready value, like a resolved Ray ObjectRef."""

    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def _ready():
            return self.value

        return _ready().__await__()


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return Ref(self._fn(*args, **kwargs))


class FakeActor:
    """Wraps a plain object so ``handle.method.remote(...)`` calls it synchronously."""

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        return _RemoteMethod(getattr(self._obj, name))


class FakeServer:
    """Records engine calls and replays scripted ``TokenOutput`` results."""

    def __init__(self, outputs=None):
        self.calls: list[dict] = []
        self.outputs = list(outputs or [])

    def generate(self, request_id, prompt_ids, sampling_params, **kwargs):
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids), **kwargs})
        if self.outputs:
            return self.outputs.pop(0)
        return TokenOutput(token_ids=[7, 8, 9], stop_reason="completed", extra_fields={"global_steps": 1})


def fake_ray_get(ref, timeout=None):
    """Drop-in for ``ray.get`` on :class:`Ref` values."""
    return ref.value if isinstance(ref, Ref) else ref
