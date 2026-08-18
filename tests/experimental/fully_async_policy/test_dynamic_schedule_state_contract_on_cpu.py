# Copyright 2026 The verl team.
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

"""State-conservation contract for native Hybrid + Standalone scheduling.

This test keeps veRL's controller, policy, load balancer, and replica APIs.  It
only requires transitions to be fail-closed, resource classes to remain
disjoint, and adaptive ratios to stay inside their declared domain.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from verl.experimental.fully_async_policy.dynamic_schedule import dynamic_resource_controller as controller_module
from verl.experimental.fully_async_policy.dynamic_schedule.base import DynamicScheduleContext
from verl.experimental.fully_async_policy.dynamic_schedule.default_policy import DefaultDynamicSchedulePolicy
from verl.experimental.fully_async_policy.dynamic_schedule.dynamic_resource_controller import DynamicResourceController
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerManager


class _AsyncRemoteMethod:
    def __init__(self, result=None, error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def _result():
            if self.error is not None:
                raise self.error
            return self.result

        return _result()


class _SyncRemoteMethod:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _Replica(SimpleNamespace):
    pass


def _replica(name: str) -> _Replica:
    return _Replica(_server_address=f"http://{name}", _server_handle=f"handle-{name}")


def _schedule_context() -> DynamicScheduleContext:
    return DynamicScheduleContext(
        required_samples=64,
        trigger_parameter_sync_step=4,
        total_generated_samples=0,
        expected_samples=0,
        buffer_samples=0,
        step_wait_times=[0.0, 0.0, 0.0, 0.0],
        step_wait_samples=[0, 0, 0, 0],
    )


class TestDefaultDynamicSchedulePolicyDomain(TestCase):
    def test_initial_ratio_is_validated(self) -> None:
        for invalid in (-0.1, 0.0, 1.01):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DefaultDynamicSchedulePolicy(deactivate_ratio=invalid)

    def test_repeated_no_wait_cycles_never_make_ratio_negative(self) -> None:
        policy = DefaultDynamicSchedulePolicy(deactivate_ratio=0.03)
        ctx = _schedule_context()

        for step in range(1, 8):
            policy.update_after_step(step, ctx)
            self.assertGreaterEqual(policy.deactivate_ratio, 0.0)
            self.assertLessEqual(policy.deactivate_ratio, 1.0)
            self.assertGreaterEqual(policy.deactivate_wait_samples(ctx), 0)

        self.assertEqual(policy.deactivate_ratio, 0.0)


class TestHybridStandaloneBookkeeping(IsolatedAsyncioTestCase):
    async def test_hybrid_activation_does_not_pollute_standalone_membership(self) -> None:
        standalone = [_replica("standalone-0"), _replica("standalone-1")]
        hybrid = {"hybrid_0": _replica("hybrid-0"), "hybrid_1": _replica("hybrid-1")}

        manager = object.__new__(FullyAsyncLLMServerManager)
        manager.rollout_replicas = list(standalone)
        manager.hybrid_replicas = dict(hybrid)
        manager.alive_replicas = {}
        manager.alive_addresses = {}
        manager.server_addresses = [replica._server_address for replica in standalone]
        manager.server_handles = [replica._server_handle for replica in standalone]
        manager.last_hybrid_add_time = 0.0
        manager.last_hybrid_remove_time = 0.0
        manager.global_load_balancer = SimpleNamespace(add_servers=_AsyncRemoteMethod(result=None))

        added = await manager.add_replicas(list(hybrid))

        self.assertEqual(added, 2)
        self.assertEqual(manager.get_standalone_replicas(), standalone)
        self.assertEqual(set(manager.alive_replicas), set(hybrid))
        self.assertTrue(
            {id(replica) for replica in manager.get_standalone_replicas()}.isdisjoint(
                {id(replica) for replica in manager.alive_replicas.values()}
            )
        )
        self.assertEqual(manager.get_active_server_count(), 4)


class TestDynamicResourceControllerFailClosed(IsolatedAsyncioTestCase):
    def _controller(self, add_result=2, remove_result=2):
        replicas = {"hybrid_0": _replica("hybrid-0"), "hybrid_1": _replica("hybrid-1")}
        rollouter = SimpleNamespace(
            get_all_hybrid_replicas=_SyncRemoteMethod(replicas),
            add_replicas=_AsyncRemoteMethod(add_result),
            remove_replicas=_AsyncRemoteMethod(remove_result),
        )
        checkpoint = SimpleNamespace(
            replicas=list(replicas.values()),
            abort_replicas=mock.AsyncMock(),
            update_weights=mock.AsyncMock(),
            resume_generation_replicas=mock.AsyncMock(),
            sleep_replicas=mock.AsyncMock(),
        )
        controller = DynamicResourceController(
            rollouter=rollouter,
            hybrid_checkpoint_manager=checkpoint,
            num_standalone_replicas=2,
            num_hybrid_replicas=2,
        )
        return controller, rollouter, checkpoint

    async def test_failed_add_cannot_flip_active_state_or_resume(self) -> None:
        controller, _rollouter, checkpoint = self._controller(add_result=0)

        with mock.patch.object(controller_module.ray, "get", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "activate|add|replica"):
                await controller.activate_hybrid_replicas(global_steps=3)

        self.assertFalse(controller.is_hybrid_active)
        self.assertEqual(controller.activate_count, 0)
        checkpoint.resume_generation_replicas.assert_not_awaited()

    async def test_failed_remove_cannot_sleep_or_flip_active_state(self) -> None:
        controller, _rollouter, checkpoint = self._controller(remove_result=0)
        controller._hybrid_active = True

        with mock.patch.object(controller_module.ray, "get", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "deactivate|remove|replica"):
                await controller.deactivate_hybrid_replicas(global_steps=4)

        self.assertTrue(controller.is_hybrid_active)
        self.assertEqual(controller.deactivate_count, 0)
        checkpoint.abort_replicas.assert_not_awaited()
        checkpoint.sleep_replicas.assert_not_awaited()

    async def test_successful_transitions_flip_state_exactly_once(self) -> None:
        controller, _rollouter, checkpoint = self._controller(add_result=2, remove_result=2)

        with mock.patch.object(controller_module.ray, "get", side_effect=lambda value: value):
            await controller.activate_hybrid_replicas(global_steps=5)
            self.assertTrue(controller.is_hybrid_active)
            self.assertEqual(controller.activate_count, 1)
            checkpoint.resume_generation_replicas.assert_awaited_once_with()

            await controller.deactivate_hybrid_replicas(global_steps=6)

        self.assertFalse(controller.is_hybrid_active)
        self.assertEqual(controller.deactivate_count, 1)
        checkpoint.abort_replicas.assert_awaited_once_with()
        checkpoint.sleep_replicas.assert_awaited_once_with()


if __name__ == "__main__":
    import unittest

    unittest.main()
