# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Fault-cleanup contracts for native publication and dynamic transitions.

CheckpointEngineManager and DynamicResourceController remain the sole owners.
The fixture does not authorize a synchronizer, rollback ledger, scheduler, load
balancer, or recovery stack. It distinguishes failures before any weight
mutation from ambiguous transfer/finalize/update failures:

* pre-mutation failures may restore the previous serving state;
* after an ambiguous mutation, replicas stay aborted/non-serving and the
  lineage fails rather than exposing mixed weights;
* partial LB transitions are read back from state, not inferred only from RPC
  attempts.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, mock

from verl.checkpoint_engine import base as checkpoint_module
from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.experimental.fully_async_policy.dynamic_schedule import dynamic_resource_controller as controller_module
from verl.experimental.fully_async_policy.dynamic_schedule.dynamic_resource_controller import DynamicResourceController


class _RemoteMethod:
    def __init__(self, result=None, callback=None):
        self.result = result
        self.callback = callback
        self.calls = []
        self.error: Exception | None = None
        self.fail_once_after_callback = False

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def _result():
            if self.callback is not None:
                self.callback(*args, **kwargs)
            if self.error is not None:
                error = self.error
                if self.fail_once_after_callback:
                    self.error = None
                raise error
            return self.result

        return _result()


class _SyncRemoteMethod:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _StatefulRollouter:
    def __init__(self, replicas: dict[str, object]):
        self.replicas = replicas
        self.lb_membership: set[str] = set()
        self.get_all_hybrid_replicas = _SyncRemoteMethod(replicas)
        self.add_replicas = _RemoteMethod(result=len(replicas), callback=self._add)
        self.remove_replicas = _RemoteMethod(result=len(replicas), callback=self._remove)

    def _add(self, resource_ids: list[str]) -> None:
        ids = list(resource_ids)
        # A failing RPC may have changed only part of LB state.
        if self.add_replicas.error is not None:
            self.lb_membership.update(ids[:1])
        else:
            self.lb_membership.update(ids)

    def _remove(self, resource_ids: list[str]) -> None:
        ids = list(resource_ids)
        # A failing RPC may have removed only part of the membership.
        targets = ids[:1] if self.remove_replicas.error is not None else ids
        self.lb_membership.difference_update(targets)


class _FakeWorkerGroup:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.world_size = 2

    def update_weights(self, **kwargs):
        del kwargs
        return ["rollout-transfer-0", "rollout-transfer-1"]

    def execute_checkpoint_engine(self, methods):
        return [f"rollout-{method}" for method in methods]


class _FakeActorGroup:
    world_size = 2

    def update_weights(self, **kwargs):
        del kwargs
        return ["actor-transfer-0", "actor-transfer-1"]

    def execute_checkpoint_engine(self, methods):
        return [f"actor-{method}" for method in methods]


class TestCheckpointPublicationFaultCleanup(IsolatedAsyncioTestCase):
    def _manager(self):
        manager = object.__new__(CheckpointEngineManager)
        manager.backend = "nccl"
        manager.actor_wg = _FakeActorGroup()
        manager.replicas = [SimpleNamespace(workers=[object()]), SimpleNamespace(workers=[object()])]
        manager.serving_readback = [True, True]
        manager.kv_ready_readback = [True, True]
        manager.version_readback = [8, 8]

        async def _abort():
            manager.serving_readback[:] = [False, False]

        async def _release_kv():
            manager.kv_ready_readback[:] = [False, False]

        async def _resume_kv():
            manager.kv_ready_readback[:] = [True, True]

        async def _resume_generation():
            manager.serving_readback[:] = [True, True]

        manager.abort_replicas = mock.AsyncMock(side_effect=_abort)
        manager.release_kv_cache_replicas = mock.AsyncMock(side_effect=_release_kv)
        manager.resume_kv_cache_replicas = mock.AsyncMock(side_effect=_resume_kv)
        manager.resume_generation_replicas = mock.AsyncMock(side_effect=_resume_generation)
        manager.build_process_group = mock.Mock()
        return manager

    async def _run_with_failure(self, stage: str):
        manager = self._manager()
        if stage == "abort":

            async def _partial_abort():
                manager.serving_readback[:] = [False, True]
                raise RuntimeError("abort fault")

            manager.abort_replicas.side_effect = _partial_abort
        elif stage == "kv_release":

            async def _partial_release():
                manager.kv_ready_readback[:] = [False, True]
                raise RuntimeError("kv release fault")

            manager.release_kv_cache_replicas.side_effect = _partial_release
        elif stage == "kv_restore":

            async def _partial_kv_restore():
                manager.kv_ready_readback[:] = [True, False]
                raise RuntimeError("kv restore fault")

            manager.resume_kv_cache_replicas.side_effect = _partial_kv_restore
        elif stage == "generation_resume":

            async def _partial_generation_resume():
                manager.serving_readback[:] = [True, False]
                raise RuntimeError("generation resume fault")

            manager.resume_generation_replicas.side_effect = _partial_generation_resume

        ray_calls = 0

        def _ray_get(values):
            nonlocal ray_calls
            del values
            ray_calls += 1
            if ray_calls == 1:
                if stage == "transfer":
                    manager.version_readback[:] = [9, 8]
                    raise RuntimeError("transfer fault")
                manager.version_readback[:] = [9, 9]
                return [{"published_candidate": 9}, {"published_candidate": 9}, None, None]
            if stage == "finalize":
                manager.version_readback[:] = [9, 8]
                raise RuntimeError("finalize fault")
            return [None, None, None, None]

        with (
            mock.patch.object(checkpoint_module, "RayWorkerGroup", _FakeWorkerGroup),
            mock.patch.object(checkpoint_module.ray, "get", side_effect=_ray_get),
        ):
            with self.assertRaisesRegex(RuntimeError, "fault"):
                await manager.update_weights(global_steps=9)
        return manager

    async def test_abort_fault_is_pre_mutation_and_may_restore_previous_serving_state(self) -> None:
        manager = await self._run_with_failure("abort")
        manager.resume_generation_replicas.assert_awaited()
        manager.release_kv_cache_replicas.assert_not_awaited()
        self.assertEqual(manager.version_readback, [8, 8])
        self.assertEqual(manager.serving_readback, [True, True])

    async def test_kv_release_fault_is_pre_mutation_and_may_restore_kv_and_serving(self) -> None:
        manager = await self._run_with_failure("kv_release")
        manager.resume_kv_cache_replicas.assert_awaited()
        manager.resume_generation_replicas.assert_awaited()
        self.assertEqual(manager.version_readback, [8, 8])
        self.assertEqual(manager.kv_ready_readback, [True, True])
        self.assertEqual(manager.serving_readback, [True, True])

    async def test_transfer_fault_keeps_ambiguous_replicas_aborted_and_uncommitted(self) -> None:
        manager = await self._run_with_failure("transfer")
        manager.resume_kv_cache_replicas.assert_not_awaited()
        manager.resume_generation_replicas.assert_not_awaited()
        self.assertEqual(manager.serving_readback, [False, False])
        self.assertNotEqual(manager.version_readback, [9, 9])

    async def test_finalize_fault_keeps_ambiguous_replicas_aborted_and_uncommitted(self) -> None:
        manager = await self._run_with_failure("finalize")
        manager.resume_kv_cache_replicas.assert_not_awaited()
        manager.resume_generation_replicas.assert_not_awaited()
        self.assertEqual(manager.serving_readback, [False, False])
        self.assertNotEqual(manager.version_readback, [9, 9])

    async def test_kv_restore_fault_does_not_resume_serving_without_complete_kv(self) -> None:
        manager = await self._run_with_failure("kv_restore")
        manager.resume_kv_cache_replicas.assert_awaited()
        manager.resume_generation_replicas.assert_not_awaited()
        self.assertEqual(manager.serving_readback, [False, False])
        self.assertNotEqual(manager.kv_ready_readback, [True, True])

    async def test_generation_resume_fault_reaborts_any_partially_serving_replica(self) -> None:
        manager = await self._run_with_failure("generation_resume")
        manager.resume_kv_cache_replicas.assert_awaited()
        manager.resume_generation_replicas.assert_awaited_once_with()
        self.assertGreaterEqual(manager.abort_replicas.await_count, 2)
        self.assertEqual(manager.serving_readback, [False, False])


class TestDynamicTransitionFaultCleanup(IsolatedAsyncioTestCase):
    def _controller(self):
        replicas = {"hybrid_0": object(), "hybrid_1": object()}
        rollouter = _StatefulRollouter(replicas)
        checkpoint = SimpleNamespace(
            replicas=list(replicas.values()),
            serving_readback=[False, False],
            version_readback=[2, 2],
        )

        async def _abort():
            checkpoint.serving_readback[:] = [False, False]

        async def _update_weights(global_steps: int):
            checkpoint.version_readback[:] = [global_steps, global_steps]

        async def _resume_generation():
            checkpoint.serving_readback[:] = [True, True]

        checkpoint.abort_replicas = mock.AsyncMock(side_effect=_abort)
        checkpoint.update_weights = mock.AsyncMock(side_effect=_update_weights)
        checkpoint.resume_generation_replicas = mock.AsyncMock(side_effect=_resume_generation)
        checkpoint.sleep_replicas = mock.AsyncMock()
        checkpoint.wake_up_replicas = mock.AsyncMock()
        controller = DynamicResourceController(
            rollouter=rollouter,
            hybrid_checkpoint_manager=checkpoint,
            num_standalone_replicas=2,
            num_hybrid_replicas=2,
        )
        return controller, rollouter, checkpoint

    async def test_hybrid_sync_update_fault_keeps_ambiguous_replicas_nonserving(self) -> None:
        controller, rollouter, checkpoint = self._controller()

        async def _partial_update(global_steps: int):
            checkpoint.version_readback[:] = [global_steps, global_steps - 1]
            raise RuntimeError("hybrid update fault")

        checkpoint.update_weights.side_effect = _partial_update
        with self.assertRaisesRegex(RuntimeError, "hybrid update fault"):
            await controller.sync_hybrid_weights(global_steps=3)

        self.assertFalse(controller.is_hybrid_active)
        self.assertEqual(rollouter.lb_membership, set())
        self.assertEqual(checkpoint.serving_readback, [False, False])
        self.assertNotEqual(checkpoint.version_readback, [3, 3])
        checkpoint.resume_generation_replicas.assert_not_awaited()
        self.assertGreaterEqual(checkpoint.abort_replicas.await_count, 1)

    async def test_hybrid_sync_abort_fault_is_pre_mutation_and_may_restore_serving(self) -> None:
        controller, _rollouter, checkpoint = self._controller()

        async def _partial_abort():
            checkpoint.serving_readback[:] = [False, True]
            raise RuntimeError("hybrid abort fault")

        checkpoint.serving_readback[:] = [True, True]
        checkpoint.abort_replicas.side_effect = _partial_abort
        with self.assertRaisesRegex(RuntimeError, "hybrid abort fault"):
            await controller.sync_hybrid_weights(global_steps=3)

        checkpoint.update_weights.assert_not_awaited()
        checkpoint.resume_generation_replicas.assert_awaited()
        self.assertEqual(checkpoint.version_readback, [2, 2])
        self.assertEqual(checkpoint.serving_readback, [True, True])

    async def test_hybrid_sync_resume_fault_reaborts_partial_serving_state(self) -> None:
        controller, rollouter, checkpoint = self._controller()

        async def _partial_resume():
            checkpoint.serving_readback[:] = [True, False]
            raise RuntimeError("hybrid resume fault")

        checkpoint.resume_generation_replicas.side_effect = _partial_resume
        with self.assertRaisesRegex(RuntimeError, "hybrid resume fault"):
            await controller.sync_hybrid_weights(global_steps=3)

        self.assertFalse(controller.is_hybrid_active)
        self.assertEqual(controller.activate_count, 0)
        self.assertEqual(rollouter.lb_membership, set())
        self.assertEqual(checkpoint.serving_readback, [False, False])
        self.assertEqual(checkpoint.version_readback, [3, 3])
        self.assertGreaterEqual(checkpoint.abort_replicas.await_count, 2)

    async def test_activation_resume_fault_removes_lb_membership_and_aborts(self) -> None:
        controller, rollouter, checkpoint = self._controller()

        async def _partial_resume():
            checkpoint.serving_readback[:] = [True, False]
            raise RuntimeError("hybrid resume fault")

        checkpoint.resume_generation_replicas.side_effect = _partial_resume
        with mock.patch.object(controller_module.ray, "get", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "hybrid resume fault"):
                await controller.activate_hybrid_replicas(global_steps=4)

        self.assertFalse(controller.is_hybrid_active)
        self.assertEqual(controller.activate_count, 0)
        self.assertEqual(rollouter.lb_membership, set())
        self.assertEqual(checkpoint.serving_readback, [False, False])
        checkpoint.abort_replicas.assert_awaited()

    async def test_activation_add_fault_cleans_partial_lb_membership_and_aborts(self) -> None:
        controller, rollouter, checkpoint = self._controller()
        rollouter.add_replicas.error = RuntimeError("lb add fault")
        rollouter.add_replicas.fail_once_after_callback = True

        with mock.patch.object(controller_module.ray, "get", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "lb add fault"):
                await controller.activate_hybrid_replicas(global_steps=4)

        self.assertFalse(controller.is_hybrid_active)
        self.assertEqual(controller.activate_count, 0)
        self.assertEqual(rollouter.lb_membership, set())
        self.assertEqual(checkpoint.serving_readback, [False, False])
        checkpoint.abort_replicas.assert_awaited()

    async def test_deactivation_abort_fault_restores_previous_lb_and_serving_state(self) -> None:
        controller, rollouter, checkpoint = self._controller()
        controller._hybrid_active = True
        rollouter.lb_membership.update(rollouter.replicas)
        checkpoint.serving_readback[:] = [True, True]

        async def _partial_abort():
            checkpoint.serving_readback[:] = [False, True]
            raise RuntimeError("deactivate abort fault")

        checkpoint.abort_replicas.side_effect = _partial_abort
        with mock.patch.object(controller_module.ray, "get", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "deactivate abort fault"):
                await controller.deactivate_hybrid_replicas(global_steps=5)

        self.assertTrue(controller.is_hybrid_active)
        self.assertEqual(controller.deactivate_count, 0)
        self.assertEqual(rollouter.lb_membership, set(rollouter.replicas))
        self.assertEqual(checkpoint.serving_readback, [True, True])

    async def test_deactivation_sleep_fault_wakes_and_restores_previous_state(self) -> None:
        controller, rollouter, checkpoint = self._controller()
        controller._hybrid_active = True
        rollouter.lb_membership.update(rollouter.replicas)
        checkpoint.serving_readback[:] = [True, True]
        checkpoint.sleep_replicas.side_effect = RuntimeError("sleep fault")

        with mock.patch.object(controller_module.ray, "get", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "sleep fault"):
                await controller.deactivate_hybrid_replicas(global_steps=5)

        self.assertTrue(controller.is_hybrid_active)
        self.assertEqual(controller.deactivate_count, 0)
        checkpoint.wake_up_replicas.assert_awaited()
        self.assertEqual(rollouter.lb_membership, set(rollouter.replicas))
        self.assertEqual(checkpoint.serving_readback, [True, True])

    async def test_deactivation_remove_fault_restores_partial_lb_state(self) -> None:
        controller, rollouter, checkpoint = self._controller()
        controller._hybrid_active = True
        rollouter.lb_membership.update(rollouter.replicas)
        checkpoint.serving_readback[:] = [True, True]
        rollouter.remove_replicas.error = RuntimeError("lb remove fault")
        rollouter.remove_replicas.fail_once_after_callback = True

        with mock.patch.object(controller_module.ray, "get", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "lb remove fault"):
                await controller.deactivate_hybrid_replicas(global_steps=5)

        self.assertTrue(controller.is_hybrid_active)
        self.assertEqual(controller.deactivate_count, 0)
        self.assertEqual(rollouter.lb_membership, set(rollouter.replicas))
        self.assertEqual(checkpoint.serving_readback, [True, True])
        checkpoint.abort_replicas.assert_not_awaited()


if __name__ == "__main__":
    import unittest

    unittest.main()
