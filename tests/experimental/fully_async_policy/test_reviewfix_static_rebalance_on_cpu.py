# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Focused regressions for static hybrid bootstrap and rebalance fail-closed behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock

from verl.experimental.fully_async_policy.dynamic_schedule import default_policy as default_policy_module
from verl.experimental.fully_async_policy.dynamic_schedule import fixed_ratio_policy as fixed_ratio_policy_module
from verl.experimental.fully_async_policy.dynamic_schedule.base import DynamicScheduleContext
from verl.experimental.fully_async_policy.dynamic_schedule.default_policy import DefaultDynamicSchedulePolicy
from verl.experimental.fully_async_policy.dynamic_schedule.fixed_ratio_policy import FixedRatioDynamicSchedulePolicy
from verl.experimental.fully_async_policy.dynamic_schedule.static_fully_async_policy import StaticFullyAsyncPolicy
from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter


def _local_actor_class(actor_class):
    return actor_class.__ray_metadata__.modified_class


def _context() -> DynamicScheduleContext:
    return DynamicScheduleContext(
        required_samples=64,
        trigger_parameter_sync_step=4,
        total_generated_samples=0,
        expected_samples=0,
        buffer_samples=0,
        only_hybrid=True,
    )


class _AsyncRemoteMethod:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def _result():
            return self.fn(*args, **kwargs)

        return _result()


class _LoadBalancer:
    def __init__(self, server_ids, *, remove_failures: int = 0, inflight_failures: int = 0):
        self.servers = set(server_ids)
        self.remove_failures = remove_failures
        self.inflight_failures = inflight_failures
        self.clear_sticky_cache = _AsyncRemoteMethod(
            lambda: {"cleared_entries": 2, "server_loads": {server_id: 0 for server_id in self.servers}}
        )
        self.get_total_inflight = _AsyncRemoteMethod(self._get_total_inflight)
        self.get_all_servers = _AsyncRemoteMethod(lambda: list(self.servers))
        self.remove_servers = _AsyncRemoteMethod(self._remove_servers)

    def _remove_servers(self, server_ids):
        if self.remove_failures:
            self.remove_failures -= 1
            raise RuntimeError("remove failed")
        self.servers.difference_update(server_ids)

    def _get_total_inflight(self):
        if self.inflight_failures:
            self.inflight_failures -= 1
            raise RuntimeError("inflight read failed")
        return 0


class _Replica:
    def __init__(self, name: str, *, abort_failures: int = 0, resume_failures: int = 0):
        self._server_address = f"http://{name}"
        self.serving = True
        self.abort_failures = abort_failures
        self.resume_failures = resume_failures
        self.abort_calls = 0
        self.resume_calls = 0

    async def abort_all_requests(self):
        self.abort_calls += 1
        self.serving = False
        if self.abort_failures:
            self.abort_failures -= 1
            raise RuntimeError("abort failed")

    async def resume_generation(self):
        self.resume_calls += 1
        self.serving = True
        if self.resume_failures:
            self.resume_failures -= 1
            raise RuntimeError("resume failed")


class TestStaticHybridBootstrap(TestCase):
    def test_static_policy_activates_hybrid_only_after_every_publication(self) -> None:
        policy = StaticFullyAsyncPolicy(only_hybrid=True)
        ctx = _context()

        # With no standalone rollouter, the trainer must first leave a complete
        # publication-cycle batch in the queue before sleeping the sole rollout
        # pool.  A zero threshold creates activate -> immediate deactivate ->
        # empty-queue deadlock before the first optimizer update.
        self.assertEqual(policy.deactivate_wait_samples(ctx), ctx.step_required_samples)
        self.assertTrue(policy.should_activate_after_step(0, False, ctx))
        self.assertTrue(policy.should_activate_after_step(1, False, ctx))


class TestRebalanceFailClosed(IsolatedAsyncioTestCase):
    @staticmethod
    def _rollouter(
        standalone: _Replica,
        hybrid: _Replica,
        *,
        remove_failures: int = 0,
        inflight_failures: int = 0,
    ):
        load_balancer = _LoadBalancer(
            [standalone._server_address, hybrid._server_address],
            remove_failures=remove_failures,
            inflight_failures=inflight_failures,
        )
        manager = SimpleNamespace(
            global_load_balancer=load_balancer,
            get_standalone_replicas=lambda: [standalone],
            alive_replicas={"hybrid_0": hybrid},
            alive_addresses={"hybrid_0": hybrid._server_address},
            server_addresses=[standalone._server_address, hybrid._server_address],
            server_handles=[object(), object()],
        )
        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.llm_server_manager = manager
        return rollouter, load_balancer

    async def test_abort_failure_quarantines_and_reaborts_every_advertised_replica(self) -> None:
        standalone = _Replica("standalone")
        hybrid = _Replica("hybrid", abort_failures=1)
        rollouter, load_balancer = self._rollouter(standalone, hybrid)

        with self.assertRaisesRegex(RuntimeError, "abort failed"):
            await rollouter.rebalance_requests(wait_inflight_timeout_s=0)

        self.assertEqual(load_balancer.servers, set())
        self.assertFalse(standalone.serving)
        self.assertFalse(hybrid.serving)
        self.assertGreaterEqual(standalone.abort_calls, 2)
        self.assertGreaterEqual(hybrid.abort_calls, 2)

    async def test_quarantine_retries_ambiguous_load_balancer_removal(self) -> None:
        standalone = _Replica("standalone")
        hybrid = _Replica("hybrid", resume_failures=1)
        rollouter, load_balancer = self._rollouter(standalone, hybrid, remove_failures=1)

        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            await rollouter.rebalance_requests(wait_inflight_timeout_s=0)

        self.assertEqual(load_balancer.servers, set())
        self.assertFalse(standalone.serving)
        self.assertFalse(hybrid.serving)
        self.assertGreaterEqual(len(load_balancer.remove_servers.calls), 2)

    async def test_inflight_read_failure_quarantines_aborted_replicas(self) -> None:
        standalone = _Replica("standalone")
        hybrid = _Replica("hybrid")
        rollouter, load_balancer = self._rollouter(standalone, hybrid, inflight_failures=1)

        with self.assertRaisesRegex(RuntimeError, "inflight read failed"):
            await rollouter.rebalance_requests(wait_inflight_timeout_s=0)

        self.assertEqual(load_balancer.servers, set())
        self.assertFalse(standalone.serving)
        self.assertFalse(hybrid.serving)
        self.assertGreaterEqual(standalone.abort_calls, 2)
        self.assertGreaterEqual(hybrid.abort_calls, 2)

    async def test_partial_resume_failure_quarantines_and_reaborts_every_replica(self) -> None:
        standalone = _Replica("standalone")
        hybrid = _Replica("hybrid", resume_failures=1)
        rollouter, load_balancer = self._rollouter(standalone, hybrid)

        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            await rollouter.rebalance_requests(wait_inflight_timeout_s=0)

        self.assertEqual(load_balancer.servers, set())
        self.assertFalse(standalone.serving)
        self.assertFalse(hybrid.serving)
        self.assertGreaterEqual(standalone.abort_calls, 2)
        self.assertGreaterEqual(hybrid.abort_calls, 2)


class TestRebalancePolicyPropagation(TestCase):
    def test_default_and_fixed_ratio_policies_propagate_rollouter_failure(self) -> None:
        ctx = _context()
        rollouter = SimpleNamespace(rebalance_requests=SimpleNamespace(remote=mock.Mock(return_value=object())))

        for policy, module in (
            (DefaultDynamicSchedulePolicy(), default_policy_module),
            (FixedRatioDynamicSchedulePolicy(), fixed_ratio_policy_module),
        ):
            with self.subTest(policy=type(policy).__name__):
                policy._rollouter = rollouter
                with (
                    mock.patch.object(module.ray, "get", side_effect=RuntimeError("rebalance failed")),
                    self.assertRaisesRegex(RuntimeError, "rebalance failed"),
                ):
                    policy.request_rebalance(global_steps=3, ctx=ctx)


if __name__ == "__main__":
    import unittest

    unittest.main()
