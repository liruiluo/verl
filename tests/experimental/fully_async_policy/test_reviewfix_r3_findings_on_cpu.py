# Copyright 2026 The verl team.
# Licensed under the Apache License, Version 2.0.

"""Regressions for the focused R2 review findings on the fully-async patch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase, mock

import numpy as np
import torch
from tensordict import TensorDict

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.padding_utils import upsample_dataproto_to_divisible_size
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage


def _local_actor_class(actor_class):
    return actor_class.__ray_metadata__.modified_class


def _action_row_batch(group_sizes: list[int]) -> DataProto:
    group_ids = [group for group, size in enumerate(group_sizes) for _ in range(size)]
    rows = len(group_ids)
    prompts = torch.tensor(group_ids, dtype=torch.long).unsqueeze(1).repeat(1, 2)
    responses = prompts + 1
    return DataProto(
        batch=TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": torch.cat([prompts, responses], dim=1),
                "attention_mask": torch.ones((rows, 4), dtype=torch.long),
                "position_ids": torch.arange(4, dtype=torch.long).repeat(rows, 1),
                "response_mask": torch.ones((rows, 2), dtype=torch.long),
                "loss_mask": torch.ones((rows, 2), dtype=torch.long),
                "token_level_rewards": torch.ones((rows, 2), dtype=torch.float32),
            },
            batch_size=rows,
        ),
        non_tensor_batch={
            "uid": np.array(
                [f"episode-{group}" for group, size in enumerate(group_sizes) for _ in range(size)],
                dtype=object,
            ),
            "is_padding": np.zeros(rows, dtype=bool),
        },
    )


def _uid_split_across_equal_chunks(batch: DataProto, chunks: int) -> set[str]:
    uid_to_rank: dict[str, int] = {}
    split: set[str] = set()
    for rank, chunk in enumerate(batch.chunk(chunks)):
        for uid in set(chunk.non_tensor_batch["uid"]):
            previous = uid_to_rank.setdefault(uid, rank)
            if previous != rank:
                split.add(uid)
    return split


class TestPaddingLossNeutralityAndSharedAlignment(TestCase):
    def test_padding_is_loss_neutral_for_real_grpo_advantages(self) -> None:
        original = _action_row_batch([16] * 63 + [15])
        baseline = (
            compute_advantage(
                original,
                adv_estimator=AdvantageEstimator.GRPO,
                norm_adv_by_std_in_grpo=True,
            )
            .batch["advantages"]
            .clone()
        )

        padded = upsample_dataproto_to_divisible_size(
            _action_row_batch([16] * 63 + [15]),
            batch_multiple=1024,
            eos_token_id=0,
            uid_group_divisor=8,
        )
        padded = compute_advantage(
            padded,
            adv_estimator=AdvantageEstimator.GRPO,
            norm_adv_by_std_in_grpo=True,
        )
        real_mask = ~padded.non_tensor_batch["is_padding"].astype(bool)

        torch.testing.assert_close(padded.batch["advantages"][real_mask], baseline)
        self.assertEqual(int(padded.batch["advantages"][~torch.as_tensor(real_mask)].count_nonzero()), 0)

    def test_role_updates_keep_the_shared_balanced_batch_unchanged(self) -> None:
        shared = upsample_dataproto_to_divisible_size(
            _action_row_batch([16] * 41 + [15] * 23),
            batch_multiple=1024,
            eos_token_id=0,
            uid_group_divisor=8,
        )
        balance_owner = object.__new__(RayPPOTrainer)
        balance_owner.use_prefix_grouper = True
        balance_owner.actor_rollout_wg = SimpleNamespace(_dispatch_info={"actor": list(range(8))})
        balance_owner._balance_batch(shared, metrics={})
        self.assertEqual(_uid_split_across_equal_chunks(shared, 8), set())

        trainer = object.__new__(_local_actor_class(FullyAsyncTrainer))
        trainer.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(ppo_mini_batch_size=512),
                rollout=SimpleNamespace(n=1),
            ),
            critic=SimpleNamespace(ppo_mini_batch_size=256),
        )
        trainer.tokenizer = SimpleNamespace(eos_token_id=0)

        with (
            mock.patch.object(SeparateRayPPOTrainer, "_update_actor", return_value="actor") as update_actor,
            mock.patch.object(SeparateRayPPOTrainer, "_update_critic", return_value="critic") as update_critic,
        ):
            self.assertEqual(trainer._update_actor(shared), "actor")
            self.assertEqual(trainer._update_critic(shared), "critic")

        actor_batch = update_actor.call_args.args[0]
        critic_batch = update_critic.call_args.args[0]
        self.assertIs(actor_batch, shared)
        self.assertIs(critic_batch, shared)
        self.assertEqual(_uid_split_across_equal_chunks(actor_batch, 8), set())
        self.assertEqual(_uid_split_across_equal_chunks(critic_batch, 8), set())


class TestProducerResumeHorizon(IsolatedAsyncioTestCase):
    async def test_fresh_producer_emits_each_horizon_index_once_and_advances_cursor(self) -> None:
        from verl.experimental.fully_async_policy import fully_async_rollouter as rollouter_module

        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.pending_queue = asyncio.Queue()
        rollouter.global_steps = 1
        rollouter.total_rollout_steps = 4
        rollouter.config = object()
        rollouter._create_continuous_iterator = lambda: iter((0, {"row": [index]}) for index in range(8))

        with mock.patch.object(
            rollouter_module,
            "prepare_single_generation_data",
            side_effect=lambda batch, _config: batch,
        ):
            await rollouter._feed_samples()

        emitted = [await rollouter.pending_queue.get() for _ in range(4)]
        self.assertEqual([sample.sample_id for sample in emitted], [f"sample_0_{index}" for index in range(1, 5)])
        self.assertIs(await rollouter.pending_queue.get(), None)
        self.assertTrue(rollouter.pending_queue.empty())
        self.assertEqual(rollouter.global_steps, 5)

    async def test_resume_at_boundary_enqueues_only_terminal_signal(self) -> None:
        from verl.experimental.fully_async_policy import fully_async_rollouter as rollouter_module

        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.pending_queue = asyncio.Queue()
        rollouter.global_steps = 5  # next cursor after a four-sample horizon
        rollouter.total_rollout_steps = 4
        rollouter.config = object()
        rollouter._create_continuous_iterator = lambda: iter([(0, {"row": [1]})])

        with mock.patch.object(rollouter_module, "prepare_single_generation_data", return_value=object()) as prepare:
            await rollouter._feed_samples()

        self.assertIs(await rollouter.pending_queue.get(), None)
        self.assertTrue(rollouter.pending_queue.empty())
        prepare.assert_not_called()

    async def test_checkpoint_beyond_shortened_horizon_fails_closed(self) -> None:
        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.config = SimpleNamespace(
            trainer=SimpleNamespace(
                resume_mode="resume_path",
                resume_from_path="/tmp/global_step_3",
                default_hdfs_dir=None,
                default_local_dir="/tmp/unused",
            ),
            async_training=SimpleNamespace(trigger_parameter_sync_step=1),
        )
        rollouter.required_samples = 2
        rollouter.total_rollout_steps = 4
        rollouter.train_dataloader = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "checkpoint.*horizon|horizon.*checkpoint|shortened"):
            rollouter.load_checkpoint()


class _AsyncRemoteMethod:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        async def _result():
            result = self.fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result

        return _result()


class _Replica:
    def __init__(self, name: str, *, resume_failure: BaseException | None = None):
        self._server_address = f"http://{name}"
        self.resume_failure = resume_failure
        self.abort_calls = 0
        self.serving = True

    async def abort_all_requests(self):
        self.abort_calls += 1
        self.serving = False

    async def resume_generation(self):
        self.serving = True
        if self.resume_failure is not None:
            error, self.resume_failure = self.resume_failure, None
            raise error


class TestCancellationSafeQuarantine(IsolatedAsyncioTestCase):
    async def test_cancellation_during_quarantine_preserves_original_fault_and_removes_servers(self) -> None:
        entered_removal = asyncio.Event()
        allow_removal = asyncio.Event()
        standalone = _Replica("standalone")
        hybrid = _Replica("hybrid", resume_failure=RuntimeError("resume failed"))
        servers = {standalone._server_address, hybrid._server_address}

        async def _remove_servers(server_ids):
            entered_removal.set()
            await allow_removal.wait()
            servers.difference_update(server_ids)

        load_balancer = SimpleNamespace(
            clear_sticky_cache=_AsyncRemoteMethod(
                lambda: {"cleared_entries": 0, "server_loads": {server_id: 0 for server_id in servers}}
            ),
            get_total_inflight=_AsyncRemoteMethod(lambda: 0),
            get_all_servers=_AsyncRemoteMethod(lambda: list(servers)),
            remove_servers=_AsyncRemoteMethod(_remove_servers),
        )
        manager = SimpleNamespace(
            global_load_balancer=load_balancer,
            get_standalone_replicas=lambda: [standalone],
            alive_replicas={"hybrid_0": hybrid},
            alive_addresses={"hybrid_0": hybrid._server_address},
            rollout_replicas=[standalone, hybrid],
            server_addresses=[standalone._server_address, hybrid._server_address],
            server_handles=[object(), object()],
        )
        rollouter = object.__new__(_local_actor_class(FullyAsyncRollouter))
        rollouter.llm_server_manager = manager

        task = asyncio.create_task(rollouter.rebalance_requests(wait_inflight_timeout_s=0))
        await asyncio.wait_for(entered_removal.wait(), timeout=1)
        task.cancel()
        allow_removal.set()

        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            await task

        self.assertEqual(servers, set())
        self.assertGreaterEqual(standalone.abort_calls, 2)
        self.assertGreaterEqual(hybrid.abort_calls, 2)


if __name__ == "__main__":
    import unittest

    unittest.main()
